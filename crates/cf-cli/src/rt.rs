//! Wall-clock paced real-time runner (virtual audio device).
//!
//! No physical audio endpoint is opened: the runner is a single paced thread
//! that wakes on an absolute schedule derived from the process's high-resolution
//! monotonic clock (`Instant`, i.e. QPC on Windows), runs one engine frame, and
//! accounts for the result.  This is the same shape as an audio callback, with
//! the device I/O replaced by deterministic or captured feature frames.
//!
//! For frame `n` the callback is scheduled at `T0 + n * period` and its output
//! must be ready before the next callback at `T0 + (n + 1) * period` (one
//! output buffer of slack).  A deadline miss ("xrun") is counted when the
//! frame is not finished by that boundary.  Wake lateness and the full
//! callback latency (`wake -> output ready`) are recorded separately, so a slow
//! wake that still finishes in time is visible without being called an xrun.
//!
//! On Windows the runner raises the system timer resolution to 1 ms for the
//! duration of the run and uses a hybrid wait (sleep until 1.5 ms before the
//! deadline, then spin), unless another pacing mode is selected.

use std::fs;
use std::time::{Duration, Instant};

use cf_core::Backend;
use cf_echo::EchoModel;
use cf_voice::VoiceModel;

use crate::npz;

const DEFAULT_ECHO_CFWM: &str = "models/echo-default.cfwm";
const DEFAULT_VOICE_CFWM: &str = "models/voice-default.cfwm";

const HYBRID_MARGIN: Duration = Duration::from_micros(1_500);

/// Maximum frame count accepted (bounds the sample vectors and overflow).
const MAX_FRAMES: u64 = 5_000_000;
/// Maximum soak length accepted, in seconds (4 hours).
const MAX_SOAK_SECONDS: f64 = 4.0 * 3600.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelKind {
    Voice,
    Echo,
}

impl ModelKind {
    fn name(self) -> &'static str {
        match self {
            ModelKind::Voice => "voice",
            ModelKind::Echo => "echo",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pacing {
    Hybrid,
    Sleep,
    Spin,
}

impl Pacing {
    fn parse(s: &str) -> Option<Pacing> {
        match s {
            "hybrid" => Some(Pacing::Hybrid),
            "sleep" => Some(Pacing::Sleep),
            "spin" => Some(Pacing::Spin),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Pacing::Hybrid => "hybrid",
            Pacing::Sleep => "sleep",
            Pacing::Spin => "spin",
        }
    }
}

enum Engine {
    Voice(VoiceModel),
    Echo(EchoModel),
}

impl Engine {
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        match self {
            Engine::Voice(m) => {
                m.process(input).map_err(|e| e.to_string())?;
                output.copy_from_slice(m.output());
            }
            Engine::Echo(m) => {
                m.process(input).map_err(|e| e.to_string())?;
                output.copy_from_slice(m.output());
            }
        }
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        match self {
            Engine::Voice(m) => m.backend_name(),
            Engine::Echo(m) => m.backend_name(),
        }
    }

    fn input_len(&self) -> usize {
        match self {
            Engine::Voice(_) => cf_voice::INPUT_LEN,
            Engine::Echo(_) => cf_echo::INPUT_LEN,
        }
    }

    fn output_len(&self) -> usize {
        match self {
            Engine::Voice(_) => cf_voice::L18_LEN,
            Engine::Echo(_) => cf_echo::OUTPUT_LEN,
        }
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `f32` in `[-0.5, 0.5)`.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0) - 0.5
    }
}

/// A vector of nanosecond samples with nearest-rank percentiles.
struct Samples {
    ns: Vec<u64>,
    sum: u128,
}

impl Samples {
    fn with_capacity(n: u64) -> Samples {
        let cap = n.min(MAX_FRAMES) as usize;
        Samples {
            ns: Vec::with_capacity(cap),
            sum: 0,
        }
    }

    #[inline]
    fn push(&mut self, ns: u64) {
        self.sum += u128::from(ns);
        self.ns.push(ns);
    }

    fn sort(&mut self) {
        self.ns.sort_unstable();
    }

    fn mean_ns(&self) -> f64 {
        if self.ns.is_empty() {
            0.0
        } else {
            self.sum as f64 / self.ns.len() as f64
        }
    }

    /// Nearest-rank percentile (`p` in 0..=100).
    fn pct_ns(&self, p: f64) -> u64 {
        if self.ns.is_empty() {
            return 0;
        }
        let rank = ((p / 100.0) * self.ns.len() as f64).ceil() as usize;
        self.ns[rank.clamp(1, self.ns.len()) - 1]
    }

    fn max_ns(&self) -> u64 {
        self.ns.last().copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// platform: 1 ms timer resolution and optional above-normal priority
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;

    #[link(name = "winmm")]
    extern "system" {
        fn timeBeginPeriod(period: u32) -> u32;
        fn timeEndPeriod(period: u32) -> u32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn SetThreadPriority(handle: *mut c_void, priority: i32) -> i32;
    }

    /// `THREAD_PRIORITY_HIGHEST`; above normal but not the real-time class.
    const THREAD_PRIORITY_HIGHEST: i32 = 2;

    /// Holds the 1 ms system timer resolution while alive.
    pub struct TimerResolution {
        active: bool,
    }

    impl TimerResolution {
        pub fn acquire() -> TimerResolution {
            let active = unsafe { timeBeginPeriod(1) } == 0;
            TimerResolution { active }
        }
    }

    impl Drop for TimerResolution {
        fn drop(&mut self) {
            if self.active {
                unsafe {
                    timeEndPeriod(1);
                }
            }
        }
    }

    pub fn raise_thread_priority() -> bool {
        unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) != 0 }
    }
}

#[cfg(not(windows))]
mod platform {
    pub struct TimerResolution;

    impl TimerResolution {
        pub fn acquire() -> TimerResolution {
            TimerResolution
        }
    }

    pub fn raise_thread_priority() -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// pacing
// ---------------------------------------------------------------------------

#[inline]
fn wait_until(deadline: Instant, pacing: Pacing) {
    match pacing {
        Pacing::Spin => {
            while Instant::now() < deadline {
                std::hint::spin_loop();
            }
        }
        Pacing::Sleep => {
            let now = Instant::now();
            if deadline > now {
                std::thread::sleep(deadline - now);
            }
        }
        Pacing::Hybrid => loop {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline - now;
            if remaining > HYBRID_MARGIN {
                std::thread::sleep(remaining - HYBRID_MARGIN);
            } else {
                std::hint::spin_loop();
            }
        },
    }
}

#[inline]
fn spin_for(duration: Duration) {
    let end = Instant::now() + duration;
    while Instant::now() < end {
        std::hint::spin_loop();
    }
}

// ---------------------------------------------------------------------------
// options and command
// ---------------------------------------------------------------------------

struct Options {
    model: ModelKind,
    period_ns: u64,
    frames: u64,
    backend: Backend,
    cfwm: String,
    input: Option<String>,
    jitter_us: u64,
    spike_every: u64,
    spike_us: u64,
    seed: u64,
    warmup: u64,
    pacing: Pacing,
    json: Option<String>,
    progress_seconds: f64,
    high_priority: bool,
}

fn parse_f64(args: &[String], i: &mut usize, name: &str) -> Result<f64, String> {
    let v = args
        .get(*i + 1)
        .ok_or_else(|| format!("{name} needs a value"))?;
    *i += 2;
    v.parse::<f64>()
        .map_err(|_| format!("{name}: {v:?} is not a number"))
}

fn parse_u64(args: &[String], i: &mut usize, name: &str) -> Result<u64, String> {
    let v = args
        .get(*i + 1)
        .ok_or_else(|| format!("{name} needs a value"))?;
    *i += 2;
    v.parse::<u64>()
        .map_err(|_| format!("{name}: {v:?} is not an integer"))
}

fn take_value(args: &[String], i: &mut usize, name: &str) -> Result<String, String> {
    let v = args
        .get(*i + 1)
        .ok_or_else(|| format!("{name} needs a value"))?;
    *i += 2;
    Ok(v.clone())
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut model: Option<ModelKind> = None;
    let mut period_ms: f64 = 16.0;
    let mut period_given = false;
    let mut soak_minutes: Option<f64> = None;
    let mut seconds: Option<f64> = None;
    let mut frames: Option<u64> = None;
    let mut backend = Backend::detect();
    let mut cfwm: Option<String> = None;
    let mut input: Option<String> = None;
    let mut jitter_us = 0u64;
    let mut spike_every = 0u64;
    let mut spike_us = 0u64;
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut warmup = 100u64;
    let mut pacing = Pacing::Hybrid;
    let mut json = None;
    let mut progress_seconds = 60.0f64;
    let mut high_priority = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--period-ms" => {
                period_ms = parse_f64(args, &mut i, "--period-ms")?;
                period_given = true;
            }
            "--soak" => soak_minutes = Some(parse_f64(args, &mut i, "--soak")?),
            "--seconds" => seconds = Some(parse_f64(args, &mut i, "--seconds")?),
            "--frames" => frames = Some(parse_u64(args, &mut i, "--frames")?),
            "--backend" => {
                let v = take_value(args, &mut i, "--backend")?;
                backend = match v.as_str() {
                    "auto" => Backend::detect(),
                    "scalar" => Backend::Scalar,
                    "avx2" => Backend::Avx2,
                    _ => return Err(format!("unknown backend {v:?} (auto|scalar|avx2)")),
                };
            }
            "--cfwm" => cfwm = Some(take_value(args, &mut i, "--cfwm")?),
            "--input" => input = Some(take_value(args, &mut i, "--input")?),
            "--jitter-us" => jitter_us = parse_u64(args, &mut i, "--jitter-us")?,
            "--spike-every" => spike_every = parse_u64(args, &mut i, "--spike-every")?,
            "--spike-us" => spike_us = parse_u64(args, &mut i, "--spike-us")?,
            "--seed" => seed = parse_u64(args, &mut i, "--seed")?,
            "--warmup" => warmup = parse_u64(args, &mut i, "--warmup")?,
            "--pacing" => {
                let v = take_value(args, &mut i, "--pacing")?;
                pacing = Pacing::parse(&v)
                    .ok_or_else(|| format!("unknown pacing {v:?} (hybrid|sleep|spin)"))?;
            }
            "--json" => json = Some(take_value(args, &mut i, "--json")?),
            "--progress-seconds" => {
                progress_seconds = parse_f64(args, &mut i, "--progress-seconds")?
            }
            "--priority" => {
                let v = take_value(args, &mut i, "--priority")?;
                high_priority = match v.as_str() {
                    "normal" => false,
                    "high" => true,
                    _ => return Err(format!("unknown priority {v:?} (normal|high)")),
                };
            }
            "--help" | "-h" => return Err(String::new()),
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other:?}"));
            }
            other => {
                if model.is_some() {
                    return Err(format!("unexpected argument {other:?}"));
                }
                model = Some(match other {
                    "voice" => ModelKind::Voice,
                    "echo" => ModelKind::Echo,
                    _ => return Err(format!("unknown model {other:?} (voice|echo)")),
                });
                i += 1;
            }
        }
    }

    let model = model.ok_or_else(|| "missing model (voice|echo)".to_string())?;

    if !period_given {
        // 16 ms is the default cadence; 10 ms is the other documented cadence.
        period_ms = 16.0;
    }
    if !(0.1..=1_000.0).contains(&period_ms) {
        return Err(format!("--period-ms {period_ms} is outside 0.1..1000"));
    }
    let period_ns = (period_ms * 1_000_000.0).round() as u64;
    if period_ns == 0 {
        return Err("period rounds to zero".into());
    }

    let sources = usize::from(soak_minutes.is_some())
        + usize::from(seconds.is_some())
        + usize::from(frames.is_some());
    if sources > 1 {
        return Err("use only one of --soak, --seconds, --frames".into());
    }
    let frames = if let Some(n) = frames {
        n
    } else {
        let secs = if let Some(m) = soak_minutes {
            m * 60.0
        } else if let Some(s) = seconds {
            s
        } else {
            60.0
        };
        if !(secs > 0.0) || !secs.is_finite() {
            return Err("duration must be a positive finite number".into());
        }
        if secs > MAX_SOAK_SECONDS {
            return Err(format!(
                "duration {secs:.1} s exceeds the {MAX_SOAK_SECONDS:.0} s cap"
            ));
        }
        ((secs * 1e9) / period_ns as f64).round() as u64
    };
    if frames == 0 || frames > MAX_FRAMES {
        return Err(format!("frame count {frames} is outside 1..={MAX_FRAMES}"));
    }
    if jitter_us > 5_000_000 {
        return Err("--jitter-us is capped at 5,000,000".into());
    }
    if spike_us > 5_000_000 {
        return Err("--spike-us is capped at 5,000,000".into());
    }
    if spike_every > 0 && spike_us == 0 {
        return Err("--spike-every needs a non-zero --spike-us".into());
    }
    if warmup > 1_000_000 {
        return Err("--warmup is capped at 1,000,000".into());
    }
    if !(progress_seconds >= 0.0) {
        return Err("--progress-seconds must be >= 0".into());
    }

    let cfwm = cfwm.unwrap_or_else(|| match model {
        ModelKind::Voice => DEFAULT_VOICE_CFWM.to_string(),
        ModelKind::Echo => DEFAULT_ECHO_CFWM.to_string(),
    });

    Ok(Options {
        model,
        period_ns,
        frames,
        backend,
        cfwm,
        input,
        jitter_us,
        spike_every,
        spike_us,
        seed,
        warmup,
        pacing,
        json,
        progress_seconds,
        high_priority,
    })
}

fn usage() {
    eprintln!("usage: cf-cli rt voice|echo [--period-ms 16|10] [--soak MINUTES | --seconds S | --frames N]");
    eprintln!("       [--backend auto|scalar|avx2] [--cfwm PATH] [--input PATH]");
    eprintln!("       [--jitter-us N] [--spike-every N --spike-us N] [--seed N]");
    eprintln!("       [--warmup N] [--pacing hybrid|sleep|spin] [--priority normal|high]");
    eprintln!("       [--progress-seconds S] [--json PATH]");
    eprintln!();
    eprintln!("Virtual real-time runner: no audio device is opened.  The thread wakes on");
    eprintln!("an absolute high-resolution schedule, runs one frame, and records deadline");
    eprintln!("misses (xruns), compute latency, wake lateness and the observed RTF.");
}

pub fn cmd_rt(args: &[String]) -> i32 {
    let opts = match parse_options(args) {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}");
            }
            usage();
            return 2;
        }
    };

    match run(opts) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            2
        }
    }
}

// ---------------------------------------------------------------------------
// input
// ---------------------------------------------------------------------------

fn load_input(path: &str, in_len: usize) -> Result<(Vec<f32>, usize), String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let reference = npz::RefData::open(&bytes).map_err(|e| format!("cannot parse {path}: {e}"))?;
    let (data, name) = if let Some(v) = reference.f32("frames") {
        (v, "frames")
    } else if let Some(v) = reference.f32("features") {
        (v, "features")
    } else {
        return Err(format!(
            "{path}: no f32 'frames' or 'features' array (not a feature dump)"
        ));
    };
    if data.len() < in_len || data.len() % in_len != 0 {
        return Err(format!(
            "{path}: {name} has {} values, not a multiple of the {in_len}-value frame",
            data.len()
        ));
    }
    let n = data.len() / in_len;
    if let Some(shape) = reference.shape(name) {
        if shape.len() > 1 {
            let per_frame: usize = shape[1..].iter().product();
            if per_frame != in_len {
                return Err(format!(
                    "{path}: {name} shape {shape:?} is {per_frame} values per frame, expected {in_len}"
                ));
            }
        }
    }
    Ok((data.to_vec(), n))
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn run(opts: Options) -> Result<i32, String> {
    let _timer = platform::TimerResolution::acquire();
    if opts.high_priority && !platform::raise_thread_priority() {
        eprintln!("note: could not raise the thread priority; continuing at normal");
    }

    let bytes = fs::read(&opts.cfwm).map_err(|e| format!("cannot read {}: {e}", opts.cfwm))?;
    let mut engine = match opts.model {
        ModelKind::Voice => Engine::Voice(
            VoiceModel::with_backend(&bytes, opts.backend)
                .map_err(|e| format!("cannot load {}: {e}", opts.cfwm))?,
        ),
        ModelKind::Echo => Engine::Echo(
            EchoModel::with_backend(&bytes, opts.backend)
                .map_err(|e| format!("cannot load {}: {e}", opts.cfwm))?,
        ),
    };

    let in_len = engine.input_len();
    let out_len = engine.output_len();

    let (input_data, input_frames, source) = match &opts.input {
        Some(path) => {
            let (data, n) = load_input(path, in_len)?;
            (data, n, format!("{path} ({n} frames, cycled)"))
        }
        None => (
            Vec::new(),
            0,
            "synthetic (deterministic xorshift)".to_string(),
        ),
    };

    let backend_name = engine.backend_name().to_string();
    let model_id = match opts.model {
        ModelKind::Voice => 1,
        ModelKind::Echo => 2,
    };

    println!("CleanField real-time runner (virtual device)");
    println!(
        "model: {} (id={}) backend={} in={} out={}",
        opts.model.name(),
        model_id,
        backend_name,
        in_len,
        out_len
    );
    println!("device: none - paced thread only, no audio hardware was opened");
    println!(
        "period: {:.3} ms | frames: {} | audio: {:.3} s | warmup: {} frames",
        opts.period_ns as f64 / 1e6,
        opts.frames,
        (opts.frames * opts.period_ns) as f64 / 1e9,
        opts.warmup
    );
    println!(
        "pacing: {} | jitter: {} us | spike: {} us every {} frames | input: {}",
        opts.pacing.name(),
        opts.jitter_us,
        opts.spike_us,
        if opts.spike_every == 0 {
            "never".to_string()
        } else {
            opts.spike_every.to_string()
        },
        source
    );
    if opts.jitter_us > 0 || opts.spike_us > 0 {
        println!("note: injected delays busy-spin, so the CPU is intentionally busy");
    }

    let mut rng = Rng::new(opts.seed);
    let mut input = vec![0.0f32; in_len];
    let mut output = vec![0.0f32; out_len];

    // Warm-up (unpaced): populate caches and recurrent state.
    for w in 0..opts.warmup {
        if input_frames > 0 {
            let base = (w as usize % input_frames) * in_len;
            input.copy_from_slice(&input_data[base..base + in_len]);
        } else {
            for v in input.iter_mut() {
                *v = rng.next_f32();
            }
        }
        engine
            .process(&input, &mut output)
            .map_err(|e| format!("warmup frame {w}: {e}"))?;
    }

    let period = Duration::from_nanos(opts.period_ns);
    let mut compute = Samples::with_capacity(opts.frames);
    let mut callback = Samples::with_capacity(opts.frames);
    let mut wake_late = Samples::with_capacity(opts.frames);

    let mut xruns = 0u64;
    let mut consecutive = 0u64;
    let mut max_consecutive = 0u64;
    let mut max_overrun_ns = 0u64;
    let mut late_wakes_1ms = 0u64;
    let mut checksum = 0u64;
    let mut first_xrun: Option<(u64, u64)> = None;
    let progress_every = if opts.progress_seconds > 0.0 {
        Some(Duration::from_secs_f64(opts.progress_seconds))
    } else {
        None
    };

    let start = Instant::now();
    let mut next_progress = progress_every.map(|every| start + every);

    for n in 0..opts.frames {
        // Prepare the next feature frame before the deadline (host feature
        // extraction is outside the engine's real-time budget).
        if input_frames > 0 {
            let base = (n as usize % input_frames) * in_len;
            input.copy_from_slice(&input_data[base..base + in_len]);
        } else {
            for v in input.iter_mut() {
                *v = rng.next_f32();
            }
        }

        let scheduled = start + period * n as u32;
        wait_until(scheduled, opts.pacing);
        let wake = Instant::now();
        let late = wake.saturating_duration_since(scheduled);
        wake_late.push(late.as_nanos() as u64);
        if late >= Duration::from_millis(1) {
            late_wakes_1ms += 1;
        }

        // Optional jitter / spike injection, after the schedule point and
        // before the engine work (a late callback, not a slow engine).
        let spike = opts.spike_every > 0 && opts.spike_us > 0 && (n + 1) % opts.spike_every == 0;
        let injected_us = if spike {
            opts.spike_us
        } else if opts.jitter_us > 0 {
            rng.next_u64() % (opts.jitter_us + 1)
        } else {
            0
        };
        if injected_us > 0 {
            spin_for(Duration::from_micros(injected_us));
        }

        let t0 = Instant::now();
        engine
            .process(&input, &mut output)
            .map_err(|e| format!("frame {n}: {e}"))?;
        let t1 = Instant::now();

        compute.push((t1 - t0).as_nanos() as u64);
        let cb = t1.saturating_duration_since(scheduled).as_nanos() as u64;
        callback.push(cb);
        for v in &output {
            checksum = checksum.rotate_left(5) ^ u64::from(v.to_bits());
        }

        let deadline = scheduled + period;
        if t1 > deadline {
            xruns += 1;
            consecutive += 1;
            max_consecutive = max_consecutive.max(consecutive);
            let overrun = (t1 - deadline).as_nanos() as u64;
            max_overrun_ns = max_overrun_ns.max(overrun);
            if first_xrun.is_none() {
                first_xrun = Some((n, overrun));
            }
        } else {
            consecutive = 0;
        }

        if let (Some(every), Some(next)) = (progress_every, next_progress) {
            let now = Instant::now();
            if now >= next {
                let elapsed = now.saturating_duration_since(start);
                println!(
                    "progress: audio={:.1}s frames={}/{} xruns={} max_cb={:.3}ms",
                    elapsed.as_secs_f64(),
                    n + 1,
                    opts.frames,
                    xruns,
                    callback.ns.iter().copied().max().unwrap_or(0) as f64 / 1e6
                );
                next_progress = Some(next + every);
            }
        }
    }
    let duration = start.elapsed();

    compute.sort();
    callback.sort();
    wake_late.sort();

    let audio_ns = opts.frames * opts.period_ns; // exact intended audio time
    let audio_s = audio_ns as f64 / 1e9;
    let wall_s = duration.as_secs_f64();
    let compute_sum = compute.sum as f64;
    let compute_rtf = compute_sum / audio_ns as f64;
    let wall_rtf = wall_s / audio_s;
    let speed = audio_ns as f64 / compute_sum;

    let per_minute = if audio_s > 0.0 {
        xruns as f64 / (audio_s / 60.0)
    } else {
        0.0
    };

    println!();
    println!("RESULT (measured)");
    println!(
        "  wall: {:.4} s | intended audio: {:.4} s | wall RTF {:.9} | frames {}",
        wall_s, audio_s, wall_rtf, opts.frames
    );
    println!(
        "  compute: mean {:.3} us | p50 {:.3} | p95 {:.3} | p99 {:.3} | p99.9 {:.3} | max {:.3}",
        compute.mean_ns() / 1e3,
        compute.pct_ns(50.0) as f64 / 1e3,
        compute.pct_ns(95.0) as f64 / 1e3,
        compute.pct_ns(99.0) as f64 / 1e3,
        compute.pct_ns(99.9) as f64 / 1e3,
        compute.max_ns() as f64 / 1e3
    );
    println!(
        "  callback (scheduled start -> output ready): mean {:.3} us | p50 {:.3} | p95 {:.3} | p99 {:.3} | p99.9 {:.3} | max {:.3}",
        callback.mean_ns() / 1e3,
        callback.pct_ns(50.0) as f64 / 1e3,
        callback.pct_ns(95.0) as f64 / 1e3,
        callback.pct_ns(99.0) as f64 / 1e3,
        callback.pct_ns(99.9) as f64 / 1e3,
        callback.max_ns() as f64 / 1e3
    );
    println!(
        "  wake lateness: p50 {:.3} us | p95 {:.3} | p99 {:.3} | p99.9 {:.3} | max {:.3} | >=1ms {}",
        wake_late.pct_ns(50.0) as f64 / 1e3,
        wake_late.pct_ns(95.0) as f64 / 1e3,
        wake_late.pct_ns(99.0) as f64 / 1e3,
        wake_late.pct_ns(99.9) as f64 / 1e3,
        wake_late.max_ns() as f64 / 1e3,
        late_wakes_1ms
    );
    println!(
        "  xruns: {} ({:.4} per minute) | max consecutive {} | max overrun {:.3} ms | first {}",
        xruns,
        per_minute,
        max_consecutive,
        max_overrun_ns as f64 / 1e6,
        match first_xrun {
            Some((n, over)) => format!("frame {n} by {:.3} ms", over as f64 / 1e6),
            None => "none".to_string(),
        }
    );
    println!(
        "  compute RTF: {:.6} ({:.2}x real time) | output checksum {:016x}",
        compute_rtf, speed, checksum
    );
    println!(
        "  deadline model: output of frame n must be ready by T0+(n+1)*period; \
         late wake is not an xrun unless it overruns"
    );

    if let Some(path) = &opts.json {
        let json = format!(
            concat!(
                "{{\n",
                "  \"schema\": \"cleanfield-rt-1\",\n",
                "  \"model\": \"{}\",\n",
                "  \"model_id\": {},\n",
                "  \"backend\": \"{}\",\n",
                "  \"device\": \"virtual\",\n",
                "  \"period_ms\": {:.6},\n",
                "  \"frames\": {},\n",
                "  \"warmup_frames\": {},\n",
                "  \"audio_seconds\": {:.6},\n",
                "  \"wall_seconds\": {:.6},\n",
                "  \"wall_rtf\": {:.9},\n",
                "  \"compute_rtf\": {:.9},\n",
                "  \"speed_factor\": {:.6},\n",
                "  \"pacing\": \"{}\",\n",
                "  \"jitter_us\": {},\n",
                "  \"spike_every\": {},\n",
                "  \"spike_us\": {},\n",
                "  \"input\": \"{}\",\n",
                "  \"xruns\": {},\n",
                "  \"xruns_per_minute\": {:.6},\n",
                "  \"max_consecutive_xruns\": {},\n",
                "  \"max_overrun_ms\": {:.6},\n",
                "  \"late_wakes_1ms\": {},\n",
                "  \"compute_us\": {{\"mean\": {:.6}, \"p50\": {:.6}, \"p95\": {:.6}, \"p99\": {:.6}, \"p99_9\": {:.6}, \"max\": {:.6}}},\n",
                "  \"callback_us\": {{\"mean\": {:.6}, \"p50\": {:.6}, \"p95\": {:.6}, \"p99\": {:.6}, \"p99_9\": {:.6}, \"max\": {:.6}}},\n",
                "  \"wake_late_us\": {{\"p50\": {:.6}, \"p95\": {:.6}, \"p99\": {:.6}, \"p99_9\": {:.6}, \"max\": {:.6}}},\n",
                "  \"checksum\": \"{:016x}\"\n",
                "}}\n"
            ),
            opts.model.name(),
            model_id,
            backend_name,
            opts.period_ns as f64 / 1e6,
            opts.frames,
            opts.warmup,
            audio_s,
            wall_s,
            wall_rtf,
            compute_rtf,
            speed,
            opts.pacing.name(),
            opts.jitter_us,
            opts.spike_every,
            opts.spike_us,
            source.replace('\\', "\\\\").replace('"', "\\\""),
            xruns,
            per_minute,
            max_consecutive,
            max_overrun_ns as f64 / 1e6,
            late_wakes_1ms,
            compute.mean_ns() / 1e3,
            compute.pct_ns(50.0) as f64 / 1e3,
            compute.pct_ns(95.0) as f64 / 1e3,
            compute.pct_ns(99.0) as f64 / 1e3,
            compute.pct_ns(99.9) as f64 / 1e3,
            compute.max_ns() as f64 / 1e3,
            callback.mean_ns() / 1e3,
            callback.pct_ns(50.0) as f64 / 1e3,
            callback.pct_ns(95.0) as f64 / 1e3,
            callback.pct_ns(99.0) as f64 / 1e3,
            callback.pct_ns(99.9) as f64 / 1e3,
            callback.max_ns() as f64 / 1e3,
            wake_late.pct_ns(50.0) as f64 / 1e3,
            wake_late.pct_ns(95.0) as f64 / 1e3,
            wake_late.pct_ns(99.0) as f64 / 1e3,
            wake_late.pct_ns(99.9) as f64 / 1e3,
            wake_late.max_ns() as f64 / 1e3,
            checksum
        );
        fs::write(path, json).map_err(|e| format!("cannot write {path}: {e}"))?;
        println!("  json: {path}");
    }

    Ok(0)
}
