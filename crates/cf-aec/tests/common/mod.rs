#![allow(dead_code)]

//! Deterministic synthetic material and measurements shared by the validation
//! tests and the report example.
//!
//! Everything here is generated from a fixed seed and integer arithmetic, so a
//! run is reproducible across machines and between the scalar and test builds.
//! There is no external audio and no `f64`: the measurements use the same `f32`
//! arithmetic as the filter.

use cf_aec::{Aec, AecConfig, FRAME_LEN};

/// Sample rate assumed throughout the validation, in hertz.
pub const FS: f32 = 16_000.0;

/// Frame duration, in milliseconds.
pub const FRAME_MS: f32 = 16.0;

/// Small deterministic integer PRNG (xorshift32).
pub struct Rng {
    state: u32,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self { state: seed | 1 }
    }

    /// Uniform sample in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        ((x >> 9) as f32) * (1.0 / 4_194_304.0) - 1.0
    }
}

/// Two-pole resonator used to give a pulse train a formant-like spectrum.
struct Resonator {
    pole_1: f32,
    pole_2: f32,
    previous_1: f32,
    previous_2: f32,
}

impl Resonator {
    fn new(frequency: f32, radius: f32) -> Self {
        let angle = 2.0 * std::f32::consts::PI * frequency / FS;
        Self {
            pole_1: 2.0 * radius * angle.cos(),
            pole_2: -(radius * radius),
            previous_1: 0.0,
            previous_2: 0.0,
        }
    }

    fn step(&mut self, input: f32) -> f32 {
        let output = input + self.pole_1 * self.previous_1 + self.pole_2 * self.previous_2;
        self.previous_2 = self.previous_1;
        self.previous_1 = output;
        output
    }
}

/// Speech-like reference: a jittered glottal pulse train plus aspiration,
/// shaped by two drifting formant resonators and a syllabic envelope.
///
/// It is periodic enough to be "speech", non-stationary enough that a delay
/// estimator cannot lock onto a single pitch multiple, and it exercises the
/// 300-3000 Hz band where echo paths live.
pub fn speech_like(len: usize, seed: u32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut voice = Rng::new(seed ^ 0x7F4A_7C15);
    // Every seed gets its own pitch and syllable schedule, so two generated
    // signals with different seeds are uncorrelated (important when one is the
    // reference and one is near-end speech).
    let unit = |value: f32| 0.5 * (value + 1.0);
    let fundamental_base = 95.0 + 55.0 * unit(voice.next_f32());
    let fundamental_rate = 0.45 + 0.8 * unit(voice.next_f32());
    let fundamental_depth = 10.0 + 25.0 * unit(voice.next_f32());
    let syllable_rate = 2.2 + 2.4 * unit(voice.next_f32());
    let phrase_rate = 0.3 + 0.6 * unit(voice.next_f32());
    let formant_1_frequency = 450.0 + 450.0 * unit(voice.next_f32());
    let formant_2_frequency = 1_000.0 + 1_000.0 * unit(voice.next_f32());
    let aspiration = 0.02 + 0.08 * unit(voice.next_f32());

    let mut formant_1 = Resonator::new(formant_1_frequency, 0.975);
    let mut formant_2 = Resonator::new(formant_2_frequency, 0.965);
    let mut high_pass_input = 0.0f32;
    let mut high_pass_output = 0.0f32;
    let mut phase = 0.0f32;
    let mut out = vec![0.0f32; len];

    for (n, sample) in out.iter_mut().enumerate() {
        let t = (n as f32) / FS;
        let fundamental = fundamental_base
            + fundamental_depth * (2.0 * std::f32::consts::PI * fundamental_rate * t).sin();
        let period = FS / fundamental;
        phase += 1.0;
        let mut source = 0.0f32;
        if phase >= period {
            phase -= period;
            source = 1.0;
        }
        source += aspiration * rng.next_f32();
        let voiced = formant_2.step(formant_1.step(source));
        // One-pole DC block: the pulse train has a large DC term.
        high_pass_output = voiced - high_pass_input + 0.995 * high_pass_output;
        high_pass_input = voiced;
        let syllable = (0.5 + 0.5 * (2.0 * std::f32::consts::PI * syllable_rate * t).sin()).powi(2);
        let phrase = 0.55 + 0.45 * (2.0 * std::f32::consts::PI * phrase_rate * t).sin();
        *sample = high_pass_output * syllable * phrase;
    }

    normalize_rms(&mut out, 0.08);
    out
}

/// Synthetic room impulse response: direct tap, a few early reflections and an
/// exponentially decaying diffuse tail, normalised to unit energy.
pub fn synthetic_rir(taps: usize, seed: u32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut impulse = vec![0.0f32; taps];
    impulse[0] = 0.7;
    if taps > 104 {
        impulse[13] = 0.3;
        impulse[29] = -0.22;
        impulse[57] = 0.16;
        impulse[101] = -0.1;
    }
    for k in 1..taps {
        let decay = (-(k as f32) / 90.0).exp();
        impulse[k] += 0.12 * rng.next_f32() * decay;
    }
    let energy: f32 = impulse.iter().map(|v| v * v).sum();
    let scale = 1.0 / energy.sqrt();
    for value in &mut impulse {
        *value *= scale;
    }
    impulse
}

/// Linear convolution, truncated to the input length.
pub fn convolve(input: &[f32], kernel: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for (n, value) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (k, &tap) in kernel.iter().enumerate() {
            if k > n {
                break;
            }
            sum += tap * input[n - k];
        }
        *value = sum;
    }
    out
}

/// A synthetic echo path plus room for near-end speech.
pub struct Scenario {
    /// Reference as fed to the AEC (already carrying the bulk delay).
    pub reference: Vec<f32>,
    /// Microphone as fed to the AEC.
    pub mic: Vec<f32>,
    /// Exact echo component of `mic`.
    pub echo: Vec<f32>,
    /// Exact near-end component of `mic`.
    pub near: Vec<f32>,
    /// Bulk delay inserted between reference and microphone, in samples.
    pub true_delay: usize,
}

impl Scenario {
    /// Builds `frames` frames of an echo-only path with `delay` samples of bulk
    /// delay and a `rir_taps`-sample synthetic room.
    ///
    /// The AEC receives the undelayed reference; the echo path itself carries
    /// the bulk delay. This is the situation the estimator must recover: the
    /// microphone hears the reference `delay` samples later.
    pub fn echo_only(frames: usize, delay: usize, rir_taps: usize, seed: u32) -> Self {
        let len = frames * FRAME_LEN;
        let reference = speech_like(len, seed);
        let mut delayed = vec![0.0f32; len];
        let copied = len.saturating_sub(delay);
        if copied > 0 {
            delayed[delay..].copy_from_slice(&reference[..copied]);
        }
        let rir = synthetic_rir(rir_taps, seed ^ 0x9E37_79B9);
        let echo = convolve(&delayed, &rir);
        Self {
            reference,
            mic: echo.clone(),
            echo,
            near: vec![0.0f32; len],
            true_delay: delay,
        }
    }

    /// Adds near-end speech over `[first_frame, last_frame)` at `level_db`
    /// relative to the echo power in the same interval.
    pub fn add_near_end(
        &mut self,
        first_frame: usize,
        last_frame: usize,
        level_db: f32,
        seed: u32,
    ) -> usize {
        let start = first_frame * FRAME_LEN;
        let end = (last_frame * FRAME_LEN).min(self.mic.len());
        let generated = speech_like(self.mic.len(), seed);
        let echo_rms = rms(&self.echo, start, end);
        let generated_rms = rms(&generated, start, end);
        let gain = echo_rms * 10.0f32.powf(level_db / 20.0) / generated_rms;
        for n in start..end {
            let value = generated[n] * gain;
            self.near[n] = value;
            self.mic[n] += value;
        }
        start
    }
}

/// Per-frame diagnostics captured while replaying a scenario.
#[derive(Clone, Copy, Default)]
pub struct FrameStat {
    pub mic_power: f32,
    pub residual_power: f32,
    pub echo_power: f32,
    pub instantaneous_erle_db: f32,
    pub reported_erle_db: f32,
    pub filter_norm: f32,
    pub double_talk: bool,
    pub delay_samples: usize,
}

/// Replays a scenario through `aec`, returning the concatenated residual and
/// one statistic per frame.
pub fn run(aec: &mut Aec, scenario: &Scenario) -> (Vec<f32>, Vec<FrameStat>) {
    let frames = scenario.mic.len() / FRAME_LEN;
    let mut residual = vec![0.0f32; scenario.mic.len()];
    let mut stats = Vec::with_capacity(frames);
    for frame in 0..frames {
        let start = frame * FRAME_LEN;
        let end = start + FRAME_LEN;
        let out = aec.process(&scenario.mic[start..end], &scenario.reference[start..end]);
        residual[start..end].copy_from_slice(out);

        let mic_power = power(&scenario.mic, start, end);
        let residual_power = power(&residual, start, end);
        let echo_power = power(&scenario.echo, start, end);
        stats.push(FrameStat {
            mic_power,
            residual_power,
            echo_power,
            instantaneous_erle_db: ratio_db(mic_power, residual_power),
            reported_erle_db: aec.erle_db(),
            filter_norm: aec.filter_norm(),
            double_talk: aec.double_talk(),
            delay_samples: aec.delay_samples(),
        });
    }
    (residual, stats)
}

/// Mean power over a sample range.
pub fn power(signal: &[f32], start: usize, end: usize) -> f32 {
    let mut sum = 0.0f32;
    for &value in &signal[start..end] {
        sum += value * value;
    }
    sum / ((end - start) as f32)
}

pub fn rms(signal: &[f32], start: usize, end: usize) -> f32 {
    power(signal, start, end).sqrt()
}

/// `10 log10(numerator / denominator)` with a floor on the denominator.
pub fn ratio_db(numerator: f32, denominator: f32) -> f32 {
    10.0 * ((numerator + 1.0e-20) / (denominator + 1.0e-20)).log10()
}

/// ERLE over a sample range: `10 log10(mic power / residual power)`.
pub fn erle_db(mic: &[f32], residual: &[f32], start: usize, end: usize) -> f32 {
    ratio_db(power(mic, start, end), power(residual, start, end))
}

/// Least-squares gain of the near-end component in the residual, in dB.
///
/// A value of 0 dB means the near-end signal passed through untouched; -1 dB
/// would mean it lost 1 dB of amplitude.
pub fn near_end_gain_db(residual: &[f32], near: &[f32], start: usize, end: usize) -> f32 {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for n in start..end {
        numerator += residual[n] * near[n];
        denominator += near[n] * near[n];
    }
    if denominator <= 0.0 {
        return 0.0;
    }
    20.0 * (numerator / denominator).log10()
}

/// Signal-to-interference ratio in dB over a sample range.
pub fn snr_db(desired: &[f32], interference: &[f32], start: usize, end: usize) -> f32 {
    ratio_db(power(desired, start, end), power(interference, start, end))
}

/// First frame at which the 10-frame ERLE reaches `target_db`, in milliseconds.
///
/// The statistic is the ratio of summed powers over the window, not a mean of
/// per-frame dB values, so silent frames cannot dominate.
pub fn convergence_ms(stats: &[FrameStat], target_db: f32) -> f32 {
    const WINDOW: usize = 10;
    for start in 0..stats.len().saturating_sub(WINDOW) {
        let mut mic = 0.0f32;
        let mut residual = 0.0f32;
        for stat in &stats[start..start + WINDOW] {
            mic += stat.mic_power;
            residual += stat.residual_power;
        }
        if ratio_db(mic, residual) >= target_db {
            return (start as f32) * FRAME_MS;
        }
    }
    f32::NAN
}

/// Find the frame indices `[start, end)` covered by a near-end segment.
pub fn near_end_range(scenario: &Scenario) -> (usize, usize) {
    let mut first = None;
    let mut last = 0usize;
    for (index, &value) in scenario.near.iter().enumerate() {
        if value != 0.0 {
            if first.is_none() {
                first = Some(index);
            }
            last = index;
        }
    }
    match first {
        Some(start) => (start, last + 1),
        None => (0, 0),
    }
}

/// One completed run: scenario, residual, per-frame statistics and the AEC.
pub struct EchoRun {
    pub scenario: Scenario,
    pub residual: Vec<f32>,
    pub stats: Vec<FrameStat>,
    pub aec: Aec,
}

impl EchoRun {
    /// ERLE over whole frames `[first_frame, last_frame)`.
    pub fn erle_db(&self, first_frame: usize, last_frame: usize) -> f32 {
        let start = first_frame * FRAME_LEN;
        let end = (last_frame * FRAME_LEN).min(self.residual.len());
        erle_db(&self.scenario.mic, &self.residual, start, end)
    }
}

/// Runs an echo-only path for `frames` frames with `delay` samples of bulk
/// delay and a 512-tap synthetic room.
pub fn run_echo_only(frames: usize, delay: usize, seed: u32) -> EchoRun {
    let scenario = Scenario::echo_only(frames, delay, 512, seed);
    let mut aec = Aec::new(AecConfig::default());
    let (residual, stats) = run(&mut aec, &scenario);
    EchoRun {
        scenario,
        residual,
        stats,
        aec,
    }
}

/// Runs `frames` frames of echo, with near-end speech over
/// `[near_from, near_to)` at `level_db` relative to the echo.
pub fn run_double_talk(
    frames: usize,
    near_from: usize,
    near_to: usize,
    level_db: f32,
    seed: u32,
) -> EchoRun {
    let mut scenario = Scenario::echo_only(frames, 0, 512, seed);
    scenario.add_near_end(near_from, near_to, level_db, seed ^ 0x51ED_2701);
    let mut aec = Aec::new(AecConfig::default());
    let (residual, stats) = run(&mut aec, &scenario);
    EchoRun {
        scenario,
        residual,
        stats,
        aec,
    }
}

/// Detection rate over the frames where near-end speech is actually present.
///
/// Returns `(flagged, active)`. A frame counts as active when its near-end
/// power is at least 10 % of the segment mean, so syllable pauses (where there
/// is nothing to detect) do not dilute the rate.
pub fn detection_rate(run: &EchoRun, first_frame: usize, last_frame: usize) -> (usize, usize) {
    let mut mean_near = 0.0f32;
    for frame in first_frame..last_frame {
        let start = frame * FRAME_LEN;
        mean_near += power(&run.scenario.near, start, start + FRAME_LEN);
    }
    mean_near /= (last_frame - first_frame) as f32;
    let mut flagged = 0;
    let mut active = 0;
    for frame in first_frame..last_frame {
        let start = frame * FRAME_LEN;
        let near_power = power(&run.scenario.near, start, start + FRAME_LEN);
        if near_power >= 0.1 * mean_near {
            active += 1;
            flagged += usize::from(run.stats[frame].double_talk);
        }
    }
    (flagged, active)
}

/// Scenario seeds used by the report and the validation tests.
pub const REPORT_SEED: u32 = 0x00C0_FFEE;

/// Runs every validation scenario and prints the compact results table.
///
/// The numbers are produced by exactly the same code the tests use, so the
/// table in `README.md` and the asserted thresholds cannot drift apart.
pub fn print_report() {
    use std::time::Instant;

    println!(
        "CleanField linear AEC - functional report\n\
         16 kHz, {FRAME_LEN}-sample frames, 512-point FFT, 8 partitions ({} taps), f32 only\n",
        8 * FRAME_LEN
    );

    // 1. Echo-only cancellation and convergence. The CPU figure is the best of
    // three runs: the suite also runs this function from a parallel test
    // binary, so a single timing would mostly measure scheduler contention.
    let started = Instant::now();
    let mut echo = run_echo_only(600, 0, REPORT_SEED);
    let mut cpu_us = started.elapsed().as_secs_f32() * 1.0e6 / 600.0;
    for _ in 1..3 {
        let started = Instant::now();
        echo = run_echo_only(600, 0, REPORT_SEED);
        cpu_us = cpu_us.min(started.elapsed().as_secs_f32() * 1.0e6 / 600.0);
    }
    let erle_tail = echo.erle_db(400, 600);
    let convergence = convergence_ms(&echo.stats, 10.0);
    let reported = echo.aec.erle_db();

    // 2. Double talk at -6 dB and near-end preservation.
    let talk_frames = 700;
    let talk = run_double_talk(talk_frames, 300, 520, -6.0, REPORT_SEED ^ 0x1111);
    let (near_start, near_end) = near_end_range(&talk.scenario);
    let near_gain = near_end_gain_db(&talk.residual, &talk.scenario.near, near_start, near_end);
    let snr_before = snr_db(
        &talk.scenario.near,
        &talk.scenario.echo,
        near_start,
        near_end,
    );
    let leakage: Vec<f32> = talk
        .residual
        .iter()
        .zip(talk.scenario.near.iter())
        .map(|(residual, near)| residual - near)
        .collect();
    let snr_after = snr_db(&talk.scenario.near, &leakage, near_start, near_end);
    let erle_during = erle_db(&talk.scenario.mic, &talk.residual, near_start, near_end);
    let erle_recovered = talk.erle_db(620, talk_frames);
    let (flagged, active) = detection_rate(&talk, near_start / FRAME_LEN, near_end / FRAME_LEN);
    let detected = 100.0 * flagged as f32 / (active.max(1) as f32);

    // 3. Strong double talk: filter norm must stay bounded.
    let strong_frames = 500;
    let strong = run_double_talk(strong_frames, 150, 350, 3.0, REPORT_SEED ^ 0x2222);
    let norm_before = strong.stats[149].filter_norm;
    let norm_max = strong.stats[150..350]
        .iter()
        .map(|stat| stat.filter_norm)
        .fold(0.0f32, f32::max);
    let erle_strong_after = strong.erle_db(420, strong_frames);
    let (strong_flagged, strong_active) = detection_rate(&strong, 150, 350);
    let strong_detected = 100.0 * strong_flagged as f32 / (strong_active.max(1) as f32);

    // 4. Bulk-delay recovery: 40 ms between the reference and the echo.
    let delay_frames = 400;
    let delayed = run_echo_only(delay_frames, 640, REPORT_SEED ^ 0x3333);
    let delay = delayed.aec.delay_samples();
    let delay_erle = delayed.erle_db(200, delay_frames);

    // 5. Manual delay override.
    let override_delay = 640usize;
    let scenario = Scenario::echo_only(400, override_delay, 512, REPORT_SEED ^ 0x4444);
    let mut fixed = Aec::new(AecConfig {
        delay_override: Some(override_delay),
        ..AecConfig::default()
    });
    let (fixed_residual, _) = run(&mut fixed, &scenario);
    let override_erle = erle_db(
        &scenario.mic,
        &fixed_residual,
        200 * FRAME_LEN,
        400 * FRAME_LEN,
    );

    println!("{:<44} {:>12} {:>14}", "metric", "measured", "target");
    println!("{}", "-".repeat(74));
    row(
        "ERLE, echo-only tail (frames 400-600)",
        &format!("{erle_tail:.1} dB"),
        ">= 15 dB",
    );
    row(
        "Convergence to 10 dB ERLE",
        &format!("{convergence:.0} ms"),
        "report",
    );
    row(
        "Reported ERLE at end of run",
        &format!("{reported:.1} dB"),
        "accessor",
    );
    row(
        "CPU per 16 ms frame (release)",
        &format!("{cpu_us:.1} us"),
        "report",
    );
    row(
        "Near-end gain during -6 dB double talk",
        &format!("{near_gain:+.2} dB"),
        ">= -1 dB",
    );
    row(
        "SNR near/echo before processing",
        &format!("{snr_before:+.1} dB"),
        "report",
    );
    row(
        "SNR near/echo after processing",
        &format!("{snr_after:+.1} dB"),
        "report",
    );
    row(
        "ERLE during -6 dB double talk",
        &format!("{erle_during:.1} dB"),
        "report",
    );
    row(
        "ERLE 100 frames after double talk",
        &format!("{erle_recovered:.1} dB"),
        ">= 12 dB",
    );
    row(
        "DT flag rate while near-end active",
        &format!("{detected:.0} %"),
        "report",
    );
    row(
        "DT flag rate, +3 dB near-end",
        &format!("{strong_detected:.0} %"),
        "report",
    );
    row(
        "Filter norm, before / peak at +3 dB DT",
        &format!("{norm_before:.2} / {norm_max:.2}"),
        "bounded",
    );
    row(
        "ERLE 70 frames after strong DT",
        &format!("{erle_strong_after:.1} dB"),
        ">= 12 dB",
    );
    row(
        "Estimated bulk delay (true 40 ms)",
        &format!("{delay} smp ({:.1} ms)", 1000.0 * delay as f32 / FS),
        "640 +- 256",
    );
    row(
        "ERLE with 40 ms path",
        &format!("{delay_erle:.1} dB"),
        ">= 15 dB",
    );
    row(
        "Manual override / measured delay",
        &format!("{override_delay} / {}", fixed.delay_samples()),
        "exact",
    );
    row(
        "ERLE with manual override",
        &format!("{override_erle:.1} dB"),
        ">= 15 dB",
    );
    println!(
        "\nAEC workspace {} bytes; bulk-delay search window 0..=2048 samples (128 ms). \
         CPU is the best of three release-build runs on the development machine.",
        std::mem::size_of::<Aec>()
    );
}

fn row(name: &str, value: &str, target: &str) {
    println!("{name:<44} {value:>12} {target:>14}");
}

fn normalize_rms(signal: &mut [f32], target: f32) {
    let mut sum = 0.0f32;
    for &value in signal.iter() {
        sum += value * value;
    }
    let current = (sum / (signal.len() as f32)).sqrt();
    if current > 0.0 {
        let scale = target / current;
        for value in signal {
            *value *= scale;
        }
    }
}
