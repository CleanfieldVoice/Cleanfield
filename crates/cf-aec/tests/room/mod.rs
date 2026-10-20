#![allow(dead_code)]

//! Realistic room-condition scene generator and measurements for the linear AEC.
//!
//! The module is test-only. It is included by the `room_conditions` integration
//! test, by the `room_report` example that produces `ROBUSTNESS.md`, and by the
//! echo-pipeline room test through a `#[path]` include. Nothing here is part of
//! the public `cf-aec` API.
//!
//! A scene is fully determined by its [`SceneSpec`] and a `u32` seed:
//!
//! * the far-end reference is a speech-like source-filter signal;
//! * the echo path is a fractional-delay reference through a unit-energy room
//!   response, with a static, stepping or slowly drifting bulk delay;
//! * optional background noise is white, pink or four-talker babble at a chosen
//!   echo-to-noise ratio;
//! * optional near-end speech overlaps the echo for a fixed 2.88 s window at a
//!   chosen near-to-echo ratio.
//!
//! Measurements use the generator's ground truth (`echo`, `noise`, `near`) to
//! separate cancellation from the noise floor. That separation matters: at
//! 5 dB echo-to-noise ratio even a perfect canceller can only reach about 6 dB
//! of raw microphone-to-residual ERLE.

mod fft;
mod matrix;
mod real;
mod real_rir;
mod rir;
mod signal;

#[allow(unused_imports)]
pub use matrix::{scenes, SEEDS};
#[allow(unused_imports)]
pub use real::{c50_db as real_c50_db, rt20_s as real_rt20_s, RealRir, ALL as REAL_RIRS};
#[allow(unused_imports)]
pub use real_rir::{
    run_matrix as run_real_matrix, scenes as real_scenes, RealCondition, RealSceneRuns,
};
#[allow(unused_imports)]
pub use rir::{profile as rir_profile, RirKind};
pub use signal::{noise_like, speech_like, NoiseKind};

use cf_aec::{Aec, AecConfig, FRAME_LEN};

/// Sample rate in hertz.
pub const SAMPLE_RATE: usize = 16_000;
/// Sample rate as `f32`, used by the generators.
pub const FS: f32 = 16_000.0;
/// Frame length in samples (one AEC frame = 16 ms).
pub const FRAME: usize = FRAME_LEN;
/// Scene length in frames (640 frames = 10.24 s).
pub const SCENE_FRAMES: usize = 640;
/// Static bulk delay used by the clean scenes, in samples (30 ms).
pub const STATIC_DELAY: usize = 480;
/// Step change applied at [`STEP_FRAME`], in samples (+5 ms).
pub const STEP_DELTA: usize = 80;
/// Frame at which the delay step happens (4.8 s).
pub const STEP_FRAME: usize = 300;
/// Slow clock drift, in parts per million over the file.
pub const DRIFT_PPM: f32 = 200.0;
/// First frame of the near-end segment (3.84 s).
pub const DT_ONSET_FRAME: usize = 240;
/// First frame after the near-end segment (6.72 s).
pub const DT_END_FRAME: usize = 420;
/// End of the pre-event measurement window (3.84 s).
pub const PRE_END_FRAME: usize = DT_ONSET_FRAME;
/// First frame of the pre-event measurement window (2.40 s).
pub const PRE_START_FRAME: usize = 150;
/// First frame of the post-event measurement window (7.68 s).
pub const POST_START_FRAME: usize = 480;
/// End of the post-event measurement window (9.92 s).
pub const POST_END_FRAME: usize = 620;
/// Frames used by the trailing ERLE window for recovery timing.
pub const RECOVERY_WINDOW: usize = 10;
/// A filter norm above this value is treated as divergence. A unit-energy room
/// needs a filter norm around one, so this is an order-of-magnitude bound, not a
/// tuned threshold.
pub const FILTER_NORM_LIMIT: f32 = 10.0;

/// Bulk-delay behaviour over the file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DelayKind {
    /// Constant delay.
    Static,
    /// Constant, then +5 ms from [`STEP_FRAME`] onwards.
    Step,
    /// A linear drift, in parts per million over the file. Positive means the
    /// echo delay grows (the far-end clock runs slower than the microphone
    /// clock). [`DRIFT_PPM`] is the 200 ppm scene the matrix uses.
    Drift(f32),
}

impl DelayKind {
    pub fn label(self) -> String {
        match self {
            Self::Static => "static".to_string(),
            Self::Step => "step+5ms".to_string(),
            Self::Drift(ppm) => format!("drift{ppm:.0}ppm"),
        }
    }

    /// Clock drift in ppm, if this is a drift condition.
    pub fn drift_ppm(self) -> Option<f32> {
        match self {
            Self::Drift(ppm) => Some(ppm),
            _ => None,
        }
    }
}

/// One scene definition. The seed is supplied to [`Scene::generate`].
#[derive(Clone, Debug)]
pub struct SceneSpec {
    pub name: String,
    pub rir: RirKind,
    /// When set, the scene uses that bundled measured response instead of
    /// `rir`. `rir` is then only a fallback for code paths that aggregate by
    /// synthetic family; real scenes are selected by this field.
    pub real_rir: Option<&'static str>,
    /// `Some((kind, snr_db))`: noise RMS is set relative to the echo RMS.
    pub noise: Option<(NoiseKind, f32)>,
    pub delay: DelayKind,
    /// `Some(db)`: near-end speech at `db` relative to the echo, over
    /// `[DT_ONSET_FRAME, DT_END_FRAME)`.
    pub near_db: Option<f32>,
}

impl SceneSpec {
    pub fn new(name: impl Into<String>, rir: RirKind) -> Self {
        Self {
            name: name.into(),
            rir,
            real_rir: None,
            noise: None,
            delay: DelayKind::Static,
            near_db: None,
        }
    }

    /// A scene whose echo path is the bundled measured response `id`.
    ///
    /// The response is fixed; the seed still changes the talkers and noise, so
    /// three seeds are three different speech/noise realisations of the same
    /// room.
    pub fn new_real(name: impl Into<String>, id: &'static str) -> Self {
        Self {
            name: name.into(),
            rir: RirKind::Office,
            real_rir: Some(id),
            noise: None,
            delay: DelayKind::Static,
            near_db: None,
        }
    }

    pub fn with_noise(mut self, kind: NoiseKind, snr_db: f32) -> Self {
        self.noise = Some((kind, snr_db));
        self
    }

    pub fn with_delay(mut self, delay: DelayKind) -> Self {
        self.delay = delay;
        self
    }

    pub fn with_near_end(mut self, level_db: f32) -> Self {
        self.near_db = Some(level_db);
        self
    }

    /// Text naming the room in the report's condition column.
    pub fn room_label(&self) -> &str {
        self.real_rir.unwrap_or_else(|| self.rir.label())
    }

    /// Text used in the report's scene column.
    pub fn condition_text(&self) -> String {
        let noise = match self.noise {
            Some((kind, snr)) => format!("{} {snr:.0} dB", kind.label()),
            None => "no noise".to_string(),
        };
        let double_talk = match self.near_db {
            Some(level) => format!("DT {level:+.0} dB"),
            None => "no DT".to_string(),
        };
        format!(
            "{} / {noise} / {} / {double_talk}",
            self.room_label(),
            self.delay.label()
        )
    }
}

/// Builds a real-RIR scene with the standard 30 ms static bulk delay.
pub fn real_scene(
    id: &'static str,
    noise: Option<(NoiseKind, f32)>,
    near_db: Option<f32>,
) -> SceneSpec {
    let mut spec = SceneSpec::new_real(id, id);
    spec.noise = noise;
    spec.near_db = near_db;
    spec
}

/// A generated scene: the signals fed to the AEC and the ground-truth parts.
pub struct Scene {
    pub spec: SceneSpec,
    pub seed: u32,
    pub reference: Vec<f32>,
    pub mic: Vec<f32>,
    /// Exact echo component of `mic`.
    pub echo: Vec<f32>,
    /// Exact background-noise component of `mic`.
    pub noise: Vec<f32>,
    /// Exact near-end component of `mic`.
    pub near: Vec<f32>,
    /// The room response used, before the bulk delay.
    pub rir: Vec<f32>,
    /// Gain applied to the whole scene to keep the input peak below 0.95.
    pub input_scale: f32,
    /// True bulk delay at the end of the file, in samples.
    pub final_delay: f32,
}

impl Scene {
    pub fn generate(spec: SceneSpec, seed: u32) -> Self {
        let len = SCENE_FRAMES * FRAME;
        let reference = speech_like(len, seed ^ 0xC001_D00D);
        let rir = match spec.real_rir {
            Some(id) => real::find(id).impulse(),
            None => spec.rir.build(seed ^ 0x1234_5678),
        };

        let delay_at = |n: usize| -> f32 {
            match spec.delay {
                DelayKind::Static => STATIC_DELAY as f32,
                DelayKind::Step => {
                    if n < STEP_FRAME * FRAME {
                        STATIC_DELAY as f32
                    } else {
                        (STATIC_DELAY + STEP_DELTA) as f32
                    }
                }
                DelayKind::Drift(ppm) => STATIC_DELAY as f32 + ppm * 1.0e-6 * n as f32,
            }
        };
        let delayed = signal::fractional_delay(&reference, delay_at);
        let mut echo = fft::convolve(&delayed, &rir, len);

        let echo_rms = signal::rms(&echo);
        let mut noise = vec![0.0f32; len];
        if let Some((kind, snr_db)) = spec.noise {
            let mut generated = noise_like(len, kind, seed);
            let generated_rms = signal::rms(&generated);
            let gain = echo_rms / 10.0f32.powf(snr_db / 20.0) / generated_rms;
            for value in &mut generated {
                *value *= gain;
            }
            noise = generated;
        }

        let mut near = vec![0.0f32; len];
        if let Some(level_db) = spec.near_db {
            let start = DT_ONSET_FRAME * FRAME;
            let end = DT_END_FRAME * FRAME;
            let generated = speech_like(len, seed ^ 0x5151_5151);
            let fade = (0.010 * FS) as usize;
            for n in start..end {
                let rising = n - start;
                let falling = end - 1 - n;
                let ramp_in = if rising < fade {
                    0.5 - 0.5 * (std::f32::consts::PI * rising as f32 / fade as f32).cos()
                } else {
                    1.0
                };
                let ramp_out = if falling < fade {
                    0.5 - 0.5 * (std::f32::consts::PI * falling as f32 / fade as f32).cos()
                } else {
                    1.0
                };
                near[n] = generated[n] * ramp_in.min(ramp_out);
            }
            let echo_rms = signal::rms(&echo[start..end]);
            let near_rms = signal::rms(&near[start..end]);
            let gain = echo_rms * 10.0f32.powf(level_db / 20.0) / near_rms;
            for value in &mut near {
                *value *= gain;
            }
        }

        let final_delay = delay_at(len - 1);
        let mut mic: Vec<f32> = (0..len).map(|n| echo[n] + noise[n] + near[n]).collect();
        let guard_peak = 0.95;
        let peak = signal::peak(&mic);
        let input_scale = if peak > guard_peak {
            guard_peak / peak
        } else {
            1.0
        };
        if input_scale != 1.0 {
            for value in &mut mic {
                *value *= input_scale;
            }
            for value in &mut noise {
                *value *= input_scale;
            }
            for value in &mut near {
                *value *= input_scale;
            }
            for value in &mut echo {
                *value *= input_scale;
            }
            // The reference must be scaled with the echo because the echo was
            // produced from it; the ratio between them is unchanged.
            let mut scaled = reference;
            for value in &mut scaled {
                *value *= input_scale;
            }
            return Self {
                spec,
                seed,
                reference: scaled,
                mic,
                echo,
                noise,
                near,
                rir,
                input_scale,
                final_delay,
            };
        }

        Self {
            spec,
            seed,
            reference,
            mic,
            echo,
            noise,
            near,
            rir,
            input_scale,
            final_delay,
        }
    }
}

/// Per-frame diagnostics captured while replaying a scene.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStat {
    pub mic_power: f32,
    pub residual_power: f32,
    pub echo_power: f32,
    pub noise_power: f32,
    pub near_power: f32,
    pub filter_norm: f32,
    pub reported_erle_db: f32,
    pub double_talk: bool,
    pub delay_samples: usize,
    /// Clock-rate correction applied on this frame, in ppm (0 while bypassed).
    pub clock_ppm: f32,
    /// Whether the clock-rate correction was active on this frame.
    pub clock_corrected: bool,
}

/// Everything measured for one `(scene, seed)` run.
#[derive(Clone, Debug)]
pub struct Metrics {
    /// Raw microphone-to-residual ERLE before the event, in dB.
    pub erle_pre_db: f32,
    /// Raw microphone-to-residual ERLE after the event, in dB.
    pub erle_post_db: f32,
    /// Echo-cancellation ERLE before the event, with the known noise removed
    /// from the residual: `10 log10(P_echo / P_echo-residual)`.
    pub echo_erle_pre_db: f32,
    /// Echo-cancellation ERLE after the event, noise removed.
    pub echo_erle_post_db: f32,
    /// Time from the end of double talk until the 10-frame echo-cancellation
    /// ERLE is within 3 dB of its pre-double-talk value. `None` when the scene
    /// has no double talk or the filter did not recover inside the measurement
    /// window; `has_double_talk` distinguishes the two.
    pub recovery_ms: Option<f32>,
    pub has_double_talk: bool,
    /// Least-squares gain of the near-end signal in the residual, in dB.
    pub near_gain_db: Option<f32>,
    /// Near-to-interference ratio improvement over the double-talk segment, in dB.
    pub near_sir_gain_db: Option<f32>,
    pub norm_peak: f32,
    pub norm_final: f32,
    pub delay_final: usize,
    pub delay_target_final: f32,
    /// Clock-rate correction applied on the last frame, in ppm.
    pub clock_ppm_final: f32,
    /// First frame on which the clock-rate correction was active, if it ever
    /// was. `None` for a static path, where the correction stays bypassed.
    pub clock_engaged_frame: Option<usize>,
    /// Mean absolute difference between the applied rate and the scene's true
    /// drift over the post-event window, in ppm. `None` without drift.
    ///
    /// It includes the frames before the correction engages (applied rate 0),
    /// so it is an upper bound on the steady-state rate error; the post window
    /// starts long after the acquisition in every reported run.
    pub clock_residual_ppm_mean_abs: Option<f32>,
    /// Time from the delay step until a new applied delay is reported.
    pub delay_change_ms: Option<f32>,
    /// Percentage of near-end-active frames in the double-talk segment on which
    /// the detector froze adaptation. `None` without double talk.
    pub dt_flag_rate: Option<f32>,
    /// Wall-clock time per 16 ms frame for this run, in milliseconds. Report
    /// only: it is machine-dependent and is not asserted against a budget.
    pub cpu_ms_per_frame: f32,
    pub has_nan: bool,
    pub diverged: bool,
    pub collapsed: bool,
}

/// A scene, the residual and the per-frame statistics.
pub struct SceneRun {
    pub scene: Scene,
    pub residual: Vec<f32>,
    pub stats: Vec<FrameStat>,
    pub metrics: Metrics,
}

impl SceneRun {
    pub fn erle_db(&self, first_frame: usize, last_frame: usize) -> f32 {
        let start = first_frame * FRAME;
        let end = (last_frame * FRAME).min(self.residual.len());
        ratio_db(
            power(&self.scene.mic[start..end]),
            power(&self.residual[start..end]),
        )
    }

    pub fn echo_erle_db(&self, first_frame: usize, last_frame: usize) -> f32 {
        let start = first_frame * FRAME;
        let end = (last_frame * FRAME).min(self.residual.len());
        let mut echo_power = 0.0f32;
        let mut error_power = 0.0f32;
        for n in start..end {
            echo_power += self.scene.echo[n] * self.scene.echo[n];
            let error = self.residual[n] - self.scene.noise[n];
            error_power += error * error;
        }
        ratio_db(echo_power, error_power)
    }
}

/// Generates and replays one scene with the AEC default configuration.
pub fn run_scene(spec: &SceneSpec, seed: u32) -> SceneRun {
    run_scene_with_config(spec, seed, AecConfig::default())
}

/// Generates and replays one scene with an explicit AEC configuration.
pub fn run_scene_with_config(spec: &SceneSpec, seed: u32, config: AecConfig) -> SceneRun {
    let scene = Scene::generate(spec.clone(), seed);
    let mut aec = Aec::new(config);
    let frames = scene.mic.len() / FRAME;
    let mut residual = vec![0.0f32; scene.mic.len()];
    let mut stats = Vec::with_capacity(frames);
    let started = std::time::Instant::now();
    for frame in 0..frames {
        let start = frame * FRAME;
        let end = start + FRAME;
        let output = aec.process(&scene.mic[start..end], &scene.reference[start..end]);
        residual[start..end].copy_from_slice(output);
        stats.push(FrameStat {
            mic_power: power(&scene.mic[start..end]),
            residual_power: power(&residual[start..end]),
            echo_power: power(&scene.echo[start..end]),
            noise_power: power(&scene.noise[start..end]),
            near_power: power(&scene.near[start..end]),
            filter_norm: aec.filter_norm(),
            reported_erle_db: aec.erle_db(),
            double_talk: aec.double_talk(),
            delay_samples: aec.delay_samples(),
            clock_ppm: aec.clock_rate_ppm(),
            clock_corrected: aec.clock_rate_corrected(),
        });
    }
    let cpu_ms_per_frame = started.elapsed().as_secs_f64() * 1_000.0 / frames as f64;
    let metrics = measure(&scene, &residual, &stats, cpu_ms_per_frame as f32);
    SceneRun {
        scene,
        residual,
        stats,
        metrics,
    }
}

fn measure(scene: &Scene, residual: &[f32], stats: &[FrameStat], cpu_ms_per_frame: f32) -> Metrics {
    let erle_pre_db = window_erle(&scene.mic, residual, PRE_START_FRAME, PRE_END_FRAME);
    let erle_post_db = window_erle(&scene.mic, residual, POST_START_FRAME, POST_END_FRAME);
    let echo_erle_pre_db = window_echo_erle(scene, residual, PRE_START_FRAME, PRE_END_FRAME);
    let echo_erle_post_db = window_echo_erle(scene, residual, POST_START_FRAME, POST_END_FRAME);

    let has_double_talk = scene.spec.near_db.is_some();
    let recovery_ms = if has_double_talk {
        recovery_time(scene, residual, echo_erle_pre_db)
    } else {
        None
    };

    let (near_gain_db, near_sir_gain_db) = if has_double_talk {
        let start = DT_ONSET_FRAME * FRAME;
        let end = DT_END_FRAME * FRAME;
        let gain = projection_gain_db(&residual[start..end], &scene.near[start..end]);
        let before = ratio_db(
            power(&scene.near[start..end]),
            power(&scene.echo[start..end]) + power(&scene.noise[start..end]),
        );
        let leak: Vec<f32> = residual[start..end]
            .iter()
            .zip(scene.near[start..end].iter())
            .map(|(r, n)| r - n)
            .collect();
        let after = ratio_db(power(&scene.near[start..end]), power(&leak));
        (Some(gain), Some(after - before))
    } else {
        (None, None)
    };

    let norm_peak = stats
        .iter()
        .map(|stat| stat.filter_norm)
        .fold(0.0f32, f32::max);
    let norm_final = stats.last().map_or(0.0, |stat| stat.filter_norm);
    let delay_final = stats.last().map_or(0, |stat| stat.delay_samples);
    let has_nan = !residual.iter().all(|value| value.is_finite())
        || stats.iter().any(|stat| !stat.filter_norm.is_finite());
    // Collapse is a steady-state property: the pre-double-talk window also
    // measures convergence, and a slow start is not a collapse. The post window
    // is echo-only and late enough for the filter to have settled.
    let echoed_collapse = echo_erle_post_db < 3.0;
    let diverged = has_nan
        || !norm_peak.is_finite()
        || norm_peak > FILTER_NORM_LIMIT
        || echo_erle_pre_db < -6.0
        || echo_erle_post_db < -6.0;

    let delay_change_ms = if scene.spec.delay == DelayKind::Step {
        let before = stats
            .get(STEP_FRAME - 1)
            .map_or(0, |stat| stat.delay_samples);
        stats
            .iter()
            .enumerate()
            .skip(STEP_FRAME)
            .find(|(_, stat)| stat.delay_samples != before)
            .map(|(frame, _)| (frame - STEP_FRAME) as f32 * 16.0)
    } else {
        None
    };

    let mut near_mean = 0.0f32;
    for frame in DT_ONSET_FRAME..DT_END_FRAME {
        near_mean += stats[frame].near_power;
    }
    near_mean /= (DT_END_FRAME - DT_ONSET_FRAME) as f32;
    let mut near_active = 0usize;
    let mut near_flagged = 0usize;
    for frame in DT_ONSET_FRAME..DT_END_FRAME {
        if stats[frame].near_power >= 0.1 * near_mean {
            near_active += 1;
            near_flagged += usize::from(stats[frame].double_talk);
        }
    }
    let dt_flag_rate = if has_double_talk && near_active > 0 {
        Some(100.0 * near_flagged as f32 / near_active as f32)
    } else {
        None
    };

    let clock_ppm_final = stats.last().map_or(0.0, |stat| stat.clock_ppm);
    let clock_engaged_frame = stats.iter().position(|stat| stat.clock_corrected);
    let clock_residual_ppm_mean_abs = scene.spec.delay.drift_ppm().map(|true_ppm| {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for stat in &stats[POST_START_FRAME..POST_END_FRAME.min(stats.len())] {
            sum += (stat.clock_ppm - true_ppm).abs();
            count += 1;
        }
        if count > 0 {
            sum / count as f32
        } else {
            f32::NAN
        }
    });

    Metrics {
        erle_pre_db,
        erle_post_db,
        echo_erle_pre_db,
        echo_erle_post_db,
        recovery_ms,
        has_double_talk,
        near_gain_db,
        near_sir_gain_db,
        norm_peak,
        norm_final,
        delay_final,
        delay_target_final: scene.final_delay,
        clock_ppm_final,
        clock_engaged_frame,
        clock_residual_ppm_mean_abs,
        delay_change_ms,
        dt_flag_rate,
        cpu_ms_per_frame,
        has_nan,
        diverged,
        collapsed: echoed_collapse,
    }
}

fn recovery_time(scene: &Scene, residual: &[f32], target_pre_db: f32) -> Option<f32> {
    let target = (target_pre_db - 3.0).max(3.0);
    let end = POST_END_FRAME.saturating_sub(RECOVERY_WINDOW);
    for frame in DT_END_FRAME..end {
        let value = window_echo_erle(scene, residual, frame, frame + RECOVERY_WINDOW);
        if value >= target {
            return Some((frame - DT_END_FRAME) as f32 * 16.0);
        }
    }
    None
}

fn window_erle(mic: &[f32], residual: &[f32], first_frame: usize, last_frame: usize) -> f32 {
    let start = first_frame * FRAME;
    let end = (last_frame * FRAME).min(mic.len());
    ratio_db(power(&mic[start..end]), power(&residual[start..end]))
}

/// Echo-cancellation ERLE over a frame window, with the known noise component
/// removed from the residual. Shared with the bottleneck-split experiment so
/// its numbers use exactly the same definition as `ROBUSTNESS.md`.
pub fn window_echo_erle(
    scene: &Scene,
    residual: &[f32],
    first_frame: usize,
    last_frame: usize,
) -> f32 {
    let start = first_frame * FRAME;
    let end = (last_frame * FRAME).min(residual.len());
    let mut echo_power = 0.0f32;
    let mut error_power = 0.0f32;
    for n in start..end {
        echo_power += scene.echo[n] * scene.echo[n];
        let error = residual[n] - scene.noise[n];
        error_power += error * error;
    }
    ratio_db(echo_power, error_power)
}

/// Least-squares amplitude gain of `near` in `residual`, in dB.
fn projection_gain_db(residual: &[f32], near: &[f32]) -> f32 {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (r, n) in residual.iter().zip(near.iter()) {
        numerator += r * n;
        denominator += n * n;
    }
    if denominator <= 0.0 {
        return 0.0;
    }
    20.0 * (numerator / denominator).abs().max(1.0e-12).log10()
}

pub fn power(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    signal.iter().map(|value| value * value).sum::<f32>() / signal.len() as f32
}

pub fn ratio_db(numerator: f32, denominator: f32) -> f32 {
    10.0 * ((numerator + 1.0e-20) / (denominator + 1.0e-20)).log10()
}

/// Mean/min/max of the finite values in `values`.
pub fn mean_min_max(values: &[f32]) -> Option<(f32, f32, f32)> {
    let mut count = 0usize;
    let mut sum = 0.0f32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &value in values {
        if value.is_finite() {
            count += 1;
            sum += value;
            min = min.min(value);
            max = max.max(value);
        }
    }
    if count == 0 {
        None
    } else {
        Some((sum / count as f32, min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scene_is_reproducible_from_its_seed() {
        let spec = SceneSpec::new("repro", RirKind::Office)
            .with_noise(NoiseKind::Pink, 15.0)
            .with_delay(DelayKind::Drift(DRIFT_PPM))
            .with_near_end(-6.0);
        let first = Scene::generate(spec.clone(), 0xABCD);
        let second = Scene::generate(spec, 0xABCD);
        assert_eq!(first.reference, second.reference);
        assert_eq!(first.mic, second.mic);
        assert_eq!(first.echo, second.echo);
        assert_eq!(first.near, second.near);
    }

    #[test]
    fn echo_matches_the_requested_snr() {
        for kind in NoiseKind::ALL {
            let spec = SceneSpec::new("snr", RirKind::ShortDirect).with_noise(kind, 15.0);
            let scene = Scene::generate(spec, 0x1234);
            let measured = ratio_db(
                signal::rms(&scene.echo).powi(2),
                signal::rms(&scene.noise).powi(2),
            );
            assert!((measured - 15.0).abs() < 0.1, "{kind:?}: {measured}");
        }
    }

    #[test]
    fn near_end_level_is_relative_to_the_echo() {
        for level in [-6.0f32, 0.0, 3.0] {
            let spec = SceneSpec::new("dt", RirKind::Office).with_near_end(level);
            let scene = Scene::generate(spec, 0x4321);
            let start = DT_ONSET_FRAME * FRAME;
            let end = DT_END_FRAME * FRAME;
            let measured = ratio_db(
                signal::rms(&scene.near[start..end]).powi(2),
                signal::rms(&scene.echo[start..end]).powi(2),
            );
            assert!((measured - level).abs() < 0.1, "{level}: {measured}");
        }
    }

    #[test]
    fn a_clean_scene_keeps_the_input_below_full_scale() {
        for rir in RirKind::ALL {
            let scene = Scene::generate(SceneSpec::new("peak", rir), 0x7777);
            assert!(signal::peak(&scene.mic) <= 0.95 + 1.0e-6);
            assert!(scene.input_scale > 0.0 && scene.input_scale.is_finite());
        }
    }
}
