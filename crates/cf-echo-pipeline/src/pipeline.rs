//! Two-stage echo cancellation: linear AEC, then the neural echo model.
//!
//! The front end mirrors the documented reference pipeline:
//!
//! 1. the linear AEC returns the residual for the current 256-sample frame and
//!    exposes the reference advanced by the applied bulk delay;
//! 2. the residual, the microphone and the aligned reference are pushed through
//!    a 768-sample periodic-Hann / 1024-point-FFT / 256-hop analysis stage;
//! 3. per bin, the smaller-power of the residual and microphone spectra is
//!    selected (full complex value, phase preserved);
//! 4. the model input is three 513-bin vectors in the order
//!    `[residual, microphone, aligned reference]`, each transformed as
//!    `0.5 * log10(power * preemphasis + 1e-16)`;
//! 5. the 513-value model output is clamped to `[0, 1]` and applied to the
//!    selected spectrum of the **previous** hop (one-hop look-ahead);
//! 6. the masked spectrum is inverse transformed, windowed, overlap-added and
//!    emitted with the same 256-sample framing.
//!
//! The stage adds [`LATENCY_SAMPLES`] = 1024 samples (64 ms): the 768-sample
//! analysis window plus one 256-sample look-ahead hop. The first 1024 output
//! samples are therefore warm-up silence.

use cf_aec::{Aec, AecConfig, MAX_FRAME_LEN};
use cf_echo::{EchoError, EchoModel, INPUT_LEN};

use crate::fft::{Fft, BINS, FFT_SIZE, WINDOW_LEN};

/// Native sample rate of the pipeline, in hertz.
pub const SAMPLE_RATE: usize = 16_000;

/// Frame length: 256 samples = 16 ms at [`SAMPLE_RATE`].
pub const FRAME_LEN: usize = cf_aec::FRAME_LEN;

/// Analysis hop: one frame.
pub const HOP_LEN: usize = FRAME_LEN;

/// Analysis/synthesis window length.
pub const WINDOW_SIZE: usize = WINDOW_LEN;

/// Bins kept from the 1024-point FFT.
pub const FEATURE_BINS: usize = BINS;

/// Model input channels: residual, microphone, aligned reference.
pub const INPUT_CHANNELS: usize = 3;

/// One model input frame (`3 x 513`).
pub const FEATURE_LEN: usize = INPUT_LEN;

/// Number of hops the learned mask looks ahead.
pub const LOOKAHEAD_HOPS: usize = 1;

/// Overlap-add normalisation: the periodic Hann window squared sums to `9/8`.
pub const SYNTHESIS_SCALE: f32 = 8.0 / 9.0;

/// Feature denominator floor.
pub const EPSILON: f32 = 1.0e-16;

/// Algorithmic latency added by the spectral stage, in samples.
pub const LATENCY_SAMPLES: usize = WINDOW_SIZE + LOOKAHEAD_HOPS * HOP_LEN;

/// Largest accepted `process` block, in samples (8 frames).
pub const MAX_BLOCK_LEN: usize = MAX_FRAME_LEN;

const _: () = assert!(FRAME_LEN == 256);
const _: () = assert!(FFT_SIZE == 1024);
const _: () = assert!(WINDOW_SIZE == 768);
const _: () = assert!(FEATURE_BINS == 513);
const _: () = assert!(FEATURE_LEN == INPUT_CHANNELS * FEATURE_BINS);

/// Tuning of one [`EchoPipeline`].
#[derive(Clone, Copy, Debug)]
pub struct EchoPipelineConfig {
    /// Linear AEC tuning.
    pub aec: AecConfig,
    /// Run the linear AEC stage. When `false` the residual channel is the raw
    /// microphone and the aligned-reference channel is the raw reference.
    pub aec_enabled: bool,
    /// Run the neural mask stage. When `false` the output is the linear
    /// residual with no spectral delay.
    pub neural_enabled: bool,
    /// Learned-mask strength in `[0, 1]`: `effective = (1 - strength) +
    /// strength * mask`. Out-of-range and non-finite values are clamped.
    pub strength: f32,
}

impl Default for EchoPipelineConfig {
    fn default() -> Self {
        Self {
            // The AEC kernel default (`1e-2`) is tuned for its own synthetic
            // suite. On speech it converges too slowly to satisfy the 48-frame
            // double-talk warm-up, after which the detector freezes adaptation;
            // measured on the project's real capture, `1e-5` lifts the linear
            // echo-only ERLE from 1.0 dB to 25.9 dB without a near-end penalty.
            // This is a configuration choice only; the kernels are unchanged.
            aec: AecConfig {
                step_size: 0.5,
                regularization: 1.0e-5,
                delay_override: None,
            },
            aec_enabled: true,
            neural_enabled: true,
            strength: 1.0,
        }
    }
}

impl EchoPipelineConfig {
    fn normalized(mut self) -> Self {
        self.strength = if self.strength.is_finite() {
            self.strength.max(0.0).min(1.0)
        } else {
            1.0
        };
        self
    }
}

/// Pipeline construction failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineError {
    /// The echo model container could not be loaded.
    Model(EchoError),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Model(error) => write!(f, "echo model: {error}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<EchoError> for PipelineError {
    fn from(error: EchoError) -> Self {
        PipelineError::Model(error)
    }
}

/// Stateful end-to-end echo canceller.
///
/// One instance per stream. [`EchoPipeline::process`] accepts a whole number of
/// 256-sample frames, returns the same number of samples, and does not allocate.
/// Large buffers are heap-allocated once by [`EchoPipeline::new`] so the value
/// itself stays small enough to pass by value on a host's audio thread stack.
pub struct EchoPipeline {
    config: EchoPipelineConfig,
    aec: Box<Aec>,
    model: EchoModel,
    fft: Fft,

    window: [f32; WINDOW_SIZE],
    weights: [f32; FEATURE_BINS],

    /// `[channel][window sample]`, channel-major.
    analysis: Vec<f32>,
    /// `[channel][bin]` spectra and powers.
    spec_re: Vec<f32>,
    spec_im: Vec<f32>,
    powers: Vec<f32>,
    overlap: Vec<f32>,
    selected_re: Vec<f32>,
    selected_im: Vec<f32>,
    previous_re: Vec<f32>,
    previous_im: Vec<f32>,
    features: Vec<f32>,
    last_mask: Vec<f32>,

    mix: f32,
    pending: [f32; FRAME_LEN],

    residual_block: Vec<f32>,
    aligned_block: Vec<f32>,
    result: Vec<f32>,
}

impl EchoPipeline {
    /// Loads the echo container and builds the fixed front-end tables.
    pub fn new(
        echo_container_bytes: &[u8],
        config: EchoPipelineConfig,
    ) -> Result<Self, PipelineError> {
        let config = config.normalized();
        let model = EchoModel::new(echo_container_bytes)?;

        let mut window = [0.0f32; WINDOW_SIZE];
        for (n, value) in window.iter_mut().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * n as f64 / WINDOW_SIZE as f64;
            *value = (0.5 - 0.5 * angle.cos()) as f32;
        }
        let mut weights = [0.0f32; FEATURE_BINS];
        for (k, value) in weights.iter_mut().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / FFT_SIZE as f64;
            *value = (1.0 + 0.92 * 0.92 - 2.0 * 0.92 * angle.cos()) as f32;
        }

        Ok(Self {
            config,
            aec: Box::new(Aec::new(config.aec)),
            model,
            fft: Fft::new(),
            window,
            weights,
            analysis: vec![0.0; INPUT_CHANNELS * WINDOW_SIZE],
            spec_re: vec![0.0; INPUT_CHANNELS * FEATURE_BINS],
            spec_im: vec![0.0; INPUT_CHANNELS * FEATURE_BINS],
            powers: vec![0.0; INPUT_CHANNELS * FEATURE_BINS],
            overlap: vec![0.0; WINDOW_SIZE],
            selected_re: vec![0.0; FEATURE_BINS],
            selected_im: vec![0.0; FEATURE_BINS],
            previous_re: vec![0.0; FEATURE_BINS],
            previous_im: vec![0.0; FEATURE_BINS],
            features: vec![0.0; FEATURE_LEN],
            last_mask: vec![0.0; FEATURE_BINS],
            mix: config.strength,
            pending: [0.0; FRAME_LEN],
            residual_block: vec![0.0; MAX_BLOCK_LEN],
            aligned_block: vec![0.0; MAX_BLOCK_LEN],
            result: vec![0.0; MAX_BLOCK_LEN],
        })
    }

    /// Processes microphone and reference frames on the same timeline.
    ///
    /// Both slices must have the same length, be a whole number of 256-sample
    /// frames and be at most [`MAX_BLOCK_LEN`] samples. Any other call is a
    /// no-op that returns an empty slice. The result is valid until the next
    /// call and lags the input by [`LATENCY_SAMPLES`] when the neural stage is
    /// enabled; with `neural_enabled = false` it is same-frame.
    pub fn process(&mut self, mic_frame: &[f32], reference_frame: &[f32]) -> &[f32] {
        let len = mic_frame.len();
        if len != reference_frame.len()
            || len == 0
            || !len.is_multiple_of(FRAME_LEN)
            || len > MAX_BLOCK_LEN
        {
            return &self.result[..0];
        }

        if self.config.aec_enabled {
            let residual = self.aec.process(mic_frame, reference_frame);
            self.residual_block[..len].copy_from_slice(residual);
            let aligned = self.aec.aligned_reference();
            if aligned.len() == len {
                self.aligned_block[..len].copy_from_slice(aligned);
            } else {
                // The AEC was enabled after a call that skipped it (or after a
                // reset while it was off), so it has no aligned block matching
                // this call yet.  Fall back to the raw reference for this one
                // frame instead of panicking; the next call is aligned again.
                self.aligned_block[..len].copy_from_slice(reference_frame);
            }
        } else {
            self.residual_block[..len].copy_from_slice(mic_frame);
            self.aligned_block[..len].copy_from_slice(reference_frame);
        }

        if !self.config.neural_enabled {
            self.result[..len].copy_from_slice(&self.residual_block[..len]);
            return &self.result[..len];
        }

        for start in (0..len).step_by(FRAME_LEN) {
            let end = start + FRAME_LEN;
            let mut residual = [0.0f32; FRAME_LEN];
            let mut microphone = [0.0f32; FRAME_LEN];
            let mut aligned = [0.0f32; FRAME_LEN];
            residual.copy_from_slice(&self.residual_block[start..end]);
            microphone.copy_from_slice(&mic_frame[start..end]);
            aligned.copy_from_slice(&self.aligned_block[start..end]);
            let out = self.process_frame(&residual, &microphone, &aligned);
            self.result[start..end].copy_from_slice(&out);
        }
        &self.result[..len]
    }

    /// Clears the linear filter, the delay estimate, the model history, the
    /// spectral buffers and the sample delay. Configuration is kept.
    pub fn reset(&mut self) {
        self.aec.reset();
        self.model.reset();
        self.analysis.fill(0.0);
        self.overlap.fill(0.0);
        self.spec_re.fill(0.0);
        self.spec_im.fill(0.0);
        self.powers.fill(0.0);
        self.selected_re.fill(0.0);
        self.selected_im.fill(0.0);
        self.previous_re.fill(0.0);
        self.previous_im.fill(0.0);
        self.features.fill(0.0);
        self.last_mask.fill(0.0);
        self.mix = self.config.strength;
        self.pending = [0.0; FRAME_LEN];
        self.result.fill(0.0);
    }

    /// Algorithmic latency added by the spectral stage, in samples.
    pub fn latency_samples(&self) -> usize {
        LATENCY_SAMPLES
    }

    /// Configuration in force.
    pub fn config(&self) -> EchoPipelineConfig {
        self.config
    }

    /// Changes the learned-mask strength without rebuilding the pipeline.
    ///
    /// The internal mix keeps ramping towards the new value with the same 30 ms
    /// smoothing used at construction, so a live change neither steps the
    /// output nor resets the model. Values are clamped exactly like
    /// [`EchoPipelineConfig::strength`], and a non-finite value selects `1.0`.
    pub fn set_strength(&mut self, strength: f32) {
        self.config.strength = if strength.is_finite() {
            strength.max(0.0).min(1.0)
        } else {
            1.0
        };
    }

    /// Enables or disables the linear AEC stage without rebuilding the
    /// pipeline (or touching the neural model).
    ///
    /// While it is disabled the microphone is the residual and the raw
    /// reference is the aligned channel. The linear filter stops adapting and
    /// resumes from its previous taps when the stage is re-enabled, so a
    /// deliberate A/B switch costs a short re-convergence, not a model reset.
    pub fn set_aec_enabled(&mut self, enabled: bool) {
        self.config.aec_enabled = enabled;
    }

    /// Pins or releases the linear-stage bulk delay, in samples, without
    /// rebuilding the pipeline. `None` returns the delay to the estimator.
    ///
    /// Delegates to [`Aec::set_delay_override`]: a changed applied delay resets
    /// the adaptive filter, because its taps describe the previous alignment.
    /// The neural model, its history and the audio stream are untouched.
    pub fn set_aec_delay_override(&mut self, delay: Option<usize>) {
        self.aec.set_delay_override(delay);
        self.config.aec.delay_override = delay.map(|d| d.min(cf_aec::MAX_DELAY_SAMPLES));
    }

    /// Linear AEC stage.
    pub fn aec(&self) -> &Aec {
        self.aec.as_ref()
    }

    /// Smoothed linear-stage ERLE, in dB.
    pub fn aec_erle_db(&self) -> f32 {
        self.aec.erle_db()
    }

    /// Applied bulk-delay compensation, in samples.
    pub fn aec_delay_samples(&self) -> usize {
        self.aec.delay_samples()
    }

    /// Whether the double-talk detector is currently freezing adaptation.
    pub fn aec_double_talk(&self) -> bool {
        self.aec.double_talk()
    }

    /// Learned mask of the most recent frame, before strength mixing. Diagnostic
    /// only; it is not needed by [`EchoPipeline::process`].
    pub fn last_mask(&self) -> &[f32] {
        &self.last_mask
    }

    fn process_frame(
        &mut self,
        residual: &[f32; FRAME_LEN],
        microphone: &[f32; FRAME_LEN],
        aligned_reference: &[f32; FRAME_LEN],
    ) -> [f32; FRAME_LEN] {
        for channel in 0..INPUT_CHANNELS {
            let frame = match channel {
                0 => residual,
                1 => microphone,
                _ => aligned_reference,
            };
            let base = channel * WINDOW_SIZE;
            self.analysis
                .copy_within(base + HOP_LEN..base + WINDOW_SIZE, base);
            self.analysis[base + WINDOW_SIZE - HOP_LEN..base + WINDOW_SIZE].copy_from_slice(frame);
        }

        for channel in 0..INPUT_CHANNELS {
            let base = channel * WINDOW_SIZE;
            let mut windowed = [0.0f32; WINDOW_SIZE];
            for n in 0..WINDOW_SIZE {
                windowed[n] = self.analysis[base + n] * self.window[n];
            }
            let mut re = [0.0f32; FEATURE_BINS];
            let mut im = [0.0f32; FEATURE_BINS];
            self.fft.real_spectrum(&windowed, &mut re, &mut im);

            let spec = channel * FEATURE_BINS;
            self.spec_re[spec..spec + FEATURE_BINS].copy_from_slice(&re);
            self.spec_im[spec..spec + FEATURE_BINS].copy_from_slice(&im);
            for k in 0..FEATURE_BINS {
                self.powers[spec + k] = re[k] * re[k] + im[k] * im[k];
            }
        }

        for k in 0..FEATURE_BINS {
            let residual_power = self.powers[k];
            let microphone_power = self.powers[FEATURE_BINS + k];
            let reference_power = self.powers[2 * FEATURE_BINS + k];
            let use_residual = residual_power <= microphone_power;
            let (selected_re, selected_im) = if use_residual {
                (self.spec_re[k], self.spec_im[k])
            } else {
                (
                    self.spec_re[FEATURE_BINS + k],
                    self.spec_im[FEATURE_BINS + k],
                )
            };
            self.selected_re[k] = selected_re;
            self.selected_im[k] = selected_im;

            let selected_power = residual_power.min(microphone_power);
            self.features[k] = feature(selected_power, self.weights[k]);
            self.features[FEATURE_BINS + k] = feature(microphone_power, self.weights[k]);
            self.features[2 * FEATURE_BINS + k] = feature(reference_power, self.weights[k]);
        }

        self.model
            .process(&self.features)
            .expect("the echo model frame length is fixed");

        let mut mask = [0.0f32; FEATURE_BINS];
        mask.copy_from_slice(self.model.output());
        self.last_mask.copy_from_slice(&mask);

        let smoothing = 1.0 - (-(HOP_LEN as f32) / (SAMPLE_RATE as f32 * 0.03)).exp();
        self.mix += (self.config.strength - self.mix) * smoothing;

        let mut processed_re = [0.0f32; FEATURE_BINS];
        let mut processed_im = [0.0f32; FEATURE_BINS];
        for k in 0..FEATURE_BINS {
            let learned = mask[k].clamp(0.0, 1.0);
            let effective = (1.0 - self.mix) + self.mix * learned;
            processed_re[k] = self.previous_re[k] * effective;
            processed_im[k] = self.previous_im[k] * effective;
        }
        self.previous_re.copy_from_slice(&self.selected_re);
        self.previous_im.copy_from_slice(&self.selected_im);

        let mut reconstruction = [0.0f32; WINDOW_SIZE];
        self.fft
            .real_inverse(&processed_re, &processed_im, &mut reconstruction);
        for n in 0..WINDOW_SIZE {
            self.overlap[n] += reconstruction[n] * self.window[n] * SYNTHESIS_SCALE;
        }

        let mut block = [0.0f32; FRAME_LEN];
        block.copy_from_slice(&self.overlap[..FRAME_LEN]);
        self.overlap.copy_within(FRAME_LEN.., 0);
        self.overlap[WINDOW_SIZE - FRAME_LEN..].fill(0.0);

        let out = self.pending;
        self.pending = block;
        out
    }
}

/// One model feature: `0.5 * log10(power * preemphasis + 1e-16)`.
#[inline]
fn feature(power: f32, weight: f32) -> f32 {
    0.5 * (power * weight + EPSILON).log10()
}

/// Periodic Hann window used by the front end, for cross-checks.
pub fn periodic_hann() -> [f32; WINDOW_SIZE] {
    let mut window = [0.0f32; WINDOW_SIZE];
    for (n, value) in window.iter_mut().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * n as f64 / WINDOW_SIZE as f64;
        *value = (0.5 - 0.5 * angle.cos()) as f32;
    }
    window
}
