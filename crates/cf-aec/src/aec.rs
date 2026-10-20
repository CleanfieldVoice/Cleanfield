//! Partitioned-block frequency-domain adaptive filter.
//!
//! Every 256-sample frame:
//!
//! 1. push the raw reference and microphone into the delay estimator;
//! 2. take the 256-sample reference frame advanced by the applied bulk delay;
//! 3. FFT the 512-sample window `[previous frame, current frame]` and store it
//!    in the partition ring;
//! 4. form the echo estimate `Y = sum_p W_p X_{m-p}`, inverse-transform it and
//!    keep the last 256 samples (overlap-save);
//! 5. `residual = microphone - echo estimate`;
//! 6. unless double talk is detected, update every partition with the per-bin
//!    NLMS step.
//!
//! The frequency-domain update is deliberately unconstrained (no gradient
//! projection). With the per-bin reference-power normalization the projection
//! is not needed for stability, and measurements at this 256-tap partition
//! length showed it costing about 15 dB of ERLE, so the simpler form is kept.

use crate::delay::DelayEstimator;
use crate::dtd::DoubleTalkDetector;
use crate::fft::Fft;
use crate::rate::RateEstimator;
use crate::{BINS, FFT_SIZE, FRAME_LEN, MAX_DELAY_SAMPLES, MAX_FRAME_LEN, PARTITIONS};

/// One-sided FFT half point (the Nyquist bin).
const HALF_SIZE: usize = FFT_SIZE / 2;

/// Smoothing factor of the reported ERLE (about 20 frames = 320 ms).
const ERLE_SMOOTHING: f32 = 0.05;

/// Lower clamp of the reported ERLE, in dB.
const ERLE_FLOOR_DB: f32 = -40.0;

/// Upper clamp of the reported ERLE, in dB.
const ERLE_CEIL_DB: f32 = 80.0;

/// Denominator floor, in power units.
const EPSILON: f32 = 1.0e-12;

/// Additional denominator floor as a fraction of a slowly decaying estimate of
/// the mean per-bin reference power. The configured value is an absolute power
/// and is far below a bin that carries speech, but it is far above a spectral
/// valley; without a relative floor the noise-driven update in those valleys is
/// effectively unregularized and the filter norm grows. Using a slow peak-hold
/// rather than the instantaneous mean keeps the floor from collapsing during
/// speech pauses, which is exactly when the noise floor would otherwise drive
/// the update. See [`Aec::adapt`].
const RELATIVE_REGULARIZATION: f32 = 0.1;

/// Per-frame decay of the peak-hold reference power. At 16 ms per frame, 0.999
/// is a 16 s time constant: fast enough to follow a level change, slow enough
/// that a normal speech pause does not empty the floor.
const POWER_FLOOR_DECAY: f32 = 0.999;

/// Tuning of one [`Aec`] instance.
///
/// The defaults are the values used in the validation suite.
///
/// Clock-rate correction is not a field here: it is switched per instance with
/// [`Aec::set_clock_rate_correction`] and is on by default, so this struct keeps
/// its layout and existing struct-literal configurations keep compiling.
#[derive(Clone, Copy, Debug)]
pub struct AecConfig {
    /// NLMS step size, `[0, 1]`. Larger converges faster and is less robust to
    /// residual double talk; zero freezes adaptation. Values outside the range
    /// are clamped.
    pub step_size: f32,
    /// NLMS regularization added to the per-bin reference power. Absolute
    /// power, not a ratio.
    pub regularization: f32,
    /// Pins the bulk delay and disables estimation when set. The value is
    /// clamped to [`MAX_DELAY_SAMPLES`].
    pub delay_override: Option<usize>,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            step_size: 0.5,
            regularization: 1.0e-2,
            delay_override: None,
        }
    }
}

impl AecConfig {
    /// Clamps user values into the supported ranges without panicking.
    fn normalized(mut self) -> Self {
        let defaults = Self::default();
        self.step_size = if self.step_size.is_finite() {
            self.step_size.max(0.0).min(1.0)
        } else {
            defaults.step_size
        };
        self.regularization = if self.regularization.is_finite() {
            self.regularization.max(0.0)
        } else {
            defaults.regularization
        };
        self.delay_override = self.delay_override.map(|d| d.min(MAX_DELAY_SAMPLES));
        self
    }
}

/// Linear acoustic echo canceller.
///
/// One instance per stream. [`Aec::process`] is the only method with a hot
/// path; it does not allocate. The residual slice it returns is valid until the
/// next call on the same instance.
pub struct Aec {
    config: AecConfig,
    fft: Fft,
    delay: DelayEstimator,
    dtd: DoubleTalkDetector,
    /// Reference-to-microphone clock-rate estimator and its applied rate.
    rate: RateEstimator,

    /// Reference spectra for the last `PARTITIONS` frames, one-sided.
    x_re: [[f32; BINS]; PARTITIONS],
    x_im: [[f32; BINS]; PARTITIONS],
    /// Adaptive filter, one 256-tap partition per slot, one-sided.
    w_re: [[f32; BINS]; PARTITIONS],
    w_im: [[f32; BINS]; PARTITIONS],
    /// Slot of the most recent reference spectrum.
    head: usize,
    /// Previous aligned reference frame (first half of the FFT window).
    overlap_ref: [f32; FRAME_LEN],

    block_residual: [f32; FRAME_LEN],
    residual: [f32; MAX_FRAME_LEN],
    /// Aligned reference produced alongside `residual`.
    block_aligned: [f32; FRAME_LEN],
    aligned: [f32; MAX_FRAME_LEN],
    aligned_len: usize,

    /// Error spectrum and inverse-transform workspace.
    err_re: [f32; FFT_SIZE],
    err_im: [f32; FFT_SIZE],
    work_re: [f32; FFT_SIZE],
    work_im: [f32; FFT_SIZE],

    mic_power_smoothed: f32,
    residual_power_smoothed: f32,
    erle: f32,
    /// Slow peak-hold of the mean per-bin reference power, used as the scale of
    /// the relative regularization floor.
    power_floor: f32,
}

impl Aec {
    /// Creates an instance. Builds the FFT tables and the fixed-size workspace;
    /// it performs no heap allocation.
    pub fn new(config: AecConfig) -> Self {
        let config = config.normalized();
        Self {
            config,
            fft: Fft::new(),
            delay: DelayEstimator::new(config.delay_override),
            dtd: DoubleTalkDetector::new(),
            rate: RateEstimator::new(true),
            x_re: [[0.0; BINS]; PARTITIONS],
            x_im: [[0.0; BINS]; PARTITIONS],
            w_re: [[0.0; BINS]; PARTITIONS],
            w_im: [[0.0; BINS]; PARTITIONS],
            head: 0,
            overlap_ref: [0.0; FRAME_LEN],
            block_residual: [0.0; FRAME_LEN],
            residual: [0.0; MAX_FRAME_LEN],
            block_aligned: [0.0; FRAME_LEN],
            aligned: [0.0; MAX_FRAME_LEN],
            aligned_len: 0,
            err_re: [0.0; FFT_SIZE],
            err_im: [0.0; FFT_SIZE],
            work_re: [0.0; FFT_SIZE],
            work_im: [0.0; FFT_SIZE],
            mic_power_smoothed: 0.0,
            residual_power_smoothed: 0.0,
            erle: 0.0,
            power_floor: 0.0,
        }
    }

    /// Processes a block of microphone and reference samples and returns the
    /// residual (microphone minus estimated echo).
    ///
    /// Both slices must have the same length, be a whole number of
    /// [`FRAME_LEN`] frames, and be at most [`MAX_FRAME_LEN`] samples. Any
    /// other call is a no-op and returns an empty slice; the AEC is block
    /// synchronous and cannot consume a partial frame without adding latency.
    /// The returned slice is valid until the next call.
    pub fn process(&mut self, mic: &[f32], reference: &[f32]) -> &[f32] {
        let len = mic.len();
        if len != reference.len()
            || len == 0
            || !len.is_multiple_of(FRAME_LEN)
            || len > MAX_FRAME_LEN
        {
            self.aligned_len = 0;
            return &self.residual[..0];
        }

        for start in (0..len).step_by(FRAME_LEN) {
            let end = start + FRAME_LEN;
            self.process_block(&mic[start..end], &reference[start..end]);
            self.residual[start..end].copy_from_slice(&self.block_residual);
            self.aligned[start..end].copy_from_slice(&self.block_aligned);
        }
        self.aligned_len = len;
        &self.residual[..len]
    }

    /// Clears the adaptive filter, the delay estimator, the clock-rate
    /// estimate, the detector and the ERLE history. The manual delay override,
    /// if any, is kept, as is the clock-rate-correction switch.
    pub fn reset(&mut self) {
        self.delay.reset();
        self.rate.reset();
        self.reset_filter_state();
        self.power_floor = 0.0;
        self.aligned.fill(0.0);
        self.aligned_len = 0;
    }

    /// Applied bulk-delay compensation, in samples. The reference is advanced
    /// by this many samples before adaptation. When the estimator is enabled
    /// the value is the confirmed first onset minus [`crate::DELAY_CAUSAL_GUARD`],
    /// so it is a little earlier than the true delay; a late alignment would
    /// make the leading echo taps non-causal.
    pub fn delay_samples(&self) -> usize {
        self.delay.applied()
    }

    /// Aligned far-end reference for the residual block most recently returned.
    ///
    /// The samples are the raw reference advanced by [`Aec::delay_samples`], so
    /// they share the residual frame's timeline and can be used as the third
    /// feature channel of the neural echo model. When the clock-rate correction
    /// is active ([`Aec::clock_rate_corrected`]) the samples are additionally
    /// resampled by the estimated rate, which is what keeps them aligned with
    /// the drifted microphone, and the read is two samples later than the plain
    /// delay because of the interpolant's lookahead. The slice length equals the
    /// last successful [`Aec::process`] block; it is empty before the first
    /// call, after [`Aec::reset`], and after an invalid call. The slice is valid
    /// until the next call and the accessor performs no allocation.
    pub fn aligned_reference(&self) -> &[f32] {
        &self.aligned[..self.aligned_len]
    }

    /// Pins the bulk delay to `Some(samples)` or releases it back to the
    /// estimator with `None`. A changed applied delay resets the adaptive
    /// filter, because its taps no longer describe the aligned path.
    pub fn set_delay_override(&mut self, delay: Option<usize>) {
        let changed = self.delay.set_override(delay);
        if changed {
            self.reset_filter_state();
        }
        self.config.delay_override = delay.map(|d| d.min(MAX_DELAY_SAMPLES));
    }

    /// Enables or disables clock-rate correction. Enabled by default.
    ///
    /// When enabled, the far-end reference is resampled by the estimated
    /// reference-to-microphone clock ratio before it enters the adaptive
    /// filter, but only once the estimate has converged; before that the
    /// reference path is the identity and the output is bit-identical to the
    /// uncorrected behaviour. Disabling clears the estimate, so re-enabling
    /// starts a fresh acquisition. Changing the switch never resets the
    /// adaptive filter.
    pub fn set_clock_rate_correction(&mut self, enabled: bool) {
        self.rate.set_enabled(enabled);
    }

    /// Whether the far-end reference is currently resampled by the estimated
    /// clock rate. `false` before the estimate has converged and after
    /// [`Aec::reset`].
    pub fn clock_rate_corrected(&self) -> bool {
        self.rate.is_corrected()
    }

    /// Clock-rate correction applied to the far-end reference, in ppm.
    ///
    /// Positive means the echo delay is growing, i.e. the reference clock runs
    /// slower than the microphone clock. The value is zero while the correction
    /// is bypassed and while the estimate is inside the dead band
    /// ([`crate::RATE_DEAD_BAND_PPM`]), and is bounded by
    /// [`crate::RATE_MAX_PPM`]. It is a diagnostic: the correction itself is
    /// applied inside [`Aec::process`].
    pub fn clock_rate_ppm(&self) -> f32 {
        self.rate.applied_ppm()
    }

    /// Smoothed echo return loss enhancement, in dB:
    /// `10 log10(microphone power / residual power)`, clamped to
    /// `[-40, 80]`. It includes near-end speech in the residual, so it drops
    /// during double talk by construction; use a near-end-free segment to
    /// judge convergence.
    pub fn erle_db(&self) -> f32 {
        self.erle
    }

    /// Whether the double-talk detector is currently freezing adaptation.
    pub fn double_talk(&self) -> bool {
        self.dtd.is_active()
    }

    /// L2 norm of the adaptive filter (all partitions), computed from the
    /// frequency-domain taps with Parseval's relation. Diagnostic accessor; it
    /// is not needed by [`Aec::process`].
    pub fn filter_norm(&self) -> f32 {
        let mut sum = 0.0f32;
        for p in 0..PARTITIONS {
            let re = &self.w_re[p];
            let im = &self.w_im[p];
            sum += re[0] * re[0] + im[0] * im[0];
            for k in 1..HALF_SIZE {
                sum += 2.0 * (re[k] * re[k] + im[k] * im[k]);
            }
            sum += re[HALF_SIZE] * re[HALF_SIZE] + im[HALF_SIZE] * im[HALF_SIZE];
        }
        (sum / (FFT_SIZE as f32)).sqrt()
    }

    fn process_block(&mut self, mic: &[f32], reference: &[f32]) {
        self.delay.push(reference, mic);
        if let Some(observation) = self.delay.take_rate_observation() {
            self.rate.observe(observation);
        }
        self.delay
            .aligned_block(&mut self.block_aligned, self.rate.correction());

        // 512-sample window: previous aligned frame, then the current one.
        let mut window_re = [0.0f32; FFT_SIZE];
        let mut window_im = [0.0f32; FFT_SIZE];
        window_re[..FRAME_LEN].copy_from_slice(&self.overlap_ref);
        window_re[FRAME_LEN..].copy_from_slice(&self.block_aligned);
        self.fft.forward(&mut window_re, &mut window_im);

        self.head = (self.head + 1) % PARTITIONS;
        let slot = self.head;
        self.x_re[slot].copy_from_slice(&window_re[..BINS]);
        self.x_im[slot].copy_from_slice(&window_im[..BINS]);
        // A real input has purely real DC and Nyquist bins; storing the numerical
        // residue would slowly leak imaginary energy into those taps.
        self.x_im[slot][0] = 0.0;
        self.x_im[slot][BINS - 1] = 0.0;

        // Echo estimate Y = sum_p W_p * X_{m-p}, one-sided.
        let mut y_re = [0.0f32; BINS];
        let mut y_im = [0.0f32; BINS];
        for p in 0..PARTITIONS {
            let history = (self.head + PARTITIONS - p) % PARTITIONS;
            let x_re = &self.x_re[history];
            let x_im = &self.x_im[history];
            let w_re = &self.w_re[p];
            let w_im = &self.w_im[p];
            for k in 0..BINS {
                y_re[k] += w_re[k] * x_re[k] - w_im[k] * x_im[k];
                y_im[k] += w_re[k] * x_im[k] + w_im[k] * x_re[k];
            }
        }

        // Inverse transform, then keep the last 256 samples (overlap-save).
        self.work_re.fill(0.0);
        self.work_im.fill(0.0);
        self.work_re[..BINS].copy_from_slice(&y_re);
        self.work_im[..BINS].copy_from_slice(&y_im);
        for k in 1..HALF_SIZE {
            self.work_re[FFT_SIZE - k] = y_re[k];
            self.work_im[FFT_SIZE - k] = -y_im[k];
        }
        self.work_im[0] = 0.0;
        self.work_im[HALF_SIZE] = 0.0;
        self.fft.inverse(&mut self.work_re, &mut self.work_im);

        let mut mic_power = 0.0f32;
        let mut echo_power = 0.0f32;
        let mut residual_power = 0.0f32;
        let mut cross = 0.0f32;
        for n in 0..FRAME_LEN {
            let m = mic[n];
            let y = self.work_re[FRAME_LEN + n];
            let e = m - y;
            self.block_residual[n] = e;
            mic_power += m * m;
            echo_power += y * y;
            residual_power += e * e;
            cross += m * y;
        }
        let scale = 1.0 / (FRAME_LEN as f32);
        mic_power *= scale;
        echo_power *= scale;
        residual_power *= scale;
        cross *= scale;

        // A non-finite block means the input or the state is already broken.
        // Pass the microphone through and restart from clean state rather than
        // propagate NaN forever. The delay history is poisoned too, reset it.
        if !self.block_residual.iter().all(|v| v.is_finite()) {
            self.delay.reset();
            self.reset_filter_state();
            self.block_residual.copy_from_slice(mic);
            return;
        }

        let denominator = (mic_power * echo_power).sqrt();
        let coherence = if denominator > EPSILON {
            (cross / denominator).max(0.0).min(1.0)
        } else {
            0.0
        };
        self.dtd
            .update(mic_power, echo_power, residual_power, coherence);

        if !self.dtd.is_active() {
            self.adapt();
        }

        self.mic_power_smoothed += ERLE_SMOOTHING * (mic_power - self.mic_power_smoothed);
        self.residual_power_smoothed +=
            ERLE_SMOOTHING * (residual_power - self.residual_power_smoothed);
        let ratio = (self.mic_power_smoothed + EPSILON) / (self.residual_power_smoothed + EPSILON);
        self.erle = (10.0 * ratio.log10()).max(ERLE_FLOOR_DB).min(ERLE_CEIL_DB);

        self.overlap_ref.copy_from_slice(&self.block_aligned);

        if self
            .delay
            .tick(self.dtd.is_active(), self.rate.wants_fast())
            .is_some()
        {
            // The alignment changed: the stored spectra and taps describe the
            // old delay and must not be reused.
            self.reset_filter_state();
        }
    }

    /// Per-bin NLMS update over every partition.
    fn adapt(&mut self) {
        let mut x_power = [0.0f32; BINS];
        for p in 0..PARTITIONS {
            let history = (self.head + PARTITIONS - p) % PARTITIONS;
            let x_re = &self.x_re[history];
            let x_im = &self.x_im[history];
            for k in 0..BINS {
                x_power[k] += x_re[k] * x_re[k] + x_im[k] * x_im[k];
            }
        }

        let mean_power = x_power.iter().sum::<f32>() / BINS as f32;
        let reference_power = mean_power.max(self.power_floor);
        self.power_floor = reference_power * POWER_FLOOR_DECAY;
        let regularization = self.config.regularization + RELATIVE_REGULARIZATION * reference_power;

        // Error spectrum: the current residual in the second half, zeros in
        // the first half, matching the overlap-save window of the estimate.
        self.err_re.fill(0.0);
        self.err_im.fill(0.0);
        self.err_re[FRAME_LEN..].copy_from_slice(&self.block_residual);
        self.fft.forward(&mut self.err_re, &mut self.err_im);
        self.err_im[0] = 0.0;
        self.err_im[HALF_SIZE] = 0.0;

        for p in 0..PARTITIONS {
            let history = (self.head + PARTITIONS - p) % PARTITIONS;
            let x_re = &self.x_re[history];
            let x_im = &self.x_im[history];
            let w_re = &mut self.w_re[p];
            let w_im = &mut self.w_im[p];
            for k in 0..BINS {
                let denominator = x_power[k] + regularization;
                if denominator <= EPSILON {
                    continue;
                }
                let step = self.config.step_size / denominator;
                let xr = x_re[k];
                let xi = x_im[k];
                let er = self.err_re[k];
                let ei = self.err_im[k];
                // conj(X_p) * E, scaled by the per-bin reference power.
                w_re[k] += step * (xr * er + xi * ei);
                w_im[k] += step * (xr * ei - xi * er);
            }
            // Keep the real-filter constraint at DC and Nyquist.
            w_im[0] = 0.0;
            w_im[HALF_SIZE] = 0.0;
        }
    }

    fn reset_filter_state(&mut self) {
        for p in 0..PARTITIONS {
            self.x_re[p].fill(0.0);
            self.x_im[p].fill(0.0);
            self.w_re[p].fill(0.0);
            self.w_im[p].fill(0.0);
        }
        self.head = 0;
        self.overlap_ref.fill(0.0);
        self.block_residual.fill(0.0);
        self.mic_power_smoothed = 0.0;
        self.residual_power_smoothed = 0.0;
        self.erle = 0.0;
        self.dtd.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize) -> Vec<f32> {
        (0..len).map(|n| n as f32 * 0.001).collect()
    }

    #[test]
    fn aligned_reference_is_the_delay_advanced_frame() {
        let delay = 64usize;
        let mut aec = Aec::new(AecConfig {
            delay_override: Some(delay),
            ..AecConfig::default()
        });
        let reference = ramp(4 * FRAME_LEN);

        let residual = aec.process(&[0.0; FRAME_LEN], &reference[..FRAME_LEN]);
        assert_eq!(residual.len(), FRAME_LEN);
        let aligned = aec.aligned_reference();
        assert_eq!(aligned.len(), FRAME_LEN);
        assert!(aligned[..delay].iter().all(|&v| v == 0.0));
        for n in delay..FRAME_LEN {
            assert_eq!(aligned[n], reference[n - delay]);
        }

        let residual = aec.process(&[0.0; 2 * FRAME_LEN], &reference[FRAME_LEN..3 * FRAME_LEN]);
        assert_eq!(residual.len(), 2 * FRAME_LEN);
        let aligned = aec.aligned_reference();
        assert_eq!(aligned.len(), 2 * FRAME_LEN);
        for n in 0..2 * FRAME_LEN {
            assert_eq!(aligned[n], reference[FRAME_LEN + n - delay]);
        }

        aec.reset();
        assert!(aec.aligned_reference().is_empty());
    }
}
