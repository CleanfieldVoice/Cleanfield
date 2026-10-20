//! Linear acoustic echo canceller (AEC) front end.
//!
//! This crate supplies the linear stage in front of the neural echo model: it
//! takes the microphone and the far-end reference, estimates the echo path with
//! a partitioned-block frequency-domain adaptive filter (PBFDAF), removes the
//! estimated echo and reports how much echo it removed (ERLE).
//!
//! # Signal contract
//!
//! * 16 000 Hz mono, `f32`, nominal range +-1.0.
//! * One frame is [`FRAME_LEN`] = 256 samples = 16 ms. The host reblocks to that
//!   size; see `CONTRACT.md`. A call may carry more than one frame (up to
//!   [`MAX_FRAME_LEN`]), but the length must be a whole number of frames.
//! * One [`Aec`] instance per stream, stateful across calls, never shared.
//! * No heap allocation at all: `new` and `process` only touch fixed-size
//!   fields, including the FFT tables and the delay-line history.
//!
//! # Algorithm
//!
//! * FFT size 512 with a 256-sample hop: 50 % overlap-save, one 256-tap
//!   partition per frame, [`PARTITIONS`] = 8 partitions (2 048 taps = 128 ms of
//!   modelled tail after the bulk delay is removed).
//! * Adaptive filter: one NLMS tap group per frequency bin,
//!   `W_p += mu * conj(X_p) * E / (sum_p |X_p|^2 + delta)`. The update is
//!   applied directly in the frequency domain without a gradient projection;
//!   the per-bin normalization already bounds it, and the projection was
//!   measured to cost ERLE at this window length (see `README.md`).
//! * Bulk delay: GCC-PHAT (phase-transform) correlation of the microphone
//!   against a bounded reference history, followed by first-onset selection
//!   rather than a global peak. The estimator locks to the direct path even
//!   when a late reflection is stronger. Once locked, a two-tier re-lock gate
//!   admits one-frame tracking in either direction, requires a dominant plain
//!   peak and an extra agreeing measurement for a large forward jump, and
//!   refuses large backward jumps (the pitch-period artefact of the onset
//!   scan). The applied alignment delay is available from
//!   [`Aec::delay_samples`] and can be pinned with [`AecConfig::delay_override`]
//!   / [`Aec::set_delay_override`].
//! * Clock rate: a slow, robust estimate of the reference-to-microphone sample
//!   clock ratio, fitted as the slope of the dominant correlation-peak lag over
//!   time, and a four-point cubic fractional-delay read of the reference
//!   history that removes the ratio from the aligned path before adaptation.
//!   The correction is bypassed (identity) until the estimate has converged and
//!   inside the +-5 ppm dead band, and a rate update never resets the filter.
//!   See the `rate` module for the estimator, its window and its response time.
//!   The applied rate is reported by [`Aec::clock_rate_ppm`], whether it is
//!   active by [`Aec::clock_rate_corrected`], and it is switched with
//!   [`Aec::set_clock_rate_correction`]. This is what keeps the echo return loss
//!   up when the far end and the near end run on free-running clocks: the
//!   uncorrected filter sees the bulk delay ramp by the clock ratio and cannot
//!   follow it.
//! * Double talk: adaptation freezes when the microphone is not explained by
//!   the current echo estimate, the two are not coherent, and the microphone
//!   stands clear of a slow stationary-noise floor, so a far-end pause with
//!   only background noise does not look like near-end speech. The thresholds
//!   are documented on the exported `DT_*` constants.
//! * Output: the residual frame (`microphone - estimated echo`), written to a
//!   fixed buffer and valid until the next call.
//!
//! # Verification boundary
//!
//! The neural model has a reference implementation and is verified bit for
//! bit. The linear stage has **no oracle**: it is validated **functionally**
//! (echo return loss, near-end preservation, double-talk stability, delay
//! recovery, determinism, allocation). No bit-exactness is claimed for this
//! crate; small changes to the arithmetic are expected to move the measured
//! numbers, not to break a bit pattern. See `README.md` for the measured
//! results.
//!
//! # Example
//!
//! ```no_run
//! use cf_aec::{Aec, AecConfig, FRAME_LEN};
//!
//! let mut aec = Aec::new(AecConfig::default());
//! let mic = [0.0f32; FRAME_LEN];
//! let reference = [0.0f32; FRAME_LEN];
//! let residual = aec.process(&mic, &reference);
//! assert_eq!(residual.len(), FRAME_LEN);
//! println!("ERLE {:.1} dB at delay {}", aec.erle_db(), aec.delay_samples());
//! ```

mod aec;
mod delay;
mod dtd;
mod fft;
mod rate;

pub use aec::{Aec, AecConfig};
pub use delay::{
    DELAY_CAUSAL_GUARD, DELAY_ONSET_FRACTION, DELAY_ONSET_SMOOTHING, DELAY_PHAT_REGULARIZATION,
    DELAY_RELOCK_JUMP_SAMPLES, DELAY_RELOCK_LARGE_CONFIRM_SEARCHES, DELAY_RELOCK_LARGE_SCORE_RATIO,
    DELAY_RELOCK_SCORE_RATIO,
};
pub use dtd::{
    DT_COHERENCE, DT_HANGOVER_FRAMES, DT_MIC_MEAN_RISE, DT_MIC_POWER_FLOOR, DT_NEAR_END_RATIO,
    DT_NOISE_FLOOR_GATE, DT_NOISE_FLOOR_RISE, DT_NOISE_PRESENCE_MAX_SHARE,
    DT_NOISE_PRESENCE_SETTLE_FRAMES, DT_NOISE_PRESENCE_SHARE, DT_WARMUP_FRAMES,
};
pub use rate::{RATE_DEAD_BAND_PPM, RATE_MAX_PPM};

/// Native sample rate of the front end, in hertz.
pub const SAMPLE_RATE: usize = 16_000;

/// Frame length in samples: 256 samples = 16 ms at [`SAMPLE_RATE`].
pub const FRAME_LEN: usize = 256;

/// FFT size. Two frames fit exactly, which gives 50 % overlap-save.
pub const FFT_SIZE: usize = 512;

/// Number of one-sided frequency bins (`0 ..= FFT_SIZE / 2`).
pub const BINS: usize = FFT_SIZE / 2 + 1;

/// Number of 256-tap partitions in the adaptive filter.
pub const PARTITIONS: usize = 8;

/// Total modelled echo tail after bulk-delay compensation, in samples.
pub const FILTER_TAPS: usize = PARTITIONS * FRAME_LEN;

/// Largest bulk delay the estimator searches and the filter compensates.
pub const MAX_DELAY_SAMPLES: usize = 2_048;

/// Largest accepted `process` block, in samples (8 frames).
pub const MAX_FRAME_LEN: usize = 8 * FRAME_LEN;
