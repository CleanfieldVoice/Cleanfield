//! Bulk-delay estimation.
//!
//! The echo path is `microphone[n] ~ h * reference[n - D]`, where `D` is the
//! bulk delay (reference buffering, device latency, propagation). The
//! partitioned filter only has causal taps, so the delay is estimated and
//! removed before adaptation.
//!
//! # Estimator
//!
//! The estimator correlates the last 256 ms of microphone against a reference
//! history and searches `0 ..= MAX_DELAY_SAMPLES` (128 ms). To keep the
//! transform small, both signals are decimated by four with a centred
//! `[1, 2, 3, 4, 3, 2, 1] / 16` low-pass before the transform; the same filter
//! is applied to both paths, so its group delay cancels and the lag resolution
//! is four samples, far below the 32-sample confirmation tolerance.
//!
//! The correlation method is **GCC-PHAT**: the cross-spectrum is whitened to
//! unit magnitude (with a small relative regularization so near-silent bins do
//! not dominate) before the inverse transform. Whitening removes the strong
//! periodicity of the speech-like reference that otherwise produces correlation
//! peaks at multiples of the pitch period.
//!
//! The lag is not the global peak. A diffuse or late-reflection room can make a
//! reflection stronger than the direct path, so the estimator takes the **first
//! onset**: the smallest non-negative lag whose smoothed, normalized correlation
//! magnitude reaches [`DELAY_ONSET_FRACTION`] of the maximum. The onset is where
//! the direct arrival starts to rise; it can precede the direct peak, and the
//! [`DELAY_CAUSAL_GUARD`] is subtracted so the applied alignment is early rather
//! than late. A plain normalized correlation peak is still computed as a
//! quality gate (the same [`MIN_PEAK_SCORE`] test as before); the PHAT onset is
//! only trusted when a real correlation peak exists at all.
//!
//! The search runs every 16 frames until a candidate is confirmed, then every
//! 64 frames. A candidate is accepted after two consecutive searches agree
//! within 32 samples. While an alignment is already applied, every measurement
//! must first clear a **two-tier re-lock gate**: a move of at most one frame
//! (small forward or backward step, clock drift) is tracking and needs only a
//! modest plain-correlation quality, a larger move forwards needs a dominant
//! plain peak and one extra agreeing measurement, and a larger move backwards
//! is refused because the onset scan turns a pitch period into a spurious
//! earlier lag. The pair keeps the candidate with the better quality. A delay
//! change resets the adaptive filter in the caller.
//!
//! # Clock-rate measurement and reference resampling
//!
//! Every search also contributes one **clock-rate observation** to
//! [`crate::rate`]: the sub-sample lag of the dominant peak of the same PHAT
//! profile. The onset used for alignment is quantised to the four-sample
//! decimation grid and is a threshold crossing, so it is the wrong quantity to
//! difference over time; the peak is a smooth extremum, and the whole echo
//! path moves with the same rate, so any stable peak carries it. A tracker
//! keeps the measurement on one arrival by searching a
//! [`crate::rate::RATE_PEAK_TRACK_BINS`]-wide window around the previous peak.
//!
//! The raw reference history is never modified, so the estimator always
//! measures the *full* reference-to-microphone ratio. The correction is
//! applied when the frame is read out of that history for the adaptive filter:
//! [`DelayEstimator::aligned_block`] can evaluate a cubic interpolant at a read
//! position that advances by `1 - rate` samples per output sample. That keeps
//! the delay line, the onset search and the rate measurement looking at the
//! microphone's own timeline, and only the aligned path - the one the filter
//! and the neural model see - is rate-corrected.
//!
//! The accumulated warp walks backwards through the stored history, so the
//! correction has a finite endurance: [`REF_RING`] is 384 ms, which at 200 ppm
//! is about 32 minutes of stream before the read reaches the oldest stored
//! sample and the warp clamps (the correction then stops instead of jumping the
//! signal). The 128 ms bulk-delay budget, [`MAX_DELAY_SAMPLES`], is reached
//! first at about 10.7 minutes, so in practice the existing re-lock path - and
//! its filter reset - takes over long before the history bound matters.

use crate::fft::FftN;
use crate::rate::{
    RateObservation, RATE_JUMP_BINS, RATE_MIN_QUALITY, RATE_PEAK_TRACK_BINS, RATE_REANCHOR_ANALYSES,
};
use crate::{FRAME_LEN, MAX_DELAY_SAMPLES};

/// Decimation factor applied before the delay transform.
const DECIMATION: usize = 4;

/// Raw microphone analysis block, in samples (256 ms).
const CORR_LEN: usize = 4_096;

/// Decimated microphone analysis block.
const CORR_LEN_DEC: usize = CORR_LEN / DECIMATION;

/// Raw reference history: the analysis block plus the whole lag window.
const REF_RING: usize = CORR_LEN + MAX_DELAY_SAMPLES;

/// Decimated reference history used by the transform.
const REF_DEC: usize = REF_RING / DECIMATION;

/// Transform size: two decimated analysis blocks.
const DELAY_FFT_SIZE: usize = 2 * CORR_LEN_DEC;

/// First bin of the decimated reference inside the transform buffer.
const REF_DEC_OFFSET: usize = DELAY_FFT_SIZE - REF_DEC;

/// Number of decimated lags searched (`0 ..= MAX_DELAY_DEC`).
const MAX_DELAY_DEC: usize = MAX_DELAY_SAMPLES / DECIMATION;

/// Index of lag zero in the PHAT correlation (a lag `d` is at `BASE - d`).
const PHAT_BASE: usize = DELAY_FFT_SIZE - CORR_LEN_DEC;

/// Search cadence before the first lock, in frames.
const FAST_PERIOD: usize = 16;

/// Search cadence after the first lock, in frames.
const SLOW_PERIOD: usize = 64;

/// Number of agreeing searches needed before a delay is applied.
const CONFIRM_SEARCHES: u8 = 2;

/// Correlation analyses allowed at the fast cadence while the clock-rate
/// estimator is acquiring.
///
/// The estimator needs sixteen measurements to fill its window, so 40 analyses
/// (10.2 s at [`FAST_PERIOD`]) is more than enough for a drifting stream and
/// bounds what a stream with no drift - which never converges and therefore
/// keeps asking for fast analyses - can cost. After the budget is spent the
/// analysis runs at [`SLOW_PERIOD`], which is the pre-existing cadence.
const FAST_ANALYSES: usize = 40;

/// Maximum difference between two candidates for them to count as agreeing.
const CONFIRM_TOLERANCE: usize = 32;

/// Fraction of the maximum smoothed PHAT correlation that marks the onset.
///
/// The direct arrival is not always the strongest correlation peak: in a
/// diffuse room a late reflection can be louder, and in the weak-direct
/// response the first strong reflection is 96 samples after the direct tap.
/// The onset rule therefore looks for the first lag that reaches this fraction
/// of the maximum instead of taking the maximum itself. Values above about 0.7
/// start to miss the direct path when it is partially cancelled by the diffuse
/// tail; values below about 0.5 admit early pitch and noise sidelobes.
pub const DELAY_ONSET_FRACTION: f32 = 0.55;

/// Relative regularization of the PHAT whitening denominator.
///
/// The denominator is `|G(f)| + DELAY_PHAT_REGULARIZATION * mean(|G|)`. A
/// perfectly silent bin has `|G| = 0`; without the floor its unit-magnitude
/// weight would amplify numerical noise and microphone noise across the whole
/// band. One percent of the mean keeps those bins negligible while leaving
/// speech bins effectively whitened.
pub const DELAY_PHAT_REGULARIZATION: f32 = 0.01;

/// Length of the triangular smoothing window applied to the PHAT profile.
///
/// Five decimated samples is 20 raw samples (1.25 ms): enough to suppress
/// single-bin numerical jitter, short enough not to blur the onset by more than
/// the confirmation tolerance. Must be odd.
pub const DELAY_ONSET_SMOOTHING: usize = 5;

/// Subtracted from the detected onset so the applied delay is never late.
///
/// Onset detection already removes the "late reflection" bias that the old
/// global peak had, but it does not remove the early-reflection bias of the
/// weak-direct response: there the direct tap is weaker than the reflection at
/// +96 samples, and a 0.55 onset threshold can land on that reflection. A
/// 96-sample guard makes the direct tap causal in that case, and also covers
/// the 4-sample decimation grid and the 20-raw-sample smoothing window. Being
/// early is harmless: the extra leading taps are simply modelled at a later tap
/// index.
pub const DELAY_CAUSAL_GUARD: usize = 96;

/// Mean power below which a signal is treated as silence, in `f32` units.
const MIN_MEAN_POWER: f32 = 1.0e-7;

/// Minimum normalized correlation for a candidate to be considered at all.
const MIN_PEAK_SCORE: f32 = 0.2;

/// Re-lock magnitude boundary, in samples.
///
/// A move of at most this many samples (one frame) is **tracking** and is
/// accepted in either direction when it clears [`DELAY_RELOCK_SCORE_RATIO`].
/// A move larger than this is a **jump** and falls into the asymmetric tier
/// described on [`DELAY_RELOCK_LARGE_SCORE_RATIO`]. One frame is the natural
/// scale: the suite's +5 ms step is 80 samples and 200 ppm drift moves about
/// 33 samples over the file, while a pitch-period artefact is hundreds of
/// samples.
pub const DELAY_RELOCK_JUMP_SAMPLES: usize = 256;

/// Relative correlation quality needed to track a small delay change, in
/// either direction, while an alignment is already applied.
///
/// A periodic reference can make a pitch multiple (including lag zero) exceed
/// the onset threshold on a few frames, and a slow re-search must not abandon a
/// good alignment for such a lag. The plain normalized correlation around the
/// new onset (within one decimated bin, so the four-sample grid does not
/// penalise a genuine step) must be at least this fraction of the best plain
/// correlation in the same search. A genuine small delay change (the suite's
/// +5 ms step, clock drift) keeps a direct path near that peak; a pitch
/// artefact does not. The initial lock is not gated because there is no working
/// alignment to protect.
pub const DELAY_RELOCK_SCORE_RATIO: f32 = 0.25;

/// Relative correlation quality required for a **forward** jump larger than
/// [`DELAY_RELOCK_JUMP_SAMPLES`].
///
/// The gate is deliberately direction-aware. The onset scan walks lags from
/// zero upwards, so when a periodic reference produces a spurious peak it
/// faults *before* the direct arrival: the measured lag moves **backwards** by
/// roughly a pitch period. A working alignment is therefore never abandoned
/// for a jump backwards of more than one frame, whatever its plain correlation
/// looks like — that is the pitch/period artefact the two-tier gate exists to
/// reject. A jump forwards can be a genuine change of the path (a different
/// device buffer or a large step), but it is also where a late reflection
/// would appear, so its onset must be near the dominant plain-correlation
/// peak. This ratio is the pre-existing large-jump bar; only the direction
/// asymmetry and the extra confirming measurement are new.
pub const DELAY_RELOCK_LARGE_SCORE_RATIO: f32 = 0.5;

/// Number of agreeing searches needed before a *large* re-lock is applied.
///
/// Tracking re-locks still need only [`CONFIRM_SEARCHES`] measurements. A far
/// larger move gets one extra measurement so a transient artefact that happens
/// to clear the score bar on two consecutive searches cannot move the
/// alignment by hundreds of samples.
pub const DELAY_RELOCK_LARGE_CONFIRM_SEARCHES: u8 = CONFIRM_SEARCHES + 1;

/// Centred decimation low-pass: `[1, 2, 3, 4, 3, 2, 1] / 16`.
const DECIMATE_TAPS: [f32; 7] = [
    1.0 / 16.0,
    2.0 / 16.0,
    3.0 / 16.0,
    4.0 / 16.0,
    3.0 / 16.0,
    2.0 / 16.0,
    1.0 / 16.0,
];

/// Samples of lookahead the cubic rate-correction read needs.
///
/// The interpolant is evaluated at `index + frac` from the four taps
/// `index - 1 ..= index + 2`, so the newest tap is two samples beyond the
/// integer read position. Keeping the read position two samples behind the
/// write head makes every tap a sample that has already been pushed: the
/// correction path is causal and adds exactly two samples of latency to the
/// aligned reference (the cubic itself interpolates, so a static path is
/// reproduced exactly at integer positions and its group delay is zero).
/// Two samples is 0.125 ms, far inside the 96-sample causal guard the
/// alignment already carries.
const RATE_LOOKAHEAD: usize = 2;

/// Bulk-delay tracker and reference delay line.
pub(crate) struct DelayEstimator {
    /// Raw reference history, oldest at `ref_pos`.
    ref_ring: [f32; REF_RING],
    ref_pos: usize,
    /// Raw microphone history, oldest at `mic_pos`.
    mic_ring: [f32; CORR_LEN],
    mic_pos: usize,
    /// Decimated analysis blocks.
    mic_dec: [f32; CORR_LEN_DEC],
    ref_dec: [f32; REF_DEC],
    /// Transform workspace. The microphone spectrum is transformed in place
    /// into the cross-spectrum, whitened and inverted; the reference spectrum is
    /// only needed while forming the cross-spectrum.
    corr_re: [f32; DELAY_FFT_SIZE],
    corr_im: [f32; DELAY_FFT_SIZE],
    reference_re: [f32; DELAY_FFT_SIZE],
    reference_im: [f32; DELAY_FFT_SIZE],
    fft: FftN<DELAY_FFT_SIZE>,
    /// Smoothed PHAT magnitude, indexed by decimated lag.
    profile: [f32; MAX_DELAY_DEC + 1],
    applied: usize,
    override_delay: Option<usize>,
    /// Latest candidate lag, used to decide whether two searches agree.
    pending: Option<usize>,
    /// Candidate lag with the best plain-correlation quality among the
    /// agreeing searches. This is the value applied when the pair confirms.
    pending_best: usize,
    /// Whether any agreeing measurement proposed a move larger than
    /// [`DELAY_RELOCK_JUMP_SAMPLES`], which raises the confirmation count.
    pending_large: bool,
    pending_count: u8,
    /// Best plain-correlation quality among the searches agreeing on `pending`.
    pending_quality: f32,
    /// Smoothed PHAT magnitude at every decimated lag, from the last analysis.
    smoothed: [f32; MAX_DELAY_DEC + 1],
    /// Largest value in `smoothed`, from the last analysis.
    profile_max: f32,
    /// First-onset bin of the last analysis, in decimated lags.
    onset: usize,
    /// Microphone block energy, from the last analysis.
    mic_energy: f32,
    blocks_since_search: usize,
    blocks_since_analysis: usize,
    /// Fast analyses left before the analysis cadence falls back to
    /// [`SLOW_PERIOD`]. Bounds the clock-rate acquisition cost at
    /// [`FAST_ANALYSES`] extra correlations per stream.
    fast_budget: usize,
    samples_pushed: usize,
    locked: bool,
    /// Latest clock-rate observation, taken by the caller after an analysis.
    rate_observation: Option<RateObservation>,
    /// Centre of the peak-tracking window, in decimated lags.
    rate_peak: Option<f32>,
    /// Consecutive analyses on which the tracked peak has been too weak to
    /// trust, used to decide when to snap to the strongest peak instead.
    lost: u8,
    /// Whole samples of accumulated rate-correction warp (the amount by which
    /// the read position has fallen behind the uncorrected one), and its
    /// fractional part in `[0, 1)`.
    rate_whole: isize,
    rate_frac: f32,
}

impl DelayEstimator {
    pub(crate) fn new(override_delay: Option<usize>) -> Self {
        let override_delay = override_delay.map(|d| d.min(MAX_DELAY_SAMPLES));
        Self {
            ref_ring: [0.0; REF_RING],
            ref_pos: 0,
            mic_ring: [0.0; CORR_LEN],
            mic_pos: 0,
            mic_dec: [0.0; CORR_LEN_DEC],
            ref_dec: [0.0; REF_DEC],
            corr_re: [0.0; DELAY_FFT_SIZE],
            corr_im: [0.0; DELAY_FFT_SIZE],
            reference_re: [0.0; DELAY_FFT_SIZE],
            reference_im: [0.0; DELAY_FFT_SIZE],
            fft: FftN::new(),
            profile: [0.0; MAX_DELAY_DEC + 1],
            applied: override_delay.unwrap_or(0),
            override_delay,
            pending: None,
            pending_best: 0,
            pending_large: false,
            pending_count: 0,
            pending_quality: 0.0,
            smoothed: [0.0; MAX_DELAY_DEC + 1],
            profile_max: 0.0,
            onset: 0,
            mic_energy: 0.0,
            blocks_since_search: 0,
            blocks_since_analysis: 0,
            fast_budget: FAST_ANALYSES,
            samples_pushed: 0,
            locked: override_delay.is_some(),
            rate_observation: None,
            rate_peak: None,
            lost: 0,
            rate_whole: 0,
            rate_frac: 0.0,
        }
    }

    /// Appends one frame of raw reference and microphone samples.
    pub(crate) fn push(&mut self, reference: &[f32], mic: &[f32]) {
        let mut written = 0usize;
        for (&r, &m) in reference.iter().zip(mic.iter()).take(FRAME_LEN) {
            self.ref_ring[self.ref_pos] = r;
            self.ref_pos += 1;
            if self.ref_pos == REF_RING {
                self.ref_pos = 0;
            }
            self.mic_ring[self.mic_pos] = m;
            self.mic_pos += 1;
            if self.mic_pos == CORR_LEN {
                self.mic_pos = 0;
            }
            written += 1;
        }
        self.samples_pushed = self.samples_pushed.saturating_add(written);
    }

    /// Copies the current frame of reference, advanced by the applied delay, so
    /// that it lines up with the current microphone frame.
    ///
    /// `correction` is the clock-rate correction in ppm. `None` copies the
    /// stored samples verbatim, which is the behaviour before the rate estimate
    /// converges and is bit-identical to the pre-correction path. `Some(ppm)`
    /// instead evaluates a four-point cubic (Lagrange) interpolant at a read
    /// position of the reference history that advances by `1 - ppm * 1e-6`
    /// samples per output sample, so the aligned reference loses the
    /// reference-to-microphone clock ratio.
    ///
    /// The read position is `start + n - warp`, where `warp` is the accumulated
    /// stream time lost to the ratio. Both the whole and the fractional part of
    /// `warp` are carried across calls, so enabling the correction, changing the
    /// estimate and dropping into the dead band never step the aligned signal.
    /// `warp` is clamped to the delay line: it may not run past the write head
    /// (the path must stay causal) or further back than the stored history.
    /// Reaching a bound stops the correction instead of jumping the signal.
    pub(crate) fn aligned_block(&mut self, out: &mut [f32; FRAME_LEN], correction: Option<f32>) {
        // `ref_pos` is the oldest sample, so the newest frame starts
        // `FRAME_LEN + applied` samples before it in ring order.
        let start = (self.ref_pos + REF_RING - FRAME_LEN - self.applied) % REF_RING;
        let Some(ppm) = correction else {
            for (n, slot) in out.iter_mut().enumerate() {
                let mut index = start + n;
                if index >= REF_RING {
                    index -= REF_RING;
                }
                *slot = self.ref_ring[index];
            }
            return;
        };

        // Causality needs `applied + warp >= RATE_LOOKAHEAD`: the newest tap
        // must be a sample that has been pushed. The history end is the whole
        // stored delay line. Clamping is a rate change, not a jump: the read
        // position simply stops warping.
        let lowest = RATE_LOOKAHEAD as isize - self.applied as isize;
        let highest = (REF_RING - FRAME_LEN - RATE_LOOKAHEAD) as isize - self.applied as isize;
        let mut whole = self.rate_whole.clamp(lowest, highest);
        let mut frac = self.rate_frac;
        let rate = ppm * 1.0e-6;

        // Read position `start - whole - frac` on the first sample: evaluate the
        // cubic at `index = start - whole - 1` with the fraction complemented.
        let mut index = (start as isize - whole - 1).rem_euclid(REF_RING as isize) as usize;
        for slot in out.iter_mut() {
            *slot = cubic(&self.ref_ring, index, 1.0 - frac);
            // One output sample advances the read position by `1 - rate`.
            index += 1;
            if index == REF_RING {
                index = 0;
            }
            frac += rate;
            if frac >= 1.0 {
                frac -= 1.0;
                whole += 1;
                index = if index == 0 { REF_RING - 1 } else { index - 1 };
            } else if frac < 0.0 {
                frac += 1.0;
                whole -= 1;
                index += 1;
                if index == REF_RING {
                    index = 0;
                }
            }
        }
        self.rate_whole = whole;
        self.rate_frac = frac;
    }

    /// Takes the clock-rate observation produced by the most recent search.
    pub(crate) fn take_rate_observation(&mut self) -> Option<RateObservation> {
        self.rate_observation.take()
    }

    /// Advances the search schedule by one frame and returns a newly applied
    /// delay when one was accepted. `double_talk` holds the analysis back
    /// because near-end speech biases the correlation; `fast_rate_search` runs
    /// the correlation every [`FAST_PERIOD`] frames instead of every
    /// [`SLOW_PERIOD`] while the clock-rate estimator is still acquiring, so
    /// that it converges early in a stream.
    ///
    /// The correlation analysis and the re-lock *decision* have separate
    /// schedules: the analysis is what the clock-rate estimate needs, the
    /// decision is what protects the alignment, and slowing the decision down
    /// must not slow the rate estimate down (or the other way round).
    pub(crate) fn tick(&mut self, double_talk: bool, fast_rate_search: bool) -> Option<usize> {
        self.blocks_since_analysis += 1;
        self.blocks_since_search += 1;
        // The fast cadence is a budget of extra analyses, not of frames: a
        // stream that never converges must not keep paying for it.
        let fast = fast_rate_search && self.fast_budget > 0;
        let period = if fast { FAST_PERIOD } else { SLOW_PERIOD };
        let analysis_due = self.blocks_since_analysis >= period;
        let search_period = if self.locked {
            SLOW_PERIOD
        } else {
            FAST_PERIOD
        };
        let search_due = self.blocks_since_search >= search_period;
        if !analysis_due && !search_due {
            return None;
        }
        if analysis_due {
            self.blocks_since_analysis = 0;
        }
        if search_due {
            self.blocks_since_search = 0;
        }
        if self.override_delay.is_some() || double_talk {
            return None;
        }
        if !self.analyse() {
            return None;
        }
        if fast {
            self.fast_budget = self.fast_budget.saturating_sub(1);
        }
        self.measure_rate();
        if !search_due {
            return None;
        }
        self.decide()
    }

    /// Re-lock decision for one finished analysis.
    fn decide(&mut self) -> Option<usize> {
        let (lag, score, peak_score) = self.best_lag()?;
        let quality = if peak_score > 0.0 {
            score / peak_score
        } else {
            0.0
        };
        let target = lag.saturating_sub(DELAY_CAUSAL_GUARD);

        // Re-lock gate. Every measurement that could become part of a
        // confirming pair must clear the tier for the move it proposes, so a
        // pair can never confirm on one good and one bad reading. A rejected
        // measurement also breaks the pair: the count restarts.
        let was_locked = self.locked;
        if was_locked && !self.relock_allowed(target, quality) {
            self.clear_pending();
            return None;
        }

        let agrees = matches!(
            self.pending,
            Some(previous) if previous.abs_diff(lag) <= CONFIRM_TOLERANCE
        );
        if agrees {
            self.pending_count = self.pending_count.saturating_add(1);
            self.pending_large |= self.is_large_jump(target);
            // Keep the better of the two candidates, not whichever came last.
            if quality > self.pending_quality {
                self.pending_quality = quality;
                self.pending_best = lag;
            }
        } else {
            self.pending_count = 1;
            self.pending_large = self.is_large_jump(target);
            self.pending_quality = quality;
            self.pending_best = lag;
        }
        self.pending = Some(lag);

        // A large move needs one extra agreeing measurement: a forward jump is
        // also where a late reflection would appear, and a short-lived artefact
        // must not be able to move the alignment by hundreds of samples.
        let required = if was_locked && self.pending_large {
            DELAY_RELOCK_LARGE_CONFIRM_SEARCHES
        } else {
            CONFIRM_SEARCHES
        };
        if self.pending_count < required {
            return None;
        }

        let confirmed_lag = self.pending_best;
        let confirmed_quality = self.pending_quality;
        self.clear_pending();
        self.locked = true;
        let target = confirmed_lag.saturating_sub(DELAY_CAUSAL_GUARD);
        if target.abs_diff(self.applied) <= CONFIRM_TOLERANCE {
            return None;
        }
        // Both measurements already cleared the gate against the same applied
        // alignment; re-check the selected candidate because it is not
        // necessarily the latest one that incremented the count.
        if was_locked && !self.relock_allowed(target, confirmed_quality) {
            return None;
        }
        self.applied = target;
        Some(target)
    }

    /// Whether `target` would move the applied delay by more than one frame.
    fn is_large_jump(&self, target: usize) -> bool {
        target.abs_diff(self.applied) > DELAY_RELOCK_JUMP_SAMPLES
    }

    /// Two-tier, direction-aware admission test for a re-lock.
    ///
    /// * Tracking (at most [`DELAY_RELOCK_JUMP_SAMPLES`], either direction):
    ///   needs only [`DELAY_RELOCK_SCORE_RATIO`], because a small forward step
    ///   or clock drift legitimately moves the direct path.
    /// * Jump forwards: needs [`DELAY_RELOCK_LARGE_SCORE_RATIO`], because a
    ///   genuine large change must still be the dominant plain correlation.
    /// * Jump backwards: always refused. The onset scan runs from lag zero
    ///   upwards, so a periodic reference faults at a pitch period *before* the
    ///   direct arrival; a large negative move is that artefact, not tracking.
    fn relock_allowed(&self, target: usize, quality: f32) -> bool {
        if !self.is_large_jump(target) {
            quality >= DELAY_RELOCK_SCORE_RATIO
        } else if target > self.applied {
            quality >= DELAY_RELOCK_LARGE_SCORE_RATIO
        } else {
            false
        }
    }

    /// Drops any partially confirmed candidate.
    fn clear_pending(&mut self) {
        self.pending = None;
        self.pending_best = 0;
        self.pending_large = false;
        self.pending_count = 0;
        self.pending_quality = 0.0;
    }

    /// Returns the applied compensation delay, in samples.
    pub(crate) fn applied(&self) -> usize {
        self.applied
    }

    /// Pins (`Some`) or releases (`None`) the manual override. Returns `true`
    /// when the applied delay changed and the caller must reset alignment.
    pub(crate) fn set_override(&mut self, delay: Option<usize>) -> bool {
        let delay = delay.map(|d| d.min(MAX_DELAY_SAMPLES));
        self.override_delay = delay;
        match delay {
            Some(value) => {
                self.locked = true;
                self.clear_pending();
                let changed = value != self.applied;
                self.applied = value;
                changed
            }
            None => {
                self.locked = false;
                self.blocks_since_search = 0;
                false
            }
        }
    }

    /// Clears all history and returns the applied delay to the override (or 0).
    pub(crate) fn reset(&mut self) {
        self.ref_ring.fill(0.0);
        self.mic_ring.fill(0.0);
        self.mic_dec.fill(0.0);
        self.ref_dec.fill(0.0);
        self.corr_re.fill(0.0);
        self.corr_im.fill(0.0);
        self.reference_re.fill(0.0);
        self.reference_im.fill(0.0);
        self.profile.fill(0.0);
        self.ref_pos = 0;
        self.mic_pos = 0;
        self.clear_pending();
        self.blocks_since_search = 0;
        self.blocks_since_analysis = 0;
        self.fast_budget = FAST_ANALYSES;
        self.samples_pushed = 0;
        self.locked = self.override_delay.is_some();
        self.applied = self.override_delay.unwrap_or(0);
        self.rate_observation = None;
        self.rate_peak = None;
        self.lost = 0;
        self.rate_whole = 0;
        self.rate_frac = 0.0;
    }

    /// Computes one whitened correlation profile.
    ///
    /// Decimates the microphone analysis block and the reference history,
    /// transforms both, whitens the cross-spectrum (GCC-PHAT) and inverse
    /// transforms it into the lag profile, then smooths it. Returns `false`
    /// without touching the profile when the block is silence or the history is
    /// not yet full.
    fn analyse(&mut self) -> bool {
        if self.samples_pushed < REF_RING || self.samples_pushed < CORR_LEN {
            return false;
        }

        decimate(&self.ref_ring, self.ref_pos, &mut self.ref_dec);
        decimate(&self.mic_ring, self.mic_pos, &mut self.mic_dec);

        let mic_energy = self.mic_dec.iter().map(|v| v * v).sum::<f32>();
        let floor = (CORR_LEN_DEC as f32) * MIN_MEAN_POWER;
        if !mic_energy.is_finite() || mic_energy < floor {
            return false;
        }
        self.mic_energy = mic_energy;

        // Microphone spectrum: the decimated block zero-padded to the transform.
        self.corr_re.fill(0.0);
        self.corr_im.fill(0.0);
        self.corr_re[..CORR_LEN_DEC].copy_from_slice(&self.mic_dec);

        // Reference spectrum: the decimated history at its offset, with the
        // unused leading bins zero.
        self.reference_re.fill(0.0);
        self.reference_im.fill(0.0);
        self.reference_re[REF_DEC_OFFSET..REF_DEC_OFFSET + REF_DEC].copy_from_slice(&self.ref_dec);

        self.fft.forward(&mut self.corr_re, &mut self.corr_im);
        self.fft
            .forward(&mut self.reference_re, &mut self.reference_im);

        // Cross-spectrum G = conj(M) * R and its mean magnitude.
        let mut mean_magnitude = 0.0f32;
        for k in 0..DELAY_FFT_SIZE {
            let ar = self.corr_re[k];
            let ai = self.corr_im[k];
            let br = self.reference_re[k];
            let bi = self.reference_im[k];
            let re = ar * br + ai * bi;
            let im = ar * bi - ai * br;
            self.corr_re[k] = re;
            self.corr_im[k] = im;
            mean_magnitude += (re * re + im * im).sqrt();
        }
        mean_magnitude /= DELAY_FFT_SIZE as f32;

        // PHAT whitening. The relative floor keeps near-silent bins from
        // contributing full-weight numerical noise.
        let epsilon = DELAY_PHAT_REGULARIZATION * mean_magnitude;
        for k in 0..DELAY_FFT_SIZE {
            let re = self.corr_re[k];
            let im = self.corr_im[k];
            let denominator = (re * re + im * im).sqrt() + epsilon;
            if denominator > 0.0 {
                self.corr_re[k] = re / denominator;
                self.corr_im[k] = im / denominator;
            } else {
                self.corr_re[k] = 0.0;
                self.corr_im[k] = 0.0;
            }
        }
        self.fft.inverse(&mut self.corr_re, &mut self.corr_im);

        // The inverse transform is the correlation indexed by shift; lag `d`
        // sits at `PHAT_BASE - d`.
        for (lag, slot) in self.profile.iter_mut().enumerate() {
            *slot = self.corr_re[PHAT_BASE - lag].abs();
        }

        let mut maximum = 0.0f32;
        for lag in 0..=MAX_DELAY_DEC {
            let value = smooth(&self.profile, lag);
            self.smoothed[lag] = value;
            maximum = maximum.max(value);
        }
        if !maximum.is_finite() || maximum <= 0.0 {
            return false;
        }
        self.profile_max = maximum;

        // First onset: the smallest lag that reaches the onset fraction of the
        // strongest peak. Both the alignment decision and the initial
        // clock-rate anchor need it, so it is resolved once here.
        let threshold = DELAY_ONSET_FRACTION * maximum;
        let mut onset = MAX_DELAY_DEC;
        for (lag, &value) in self.smoothed.iter().enumerate() {
            if value >= threshold {
                onset = lag;
                break;
            }
        }
        self.onset = onset;
        true
    }

    /// GCC-PHAT first-onset lag over the bounded lag window.
    ///
    /// Returns the onset in raw samples, the plain normalized correlation at
    /// that onset, and the best plain correlation in the search. The last two
    /// feed the detection gate and the re-lock gate.
    fn best_lag(&mut self) -> Option<(usize, f32, f32)> {
        // Detection gate: there must be a real plain correlation peak.
        let (_, peak_score) = self.plain_peak(self.mic_energy)?;

        let onset = self.onset;
        let lag = (onset * DECIMATION).min(MAX_DELAY_SAMPLES);
        let score = self.plain_score_near(onset, self.mic_energy);
        Some((lag, score, peak_score))
    }

    /// Records the clock-rate observation for the finished analysis.
    ///
    /// The measurement is the sub-sample lag of the dominant smoothed PHAT peak,
    /// followed from analysis to analysis. Every arrival of the echo path - the
    /// direct sound and every reflection - carries the same clock rate, so
    /// following one of them is enough; what matters is that it is the *same*
    /// one each time. A [`RATE_PEAK_TRACK_BINS`] window around the previous
    /// measurement does that: it is wider than the drift between analyses (under
    /// one sample at the fast cadence, about fourteen across a double-talk gap)
    /// and narrower than the separation between the arrivals of the response
    /// families in use, so a competing reflection cannot capture the track. The
    /// three-point parabola of the peak resolves it to a fraction of a sample.
    ///
    /// A peak that falls to [`RATE_MIN_QUALITY`] of the strongest in its own
    /// analysis is a lost track - the excitation changed character, or the
    /// alignment is still settling - and its observation is dropped rather than
    /// followed. After [`RATE_REANCHOR_ANALYSES`] consecutive lost analyses the
    /// tracker snaps to the strongest peak instead, which recovers from a bad
    /// first anchor.
    fn measure_rate(&mut self) {
        let smoothed = &self.smoothed;
        let maximum = self.profile_max;
        let global = argmax_profile(smoothed);
        let anchor = self.rate_peak.unwrap_or(global as f32);
        let mut centre = {
            let half = RATE_PEAK_TRACK_BINS as f32;
            let low = ((anchor - half).round().max(0.0) as usize).min(MAX_DELAY_DEC);
            let high = ((anchor + half).round() as usize).min(MAX_DELAY_DEC);
            if (low..=high).contains(&global) {
                global
            } else {
                let mut best = low;
                for lag in low..=high {
                    if smoothed[lag] > smoothed[best] {
                        best = lag;
                    }
                }
                best
            }
        };
        let height = |lag: usize| {
            if maximum > 0.0 {
                smoothed[lag] / maximum
            } else {
                0.0
            }
        };
        if height(centre) < RATE_MIN_QUALITY {
            self.lost = self.lost.saturating_add(1);
            if self.lost < RATE_REANCHOR_ANALYSES {
                return;
            }
            self.lost = 0;
            centre = global;
        } else {
            self.lost = 0;
        }

        let (position, quality) = refine_peak(smoothed, centre as f32, maximum);
        if quality < RATE_MIN_QUALITY {
            return;
        }
        let restart = matches!(
            self.rate_peak,
            Some(previous) if (position - previous).abs() >= RATE_JUMP_BINS as f32
        );
        self.rate_peak = Some(position);
        self.rate_observation = Some(RateObservation {
            at: self.samples_pushed,
            lag: position * DECIMATION as f32,
            quality,
            restart,
        });
    }

    /// Best plain normalized correlation within one decimated bin of `lag`.
    ///
    /// The PHAT onset is resolved on the decimated grid, so even a clean delay
    /// can put the onset one bin before the plain correlation peak (four raw
    /// samples, far inside the 32-sample confirmation tolerance). Scoring the
    /// exact bin alone would reject a genuine step whenever the grid rounded
    /// the wrong way; the gate asks whether a real plain peak supports the
    /// candidate, so it looks one bin either side. The thresholds themselves
    /// are unchanged.
    fn plain_score_near(&self, lag: usize, mic_energy: f32) -> f32 {
        let first = lag.saturating_sub(1);
        let last = (lag + 1).min(MAX_DELAY_DEC);
        let mut best = 0.0f32;
        for candidate in first..=last {
            best = best.max(self.plain_score_at(candidate, mic_energy));
        }
        best
    }

    /// Plain normalized correlation at one decimated lag.
    fn plain_score_at(&self, lag: usize, mic_energy: f32) -> f32 {
        let start = MAX_DELAY_DEC - lag;
        let mut correlation = 0.0f32;
        let mut reference_energy = 0.0f32;
        for i in 0..CORR_LEN_DEC {
            let reference = self.ref_dec[start + i];
            correlation += self.mic_dec[i] * reference;
            reference_energy += reference * reference;
        }
        let denominator = (mic_energy * reference_energy).sqrt();
        if denominator > 0.0 {
            correlation.abs() / denominator
        } else {
            0.0
        }
    }

    /// Plain normalized correlation peak over the decimated lag window.
    ///
    /// This is only a detection-quality gate: it answers "is there a real
    /// correlation peak at all", not "where is the direct path". The PHAT onset
    /// decides the lag. Returns `None` below [`MIN_PEAK_SCORE`].
    fn plain_peak(&self, mic_energy: f32) -> Option<(usize, f32)> {
        let mut best_lag = 0usize;
        let mut best_score = 0.0f32;
        for lag in 0..=MAX_DELAY_DEC {
            let start = MAX_DELAY_DEC - lag;
            let mut correlation = 0.0f32;
            let mut reference_energy = 0.0f32;
            for i in 0..CORR_LEN_DEC {
                let reference = self.ref_dec[start + i];
                correlation += self.mic_dec[i] * reference;
                reference_energy += reference * reference;
            }
            let denominator = (mic_energy * reference_energy).sqrt();
            if denominator > 0.0 {
                let score = correlation.abs() / denominator;
                if score > best_score {
                    best_score = score;
                    best_lag = lag;
                }
            }
        }
        if best_score.is_finite() && best_score >= MIN_PEAK_SCORE {
            Some((best_lag, best_score))
        } else {
            None
        }
    }
}

/// Four-point cubic (Lagrange) interpolation at `index + frac`.
///
/// The interpolant passes exactly through `ring[index]` at `frac = 0`, so a
/// fractional delay of zero reproduces the stored samples bit for bit. Its
/// group delay is zero at the read position by construction (it interpolates,
/// it does not filter), and the error stays below a hundredth of a sample up to
/// 4 kHz, which is the useful band of the 16 kHz front end. Four taps is the
/// smallest cubic: three taps can only fit a parabola, which has a visible
/// group-delay ripple at the top of the band.
#[inline]
fn cubic(ring: &[f32; REF_RING], index: usize, frac: f32) -> f32 {
    let prev = if index == 0 { REF_RING - 1 } else { index - 1 };
    let next = if index + 1 == REF_RING { 0 } else { index + 1 };
    let next2 = if index + 2 >= REF_RING {
        index + 2 - REF_RING
    } else {
        index + 2
    };
    let p0 = ring[prev];
    let p1 = ring[index];
    let p2 = ring[next];
    let p3 = ring[next2];
    let t2 = frac * frac;
    let t3 = t2 * frac;
    // Lagrange basis of the four taps at positions -1, 0, 1, 2 evaluated at
    // `frac` in [0, 1).
    let h0 = (-t3 + 3.0 * t2 - 2.0 * frac) * (1.0 / 6.0);
    let h1 = (t3 - 2.0 * t2 - frac + 2.0) * 0.5;
    let h2 = (-t3 + t2 + 2.0 * frac) * 0.5;
    let h3 = (t3 - frac) * (1.0 / 6.0);
    h0 * p0 + h1 * p1 + h2 * p2 + h3 * p3
}

/// Index of the strongest value of a lag profile.
fn argmax_profile(profile: &[f32; MAX_DELAY_DEC + 1]) -> usize {
    let mut best = 0usize;
    for lag in 1..=MAX_DELAY_DEC {
        if profile[lag] > profile[best] {
            best = lag;
        }
    }
    best
}

/// Sub-sample position and relative height of the peak nearest `centre`.
///
/// Looks one decimated bin either side of `centre`, which absorbs a half-bin
/// rounding of the tracked position without ever reaching a different arrival,
/// then refines with the three-point parabola of the peak. Returns the position
/// in decimated bins and the peak height relative to the profile maximum.
fn refine_peak(profile: &[f32; MAX_DELAY_DEC + 1], centre: f32, maximum: f32) -> (f32, f32) {
    let coarse = (centre.round().max(0.0) as usize).min(MAX_DELAY_DEC);
    let low = coarse.saturating_sub(1);
    let high = (coarse + 1).min(MAX_DELAY_DEC);
    let mut centre = low;
    for lag in low..=high {
        if profile[lag] > profile[centre] {
            centre = lag;
        }
    }
    let mut offset = 0.0f32;
    if centre > 0 && centre < MAX_DELAY_DEC {
        let left = profile[centre - 1];
        let middle = profile[centre];
        let right = profile[centre + 1];
        let denominator = left - 2.0 * middle + right;
        if denominator < 0.0 {
            offset = (0.5 * (left - right) / denominator).clamp(-0.5, 0.5);
        }
    }
    let height = if maximum > 0.0 {
        (profile[centre] / maximum).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (centre as f32 + offset, height)
}

/// Decimates a circular history by [`DECIMATION`] with the centred low-pass.
///
/// `pos` is the oldest sample. Output sample `j` is centred on raw sample
/// `DECIMATION * j`; samples before the start of the history are zero.
fn decimate<const R: usize>(ring: &[f32; R], pos: usize, out: &mut [f32]) {
    for (j, slot) in out.iter_mut().enumerate() {
        let center = DECIMATION * j;
        let mut sum = 0.0f32;
        for (tap, &weight) in DECIMATE_TAPS.iter().enumerate() {
            let offset = center + tap;
            if offset >= 3 {
                let mut index = pos + offset - 3;
                if index >= R {
                    index -= R;
                }
                sum += weight * ring[index];
            }
        }
        *slot = sum;
    }
}

/// Triangular smoothing of the PHAT magnitude at `lag`.
fn smooth(profile: &[f32; MAX_DELAY_DEC + 1], lag: usize) -> f32 {
    let half = (DELAY_ONSET_SMOOTHING - 1) / 2;
    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    for offset in -(half as isize)..=(half as isize) {
        let index = lag as isize + offset;
        if index < 0 || index as usize > MAX_DELAY_DEC {
            continue;
        }
        let weight = (half as isize + 1 - offset.abs()) as f32;
        sum += weight * profile[index as usize];
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        sum / weight_sum
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, seed: u32) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        let mut state = seed | 1;
        for value in &mut out {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *value = ((state >> 9) as f32) * (1.0 / 4_194_304.0) - 1.0;
        }
        out
    }

    fn run_estimator(reference: &[f32], mic: &[f32]) -> DelayEstimator {
        let frames = mic.len() / FRAME_LEN;
        let mut estimator = DelayEstimator::new(None);
        for frame in 0..frames {
            let range = frame * FRAME_LEN..(frame + 1) * FRAME_LEN;
            estimator.push(&reference[range.clone()], &mic[range]);
            estimator.tick(false, false);
        }
        estimator
    }

    /// Builds a band-limited reference (one-pole smoothed noise) and a
    /// microphone that is delay `first` before `switch_frame` and delay
    /// `second` afterwards.
    ///
    /// The reference is smoothed because the estimator resolves the onset on a
    /// four-sample grid: against white noise a one-bin offset already decorrelates,
    /// which is not what a speech-like reference does. The step tests care about
    /// the gate, not about that grid artifact.
    fn stepped_delay(
        frames: usize,
        first: usize,
        second: usize,
        switch_frame: usize,
        seed: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let len = frames * FRAME_LEN;
        let raw = noise(len, seed);
        let mut reference = vec![0.0f32; len];
        let mut state = 0.0f32;
        for (out, &value) in reference.iter_mut().zip(raw.iter()) {
            state = 0.8 * state + value;
            *out = state;
        }
        let switch = switch_frame * FRAME_LEN;
        let mut mic = vec![0.0f32; len];
        for n in 0..len {
            let delay = if n < switch { first } else { second };
            if n >= delay {
                mic[n] = 0.8 * reference[n - delay];
            }
        }
        (reference, mic)
    }

    #[test]
    fn recovers_a_known_delay() {
        let true_delay = 640usize;
        let frames = 64;
        let len = frames * FRAME_LEN;
        let reference = noise(len, 0xA5A5_1234);
        let mut mic = vec![0.0f32; len];
        for n in true_delay..len {
            mic[n] = 0.8 * reference[n - true_delay];
        }

        let estimator = run_estimator(&reference, &mic);
        // The estimator deliberately applies the onset minus the causal guard,
        // so compare the onset, not the applied value, with the true lag.
        let onset = estimator.applied() + DELAY_CAUSAL_GUARD;
        assert!(
            onset.abs_diff(true_delay) <= CONFIRM_TOLERANCE,
            "estimated onset {onset} (applied {}) for true {true_delay}",
            estimator.applied()
        );
    }

    #[test]
    fn first_onset_beats_a_stronger_late_reflection() {
        // Direct path at 480 and a louder reflection at 1780, the shape of the
        // diffuse-room failure. The global peak is the reflection; the onset
        // rule must still pick the direct arrival.
        let true_delay = 480usize;
        let reflection_delay = 1_780usize;
        let frames = 64;
        let len = frames * FRAME_LEN;
        let reference = noise(len, 0x0BAD_F00D);
        let mut mic = vec![0.0f32; len];
        for n in true_delay..len {
            mic[n] += 0.8 * reference[n - true_delay];
        }
        for n in reflection_delay..len {
            mic[n] += 1.0 * reference[n - reflection_delay];
        }

        let estimator = run_estimator(&reference, &mic);
        let onset = estimator.applied() + DELAY_CAUSAL_GUARD;
        assert!(
            onset.abs_diff(true_delay) <= CONFIRM_TOLERANCE,
            "estimated onset {onset} (applied {}) for direct {true_delay}",
            estimator.applied()
        );
    }

    #[test]
    fn re_lock_gate_is_direction_and_magnitude_aware() {
        let mut estimator = DelayEstimator::new(None);
        estimator.applied = 480;
        estimator.locked = true;

        // Small moves are tracking, forward or backward, at the tracking bar.
        assert!(estimator.relock_allowed(560, DELAY_RELOCK_SCORE_RATIO));
        assert!(estimator.relock_allowed(400, DELAY_RELOCK_SCORE_RATIO));
        assert!(!estimator.relock_allowed(400, DELAY_RELOCK_SCORE_RATIO - 0.01));

        // A large backward move is a pitch-period artefact, even when its plain
        // correlation is perfect.
        assert!(!estimator.relock_allowed(180, 1.0));
        assert!(!estimator.relock_allowed(0, 1.0));

        // A large forward move is possible in principle but must dominate the
        // plain correlation.
        assert!(estimator.relock_allowed(780, DELAY_RELOCK_LARGE_SCORE_RATIO));
        assert!(!estimator.relock_allowed(780, DELAY_RELOCK_LARGE_SCORE_RATIO - 0.01));
    }

    #[test]
    fn small_forward_step_is_tracked() {
        // The suite's +5 ms delay step is 80 samples: inside the tracking tier,
        // so it must be applied once two searches agree.
        let (reference, mic) = stepped_delay(640, 480, 560, 300, 0x51E9_0001);
        let estimator = run_estimator(&reference, &mic);
        let onset = estimator.applied() + DELAY_CAUSAL_GUARD;
        assert!(
            onset.abs_diff(560) <= CONFIRM_TOLERANCE,
            "estimated onset {onset} (applied {}) after a +80 step",
            estimator.applied()
        );
    }

    #[test]
    fn large_backward_jump_is_refused() {
        // Lock at 480, then the echo moves 320 samples earlier. A periodic
        // reference produces exactly this shape by faulting one pitch period
        // early; the gate must not follow it.
        let (reference, mic) = stepped_delay(640, 480, 160, 300, 0xBAAD_0002);
        let estimator = run_estimator(&reference, &mic);
        let onset = estimator.applied() + DELAY_CAUSAL_GUARD;
        assert!(
            onset.abs_diff(480) <= CONFIRM_TOLERANCE,
            "estimated onset {onset} (applied {}) followed a large backward jump",
            estimator.applied()
        );
    }

    #[test]
    fn large_forward_jump_needs_the_extra_measurement_and_dominance() {
        // Lock at 480, then the echo moves 520 samples later. This is a real
        // path change, but it is also where a late reflection would appear, so
        // it must be confirmed by three agreeing, dominant measurements.
        let (reference, mic) = stepped_delay(760, 480, 1_000, 300, 0xF0FA_0003);
        let estimator = run_estimator(&reference, &mic);
        let onset = estimator.applied() + DELAY_CAUSAL_GUARD;
        assert!(
            onset.abs_diff(1_000) <= CONFIRM_TOLERANCE,
            "estimated onset {onset} (applied {}) after a large forward jump",
            estimator.applied()
        );
    }

    #[test]
    fn a_stronger_late_reflection_does_not_steal_a_locked_alignment() {
        // The same response as `first_onset_beats_a_stronger_late_reflection`,
        // but run long enough for several slow re-searches. The reflection is
        // the dominant plain-correlation peak; the estimator must not abandon
        // the already-locked direct path for it. This is the A5 diffuse-room
        // failure in miniature.
        let true_delay = 480usize;
        let reflection_delay = 1_780usize;
        let frames = 640;
        let len = frames * FRAME_LEN;
        let reference = noise(len, 0x0BAD_F00D);
        let mut mic = vec![0.0f32; len];
        for n in true_delay..len {
            mic[n] += 0.8 * reference[n - true_delay];
        }
        for n in reflection_delay..len {
            mic[n] += 1.0 * reference[n - reflection_delay];
        }

        let estimator = run_estimator(&reference, &mic);
        let onset = estimator.applied() + DELAY_CAUSAL_GUARD;
        assert!(
            onset.abs_diff(true_delay) <= CONFIRM_TOLERANCE,
            "estimated onset {onset} (applied {}) moved to the late reflection at {reflection_delay}",
            estimator.applied()
        );
    }

    #[test]
    fn manual_override_is_reported_verbatim() {
        let mut estimator = DelayEstimator::new(Some(512));
        assert_eq!(estimator.applied(), 512);
        assert!(!estimator.set_override(Some(512)));
        assert!(estimator.set_override(Some(256)));
        assert_eq!(estimator.applied(), 256);

        let mut block = [0.0f32; FRAME_LEN];
        estimator.aligned_block(&mut block, None);
    }

    #[test]
    fn silence_is_not_locked() {
        let mut estimator = DelayEstimator::new(None);
        let silence = [0.0f32; FRAME_LEN];
        for _ in 0..64 {
            estimator.push(&silence, &silence);
            assert_eq!(estimator.tick(false, false), None);
        }
        assert_eq!(estimator.applied(), 0);
    }

    /// Fills the reference history with a unit ramp `ring[k] = k`, so the
    /// aligned read position is directly observable in the output samples.
    fn ramp_estimator() -> DelayEstimator {
        let mut estimator = DelayEstimator::new(Some(0));
        let mut reference = vec![0.0f32; 2 * REF_RING];
        for (index, value) in reference.iter_mut().enumerate() {
            *value = index as f32;
        }
        let mic = vec![0.0f32; FRAME_LEN];
        for frame in 0..(REF_RING / FRAME_LEN) {
            let range = frame * FRAME_LEN..(frame + 1) * FRAME_LEN;
            estimator.push(&reference[range], &mic);
        }
        estimator
    }

    #[test]
    fn rate_correction_slows_the_read_position() {
        // A ramp is reproduced exactly by a cubic, so the output is the read
        // position. 1000 ppm for one frame is a quarter of a sample: visible,
        // and small enough to stay well inside the causal window.
        let mut estimator = ramp_estimator();
        let mut out = [0.0f32; FRAME_LEN];
        estimator.aligned_block(&mut out, Some(1_000.0));
        // The uncorrected read starts at `REF_RING - FRAME_LEN`; the cubic path
        // deliberately reads `RATE_LOOKAHEAD` samples earlier.
        let first = (REF_RING - FRAME_LEN - RATE_LOOKAHEAD) as f32;
        for (n, &value) in out.iter().enumerate() {
            let expected = first + (n as f32) * (1.0 - 1_000.0e-6);
            assert!(
                (value - expected).abs() < 0.01,
                "sample {n}: {value} instead of {expected}"
            );
        }
    }

    #[test]
    fn a_negative_rate_advances_the_read_position() {
        let mut estimator = ramp_estimator();
        let mut out = [0.0f32; FRAME_LEN];
        estimator.aligned_block(&mut out, Some(-1_000.0));
        let first = (REF_RING - FRAME_LEN - RATE_LOOKAHEAD) as f32;
        for (n, &value) in out.iter().enumerate() {
            let expected = first + (n as f32) * (1.0 + 1_000.0e-6);
            assert!(
                (value - expected).abs() < 0.01,
                "sample {n}: {value} instead of {expected}"
            );
        }
    }

    #[test]
    fn the_rate_correction_read_position_is_continuous_across_frames() {
        // Two consecutive frames must continue the same warp: the second frame
        // starts where the first ended, with no integer or fractional jump.
        let mut estimator = ramp_estimator();
        let mut first = [0.0f32; FRAME_LEN];
        let mut second = [0.0f32; FRAME_LEN];
        estimator.aligned_block(&mut first, Some(1_000.0));
        // Push one more frame so the write head keeps up with the read.
        let frame: Vec<f32> = (0..FRAME_LEN).map(|n| (REF_RING + n) as f32).collect();
        estimator.push(&frame, &[0.0; FRAME_LEN]);
        estimator.aligned_block(&mut second, Some(1_000.0));
        // The ramp continues across the pushed frame, so the outputs are one
        // linear sequence: no discontinuity at the boundary.
        let step = second[0] - first[FRAME_LEN - 1];
        let expected = 1.0 - 1_000.0e-6;
        assert!(
            (step - expected).abs() < 0.01,
            "frame boundary advanced by {step} instead of {expected}"
        );
    }
}
