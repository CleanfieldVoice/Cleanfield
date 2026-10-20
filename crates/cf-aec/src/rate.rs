//! Clock-rate (sample-clock ratio) estimation for the far-end reference.
//!
//! # Why a rate estimate
//!
//! The far-end reference and the microphone are sampled by two free-running
//! clocks. A ratio of `1 + r` between them turns the echo path
//! `microphone[n] ~ reference[n - D]` into `microphone[n] ~ reference[(1 - r) n - D]`:
//! the bulk delay the adaptive filter has to model moves by `r` samples per
//! sample. The partitioned filter cannot follow that moving optimum (it is 33
//! samples over a 10 s file at 200 ppm), so the echo return loss collapses even
//! though the bulk-delay estimator never re-locks. Removing `r` from the
//! reference before adaptation turns the path back into a static one; the
//! measured ceiling for an exact correction is the static bound.
//!
//! # Estimator
//!
//! The estimator is deliberately slow and lives on data the AEC already
//! produces: every bulk-delay search correlates the microphone against a
//! bounded reference history and builds a whitened correlation profile. The
//! search resolves the *first onset* (a quantised, decimated quantity used for
//! the alignment); this module instead takes the **dominant peak of the same
//! profile**, which is a smooth extremum, and refines it to sub-sample
//! resolution with a parabolic fit over the three bins around it. The whole
//! echo path - direct arrival and every reflection - is delayed by the same
//! `r` samples per sample, so any stable peak of the profile carries the rate;
//! a peak tracker that stays within [`RATE_PEAK_TRACK_BINS`] of the previous
//! measurement keeps the estimate on one arrival instead of hopping between
//! reflections.
//!
//! Each measurement is `(sample index, peak lag)`. The rate is the slope of a
//! straight line through the last [`RATE_WINDOW`] measurements, in samples per
//! sample, reported in ppm. The fit is made robust in four ways:
//!
//! * **outlier rejection**: a first fit, then every measurement further than
//!   `3.5 * 1.4826 * MAD` from the median residual (and at least
//!   `RATE_REJECT_FLOOR_SAMPLES`) is dropped and the line refitted;
//! * **peak weight**: a measurement whose peak is less than
//!   [`RATE_MIN_QUALITY`] of the strongest peak in the same search never
//!   enters the window;
//! * **residual bound**: a fit whose residual RMS exceeds
//!   `RATE_MAX_FIT_RMS_SAMPLES` is not trusted, which is what keeps a delay
//!   *step* from being mistaken for a ramp - a step leaves a large residual
//!   against any straight line;
//! * **statistics**: the full-window slope must be at least
//!   `RATE_SIGNIFICANCE` standard errors from zero, the total displacement it
//!   predicts must be at least `RATE_MIN_DISPLACEMENT_SAMPLES`, and each half
//!   of the window must show a slope of the same sign at
//!   `RATE_HALF_SIGNIFICANCE` standard errors. A static path has no consistent
//!   slope to find, so it never reaches the significance bar and the
//!   correction stays bypassed.
//!
//! The estimate is only *applied* once it has been significant for
//! [`RATE_CONFIRM_UPDATES`] consecutive updates, after which the correction is
//! latched. From then on the applied rate is an exponential average of the
//! fitted slope (`RATE_SMOOTHING`) with a per-update slew bound
//! (`RATE_SLEW_PPM`), clamped to `+-RATE_MAX_PPM`, and set to zero while the
//! average is inside the `+-RATE_DEAD_BAND_PPM` dead band. A single bad
//! measurement carries weight `1 / n` in the least-squares fit and at most
//! `RATE_SMOOTHING / n` after smoothing, so it cannot move the applied rate.
//!
//! # Window and response time
//!
//! The window holds [`RATE_WINDOW`] measurements. A fit needs at least
//! `RATE_MIN_OBSERVATIONS` of them spanning at least
//! [`RATE_MIN_SPAN_SAMPLES`] (1.28 s), which is why it cannot be fooled by a few
//! measurements bunched together.
//!
//! Measurements arrive with the delay analysis. While the estimator is still
//! acquiring (and before the first delay lock) the analysis runs every 16 frames
//! (256 ms), so the six-measurement first fit is possible after about 1.4 s of
//! stream; once the window is full, or the correction has converged, the
//! analysis falls back to every 64 frames (1.024 s), and it does not run at all
//! while the double-talk detector holds it back. The acquisition cadence is
//! capped at 40 extra analyses (10.2 s of stream) by the delay estimator, so a
//! stream with no rate to find pays that budget and then returns to the
//! pre-existing cadence without ever engaging the resampler.
//!
//! The measured engagement on the drift suite is frame 206 mean (3.3 s,
//! range 160-400 frames) and the applied rate settles within a few ppm; the
//! exponential smoothing adds about one further update interval of lag. The
//! window is a ring, so a rate *change* is fully absorbed after
//! [`RATE_WINDOW`] measurements (about 16 s at the slow cadence) - slow by
//! design, because a clock ratio is a stable property of the hardware.
//!
//! The estimator performs no allocation and no transcendental functions.

/// Largest clock-rate correction that can be applied, in ppm.
///
/// Consumer audio clocks are specified to about +-100 ppm; 500 ppm is five
/// times that and still small enough that the fractional resampler stays in
/// its accurately interpolated range. An estimate outside this bound is
/// clamped rather than rejected, so a wrong fit cannot run away.
pub const RATE_MAX_PPM: f32 = 500.0;

/// Rate magnitude below which no correction is applied, in ppm.
///
/// At 5 ppm the residual drift costs about 0.4 dB of the static bound (10 ppm
/// costs 1.7 dB), which is well below the measurement noise of this estimator;
/// applying smaller estimates would only add interpolation work.
pub const RATE_DEAD_BAND_PPM: f32 = 5.0;

/// Measurements kept for the slope fit. A ring; the oldest is dropped.
///
/// Sixteen measurements at the acquisition cadence is 4.1 s of history, which
/// is long enough that the standard error of the slope is a few ppm and short
/// enough that the estimate follows a real clock change within seconds.
const RATE_WINDOW: usize = 16;

/// Minimum number of measurements in a fit.
///
/// Six is the smallest count that gives each half of the window three points,
/// which the half-window consistency test needs.
const RATE_MIN_OBSERVATIONS: usize = 6;

/// Minimum span of the fit window, in samples (1.28 s).
///
/// Six measurements at the acquisition cadence span exactly this, so the first
/// fit is possible 1.28 s after the sixth measurement. The bound exists so a
/// fit is never made from measurements bunched into a few frames.
const RATE_MIN_SPAN_SAMPLES: usize = 20_480;

/// Largest residual RMS of a fit that may be trusted, in samples.
///
/// A drifting path fits a straight line to well under a sample. A delay step
/// (the suite's +5 ms step is 80 samples) leaves a residual of the order of the
/// step, so this single test is what separates "the path is ramping" from "the
/// path moved".
const RATE_MAX_FIT_RMS_SAMPLES: f32 = 4.0;

/// Full-window slope significance required for the correction to engage, in
/// standard errors.
const RATE_SIGNIFICANCE: f32 = 3.0;

/// Per-half slope significance required for the correction to engage, in
/// standard errors. Both halves must also agree in sign with the full fit.
const RATE_HALF_SIGNIFICANCE: f32 = 2.5;

/// Total displacement over the fit window required to engage, in samples.
///
/// A static path can produce a significant slope by chance when the peak
/// tracker wobbles, but the displacement it predicts over the window is then a
/// small fraction of a sample. A real 200 ppm drift moves about six samples
/// over the six-measurement first fit. Four samples is several standard errors
/// of the tracker noise at every window length used here, and it is the bound
/// that keeps a static path bypassed.
const RATE_MIN_DISPLACEMENT_SAMPLES: f32 = 4.0;

/// Consecutive significant updates required before the correction is applied.
const RATE_CONFIRM_UPDATES: u8 = 3;

/// Rejection floor of the robust fit, in samples. Residuals below this are
/// always kept, so a nearly perfect line is not thinned by the MAD test.
const RATE_REJECT_FLOOR_SAMPLES: f32 = 1.0;

/// Exponential smoothing of the applied rate, per update.
const RATE_SMOOTHING: f32 = 0.5;

/// Bound on how far the smoothed rate may move in one update, in ppm.
const RATE_SLEW_PPM: f32 = 60.0;

/// Half-width of the peak-tracking window, in decimated bins (four samples
/// each).
///
/// Six bins is 24 samples: wider than the drift between two analyses (under one
/// sample at the fast cadence, about fourteen across a double-talk gap) and
/// narrower than the separation between the arrivals of the responses in use,
/// so a competing reflection cannot take over the track. The weak-direct
/// response sets the bound: its direct path and its first strong reflection are
/// about 72 samples apart, and a wider window lets the tracker alternate
/// between them, which destroys the slope fit.
pub(crate) const RATE_PEAK_TRACK_BINS: usize = 6;

/// Relative peak height below which a measurement is discarded.
pub(crate) const RATE_MIN_QUALITY: f32 = 0.5;

/// Consecutive lost analyses after which the peak tracker snaps to the
/// strongest peak in its own analysis.
pub(crate) const RATE_REANCHOR_ANALYSES: u8 = 3;

/// Lag jump, in decimated bins, that marks a measurement as discontinuous from
/// the previous one.
///
/// The tracker reports a jump this large only when it has moved to a different
/// arrival (a re-anchor after losing its peak). The two measurements are then
/// in different lag coordinates and cannot both belong to one ramp, so the fit
/// window is restarted instead of mixing them.
pub(crate) const RATE_JUMP_BINS: usize = 8;

/// Per-half slope significance as `f64`, for the fit arithmetic.
const RATE_HALF_SIGNIFICANCE_F64: f64 = RATE_HALF_SIGNIFICANCE as f64;

/// One delay measurement contributed by the bulk-delay search.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RateObservation {
    /// Samples processed when the measurement was taken.
    pub(crate) at: usize,
    /// Lag of the dominant correlation peak, in samples (sub-sample).
    pub(crate) lag: f32,
    /// Peak height relative to the strongest peak in the same search, `0..=1`.
    pub(crate) quality: f32,
    /// Set when the measurement is discontinuous with the ones before it, so
    /// the fit window must restart before it is added.
    pub(crate) restart: bool,
}

/// One straight-line fit, in the units of the fit (`samples` per `samples`).
#[derive(Clone, Copy, Debug)]
struct Line {
    intercept: f64,
    slope: f64,
    rss: f64,
    sxx: f64,
    count: usize,
}

impl Line {
    fn at(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }

    /// Standard error of the slope, from the residual sum of squares.
    fn slope_error(&self) -> f64 {
        if self.count > 2 && self.sxx > 0.0 {
            (self.rss / ((self.count - 2) as f64) / self.sxx).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Residual RMS of the fit.
    fn residual_rms(&self) -> f64 {
        if self.count == 0 {
            return f64::INFINITY;
        }
        (self.rss / (self.count as f64)).sqrt()
    }
}

/// Result of a trusted rate fit, in ppm.
#[derive(Clone, Copy, Debug)]
struct Fit {
    ppm: f32,
    ppm_error: f32,
    rms: f32,
    /// Total displacement over the window, in samples.
    displacement: f32,
    /// Whether both halves of the window show the same significant slope.
    consistent: bool,
}

/// Ordinary least squares on `(xs, ys)`.
fn least_squares(xs: &[f64], ys: &[f64]) -> Option<Line> {
    let count = xs.len().min(ys.len());
    if count < 3 {
        return None;
    }
    let scale = 1.0 / (count as f64);
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for i in 0..count {
        mean_x += xs[i];
        mean_y += ys[i];
    }
    mean_x *= scale;
    mean_y *= scale;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..count {
        let dx = xs[i] - mean_x;
        sxx += dx * dx;
        sxy += dx * (ys[i] - mean_y);
    }
    if !(sxx > 0.0) {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let mut rss = 0.0;
    for i in 0..count {
        let residual = ys[i] - (intercept + slope * xs[i]);
        rss += residual * residual;
    }
    Some(Line {
        intercept,
        slope,
        rss,
        sxx,
        count,
    })
}

/// Median of the first `count` values, sorting them in place.
fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let count = values.len();
    // Insertion sort: the window is at most `RATE_WINDOW` entries.
    for i in 1..count {
        let value = values[i];
        let mut j = i;
        while j > 0 && values[j - 1] > value {
            values[j] = values[j - 1];
            j -= 1;
        }
        values[j] = value;
    }
    if count % 2 == 1 {
        values[count / 2]
    } else {
        0.5 * (values[count / 2 - 1] + values[count / 2])
    }
}

/// Slow, robust clock-rate estimator.
pub(crate) struct RateEstimator {
    observations: [RateObservation; RATE_WINDOW],
    /// Index one past the newest observation.
    head: usize,
    /// Number of valid entries in the ring.
    filled: usize,
    /// Smoothed rate estimate, in ppm, clamped to `+-RATE_MAX_PPM`.
    ppm: f32,
    /// Rate handed to the resampler, in ppm. Zero while bypassed.
    applied_ppm: f32,
    /// Latched once the estimate has been significant for
    /// `RATE_CONFIRM_UPDATES` updates. Never cleared by an update, so a later
    /// quiet window cannot switch the correction off discontinuously.
    converged: bool,
    /// Consecutive significant updates so far.
    significant: u8,
    /// Master switch.
    enabled: bool,
}

impl RateEstimator {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            observations: [RateObservation {
                at: 0,
                lag: 0.0,
                quality: 0.0,
                restart: false,
            }; RATE_WINDOW],
            head: 0,
            filled: 0,
            ppm: 0.0,
            applied_ppm: 0.0,
            converged: false,
            significant: 0,
            enabled,
        }
    }

    /// Enables or disables the estimator. A change clears the estimate so that
    /// a re-enable starts a fresh acquisition instead of applying a stale rate.
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        if self.enabled != enabled {
            self.reset();
            self.enabled = enabled;
        }
    }

    /// Clears every estimate and the observation window.
    pub(crate) fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
        self.ppm = 0.0;
        self.applied_ppm = 0.0;
        self.converged = false;
        self.significant = 0;
    }

    /// Feeds one delay measurement and updates the estimate.
    pub(crate) fn observe(&mut self, observation: RateObservation) {
        if !observation.quality.is_finite()
            || !observation.lag.is_finite()
            || observation.quality < RATE_MIN_QUALITY
        {
            return;
        }
        if observation.restart {
            // The measurements before the jump are in another lag coordinate.
            self.head = 0;
            self.filled = 0;
            self.significant = 0;
        }
        self.observations[self.head] = observation;
        self.head = (self.head + 1) % RATE_WINDOW;
        self.filled = (self.filled + 1).min(RATE_WINDOW);
        self.update();
    }

    /// Rate correction to apply, in ppm: `None` while the estimate has not
    /// converged (the reference path is the identity), `Some(ppm)` afterwards.
    pub(crate) fn correction(&self) -> Option<f32> {
        if self.converged {
            Some(self.applied_ppm)
        } else {
            None
        }
    }

    /// Applied rate in ppm, zero while bypassed or inside the dead band.
    pub(crate) fn applied_ppm(&self) -> f32 {
        self.applied_ppm
    }

    /// Whether the estimate has converged and the correction is active.
    pub(crate) fn is_corrected(&self) -> bool {
        self.converged
    }

    /// Whether the estimator still needs fast delay analyses.
    ///
    /// True until the correction has engaged, or until the fit window is full
    /// without one: the acquisition phase then sees four times as many
    /// measurements per second, and a static path - which never converges -
    /// falls back to the slow cadence once its window is full instead of paying
    /// the fast cadence forever.
    pub(crate) fn wants_fast(&self) -> bool {
        self.enabled && !self.converged && self.filled < RATE_WINDOW
    }

    /// Fits the observation window and advances the estimate.
    fn update(&mut self) {
        if !self.enabled {
            self.applied_ppm = 0.0;
            return;
        }
        let Some(fit) = self.fit() else {
            return;
        };
        let accepted = fit.ppm.abs() >= RATE_SIGNIFICANCE * fit.ppm_error
            && fit.ppm.abs() >= RATE_DEAD_BAND_PPM
            && fit.displacement >= RATE_MIN_DISPLACEMENT_SAMPLES
            && fit.consistent
            && fit.rms <= RATE_MAX_FIT_RMS_SAMPLES;
        if !self.converged {
            if accepted {
                self.significant = self.significant.saturating_add(1);
            } else {
                self.significant = 0;
            }
            if self.significant < RATE_CONFIRM_UPDATES {
                self.applied_ppm = 0.0;
                return;
            }
            self.converged = true;
            // Start at the fitted value: the fit is already an average over
            // the whole window, so a single bad measurement cannot have
            // dominated it.
            self.ppm = clamp_ppm(fit.ppm);
        } else {
            let target = clamp_ppm(fit.ppm);
            let delta = (target - self.ppm).clamp(-RATE_SLEW_PPM, RATE_SLEW_PPM);
            self.ppm = clamp_ppm(self.ppm + RATE_SMOOTHING * delta);
        }
        self.applied_ppm = if self.ppm.abs() < RATE_DEAD_BAND_PPM {
            0.0
        } else {
            self.ppm
        };
    }

    /// Robust fit of the observation window.
    fn fit(&self) -> Option<Fit> {
        let mut xs = [0.0f64; RATE_WINDOW];
        let mut ys = [0.0f64; RATE_WINDOW];
        if self.filled < RATE_MIN_OBSERVATIONS {
            return None;
        }
        let oldest = (self.head + RATE_WINDOW - self.filled) % RATE_WINDOW;
        let base = self.observations[oldest].at;
        for k in 0..self.filled {
            let observation = self.observations[(oldest + k) % RATE_WINDOW];
            xs[k] = observation.at.saturating_sub(base) as f64;
            ys[k] = f64::from(observation.lag);
        }
        let count = self.filled;
        if xs[count - 1] - xs[0] < RATE_MIN_SPAN_SAMPLES as f64 {
            return None;
        }

        // First pass, then reject measurements that the robust spread says are
        // not on the line, then refit. At least `RATE_MIN_OBSERVATIONS` points
        // are always kept so the rejection cannot empty the window.
        let first = least_squares(&xs[..count], &ys[..count])?;
        let mut residuals = [0.0f32; RATE_WINDOW];
        let mut deviations = [0.0f32; RATE_WINDOW];
        for i in 0..count {
            residuals[i] = (ys[i] - first.at(xs[i])) as f32;
        }
        // `median` sorts its argument, so work on copies: the per-point
        // deviations must stay in observation order for the selection below.
        let mut scratch = residuals;
        let centre = median(&mut scratch[..count]);
        for i in 0..count {
            deviations[i] = (residuals[i] - centre).abs();
        }
        let mut scratch = deviations;
        let mad = median(&mut scratch[..count]);
        let threshold = (3.5 * 1.4826 * mad).max(RATE_REJECT_FLOOR_SAMPLES);

        let mut passing = 0usize;
        for &deviation in deviations.iter().take(count) {
            if deviation <= threshold {
                passing += 1;
            }
        }
        let keep_all = passing < RATE_MIN_OBSERVATIONS;
        let mut kept_x = [0.0f64; RATE_WINDOW];
        let mut kept_y = [0.0f64; RATE_WINDOW];
        let mut kept = 0usize;
        for i in 0..count {
            if keep_all || deviations[i] <= threshold {
                kept_x[kept] = xs[i];
                kept_y[kept] = ys[i];
                kept += 1;
            }
        }
        let line = if kept == count {
            first
        } else {
            least_squares(&kept_x[..kept], &kept_y[..kept])?
        };
        if !line.slope.is_finite() || !line.slope_error().is_finite() {
            return None;
        }
        let slope = line.slope;
        let error = line.slope_error();
        let span = kept_x[kept - 1] - kept_x[0];

        // Each half of the window must show a significant slope of the same
        // sign as the full fit. A static path produces two independent noise
        // slopes, so they rarely agree; a real ramp produces two half-slopes
        // that both estimate the same rate.
        let half = kept / 2;
        let consistent = half >= 3 && kept - half >= 3 && {
            let older = least_squares(&kept_x[..half], &kept_y[..half]);
            let newer = least_squares(&kept_x[half..kept], &kept_y[half..kept]);
            match (older, newer) {
                (Some(older), Some(newer)) => {
                    let positive = slope > 0.0;
                    let agrees = |part: &Line| {
                        let error = part.slope_error();
                        part.slope.is_finite()
                            && error.is_finite()
                            && part.slope.abs() >= RATE_HALF_SIGNIFICANCE_F64 * error
                            && (part.slope > 0.0) == positive
                    };
                    agrees(&older) && agrees(&newer)
                }
                _ => false,
            }
        };

        Some(Fit {
            ppm: (slope * 1.0e6) as f32,
            ppm_error: (error * 1.0e6) as f32,
            rms: line.residual_rms() as f32,
            displacement: (slope * span).abs() as f32,
            consistent,
        })
    }
}

/// Clamps a rate to the supported range, replacing non-finite values.
fn clamp_ppm(ppm: f32) -> f32 {
    if !ppm.is_finite() {
        return 0.0;
    }
    ppm.clamp(-RATE_MAX_PPM, RATE_MAX_PPM)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: usize = crate::FRAME_LEN;

    fn observation(at: usize, lag: f32) -> RateObservation {
        RateObservation {
            at,
            lag,
            quality: 1.0,
            restart: false,
        }
    }

    /// Feeds a ramp of `ppm` at the production slow cadence and returns the
    /// estimator after `count` measurements.
    fn ramp(ppm: f32, count: usize) -> RateEstimator {
        let mut estimator = RateEstimator::new(true);
        let rate = ppm * 1.0e-6;
        for k in 0..count {
            let at = (48 + 64 * k) * FRAME;
            estimator.observe(observation(at, 480.0 + rate * at as f32));
        }
        estimator
    }

    #[test]
    fn a_high_confidence_ramp_is_estimated() {
        let estimator = ramp(200.0, 9);
        let correction = estimator.correction().expect("the ramp must converge");
        assert!(
            (correction - 200.0).abs() < 1.0,
            "estimated {correction:.1} ppm"
        );
        assert!(estimator.is_corrected());
    }

    #[test]
    fn negative_rates_are_estimated_too() {
        let estimator = ramp(-180.0, 9);
        let correction = estimator.correction().expect("the ramp must converge");
        assert!(
            (correction + 180.0).abs() < 1.0,
            "estimated {correction:.1} ppm"
        );
    }

    #[test]
    fn a_static_path_is_not_corrected() {
        // Pure measurement noise around a constant lag, the shape a static
        // path produces. The correction must stay bypassed.
        let mut state = 0x1234_5678u32;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 9) as f32) * (1.0 / 4_194_304.0) - 1.0
        };
        for seed in 0..8 {
            let mut estimator = RateEstimator::new(true);
            for k in 0..24 {
                let at = (48 + 64 * k) * FRAME;
                let lag = 480.0 + 0.6 * noise() + 0.1 * (seed as f32);
                estimator.observe(observation(at, lag));
            }
            assert!(
                estimator.correction().is_none(),
                "a static path was corrected by {:?} ppm",
                estimator.correction()
            );
        }
    }

    #[test]
    fn a_delay_step_is_not_mistaken_for_a_ramp() {
        let mut estimator = RateEstimator::new(true);
        for k in 0..9 {
            let at = (48 + 64 * k) * FRAME;
            let lag = if k < 4 { 480.0 } else { 560.0 };
            estimator.observe(observation(at, lag));
        }
        assert!(
            estimator.correction().is_none(),
            "a delay step was read as {:?} ppm",
            estimator.correction()
        );
    }

    #[test]
    fn one_outlier_cannot_move_the_estimate() {
        let mut estimator = RateEstimator::new(true);
        let rate = 200.0e-6;
        for k in 0..9 {
            let at = (48 + 64 * k) * FRAME;
            let mut lag = 480.0 + rate * at as f32;
            if k == 5 {
                lag += 40.0;
            }
            estimator.observe(observation(at, lag));
        }
        let correction = estimator.correction().expect("the ramp must converge");
        assert!(
            (correction - 200.0).abs() < 2.0,
            "an outlier moved the estimate to {correction:.1} ppm"
        );
    }

    #[test]
    fn a_low_quality_measurement_is_ignored() {
        let mut estimator = RateEstimator::new(true);
        for k in 0..9 {
            let at = (48 + 64 * k) * FRAME;
            estimator.observe(observation(at, 480.0 + 200.0e-6 * at as f32));
            estimator.observe(RateObservation {
                at,
                lag: 1_500.0,
                quality: 0.1,
                restart: false,
            });
        }
        let correction = estimator.correction().expect("the ramp must converge");
        assert!(
            (correction - 200.0).abs() < 2.0,
            "a low-quality measurement moved the estimate to {correction:.1} ppm"
        );
    }

    #[test]
    fn the_applied_rate_is_clamped() {
        let estimator = ramp(900.0, 9);
        let correction = estimator.correction().expect("the ramp must converge");
        assert!(
            correction <= RATE_MAX_PPM + 1.0e-3,
            "clamp did not hold: {correction}"
        );
    }

    #[test]
    fn a_disabled_estimator_stays_bypassed() {
        let mut estimator = RateEstimator::new(false);
        for k in 0..16 {
            let at = (48 + 64 * k) * FRAME;
            estimator.observe(observation(at, 480.0 + 200.0e-6 * at as f32));
        }
        assert!(estimator.correction().is_none());
        assert_eq!(estimator.applied_ppm(), 0.0);
    }

    #[test]
    fn the_estimate_latches_and_settles_inside_the_slew_bound() {
        // A ramp that starts late: the applied rate must move by no more than
        // the slew bound per update, and it must not overshoot.
        let mut estimator = RateEstimator::new(true);
        let mut at = 48 * FRAME;
        let mut lag = 480.0f32;
        for _ in 0..9 {
            estimator.observe(observation(at, lag));
            at += 64 * FRAME;
            lag += 200.0e-6 * (64 * FRAME) as f32;
        }
        let mut previous = estimator.applied_ppm();
        for _ in 0..12 {
            estimator.observe(observation(at, lag));
            at += 64 * FRAME;
            lag += 200.0e-6 * (64 * FRAME) as f32;
            let next = estimator.applied_ppm();
            assert!(
                (next - previous).abs() <= RATE_SLEW_PPM + 1.0e-3,
                "applied rate moved {:.1} ppm in one update",
                next - previous
            );
            previous = next;
        }
        assert!(
            (previous - 200.0).abs() < 1.0,
            "settled at {previous:.1} ppm"
        );
    }

    #[test]
    fn a_reset_clears_the_estimate() {
        let mut estimator = ramp(200.0, 9);
        assert!(estimator.is_corrected());
        estimator.reset();
        assert!(!estimator.is_corrected());
        assert_eq!(estimator.applied_ppm(), 0.0);
        assert!(estimator.correction().is_none());
    }
}
