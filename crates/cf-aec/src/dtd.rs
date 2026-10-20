//! Double-talk detection.
//!
//! The rule is deliberately simple and explainable. After a short warm-up it
//! compares two per-frame quantities:
//!
//! * `near_end_ratio = 1 - echo_power / mic_power` -- how much of the
//!   microphone power the current echo estimate does not explain;
//! * `coherence = |<mic, echo>| / sqrt(<mic, mic> * <echo, echo>)` -- how much
//!   the microphone and the echo estimate look like the same signal.
//!
//! Echo alone, with a converged filter, gives a ratio near zero and a
//! coherence near one. Near-end speech raises the ratio and lowers the
//! coherence. Adaptation is frozen while either the rule fires or a short
//! hangover from the last detection lasts, which avoids chattering at syllable
//! boundaries. A freeze is capped and followed by a guaranteed adaptation
//! window, so a detection that stayed active after the echo path moved cannot
//! latch the filter forever (see `DT_MAX_FREEZE_FRAMES`).
//!
//! # The stationary-floor gate
//!
//! The ratio and coherence tests alone fire during a far-end pause: the echo
//! estimate is then near zero, so the ratio is one and the coherence is
//! undefined (zero), which is indistinguishable from near-end speech. With a
//! stationary background noise that happens for a large share of the frames,
//! and in a drifted stream it keeps the filter from re-converging after double
//! talk. Measurement on the office/pink/15 dB/double-talk scene showed that 40 %
//! of the false post-window freezes are literal far-end pauses and 67 % have a
//! microphone power within 2x of the true background noise power, while only
//! 20 % of the freezes during real double talk are that close to the floor.
//!
//! The detector therefore keeps a slow noise-floor estimate of the microphone
//! power -- fast down, slow up, so it sits on the quietest frames (pauses) and
//! does not climb through speech -- and requires the microphone to be at least
//! [`DT_NOISE_FLOOR_GATE`] times that floor before it may call a frame
//! near-end. A far-end pause with noise only is no longer near-end; real speech,
//! and a near end that is active while the far end is silent, are far above the
//! floor and are still detected. The estimate is deliberately slow: during a
//! long double-talk burst it must not climb up to the near-end level and start
//! masking the detection it is supposed to allow.
//!
//! The warm-up is not a tuning excuse: immediately after a reset the echo
//! estimate is zero, so the ratio is exactly one and the coherence is
//! undefined. Disabling the rule for the first [`DT_WARMUP_FRAMES`] frames lets
//! the estimate form; a reset (stream start, or an accepted delay change)
//! restarts the warm-up.

/// Frames after a reset during which adaptation is never frozen.
pub const DT_WARMUP_FRAMES: u32 = 48;

/// Near-end ratio above which the microphone is "not explained by the echo".
/// A value of 0.5 means the echo estimate has to explain at least half of the
/// microphone power before the ratio test is even considered.
pub const DT_NEAR_END_RATIO: f32 = 0.5;

/// Coherence below which the microphone is "not the same signal as the echo".
pub const DT_COHERENCE: f32 = 0.7;

/// Microphone mean power below which double talk is not declared.
pub const DT_MIC_POWER_FLOOR: f32 = 1.0e-8;

/// Microphone power must exceed this multiple of the slow background-noise
/// floor before a frame may be near-end. At the floor itself (a far-end pause
/// with stationary noise) the ratio test would otherwise always fire, because
/// there is no echo left to explain and the coherence is undefined.
pub const DT_NOISE_FLOOR_GATE: f32 = 3.0;

/// Per-frame upward leakage of the slow noise-floor estimate. At 16 ms per
/// frame, 0.001 is a 16 s time constant: slow enough that a 2.9 s double-talk
/// segment cannot drag the floor up to the near-end level, fast enough to
/// follow a real change in the background noise over a few seconds.
pub const DT_NOISE_FLOOR_RISE: f32 = 0.001;

/// The floor gate only applies when the stream actually has a stationary
/// background: the quietest valid frame must be at least this fraction of the
/// slow mean microphone power.
///
/// A clean room has no such floor. Its quietest frames are the decaying room
/// tail, which is far below the mean (measured 0.00038 on the clean hall),
/// whereas the noise scenes sit at about 0.01-0.027. Without this condition the
/// gate also removes the harmless pause freezes in a clean long room and costs
/// about 0.4 dB there; with it, the clean static controls keep the old rule
/// (unchanged to the last displayed digit).
pub const DT_NOISE_PRESENCE_SHARE: f32 = 0.0025;

/// Per-frame rise of the slow microphone-power mean used by
/// [`DT_NOISE_PRESENCE_SHARE`]. At 16 ms per frame, 0.002 is about 8 s.
pub const DT_MIC_MEAN_RISE: f32 = 0.002;

/// Frames before the stationary-floor gate may arm. At stream start the
/// quietest frame seen so far is simply the most recent one, so the presence
/// ratio is one and the gate would arm in every room for the first second.
pub const DT_NOISE_PRESENCE_SETTLE_FRAMES: u32 = 64;

/// The quietest frame must also be meaningfully below the mean, or no pause has
/// been observed yet and there is nothing to call a floor.
pub const DT_NOISE_PRESENCE_MAX_SHARE: f32 = 0.5;

/// Cancellation the filter must have demonstrated before double talk can be
/// declared, in dB, sustained for [`DT_CONFIRM_FRAMES`] consecutive frames.
///
/// This is a cold-start guard, not a double-talk threshold. Immediately after
/// a reset the estimate is exactly zero and grows only while adaptation runs;
/// without the guard a slow-converging room (a long hall, a diffuse tail, a
/// late direct path) still looks like double talk after the fixed warm-up and
/// the detector freezes adaptation permanently. Requiring sustained measured
/// cancellation first makes the detector wait for evidence that the filter
/// works, and the latch is kept until reset, so a later strong near-end signal
/// cannot re-enter the cold-start path and suppress a legitimate detection. The
/// latch only gates the first declaration; [`DT_MAX_FREEZE_FRAMES`] independently
/// prevents a stale detection from freezing adaptation forever.
pub const DT_MIN_CANCELLATION_DB: f32 = 6.0;

/// Consecutive frames above [`DT_MIN_CANCELLATION_DB`] needed to latch the
/// cold-start guard. A single lucky block must not unlock the detector.
pub const DT_CONFIRM_FRAMES: u8 = 8;

/// Frames the detection is held after the rule stops firing.
pub const DT_HANGOVER_FRAMES: u8 = 4;

/// Longest uninterrupted freeze, in frames. A detector that was right when it
/// fired can still be wrong about the reason: if the echo path moves while the
/// filter is frozen (clock drift, a delay step), the stale estimate leaves a
/// residual that keeps the detector firing, so adaptation never restarts. A
/// hard cap turns that latch into a duty cycle.
const DT_MAX_FREEZE_FRAMES: u8 = 32;

/// Frames of unconditional adaptation after [`DT_MAX_FREEZE_FRAMES`]. Short
/// enough that near-end speech is still mostly protected, long enough for the
/// filter to reduce a stale-residual deadlock.
const DT_RELEASE_FRAMES: u8 = 16;

/// Stateful per-stream detector.
pub(crate) struct DoubleTalkDetector {
    frames: u32,
    hangover: u8,
    active: bool,
    /// Consecutive frames that cleared the cold-start cancellation bar.
    formed_frames: u8,
    /// Latched once the filter has demonstrated sustained cancellation. Kept
    /// until reset so a later strong near-end signal cannot re-enter the
    /// cold-start path and suppress a legitimate detection.
    formed: bool,
    /// Consecutive frames the detector has frozen adaptation.
    frozen_frames: u8,
    /// Remaining release frames during which detection is suppressed.
    release_left: u8,
    /// Slow estimate of the stationary background power (fast down, slow up).
    noise_floor: f32,
    /// Whether [`Self::noise_floor`] has been seeded from a first frame.
    noise_floor_ready: bool,
    /// Quietest valid microphone frame seen since the reset. A clean room's
    /// tail decays far below its mean; a stationary noise floor does not.
    quiet_floor: f32,
    /// Whether [`Self::quiet_floor`] has been seeded from a valid frame.
    quiet_floor_ready: bool,
    /// Slow mean of the microphone power, the scale for the presence test.
    mic_mean: f32,
    /// Whether [`Self::mic_mean`] has been seeded from a valid frame.
    mic_mean_ready: bool,
}

impl DoubleTalkDetector {
    pub(crate) fn new() -> Self {
        Self {
            frames: 0,
            hangover: 0,
            active: false,
            formed_frames: 0,
            formed: false,
            frozen_frames: 0,
            release_left: 0,
            noise_floor: 0.0,
            noise_floor_ready: false,
            quiet_floor: 0.0,
            quiet_floor_ready: false,
            mic_mean: 0.0,
            mic_mean_ready: false,
        }
    }

    /// Clears the warm-up and the hangover counter.
    pub(crate) fn reset(&mut self) {
        self.frames = 0;
        self.hangover = 0;
        self.active = false;
        self.formed_frames = 0;
        self.formed = false;
        self.frozen_frames = 0;
        self.release_left = 0;
        self.noise_floor = 0.0;
        self.noise_floor_ready = false;
        self.quiet_floor = 0.0;
        self.quiet_floor_ready = false;
        self.mic_mean = 0.0;
        self.mic_mean_ready = false;
    }

    /// Feeds one frame of features and returns the new adaptation-freeze state.
    pub(crate) fn update(
        &mut self,
        mic_power: f32,
        echo_power: f32,
        residual_power: f32,
        coherence: f32,
    ) -> bool {
        self.frames = self.frames.saturating_add(1);
        self.track_noise_floor(mic_power);
        if self.frames <= DT_WARMUP_FRAMES {
            self.hangover = 0;
            self.active = false;
            return false;
        }

        let near_end_ratio = if mic_power > DT_MIC_POWER_FLOOR {
            1.0 - echo_power / mic_power
        } else {
            0.0
        };
        // A filter that has not formed yet is not evidence of double talk. The
        // guard only delays the first declaration; it never suppresses one made
        // with a developed echo estimate.
        if !self.formed {
            let cancellation_db =
                10.0 * ((mic_power + 1.0e-20) / (residual_power + 1.0e-20)).log10();
            if mic_power > DT_MIC_POWER_FLOOR && cancellation_db >= DT_MIN_CANCELLATION_DB {
                self.formed_frames = self.formed_frames.saturating_add(1);
                if self.formed_frames >= DT_CONFIRM_FRAMES {
                    self.formed = true;
                }
            } else {
                self.formed_frames = 0;
            }
            if !self.formed {
                self.hangover = 0;
                self.active = false;
                return false;
            }
        }
        // A far-end pause leaves only the background floor: the ratio test
        // would fire there (nothing is explained, the coherence is undefined),
        // so require the microphone to stand clear of that floor first. The
        // gate is only armed when the stream has a real stationary floor; a
        // clean room's quiet frames are decaying tail, not background, and
        // there the old rule is kept untouched.
        let floor_share = if self.mic_mean_ready && self.mic_mean > 0.0 {
            self.quiet_floor / self.mic_mean
        } else {
            1.0
        };
        let noise_present = self.frames > DT_NOISE_PRESENCE_SETTLE_FRAMES
            && self.quiet_floor_ready
            && floor_share > DT_NOISE_PRESENCE_SHARE
            && floor_share < DT_NOISE_PRESENCE_MAX_SHARE;
        let above_floor = !noise_present || mic_power >= DT_NOISE_FLOOR_GATE * self.noise_floor;
        let near_end =
            above_floor && near_end_ratio > DT_NEAR_END_RATIO && coherence < DT_COHERENCE;

        if self.release_left > 0 {
            self.release_left -= 1;
            self.hangover = 0;
            self.frozen_frames = 0;
            self.active = false;
            return false;
        }

        if near_end {
            self.hangover = DT_HANGOVER_FRAMES;
            self.frozen_frames = self.frozen_frames.saturating_add(1);
        } else if self.hangover > 0 {
            self.hangover -= 1;
            self.frozen_frames = 0;
        } else {
            self.frozen_frames = 0;
        }
        self.active = near_end || self.hangover > 0;
        if self.active && self.frozen_frames >= DT_MAX_FREEZE_FRAMES {
            // The stale-estimate deadlock: give the filter a short, guaranteed
            // adaptation window instead of staying frozen forever.
            self.release_left = DT_RELEASE_FRAMES.saturating_sub(1);
            self.hangover = 0;
            self.frozen_frames = 0;
            self.active = false;
        }
        self.active
    }

    pub(crate) fn is_active(&self) -> bool {
        self.active
    }

    /// Tracks the stationary background floor of the microphone power: jump
    /// down immediately to a quieter frame, leak upward slowly otherwise.
    ///
    /// The quietest frames of a speech-like far end are the pauses, and those
    /// contain the background noise plus only the decaying room tail, so the
    /// estimate sits at or just above the true noise power. Jumping *down*
    /// immediately is what lets it find that floor; the slow upward leak is what
    /// keeps it from climbing through speech and from masking a long near-end
    /// burst.
    fn track_noise_floor(&mut self, mic_power: f32) {
        // Digital silence says nothing about the background and would pin the
        // floor to zero forever after a zeroed start-up block.
        if mic_power <= DT_MIC_POWER_FLOOR {
            return;
        }
        if !self.noise_floor_ready {
            self.noise_floor = mic_power;
            self.noise_floor_ready = true;
        } else if mic_power < self.noise_floor {
            self.noise_floor = mic_power;
        } else {
            self.noise_floor += DT_NOISE_FLOOR_RISE * (mic_power - self.noise_floor);
        }

        if !self.quiet_floor_ready {
            self.quiet_floor = mic_power;
            self.quiet_floor_ready = true;
        } else if mic_power < self.quiet_floor {
            self.quiet_floor = mic_power;
        }
        if !self.mic_mean_ready {
            self.mic_mean = mic_power;
            self.mic_mean_ready = true;
        } else {
            self.mic_mean += DT_MIC_MEAN_RISE * (mic_power - self.mic_mean);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warm(detector: &mut DoubleTalkDetector) {
        for _ in 0..DT_WARMUP_FRAMES {
            assert!(!detector.update(1.0, 0.0, 1.0, 0.0));
        }
    }

    fn form(detector: &mut DoubleTalkDetector) {
        for _ in 0..DT_CONFIRM_FRAMES {
            assert!(!detector.update(1.0, 0.8, 0.06, 0.99));
        }
    }

    /// Runs enough ordinary echo-only frames for the stationary-floor
    /// presence test to settle: loud speech frames interleaved with quiet ones,
    /// which puts the quiet floor at 1.0 and the mean well above it.
    fn settle(detector: &mut DoubleTalkDetector) {
        for frame in 0..256 {
            if frame % 4 == 0 {
                assert!(!detector.update(1.0, 0.9, 0.1, 0.99));
            } else {
                assert!(!detector.update(16.0, 14.4, 1.6, 0.99));
            }
        }
    }

    #[test]
    fn cold_start_is_not_double_talk() {
        let mut detector = DoubleTalkDetector::new();
        warm(&mut detector);
        // The estimate is still zero and nothing has been cancelled, which is
        // exactly what the detector used to interpret as near-end speech and
        // freeze adaptation forever.
        assert!(!detector.update(1.0, 0.0, 1.0, 0.0));
        assert!(!detector.update(1.0, 0.01, 0.9, 0.1));
        assert!(!detector.is_active());
    }

    #[test]
    fn a_formed_filter_detects_near_end_speech() {
        let mut detector = DoubleTalkDetector::new();
        warm(&mut detector);
        // 12 dB of measured cancellation for long enough: the filter has formed.
        form(&mut detector);
        settle(&mut detector);
        // Near-end speech: most of the microphone is unexplained, the two
        // signals are not coherent, and the microphone is well above the
        // background floor (which the warm-up frames put at 1.0).
        assert!(detector.update(8.0, 1.6, 6.4, 0.3));
        assert!(detector.is_active());
    }

    #[test]
    fn a_formed_filter_keeps_detecting_a_strong_near_end_signal() {
        let mut detector = DoubleTalkDetector::new();
        warm(&mut detector);
        form(&mut detector);
        settle(&mut detector);
        // +20 dB near-end leaves the filter with almost nothing to explain.
        // The cold-start guard must not re-engage after the filter has formed.
        assert!(detector.update(8.0, 0.08, 7.92, 0.05));
    }

    #[test]
    fn a_stationary_floor_does_not_freeze_adaptation() {
        let mut detector = DoubleTalkDetector::new();
        warm(&mut detector);
        form(&mut detector);
        settle(&mut detector);
        // A far-end pause with only the background floor left: nothing is
        // explained by the echo estimate, so the ratio is one and the
        // coherence is zero. That used to look exactly like near-end speech.
        for _ in 0..200 {
            assert!(
                !detector.update(1.0, 0.0, 1.0, 0.0),
                "a stationary noise floor must not freeze adaptation"
            );
        }
        assert!(!detector.is_active());
    }

    #[test]
    fn a_long_near_end_burst_keeps_the_floor_below_the_burst() {
        let mut detector = DoubleTalkDetector::new();
        warm(&mut detector);
        form(&mut detector);
        settle(&mut detector);
        // A 2.9 s near-end burst at 12 dB above the floor. The floor leaks
        // upward, but far too slowly to mask the burst: the 32/16 freeze cap
        // releases some frames, yet the burst is still detected late in the
        // segment rather than being gated out.
        let mut active = 0;
        for _ in 0..180 {
            if detector.update(16.0, 3.2, 12.8, 0.3) {
                active += 1;
            }
        }
        assert!(
            active > 120,
            "only {active} of 180 burst frames were detected"
        );
        assert!(
            detector.noise_floor < 16.0 / DT_NOISE_FLOOR_GATE,
            "the floor climbed to {} during the burst, into the gate",
            detector.noise_floor
        );
    }

    #[test]
    fn reset_rearms_the_cold_start_guard() {
        let mut detector = DoubleTalkDetector::new();
        warm(&mut detector);
        form(&mut detector);
        detector.reset();
        warm(&mut detector);
        assert!(!detector.update(1.0, 0.0, 1.0, 0.0));
    }
}
