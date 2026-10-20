//! Room impulse responses for the robustness matrix.
//!
//! Six response families cover the practical failure modes: a short direct-path
//! room, an office, a long hall, a diffuse tail with sparse early reflections, a
//! late direct path and a weak direct path with strong early reflections.
//!
//! Every response is normalised to **unit L2 energy**, `sum h[n]^2 = 1`. That is
//! the level-normalisation policy for the whole suite: a unit-energy response
//! makes the echo power equal to the far-end power on average (for a
//! white-enough reference), so an ERLE number is not flattered or penalised by
//! the absolute gain of the synthetic room. The direct-to-reverberant ratio is
//! still whatever the response shape implies, and is reported separately.

use super::signal::Rng;

/// Colourless unity gain tap used by the short direct-path room.
const DIRECT: f32 = 1.0;

/// Room family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RirKind {
    /// (a) Short direct-path room, response span about 20 ms.
    ShortDirect,
    /// (b) Typical office, response span about 60 ms.
    Office,
    /// (c) Long reverberant hall, response span about 250 ms.
    Hall,
    /// (d) Diffuse tail with sparse early reflections, span about 150 ms.
    DiffuseSparse,
    /// (e) Late direct path: the energy peak is at 45 ms, beyond the 40 ms
    /// requirement.
    LateDirect,
    /// (f) Weak direct path and strong early reflections, span about 120 ms.
    WeakDirectStrongEarly,
}

impl RirKind {
    pub const ALL: [Self; 6] = [
        Self::ShortDirect,
        Self::Office,
        Self::Hall,
        Self::DiffuseSparse,
        Self::LateDirect,
        Self::WeakDirectStrongEarly,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ShortDirect => "short-direct-20ms",
            Self::Office => "office-60ms",
            Self::Hall => "hall-250ms",
            Self::DiffuseSparse => "diffuse-sparse-150ms",
            Self::LateDirect => "late-direct-45ms",
            Self::WeakDirectStrongEarly => "weak-direct-strong-early",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::ShortDirect => "direct path, four reflections, 24 ms span",
            Self::Office => "direct path, early reflections, 64 ms span",
            Self::Hall => "direct path, sparse early taps, 256 ms diffuse tail",
            Self::DiffuseSparse => "0.65 direct, four sparse reflections, 150 ms diffuse tail",
            Self::LateDirect => "first energy at 45 ms, peak at 45 ms, 200 ms tail",
            Self::WeakDirectStrongEarly => "0.25 direct, reflections at 2x direct, 120 ms span",
        }
    }

    /// Builds the response for `seed`, normalised to unit energy.
    pub fn build(self, seed: u32) -> Vec<f32> {
        let taps = self.taps();
        let mut impulse = vec![0.0f32; taps];
        let mut rng = Rng::new(seed ^ 0x9E37_79B9);

        match self {
            Self::ShortDirect => {
                impulse[0] = DIRECT;
                impulse[36] = 0.45;
                impulse[74] = -0.30;
                impulse[118] = 0.22;
                impulse[180] = -0.15;
                add_diffuse(&mut impulse, 0, &mut rng, 6.0, 0.045);
            }
            Self::Office => {
                impulse[0] = DIRECT;
                for &(offset, gain) in &[
                    (29usize, 0.50f32),
                    (61, -0.40),
                    (103, 0.32),
                    (157, -0.25),
                    (221, 0.20),
                    (310, -0.14),
                    (412, 0.10),
                    (540, -0.08),
                ] {
                    impulse[offset] += gain;
                }
                add_diffuse(&mut impulse, 0, &mut rng, 22.0, 0.060);
            }
            Self::Hall => {
                impulse[0] = DIRECT;
                for &(offset, gain) in &[
                    (53usize, 0.42f32),
                    (131, -0.33),
                    (229, 0.27),
                    (359, -0.22),
                    (521, 0.18),
                    (733, -0.14),
                    (971, 0.11),
                    (1_259, -0.09),
                ] {
                    impulse[offset] += gain;
                }
                add_diffuse(&mut impulse, 0, &mut rng, 70.0, 0.090);
            }
            Self::DiffuseSparse => {
                impulse[0] = 0.65;
                for &(offset, gain) in &[
                    (240usize, 0.55f32),
                    (520, -0.60),
                    (870, 0.50),
                    (1_300, -0.45),
                    (1_700, 0.35),
                ] {
                    impulse[offset] += gain;
                }
                add_diffuse(&mut impulse, 0, &mut rng, 45.0, 0.110);
            }
            Self::LateDirect => {
                // No energy at all before 40 ms: the visible energy peak is the
                // direct arrival at 45 ms (720 samples). The search window sees
                // this as a large bulk delay, which is the point of the scene.
                impulse[720] = DIRECT;
                for &(offset, gain) in &[
                    (790usize, 0.55f32),
                    (860, -0.50),
                    (950, 0.40),
                    (1_080, -0.30),
                    (1_280, 0.25),
                ] {
                    impulse[offset] += gain;
                }
                add_diffuse(&mut impulse, 720, &mut rng, 70.0, 0.100);
            }
            Self::WeakDirectStrongEarly => {
                impulse[0] = 0.25;
                for &(offset, gain) in &[
                    (96usize, -0.90f32),
                    (160, 0.85),
                    (240, -0.75),
                    (352, 0.60),
                    (500, -0.45),
                ] {
                    impulse[offset] += gain;
                }
                add_diffuse(&mut impulse, 0, &mut rng, 30.0, 0.070);
            }
        }

        normalize_energy(&mut impulse);
        impulse
    }

    /// Response length in samples.
    pub fn taps(self) -> usize {
        match self {
            Self::ShortDirect => 384,             // 24 ms
            Self::Office => 1_024,                // 64 ms
            Self::Hall => 4_096,                  // 256 ms
            Self::DiffuseSparse => 2_400,         // 150 ms
            Self::LateDirect => 3_200,            // 200 ms
            Self::WeakDirectStrongEarly => 1_920, // 120 ms
        }
    }
}

/// Adds a seeded, exponentially decaying diffuse tail starting at `start`.
fn add_diffuse(impulse: &mut [f32], start: usize, rng: &mut Rng, tau_ms: f32, level: f32) {
    let tau = tau_ms * 16.0; // milliseconds -> samples at 16 kHz
    for (offset, value) in impulse.iter_mut().enumerate().skip(start) {
        let distance = (offset - start) as f32;
        *value += level * rng.next_f32() * (-distance / tau).exp();
    }
}

fn normalize_energy(impulse: &mut [f32]) {
    let energy = impulse.iter().map(|value| value * value).sum::<f32>();
    let scale = 1.0 / energy.sqrt();
    for value in impulse {
        *value *= scale;
    }
}

/// Measured shape of one response, used in the report's generator table.
#[derive(Clone, Copy, Debug)]
pub struct RirProfile {
    pub taps: usize,
    pub peak_index: usize,
    pub peak_amplitude: f32,
    /// Energy-weighted mean delay, in milliseconds.
    pub centroid_ms: f32,
    /// Share of the total energy that arrives before 40 ms.
    pub energy_before_40ms: f32,
    /// Energy of the single largest tap as a share of the total.
    pub peak_energy_share: f32,
}

pub fn profile(impulse: &[f32]) -> RirProfile {
    let energy = impulse.iter().map(|value| value * value).sum::<f32>();
    let mut peak_index = 0usize;
    let mut peak_amplitude = 0.0f32;
    let mut centroid = 0.0f32;
    let mut early = 0.0f32;
    for (index, &value) in impulse.iter().enumerate() {
        let power = value * value;
        if value.abs() > peak_amplitude {
            peak_amplitude = value.abs();
            peak_index = index;
        }
        centroid += power * index as f32;
        if index < 640 {
            early += power;
        }
    }
    RirProfile {
        taps: impulse.len(),
        peak_index,
        peak_amplitude,
        centroid_ms: centroid / energy / 16.0,
        energy_before_40ms: early / energy,
        peak_energy_share: peak_amplitude * peak_amplitude / energy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_response_has_unit_energy_and_a_finite_profile() {
        for kind in RirKind::ALL {
            for seed in [1u32, 2, 3] {
                let impulse = kind.build(seed);
                let energy = impulse.iter().map(|value| value * value).sum::<f32>();
                assert!((energy - 1.0).abs() < 1.0e-5, "{kind:?} energy {energy}");
                assert!(impulse.iter().all(|value| value.is_finite()));
                let profile = profile(&impulse);
                assert_eq!(profile.taps, kind.taps());
                assert!(profile.centroid_ms > 0.0 && profile.centroid_ms < 300.0);
            }
        }
    }

    #[test]
    fn late_direct_peak_is_beyond_40ms() {
        let impulse = RirKind::LateDirect.build(1);
        let profile = profile(&impulse);
        assert!(profile.peak_index >= 640, "peak at {}", profile.peak_index);
        assert!(profile.energy_before_40ms < 1.0e-3);
    }

    #[test]
    fn weak_direct_is_weaker_than_its_early_reflections() {
        let impulse = RirKind::WeakDirectStrongEarly.build(1);
        let profile = profile(&impulse);
        assert!(profile.peak_index > 0);
        assert!(impulse[0].abs() < profile.peak_amplitude);
    }

    #[test]
    fn hall_has_a_long_diffuse_tail() {
        let impulse = RirKind::Hall.build(1);
        let profile = profile(&impulse);
        // This is a direct-dominated hall: most energy still arrives in the
        // first 40 ms, but the response is 256 ms long and a substantial share
        // arrives after 40 ms. Those are the properties the AEC sees.
        assert!(profile.taps >= 4_000, "taps {}", profile.taps);
        assert!(
            profile.centroid_ms > 10.0,
            "centroid {}",
            profile.centroid_ms
        );
        assert!(
            profile.energy_before_40ms < 0.9,
            "early energy share {}",
            profile.energy_before_40ms
        );
    }
}
