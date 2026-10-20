//! The room-condition scene matrix.
//!
//! The matrix is a designed experiment, not the full Cartesian product. It
//! covers every factor asked for (six room families, four noise conditions at
//! three echo-to-noise ratios, three delay behaviours and three double-talk
//! levels), plus the interactions that are most likely to break a linear
//! canceller: long room with noise, long room with drift, and noise + drift +
//! double talk together.
//!
//! The full Cartesian product would be 6 x 10 x 3 x 4 = 720 scenes and would
//! make the report unreadable. The 53 scenes below keep one row per meaningful
//! condition while still running every seed through the same generated scene
//! definition.

use super::{DelayKind, NoiseKind, RirKind, SceneSpec, DRIFT_PPM};

/// Three independent scene seeds. The report aggregates across all three.
pub const SEEDS: [u32; 3] = [0x00C0_FFEE, 0x5EED_1234, 0xA5A5_5A5A];

/// Builds the scene matrix in a stable order.
pub fn scenes() -> Vec<SceneSpec> {
    let mut out = Vec::new();

    // 1. Clean baselines: one static, noise-free, single-talker scene per room.
    for rir in RirKind::ALL {
        add(&mut out, rir, None, DelayKind::Static, None);
    }

    // 2. Background noise on the office and the hall, every family and SNR.
    for rir in [RirKind::Office, RirKind::Hall] {
        for kind in NoiseKind::ALL {
            for snr_db in [30.0f32, 15.0, 5.0] {
                add(&mut out, rir, Some((kind, snr_db)), DelayKind::Static, None);
            }
        }
    }
    // A short room and a diffuse room at the hardest noise condition, to check
    // that the noise result is not an office/hall artefact.
    add(
        &mut out,
        RirKind::ShortDirect,
        Some((NoiseKind::Babble, 5.0)),
        DelayKind::Static,
        None,
    );
    add(
        &mut out,
        RirKind::DiffuseSparse,
        Some((NoiseKind::Babble, 5.0)),
        DelayKind::Static,
        None,
    );

    // 3. Delay dynamics, one scene per room family, clean.
    for rir in RirKind::ALL {
        add(&mut out, rir, None, DelayKind::Step, None);
    }
    for rir in RirKind::ALL {
        add(&mut out, rir, None, DelayKind::Drift(DRIFT_PPM), None);
    }

    // 4. Double talk on a small, typical and difficult room.
    for rir in [
        RirKind::Office,
        RirKind::Hall,
        RirKind::WeakDirectStrongEarly,
    ] {
        for level_db in [-6.0f32, 0.0, 3.0] {
            add(&mut out, rir, None, DelayKind::Static, Some(level_db));
        }
    }

    // 5. Combined stress: the conditions that interact.
    add(
        &mut out,
        RirKind::Hall,
        Some((NoiseKind::Babble, 5.0)),
        DelayKind::Drift(DRIFT_PPM),
        Some(0.0),
    );
    add(
        &mut out,
        RirKind::Hall,
        Some((NoiseKind::White, 15.0)),
        DelayKind::Step,
        Some(-6.0),
    );
    add(
        &mut out,
        RirKind::DiffuseSparse,
        Some((NoiseKind::Babble, 5.0)),
        DelayKind::Drift(DRIFT_PPM),
        Some(3.0),
    );
    add(
        &mut out,
        RirKind::Office,
        Some((NoiseKind::Pink, 15.0)),
        DelayKind::Drift(DRIFT_PPM),
        Some(0.0),
    );
    add(
        &mut out,
        RirKind::LateDirect,
        Some((NoiseKind::Babble, 15.0)),
        DelayKind::Static,
        Some(-6.0),
    );
    add(
        &mut out,
        RirKind::WeakDirectStrongEarly,
        Some((NoiseKind::White, 5.0)),
        DelayKind::Step,
        Some(3.0),
    );

    out
}

fn add(
    out: &mut Vec<SceneSpec>,
    rir: RirKind,
    noise: Option<(NoiseKind, f32)>,
    delay: DelayKind,
    near_db: Option<f32>,
) {
    let mut spec = SceneSpec::new("", rir).with_delay(delay);
    spec.noise = noise;
    spec.near_db = near_db;
    spec.name = format!("{:02} {}", out.len() + 1, spec.condition_text());
    out.push(spec);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::room::{NoiseKind, RirKind};

    #[test]
    fn matrix_covers_every_requested_factor() {
        let scenes = scenes();
        for rir in RirKind::ALL {
            assert!(scenes.iter().any(|spec| spec.rir == rir));
        }
        for kind in NoiseKind::ALL {
            for snr in [30.0f32, 15.0, 5.0] {
                assert!(scenes.iter().any(|spec| spec.noise == Some((kind, snr))));
            }
        }
        for delay in [
            DelayKind::Static,
            DelayKind::Step,
            DelayKind::Drift(DRIFT_PPM),
        ] {
            assert!(scenes.iter().any(|spec| spec.delay == delay));
        }
        for level in [-6.0f32, 0.0, 3.0] {
            assert!(scenes.iter().any(|spec| spec.near_db == Some(level)));
        }
    }

    #[test]
    fn scene_names_are_unique() {
        let scenes = scenes();
        let mut names: Vec<&str> = scenes.iter().map(|spec| spec.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), scenes.len());
    }
}
