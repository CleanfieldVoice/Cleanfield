//! Room-condition robustness suite for the linear AEC.
//!
//! The suite is a designed matrix, not the full Cartesian product: six room
//! families, four noise conditions (none, white, pink, babble) at three
//! echo-to-noise ratios, three bulk-delay behaviours and three double-talk
//! levels, plus the interactions that are most likely to break a linear
//! canceller. See `room/matrix.rs` for the exact scene list.
//!
//! Every scene is deterministic from a `u32` seed. The report aggregates three
//! seeds; the assertions below run the same three seeds and state what the
//! measured behaviour actually is. Thresholds are properties the stage must
//! have, not values tuned to make the table look good: the known weak cases
//! (diffuse room, clock drift, drift + noise + double talk) are asserted as
//! stability properties and reported as degradations in `ROBUSTNESS.md`.
//!
//! Run with `cargo test -p cf-aec --test room_conditions -- --nocapture` to see
//! the compact table.

#[path = "room/mod.rs"]
mod room;

use std::sync::OnceLock;

use room::{
    mean_min_max, run_scene, scenes, DelayKind, Metrics, NoiseKind, RirKind, SceneSpec, DRIFT_PPM,
    FILTER_NORM_LIMIT, SEEDS,
};
struct SceneRuns {
    spec: SceneSpec,
    runs: Vec<Metrics>,
}

/// Runs the whole matrix once and shares it across the assertions. Each test
/// binary pays for the matrix generation a single time.
fn matrix() -> &'static [SceneRuns] {
    static MATRIX: OnceLock<Vec<SceneRuns>> = OnceLock::new();
    MATRIX.get_or_init(|| {
        scenes()
            .into_iter()
            .map(|spec| {
                let runs = SEEDS
                    .iter()
                    .map(|&seed| run_scene(&spec, seed).metrics)
                    .collect();
                SceneRuns { spec, runs }
            })
            .collect()
    })
}

fn runs_for(rir: RirKind, delay: DelayKind, noise: bool, near: bool) -> Vec<&'static SceneRuns> {
    matrix()
        .iter()
        .filter(|scene| {
            scene.spec.rir == rir
                && scene.spec.delay == delay
                && scene.spec.noise.is_some() == noise
                && scene.spec.near_db.is_some() == near
        })
        .collect()
}

fn values(runs: &[&SceneRuns], metric: impl Fn(&Metrics) -> f32) -> Vec<f32> {
    runs.iter()
        .flat_map(|scene| scene.runs.iter().map(&metric))
        .collect()
}

fn mean(values: &[f32]) -> f32 {
    mean_min_max(values).map_or(f32::NAN, |(mean, _, _)| mean)
}

#[test]
fn every_scene_is_finite_and_bounded() {
    for scene in matrix() {
        for run in &scene.runs {
            assert!(
                !run.has_nan,
                "{}: non-finite residual or filter state",
                scene.spec.name
            );
            assert!(
                run.norm_peak.is_finite() && run.norm_peak <= FILTER_NORM_LIMIT,
                "{}: filter norm peak {} exceeds the divergence bound {FILTER_NORM_LIMIT}",
                scene.spec.name,
                run.norm_peak
            );
            assert!(
                !run.diverged,
                "{}: divergence flag set (echo ERLE pre {} post {})",
                scene.spec.name, run.echo_erle_pre_db, run.echo_erle_post_db
            );
        }
    }
}

#[test]
fn clean_short_office_and_hall_cancel_echo() {
    for (rir, target) in [
        (RirKind::ShortDirect, 20.0f32),
        (RirKind::Office, 20.0),
        (RirKind::Hall, 8.0),
    ] {
        let scenes = runs_for(rir, DelayKind::Static, false, false);
        assert!(!scenes.is_empty());
        for scene in scenes {
            for run in &scene.runs {
                assert!(
                    run.echo_erle_post_db >= target,
                    "{}: clean static echo ERLE post {:.1} dB is below {target:.0} dB",
                    scene.spec.name,
                    run.echo_erle_post_db
                );
            }
        }
    }
}

#[test]
fn diffuse_and_late_direct_rooms_are_not_worse_than_the_microphone() {
    // These two families are the hard cases: the modelled tail (128 ms) is
    // shorter than the room, and the correlation peak can sit on a late
    // reflection. A converged filter is not guaranteed; not making the signal
    // worse than the input is.
    for scene in runs_for(RirKind::LateDirect, DelayKind::Static, false, false) {
        for run in &scene.runs {
            assert!(
                run.echo_erle_post_db > 0.0,
                "{}: echo ERLE post {:.1} dB is worse than the microphone",
                scene.spec.name,
                run.echo_erle_post_db
            );
        }
    }
    // The diffuse room has a high seed variance because the delay estimator may
    // choose the reflection at 33 ms instead of the direct path. The mean must
    // still show cancellation, at least one seed must converge well, and at
    // most one of the three seeds may fall below the input level.
    let diffuse = runs_for(RirKind::DiffuseSparse, DelayKind::Static, false, false);
    let posts = values(&diffuse, |m| m.echo_erle_post_db);
    assert!(
        mean(&posts) >= 8.0,
        "diffuse room mean echo ERLE {:.1} dB is below 8 dB",
        mean(&posts)
    );
    assert!(
        posts.iter().any(|&value| value >= 15.0),
        "diffuse room never reaches 15 dB on any seed: {posts:?}"
    );
    assert!(
        posts.iter().filter(|&&value| value < 0.0).count() <= 1,
        "diffuse room is below the input on more than one seed: {posts:?}"
    );
}

#[test]
fn weak_direct_path_is_recovered_by_the_causal_guard() {
    for scene in runs_for(
        RirKind::WeakDirectStrongEarly,
        DelayKind::Static,
        false,
        false,
    ) {
        for run in &scene.runs {
            assert!(
                run.echo_erle_post_db >= 8.0,
                "{}: weak-direct echo ERLE post {:.1} dB",
                scene.spec.name,
                run.echo_erle_post_db
            );
        }
    }
}

#[test]
fn noise_degrades_in_proportion_to_snr() {
    // At 30 dB the linear filter should still work; at 15 dB it should cancel
    // some echo; at 5 dB the microphone is close to a noise-only signal and the
    // reported ERLE is dominated by the noise floor. These are per-group means
    // because individual 5 dB seeds can be unlucky.
    let noisy = matrix()
        .iter()
        .filter(|scene| scene.spec.noise.is_some())
        .collect::<Vec<_>>();
    let mut by_snr: Vec<(f32, Vec<f32>)> = Vec::new();
    for snr in [30.0f32, 15.0, 5.0] {
        let selected: Vec<&SceneRuns> = noisy
            .iter()
            .copied()
            .filter(|scene| scene.spec.noise.unwrap().1 == snr)
            .collect();
        let posts = values(&selected, |m| m.echo_erle_post_db);
        by_snr.push((snr, posts));
    }
    for (snr, posts) in &by_snr {
        let value = mean(posts);
        let target = match snr {
            30.0 => 5.0,
            15.0 => 0.0,
            _ => -2.0,
        };
        assert!(
            value >= target,
            "noise {snr:.0} dB: mean echo ERLE {value:.1} dB is below {target:.1} dB"
        );
    }
    let high = mean(&by_snr[0].1);
    let low = mean(&by_snr[2].1);
    assert!(
        high - low > 5.0,
        "echo ERLE does not degrade with SNR: 30 dB {high:.1}, 5 dB {low:.1}"
    );
}

#[test]
fn double_talk_preserves_the_near_end_signal() {
    for scene in matrix().iter().filter(|scene| scene.spec.near_db.is_some()) {
        for run in &scene.runs {
            let gain = run
                .near_gain_db
                .expect("a double-talk scene must report a near-end gain");
            assert!(
                (-1.0..=1.0).contains(&gain),
                "{}: near-end gain {gain:+.2} dB is outside +-1 dB",
                scene.spec.name
            );
        }
    }
}

#[test]
fn double_talk_recovers_in_static_rooms() {
    for rir in [
        RirKind::Office,
        RirKind::Hall,
        RirKind::WeakDirectStrongEarly,
    ] {
        for scene in runs_for(rir, DelayKind::Static, false, true) {
            for run in &scene.runs {
                let recovery = run.recovery_ms.unwrap_or_else(|| {
                    panic!(
                        "{}: no echo ERLE recovery after double talk (pre {:.1} dB, post {:.1} dB)",
                        scene.spec.name, run.echo_erle_pre_db, run.echo_erle_post_db
                    )
                });
                assert!(
                    recovery <= 1_500.0,
                    "{}: recovery took {recovery:.0} ms",
                    scene.spec.name
                );
            }
        }
    }
}

#[test]
fn delay_step_is_absorbed() {
    for rir in [RirKind::ShortDirect, RirKind::Office] {
        let scenes = runs_for(rir, DelayKind::Step, false, false);
        assert!(!scenes.is_empty());
        for scene in scenes {
            let posts = scene
                .runs
                .iter()
                .map(|m| m.echo_erle_post_db)
                .collect::<Vec<_>>();
            assert!(
                mean(&posts) >= 10.0,
                "{}: mean echo ERLE after step {:.1} dB",
                scene.spec.name,
                mean(&posts)
            );
            for run in &scene.runs {
                let changed = run.delay_change_ms.unwrap_or_else(|| {
                    panic!("{}: delay step was never reported", scene.spec.name)
                });
                assert!(
                    changed <= 3_000.0,
                    "{}: delay step applied after {changed:.0} ms",
                    scene.spec.name
                );
                assert!(
                    run.echo_erle_post_db >= 5.0,
                    "{}: echo ERLE post after step {:.1} dB",
                    scene.spec.name,
                    run.echo_erle_post_db
                );
            }
        }
    }
}

#[test]
fn drift_does_not_destabilise_the_filter() {
    // Clock drift is the weakest delay behaviour: the filter has to follow a
    // moving path while the bulk alignment stays fixed. Requiring stability
    // (not a particular ERLE) is the honest contract.
    for scene in matrix()
        .iter()
        .filter(|scene| matches!(scene.spec.delay, DelayKind::Drift(_)))
    {
        for run in &scene.runs {
            assert!(
                !run.has_nan && run.norm_peak.is_finite() && run.norm_peak <= FILTER_NORM_LIMIT,
                "{}: drift run left the stable region (norm peak {})",
                scene.spec.name,
                run.norm_peak
            );
            assert!(
                run.erle_post_db.is_finite(),
                "{}: drift run has no finite post ERLE",
                scene.spec.name
            );
        }
    }
}

#[test]
fn scene_generation_is_seed_deterministic() {
    let spec = SceneSpec::new("repro", RirKind::Office)
        .with_noise(NoiseKind::Pink, 15.0)
        .with_delay(DelayKind::Drift(DRIFT_PPM))
        .with_near_end(-6.0);
    let first = room::Scene::generate(spec.clone(), 0xABCD);
    let second = room::Scene::generate(spec, 0xABCD);
    assert_eq!(first.reference, second.reference);
    assert_eq!(first.mic, second.mic);
    assert_eq!(first.echo, second.echo);
    assert_eq!(first.noise, second.noise);
    assert_eq!(first.near, second.near);
}

#[test]
fn aec_replay_is_bit_deterministic() {
    let spec = SceneSpec::new("determinism", RirKind::Office)
        .with_noise(NoiseKind::White, 15.0)
        .with_delay(DelayKind::Step)
        .with_near_end(0.0);
    let first = run_scene(&spec, 0x1234_5678);
    let second = run_scene(&spec, 0x1234_5678);
    assert_eq!(first.residual.len(), second.residual.len());
    for (index, (&a, &b)) in first
        .residual
        .iter()
        .zip(second.residual.iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "sample {index} differs between identical runs"
        );
    }
}

#[test]
fn cpu_per_frame_is_measured_and_inside_the_frame_budget() {
    let spec = SceneSpec::new("cpu", RirKind::Office).with_noise(NoiseKind::White, 15.0);
    let run = run_scene(&spec, 0x00C0_FFEE);
    assert!(
        run.metrics.cpu_ms_per_frame > 0.0,
        "CPU metric was not measured"
    );
    assert!(
        run.metrics.cpu_ms_per_frame < 16.0,
        "CPU per frame {:.2} ms exceeds the 16 ms frame budget",
        run.metrics.cpu_ms_per_frame
    );
}

#[test]
fn prints_compact_matrix_table() {
    let mut rows = 0usize;
    for scene in matrix() {
        let stats = |metric: fn(&Metrics) -> f32| {
            let values = scene.runs.iter().map(metric).collect::<Vec<_>>();
            mean_min_max(&values)
                .map(|(mean, min, max)| format!("{mean:.1} [{min:.1}, {max:.1}]"))
                .unwrap_or_else(|| "-".to_string())
        };
        println!(
            "{:<58} ERLE {:<16} echoERLE {:<16} norm {:<16} ms/frame {}",
            scene.spec.name,
            stats(|m| m.erle_post_db),
            stats(|m| m.echo_erle_post_db),
            stats(|m| m.norm_peak),
            stats(|m| m.cpu_ms_per_frame),
        );
        rows += 1;
    }
    assert_eq!(rows, scenes().len());
}
