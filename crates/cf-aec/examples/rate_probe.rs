//! Diagnostic probe for the clock-rate estimator and the drift scenes.
//!
//! ```text
//! cargo run --release -p cf-aec --example rate_probe
//! ```
//!
//! Prints, per scene, the frame at which the rate correction engages, the rate
//! it applies over time, and the echo ERLE in the pre- and post-event windows,
//! so the convergence lag and the residual ppm can be read off directly. It
//! ends with a CPU measurement that separates the two costs of the correction
//! (the acquisition-phase analysis cadence and the resampler) by switching the
//! correction off and on for the same scene. The assertions live in
//! `tests/drift_split.rs`; this is a development tool.

#[path = "../tests/room/mod.rs"]
mod room;

use std::time::Instant;

use cf_aec::{Aec, AecConfig};
use room::{
    window_echo_erle, DelayKind, Scene, SceneSpec, DRIFT_PPM, FRAME, POST_END_FRAME,
    POST_START_FRAME, PRE_END_FRAME, PRE_START_FRAME, SCENE_FRAMES, SEEDS,
};

fn main() {
    let mut specs: Vec<SceneSpec> = room::scenes()
        .into_iter()
        .filter(|spec| matches!(spec.delay, DelayKind::Drift(_)))
        .collect();
    for spec in room::scenes() {
        let clean = spec.noise.is_none() && spec.near_db.is_none();
        if clean && (spec.delay == DelayKind::Static || spec.delay == DelayKind::Step) {
            specs.push(spec);
        }
    }

    for spec in specs {
        for &seed in &SEEDS {
            println!("\n=== {} seed {:08X} ===", spec.name, seed);
            let scene = Scene::generate(spec.clone(), seed);
            run(&scene);
        }
    }

    timing();
}

fn run(scene: &Scene) {
    let mut aec = Aec::new(AecConfig::default());
    let mut residual = vec![0.0f32; scene.mic.len()];
    let mut engaged = None;
    let mut ppm_at: Vec<(usize, f32)> = Vec::new();
    for frame in 0..SCENE_FRAMES {
        let start = frame * FRAME;
        let end = start + FRAME;
        let output = aec.process(&scene.mic[start..end], &scene.reference[start..end]);
        residual[start..end].copy_from_slice(output);
        if aec.clock_rate_corrected() && engaged.is_none() {
            engaged = Some(frame);
        }
        if frame % 64 == 0 || frame + 1 == SCENE_FRAMES {
            ppm_at.push((frame, aec.clock_rate_ppm()));
        }
    }
    let pre = window_echo_erle(scene, &residual, PRE_START_FRAME, PRE_END_FRAME);
    let post = window_echo_erle(scene, &residual, POST_START_FRAME, POST_END_FRAME);
    println!(
        "engaged at {:?}; pre {:.1} dB post {:.1} dB; norm {:.2}",
        engaged,
        pre,
        post,
        aec.filter_norm()
    );
    let rates: Vec<String> = ppm_at
        .iter()
        .map(|(frame, ppm)| format!("{frame}:{ppm:.0}"))
        .collect();
    println!("ppm {}", rates.join(" "));
}

/// Microseconds per 16 ms frame for the four configurations that separate the
/// cost of the correction into its two parts.
///
/// * a static scene, correction on: the acquisition-phase analysis cadence,
/// * a static scene, correction off: the pre-existing analysis cadence,
/// * a drift scene, correction on: acquisition plus the engaged resampler,
/// * a drift scene, correction off: acquisition only, no resampler.
fn timing() {
    let static_spec = room::SceneSpec::new("t", room::RirKind::Office);
    let drift_spec =
        room::SceneSpec::new("t", room::RirKind::Office).with_delay(DelayKind::Drift(DRIFT_PPM));
    let static_scene = Scene::generate(static_spec, SEEDS[0]);
    let drift_scene = Scene::generate(drift_spec, SEEDS[0]);
    let repeats = 20;

    let measure = |scene: &Scene, correction: bool| -> f64 {
        let mut total = 0.0f64;
        for _ in 0..repeats {
            let mut aec = Aec::new(AecConfig::default());
            aec.set_clock_rate_correction(correction);
            let started = Instant::now();
            let mut sink = 0.0f32;
            for frame in 0..SCENE_FRAMES {
                let start = frame * FRAME;
                let end = start + FRAME;
                let output = aec.process(&scene.mic[start..end], &scene.reference[start..end]);
                sink += output[0];
            }
            std::hint::black_box(sink);
            total += started.elapsed().as_secs_f64() * 1.0e6 / SCENE_FRAMES as f64;
        }
        total / repeats as f64
    };

    let static_off = measure(&static_scene, false);
    let static_on = measure(&static_scene, true);
    let drift_off = measure(&drift_scene, false);
    let drift_on = measure(&drift_scene, true);
    println!("\n=== CPU, microseconds per 16 ms frame (release, mean of {repeats}) ===");
    println!("static, correction off (baseline cadence)   {static_off:8.2}");
    println!("static, correction on  (acquisition)       {static_on:8.2}");
    println!("drift,  correction off (no resampler)      {drift_off:8.2}");
    println!("drift,  correction on  (resampler engaged) {drift_on:8.2}");
    println!(
        "acquisition delta (static on - off)        {:8.2}",
        static_on - static_off
    );
    println!(
        "resampler delta   (drift on - off)         {:8.2}",
        drift_on - drift_off
    );
}
