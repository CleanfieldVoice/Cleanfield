//! Prints the room-condition matrix used by `ROBUSTNESS.md`.
//!
//! ```text
//! cargo run --release -p cf-aec --example room_report
//! ```
//!
//! The numbers are produced by exactly the same generator and measurement code
//! as `tests/room_conditions.rs`; the report and the assertions cannot drift
//! apart. Only `std` is used and every scene is seeded, so two runs on the same
//! machine produce identical text.

#[path = "../tests/room/mod.rs"]
mod room;

use room::{
    mean_min_max, rir_profile, run_scene, scenes, DelayKind, Metrics, NoiseKind, RirKind,
    SceneSpec, SEEDS,
};

fn main() {
    print_generator();
    let matrix = scenes();
    let mut summaries = Vec::with_capacity(matrix.len());
    for spec in &matrix {
        let runs: Vec<Metrics> = SEEDS
            .iter()
            .map(|&seed| run_scene(spec, seed).metrics)
            .collect();
        summaries.push((spec.clone(), runs));
        eprintln!("measured {}", spec.name);
    }

    println!("\n### Per-scene results\n");
    println!(
        "| # | condition | ERLE pre (dB) | ERLE post (dB) | Echo ERLE pre (dB) | Echo ERLE post (dB) | Recovery (ms) | Near gain (dB) | DT flag (%) | Norm peak | CPU (ms/frame) | Delay (applied/bulk, smp) | Step (ms) | Flags |"
    );
    println!("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
    for (spec, runs) in &summaries {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            scene_number(spec),
            spec.condition_text(),
            stat(&collect(runs, |m| m.erle_pre_db)),
            stat(&collect(runs, |m| m.erle_post_db)),
            stat(&collect(runs, |m| m.echo_erle_pre_db)),
            stat(&collect(runs, |m| m.echo_erle_post_db)),
            recovery_stat(runs),
            gain_stat(runs),
            detection_stat(runs),
            stat(&collect(runs, |m| m.norm_peak)),
            stat(&collect(runs, |m| m.cpu_ms_per_frame)),
            delay_stat(runs),
            step_stat(runs),
            flags(runs),
        );
    }

    println!("\n### Aggregate by room family\n");
    println!("| Room family | Scenes | ERLE pre | ERLE post | Echo ERLE pre | Echo ERLE post | Norm peak |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
    for rir in RirKind::ALL {
        let selected: Vec<&Metrics> = summaries
            .iter()
            .filter(|(spec, _)| spec.rir == rir)
            .flat_map(|(_, runs)| runs.iter())
            .collect();
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            rir.label(),
            selected.len(),
            stat(&collect_refs(&selected, |m| m.erle_pre_db)),
            stat(&collect_refs(&selected, |m| m.erle_post_db)),
            stat(&collect_refs(&selected, |m| m.echo_erle_pre_db)),
            stat(&collect_refs(&selected, |m| m.echo_erle_post_db)),
            stat(&collect_refs(&selected, |m| m.norm_peak)),
        );
    }

    println!("\n### Aggregate by background noise\n");
    println!(
        "| Noise | Scenes | ERLE pre | ERLE post | Echo ERLE pre | Echo ERLE post | Norm peak |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
    let noise_groups = noise_groups();
    for (label, filter) in noise_groups {
        let selected: Vec<&Metrics> = summaries
            .iter()
            .filter(|(spec, _)| filter.matches(spec))
            .flat_map(|(_, runs)| runs.iter())
            .collect();
        if selected.is_empty() {
            continue;
        }
        println!(
            "| {label} | {} | {} | {} | {} | {} | {} |",
            selected.len(),
            stat(&collect_refs(&selected, |m| m.erle_pre_db)),
            stat(&collect_refs(&selected, |m| m.erle_post_db)),
            stat(&collect_refs(&selected, |m| m.echo_erle_pre_db)),
            stat(&collect_refs(&selected, |m| m.echo_erle_post_db)),
            stat(&collect_refs(&selected, |m| m.norm_peak)),
        );
    }

    println!("\n### Aggregate by delay behaviour\n");
    println!(
        "| Delay | Scenes | ERLE pre | ERLE post | Echo ERLE pre | Echo ERLE post | Norm peak |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
    for delay in [
        DelayKind::Static,
        DelayKind::Step,
        DelayKind::Drift(room::DRIFT_PPM),
    ] {
        let selected: Vec<&Metrics> = summaries
            .iter()
            .filter(|(spec, _)| spec.delay == delay)
            .flat_map(|(_, runs)| runs.iter())
            .collect();
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            delay.label(),
            selected.len(),
            stat(&collect_refs(&selected, |m| m.erle_pre_db)),
            stat(&collect_refs(&selected, |m| m.erle_post_db)),
            stat(&collect_refs(&selected, |m| m.echo_erle_pre_db)),
            stat(&collect_refs(&selected, |m| m.echo_erle_post_db)),
            stat(&collect_refs(&selected, |m| m.norm_peak)),
        );
    }

    println!("\n### Aggregate by double talk\n");
    println!("| Near-end | Scenes | ERLE pre | ERLE post | Echo ERLE pre | Echo ERLE post | Recovery (ms) | Near gain (dB) | DT flag (%) | Norm peak |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for (label, filter) in double_talk_groups() {
        let selected: Vec<&Metrics> = summaries
            .iter()
            .filter(|(spec, _)| filter.matches(spec))
            .flat_map(|(_, runs)| runs.iter())
            .collect();
        println!(
            "| {label} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            selected.len(),
            stat(&collect_refs(&selected, |m| m.erle_pre_db)),
            stat(&collect_refs(&selected, |m| m.erle_post_db)),
            stat(&collect_refs(&selected, |m| m.echo_erle_pre_db)),
            stat(&collect_refs(&selected, |m| m.echo_erle_post_db)),
            recovery_stat_refs(&selected),
            gain_stat_refs(&selected),
            detection_stat_refs(&selected),
            stat(&collect_refs(&selected, |m| m.norm_peak)),
        );
    }

    println!("\n### Flags\n");
    println!("| Flag | Count |");
    println!("| --- | ---: |");
    let all: Vec<&Metrics> = summaries.iter().flat_map(|(_, runs)| runs.iter()).collect();
    println!("| Runs | {} |", all.len());
    println!(
        "| Non-finite output | {} |",
        all.iter().filter(|m| m.has_nan).count()
    );
    println!(
        "| Diverged (NaN, norm > 10, echo ERLE < -6 dB) | {} |",
        all.iter().filter(|m| m.diverged).count()
    );
    println!(
        "| ERLE collapse (echo ERLE < 3 dB, post window) | {} |",
        all.iter().filter(|m| m.collapsed).count()
    );
    println!(
        "| Recovery not reached inside the window | {} |",
        all.iter()
            .filter(|m| m.has_double_talk && m.recovery_ms.is_none())
            .count()
    );
    println!(
        "\nPer-scene cells are mean over {} seeds with `[min, max]`. CPU is wall-clock time for the AEC call only, measured on the report machine; it is reported, not asserted.",
        SEEDS.len()
    );
}

fn print_generator() {
    println!("## Generator parameters\n");
    println!(
        "Scene length {} frames = {:.2} s; frame {} samples; seeds {:08X}, {:08X}, {:08X}.",
        room::SCENE_FRAMES,
        room::SCENE_FRAMES as f32 * 16.0 / 1000.0,
        room::FRAME,
        SEEDS[0],
        SEEDS[1],
        SEEDS[2]
    );
    println!(
        "Static bulk delay {} samples (30 ms), step +{} samples (+5 ms) at frame {}, drift {} ppm over the file; near-end segment frames {}..{}.",
        room::STATIC_DELAY,
        room::STEP_DELTA,
        room::STEP_FRAME,
        room::DRIFT_PPM,
        room::DT_ONSET_FRAME,
        room::DT_END_FRAME
    );
    println!(
        "All responses are normalised to unit L2 energy. Noise RMS is set against the echo RMS; near-end RMS is set against the echo RMS in the double-talk segment.\n"
    );
    println!("| Room | Taps | Span (ms) | Peak index | Peak (ms) | Centroid (ms) | Energy < 40 ms | Peak energy share |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for rir in RirKind::ALL {
        let impulse = rir.build(0x00C0_FFEE ^ 0x1234_5678);
        let profile = rir_profile(&impulse);
        println!(
            "| {} | {} | {:.1} | {} | {:.1} | {:.1} | {:.1} % | {:.1} % |",
            rir.label(),
            profile.taps,
            profile.taps as f32 / 16.0,
            profile.peak_index,
            profile.peak_index as f32 / 16.0,
            profile.centroid_ms,
            100.0 * profile.energy_before_40ms,
            100.0 * profile.peak_energy_share,
        );
    }
}

fn scene_number(spec: &SceneSpec) -> usize {
    spec.name
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn collect(runs: &[Metrics], metric: impl Fn(&Metrics) -> f32) -> Vec<f32> {
    runs.iter().map(metric).collect()
}

fn collect_refs(runs: &[&Metrics], metric: impl Fn(&Metrics) -> f32) -> Vec<f32> {
    runs.iter().map(|metrics| metric(metrics)).collect()
}

fn stat(values: &[f32]) -> String {
    match mean_min_max(values) {
        Some((mean, min, max)) => format!("{mean:.1} [{min:.1}, {max:.1}]"),
        None => "-".to_string(),
    }
}

fn recovery_stat(runs: &[Metrics]) -> String {
    let values: Vec<f32> = runs.iter().filter_map(|m| m.recovery_ms).collect();
    if values.is_empty() {
        if runs.iter().any(|m| m.has_double_talk) {
            "no recovery".to_string()
        } else {
            "-".to_string()
        }
    } else {
        stat(&values)
    }
}

fn recovery_stat_refs(runs: &[&Metrics]) -> String {
    let values: Vec<f32> = runs.iter().filter_map(|m| m.recovery_ms).collect();
    if values.is_empty() {
        if runs.iter().any(|m| m.has_double_talk) {
            "no recovery".to_string()
        } else {
            "-".to_string()
        }
    } else {
        stat(&values)
    }
}

fn gain_stat(runs: &[Metrics]) -> String {
    let values: Vec<f32> = runs.iter().filter_map(|m| m.near_gain_db).collect();
    if values.is_empty() {
        "-".to_string()
    } else {
        stat(&values)
    }
}

fn gain_stat_refs(runs: &[&Metrics]) -> String {
    let values: Vec<f32> = runs.iter().filter_map(|m| m.near_gain_db).collect();
    if values.is_empty() {
        "-".to_string()
    } else {
        stat(&values)
    }
}

fn detection_stat(runs: &[Metrics]) -> String {
    let values: Vec<f32> = runs.iter().filter_map(|m| m.dt_flag_rate).collect();
    if values.is_empty() {
        "-".to_string()
    } else {
        stat(&values)
    }
}

fn detection_stat_refs(runs: &[&Metrics]) -> String {
    let values: Vec<f32> = runs.iter().filter_map(|m| m.dt_flag_rate).collect();
    if values.is_empty() {
        "-".to_string()
    } else {
        stat(&values)
    }
}

fn delay_stat(runs: &[Metrics]) -> String {
    if runs.is_empty() {
        return "-".to_string();
    }
    let applied = runs.iter().map(|m| m.delay_final as f32).sum::<f32>() / runs.len() as f32;
    let target = runs.iter().map(|m| m.delay_target_final).sum::<f32>() / runs.len() as f32;
    format!("{applied:.0} / {target:.0}")
}

fn step_stat(runs: &[Metrics]) -> String {
    let values: Vec<f32> = runs.iter().filter_map(|m| m.delay_change_ms).collect();
    if values.is_empty() {
        "-".to_string()
    } else {
        stat(&values)
    }
}

fn flags(runs: &[Metrics]) -> String {
    let mut flags = Vec::new();
    if runs.iter().any(|m| m.has_nan) {
        flags.push("NaN");
    }
    if runs.iter().any(|m| m.diverged) {
        flags.push("DIVERGED");
    }
    if runs.iter().any(|m| m.collapsed) {
        flags.push("collapse");
    }
    if runs
        .iter()
        .any(|m| m.has_double_talk && m.recovery_ms.is_none())
    {
        flags.push("no-recovery");
    }
    if flags.is_empty() {
        "ok".to_string()
    } else {
        flags.join(" ")
    }
}

struct NoiseFilter {
    noise: Option<(NoiseKind, f32)>,
}

impl NoiseFilter {
    fn matches(&self, spec: &SceneSpec) -> bool {
        spec.noise == self.noise
    }
}

struct NearFilter {
    level: Option<f32>,
}

impl NearFilter {
    fn matches(&self, spec: &SceneSpec) -> bool {
        spec.near_db == self.level
    }
}

fn noise_groups() -> Vec<(String, NoiseFilter)> {
    let mut groups = vec![("none".to_string(), NoiseFilter { noise: None })];
    for kind in NoiseKind::ALL {
        for snr in [30.0f32, 15.0, 5.0] {
            groups.push((
                format!("{} {snr:.0} dB", kind.label()),
                NoiseFilter {
                    noise: Some((kind, snr)),
                },
            ));
        }
    }
    groups
}

fn double_talk_groups() -> Vec<(String, NearFilter)> {
    let mut groups = vec![("none".to_string(), NearFilter { level: None })];
    for level in [-6.0f32, 0.0, 3.0] {
        groups.push((format!("{level:+.0} dB"), NearFilter { level: Some(level) }));
    }
    groups
}
