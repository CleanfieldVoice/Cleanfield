//! Functional validation of the linear AEC.
//!
//! There is no reference binary and no bit-exactness claim for this stage.
//! These tests check the properties the linear front end must have before it
//! can feed the neural model:
//!
//! 1. echo return loss on a synthetic room path (target >= 15 dB);
//! 2. near-end preservation during double talk (target >= -1 dB gain loss);
//! 3. double-talk stability (bounded filter norm, recovery afterwards);
//! 4. bulk-delay recovery (40 ms, within one 16 ms frame) and manual override;
//! 5. convergence time to 10 dB ERLE;
//! 6. determinism and block-size independence.
//!
//! Run with `cargo test -p cf-aec -- --nocapture` to see the printed table.

mod common;

use cf_aec::{Aec, AecConfig, FRAME_LEN, MAX_FRAME_LEN};
use common::{
    convergence_ms, detection_rate, erle_db, near_end_gain_db, near_end_range, run,
    run_double_talk, run_echo_only, snr_db, Scenario, REPORT_SEED,
};

fn seed_of(offset: u32) -> u32 {
    REPORT_SEED ^ offset
}

#[test]
fn echo_only_erle_exceeds_target() {
    let run = run_echo_only(600, 0, seed_of(0x0001));
    let tail = run.erle_db(400, 600);
    let reported = run.aec.erle_db();
    eprintln!("echo-only tail ERLE {tail:.1} dB, accessor {reported:.1} dB");
    assert!(
        tail >= 15.0,
        "echo-only tail ERLE {tail:.1} dB is below the 15 dB target"
    );
    assert!(
        (tail - reported).abs() <= 6.0,
        "accessor {reported:.1} dB disagrees with measured {tail:.1} dB"
    );
    // The synthetic room has unit energy; a converged causal filter is O(1).
    assert!(run.aec.filter_norm().is_finite());
    assert!(run
        .stats
        .iter()
        .all(|stat| stat.instantaneous_erle_db.is_finite()));
}

#[test]
fn converges_to_ten_db_quickly() {
    let run = run_echo_only(600, 0, seed_of(0x0002));
    let reached = convergence_ms(&run.stats, 10.0);
    eprintln!(
        "10 dB ERLE after {reached:.0} ms ({} frames)",
        reached / 16.0
    );
    assert!(!reached.is_nan(), "10 dB ERLE was never reached");
    assert!(
        reached <= 1_000.0,
        "10 dB ERLE took {reached:.0} ms, more than one second"
    );
    assert!(run.erle_db(400, 600) >= 15.0);
}

#[test]
fn near_end_speech_is_preserved() {
    let frames = 700;
    let run = run_double_talk(frames, 300, 520, -6.0, seed_of(0x0003));
    let (start, end) = near_end_range(&run.scenario);
    assert!(end > start, "scenario has no near-end segment");

    let gain = near_end_gain_db(&run.residual, &run.scenario.near, start, end);
    let leak: Vec<f32> = run
        .residual
        .iter()
        .zip(run.scenario.near.iter())
        .map(|(residual, near)| residual - near)
        .collect();
    let snr_before = snr_db(&run.scenario.near, &run.scenario.echo, start, end);
    let snr_after = snr_db(&run.scenario.near, &leak, start, end);
    eprintln!("near-end gain {gain:+.2} dB, SNR {snr_before:+.1} -> {snr_after:+.1} dB");

    assert!(
        (-1.0..=1.0).contains(&gain),
        "near-end gain {gain:+.2} dB is outside +-1 dB"
    );
    assert!(
        snr_after > snr_before,
        "near-end SNR did not improve ({snr_before:+.1} -> {snr_after:+.1} dB)"
    );
    // After the near-end stops, the filter must still cancel the echo.
    let recovered = run.erle_db(620, frames);
    assert!(
        recovered >= 15.0,
        "ERLE 100 frames after double talk is only {recovered:.1} dB"
    );
}

#[test]
fn double_talk_does_not_diverge() {
    let frames = 500;
    let run = run_double_talk(frames, 150, 350, 3.0, seed_of(0x0004));
    let before = run.stats[149].filter_norm;
    let mut peak_norm = 0.0f32;
    let mut peak_residual = 0.0f32;
    for stat in &run.stats[150..350] {
        peak_norm = peak_norm.max(stat.filter_norm);
        peak_residual = peak_residual.max(stat.residual_power);
    }
    let (flagged, active) = detection_rate(&run, 150, 350);
    let after = run.erle_db(420, frames);
    eprintln!(
        "filter norm before {before:.2}, peak during DT {peak_norm:.2}; \
         residual power peak {peak_residual:.4}; DT flagged {flagged}/{active} active frames; \
         ERLE after {after:.1} dB"
    );

    assert!(peak_norm.is_finite() && peak_norm <= before * 2.0 + 0.1);
    assert!(peak_residual.is_finite());
    assert!(
        run.residual.iter().all(|value| value.is_finite()),
        "residual contains a non-finite sample"
    );
    assert!(
        flagged * 10 >= active * 7,
        "double talk was flagged in only {flagged}/{active} active frames"
    );
    assert!(
        after >= 12.0,
        "ERLE 70 frames after strong double talk is only {after:.1} dB"
    );
}

#[test]
fn bulk_delay_is_recovered() {
    let frames = 400;
    let true_delay = 640usize;
    let run = run_echo_only(frames, true_delay, seed_of(0x0005));
    let estimate = run.aec.delay_samples();
    eprintln!("true delay {true_delay} samples, estimate {estimate} samples");
    assert!(
        estimate.abs_diff(true_delay) <= FRAME_LEN,
        "delay estimate {estimate} is more than one frame from {true_delay}"
    );
    let erle = run.erle_db(200, frames);
    eprintln!("ERLE with 40 ms path {erle:.1} dB");
    assert!(erle >= 15.0, "ERLE with a delayed path is {erle:.1} dB");
}

#[test]
fn manual_delay_override_is_honoured() {
    let true_delay = 640usize;
    let scenario = Scenario::echo_only(400, true_delay, 512, seed_of(0x0006));
    let mut aec = Aec::new(AecConfig {
        delay_override: Some(true_delay),
        ..AecConfig::default()
    });
    let (residual, _) = run(&mut aec, &scenario);
    assert_eq!(aec.delay_samples(), true_delay);
    let erle = erle_db(&scenario.mic, &residual, 200 * FRAME_LEN, 400 * FRAME_LEN);
    eprintln!("override ERLE {erle:.1} dB");
    assert!(erle >= 15.0);

    // Switching the override mid-stream must change the reported delay and
    // reset the filter, not leave stale taps aligned to the old delay.
    aec.set_delay_override(Some(0));
    assert_eq!(aec.delay_samples(), 0);
    let frame = [0.0f32; FRAME_LEN];
    let output = aec.process(&frame, &frame);
    assert_eq!(output.len(), FRAME_LEN);
    aec.set_delay_override(None);
    assert_eq!(aec.delay_samples(), 0);
}

#[test]
fn output_is_deterministic() {
    let scenario = Scenario::echo_only(300, 640, 512, seed_of(0x0007));
    let mut first = Aec::new(AecConfig::default());
    let mut second = Aec::new(AecConfig::default());
    let (a, _) = run(&mut first, &scenario);
    let (b, _) = run(&mut second, &scenario);
    assert_eq!(a.len(), b.len());
    for (index, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "sample {index}: 0x{:08x} vs 0x{:08x}",
            x.to_bits(),
            y.to_bits()
        );
    }

    // A reset must reproduce the same stream from the same input.
    first.reset();
    let (again, _) = run(&mut first, &scenario);
    for (index, (&x, &y)) in a.iter().zip(again.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "after reset, sample {index}: 0x{:08x} vs 0x{:08x}",
            x.to_bits(),
            y.to_bits()
        );
    }
}

#[test]
fn block_size_does_not_change_the_result() {
    let scenario = Scenario::echo_only(120, 0, 512, seed_of(0x0008));
    let mut split = Aec::new(AecConfig::default());
    let (split_output, _) = run(&mut split, &scenario);

    let mut whole = Aec::new(AecConfig::default());
    let mut whole_output = vec![0.0f32; scenario.mic.len()];
    for start in (0..scenario.mic.len()).step_by(2 * FRAME_LEN) {
        let end = start + 2 * FRAME_LEN;
        let output = whole.process(&scenario.mic[start..end], &scenario.reference[start..end]);
        assert_eq!(output.len(), 2 * FRAME_LEN);
        whole_output[start..end].copy_from_slice(output);
    }
    for (index, (&x, &y)) in split_output.iter().zip(whole_output.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "sample {index} differs by block size"
        );
    }
}

#[test]
fn invalid_frame_lengths_are_rejected() {
    let mut aec = Aec::new(AecConfig::default());
    let zero = [0.0f32; 4_096];
    assert_eq!(aec.process(&zero[..100], &zero[..100]).len(), 0);
    assert_eq!(aec.process(&zero[..FRAME_LEN], &zero[..128]).len(), 0);
    assert_eq!(aec.process(&[], &[]).len(), 0);
    assert_eq!(
        aec.process(
            &zero[..MAX_FRAME_LEN + FRAME_LEN],
            &zero[..MAX_FRAME_LEN + FRAME_LEN]
        )
        .len(),
        0
    );
    assert_eq!(
        aec.process(&zero[..FRAME_LEN], &zero[..FRAME_LEN]).len(),
        FRAME_LEN
    );
}

#[test]
fn silence_stays_silent() {
    let mut aec = Aec::new(AecConfig::default());
    let silence = [0.0f32; FRAME_LEN];
    for _ in 0..64 {
        let output = aec.process(&silence, &silence);
        assert!(output.iter().all(|&value| value == 0.0));
    }
    assert!(aec.erle_db().is_finite());
}

#[test]
fn config_values_are_clamped() {
    let mut aec = Aec::new(AecConfig {
        step_size: 5.0,
        regularization: -1.0,
        delay_override: Some(usize::MAX),
    });
    assert_eq!(aec.delay_samples(), cf_aec::MAX_DELAY_SAMPLES);
    let frame = [0.1f32; FRAME_LEN];
    let output = aec.process(&frame, &frame);
    assert_eq!(output.len(), FRAME_LEN);
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn reset_clears_the_state() {
    let run = run_echo_only(200, 640, seed_of(0x0009));
    let mut aec = run.aec;
    assert!(aec.delay_samples() > 0, "delay was not estimated");
    aec.reset();
    assert_eq!(aec.delay_samples(), 0);
    assert_eq!(aec.erle_db(), 0.0);
    assert_eq!(aec.filter_norm(), 0.0);
    assert!(!aec.double_talk());
}

#[test]
fn prints_results_table() {
    common::print_report();
}
