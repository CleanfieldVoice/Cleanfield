//! Integration tests for the two-stage echo pipeline.

use cf_aec::AecConfig;
use cf_echo_pipeline::{EchoPipeline, EchoPipelineConfig, FRAME_LEN, LATENCY_SAMPLES};

fn model_bytes() -> Vec<u8> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/echo-default.cfwm"
    );
    std::fs::read(path).expect("models/echo-default.cfwm")
}

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

fn run(config: EchoPipelineConfig, mic: &[f32], reference: &[f32]) -> Vec<f32> {
    let bytes = model_bytes();
    let mut pipeline = EchoPipeline::new(&bytes, config).expect("model");
    let mut out = Vec::with_capacity(mic.len());
    for start in (0..mic.len()).step_by(FRAME_LEN) {
        let end = start + FRAME_LEN;
        out.extend_from_slice(pipeline.process(&mic[start..end], &reference[start..end]));
    }
    out
}

#[test]
fn neural_stage_has_the_declared_latency() {
    let frames = 12;
    let reference = noise(frames * FRAME_LEN, 0x1234_5678);
    let mic = noise(frames * FRAME_LEN, 0x8765_4321);
    let out = run(EchoPipelineConfig::default(), &mic, &reference);
    assert_eq!(out.len(), mic.len());
    // The mask is a spectral filter, so the nominal warm-up region is not
    // bit-zero; it is many orders of magnitude below the signal.
    let early_peak = out[..LATENCY_SAMPLES]
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!(
        early_peak < 1.0e-6,
        "the first {LATENCY_SAMPLES} samples should be warm-up, peak {early_peak:e}"
    );
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn pipeline_is_deterministic() {
    let frames = 8;
    let reference = noise(frames * FRAME_LEN, 0xAAAA_0001);
    let mic = noise(frames * FRAME_LEN, 0x5555_0002);
    let first = run(EchoPipelineConfig::default(), &mic, &reference);
    let second = run(EchoPipelineConfig::default(), &mic, &reference);
    assert_eq!(first, second);
}

#[test]
fn invalid_block_is_a_no_op() {
    let bytes = model_bytes();
    let mut pipeline = EchoPipeline::new(&bytes, EchoPipelineConfig::default()).expect("model");
    assert!(pipeline.process(&[0.0; 100], &[0.0; 100]).is_empty());
    assert!(pipeline
        .process(&[0.0; FRAME_LEN], &[0.0; FRAME_LEN - 1])
        .is_empty());
    assert_eq!(
        pipeline
            .process(&[0.0; 2 * FRAME_LEN], &[0.0; 2 * FRAME_LEN])
            .len(),
        2 * FRAME_LEN
    );
}

#[test]
fn aec_only_stage_removes_the_synthetic_echo() {
    let frames = 128;
    let len = frames * FRAME_LEN;
    let reference = noise(len, 0xC0FF_EE01);
    let mut mic = vec![0.0f32; len];
    let delay = 128usize;
    for n in delay..len {
        mic[n] = 0.6 * reference[n - delay];
    }
    let config = EchoPipelineConfig {
        aec: AecConfig {
            delay_override: Some(delay),
            ..AecConfig::default()
        },
        neural_enabled: false,
        ..EchoPipelineConfig::default()
    };
    let out = run(config, &mic, &reference);
    let mic_power: f64 = mic[len / 2..]
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum();
    let out_power: f64 = out[len / 2..]
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum();
    let erle = 10.0 * (mic_power / out_power.max(1.0e-30)).log10();
    assert!(erle > 20.0, "ERLE was {erle:.1} dB");
}
