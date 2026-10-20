//! Offline driver for the echo pipeline, used to build the demo assets.
//!
//! ```text
//! echo_driver <echo-model.cfwm> <mic.wav> <reference.wav> <out.wav>
//! ```
//!
//! The pipeline's fixed latency is compensated, so `out.wav` is sample aligned
//! with `mic.wav` and has the same length.

mod wav;

use std::path::PathBuf;

use cf_echo_pipeline::{EchoPipeline, EchoPipelineConfig, FRAME_LEN, LATENCY_SAMPLES};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        eprintln!("usage: echo_driver <model.cfwm> <mic.wav> <reference.wav> <out.wav>");
        std::process::exit(2);
    }
    let model_path = PathBuf::from(&args[0]);
    let mic_path = PathBuf::from(&args[1]);
    let reference_path = PathBuf::from(&args[2]);
    let out_path = PathBuf::from(&args[3]);

    let model = std::fs::read(&model_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", model_path.display()));
    let mic = wav::read(&mic_path).expect("mic wav");
    let reference = wav::read(&reference_path).expect("reference wav");
    assert_eq!(mic.sample_rate, 16_000, "mic must be 16 kHz");
    assert_eq!(reference.sample_rate, 16_000, "reference must be 16 kHz");
    assert_eq!(
        mic.samples.len(),
        reference.samples.len(),
        "mic and reference must be the same length and start on the same timeline"
    );

    let frames = mic.samples.len() / FRAME_LEN * FRAME_LEN;
    let mut pipeline = EchoPipeline::new(&model, EchoPipelineConfig::default())
        .expect("the echo container must load");

    // The pipeline lags its input by LATENCY_SAMPLES: pad the input, then drop
    // the warm-up so the result lines up sample for sample with the microphone.
    let mut padded_mic = mic.samples[..frames].to_vec();
    padded_mic.resize(frames + LATENCY_SAMPLES, 0.0);
    let mut padded_reference = reference.samples[..frames].to_vec();
    padded_reference.resize(frames + LATENCY_SAMPLES, 0.0);

    let mut out = Vec::with_capacity(frames + LATENCY_SAMPLES);
    for start in (0..frames + LATENCY_SAMPLES).step_by(FRAME_LEN) {
        out.extend_from_slice(pipeline.process(
            &padded_mic[start..start + FRAME_LEN],
            &padded_reference[start..start + FRAME_LEN],
        ));
    }

    let aligned: Vec<f32> = out[LATENCY_SAMPLES..LATENCY_SAMPLES + frames].to_vec();
    wav::write(&out_path, 16_000, &aligned).expect("write output wav");

    let rms = |x: &[f32]| {
        (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len().max(1) as f64)
            .sqrt()
    };
    println!(
        "frames={} in_rms={:.5} ({:.1} dBFS) out_rms={:.5} ({:.1} dBFS) peak={:.4}",
        frames / FRAME_LEN,
        rms(&mic.samples[..frames]),
        20.0 * rms(&mic.samples[..frames]).max(1e-12).log10(),
        rms(&aligned),
        20.0 * rms(&aligned).max(1e-12).log10(),
        aligned.iter().fold(0.0f32, |a, v| a.max(v.abs())),
    );
}
