//! Validation and demo-asset driver for the end-to-end echo pipeline.
//!
//! Run from the workspace root:
//!
//! ```text
//! cargo run --release -p cf-echo-pipeline --example echo_demo -- all
//! ```
//!
//! Subcommands: `synthetic`, `real`, `assets`, `all`. Every reported number is
//! measured from the `cf-aec` + `cf-echo` path in this workspace.

mod voice;
mod wav;

use std::fs;
use std::path::{Path, PathBuf};

use cf_echo_pipeline::{EchoPipeline, EchoPipelineConfig, FRAME_LEN, LATENCY_SAMPLES};
use voice::{VoiceDenoiser, VOICE_LATENCY_SAMPLES};

const SR: usize = 16_000;
const DEMO_START_S: f64 = 2.0;
const DEMO_END_S: f64 = 10.0;

fn main() {
    let root = workspace_root();
    let command = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());

    let model_path = root.join("models/echo-default.cfwm");
    let model =
        fs::read(&model_path).unwrap_or_else(|e| panic!("read {}: {e}", model_path.display()));

    match command.as_str() {
        "synthetic" => synthetic(&model),
        "real" => real(&model, &root, false),
        "assets" => real(&model, &root, true),
        "all" => {
            synthetic(&model);
            real(&model, &root, true);
        }
        other => {
            eprintln!("unknown subcommand {other:?}; use synthetic|real|assets|all");
            std::process::exit(2);
        }
    }
}

fn workspace_root() -> PathBuf {
    if Path::new("models/echo-default.cfwm").exists() {
        PathBuf::from(".")
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }
}

fn samples_dir(root: &Path) -> PathBuf {
    root.join("../local-denoiser/samples")
}

fn artifacts_dir(root: &Path) -> PathBuf {
    root.join("artifacts/echo-pipeline")
}

fn assets_dir(root: &Path) -> PathBuf {
    root.join("../cleanfield-repo/assets/audio")
}

struct Run {
    out: Vec<f32>,
    aec_erle_db: f32,
    aec_delay: usize,
    double_talk_frames: usize,
    /// `(second, min, mean, max)` of the learned mask, sampled every ~0.5 s.
    mask_stats: Vec<(f64, f32, f32, f32)>,
}

fn run_pipeline(model: &[u8], mic: &[f32], reference: &[f32], config: EchoPipelineConfig) -> Run {
    let mut pipeline = EchoPipeline::new(model, config).expect("the echo container must load");
    let mut out = Vec::with_capacity(mic.len());
    let mut double_talk_frames = 0;
    let mut mask_stats = Vec::new();
    for start in (0..mic.len()).step_by(FRAME_LEN) {
        let end = start + FRAME_LEN;
        out.extend_from_slice(pipeline.process(&mic[start..end], &reference[start..end]));
        if pipeline.aec_double_talk() {
            double_talk_frames += 1;
        }
        if config.neural_enabled && start % (FRAME_LEN * 31) == 0 {
            let mask = pipeline.last_mask();
            let min = mask.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = mask.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mean = mask.iter().sum::<f32>() / mask.len() as f32;
            mask_stats.push((start as f64 / SR as f64, min, mean, max));
        }
    }
    Run {
        out,
        aec_erle_db: pipeline.aec_erle_db(),
        aec_delay: pipeline.aec_delay_samples(),
        double_talk_frames,
        mask_stats,
    }
}

fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64).sqrt()
}

fn dbfs(value: f64) -> f64 {
    20.0 * value.max(1.0e-12).log10()
}

fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
}

fn all_finite(x: &[f32]) -> bool {
    x.iter().all(|v| v.is_finite())
}

fn erle_db(mic: &[f32], out: &[f32], out_delay: usize, a: usize, b: usize) -> f64 {
    let mut mic_power = 0.0f64;
    let mut out_power = 0.0f64;
    for n in a..b {
        mic_power += (mic[n] as f64) * (mic[n] as f64);
        out_power += (out[n + out_delay] as f64) * (out[n + out_delay] as f64);
    }
    10.0 * (mic_power / out_power.max(1.0e-30)).log10()
}

fn projection_db(out: &[f32], out_delay: usize, near: &[f32], a: usize, b: usize) -> f64 {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for n in a..b {
        numerator += (out[n + out_delay] as f64) * (near[n] as f64);
        denominator += (near[n] as f64) * (near[n] as f64);
    }
    20.0 * (numerator / denominator.max(1.0e-30))
        .abs()
        .max(1.0e-12)
        .log10()
}

fn synthetic(model: &[u8]) {
    let root = workspace_root();
    println!("\n=== synthetic scene ===");
    let speech = wav::read(&samples_dir(&root).join("speech-clean.wav")).expect("speech-clean.wav");
    assert_eq!(speech.sample_rate as usize, SR);
    let len = 12 * SR;
    let reference: Vec<f32> = speech.samples[..len].to_vec();
    let near_start = (11.5 * SR as f64) as usize;
    let near_source: Vec<f32> = speech.samples[near_start..near_start + len].to_vec();

    let rir = make_rir(512, 160, 0x5EED_1234);
    let echo = convolve(&reference, &rir, len);
    let gate = build_gate(len, &[(4.0, 7.0), (8.0, 11.0)]);
    let near = gated(&near_source, &gate);
    let echo_level = rms(&echo);
    let near_level = rms(&near);
    println!(
        "reference speech {:.2} s, RIR 512 taps (direct at 160), echo RMS {:.5} ({:.1} dBFS)",
        len as f64 / SR as f64,
        echo_level,
        dbfs(echo_level)
    );

    let configs = [
        (
            "AEC only",
            EchoPipelineConfig {
                neural_enabled: false,
                ..EchoPipelineConfig::default()
            },
        ),
        ("AEC + model", EchoPipelineConfig::default()),
    ];

    for relative_db in [-6.0f64, 3.0] {
        let gain = echo_level / near_level * 10.0f64.powf(relative_db / 20.0);
        let near_scaled: Vec<f32> = near.iter().map(|v| v * gain as f32).collect();
        let mic: Vec<f32> = echo
            .iter()
            .zip(near_scaled.iter())
            .map(|(e, n)| e + n)
            .collect();
        let input_peak = peak(&mic);
        let scale = if input_peak > 0.9 {
            0.9 / input_peak
        } else {
            1.0
        };
        let mic: Vec<f32> = mic.iter().map(|v| v * scale).collect();
        let reference_scaled: Vec<f32> = reference.iter().map(|v| v * scale).collect();
        let near_scaled: Vec<f32> = near_scaled.iter().map(|v| v * scale).collect();
        let near_active = rms(&near_scaled);
        println!(
            "\n-- near-end {relative_db:+.1} dB re echo (near RMS {:.5}, input peak {:.4} after x{scale:.3} normalisation)",
            near_active,
            peak(&mic)
        );

        for (label, config) in configs {
            let delay = if config.neural_enabled {
                LATENCY_SAMPLES
            } else {
                0
            };
            let run = run_pipeline(model, &mic, &reference_scaled, config);
            let early = erle_db(
                &mic,
                &run.out,
                delay,
                (1.0 * SR as f64) as usize,
                (3.75 * SR as f64) as usize,
            );
            let late = erle_db(
                &mic,
                &run.out,
                delay,
                (11.0 * SR as f64) as usize,
                (11.8 * SR as f64) as usize,
            );
            let near_a = projection_db(
                &run.out,
                delay,
                &near_scaled,
                (4.5 * SR as f64) as usize,
                (6.5 * SR as f64) as usize,
            );
            let near_b = projection_db(
                &run.out,
                delay,
                &near_scaled,
                (8.5 * SR as f64) as usize,
                (10.5 * SR as f64) as usize,
            );
            println!(
                "{label:<12} early echo-only ERLE {early:6.2} dB | post-double-talk {late:6.2} dB | near-end gain {near_a:+.2} / {near_b:+.2} dB | output peak {:.4} over-range {} finite {} | AEC ERLE {:.1} dB @ {} samples",
                peak(&run.out),
                run.out.iter().any(|v| v.abs() > 1.0),
                all_finite(&run.out),
                run.aec_erle_db,
                run.aec_delay,
            );
            for (second, min, mean, max) in &run.mask_stats {
                if (2.5 - second).abs() < 0.3 {
                    println!(
                        "  echo-only mask t={second:.2}s min {min:.3e} mean {mean:.3e} max {max:.3e}"
                    );
                }
            }
        }
    }
}

fn real(model: &[u8], root: &Path, write_assets: bool) {
    println!("\n=== real material ===");
    let samples = samples_dir(root);
    let mic_wav = wav::read(&samples.join("echo-microphone.wav")).expect("echo-microphone.wav");
    let ref_wav =
        wav::read(&samples.join("echo-playback-speech.wav")).expect("echo-playback-speech.wav");
    let near_wav = wav::read(&samples.join("echo-near-clean.wav")).expect("echo-near-clean.wav");
    assert_eq!(mic_wav.sample_rate as usize, SR);

    let offset = detect_reference_offset(&mic_wav.samples, &ref_wav.samples);
    println!(
        "measured reference offset {offset} samples ({:.3} s); advancing the reference to the microphone timeline",
        offset as f64 / SR as f64
    );

    let frames =
        (mic_wav.samples.len().min(ref_wav.samples.len() + offset)) / FRAME_LEN * FRAME_LEN;
    let mic: Vec<f32> = mic_wav.samples[..frames].to_vec();
    let reference = align_reference(&ref_wav.samples, offset, frames);

    let aec_config = EchoPipelineConfig {
        neural_enabled: false,
        ..EchoPipelineConfig::default()
    };
    let aec_only = run_pipeline(model, &mic, &reference, aec_config);
    let full = run_pipeline(model, &mic, &reference, EchoPipelineConfig::default());

    let a = (DEMO_START_S * SR as f64) as usize;
    let b = (DEMO_END_S * SR as f64) as usize;
    println!(
        "processed {:.2} s; AEC-only ERLE {:.1} dB, applied delay {} samples, double-talk frames {}",
        frames as f64 / SR as f64,
        aec_only.aec_erle_db,
        full.aec_delay,
        full.double_talk_frames
    );
    println!(
        "demo window 2.0-10.0 s: mic RMS {:.5} ({:.1} dBFS) peak {:.4} | AEC-only RMS {:.5} ({:.1} dBFS) peak {:.4} | AEC+model RMS {:.5} ({:.1} dBFS) peak {:.4}",
        rms(&mic[a..b]),
        dbfs(rms(&mic[a..b])),
        peak(&mic[a..b]),
        rms(&aec_only.out[a..b]),
        dbfs(rms(&aec_only.out[a..b])),
        peak(&aec_only.out[a..b]),
        rms(&full.out[a + LATENCY_SAMPLES..b + LATENCY_SAMPLES]),
        dbfs(rms(&full.out[a + LATENCY_SAMPLES..b + LATENCY_SAMPLES])),
        peak(&full.out[a + LATENCY_SAMPLES..b + LATENCY_SAMPLES]),
    );
    let echo_only_a = (2.0 * SR as f64) as usize;
    let echo_only_b = (8.0 * SR as f64) as usize;
    println!(
        "echo-only 2.0-8.0 s ERLE: AEC only {:.2} dB, AEC+model {:.2} dB",
        erle_db(&mic, &aec_only.out, 0, echo_only_a, echo_only_b),
        erle_db(&mic, &full.out, LATENCY_SAMPLES, echo_only_a, echo_only_b),
    );
    println!(
        "full-file ERLE: AEC only {:.2} dB, AEC+model {:.2} dB",
        erle_db(&mic, &aec_only.out, 0, 0, frames),
        erle_db(
            &mic,
            &full.out,
            LATENCY_SAMPLES,
            0,
            frames - LATENCY_SAMPLES
        ),
    );
    for (second, min, mean, max) in &full.mask_stats {
        if [3.0f64, 5.0, 9.0].iter().any(|t| (*t - second).abs() < 0.2) {
            println!("  learned mask t={second:5.2}s min {min:.3e} mean {mean:.3e} max {max:.3e}");
        }
    }
    let near_aec = projection_db(&aec_only.out, 0, &near_wav.samples, 12 * SR, 30 * SR);
    let near_full = projection_db(
        &full.out,
        LATENCY_SAMPLES,
        &near_wav.samples,
        12 * SR,
        30 * SR,
    );
    println!(
        "near-end projection 12.0-30.0 s: AEC only {near_aec:+.2} dB, AEC+model {near_full:+.2} dB"
    );

    let runs = playback_only_runs(&mic, &reference, &near_wav.samples, frames);
    println!("playback-only stretches (near-end silent, playback active):");
    for (start, end) in &runs {
        println!(
            "  {:.2}-{:.2} s",
            *start as f64 / SR as f64,
            *end as f64 / SR as f64
        );
    }
    let mut mic_power = 0.0f64;
    let mut aec_power = 0.0f64;
    let mut full_power = 0.0f64;
    for (start, end) in &runs {
        for n in *start..*end {
            mic_power += (mic[n] as f64) * (mic[n] as f64);
            aec_power += (aec_only.out[n] as f64) * (aec_only.out[n] as f64);
            full_power +=
                (full.out[n + LATENCY_SAMPLES] as f64) * (full.out[n + LATENCY_SAMPLES] as f64);
        }
    }
    println!(
        "  total {:.2} s | ERLE AEC only {:.2} dB, AEC+model {:.2} dB",
        runs.iter()
            .map(|(a, b)| (b - a) as f64 / SR as f64)
            .sum::<f64>(),
        10.0 * (mic_power / aec_power.max(1.0e-30)).log10(),
        10.0 * (mic_power / full_power.max(1.0e-30)).log10(),
    );

    let out_dir = artifacts_dir(root);
    let after_window: Vec<f32> = full.out[a + LATENCY_SAMPLES..b + LATENCY_SAMPLES].to_vec();
    wav::write(
        &out_dir.join("real-echo-after-8s.wav"),
        SR as u32,
        &after_window,
    )
    .expect("write processed window");
    let mut aligned_full: Vec<f32> = full.out[LATENCY_SAMPLES..].to_vec();
    aligned_full.resize(frames, 0.0);
    wav::write(
        &out_dir.join("real-echo-after-full.wav"),
        SR as u32,
        &aligned_full,
    )
    .expect("write processed full file");

    if write_assets {
        write_demo_assets(root, &mic, &full.out, a, b);
    }
}

fn write_demo_assets(root: &Path, mic: &[f32], echo_output: &[f32], a: usize, b: usize) {
    let assets = assets_dir(root);
    let voice_model =
        fs::read(root.join("models/voice-default.cfwm")).expect("read models/voice-default.cfwm");
    let mut denoiser = VoiceDenoiser::new(&voice_model).expect("voice model");
    let denoised = denoiser.process(echo_output);

    let echo_before = &mic[a..b];
    let echo_after = &echo_output[a + LATENCY_SAMPLES..b + LATENCY_SAMPLES];
    let both_after_start = a + LATENCY_SAMPLES + VOICE_LATENCY_SAMPLES;
    let both_after_end = b + LATENCY_SAMPLES + VOICE_LATENCY_SAMPLES;
    let both_after = &denoised[both_after_start..both_after_end];

    let clips = [
        ("echo-before.wav", echo_before),
        ("echo-after.wav", echo_after),
        ("both-before.wav", echo_before),
        ("both-after.wav", both_after),
    ];
    println!("\n=== demo assets (8.0 s, 16 kHz mono PCM16) ===");
    for (name, clip) in clips {
        wav::write(&assets.join(name), SR as u32, clip).expect("write demo clip");
        println!(
            "{name:<18} RMS {:.5} ({:6.1} dBFS)  peak {:.4}  finite {}  over-range {}",
            rms(clip),
            dbfs(rms(clip)),
            peak(clip),
            all_finite(clip),
            clip.iter().any(|v| v.abs() > 1.0),
        );
    }
    println!("assets written to {}", assets.display());
}

fn make_rir(taps: usize, direct: usize, seed: u32) -> Vec<f32> {
    let mut h = vec![0.0f32; taps];
    h[direct] = 1.0;
    for &(delay, gain) in &[
        (23usize, 0.55f32),
        (41, -0.42),
        (77, 0.33),
        (113, -0.27),
        (157, 0.21),
        (211, -0.16),
    ] {
        if direct + delay < taps {
            h[direct + delay] += gain;
        }
    }
    let mut state = seed | 1;
    for n in (direct + 64)..taps {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let noise = ((state >> 9) as f32) * (1.0 / 4_194_304.0) - 1.0;
        let distance = (n - direct) as f32;
        h[n] += 0.12 * (-distance / 90.0).exp() * noise;
    }
    let energy = h.iter().map(|v| v * v).sum::<f32>().sqrt();
    for value in &mut h {
        *value /= energy;
    }
    h
}

fn convolve(input: &[f32], filter: &[f32], out_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_len];
    for (i, &sample) in input.iter().enumerate().take(out_len) {
        if sample == 0.0 {
            continue;
        }
        let count = filter.len().min(out_len - i);
        for j in 0..count {
            out[i + j] += sample * filter[j];
        }
    }
    out
}

fn build_gate(len: usize, intervals: &[(f64, f64)]) -> Vec<f32> {
    let mut gate = vec![0.0f32; len];
    let fade = (0.01 * SR as f64) as usize;
    for &(start_s, end_s) in intervals {
        let start = (start_s * SR as f64) as usize;
        let end = (end_s * SR as f64) as usize;
        for n in start..end {
            let rising = n - start;
            let falling = end - 1 - n;
            let ramp_in = if rising < fade {
                0.5 - 0.5 * (std::f64::consts::PI * rising as f64 / fade as f64).cos()
            } else {
                1.0
            };
            let ramp_out = if falling < fade {
                0.5 - 0.5 * (std::f64::consts::PI * falling as f64 / fade as f64).cos()
            } else {
                1.0
            };
            gate[n] = ramp_in.min(ramp_out) as f32;
        }
    }
    gate
}

fn gated(input: &[f32], gate: &[f32]) -> Vec<f32> {
    input.iter().zip(gate.iter()).map(|(x, g)| x * g).collect()
}

fn align_reference(reference: &[f32], offset: usize, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    for (n, slot) in out.iter_mut().enumerate() {
        if n >= offset {
            let index = n - offset;
            if index < reference.len() {
                *slot = reference[index];
            }
        }
    }
    out
}

fn detect_reference_offset(mic: &[f32], reference: &[f32]) -> usize {
    let decimation = 8;
    let a = 10 * SR;
    let b = 12 * SR;
    let span = b - a;
    if b > mic.len() || span == 0 {
        return 0;
    }
    let m: Vec<f32> = mic[a..b].iter().step_by(decimation).copied().collect();
    let mic_energy: f64 = m.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let mut best_lag = 0usize;
    let mut best_score = f64::MIN;
    for lag in (0..=30_000).step_by(decimation) {
        if a < lag || a - lag + span > reference.len() {
            continue;
        }
        let start = a - lag;
        let r: Vec<f32> = reference[start..start + span]
            .iter()
            .step_by(decimation)
            .copied()
            .collect();
        let reference_energy: f64 = r.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let correlation: f64 = m
            .iter()
            .zip(r.iter())
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum();
        let score = correlation / (mic_energy * reference_energy).sqrt().max(1.0e-30);
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    let mic_energy_full: f64 = mic[a..b].iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let low = best_lag.saturating_sub(decimation);
    let high = (best_lag + decimation).min(30_000);
    let mut fine = best_lag;
    for lag in low..=high {
        if a < lag || a - lag + span > reference.len() {
            continue;
        }
        let start = a - lag;
        let mut correlation = 0.0f64;
        let mut reference_energy = 0.0f64;
        for n in 0..span {
            let x = mic[a + n] as f64;
            let y = reference[start + n] as f64;
            correlation += x * y;
            reference_energy += y * y;
        }
        let score = correlation / (mic_energy_full * reference_energy).sqrt().max(1.0e-30);
        if score > best_score {
            best_score = score;
            fine = lag;
        }
    }
    fine
}

fn playback_only_runs(
    mic: &[f32],
    reference: &[f32],
    near: &[f32],
    frames: usize,
) -> Vec<(usize, usize)> {
    let near_threshold = 10.0f64.powf(-60.0 / 20.0);
    let reference_threshold = 10.0f64.powf(-45.0 / 20.0);
    let mic_threshold = 10.0f64.powf(-50.0 / 20.0);
    let count = frames / FRAME_LEN;
    let mut good = Vec::with_capacity(count);
    for frame in 0..count {
        let start = frame * FRAME_LEN;
        let end = start + FRAME_LEN;
        let near_level = rms(&near[start..end.min(near.len())]);
        let reference_level = rms(&reference[start..end]);
        let mic_level = rms(&mic[start..end]);
        good.push(
            near_level < near_threshold
                && reference_level > reference_threshold
                && mic_level > mic_threshold,
        );
    }
    let mut runs = Vec::new();
    let mut index = 0;
    while index < good.len() {
        if good[index] {
            let start = index;
            while index < good.len() && good[index] {
                index += 1;
            }
            if index - start >= 32 {
                runs.push((start * FRAME_LEN, index * FRAME_LEN));
            }
        } else {
            index += 1;
        }
    }
    runs
}
