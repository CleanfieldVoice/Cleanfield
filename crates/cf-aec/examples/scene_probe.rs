//! Temporary diagnostic probe (not part of the deliverable).

#[path = "../tests/room/mod.rs"]
mod room;

use room::{scenes, SceneSpec, SEEDS};
use std::f32::consts::PI;

const N: usize = 4096;
const L: usize = 2048;

struct Fft {
    size: usize,
    reverse: Vec<u32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Fft {
    fn new(size: usize) -> Self {
        let bits = size.trailing_zeros();
        let mut reverse = vec![0u32; size];
        for (index, slot) in reverse.iter_mut().enumerate() {
            *slot = (index as u32).reverse_bits() >> (32 - bits);
        }
        let mut cos = vec![0.0f32; size / 2];
        let mut sin = vec![0.0f32; size / 2];
        for (k, (c, s)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
            let phase = -2.0 * PI * (k as f32) / (size as f32);
            *c = phase.cos();
            *s = phase.sin();
        }
        Self {
            size,
            reverse,
            cos,
            sin,
        }
    }
    fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        self.transform(re, im, false);
    }
    fn inverse(&self, re: &mut [f32], im: &mut [f32]) {
        self.transform(re, im, true);
        let scale = 1.0 / self.size as f32;
        for i in 0..self.size {
            re[i] *= scale;
            im[i] *= scale;
        }
    }
    fn transform(&self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        let size = self.size;
        for (index, &target) in self.reverse.iter().enumerate() {
            let target = target as usize;
            if target > index {
                re.swap(index, target);
                im.swap(index, target);
            }
        }
        let sign = if inverse { -1.0f32 } else { 1.0f32 };
        let mut block = 2;
        while block <= size {
            let half = block / 2;
            let step = size / block;
            for start in (0..size).step_by(block) {
                for k in 0..half {
                    let table = k * step;
                    let wr = self.cos[table];
                    let wi = sign * self.sin[table];
                    let a = start + k;
                    let b = a + half;
                    let br = re[b];
                    let bi = im[b];
                    let tr = br * wr - bi * wi;
                    let ti = br * wi + bi * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            block *= 2;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("dump") {
        dump();
        return;
    }
    let want: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(4);
    let frame: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(240);
    let list = scenes();
    let spec: &SceneSpec = &list[want - 1];
    println!("scene {want}: {}", spec.name);
    for &seed in &SEEDS {
        probe(spec, seed, frame);
        trace(spec, seed);
    }
}

fn dump() {
    let list = scenes();
    for scene_index in [4usize, 5, 6, 27, 30] {
        let spec = &list[scene_index - 1];
        for (seed_index, &seed) in SEEDS.iter().enumerate() {
            let scene = room::Scene::generate(spec.clone(), seed);
            for (name, signal) in [
                ("reference", &scene.reference),
                ("mic", &scene.mic),
                ("echo", &scene.echo),
            ] {
                let path = format!(
                    "target/_probe_s{}_seed{}_{}.bin",
                    scene_index, seed_index, name
                );
                let mut bytes = Vec::with_capacity(signal.len() * 4);
                for value in signal {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                std::fs::write(path, bytes).unwrap();
            }
        }
    }
}

fn trace(spec: &SceneSpec, seed: u32) {
    let scene = room::Scene::generate(spec.clone(), seed);
    let mut aec = cf_aec::Aec::new(cf_aec::AecConfig::default());
    let frames = scene.mic.len() / room::FRAME;
    let mut last = aec.delay_samples();
    let mut changes = Vec::new();
    for frame in 0..frames {
        let start = frame * room::FRAME;
        let end = start + room::FRAME;
        aec.process(&scene.mic[start..end], &scene.reference[start..end]);
        let delay = aec.delay_samples();
        if delay != last {
            changes.push(format!("{}->{}@f{}", last, delay, frame));
            last = delay;
        }
    }
    println!(
        "  trace seed {:08X}: final {} (target {:.0}); changes {:?}",
        seed, last, scene.final_delay, changes
    );
    let run = room::run_scene(spec, seed);
    println!(
        "    metrics: applied {} post echo ERLE {:.1} dB, pre {:.1} dB",
        run.metrics.delay_final, run.metrics.echo_erle_post_db, run.metrics.echo_erle_pre_db
    );
}

fn probe(spec: &SceneSpec, seed: u32, frame: usize) {
    let scene = room::Scene::generate(spec.clone(), seed);
    let end = frame * room::FRAME;
    let start_mic = end - L;
    let start_ref = end - N;
    let mut a_re = vec![0.0f32; N];
    let mut a_im = vec![0.0f32; N];
    let mut b_re = vec![0.0f32; N];
    let mut b_im = vec![0.0f32; N];
    a_re[..L].copy_from_slice(&scene.mic[start_mic..end]);
    b_re.copy_from_slice(&scene.reference[start_ref..end]);

    let fft = Fft::new(N);
    fft.forward(&mut a_re, &mut a_im);
    fft.forward(&mut b_re, &mut b_im);
    let mut g_re = vec![0.0f32; N];
    let mut g_im = vec![0.0f32; N];
    let mut mags = vec![0.0f32; N];
    let mut max_mag = 0.0f32;
    for k in 0..N {
        // conj(A)*B
        g_re[k] = a_re[k] * b_re[k] + a_im[k] * b_im[k];
        g_im[k] = a_re[k] * b_im[k] - a_im[k] * b_re[k];
        let mag = (g_re[k] * g_re[k] + g_im[k] * g_im[k]).sqrt();
        mags[k] = mag;
        max_mag = max_mag.max(mag);
    }
    let mean_mag = mags.iter().sum::<f32>() / N as f32;

    // plain correlation
    let mut plain_re = g_re.clone();
    let mut plain_im = g_im.clone();
    fft.inverse(&mut plain_re, &mut plain_im);
    let plain: Vec<f32> = (0..=L).map(|k| plain_re[k]).collect();

    let mut phat_re = g_re.clone();
    let mut phat_im = g_im.clone();
    let eps = 0.01 * mean_mag;
    for k in 0..N {
        let denom = mags[k] + eps;
        phat_re[k] /= denom;
        phat_im[k] /= denom;
    }
    fft.inverse(&mut phat_re, &mut phat_im);
    let phat: Vec<f32> = (0..=L).map(|k| phat_re[k]).collect();

    // mic energy and per-lag reference energy/normalized plain score
    let mic_energy: f32 = scene.mic[start_mic..end].iter().map(|v| v * v).sum();
    let base = N - L;
    let mut best_score = (0usize, 0.0f32);
    let mut scores = vec![0.0f32; L + 1];
    for lag in 0..=L {
        let window_start = start_ref + base - lag;
        let window = &scene.reference[window_start..window_start + L];
        let mut corr = 0.0f32;
        let mut energy = 0.0f32;
        for (i, &r) in window.iter().enumerate() {
            corr += scene.mic[start_mic + i] * r;
            energy += r * r;
        }
        let score = corr.abs() / (mic_energy * energy).sqrt().max(1.0e-20);
        scores[lag] = score;
        if score > best_score.1 {
            best_score = (lag, score);
        }
    }

    let smooth = |values: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; values.len()];
        for i in 0..values.len() {
            let mut sum = 0.0;
            let mut weight = 0.0;
            for d in -2i32..=2 {
                let j = i as i32 + d;
                if j >= 0 && (j as usize) < values.len() {
                    let w = 3.0 - d.abs() as f32;
                    sum += w * values[j as usize];
                    weight += w;
                }
            }
            out[i] = sum / weight;
        }
        out
    };
    let phat_abs: Vec<f32> = phat.iter().map(|v| v.abs()).collect();
    let phat_smooth = smooth(&phat_abs);
    let phat_max = phat_smooth
        .iter()
        .take(L + 1)
        .fold(0.0f32, |a, &b| a.max(b));
    let plain_abs: Vec<f32> = plain.iter().map(|v| v.abs()).collect();
    let plain_smooth = smooth(&plain_abs);
    let plain_max = plain_smooth
        .iter()
        .take(L + 1)
        .fold(0.0f32, |a, &b| a.max(b));

    // top few PHAT peaks
    let mut peaks: Vec<(usize, f32)> = phat_smooth
        .iter()
        .take(L + 1)
        .copied()
        .enumerate()
        .collect();
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top: Vec<String> = peaks
        .iter()
        .take(6)
        .map(|&(k, v)| format!("D={} ({:.2})", L - k, v / phat_max.max(1e-20)))
        .collect();

    let onset = |frac: f32| -> Option<usize> {
        for lag in 0..=L {
            let k = L - lag;
            if phat_smooth[k] >= frac * phat_max {
                return Some(lag);
            }
        }
        None
    };
    let plain_onset = |frac: f32| -> Option<usize> {
        for lag in 0..=L {
            let k = L - lag;
            if plain_smooth[k] >= frac * plain_max {
                return Some(lag);
            }
        }
        None
    };
    println!(
        "seed {:08X}: plain peak D={} score={:.3}; smooth-plainpeak D={}; phat top {:?}",
        seed,
        best_score.0,
        best_score.1,
        L - plain_smooth
            .iter()
            .take(L + 1)
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0,
        top
    );
    let rir_e: f32 = scene.rir.iter().map(|v| v * v).sum();
    println!(
        "  rir[0]={:.3} rir[240]={:.3} rir[520]={:.3} rir[1300]={:.3} rir[1700]={:.3} energy={:.3} micE={:.1} scale={:.3}",
        scene.rir[0], scene.rir[240], scene.rir[520], scene.rir[1300], scene.rir[1700],
        rir_e, mic_energy, scene.input_scale
    );
    // Direct time-domain normalized correlation at selected delays, against the
    // echo component itself and against a delta-delayed reference.
    for lag in [240usize, 480, 520, 1000, 1300, 1780] {
        let window = &scene.reference[start_ref + base - lag..start_ref + base - lag + L];
        let mut corr_mic = 0.0f32;
        let mut corr_echo = 0.0f32;
        let mut energy = 0.0f32;
        for i in 0..L {
            corr_mic += scene.mic[start_mic + i] * window[i];
            corr_echo += scene.echo[start_mic + i] * window[i];
            energy += window[i] * window[i];
        }
        println!(
            "    D={lag}: mic/ref {:.3}, echo/ref {:.3}, scores {:.3}",
            corr_mic / (mic_energy * energy).sqrt(),
            corr_echo / (mic_energy * energy).sqrt(),
            scores[lag]
        );
    }
    for frac in [0.4f32, 0.5, 0.6, 0.7, 0.8] {
        print!(
            "  f={frac:.1}: phat_onset={:?} plain_onset={:?}",
            onset(frac),
            plain_onset(frac)
        );
    }
    println!();
    print!("  values at lags:");
    for lag in [
        0usize, 240, 480, 520, 720, 1000, 1200, 1300, 1700, 1780, 2048,
    ] {
        let k = L - lag;
        print!(
            " D={lag}:{:.2}/{:.2}",
            phat_smooth[k] / phat_max.max(1e-20),
            plain_smooth[k] / plain_max.max(1e-20)
        );
    }
    println!();
    let mut sorted: Vec<f32> = phat_smooth
        .iter()
        .take(L + 1)
        .map(|v| v / phat_max)
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  phat percentiles 50/75/90/95/99: {:.2} {:.2} {:.2} {:.2} {:.2}",
        sorted[L / 2],
        sorted[L * 3 / 4],
        sorted[L * 9 / 10],
        sorted[L * 19 / 20],
        sorted[L * 99 / 100]
    );
    // Local maxima of the smoothed plain and PHAT envelopes, above 10% of max,
    // expressed as delay in samples.
    let local_maxima = |values: &[f32], max: f32| -> Vec<String> {
        let mut out = Vec::new();
        for lag in 1..L {
            let k = L - lag;
            let previous = values[L - (lag - 1)];
            let next = values[L - (lag + 1)];
            if values[k] >= previous && values[k] > next && values[k] >= 0.10 * max {
                out.push(format!("{}:{:.2}", lag, values[k] / max.max(1.0e-20)));
            }
        }
        out
    };
    println!(
        "  plain local maxima: {:?}",
        local_maxima(&plain_smooth, plain_max)
    );
    println!(
        "  phat  local maxima: {:?}",
        local_maxima(&phat_smooth, phat_max)
    );
}
