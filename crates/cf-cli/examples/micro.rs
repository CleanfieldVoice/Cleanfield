//! Scratch microbenchmark for the int8 GEMM and the LSTM layer parts
//! (not part of the shipped CLI).

use std::hint::black_box;
use std::time::Instant;

use cf_core::{Avx2Simd, Backend, Simd};
use cf_voice::tags::*;
use cf_voice::voice::{VoiceModel, INPUT_LEN};

fn zeros(v: &[i16]) -> f64 {
    v.iter().filter(|&&x| x == 0).count() as f64 / v.len() as f64
}

fn macs(k: usize, n: usize) -> f64 {
    (k * n) as f64
}

fn bench_gemm(name: &str, k: usize, n: usize, iters: usize, w: &[i8], zp: &[i8], qs: &[Vec<i16>]) {
    let mut acc = vec![0i32; n];
    for _ in 0..10 {
        Avx2Simd::gemm_i8(&mut acc, w, zp, &qs[0], k, n);
    }
    let mut best = u128::MAX;
    let mut sum = 0u128;
    // Every iteration uses a different frame's activations.
    for _ in 0..7 {
        let t = Instant::now();
        for it in 0..iters {
            let q = &qs[it % qs.len()];
            Avx2Simd::gemm_i8(
                black_box(&mut acc),
                black_box(w),
                black_box(zp),
                black_box(q),
                black_box(k),
                black_box(n),
            );
        }
        let ns = t.elapsed().as_nanos();
        sum += ns;
        best = best.min(ns);
    }
    println!(
        "{name:>12} best {:>8.3} us  mean {:>8.3} us  {:>6.2} GMAC/s  check {}",
        best as f64 / iters as f64 / 1000.0,
        sum as f64 / 7.0 / iters as f64 / 1000.0,
        macs(k, n) * 1e-9 / (best as f64 / iters as f64 / 1e9),
        acc[n / 2]
    );
}

fn bench<T: FnMut()>(name: &str, iters: usize, mut f: T) {
    let mut best = u128::MAX;
    let mut sum = 0u128;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let ns = t.elapsed().as_nanos();
        sum += ns;
        best = best.min(ns);
    }
    println!(
        "{name:>14} best {:>8.4} us  mean {:>8.4} us",
        best as f64 / iters as f64 / 1000.0,
        sum as f64 / 7.0 / iters as f64 / 1000.0
    );
}

fn main() {
    if !Backend::Avx2.is_available() {
        println!("avx2 unavailable");
        return;
    }
    let bytes = include_bytes!("../../../models/voice-default.cfwm");
    let mut m = VoiceModel::with_backend(bytes, Backend::Avx2).expect("model");
    let _container_frames = 0usize;
    let mut st = 0x1234_5678_9abc_def0u64;
    let mut frames: Vec<[f32; INPUT_LEN]> = Vec::new();
    for _ in 0..64 {
        let mut frame = [0.0f32; INPUT_LEN];
        for v in frame.iter_mut() {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            *v = ((st % 2001) as f32) * 0.001 - 1.0;
        }
        frames.push(frame);
    }
    for frame in &frames {
        m.process(frame).unwrap();
    }
    // Capture the last 32 frames' quantised inputs is not possible after the
    // fact; recompute and capture.
    let mut qx15s: Vec<Vec<i16>> = Vec::new();
    let mut qh15s: Vec<Vec<i16>> = Vec::new();
    let mut qh16s: Vec<Vec<i16>> = Vec::new();
    let mut qx18s: Vec<Vec<i16>> = Vec::new();
    let mut y15: Vec<f32> = Vec::new();
    let mut cell: Vec<f32> = Vec::new();
    m.reset();
    for frame in &frames {
        m.process(frame).unwrap();
        qx15s.push(m.slot_i16("L15_qx").unwrap().to_vec());
        qh15s.push(m.slot_i16("L15_qh").unwrap().to_vec());
        qh16s.push(m.slot_i16("L16_qh").unwrap().to_vec());
        qx18s.push(m.slot_i16("L18_qx").unwrap().to_vec());
        if y15.is_empty() {
            y15 = m.slot("L15_y").unwrap().to_vec();
            cell = m.slot("L15_c").unwrap().to_vec();
        }
    }
    let avg =
        |v: &Vec<Vec<i16>>| -> f64 { v.iter().map(|x| zeros(x)).sum::<f64>() / v.len() as f64 };
    println!(
        "mean zero fraction: qx15 {:.3} qh15 {:.3} qh16 {:.3} qx18 {:.3}",
        avg(&qx15s),
        avg(&qh15s),
        avg(&qh16s),
        avg(&qx18s)
    );
    let c = m.container();
    let sp15i = c.f32(TAG_L15_SP_IN).unwrap();
    let b15 = c.f32(TAG_L15_BIAS).unwrap();
    let inv15 = m.consts().inv_h15();
    let act = m.consts().act();
    bench_gemm(
        "L15 in",
        1024,
        1024,
        3000,
        c.i8(TAG_L15_W_IN).unwrap(),
        c.i8(TAG_L15_ZP_IN).unwrap(),
        &qx15s,
    );
    bench_gemm(
        "L15 rec",
        256,
        1024,
        12000,
        c.i8(TAG_L15_W_REC).unwrap(),
        c.i8(TAG_L15_ZP_REC).unwrap(),
        &qh15s,
    );
    bench_gemm(
        "L16 in",
        256,
        1024,
        12000,
        c.i8(TAG_L16_W_IN).unwrap(),
        c.i8(TAG_L16_ZP_IN).unwrap(),
        &qh15s,
    );
    bench_gemm(
        "L16 rec",
        256,
        1024,
        12000,
        c.i8(TAG_L16_W_REC).unwrap(),
        c.i8(TAG_L16_ZP_REC).unwrap(),
        &qh16s,
    );
    bench_gemm(
        "L17",
        256,
        512,
        20000,
        c.i8(TAG_L17_W).unwrap(),
        c.i8(TAG_L17_ZP).unwrap(),
        &qh16s,
    );
    bench_gemm(
        "L18",
        512,
        1026,
        10000,
        c.i8(TAG_L18_W).unwrap(),
        c.i8(TAG_L18_ZP).unwrap(),
        &qx18s,
    );

    let acc: Vec<i32> = (0..1024).map(|i| i * 977 - 500_000).collect();
    let mut y = vec![0.0f32; 1024];
    bench("epilogue", 20000, || {
        for j in 0..1024 {
            y[j] = cf_core::mul_add2(black_box(acc[j]) as f32, sp15i[j], b15[j]);
        }
    });
    let mut y2 = y15.clone();
    bench("activate-1024", 20000, || {
        Avx2Simd::lstm_activate(black_box(&mut y2), 256, &act);
    });
    let mut cell2 = cell.clone();
    let mut qh = vec![0i16; 256];
    bench("cell-256", 20000, || {
        Avx2Simd::lstm_cell(
            black_box(&mut cell2),
            black_box(&mut qh),
            black_box(&y15),
            inv15,
            8192,
            &act,
        );
    });
    let mut q = vec![0i16; 1024];
    let src = m.slot("L12").unwrap().to_vec();
    bench("quantize-1024", 20000, || {
        Avx2Simd::quantize_i16(black_box(&mut q), black_box(&src), 0.0001, 8192);
    });

    // ---- conv / depthwise / pool kernels on their real shapes ----
    let mut dw_src = vec![0.0f32; 3 * 255 * 32];
    for (i, v) in dw_src.iter_mut().enumerate() {
        *v = ((i * 37 % 211) as f32) * 0.01 - 1.0;
    }
    let dw_w: Vec<f32> = (0..9 * 32).map(|i| (i as f32) * 0.001 - 0.1).collect();
    let mut dw_dst = vec![0.0f32; 253 * 32];
    bench("dw3x3-255", 20000, || {
        Avx2Simd::depthwise_3x3(
            black_box(&mut dw_dst),
            black_box(&dw_src),
            black_box(&dw_w),
            255,
        );
    });
    let mut c32_src = vec![0.0f32; 253 * 32];
    for (i, v) in c32_src.iter_mut().enumerate() {
        *v = ((i * 41 % 197) as f32) * 0.01 - 0.9;
    }
    let c32_w: Vec<f32> = (0..32 * 32).map(|i| (i as f32) * 0.0003 - 0.15).collect();
    let c32_b: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
    let mut c32_dst = vec![0.0f32; 253 * 32];
    bench("conv1x1-253", 20000, || {
        Avx2Simd::conv1x1_32(
            black_box(&mut c32_dst),
            black_box(&c32_src),
            black_box(&c32_w),
            black_box(&c32_b),
        );
    });
    let c1_src = vec![0.5f32; 3 * 511];
    let c1_w: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
    let mut c1_dst = vec![0.0f32; 3 * 511 * 32];
    bench("conv1x1_1-1533", 20000, || {
        Avx2Simd::conv1x1_1(
            black_box(&mut c1_dst),
            black_box(&c1_src),
            black_box(&c1_w),
            black_box(&c32_b),
        );
    });
    let pool_src = vec![0.25f32; 3 * 511 * 32];
    let mut pool_dst = vec![0.0f32; 3 * 255 * 32];
    bench("maxpool-511", 20000, || {
        Avx2Simd::maxpool1x3s2(black_box(&mut pool_dst), black_box(&pool_src), 511);
    });

    // Whole L15 layer (both GEMMs + epilogues + activations + cell), using the
    // real per-frame quantised inputs captured above.
    println!(
        "align: L15in {} L15rec {} L18 {}",
        c.i8(TAG_L15_W_IN).unwrap().as_ptr() as usize % 64,
        c.i8(TAG_L15_W_REC).unwrap().as_ptr() as usize % 64,
        c.i8(TAG_L18_W).unwrap().as_ptr() as usize % 64
    );
    let sp15r = c.f32(TAG_L15_SP_REC).unwrap();
    let mut y = vec![0.0f32; 1024];
    let mut cellx = vec![0.0f32; 256];
    let mut qhx = vec![0i16; 256];
    let mut accx = vec![0i32; 1024];
    let mut fi = 0usize;
    bench("L15 whole", 20000, || {
        let qx: &[i16] = &qx15s[fi % qx15s.len()];
        let qhr: &[i16] = &qh15s[fi % qh15s.len()];
        fi += 1;
        Avx2Simd::gemm_i8(
            black_box(&mut accx),
            black_box(c.i8(TAG_L15_W_IN).unwrap()),
            black_box(c.i8(TAG_L15_ZP_IN).unwrap()),
            qx,
            1024,
            1024,
        );
        Avx2Simd::scale_i32_add(black_box(&mut y), &accx, sp15i, b15);
        Avx2Simd::gemm_i8(
            black_box(&mut accx),
            black_box(c.i8(TAG_L15_W_REC).unwrap()),
            black_box(c.i8(TAG_L15_ZP_REC).unwrap()),
            qhr,
            256,
            1024,
        );
        Avx2Simd::scale_i32_add_inplace(black_box(&mut y), &accx, sp15r);
        Avx2Simd::lstm_activate(black_box(&mut y), 256, &act);
        Avx2Simd::lstm_cell(
            black_box(&mut cellx),
            black_box(&mut qhx),
            black_box(&y),
            inv15,
            8192,
            &act,
        );
    });

    // Full model, same deterministic frame sequence as the CLI bench.
    let mut m2 = VoiceModel::with_backend(bytes, Backend::Avx2).expect("model 2");
    let mut f = 0usize;
    bench("full model", 20000, || {
        let frame: &[f32] = &frames[f % frames.len()];
        f += 1;
        m2.process(black_box(frame)).unwrap();
    });
}
