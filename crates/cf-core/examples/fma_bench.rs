// Temporary micro-benchmark for soft-FMA variants (not part of the engine).
use std::hint::black_box;
use std::time::Instant;

use cf_core::fma::soft_fma;

#[inline(never)]
fn run_engine(n: usize) -> f32 {
    // 32 independent accumulators, like a 1x1 convolution output row.
    let mut x = [0.0f32; 32];
    let mut w = [0.0f32; 32];
    let mut acc = [0.0f32; 32];
    let mut s = 0x1234_5678u64;
    for j in 0..32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        x[j] = ((s >> 40) as f32 / 16777216.0) - 0.5;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        w[j] = ((s >> 40) as f32 / 16777216.0) - 0.5;
        acc[j] = 0.0;
    }
    for _ in 0..n {
        black_box(&mut acc);
        for j in 0..32 {
            acc[j] = soft_fma(x[j], w[j], acc[j]);
        }
    }
    let mut o = 0.0;
    for j in 0..32 {
        o += acc[j];
    }
    black_box(o)
}

fn main() {
    let n: usize = black_box(1000);
    let ops = 32usize * n;
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let v = run_engine(n);
        let dt = t.elapsed().as_secs_f64();
        black_box(v);
        let ns = dt * 1e9 / ops as f64;
        println!("run {dt:.4}s  {ns:.3} ns/fma");
        best = best.min(ns);
    }
    println!("best {best:.3} ns/fma");
}
