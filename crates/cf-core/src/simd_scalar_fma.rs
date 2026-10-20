//! Scalar backend that executes the reference kernels with the FMA
//! instruction.
//!
//! `Backend::Scalar` selects this type only when
//! `is_x86_feature_detected!("fma")` is true; otherwise the same graph runs on
//! [`crate::simd::ScalarSimd`].  The kernel bodies are the ones in
//! [`crate::scalar_kernels`], reached through the `#[target_feature(enable =
//! "fma")]` shims of `scalar_kernels::fma_kernels`, so the compiler can select
//! `vfmadd` for the fused operations and schedule (or vectorise) the loops
//! instead of going through the opaque scalar inline-asm helper.  Every lane
//! still computes the same correctly-rounded IEEE-754 fused multiply-add as the
//! reference's `f32::mul_add`/`fmaf`, so the output is bit-identical - this is a
//! pure instruction-selection change, not a numerical one.
//!
//! The feature is local to those shims: the rest of the crate is compiled
//! without it and the shims are only reached after the runtime FMA check, so a
//! CPU without `fma` never executes one.
//!
//! The kernel bodies live in [`crate::scalar_kernels`] and are shared between
//! the two compile-time FMA strategies, so both paths are literally the same
//! operation sequence.

use crate::scalar_kernels as kernels;
use crate::simd::{ActConsts, Simd};

/// Scalar reference graph executed with the scalar FMA instruction.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarFmaSimd;

/// Runtime dispatch to an `fma` target-feature shim.
///
/// Expands to a plain `if`, so only one side is evaluated: `$hw` is an
/// expression that calls one of the `unsafe` shims in
/// [`crate::scalar_kernels::fma_kernels`], and `$sw` is the portable
/// `HW = false` kernel.  On a CPU without `fma` the `unsafe` call is never
/// reached and the portable kernel runs instead; inside `$sw`, `fma()` resolves
/// to the correctly-rounded native/libm FMA, so the fallback only costs speed.
macro_rules! dispatch {
    ($hw:expr, $sw:expr) => {
        if kernels::fma_available() {
            // SAFETY: `fma_available()` just returned true, so the target
            // feature required by the shim is present on this CPU.
            unsafe { $hw }
        } else {
            $sw
        }
    };
}

impl Simd for ScalarFmaSimd {
    #[inline]
    fn name(&self) -> &'static str {
        "scalar"
    }

    #[inline]
    fn fma_broadcast(acc: &mut [f32], x: f32, w: &[f32]) {
        dispatch!(
            kernels::fma_kernels::fma_broadcast(acc, x, w),
            kernels::fma_broadcast::<false>(acc, x, w)
        );
    }

    #[inline]
    fn fma_tiled(acc: &mut [f32], x: &[f32], w: &[f32]) {
        dispatch!(
            kernels::fma_kernels::fma_tiled(acc, x, w),
            kernels::fma_tiled::<false>(acc, x, w)
        );
    }

    #[inline]
    fn mul_add_tap(dst: &mut [f32], src: &[f32], k: f32) {
        dispatch!(
            kernels::fma_kernels::mul_add_tap(dst, src, k),
            kernels::mul_add_tap(dst, src, k)
        );
    }

    fn conv1x1_1(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
        dispatch!(
            kernels::fma_kernels::conv1x1_1(dst, src, w, bias),
            kernels::conv1x1_1::<false>(dst, src, w, bias)
        );
    }

    fn conv1x1_32(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
        dispatch!(
            kernels::fma_kernels::conv1x1_32(dst, src, w, bias),
            kernels::conv1x1_32::<false>(dst, src, w, bias)
        );
    }

    fn depthwise_3x3(dst: &mut [f32], src: &[f32], w: &[f32], in_len: usize) {
        dispatch!(
            kernels::fma_kernels::depthwise_3x3(dst, src, w, in_len),
            kernels::depthwise_3x3::<false>(dst, src, w, in_len)
        );
    }

    fn depthwise_1x3s2(dst: &mut [f32], src: &[f32], w: &[f32], pad_left: usize) {
        dispatch!(
            kernels::fma_kernels::depthwise_1x3s2(dst, src, w, pad_left),
            kernels::depthwise_1x3s2::<false>(dst, src, w, pad_left)
        );
    }

    fn maxpool1x3s2(dst: &mut [f32], src: &[f32], in_len: usize) {
        dispatch!(
            kernels::fma_kernels::maxpool1x3s2(dst, src, in_len),
            kernels::maxpool1x3s2(dst, src, in_len)
        );
    }

    fn gemm_i8(acc: &mut [i32], w: &[i8], zp: &[i8], qx: &[i16], k: usize, n: usize) {
        dispatch!(
            kernels::fma_kernels::gemm_i8(acc, w, zp, qx, k, n),
            kernels::gemm_i8(acc, w, zp, qx, k, n)
        );
    }

    fn quantize_i16(dst: &mut [i16], src: &[f32], inv: f32, clamp: i32) {
        dispatch!(
            kernels::fma_kernels::quantize_i16(dst, src, inv, clamp),
            kernels::quantize_i16(dst, src, inv, clamp)
        );
    }

    fn lstm_activate(y: &mut [f32], h: usize, k: &ActConsts) {
        dispatch!(
            kernels::fma_kernels::lstm_activate(y, h, k),
            kernels::lstm_activate::<false>(y, h, k)
        );
    }

    fn lstm_cell(
        cell: &mut [f32],
        qh: &mut [i16],
        y: &[f32],
        inv_h: f32,
        clamp: i32,
        k: &ActConsts,
    ) {
        dispatch!(
            kernels::fma_kernels::lstm_cell(cell, qh, y, inv_h, clamp, k),
            kernels::lstm_cell::<false>(cell, qh, y, inv_h, clamp, k)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar_kernels::fma_available;
    use crate::simd::{ActConsts, ScalarSimd};

    struct Rng(u64);

    impl Rng {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
        }
        fn i8(&mut self) -> i8 {
            self.next_u32() as i8
        }
        fn i16(&mut self, limit: i32) -> i16 {
            ((self.next_u32() % (2 * limit as u32 + 1)) as i32 - limit) as i16
        }
        fn f32(&mut self) -> f32 {
            let u = self.next_u32() >> 8;
            (u as f32 / (1u32 << 24) as f32) * 8.0 - 4.0
        }
    }

    fn skip() -> bool {
        if !fma_available() {
            eprintln!("fma unavailable, skipping");
            return true;
        }
        false
    }

    fn act_consts() -> ActConsts {
        // The activation constants only need to be arbitrary finite values for
        // an op-order comparison; the real ones come from the container.
        ActConsts {
            log2e: core::f32::consts::LOG2_E,
            ln2: -core::f32::consts::LN_2,
            exp_poly: [0.0008, 0.0096, 0.0555, 0.2402, 0.6931],
            half: 0.5,
            one: 1.0,
            two: 2.0,
            four: 4.0,
            neg_four: -4.0,
            exp_clamp: 88.0,
            neg_exp_clamp: -88.0,
            cell_poly: [0.0156, 0.2369, 1.0, 1.0, 27.0],
        }
    }

    #[test]
    fn every_kernel_matches_the_scalar_reference() {
        if skip() {
            return;
        }
        let mut rng = Rng(0x5eed_0bad_cafe_f00d);
        let positions = 37usize;
        let src: Vec<f32> = (0..positions * 32).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..32 * 32).map(|_| rng.f32() * 0.25).collect();
        let bias: Vec<f32> = (0..32).map(|_| rng.f32()).collect();
        let mut a = vec![0.0f32; positions * 32];
        let mut b = vec![0.0f32; positions * 32];
        ScalarSimd::conv1x1_32(&mut a, &src, &w, &bias);
        ScalarFmaSimd::conv1x1_32(&mut b, &src, &w, &bias);
        assert_eq!(a, b, "conv1x1_32");

        let single: Vec<f32> = (0..positions).map(|_| rng.f32()).collect();
        let w1: Vec<f32> = (0..32).map(|_| rng.f32() * 0.5).collect();
        ScalarSimd::conv1x1_1(&mut a, &single, &w1, &bias);
        ScalarFmaSimd::conv1x1_1(&mut b, &single, &w1, &bias);
        assert_eq!(a, b, "conv1x1_1");

        let in_len = 255usize;
        let src3: Vec<f32> = (0..3 * in_len * 32).map(|_| rng.f32()).collect();
        let w3: Vec<f32> = (0..9 * 32).map(|_| rng.f32()).collect();
        let mut a = vec![0.0f32; (in_len - 2) * 32];
        let mut b = vec![0.0f32; (in_len - 2) * 32];
        ScalarSimd::depthwise_3x3(&mut a, &src3, &w3, in_len);
        ScalarFmaSimd::depthwise_3x3(&mut b, &src3, &w3, in_len);
        assert_eq!(a, b, "depthwise_3x3");

        for &(len, pad) in &[(126usize, 0usize), (63, 1), (32, 1), (7, 0), (5, 1)] {
            let s: Vec<f32> = (0..len * 32).map(|_| rng.f32()).collect();
            let w: Vec<f32> = (0..3 * 32).map(|_| rng.f32()).collect();
            let mut a = vec![0.0f32; len.div_ceil(2) * 32];
            let mut b = vec![0.0f32; len.div_ceil(2) * 32];
            ScalarSimd::depthwise_1x3s2(&mut a, &s, &w, pad);
            ScalarFmaSimd::depthwise_1x3s2(&mut b, &s, &w, pad);
            assert_eq!(a, b, "depthwise_1x3s2 len={len} pad={pad}");
        }

        // Pooling including signed zeros and exact ties.
        let len = 63usize;
        let mut s: Vec<f32> = (0..3 * len * 32).map(|_| rng.f32()).collect();
        for chunk in s.chunks_mut(4) {
            chunk[0] = 0.0;
            if chunk.len() > 1 {
                chunk[1] = -0.0;
            }
        }
        // `s` above holds three pool segments of `len` positions, so the
        // destination is three times the 31-position output of one segment.
        let mut a = vec![0.0f32; 3 * 31 * 32];
        let mut b = vec![0.0f32; 3 * 31 * 32];
        ScalarSimd::maxpool1x3s2(&mut a, &s, len);
        ScalarFmaSimd::maxpool1x3s2(&mut b, &s, len);
        assert_eq!(a, b, "maxpool1x3s2");

        // FMA helpers.
        let mut a = vec![1.0f32; 32];
        let mut b = vec![1.0f32; 32];
        let w: Vec<f32> = (0..32).map(|_| rng.f32()).collect();
        ScalarSimd::fma_broadcast(&mut a, 0.75, &w);
        ScalarFmaSimd::fma_broadcast(&mut b, 0.75, &w);
        assert_eq!(a, b, "fma_broadcast");
        ScalarSimd::fma_tiled(&mut a, &w, &w[..4]);
        ScalarFmaSimd::fma_tiled(&mut b, &w, &w[..4]);
        assert_eq!(a, b, "fma_tiled");
        ScalarSimd::mul_add_tap(&mut a, &w, 0.25);
        ScalarFmaSimd::mul_add_tap(&mut b, &w, 0.25);
        assert_eq!(a, b, "mul_add_tap");

        // Quantiser, including clamp edges and subnormals.
        let mut s: Vec<f32> = (0..97).map(|_| rng.f32() * 1000.0).collect();
        s[0] = f32::INFINITY;
        s[1] = f32::NEG_INFINITY;
        s[2] = f32::NAN;
        s[3] = 8191.5;
        s[4] = -8192.5;
        s[5] = f32::from_bits(1);
        let mut a = vec![0i16; s.len()];
        let mut b = vec![0i16; s.len()];
        ScalarSimd::quantize_i16(&mut a, &s, 0.001, 8192);
        ScalarFmaSimd::quantize_i16(&mut b, &s, 0.001, 8192);
        assert_eq!(a, b, "quantize_i16");

        // GEMMs with sparse rows and extreme values.
        for &(k, n, clamp) in &[
            (1024usize, 1024usize, 8192i32),
            (256, 1024, 8192),
            (512, 1026, 16384),
            (256, 512, 16384),
        ] {
            let w: Vec<i8> = (0..k * n).map(|_| rng.i8()).collect();
            let zp: Vec<i8> = (0..n).map(|_| rng.i8()).collect();
            let qx: Vec<i16> = (0..k).map(|_| rng.i16(clamp)).collect();
            let mut a = vec![0i32; n];
            let mut b = vec![0i32; n];
            ScalarSimd::gemm_i8(&mut a, &w, &zp, &qx, k, n);
            ScalarFmaSimd::gemm_i8(&mut b, &w, &zp, &qx, k, n);
            assert_eq!(a, b, "gemm_i8 k={k} n={n}");
        }

        // Activations and cell update.
        let k = act_consts();
        let h = 64usize;
        let mut y: Vec<f32> = (0..4 * h).map(|_| rng.f32() * 4.0).collect();
        let mut y2 = y.clone();
        ScalarSimd::lstm_activate(&mut y, h, &k);
        ScalarFmaSimd::lstm_activate(&mut y2, h, &k);
        assert_eq!(y, y2, "lstm_activate");
        let mut cell: Vec<f32> = (0..h).map(|_| rng.f32() * 4.0).collect();
        let mut cell2 = cell.clone();
        let mut qh = vec![0i16; h];
        let mut qh2 = vec![0i16; h];
        ScalarSimd::lstm_cell(&mut cell, &mut qh, &y, 0.01, 32767, &k);
        ScalarFmaSimd::lstm_cell(&mut cell2, &mut qh2, &y, 0.01, 32767, &k);
        assert_eq!(cell, cell2, "lstm_cell state");
        assert_eq!(qh, qh2, "lstm_cell qh");
    }
}
