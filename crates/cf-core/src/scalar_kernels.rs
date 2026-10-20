//! Scalar layer kernels shared by the runtime-selected scalar backends.
//!
//! The bodies are written once and instantiated with a compile-time FMA
//! strategy:
//!
//! * `HW = false` uses [`crate::fma::fma`] - the platform's correctly-rounded
//!   FMA (hardware `fmaf` on a native build without the `fma` target feature,
//!   the exact `f64` software FMA on wasm).
//! * `HW = true` uses `f32::mul_add`, which the compiler turns into a hardware
//!   `vfmadd` only inside a `#[target_feature(enable = "fma")]` function.  The
//!   instantiations are therefore reached exclusively through the `unsafe`
//!   shims in [`fma_kernels`], which `ScalarFmaSimd` calls after the runtime
//!   `is_x86_feature_detected!("fma")` check.  Within such a function LLVM may
//!   contract the `llvm.fma` intrinsic into `vfmadd231ss`, vectorise it to
//!   `vfmadd231ps` and schedule the loads: the per-lane rounding is the same
//!   IEEE-754 correctly-rounded fused multiply-add, so every output bit is
//!   unchanged.  (The one-instruction inline-asm helper [`hw_fma`] is opaque
//!   to the optimiser and deliberately blocks that vectorisation; it remains
//!   the runtime-dispatched implementation behind [`crate::fma::fma`].)
//!
//! Both strategies implement the same IEEE-754 correctly-rounded fused
//! multiply-add, so the operation order - and therefore every output bit - is
//! identical.  `simd_scalar_fma` has unit tests comparing every kernel against
//! the [`crate::simd::ScalarSimd`] reference, and the CLI/`simd-check`/wasm
//! gates compare complete graphs.

use crate::fma::fma;
use crate::rcp::rcp_ps;
use crate::simd::ActConsts;

/// Channel count of the convolutional stack.
pub(crate) const CHANNELS: usize = 32;
/// IEEE-754 single-precision exponent bias (structural, not a model constant).
const EXP_BIAS: i32 = 127;

/// Whether the CPU can execute the `HW = true` kernels.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn fma_available() -> bool {
    std::arch::is_x86_feature_detected!("fma")
}

/// `a * b + c` with a single rounding, using the scalar FMA instruction.
///
/// # Safety
///
/// The CPU must support the `fma` target feature; callers go through
/// [`fma_available`] (the voice dispatcher does exactly that).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn hw_fma(a: f32, b: f32, c: f32) -> f32 {
    use core::arch::asm;
    let mut out = c;
    // Intel syntax: `vfmadd231ss dest, src1, src2` -> dest = src1*src2 + dest.
    asm!(
        "vfmadd231ss {out}, {a}, {b}",
        out = inout(xmm_reg) out,
        a = in(xmm_reg) a,
        b = in(xmm_reg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    out
}

/// One FMA under the chosen compile-time strategy.
#[inline(always)]
pub(crate) fn kfma<const HW: bool>(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if HW {
            // `HW = true` is only instantiated inside the `fma` target-feature
            // shims below, where `mul_add` becomes a single `vfmadd`
            // instruction.  Compiling the same call outside such a function
            // would fall back to the correctly-rounded library `fmaf`, which is
            // still bit-identical - just slower and never selected.
            return a.mul_add(b, c);
        }
    }
    let _ = HW;
    fma(a, b, c)
}

/// `acc[j] = fma(x, w[j], acc[j])`.
#[inline(always)]
pub(crate) fn fma_broadcast<const HW: bool>(acc: &mut [f32], x: f32, w: &[f32]) {
    debug_assert_eq!(acc.len(), w.len());
    for (a, wi) in acc.iter_mut().zip(w.iter()) {
        *a = kfma::<HW>(x, *wi, *a);
    }
}

/// `acc[i] = fma(x[i], w[i % channels], acc[i])`.
#[inline(always)]
pub(crate) fn fma_tiled<const HW: bool>(acc: &mut [f32], x: &[f32], w: &[f32]) {
    debug_assert_eq!(acc.len(), x.len());
    let channels = w.len();
    debug_assert!(channels > 0 && acc.len().is_multiple_of(channels));
    for i in 0..acc.len() {
        acc[i] = kfma::<HW>(x[i], w[i % channels], acc[i]);
    }
}

/// `dst[i] = dst[i] + (src[i] * k)` with two roundings (not an FMA).
#[inline(always)]
pub(crate) fn mul_add_tap(dst: &mut [f32], src: &[f32], k: f32) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        let p = *s * k; // separate rounding, must not be contracted
        *d += p;
    }
}

/// Fused ReLU applied to one output row: `if !(v > 0) { v = 0 }`.
#[inline]
fn relu(row: &mut [f32]) {
    for v in row.iter_mut() {
        if !(*v > 0.0) {
            *v = 0.0;
        }
    }
}

/// 1x1 convolution, one input channel: bias, one FMA, fused ReLU.
#[inline(always)]
pub(crate) fn conv1x1_1<const HW: bool>(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
    let positions = dst.len() / CHANNELS;
    debug_assert_eq!(dst.len(), positions * CHANNELS);
    debug_assert_eq!(src.len(), positions);
    debug_assert_eq!(w.len(), CHANNELS);
    debug_assert_eq!(bias.len(), CHANNELS);
    for pos in 0..positions {
        let row = &mut dst[pos * CHANNELS..(pos + 1) * CHANNELS];
        let x = src[pos];
        for oc in 0..CHANNELS {
            row[oc] = kfma::<HW>(x, w[oc], bias[oc]);
        }
        relu(row);
    }
}

/// 1x1 convolution, 32 input channels -> 32 output channels, fused ReLU.
///
/// The 32 accumulators stay in registers across the input-channel loop; the
/// weight row for one input channel is 128 contiguous bytes, which is the
/// cache-blocking unit that keeps the kernel resident in L1.
#[inline(always)]
pub(crate) fn conv1x1_32<const HW: bool>(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
    let positions = dst.len() / CHANNELS;
    debug_assert_eq!(dst.len(), positions * CHANNELS);
    debug_assert_eq!(src.len(), positions * CHANNELS);
    debug_assert_eq!(w.len(), CHANNELS * CHANNELS);
    debug_assert_eq!(bias.len(), CHANNELS);
    for pos in 0..positions {
        let x = &src[pos * CHANNELS..(pos + 1) * CHANNELS];
        let row = &mut dst[pos * CHANNELS..(pos + 1) * CHANNELS];
        let mut acc = [0.0f32; CHANNELS];
        acc.copy_from_slice(&bias[..CHANNELS]);
        for ic in 0..CHANNELS {
            let xi = x[ic];
            let wrow = &w[ic * CHANNELS..(ic + 1) * CHANNELS];
            for oc in 0..CHANNELS {
                acc[oc] = kfma::<HW>(xi, wrow[oc], acc[oc]);
            }
        }
        for oc in 0..CHANNELS {
            row[oc] = if acc[oc] > 0.0 { acc[oc] } else { 0.0 };
        }
    }
}

/// Depthwise 3x3 with taps in `[kh][kw]` order.
///
/// One output position at a time: nine contiguous source rows and nine
/// contiguous weight rows feed 32 independent accumulators, so the output row
/// is written once instead of nine times.
#[inline(always)]
pub(crate) fn depthwise_3x3<const HW: bool>(
    dst: &mut [f32],
    src: &[f32],
    w: &[f32],
    in_len: usize,
) {
    let out_len = in_len - 2;
    debug_assert_eq!(src.len(), 3 * in_len * CHANNELS);
    debug_assert_eq!(dst.len(), out_len * CHANNELS);
    debug_assert_eq!(w.len(), 9 * CHANNELS);
    for p in 0..out_len {
        let r0 = p * CHANNELS;
        let r1 = (in_len + p) * CHANNELS;
        let r2 = (2 * in_len + p) * CHANNELS;
        let d = &mut dst[p * CHANNELS..(p + 1) * CHANNELS];
        for ch in 0..CHANNELS {
            let mut a = 0.0f32;
            a = kfma::<HW>(src[r0 + ch], w[ch], a);
            a = kfma::<HW>(src[r0 + CHANNELS + ch], w[CHANNELS + ch], a);
            a = kfma::<HW>(src[r0 + 2 * CHANNELS + ch], w[2 * CHANNELS + ch], a);
            a = kfma::<HW>(src[r1 + ch], w[3 * CHANNELS + ch], a);
            a = kfma::<HW>(src[r1 + CHANNELS + ch], w[4 * CHANNELS + ch], a);
            a = kfma::<HW>(src[r1 + 2 * CHANNELS + ch], w[5 * CHANNELS + ch], a);
            a = kfma::<HW>(src[r2 + ch], w[6 * CHANNELS + ch], a);
            a = kfma::<HW>(src[r2 + CHANNELS + ch], w[7 * CHANNELS + ch], a);
            a = kfma::<HW>(src[r2 + 2 * CHANNELS + ch], w[8 * CHANNELS + ch], a);
            d[ch] = a;
        }
    }
}

/// Depthwise 1x3 stride 2 with left padding, taps in tap order.
#[inline(always)]
pub(crate) fn depthwise_1x3s2<const HW: bool>(
    dst: &mut [f32],
    src: &[f32],
    w: &[f32],
    pad_left: usize,
) {
    let in_len = src.len() / CHANNELS;
    let out_len = in_len.div_ceil(2);
    debug_assert_eq!(dst.len(), out_len * CHANNELS);
    debug_assert_eq!(w.len(), 3 * CHANNELS);
    for p in 0..out_len {
        let mut acc = [0.0f32; CHANNELS];
        for t in 0..3 {
            let idx = (2 * p + t) as isize - pad_left as isize;
            if idx < 0 || idx as usize >= in_len {
                continue;
            }
            let off = idx as usize * CHANNELS;
            let s = &src[off..off + CHANNELS];
            let tap = &w[t * CHANNELS..(t + 1) * CHANNELS];
            for ch in 0..CHANNELS {
                acc[ch] = kfma::<HW>(s[ch], tap[ch], acc[ch]);
            }
        }
        dst[p * CHANNELS..(p + 1) * CHANNELS].copy_from_slice(&acc);
    }
}

/// Max pool 1x3 stride 2; ties select the later operand.
#[inline(always)]
pub(crate) fn maxpool1x3s2(dst: &mut [f32], src: &[f32], in_len: usize) {
    let out_len = (in_len - 3) / 2 + 1;
    let segments = src.len() / (in_len * CHANNELS);
    debug_assert_eq!(dst.len(), segments * out_len * CHANNELS);
    for seg in 0..segments {
        for p in 0..out_len {
            for ch in 0..CHANNELS {
                let base = (seg * in_len + p * 2) * CHANNELS + ch;
                let a = src[base];
                let b = src[base + CHANNELS];
                let c = src[base + 2 * CHANNELS];
                // Two sequential maxima; ties select the later operand.
                let m = if a > b { a } else { b };
                let m = if m > c { m } else { c };
                dst[(seg * out_len + p) * CHANNELS + ch] = m;
            }
        }
    }
}

/// `acc[j] = sum_k qx[k] * (w[k][j] - zp[j])` in `int32`, `w` input major.
#[inline(always)]
pub(crate) fn gemm_i8(acc: &mut [i32], w: &[i8], zp: &[i8], qx: &[i16], k: usize, n: usize) {
    debug_assert!(acc.len() >= n);
    debug_assert_eq!(w.len(), k * n);
    debug_assert!(zp.len() >= n);
    debug_assert!(qx.len() >= k);
    acc[..n].fill(0);
    for kk in 0..k {
        let q = qx[kk] as i32;
        if q == 0 {
            continue;
        }
        let row = &w[kk * n..kk * n + n];
        for j in 0..n {
            let wz = row[j] as i32 - zp[j] as i32;
            acc[j] = acc[j].wrapping_add(q.wrapping_mul(wz));
        }
    }
}

/// `dst = clamp(rne(src * inv))` as `int16`.
#[inline(always)]
pub(crate) fn quantize_i16(dst: &mut [i16], src: &[f32], inv: f32, clamp: i32) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        *d = crate::simd::quantize_one(v, inv, clamp);
    }
}

// ---------------------------------------------------------------------------
// activations (same constants, same operation order as the reference)
// ---------------------------------------------------------------------------

#[inline]
fn exp_kernel<const HW: bool>(t: f32, k: &ActConsts) -> f32 {
    let n = (t * k.log2e).round_ties_even();
    let r = kfma::<HW>(n, k.ln2, t);
    let mut p = kfma::<HW>(r, k.exp_poly[0], k.exp_poly[1]);
    p = kfma::<HW>(r, p, k.exp_poly[2]);
    p = kfma::<HW>(r, p, k.exp_poly[3]);
    p = kfma::<HW>(r, p, k.exp_poly[4]);
    p = kfma::<HW>(r, p, k.half);
    p = kfma::<HW>(r, p, k.one);
    p = kfma::<HW>(r, p, k.one);
    // `(int32(n) + bias) << 23`, with wrapping integer arithmetic so the
    // negative-exponent edge is identical to the reference.
    let scale_bits = ((n as i32).wrapping_add(EXP_BIAS) << 23) as u32;
    p * f32::from_bits(scale_bits)
}

#[inline]
fn sigmoid<const HW: bool>(x: f32, k: &ActConsts) -> f32 {
    let t = (-x).min(k.exp_clamp).max(k.neg_exp_clamp);
    let e = exp_kernel::<HW>(t, k);
    rcp_ps(e + k.one)
}

#[inline]
fn tanh_gate<const HW: bool>(x: f32, k: &ActConsts) -> f32 {
    let t = (x * k.two).min(k.exp_clamp).max(k.neg_exp_clamp);
    let e = exp_kernel::<HW>(t, k);
    (e - k.one) * rcp_ps(e + k.one)
}

#[inline]
fn tanh_cell<const HW: bool>(x: f32, k: &ActConsts) -> f32 {
    let v = x.max(k.neg_four).min(k.four);
    let x2 = v * v;
    let mut p = kfma::<HW>(x2, k.cell_poly[0], k.cell_poly[1]);
    p = kfma::<HW>(x2, p, k.cell_poly[2]);
    p = kfma::<HW>(x2, p, k.cell_poly[3]);
    let q = x2 + k.cell_poly[4];
    let s = v * rcp_ps(p);
    s * kfma::<HW>(x2, q, k.cell_poly[3])
}

/// Apply `[sigmoid, sigmoid, sigmoid, tanh_gate]` to the four gate blocks.
#[inline(always)]
pub(crate) fn lstm_activate<const HW: bool>(y: &mut [f32], h: usize, k: &ActConsts) {
    debug_assert_eq!(y.len(), 4 * h);
    for t in 0..h {
        y[t] = sigmoid::<HW>(y[t], k);
        y[h + t] = sigmoid::<HW>(y[h + t], k);
        y[2 * h + t] = sigmoid::<HW>(y[2 * h + t], k);
        y[3 * h + t] = tanh_gate::<HW>(y[3 * h + t], k);
    }
}

/// One LSTM cell update, elementwise over the hidden axis.
#[inline(always)]
pub(crate) fn lstm_cell<const HW: bool>(
    cell: &mut [f32],
    qh: &mut [i16],
    y: &[f32],
    inv_h: f32,
    clamp: i32,
    k: &ActConsts,
) {
    let h = cell.len();
    debug_assert_eq!(qh.len(), h);
    debug_assert_eq!(y.len(), 4 * h);
    for t in 0..h {
        let i = y[t];
        let o = y[h + t];
        let f = y[2 * h + t];
        let g = y[3 * h + t];
        let c_prev = cell[t];
        let c_new = kfma::<HW>(g, i, f * c_prev);
        cell[t] = c_new;
        let h_new = o * tanh_cell::<HW>(c_new, k);
        qh[t] = crate::simd::quantize_one(h_new, inv_h, clamp);
    }
}

/// `#[target_feature(enable = "fma")]` entry points for the `HW = true`
/// kernels.
///
/// The attribute is what unlocks the performance: inside an `fma` function the
/// compiler is allowed to use the FMA instruction for `mul_add` and to
/// vectorise/contract the surrounding loops (128/256-bit `vfmadd`), while the
/// portable instantiations above stay on [`crate::fma::fma`].  A vector
/// `vfmadd` rounds each lane exactly like the scalar `vfmadd231ss`, so the
/// output bits do not change.
///
/// # Safety
///
/// Every function in this module may only be called on a CPU for which
/// `is_x86_feature_detected!("fma")` is true (see [`fma_available`]).
/// [`crate::simd_scalar_fma::ScalarFmaSimd`] is the only caller: it checks the
/// feature before each call and falls back to the `HW = false` kernels
/// otherwise, and the voice dispatcher never even constructs it on a CPU
/// without `fma`.
#[cfg(target_arch = "x86_64")]
pub mod fma_kernels {
    use super::*;

    /// # Safety
    /// The CPU must support `fma` (and the features it implies: `avx`,
    /// `sse4.1`, `sse4.2`, ...).
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn fma_broadcast(acc: &mut [f32], x: f32, w: &[f32]) {
        super::fma_broadcast::<true>(acc, x, w);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn fma_tiled(acc: &mut [f32], x: &[f32], w: &[f32]) {
        super::fma_tiled::<true>(acc, x, w);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn mul_add_tap(dst: &mut [f32], src: &[f32], k: f32) {
        super::mul_add_tap(dst, src, k);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn conv1x1_1(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
        super::conv1x1_1::<true>(dst, src, w, bias);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn conv1x1_32(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
        super::conv1x1_32::<true>(dst, src, w, bias);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn depthwise_3x3(dst: &mut [f32], src: &[f32], w: &[f32], in_len: usize) {
        super::depthwise_3x3::<true>(dst, src, w, in_len);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn depthwise_1x3s2(dst: &mut [f32], src: &[f32], w: &[f32], pad_left: usize) {
        super::depthwise_1x3s2::<true>(dst, src, w, pad_left);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn maxpool1x3s2(dst: &mut [f32], src: &[f32], in_len: usize) {
        super::maxpool1x3s2(dst, src, in_len);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn gemm_i8(acc: &mut [i32], w: &[i8], zp: &[i8], qx: &[i16], k: usize, n: usize) {
        super::gemm_i8(acc, w, zp, qx, k, n);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn quantize_i16(dst: &mut [i16], src: &[f32], inv: f32, clamp: i32) {
        super::quantize_i16(dst, src, inv, clamp);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn lstm_activate(y: &mut [f32], h: usize, k: &ActConsts) {
        super::lstm_activate::<true>(y, h, k);
    }

    /// # Safety
    /// The CPU must support `fma`.
    #[inline]
    #[target_feature(enable = "fma")]
    pub unsafe fn lstm_cell(
        cell: &mut [f32],
        qh: &mut [i16],
        y: &[f32],
        inv_h: f32,
        clamp: i32,
        k: &ActConsts,
    ) {
        super::lstm_cell::<true>(cell, qh, y, inv_h, clamp, k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scalar FMA instruction must agree with the native correctly-rounded
    /// reference on adversarial values (full 32-bit encodings, cancellation,
    /// subnormals and midpoints).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn hw_fma_matches_native_reference() {
        if !fma_available() {
            eprintln!("fma unavailable, skipping");
            return;
        }
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut bad = 0u64;
        let cases = 2_000_000u64;
        for _ in 0..cases {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let a = f32::from_bits((s >> 32) as u32);
            s = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let b = f32::from_bits((s >> 32) as u32);
            s = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let c = f32::from_bits((s >> 32) as u32);
            let want = a.mul_add(b, c);
            let got = unsafe { hw_fma(a, b, c) };
            if want.to_bits() != got.to_bits() && !(want.is_nan() && got.is_nan()) {
                bad += 1;
            }
        }
        // Exact-cancellation and midpoint patterns.
        let specials = [
            (3.0f32, 0.1f32, -0.3f32),
            (4097.0, 4097.0, 0.0),
            (4097.0, 4097.0, f32::from_bits(1)),
            (f32::MAX, 2.0, -f32::MAX),
            (f32::from_bits(1), f32::from_bits(1), 0.0),
            (-128.0, core::f32::consts::LN_2, 88.723_785),
        ];
        for (a, b, c) in specials {
            if a.mul_add(b, c).to_bits() != unsafe { hw_fma(a, b, c) }.to_bits() {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "scalar FMA instruction diverged on {bad} cases");
    }
}
