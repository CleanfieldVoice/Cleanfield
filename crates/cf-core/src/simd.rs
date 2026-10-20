//! SIMD abstraction and the scalar reference kernels.
//!
//! The scalar implementation *is* the reference: every method has a scalar
//! body whose operation order matches the model contract element by element.
//! A wider backend (AVX2, and later wasm `simd128`) overrides the same trait
//! methods, but only with instruction sequences that perform, for each output
//! element, exactly the same sequence of IEEE-754 operations in exactly the
//! same order.  There is no re-association anywhere: the vector unit only
//! supplies independent lanes.
//!
//! The trait covers the whole layer vocabulary rather than a handful of
//! primitives, so a backend can choose its own blocking.  The integer GEMM
//! is the one place where operation *order* is irrelevant (the proven bound
//! keeps every partial sum well inside `int32`), but the arithmetic is exact
//! integer arithmetic either way.

use crate::fma::fma;
use crate::rcp::rcp_ps;

/// Channel count of the convolutional stack.
const CHANNELS: usize = 32;
/// IEEE-754 single-precision exponent bias (structural, not a model constant).
const EXP_BIAS: i32 = 127;

/// Backend selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar reference.
    Scalar,
    /// x86-64 AVX2 + FMA backend.
    Avx2,
}

impl Backend {
    /// Pick the best backend this CPU can run.
    ///
    /// The AVX2 backend needs both AVX2 (integer/float 256-bit forms, gathers)
    /// and FMA (single-rounding `vfmadd`), so both features are required.
    #[inline]
    pub fn detect() -> Backend {
        if avx2_available() {
            Backend::Avx2
        } else {
            Backend::Scalar
        }
    }

    /// Stable diagnostic name.
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Avx2 => "avx2",
        }
    }

    /// Whether this backend can run on the current CPU.
    #[inline]
    pub fn is_available(self) -> bool {
        match self {
            Backend::Scalar => true,
            Backend::Avx2 => avx2_available(),
        }
    }
}

/// Whether the AVX2 + FMA backend is usable.
#[inline]
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The scalar constants the activation kernels need.
///
/// Everything here is restored from the container's `u32` bit patterns; the
/// struct exists only to keep the `cf-core` kernels independent of the
/// container format.
#[derive(Clone, Copy, Debug)]
pub struct ActConsts {
    /// `log2(e)`, used by the exponential range reduction.
    pub log2e: f32,
    /// `-ln(2)`, the range reduction residual.
    pub ln2: f32,
    /// Degree-8 exponential polynomial, highest power first.
    pub exp_poly: [f32; 5],
    /// `0.5`, the first of the two final polynomial folds.
    pub half: f32,
    /// `1.0`, used by both activations.
    pub one: f32,
    /// `2.0`, the gate-tanh input scale.
    pub two: f32,
    /// `4.0`, cell-tanh upper clamp.
    pub four: f32,
    /// `-4.0`, cell-tanh lower clamp.
    pub neg_four: f32,
    /// Exponential upper clamp.
    pub exp_clamp: f32,
    /// Exponential lower clamp.
    pub neg_exp_clamp: f32,
    /// Cell-tanh Pade coefficients (four numerator, one denominator offset).
    pub cell_poly: [f32; 5],
}

/// The scalar constants the echo model's activation kernels need.
///
/// Like [`ActConsts`], every value is restored from the container's `u32` bit
/// patterns.  The echo activation forms differ from the voice model's (rational
/// `rcpps` sigmoid/tanh for the GRU gates, an exp polynomial for the final
/// sigmoid), so they get their own table.
#[derive(Clone, Copy, Debug)]
pub struct EchoActConsts {
    /// `1.0`.
    pub one: f32,
    /// `0.5`.
    pub half: f32,
    /// `8.0`, GRU sigmoid upper clamp.
    pub eight: f32,
    /// `-8.0`, GRU sigmoid lower clamp.
    pub neg_eight: f32,
    /// `4.0`, GRU tanh upper clamp.
    pub four: f32,
    /// `-4.0`, GRU tanh lower clamp.
    pub neg_four: f32,
    /// Rational sigmoid coefficients `c0..c6`.
    pub sig: [f32; 7],
    /// Rational tanh coefficients `c0..c4`.
    pub tanh: [f32; 5],
    /// Final-sigmoid upper clamp.
    pub sig88_clamp: f32,
    /// Final-sigmoid lower clamp (negative).
    pub sig88_neg_clamp: f32,
    /// `log2(e)` for the final-sigmoid exponential.
    pub exp_log2e: f32,
    /// `-ln(2)` for the final-sigmoid exponential.
    pub exp_ln2: f32,
    /// Degree-8 exponential polynomial, highest power first.
    pub exp_c: [f32; 5],
}

/// Operations the layer kernels need.
pub trait Simd: Copy {
    /// Backend name for diagnostics.
    fn name(&self) -> &'static str;

    /// `acc[j] = fma(x, w[j], acc[j])` - a scalar broadcast over a row.
    fn fma_broadcast(acc: &mut [f32], x: f32, w: &[f32]);

    /// `acc[i] = fma(x[i], w[i % channels], acc[i])` - one tap applied to a
    /// run of positions with a per-channel weight vector.
    fn fma_tiled(acc: &mut [f32], x: &[f32], w: &[f32]);

    /// `dst[i] = dst[i] + (src[i] * k)` with two roundings (not an FMA).
    fn mul_add_tap(dst: &mut [f32], src: &[f32], k: f32);

    /// 1x1 convolution, one input channel: bias, one FMA, fused ReLU.
    fn conv1x1_1(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]);

    /// 1x1 convolution, 32 input channels -> 32 output channels, fused ReLU.
    fn conv1x1_32(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]);

    /// Depthwise 3x3 along the length axis (`in_len` -> `in_len - 2`), taps in
    /// `[kh][kw]` order, fused multiply-add.
    fn depthwise_3x3(dst: &mut [f32], src: &[f32], w: &[f32], in_len: usize);

    /// Depthwise 1x3 stride 2 with left padding, taps in tap order.
    fn depthwise_1x3s2(dst: &mut [f32], src: &[f32], w: &[f32], pad_left: usize);

    /// Max pool 1x3 stride 2 along the length axis, 32 channels per position.
    fn maxpool1x3s2(dst: &mut [f32], src: &[f32], in_len: usize);

    /// `acc[j] = sum_k qx[k] * (w[k][j] - zp[j])` in `int32`, `w` input major.
    fn gemm_i8(acc: &mut [i32], w: &[i8], zp: &[i8], qx: &[i16], k: usize, n: usize);

    /// `dst = clamp(rne(src * inv))` as `int16`.
    fn quantize_i16(dst: &mut [i16], src: &[f32], inv: f32, clamp: i32);

    /// Apply `[sigmoid, sigmoid, sigmoid, tanh_gate]` to the four gate blocks
    /// of `y` in place.
    fn lstm_activate(y: &mut [f32], h: usize, k: &ActConsts);

    /// One LSTM cell update, elementwise over the hidden axis.
    fn lstm_cell(
        cell: &mut [f32],
        qh: &mut [i16],
        y: &[f32],
        inv_h: f32,
        clamp: i32,
        k: &ActConsts,
    );

    /// `dst[j] = f32(acc[j]) * sp[j] + bias[j]` with two roundings
    /// (multiply, then add - not an FMA).
    ///
    /// `acc` may be the larger shared arena; only `dst.len()` entries are read.
    fn scale_i32_add(dst: &mut [f32], acc: &[i32], sp: &[f32], bias: &[f32]) {
        debug_assert!(acc.len() >= dst.len());
        debug_assert_eq!(dst.len(), sp.len());
        debug_assert_eq!(dst.len(), bias.len());
        for j in 0..dst.len() {
            dst[j] = crate::fma::mul_add2(acc[j] as f32, sp[j], bias[j]);
        }
    }

    /// `dst[j] = f32(acc[j]) * sp[j] + dst[j]` with two roundings.
    fn scale_i32_add_inplace(dst: &mut [f32], acc: &[i32], sp: &[f32]) {
        debug_assert!(acc.len() >= dst.len());
        debug_assert_eq!(dst.len(), sp.len());
        for j in 0..dst.len() {
            dst[j] = crate::fma::mul_add2(acc[j] as f32, sp[j], dst[j]);
        }
    }

    /// [`Simd::scale_i32_add`] followed by `if !(y > 0.0) { y = 0.0 }`.
    fn scale_i32_add_relu(dst: &mut [f32], acc: &[i32], sp: &[f32], bias: &[f32]) {
        debug_assert!(acc.len() >= dst.len());
        debug_assert_eq!(dst.len(), sp.len());
        debug_assert_eq!(dst.len(), bias.len());
        for j in 0..dst.len() {
            let mut y = crate::fma::mul_add2(acc[j] as f32, sp[j], bias[j]);
            if !(y > 0.0) {
                y = 0.0;
            }
            dst[j] = y;
        }
    }

    /// `dst[j] = a[j] * b[j] + c[j]` with two roundings (multiply, then add).
    fn mul_add2_slices(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
        debug_assert_eq!(dst.len(), a.len());
        debug_assert_eq!(dst.len(), b.len());
        debug_assert_eq!(dst.len(), c.len());
        for j in 0..dst.len() {
            dst[j] = crate::fma::mul_add2(a[j], b[j], c[j]);
        }
    }

    // -----------------------------------------------------------------------
    // echo-model kernels
    //
    // These are part of the same trait so the echo graph dispatches exactly
    // like the voice graph.  The scalar bodies below are the reference, and
    // every weight slice keeps its container layout:
    //
    // * `echo_conv1x1`:    `w[ic][oc]`;
    // * `echo_depthwise`:  `w[channel][tap]`;
    // * `echo_deconv`:     `w[channel][tap]` (five taps);
    // * `echo_conv3`:      `w[oc][ic][tap]`;
    // * `echo_conv3_out1`: `w[ic][tap]` (one output channel);
    // * `echo_mm33`:       `w[i * 33 + j]`.
    // -----------------------------------------------------------------------

    /// Echo 1x1 convolution: bias prefill, one FMA per input channel ascending.
    fn echo_conv1x1(
        dst: &mut [f32],
        src: &[f32],
        w: &[f32],
        bias: &[f32],
        cin: usize,
        cout: usize,
    ) -> bool {
        debug_assert!(cin > 0 && cout > 0);
        debug_assert_eq!(src.len() % cin, 0);
        let positions = src.len() / cin;
        debug_assert!(dst.len() >= positions * cout);
        debug_assert_eq!(w.len(), cin * cout);
        debug_assert!(bias.len() >= cout);
        for pos in 0..positions {
            let row = &mut dst[pos * cout..(pos + 1) * cout];
            row.copy_from_slice(&bias[..cout]);
            for ic in 0..cin {
                let x = src[pos * cin + ic];
                let wr = &w[ic * cout..(ic + 1) * cout];
                for oc in 0..cout {
                    row[oc] = fma(x, wr[oc], row[oc]);
                }
            }
        }
        true
    }

    /// Echo depthwise dilated 5x1 (`h - 4 + 2t`) or adjacent 3x1 (`h - 1 + t`).
    ///
    /// `fma_mode == false` is multiply-then-add (the 12-channel layers).  An
    /// out-of-range tap contributes an exact zero operand.
    fn echo_depthwise(
        dst: &mut [f32],
        src: &[f32],
        w: &[f32],
        bias: &[f32],
        channels: usize,
        taps: usize,
        fma_mode: bool,
        bins: usize,
    ) -> bool {
        debug_assert_eq!(src.len(), bins * channels);
        debug_assert!(dst.len() >= bins * channels);
        debug_assert_eq!(w.len(), channels * taps);
        debug_assert!(bias.len() >= channels);
        for ch in 0..channels {
            for h in 0..bins {
                let mut acc = bias[ch];
                for t in 0..taps {
                    let i = if taps == 5 {
                        h as isize - 4 + 2 * t as isize
                    } else {
                        h as isize - 1 + t as isize
                    };
                    let xv = if i >= 0 && (i as usize) < bins {
                        src[i as usize * channels + ch]
                    } else {
                        0.0
                    };
                    let wt = w[ch * taps + t];
                    acc = if fma_mode {
                        fma(xv, wt, acc)
                    } else {
                        acc + (xv * wt)
                    };
                }
                dst[h * channels + ch] = acc;
            }
        }
        true
    }

    /// Echo max pool 5x1 stride 2: out-of-range taps dropped, `-inf` start,
    /// later operand on ties.
    fn echo_maxpool5(dst: &mut [f32], src: &[f32], channels: usize, bins: usize) -> bool {
        let out_bins = (bins - 1) / 2 + 1;
        debug_assert_eq!(src.len(), bins * channels);
        debug_assert!(dst.len() >= out_bins * channels);
        for p in 0..out_bins {
            for ch in 0..channels {
                let mut m = f32::NEG_INFINITY;
                for t in 0..5 {
                    let i = (2 * p) as isize - 2 + t as isize;
                    if i >= 0 && (i as usize) < bins {
                        let v = src[i as usize * channels + ch];
                        m = if m > v { m } else { v };
                    }
                }
                dst[p * channels + ch] = m;
            }
        }
        true
    }

    /// Echo int8 GEMM with an `i64` accumulator (the `K = 2112` GRU input).
    fn echo_gemm_i64(acc: &mut [i64], w: &[i8], zp: &[i8], qx: &[i16], k: usize, n: usize) -> bool {
        debug_assert!(acc.len() >= n);
        debug_assert_eq!(w.len(), k * n);
        debug_assert!(zp.len() >= n);
        debug_assert!(qx.len() >= k);
        acc[..n].fill(0);
        for kk in 0..k {
            let q = qx[kk] as i64;
            if q == 0 {
                continue;
            }
            let row = &w[kk * n..kk * n + n];
            for j in 0..n {
                acc[j] += q * (row[j] as i64 - zp[j] as i64);
            }
        }
        true
    }

    /// Echo depthwise deconvolution: only even output rows are computed, taps
    /// descending `k = 4..0`, `out[2j] = bias + sum fma(x[j+2-k], w[k])`.
    /// The odd output rows are the bias.
    fn echo_deconv(
        dst: &mut [f32],
        src: &[f32],
        w: &[f32],
        bias: &[f32],
        hin: usize,
        channels: usize,
    ) -> bool {
        let hout = 2 * hin - 1;
        debug_assert!(src.len() >= hin * channels);
        debug_assert!(dst.len() >= hout * channels);
        debug_assert_eq!(w.len(), channels * 5);
        debug_assert!(bias.len() >= channels);
        for ch in 0..channels {
            for j in 0..hin - 1 {
                dst[(2 * j + 1) * channels + ch] = bias[ch];
            }
            for j in 0..hin {
                let mut acc = bias[ch];
                for t in (0..5).rev() {
                    let i = j as isize + 2 - t as isize;
                    if i >= 0 && (i as usize) < hin {
                        acc = fma(src[i as usize * channels + ch], w[ch * 5 + t], acc);
                    }
                }
                dst[(2 * j) * channels + ch] = acc;
            }
        }
        true
    }

    /// Echo 3x1 decoder convolution, weights `[oc][ic][tap]`, `ic` outer and
    /// `k` inner, bias prefilled, FMA, optional leaky-ReLU.
    fn echo_conv3(
        dst: &mut [f32],
        src: &[f32],
        w: &[f32],
        bias: &[f32],
        bins: usize,
        cin: usize,
        cout: usize,
        leaky: bool,
        slope: f32,
    ) -> bool {
        debug_assert_eq!(src.len(), bins * cin);
        debug_assert!(dst.len() >= bins * cout);
        debug_assert_eq!(w.len(), cout * cin * 3);
        debug_assert!(bias.len() >= cout);
        for oc in 0..cout {
            for h in 0..bins {
                let mut acc = bias[oc];
                for ic in 0..cin {
                    for t in 0..3 {
                        let i = h as isize - 1 + t as isize;
                        if i >= 0 && (i as usize) < bins {
                            acc = fma(
                                src[i as usize * cin + ic],
                                w[oc * cin * 3 + ic * 3 + t],
                                acc,
                            );
                        }
                    }
                }
                dst[h * cout + oc] = if leaky && !(acc > 0.0) {
                    acc * slope
                } else {
                    acc
                };
            }
        }
        true
    }

    /// Echo layer 82: outC == 1, tap-0/1/2 nested FMA, bias-first horizontal
    /// sum in channel order.  Weights are tap-major `[tap][ic]`.
    fn echo_conv3_out1(
        dst: &mut [f32],
        src: &[f32],
        w: &[f32],
        bias: &[f32],
        bins: usize,
        cin: usize,
    ) -> bool {
        debug_assert_eq!(src.len(), bins * cin);
        debug_assert!(dst.len() >= bins);
        debug_assert_eq!(w.len(), 3 * cin);
        debug_assert!(!bias.is_empty());
        let mut acc = [0.0f32; 16];
        debug_assert!(cin <= acc.len());
        for h in 0..bins {
            for c in 0..cin {
                let x0 = if h > 0 { src[(h - 1) * cin + c] } else { 0.0 };
                let x1 = src[h * cin + c];
                let x2 = if h + 1 < bins {
                    src[(h + 1) * cin + c]
                } else {
                    0.0
                };
                acc[c] = fma(x2, w[2 * cin + c], fma(x1, w[cin + c], fma(x0, w[c], 0.0)));
            }
            let mut o = bias[0];
            for c in 0..cin {
                o += acc[c];
            }
            dst[h] = o;
        }
        true
    }

    /// Echo 33x33 MatMul: `out[j*64 + c] = sum_i fma(x[i*64+c], W[i][j])`,
    /// input positions ascending.
    fn echo_mm33(dst: &mut [f32], src: &[f32], w: &[f32]) -> bool {
        debug_assert_eq!(dst.len(), 33 * 64);
        debug_assert_eq!(src.len(), 33 * 64);
        debug_assert_eq!(w.len(), 33 * 33);
        for j in 0..33 {
            for ch in 0..64 {
                let mut acc = 0.0f32;
                for i in 0..33 {
                    acc = fma(src[i * 64 + ch], w[i * 33 + j], acc);
                }
                dst[j * 64 + ch] = acc;
            }
        }
        true
    }

    /// Echo per-frame input quantisation.  Returns `A = f32(maxabs / 8192)`.
    /// The vector body rounds ties to even; the `n % 8` tail rounds half away
    /// from zero.  The all-zero input branch reproduces the reference's
    /// `0 * inf -> NaN -> int32::MIN -> clamp` result.
    fn echo_quantize_input(dst: &mut [i16], src: &[f32], quant_scale: f32) -> f32 {
        let n = src.len();
        assert!(dst.len() >= n);
        let mx = max_abs(src);
        let a = mx / quant_scale;
        let clamp = quant_scale as i32;
        if mx == 0.0 {
            for d in dst[..n].iter_mut() {
                *d = -clamp as i16;
            }
            return a;
        }
        let inv = quant_scale / mx;
        let nvec = if n >= 8 {
            (8 * (((n - 8) >> 3) + 1)).min(n)
        } else {
            0
        };
        for i in 0..nvec {
            dst[i] = quant_rne(src[i] * inv, clamp);
        }
        for i in nvec..n {
            dst[i] = quant_half_away(src[i] * inv, clamp);
        }
        a
    }

    /// Echo hidden-state quantisation.  Returns `A_h = maxabs / 127` (or
    /// `/ 32767` when `maxabs > 1`).
    fn echo_quantize_hidden(
        dst: &mut [i16],
        src: &[f32],
        hidden_scale: f32,
        hidden_scale_wide: f32,
    ) -> f32 {
        let mx = max_abs(src);
        let c = if mx <= 1.0 {
            hidden_scale
        } else {
            hidden_scale_wide
        };
        let a = mx / c;
        let inv = if a != 0.0 { 1.0 / a } else { 0.0 };
        let lim = c as i32;
        for (d, &v) in dst.iter_mut().zip(src.iter()) {
            *d = ((v * inv).round_ties_even() as i32).clamp(-lim, lim) as i16;
        }
        a
    }

    /// Echo GRU tail: gate sums, `[z, r]` sigmoids, `n` tanh and the hidden
    /// update with two multiplies and one add (no FMA).
    fn echo_gru_tail(yih: &mut [f32], yhh: &[f32], h: &mut [f32], k: &EchoActConsts) -> bool {
        let hidden = h.len();
        debug_assert_eq!(yih.len(), 3 * hidden);
        debug_assert_eq!(yhh.len(), 3 * hidden);
        for j in 0..2 * hidden {
            yih[j] += yhh[j];
        }
        for j in 0..2 * hidden {
            yih[j] = echo_sigmoid_gru(yih[j], k);
        }
        for j in 0..hidden {
            let t = yhh[2 * hidden + j] * yih[hidden + j] + yih[2 * hidden + j];
            yih[2 * hidden + j] = echo_tanh_gru(t, k);
        }
        for j in 0..hidden {
            let z = yih[j];
            let n = yih[2 * hidden + j];
            let a = (k.one - z) * n;
            let b = z * h[j];
            h[j] = a + b;
        }
        true
    }

    /// Echo final sigmoid: exp polynomial + reciprocal table for the
    /// `8 * floor(n / 8)` body, scalar `1 / (f32(exp(-x)) + 1)` for the tail.
    fn echo_sigmoid88(dst: &mut [f32], src: &[f32], k: &EchoActConsts) -> bool {
        let n = src.len();
        debug_assert!(dst.len() >= n);
        let nvec = n - (n % 8);
        for i in 0..nvec {
            dst[i] = echo_sigmoid_vector(src[i], k);
        }
        for i in nvec..n {
            dst[i] = crate::dexp::sigmoid_tail(src[i]);
        }
        true
    }
}

/// Scalar reference backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarSimd;

impl Simd for ScalarSimd {
    #[inline]
    fn name(&self) -> &'static str {
        "scalar"
    }

    #[inline]
    fn fma_broadcast(acc: &mut [f32], x: f32, w: &[f32]) {
        debug_assert_eq!(acc.len(), w.len());
        for (a, wi) in acc.iter_mut().zip(w.iter()) {
            *a = fma(x, *wi, *a);
        }
    }

    #[inline]
    fn fma_tiled(acc: &mut [f32], x: &[f32], w: &[f32]) {
        debug_assert_eq!(acc.len(), x.len());
        let channels = w.len();
        debug_assert!(channels > 0 && acc.len().is_multiple_of(channels));
        for i in 0..acc.len() {
            acc[i] = fma(x[i], w[i % channels], acc[i]);
        }
    }

    #[inline]
    fn mul_add_tap(dst: &mut [f32], src: &[f32], k: f32) {
        debug_assert_eq!(dst.len(), src.len());
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            let p = *s * k; // separate rounding, must not be contracted
            *d += p;
        }
    }

    fn conv1x1_1(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
        let positions = dst.len() / CHANNELS;
        debug_assert_eq!(dst.len(), positions * CHANNELS);
        debug_assert_eq!(src.len(), positions);
        debug_assert_eq!(w.len(), CHANNELS);
        debug_assert_eq!(bias.len(), CHANNELS);
        for pos in 0..positions {
            let row = &mut dst[pos * CHANNELS..(pos + 1) * CHANNELS];
            row.copy_from_slice(bias);
            let x = src[pos];
            for oc in 0..CHANNELS {
                row[oc] = fma(x, w[oc], row[oc]);
            }
            relu(row);
        }
    }

    fn conv1x1_32(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
        let positions = dst.len() / CHANNELS;
        debug_assert_eq!(dst.len(), positions * CHANNELS);
        debug_assert_eq!(src.len(), positions * CHANNELS);
        debug_assert_eq!(w.len(), CHANNELS * CHANNELS);
        debug_assert_eq!(bias.len(), CHANNELS);
        for pos in 0..positions {
            let row = &mut dst[pos * CHANNELS..(pos + 1) * CHANNELS];
            row.copy_from_slice(bias);
            for ic in 0..CHANNELS {
                let x = src[pos * CHANNELS + ic];
                let wrow = &w[ic * CHANNELS..(ic + 1) * CHANNELS];
                for oc in 0..CHANNELS {
                    row[oc] = fma(x, wrow[oc], row[oc]);
                }
            }
            relu(row);
        }
    }

    fn depthwise_3x3(dst: &mut [f32], src: &[f32], w: &[f32], in_len: usize) {
        let out_len = in_len - 2;
        debug_assert_eq!(src.len(), 3 * in_len * CHANNELS);
        debug_assert_eq!(dst.len(), out_len * CHANNELS);
        debug_assert_eq!(w.len(), 9 * CHANNELS);
        dst.fill(0.0);
        for kh in 0..3 {
            for kw in 0..3 {
                let base = (kh * in_len + kw) * CHANNELS;
                let src = &src[base..base + out_len * CHANNELS];
                let tap = &w[(kh * 3 + kw) * CHANNELS..(kh * 3 + kw + 1) * CHANNELS];
                for i in 0..out_len * CHANNELS {
                    dst[i] = fma(src[i], tap[i % CHANNELS], dst[i]);
                }
            }
        }
    }

    fn depthwise_1x3s2(dst: &mut [f32], src: &[f32], w: &[f32], pad_left: usize) {
        let in_len = src.len() / CHANNELS;
        let out_len = in_len.div_ceil(2);
        debug_assert_eq!(dst.len(), out_len * CHANNELS);
        debug_assert_eq!(w.len(), 3 * CHANNELS);
        dst.fill(0.0);
        for t in 0..3 {
            let tap = &w[t * CHANNELS..(t + 1) * CHANNELS];
            for p in 0..out_len {
                let idx = (2 * p + t) as isize - pad_left as isize;
                if idx < 0 || idx as usize >= in_len {
                    continue;
                }
                let s = &src[idx as usize * CHANNELS..][..CHANNELS];
                let d = &mut dst[p * CHANNELS..][..CHANNELS];
                for ch in 0..CHANNELS {
                    d[ch] = fma(s[ch], tap[ch], d[ch]);
                }
            }
        }
    }

    fn maxpool1x3s2(dst: &mut [f32], src: &[f32], in_len: usize) {
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
                    // Two sequential maxima; ties select the later operand,
                    // which is what the reference element-wise maximum does.
                    let m = if a > b { a } else { b };
                    let m = if m > c { m } else { c };
                    dst[(seg * out_len + p) * CHANNELS + ch] = m;
                }
            }
        }
    }

    fn gemm_i8(acc: &mut [i32], w: &[i8], zp: &[i8], qx: &[i16], k: usize, n: usize) {
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

    fn quantize_i16(dst: &mut [i16], src: &[f32], inv: f32, clamp: i32) {
        debug_assert_eq!(dst.len(), src.len());
        for (d, &v) in dst.iter_mut().zip(src.iter()) {
            *d = quantize_one(v, inv, clamp);
        }
    }

    fn lstm_activate(y: &mut [f32], h: usize, k: &ActConsts) {
        debug_assert_eq!(y.len(), 4 * h);
        for t in 0..h {
            y[t] = sigmoid(y[t], k);
            y[h + t] = sigmoid(y[h + t], k);
            y[2 * h + t] = sigmoid(y[2 * h + t], k);
            y[3 * h + t] = tanh_gate(y[3 * h + t], k);
        }
    }

    fn lstm_cell(
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
            let c_new = fma(g, i, f * c_prev);
            cell[t] = c_new;
            let h_new = o * tanh_cell(c_new, k);
            qh[t] = quantize_one(h_new, inv_h, clamp);
        }
    }
}

/// `np.maximum(v, 0)` for a finite `v`: ties select `+0`, so `-0.0` becomes
/// `+0.0`.  (NaN does not occur in the verified sequences; the reference
/// maximum would propagate it and this branch would not.)
#[inline]
fn relu(row: &mut [f32]) {
    for v in row.iter_mut() {
        if !(*v > 0.0) {
            *v = 0.0;
        }
    }
}

#[inline]
pub(crate) fn quantize_one(v: f32, inv: f32, clamp: i32) -> i16 {
    let q = (v * inv).round_ties_even();
    (q as i32).clamp(-clamp, clamp) as i16
}

#[inline]
pub(crate) fn max_abs(v: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for &x in v {
        let a = x.abs();
        if a > m {
            m = a;
        }
    }
    m
}

#[inline]
pub(crate) fn quant_rne(v: f32, clamp: i32) -> i16 {
    (v.round_ties_even() as i32).clamp(-clamp, clamp) as i16
}

#[inline]
pub(crate) fn quant_half_away(v: f32, clamp: i32) -> i16 {
    let q = if v > 0.0 {
        (v + 0.5).floor()
    } else {
        (v - 0.5).ceil()
    };
    (q as i32).clamp(-clamp, clamp) as i16
}

/// Echo GRU sigmoid: rational `t * rcp(q)` form, clamp +-8.
#[inline]
pub(crate) fn echo_sigmoid_gru(x: f32, k: &EchoActConsts) -> f32 {
    let t = x.max(k.neg_eight).min(k.eight);
    let x2 = t * t;
    let mut q = fma(x2, k.sig[0], k.sig[1]);
    q = fma(x2, q, k.sig[2]);
    q = fma(x2, q, k.sig[3]);
    let s = t * rcp_ps(q);
    let mut p = fma(x2, k.sig[4], k.sig[5]);
    p = fma(x2, p, k.sig[6]);
    fma(s, p, k.half)
}

/// Echo GRU tanh: rational `v * rcp(p)` form, clamp +-4.
#[inline]
pub(crate) fn echo_tanh_gru(x: f32, k: &EchoActConsts) -> f32 {
    let v = x.max(k.neg_four).min(k.four);
    let x2 = v * v;
    let mut p = fma(x2, k.tanh[0], k.tanh[1]);
    p = fma(x2, p, k.tanh[2]);
    p = fma(x2, p, k.tanh[3]);
    let s = v * rcp_ps(p);
    let q = x2 + k.tanh[4];
    s * fma(x2, q, k.tanh[3])
}

/// Echo final-sigmoid vector form: exp polynomial + reciprocal table.
#[inline]
pub(crate) fn echo_sigmoid_vector(x: f32, k: &EchoActConsts) -> f32 {
    let t = (-x).max(k.sig88_neg_clamp).min(k.sig88_clamp);
    let n = (t * k.exp_log2e).round_ties_even();
    let r = fma(n, k.exp_ln2, t);
    let mut p = fma(r, k.exp_c[0], k.exp_c[1]);
    p = fma(r, p, k.exp_c[2]);
    p = fma(r, p, k.exp_c[3]);
    p = fma(r, p, k.exp_c[4]);
    p = fma(r, p, k.half);
    p = fma(r, p, k.one);
    p = fma(r, p, k.one);
    let scale = f32::from_bits((((n as i32).wrapping_add(127)) << 23) as u32);
    let e = p * scale;
    rcp_ps(e + k.one)
}

/// Shared range reduction and degree-8 polynomial used by both gate
/// activations: `n = rne(t * log2e)`, `r = fma(n, -ln2, t)`, then the
/// polynomial in `r`, scaled by `2^n` built from integer exponent bits.
#[inline]
pub(crate) fn exp_kernel(t: f32, k: &ActConsts) -> f32 {
    let n = (t * k.log2e).round_ties_even();
    let r = fma(n, k.ln2, t);
    let mut p = fma(r, k.exp_poly[0], k.exp_poly[1]);
    p = fma(r, p, k.exp_poly[2]);
    p = fma(r, p, k.exp_poly[3]);
    p = fma(r, p, k.exp_poly[4]);
    p = fma(r, p, k.half);
    p = fma(r, p, k.one);
    p = fma(r, p, k.one);
    // `(int32(n) + bias) << 23` is the reference's exponent construction; it
    // is evaluated with wrapping integer arithmetic so the negative-exponent
    // edge is identical.
    let scale_bits = ((n as i32).wrapping_add(EXP_BIAS) << 23) as u32;
    p * f32::from_bits(scale_bits)
}

/// `sigmoid(x) = rcp(exp(clamp(-x)) + 1)`.
#[inline]
pub(crate) fn sigmoid(x: f32, k: &ActConsts) -> f32 {
    let t = (-x).min(k.exp_clamp).max(k.neg_exp_clamp);
    let e = exp_kernel(t, k);
    rcp_ps(e + k.one)
}

/// Gate `tanh`: `(exp(clamp(2x)) - 1) * rcp(exp(clamp(2x)) + 1)`.
#[inline]
pub(crate) fn tanh_gate(x: f32, k: &ActConsts) -> f32 {
    let t = (x * k.two).min(k.exp_clamp).max(k.neg_exp_clamp);
    let e = exp_kernel(t, k);
    (e - k.one) * rcp_ps(e + k.one)
}

/// Cell `tanh`: clamp +-4, then the fixed Pade form with a reciprocal.
#[inline]
pub(crate) fn tanh_cell(x: f32, k: &ActConsts) -> f32 {
    let v = x.max(k.neg_four).min(k.four);
    let x2 = v * v;
    let mut p = fma(x2, k.cell_poly[0], k.cell_poly[1]);
    p = fma(x2, p, k.cell_poly[2]);
    p = fma(x2, p, k.cell_poly[3]);
    let q = x2 + k.cell_poly[4];
    let s = v * rcp_ps(p);
    s * fma(x2, q, k.cell_poly[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_backend_round_trips() {
        let mut acc = [1.0f32, 2.0, 3.0];
        ScalarSimd::fma_broadcast(&mut acc, 2.0, &[0.5, 0.25, 0.125]);
        assert_eq!(acc, [2.0, 2.5, 3.25]);

        let mut acc = [0.0f32; 4];
        ScalarSimd::fma_tiled(&mut acc, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0]);
        assert_eq!(acc, [10.0, 40.0, 30.0, 80.0]);

        let mut dst = [1.0f32, 1.0, 1.0];
        ScalarSimd::mul_add_tap(&mut dst, &[2.0, 3.0, 4.0], 0.5);
        assert_eq!(dst, [2.0, 2.5, 3.0]);
    }

    #[test]
    fn scalar_kernels_have_reference_shape() {
        let mut dst = vec![0.0f32; 2 * CHANNELS];
        ScalarSimd::conv1x1_32(&mut dst, &[1.0; 2 * CHANNELS], &[0.0; 1024], &[1.0; 32]);
        assert!(dst.iter().all(|&v| v == 1.0));

        let mut dst = vec![0.0f32; 2 * CHANNELS];
        ScalarSimd::conv1x1_32(&mut dst, &[-1.0; 2 * CHANNELS], &[1.0; 1024], &[0.0; 32]);
        assert!(dst.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn backend_names_are_stable() {
        assert_eq!(Backend::Scalar.name(), "scalar");
        assert_eq!(Backend::Avx2.name(), "avx2");
        assert!(Backend::Scalar.is_available());
    }
}
