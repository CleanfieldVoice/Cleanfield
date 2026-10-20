//! AVX2 + FMA backend.
//!
//! Every kernel here is a lane-wise restructuring of the scalar reference in
//! [`crate::simd`].  No operation is re-associated: the float kernels perform
//! the same sequence of FMA / multiply / add per output element, and the
//! integer GEMM relies only on the proven `int32` bound for its accumulator.
//!
//! The module is compiled only for `x86_64`; the runtime selector in
//! [`crate::simd::Backend::detect`] guarantees both AVX2 and FMA are present
//! before any of these functions is reachable.

use std::arch::x86_64::*;

use crate::rcp::RCP_TABLE_U32;
use crate::simd::{
    quantize_one, sigmoid, tanh_cell, tanh_gate, ActConsts, EchoActConsts, ScalarSimd, Simd,
};

/// AVX2 + FMA backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct Avx2Simd;

const CHANNELS: usize = 32;
const EXP_BIAS: i32 = 127;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn zero() -> __m256 {
    _mm256_setzero_ps()
}

// ---------------------------------------------------------------------------
// float kernels
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx2,fma")]
fn conv1x1_1_avx2(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
    let positions = dst.len() / CHANNELS;
    debug_assert_eq!(dst.len(), positions * CHANNELS);
    debug_assert_eq!(src.len(), positions);
    debug_assert_eq!(w.len(), CHANNELS);
    debug_assert_eq!(bias.len(), CHANNELS);
    let z = zero();
    unsafe {
        let bp = bias.as_ptr();
        let b0 = _mm256_loadu_ps(bp);
        let b1 = _mm256_loadu_ps(bp.add(8));
        let b2 = _mm256_loadu_ps(bp.add(16));
        let b3 = _mm256_loadu_ps(bp.add(24));
        let w0 = _mm256_loadu_ps(w.as_ptr());
        let w1 = _mm256_loadu_ps(w.as_ptr().add(8));
        let w2 = _mm256_loadu_ps(w.as_ptr().add(16));
        let w3 = _mm256_loadu_ps(w.as_ptr().add(24));
        for pos in 0..positions {
            let x = _mm256_set1_ps(*src.get_unchecked(pos));
            let a0 = _mm256_max_ps(_mm256_fmadd_ps(x, w0, b0), z);
            let a1 = _mm256_max_ps(_mm256_fmadd_ps(x, w1, b1), z);
            let a2 = _mm256_max_ps(_mm256_fmadd_ps(x, w2, b2), z);
            let a3 = _mm256_max_ps(_mm256_fmadd_ps(x, w3, b3), z);
            let d = dst.as_mut_ptr().add(pos * CHANNELS);
            _mm256_storeu_ps(d, a0);
            _mm256_storeu_ps(d.add(8), a1);
            _mm256_storeu_ps(d.add(16), a2);
            _mm256_storeu_ps(d.add(24), a3);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn conv1x1_32_avx2(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
    let positions = dst.len() / CHANNELS;
    debug_assert_eq!(dst.len(), positions * CHANNELS);
    debug_assert_eq!(src.len(), positions * CHANNELS);
    debug_assert_eq!(w.len(), CHANNELS * CHANNELS);
    debug_assert_eq!(bias.len(), CHANNELS);
    let z = zero();
    unsafe {
        let bp = bias.as_ptr();
        let b0 = _mm256_loadu_ps(bp);
        let b1 = _mm256_loadu_ps(bp.add(8));
        let b2 = _mm256_loadu_ps(bp.add(16));
        let b3 = _mm256_loadu_ps(bp.add(24));
        let wp = w.as_ptr();
        // Two positions per iteration: eight independent accumulator chains
        // instead of four.  Each output keeps the reference's sequence (bias
        // prefill, then one FMA per input channel in ascending order); only
        // the interleaving of two independent positions changes, and the
        // weight row is loaded once for both.
        let mut pos = 0usize;
        while pos + 2 <= positions {
            let sp0 = src.as_ptr().add(pos * CHANNELS);
            let sp1 = sp0.add(CHANNELS);
            let mut a0 = b0;
            let mut a1 = b1;
            let mut a2 = b2;
            let mut a3 = b3;
            let mut c0 = b0;
            let mut c1 = b1;
            let mut c2 = b2;
            let mut c3 = b3;
            for ic in 0..CHANNELS {
                let row = wp.add(ic * CHANNELS);
                let x = _mm256_set1_ps(*sp0.add(ic));
                let y = _mm256_set1_ps(*sp1.add(ic));
                let w0 = _mm256_loadu_ps(row);
                a0 = _mm256_fmadd_ps(x, w0, a0);
                c0 = _mm256_fmadd_ps(y, w0, c0);
                let w1 = _mm256_loadu_ps(row.add(8));
                a1 = _mm256_fmadd_ps(x, w1, a1);
                c1 = _mm256_fmadd_ps(y, w1, c1);
                let w2 = _mm256_loadu_ps(row.add(16));
                a2 = _mm256_fmadd_ps(x, w2, a2);
                c2 = _mm256_fmadd_ps(y, w2, c2);
                let w3 = _mm256_loadu_ps(row.add(24));
                a3 = _mm256_fmadd_ps(x, w3, a3);
                c3 = _mm256_fmadd_ps(y, w3, c3);
            }
            let d0 = dst.as_mut_ptr().add(pos * CHANNELS);
            _mm256_storeu_ps(d0, _mm256_max_ps(a0, z));
            _mm256_storeu_ps(d0.add(8), _mm256_max_ps(a1, z));
            _mm256_storeu_ps(d0.add(16), _mm256_max_ps(a2, z));
            _mm256_storeu_ps(d0.add(24), _mm256_max_ps(a3, z));
            let d1 = d0.add(CHANNELS);
            _mm256_storeu_ps(d1, _mm256_max_ps(c0, z));
            _mm256_storeu_ps(d1.add(8), _mm256_max_ps(c1, z));
            _mm256_storeu_ps(d1.add(16), _mm256_max_ps(c2, z));
            _mm256_storeu_ps(d1.add(24), _mm256_max_ps(c3, z));
            pos += 2;
        }
        if pos < positions {
            let sp = src.as_ptr().add(pos * CHANNELS);
            let mut a0 = b0;
            let mut a1 = b1;
            let mut a2 = b2;
            let mut a3 = b3;
            for ic in 0..CHANNELS {
                let x = _mm256_set1_ps(*sp.add(ic));
                let row = wp.add(ic * CHANNELS);
                a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(row), a0);
                a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(row.add(8)), a1);
                a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(row.add(16)), a2);
                a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(row.add(24)), a3);
            }
            let d = dst.as_mut_ptr().add(pos * CHANNELS);
            _mm256_storeu_ps(d, _mm256_max_ps(a0, z));
            _mm256_storeu_ps(d.add(8), _mm256_max_ps(a1, z));
            _mm256_storeu_ps(d.add(16), _mm256_max_ps(a2, z));
            _mm256_storeu_ps(d.add(24), _mm256_max_ps(a3, z));
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn mul_add_tap_avx2(dst: &mut [f32], src: &[f32], k: f32) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let kk = _mm256_set1_ps(k);
    unsafe {
        let mut i = 0;
        while i + 32 <= n {
            let s0 = _mm256_loadu_ps(src.as_ptr().add(i));
            let s1 = _mm256_loadu_ps(src.as_ptr().add(i + 8));
            let s2 = _mm256_loadu_ps(src.as_ptr().add(i + 16));
            let s3 = _mm256_loadu_ps(src.as_ptr().add(i + 24));
            let d0 = _mm256_loadu_ps(dst.as_ptr().add(i));
            let d1 = _mm256_loadu_ps(dst.as_ptr().add(i + 8));
            let d2 = _mm256_loadu_ps(dst.as_ptr().add(i + 16));
            let d3 = _mm256_loadu_ps(dst.as_ptr().add(i + 24));
            // multiply and add are separate rounded operations, not an FMA.
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(i),
                _mm256_add_ps(d0, _mm256_mul_ps(s0, kk)),
            );
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(i + 8),
                _mm256_add_ps(d1, _mm256_mul_ps(s1, kk)),
            );
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(i + 16),
                _mm256_add_ps(d2, _mm256_mul_ps(s2, kk)),
            );
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(i + 24),
                _mm256_add_ps(d3, _mm256_mul_ps(s3, kk)),
            );
            i += 32;
        }
        while i < n {
            let p = *src.get_unchecked(i) * k;
            *dst.get_unchecked_mut(i) += p;
            i += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn depthwise_3x3_avx2(dst: &mut [f32], src: &[f32], w: &[f32], in_len: usize) {
    let out_len = in_len - 2;
    debug_assert_eq!(src.len(), 3 * in_len * CHANNELS);
    debug_assert_eq!(dst.len(), out_len * CHANNELS);
    debug_assert_eq!(w.len(), 9 * CHANNELS);
    unsafe {
        let sp = src.as_ptr();
        let wp = w.as_ptr();
        // One output position at a time: the nine taps are accumulated in
        // registers in `(kh, kw)` order (the same per-element order as the
        // reference) and written once, instead of nine read-modify-write
        // passes over the whole output buffer.
        for p in 0..out_len {
            let mut a0 = zero();
            let mut a1 = zero();
            let mut a2 = zero();
            let mut a3 = zero();
            for kh in 0..3 {
                for kw in 0..3 {
                    let tap = wp.add((kh * 3 + kw) * CHANNELS);
                    let s = sp.add((kh * in_len + kw + p) * CHANNELS);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(s), _mm256_loadu_ps(tap), a0);
                    a1 =
                        _mm256_fmadd_ps(_mm256_loadu_ps(s.add(8)), _mm256_loadu_ps(tap.add(8)), a1);
                    a2 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(s.add(16)),
                        _mm256_loadu_ps(tap.add(16)),
                        a2,
                    );
                    a3 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(s.add(24)),
                        _mm256_loadu_ps(tap.add(24)),
                        a3,
                    );
                }
            }
            let d = dst.as_mut_ptr().add(p * CHANNELS);
            _mm256_storeu_ps(d, a0);
            _mm256_storeu_ps(d.add(8), a1);
            _mm256_storeu_ps(d.add(16), a2);
            _mm256_storeu_ps(d.add(24), a3);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn depthwise_1x3s2_avx2(dst: &mut [f32], src: &[f32], w: &[f32], pad_left: usize) {
    let in_len = src.len() / CHANNELS;
    let out_len = in_len.div_ceil(2);
    debug_assert_eq!(dst.len(), out_len * CHANNELS);
    debug_assert_eq!(w.len(), 3 * CHANNELS);
    let n = out_len * CHANNELS;
    unsafe {
        let mut i = 0;
        while i < n {
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), zero());
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), zero());
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), zero());
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), zero());
            i += 32;
        }
        for t in 0..3 {
            let tap = w.as_ptr().add(t * CHANNELS);
            let w0 = _mm256_loadu_ps(tap);
            let w1 = _mm256_loadu_ps(tap.add(8));
            let w2 = _mm256_loadu_ps(tap.add(16));
            let w3 = _mm256_loadu_ps(tap.add(24));
            for p in 0..out_len {
                let idx = (2 * p + t) as isize - pad_left as isize;
                if idx < 0 || idx as usize >= in_len {
                    continue;
                }
                let s = src.as_ptr().add(idx as usize * CHANNELS);
                let d = dst.as_mut_ptr().add(p * CHANNELS);
                let a0 = _mm256_fmadd_ps(_mm256_loadu_ps(s), w0, _mm256_loadu_ps(d));
                let a1 = _mm256_fmadd_ps(_mm256_loadu_ps(s.add(8)), w1, _mm256_loadu_ps(d.add(8)));
                let a2 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(s.add(16)), w2, _mm256_loadu_ps(d.add(16)));
                let a3 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(s.add(24)), w3, _mm256_loadu_ps(d.add(24)));
                _mm256_storeu_ps(d, a0);
                _mm256_storeu_ps(d.add(8), a1);
                _mm256_storeu_ps(d.add(16), a2);
                _mm256_storeu_ps(d.add(24), a3);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn maxpool1x3s2_avx2(dst: &mut [f32], src: &[f32], in_len: usize) {
    let out_len = (in_len - 3) / 2 + 1;
    let segments = src.len() / (in_len * CHANNELS);
    debug_assert_eq!(dst.len(), segments * out_len * CHANNELS);
    unsafe {
        for seg in 0..segments {
            for p in 0..out_len {
                let base = (seg * in_len + p * 2) * CHANNELS;
                let d = dst.as_mut_ptr().add((seg * out_len + p) * CHANNELS);
                let s = src.as_ptr().add(base);
                // `maxps(a, b)` returns `b` unless `a > b`, which is exactly
                // the reference's `if a > b { a } else { b }` chain (including
                // signed-zero and NaN behaviour).
                let m0 = _mm256_max_ps(_mm256_loadu_ps(s), _mm256_loadu_ps(s.add(CHANNELS)));
                let m0 = _mm256_max_ps(m0, _mm256_loadu_ps(s.add(2 * CHANNELS)));
                let m1 = _mm256_max_ps(
                    _mm256_loadu_ps(s.add(8)),
                    _mm256_loadu_ps(s.add(CHANNELS + 8)),
                );
                let m1 = _mm256_max_ps(m1, _mm256_loadu_ps(s.add(2 * CHANNELS + 8)));
                let m2 = _mm256_max_ps(
                    _mm256_loadu_ps(s.add(16)),
                    _mm256_loadu_ps(s.add(CHANNELS + 16)),
                );
                let m2 = _mm256_max_ps(m2, _mm256_loadu_ps(s.add(2 * CHANNELS + 16)));
                let m3 = _mm256_max_ps(
                    _mm256_loadu_ps(s.add(24)),
                    _mm256_loadu_ps(s.add(CHANNELS + 24)),
                );
                let m3 = _mm256_max_ps(m3, _mm256_loadu_ps(s.add(2 * CHANNELS + 24)));
                _mm256_storeu_ps(d, m0);
                _mm256_storeu_ps(d.add(8), m1);
                _mm256_storeu_ps(d.add(16), m2);
                _mm256_storeu_ps(d.add(24), m3);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// echo-model float kernels
//
// The echo graph's weights keep their container layout; the kernels below use
// `vpgatherdd` where a vector needs a strided weight (depthwise, deconv and
// 3x1 convolutions) and broadcast style FMA everywhere else.  Per output
// element the sequence of operations and the accumulation order are exactly
// the scalar reference's.
// ---------------------------------------------------------------------------

/// Generic 1x1 convolution with 16/32/64 output channels.  Output channels are
/// processed in one block of `ymm` lanes per 8 channels; input channels are
/// applied in ascending order, one FMA per lane.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_conv1x1_avx2(
    dst: &mut [f32],
    src: &[f32],
    w: &[f32],
    bias: &[f32],
    cin: usize,
    cout: usize,
) -> bool {
    macro_rules! run {
        ($n:expr) => {{
            let cout: usize = $n;
            let positions = src.len() / cin;
            unsafe {
                let bp = bias.as_ptr();
                let mut acc = [_mm256_setzero_ps(); $n / 8];
                for pos in 0..positions {
                    for b in 0..$n / 8 {
                        acc[b] = _mm256_loadu_ps(bp.add(b * 8));
                    }
                    let sp = src.as_ptr().add(pos * cin);
                    for ic in 0..cin {
                        let x = _mm256_set1_ps(*sp.add(ic));
                        let row = w.as_ptr().add(ic * cout);
                        for b in 0..$n / 8 {
                            acc[b] = _mm256_fmadd_ps(x, _mm256_loadu_ps(row.add(b * 8)), acc[b]);
                        }
                    }
                    let d = dst.as_mut_ptr().add(pos * cout);
                    for b in 0..$n / 8 {
                        _mm256_storeu_ps(d.add(b * 8), acc[b]);
                    }
                }
            }
            true
        }};
    }
    match cout {
        16 => run!(16),
        32 => run!(32),
        64 => run!(64),
        _ => <ScalarSimd as Simd>::echo_conv1x1(dst, src, w, bias, cin, cout),
    }
}

/// Depthwise 5x1 (dilated) / 3x1, vectorised over channels.  Strided weights
/// are gathered; the per-element tap order is unchanged.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_depthwise_avx2(
    dst: &mut [f32],
    src: &[f32],
    w: &[f32],
    bias: &[f32],
    channels: usize,
    taps: usize,
    fma_mode: bool,
    bins: usize,
) -> bool {
    debug_assert!(taps == 3 || taps == 5);
    unsafe {
        let stride = taps as i32;
        let idx = _mm256_setr_epi32(
            0,
            stride,
            2 * stride,
            3 * stride,
            4 * stride,
            5 * stride,
            6 * stride,
            7 * stride,
        );
        for h in 0..bins {
            let mut ch = 0usize;
            while ch + 8 <= channels {
                let mut acc = _mm256_loadu_ps(bias.as_ptr().add(ch));
                for t in 0..taps {
                    let i = if taps == 5 {
                        h as isize - 4 + 2 * t as isize
                    } else {
                        h as isize - 1 + t as isize
                    };
                    let xv = if i >= 0 && (i as usize) < bins {
                        _mm256_loadu_ps(src.as_ptr().add(i as usize * channels + ch))
                    } else {
                        zero()
                    };
                    let wv = _mm256_i32gather_ps(w.as_ptr().add(ch * taps + t), idx, 4);
                    acc = if fma_mode {
                        _mm256_fmadd_ps(xv, wv, acc)
                    } else {
                        _mm256_add_ps(acc, _mm256_mul_ps(xv, wv))
                    };
                }
                _mm256_storeu_ps(dst.as_mut_ptr().add(h * channels + ch), acc);
                ch += 8;
            }
            while ch < channels {
                let mut acc = bias[ch];
                for t in 0..taps {
                    let i = if taps == 5 {
                        h as isize - 4 + 2 * t as isize
                    } else {
                        h as isize - 1 + t as isize
                    };
                    let xv = if i >= 0 && (i as usize) < bins {
                        *src.get_unchecked(i as usize * channels + ch)
                    } else {
                        0.0
                    };
                    let wt = *w.get_unchecked(ch * taps + t);
                    acc = if fma_mode {
                        crate::fma::fma(xv, wt, acc)
                    } else {
                        acc + (xv * wt)
                    };
                }
                *dst.get_unchecked_mut(h * channels + ch) = acc;
                ch += 1;
            }
        }
    }
    true
}

/// Max pool 5x1 stride 2, vectorised over channels; `maxps` keeps the later
/// operand on ties, matching `if m > v { m } else { v }`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_maxpool5_avx2(dst: &mut [f32], src: &[f32], channels: usize, bins: usize) -> bool {
    let out_bins = (bins - 1) / 2 + 1;
    unsafe {
        for p in 0..out_bins {
            let mut ch = 0usize;
            while ch + 8 <= channels {
                let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
                for t in 0..5 {
                    let i = (2 * p) as isize - 2 + t as isize;
                    if i >= 0 && (i as usize) < bins {
                        let v = _mm256_loadu_ps(src.as_ptr().add(i as usize * channels + ch));
                        m = _mm256_max_ps(m, v);
                    }
                }
                _mm256_storeu_ps(dst.as_mut_ptr().add(p * channels + ch), m);
                ch += 8;
            }
            while ch < channels {
                let mut m = f32::NEG_INFINITY;
                for t in 0..5 {
                    let i = (2 * p) as isize - 2 + t as isize;
                    if i >= 0 && (i as usize) < bins {
                        let v = *src.get_unchecked(i as usize * channels + ch);
                        m = if m > v { m } else { v };
                    }
                }
                *dst.get_unchecked_mut(p * channels + ch) = m;
                ch += 1;
            }
        }
    }
    true
}

/// Depthwise deconvolution: descending taps, even output rows only, odd rows
/// are the bias.  Channels are the vector lanes; the weights are gathered with
/// the container's `[channel][tap]` stride.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_deconv_avx2(
    dst: &mut [f32],
    src: &[f32],
    w: &[f32],
    bias: &[f32],
    hin: usize,
    channels: usize,
) -> bool {
    unsafe {
        let idx = _mm256_setr_epi32(0, 5, 10, 15, 20, 25, 30, 35);
        let mut ch = 0usize;
        while ch + 8 <= channels {
            let bv = _mm256_loadu_ps(bias.as_ptr().add(ch));
            for j in 0..hin - 1 {
                _mm256_storeu_ps(dst.as_mut_ptr().add((2 * j + 1) * channels + ch), bv);
            }
            for j in 0..hin {
                let mut acc = bv;
                for t in (0..5).rev() {
                    let i = j as isize + 2 - t as isize;
                    if i >= 0 && (i as usize) < hin {
                        let xv = _mm256_loadu_ps(src.as_ptr().add(i as usize * channels + ch));
                        let wv = _mm256_i32gather_ps(w.as_ptr().add(ch * 5 + t), idx, 4);
                        acc = _mm256_fmadd_ps(xv, wv, acc);
                    }
                }
                _mm256_storeu_ps(dst.as_mut_ptr().add(2 * j * channels + ch), acc);
            }
            ch += 8;
        }
        while ch < channels {
            for j in 0..hin - 1 {
                *dst.get_unchecked_mut((2 * j + 1) * channels + ch) = bias[ch];
            }
            for j in 0..hin {
                let mut acc = bias[ch];
                for t in (0..5).rev() {
                    let i = j as isize + 2 - t as isize;
                    if i >= 0 && (i as usize) < hin {
                        acc = crate::fma::fma(
                            *src.get_unchecked(i as usize * channels + ch),
                            *w.get_unchecked(ch * 5 + t),
                            acc,
                        );
                    }
                }
                *dst.get_unchecked_mut(2 * j * channels + ch) = acc;
            }
            ch += 1;
        }
    }
    true
}

/// 3x1 decoder convolution: output channels are the vector lanes, weights are
/// gathered with the container's `[oc][ic][tap]` stride.  Input channel is
/// outer and tap inner, exactly like the scalar reference.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_conv3_avx2(
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
    unsafe {
        let step = (cin * 3) as i32;
        let idx = _mm256_setr_epi32(
            0,
            step,
            2 * step,
            3 * step,
            4 * step,
            5 * step,
            6 * step,
            7 * step,
        );
        let z = zero();
        let vslope = _mm256_set1_ps(slope);
        let mut oc = 0usize;
        while oc + 8 <= cout {
            for h in 0..bins {
                let mut acc = _mm256_loadu_ps(bias.as_ptr().add(oc));
                for ic in 0..cin {
                    for t in 0..3 {
                        let i = h as isize - 1 + t as isize;
                        if i >= 0 && (i as usize) < bins {
                            let xv = _mm256_set1_ps(*src.get_unchecked(i as usize * cin + ic));
                            let wv = _mm256_i32gather_ps(
                                w.as_ptr().add(oc * cin * 3 + ic * 3 + t),
                                idx,
                                4,
                            );
                            acc = _mm256_fmadd_ps(xv, wv, acc);
                        }
                    }
                }
                if leaky {
                    let gt = _mm256_cmp_ps(acc, z, _CMP_GT_OQ);
                    acc = _mm256_blendv_ps(_mm256_mul_ps(acc, vslope), acc, gt);
                }
                _mm256_storeu_ps(dst.as_mut_ptr().add(h * cout + oc), acc);
            }
            oc += 8;
        }
        for o in oc..cout {
            for h in 0..bins {
                let mut acc = bias[o];
                for ic in 0..cin {
                    for t in 0..3 {
                        let i = h as isize - 1 + t as isize;
                        if i >= 0 && (i as usize) < bins {
                            acc = crate::fma::fma(
                                *src.get_unchecked(i as usize * cin + ic),
                                *w.get_unchecked(o * cin * 3 + ic * 3 + t),
                                acc,
                            );
                        }
                    }
                }
                *dst.get_unchecked_mut(h * cout + o) = if leaky && !(acc > 0.0) {
                    acc * slope
                } else {
                    acc
                };
            }
        }
    }
    true
}

/// Layer 82: one output channel, tap-major weights `[tap][ic]`.  The three
/// taps are computed lane-wise over channels, then summed in channel order.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_conv3_out1_avx2(
    dst: &mut [f32],
    src: &[f32],
    w: &[f32],
    bias: &[f32],
    bins: usize,
    cin: usize,
) -> bool {
    unsafe {
        let z = zero();
        let mut lanes = [0.0f32; 8];
        for h in 0..bins {
            let mut o = *bias.get_unchecked(0);
            let mut c0 = 0usize;
            while c0 + 8 <= cin {
                let x0 = if h > 0 {
                    _mm256_loadu_ps(src.as_ptr().add((h - 1) * cin + c0))
                } else {
                    z
                };
                let x1 = _mm256_loadu_ps(src.as_ptr().add(h * cin + c0));
                let x2 = if h + 1 < bins {
                    _mm256_loadu_ps(src.as_ptr().add((h + 1) * cin + c0))
                } else {
                    z
                };
                let w0 = _mm256_loadu_ps(w.as_ptr().add(c0));
                let w1 = _mm256_loadu_ps(w.as_ptr().add(cin + c0));
                let w2 = _mm256_loadu_ps(w.as_ptr().add(2 * cin + c0));
                let acc =
                    _mm256_fmadd_ps(x2, w2, _mm256_fmadd_ps(x1, w1, _mm256_fmadd_ps(x0, w0, z)));
                _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
                for lane in 0..8 {
                    o += *lanes.get_unchecked(lane);
                }
                c0 += 8;
            }
            while c0 < cin {
                let x0 = if h > 0 {
                    *src.get_unchecked((h - 1) * cin + c0)
                } else {
                    0.0
                };
                let x1 = *src.get_unchecked(h * cin + c0);
                let x2 = if h + 1 < bins {
                    *src.get_unchecked((h + 1) * cin + c0)
                } else {
                    0.0
                };
                o += crate::fma::fma(
                    x2,
                    *w.get_unchecked(2 * cin + c0),
                    crate::fma::fma(
                        x1,
                        *w.get_unchecked(cin + c0),
                        crate::fma::fma(x0, *w.get_unchecked(c0), 0.0),
                    ),
                );
                c0 += 1;
            }
            *dst.get_unchecked_mut(h) = o;
        }
    }
    true
}

/// 33x33 MatMul: 64 output channels in eight `ymm` blocks, input positions
/// ascending.  The weight for a given `(i, j)` is a broadcast scalar.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_mm33_avx2(dst: &mut [f32], src: &[f32], w: &[f32]) -> bool {
    unsafe {
        for j in 0..33 {
            for ch in (0..64).step_by(8) {
                let mut acc = zero();
                for i in 0..33 {
                    let x = _mm256_loadu_ps(src.as_ptr().add(i * 64 + ch));
                    let wv = _mm256_set1_ps(*w.get_unchecked(i * 33 + j));
                    acc = _mm256_fmadd_ps(x, wv, acc);
                }
                _mm256_storeu_ps(dst.as_mut_ptr().add(j * 64 + ch), acc);
            }
        }
    }
    true
}

/// `i64` int8 GEMM for the `K = 2112` GRU input.
///
/// 256-input chunks keep each `int32` sub-sum below `256 * 8192 * 255 =
/// 534,773,760 < 2^31`; each chunk is widened and added into the `i64`
/// accumulator, so the result is the same exact integer sum as the scalar
/// path regardless of the (unprovable) global bound.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_gemm_i64_avx2(
    acc: &mut [i64],
    w: &[i8],
    zp: &[i8],
    qx: &[i16],
    k: usize,
    n: usize,
) -> bool {
    debug_assert!(acc.len() >= n);
    debug_assert_eq!(w.len(), k * n);
    debug_assert!(zp.len() >= n);
    debug_assert!(qx.len() >= k);
    acc[..n].fill(0);
    let pairs = k / 2;
    unsafe {
        let wp = w.as_ptr();
        let qp = qx.as_ptr();
        let mut j0 = 0usize;
        while j0 + 8 <= n {
            let zpv = _mm_cvtepi8_epi16(_mm_loadl_epi64(zp.as_ptr().add(j0) as *const __m128i));
            let mut out_lo = _mm256_setzero_si256();
            let mut out_hi = _mm256_setzero_si256();
            let mut p = 0usize;
            while p < pairs {
                let pend = (p + 128).min(pairs);
                let mut a0 = _mm_setzero_si128();
                let mut a1 = _mm_setzero_si128();
                while p < pend {
                    let q0 = *qp.add(2 * p);
                    let q1 = *qp.add(2 * p + 1);
                    if q0 != 0 || q1 != 0 {
                        let qbits = ((q1 as u16 as u32) << 16 | (q0 as u16 as u32)) as i32;
                        let qv = _mm_set1_epi32(qbits);
                        let row0 = wp.add(2 * p * n + j0);
                        let row1 = row0.add(n);
                        let wa = _mm_sub_epi16(
                            _mm_cvtepi8_epi16(_mm_loadl_epi64(row0 as *const __m128i)),
                            zpv,
                        );
                        let wb = _mm_sub_epi16(
                            _mm_cvtepi8_epi16(_mm_loadl_epi64(row1 as *const __m128i)),
                            zpv,
                        );
                        a0 = _mm_add_epi32(a0, _mm_madd_epi16(qv, _mm_unpacklo_epi16(wa, wb)));
                        a1 = _mm_add_epi32(a1, _mm_madd_epi16(qv, _mm_unpackhi_epi16(wa, wb)));
                    }
                    p += 1;
                }
                out_lo = _mm256_add_epi64(out_lo, _mm256_cvtepi32_epi64(a0));
                out_hi = _mm256_add_epi64(out_hi, _mm256_cvtepi32_epi64(a1));
            }
            let out = acc.as_mut_ptr().add(j0);
            _mm256_storeu_si256(out as *mut __m256i, out_lo);
            _mm256_storeu_si256(out.add(4) as *mut __m256i, out_hi);
            j0 += 8;
        }
        // Odd input width: the last input is added exactly to every output.
        if k & 1 != 0 {
            let q = *qp.add(k - 1) as i64;
            if q != 0 {
                for j in 0..n {
                    acc[j] += q * (*wp.add((k - 1) * n + j) as i64 - *zp.get_unchecked(j) as i64);
                }
            }
        }
        for j in j0..n {
            let mut a = 0i64;
            for kk in 0..k {
                let q = *qp.add(kk) as i64;
                if q != 0 {
                    a += q * (*wp.add(kk * n + j) as i64 - *zp.get_unchecked(j) as i64);
                }
            }
            *acc.get_unchecked_mut(j) = a;
        }
    }
    true
}

// -- quantisers -------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx2,fma")]
fn max_abs_avx2(src: &[f32]) -> f32 {
    unsafe {
        let sign = _mm256_set1_ps(-0.0);
        let mut vmax = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= src.len() {
            let x = _mm256_loadu_ps(src.as_ptr().add(i));
            vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, x));
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), vmax);
        let mut m = 0.0f32;
        for lane in lanes {
            if lane > m {
                m = lane;
            }
        }
        while i < src.len() {
            let a = src[i].abs();
            if a > m {
                m = a;
            }
            i += 1;
        }
        m
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_quantize_input_avx2(dst: &mut [i16], src: &[f32], quant_scale: f32) -> f32 {
    let n = src.len();
    let mx = max_abs_avx2(src);
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
    unsafe {
        let vinv = _mm256_set1_ps(inv);
        let hi = _mm256_set1_ps(clamp as f32);
        let lo = _mm256_set1_ps(-(clamp as f32));
        let mut i = 0usize;
        while i + 8 <= nvec {
            let x = _mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i)), vinv);
            let r = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let bad = _mm256_movemask_ps(_mm256_cmp_ps(r, r, _CMP_UNORD_Q));
            if bad != 0 {
                for lane in 0..8 {
                    *dst.get_unchecked_mut(i + lane) =
                        crate::simd::quant_rne(*src.get_unchecked(i + lane), clamp);
                }
            } else {
                let c = _mm256_min_ps(_mm256_max_ps(r, lo), hi);
                let q = _mm256_cvttps_epi32(c);
                let p = _mm256_packs_epi32(q, q);
                let p = _mm256_permute4x64_epi64(p, 0x08);
                _mm_storeu_si128(
                    dst.as_mut_ptr().add(i) as *mut __m128i,
                    _mm256_castsi256_si128(p),
                );
            }
            i += 8;
        }
        while i < n {
            *dst.get_unchecked_mut(i) =
                crate::simd::quant_half_away(*src.get_unchecked(i) * inv, clamp);
            i += 1;
        }
    }
    a
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_quantize_hidden_avx2(
    dst: &mut [i16],
    src: &[f32],
    hidden_scale: f32,
    hidden_scale_wide: f32,
) -> f32 {
    let mx = max_abs_avx2(src);
    let c = if mx <= 1.0 {
        hidden_scale
    } else {
        hidden_scale_wide
    };
    let a = mx / c;
    let inv = if a != 0.0 { 1.0 / a } else { 0.0 };
    let lim = c as i32;
    unsafe {
        let vinv = _mm256_set1_ps(inv);
        let hi = _mm256_set1_ps(lim as f32);
        let lo = _mm256_set1_ps(-(lim as f32));
        let n = src.len();
        let mut i = 0usize;
        while i + 8 <= n {
            let x = _mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i)), vinv);
            let r = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let bad = _mm256_movemask_ps(_mm256_cmp_ps(r, r, _CMP_UNORD_Q));
            if bad != 0 {
                for lane in 0..8 {
                    *dst.get_unchecked_mut(i + lane) =
                        ((*src.get_unchecked(i + lane) * inv).round_ties_even() as i32)
                            .clamp(-lim, lim) as i16;
                }
            } else {
                let c2 = _mm256_min_ps(_mm256_max_ps(r, lo), hi);
                let q = _mm256_cvttps_epi32(c2);
                let p = _mm256_packs_epi32(q, q);
                let p = _mm256_permute4x64_epi64(p, 0x08);
                _mm_storeu_si128(
                    dst.as_mut_ptr().add(i) as *mut __m128i,
                    _mm256_castsi256_si128(p),
                );
            }
            i += 8;
        }
        while i < n {
            *dst.get_unchecked_mut(i) =
                ((*src.get_unchecked(i) * inv).round_ties_even() as i32).clamp(-lim, lim) as i16;
            i += 1;
        }
    }
    a
}

// -- GRU tail and final sigmoid --------------------------------------------

#[inline]
#[target_feature(enable = "avx2,fma")]
fn echo_sigmoid_gru8(x: __m256, k: &EchoActConsts) -> __m256 {
    {
        let t = _mm256_min_ps(
            _mm256_max_ps(x, _mm256_set1_ps(k.neg_eight)),
            _mm256_set1_ps(k.eight),
        );
        let x2 = _mm256_mul_ps(t, t);
        let mut q = _mm256_fmadd_ps(x2, _mm256_set1_ps(k.sig[0]), _mm256_set1_ps(k.sig[1]));
        q = _mm256_fmadd_ps(x2, q, _mm256_set1_ps(k.sig[2]));
        q = _mm256_fmadd_ps(x2, q, _mm256_set1_ps(k.sig[3]));
        let s = _mm256_mul_ps(t, rcp8(q));
        let mut p = _mm256_fmadd_ps(x2, _mm256_set1_ps(k.sig[4]), _mm256_set1_ps(k.sig[5]));
        p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(k.sig[6]));
        _mm256_fmadd_ps(s, p, _mm256_set1_ps(k.half))
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn echo_tanh_gru8(x: __m256, k: &EchoActConsts) -> __m256 {
    {
        let v = _mm256_min_ps(
            _mm256_max_ps(x, _mm256_set1_ps(k.neg_four)),
            _mm256_set1_ps(k.four),
        );
        let x2 = _mm256_mul_ps(v, v);
        let mut p = _mm256_fmadd_ps(x2, _mm256_set1_ps(k.tanh[0]), _mm256_set1_ps(k.tanh[1]));
        p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(k.tanh[2]));
        p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(k.tanh[3]));
        let s = _mm256_mul_ps(v, rcp8(p));
        let q = _mm256_add_ps(x2, _mm256_set1_ps(k.tanh[4]));
        _mm256_mul_ps(s, _mm256_fmadd_ps(x2, q, _mm256_set1_ps(k.tanh[3])))
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn echo_gru_tail_avx2(yih: &mut [f32], yhh: &[f32], h: &mut [f32], k: &EchoActConsts) -> bool {
    let hidden = h.len();
    let n2 = 2 * hidden;
    debug_assert_eq!(yih.len(), 3 * hidden);
    debug_assert_eq!(yhh.len(), 3 * hidden);
    unsafe {
        let mut j = 0usize;
        while j + 8 <= n2 {
            let a = _mm256_loadu_ps(yih.as_ptr().add(j));
            let b = _mm256_loadu_ps(yhh.as_ptr().add(j));
            _mm256_storeu_ps(yih.as_mut_ptr().add(j), _mm256_add_ps(a, b));
            j += 8;
        }
        j = 0;
        while j + 8 <= n2 {
            let x = _mm256_loadu_ps(yih.as_ptr().add(j));
            _mm256_storeu_ps(yih.as_mut_ptr().add(j), echo_sigmoid_gru8(x, k));
            j += 8;
        }
        j = 0;
        while j + 8 <= hidden {
            let t = _mm256_add_ps(
                _mm256_mul_ps(
                    _mm256_loadu_ps(yhh.as_ptr().add(n2 + j)),
                    _mm256_loadu_ps(yih.as_ptr().add(hidden + j)),
                ),
                _mm256_loadu_ps(yih.as_ptr().add(n2 + j)),
            );
            _mm256_storeu_ps(yih.as_mut_ptr().add(n2 + j), echo_tanh_gru8(t, k));
            j += 8;
        }
        j = 0;
        while j + 8 <= hidden {
            let z = _mm256_loadu_ps(yih.as_ptr().add(j));
            let n = _mm256_loadu_ps(yih.as_ptr().add(n2 + j));
            let one_minus_z = _mm256_sub_ps(_mm256_set1_ps(k.one), z);
            let a = _mm256_mul_ps(one_minus_z, n);
            let b = _mm256_mul_ps(z, _mm256_loadu_ps(h.as_ptr().add(j)));
            _mm256_storeu_ps(h.as_mut_ptr().add(j), _mm256_add_ps(a, b));
            j += 8;
        }
    }
    true
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn echo_sigmoid_vector8(x: __m256, k: &EchoActConsts) -> __m256 {
    {
        let neg = _mm256_xor_ps(x, _mm256_set1_ps(-0.0));
        let t = _mm256_min_ps(
            _mm256_max_ps(neg, _mm256_set1_ps(k.sig88_neg_clamp)),
            _mm256_set1_ps(k.sig88_clamp),
        );
        let nf = _mm256_round_ps(
            _mm256_mul_ps(t, _mm256_set1_ps(k.exp_log2e)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let r = _mm256_fmadd_ps(nf, _mm256_set1_ps(k.exp_ln2), t);
        let mut p = _mm256_fmadd_ps(r, _mm256_set1_ps(k.exp_c[0]), _mm256_set1_ps(k.exp_c[1]));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.exp_c[2]));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.exp_c[3]));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.exp_c[4]));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.half));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.one));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.one));
        let ni = _mm256_cvttps_epi32(nf);
        let bits = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(EXP_BIAS)), 23);
        let e = _mm256_mul_ps(p, _mm256_castsi256_ps(bits));
        rcp8(_mm256_add_ps(e, _mm256_set1_ps(k.one)))
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn echo_sigmoid88_avx2(dst: &mut [f32], src: &[f32], k: &EchoActConsts) -> bool {
    let n = src.len();
    let nvec = n - (n % 8);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= nvec {
            let x = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), echo_sigmoid_vector8(x, k));
            i += 8;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = crate::dexp::sigmoid_tail(*src.get_unchecked(i));
            i += 1;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// integer GEMM
// ---------------------------------------------------------------------------

/// Sum of `k` `int16` inputs as `int32`.
///
/// Quantised activations are clamped to at most `+-16384` and `k <= 1024`, so
/// the true sum fits comfortably; the arithmetic is written modulo `2^32` like
/// the rest of the integer path.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_i16_avx2(q: *const i16, k: usize) -> i32 {
    let ones = _mm256_set1_epi16(1);
    let mut s = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 16 <= k {
        let v = _mm256_loadu_si256(q.add(i) as *const __m256i);
        s = _mm256_add_epi32(s, _mm256_madd_epi16(v, ones));
        i += 16;
    }
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, s);
    let mut sum = 0i32;
    for &v in lanes.iter() {
        sum = sum.wrapping_add(v);
    }
    while i < k {
        sum = sum.wrapping_add(*q.add(i) as i32);
        i += 1;
    }
    sum
}

/// 64 output channels per block, eight `ymm` accumulators.
///
/// `w` keeps its `[k][n]` layout, so a 64-wide block reads one complete
/// 64-byte cache line per input row (instead of half a line per 32-wide block
/// and a second pass later).  Each 16-byte group of two adjacent rows is
/// sign-extended and interleaved into 16-bit lane pairs, then accumulated with
/// `vpmaddwd`.
///
/// The per-channel zero point is *not* subtracted from every weight load.
/// Instead the kernel accumulates raw weights and folds the correction
/// `-zp[j] * sum(q)` into the finished column.  That is the same integer
/// expression, just reassociated; every step is a 32-bit wrapping add/multiply
/// and `vpmaddwd` cannot overflow a 32-bit lane for `i8 * i16` pairs, so the
/// result is bit-identical to the scalar reference for any input, not just
/// within the proven bound.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn gemm_i8_avx2(acc: &mut [i32], w: &[i8], zp: &[i8], qx: &[i16], k: usize, n: usize) {
    debug_assert!(acc.len() >= n);
    debug_assert_eq!(w.len(), k * n);
    debug_assert!(zp.len() >= n);
    debug_assert!(qx.len() >= k);
    let pairs = k / 2;
    unsafe {
        let wp = w.as_ptr();
        let qp = qx.as_ptr();
        let zpp = zp.as_ptr();
        let sumq = sum_i16_avx2(qp, k);
        let mut j0 = 0usize;

        while j0 + 64 <= n {
            let mut a0 = _mm256_setzero_si256();
            let mut a1 = _mm256_setzero_si256();
            let mut a2 = _mm256_setzero_si256();
            let mut a3 = _mm256_setzero_si256();
            let mut a4 = _mm256_setzero_si256();
            let mut a5 = _mm256_setzero_si256();
            let mut a6 = _mm256_setzero_si256();
            let mut a7 = _mm256_setzero_si256();
            let mut p = 0usize;
            while p < pairs {
                let q0 = *qp.add(2 * p);
                let q1 = *qp.add(2 * p + 1);
                if q0 != 0 || q1 != 0 {
                    let qbits = ((q1 as u16 as u32) << 16 | (q0 as u16 as u32)) as i32;
                    let qv = _mm256_set1_epi32(qbits);
                    let row0 = wp.add(2 * p * n + j0);
                    let row1 = row0.add(n);

                    let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row0 as *const __m128i));
                    let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row1 as *const __m128i));
                    a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(qv, _mm256_unpacklo_epi16(w0, w1)));
                    a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(qv, _mm256_unpackhi_epi16(w0, w1)));

                    let w2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row0.add(16) as *const __m128i));
                    let w3 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row1.add(16) as *const __m128i));
                    a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(qv, _mm256_unpacklo_epi16(w2, w3)));
                    a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(qv, _mm256_unpackhi_epi16(w2, w3)));

                    let w4 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row0.add(32) as *const __m128i));
                    let w5 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row1.add(32) as *const __m128i));
                    a4 = _mm256_add_epi32(a4, _mm256_madd_epi16(qv, _mm256_unpacklo_epi16(w4, w5)));
                    a5 = _mm256_add_epi32(a5, _mm256_madd_epi16(qv, _mm256_unpackhi_epi16(w4, w5)));

                    let w6 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row0.add(48) as *const __m128i));
                    let w7 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row1.add(48) as *const __m128i));
                    a6 = _mm256_add_epi32(a6, _mm256_madd_epi16(qv, _mm256_unpacklo_epi16(w6, w7)));
                    a7 = _mm256_add_epi32(a7, _mm256_madd_epi16(qv, _mm256_unpackhi_epi16(w6, w7)));
                }
                p += 1;
            }
            let out = acc.as_mut_ptr().add(j0) as *mut __m256i;
            _mm256_storeu_si256(out, _mm256_permute2x128_si256(a0, a1, 0x20));
            _mm256_storeu_si256(out.add(1), _mm256_permute2x128_si256(a0, a1, 0x31));
            _mm256_storeu_si256(out.add(2), _mm256_permute2x128_si256(a2, a3, 0x20));
            _mm256_storeu_si256(out.add(3), _mm256_permute2x128_si256(a2, a3, 0x31));
            _mm256_storeu_si256(out.add(4), _mm256_permute2x128_si256(a4, a5, 0x20));
            _mm256_storeu_si256(out.add(5), _mm256_permute2x128_si256(a4, a5, 0x31));
            _mm256_storeu_si256(out.add(6), _mm256_permute2x128_si256(a6, a7, 0x20));
            _mm256_storeu_si256(out.add(7), _mm256_permute2x128_si256(a6, a7, 0x31));
            j0 += 64;
        }

        while j0 + 16 <= n {
            let mut a0 = _mm256_setzero_si256();
            let mut a1 = _mm256_setzero_si256();
            let mut p = 0usize;
            while p < pairs {
                let q0 = *qp.add(2 * p);
                let q1 = *qp.add(2 * p + 1);
                if q0 != 0 || q1 != 0 {
                    let qbits = ((q1 as u16 as u32) << 16 | (q0 as u16 as u32)) as i32;
                    let qv = _mm256_set1_epi32(qbits);
                    let row0 = wp.add(2 * p * n + j0);
                    let row1 = row0.add(n);
                    let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row0 as *const __m128i));
                    let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(row1 as *const __m128i));
                    a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(qv, _mm256_unpacklo_epi16(w0, w1)));
                    a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(qv, _mm256_unpackhi_epi16(w0, w1)));
                }
                p += 1;
            }
            let out = acc.as_mut_ptr().add(j0);
            _mm256_storeu_si256(out as *mut __m256i, _mm256_permute2x128_si256(a0, a1, 0x20));
            _mm256_storeu_si256(
                out.add(8) as *mut __m256i,
                _mm256_permute2x128_si256(a0, a1, 0x31),
            );
            j0 += 16;
        }

        while j0 < n {
            let mut a = 0i32;
            for p in 0..pairs {
                let q0 = *qp.add(2 * p) as i32;
                let q1 = *qp.add(2 * p + 1) as i32;
                if q0 != 0 {
                    a = a.wrapping_add(q0.wrapping_mul(*wp.add(2 * p * n + j0) as i32));
                }
                if q1 != 0 {
                    a = a.wrapping_add(q1.wrapping_mul(*wp.add((2 * p + 1) * n + j0) as i32));
                }
            }
            *acc.get_unchecked_mut(j0) = a;
            j0 += 1;
        }

        if k & 1 != 0 {
            // Odd input widths: the last input is added after the paired
            // blocks.  Integer accumulation is order independent.
            let q = *qp.add(k - 1) as i32;
            if q != 0 {
                for j in 0..n {
                    let v = *wp.add((k - 1) * n + j) as i32;
                    let a = acc.get_unchecked_mut(j);
                    *a = a.wrapping_add(q.wrapping_mul(v));
                }
            }
        }

        // Fold the zero point back in: acc[j] -= zp[j] * sum(q).
        if sumq != 0 {
            let sv = _mm256_set1_epi32(sumq);
            let mut j = 0usize;
            while j + 8 <= n {
                let z = _mm256_cvtepi8_epi32(_mm_loadl_epi64(zpp.add(j) as *const __m128i));
                let prod = _mm256_mullo_epi32(z, sv);
                let a = _mm256_loadu_si256(acc.as_ptr().add(j) as *const __m256i);
                _mm256_storeu_si256(
                    acc.as_mut_ptr().add(j) as *mut __m256i,
                    _mm256_sub_epi32(a, prod),
                );
                j += 8;
            }
            while j < n {
                let a = acc.get_unchecked_mut(j);
                *a = a.wrapping_sub((*zpp.add(j) as i32).wrapping_mul(sumq));
                j += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// quantisation (same rounding and clamp as the reference, including the
// saturating-cast edge and the NaN branch)
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx2,fma")]
fn quantize_i16_avx2(dst: &mut [i16], src: &[f32], inv: f32, clamp: i32) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    unsafe {
        let vinv = _mm256_set1_ps(inv);
        let hi = _mm256_set1_ps(clamp as f32);
        let lo = _mm256_set1_ps(-(clamp as f32));
        let mut i = 0usize;
        while i + 8 <= n {
            let x = _mm256_mul_ps(_mm256_loadu_ps(src.as_ptr().add(i)), vinv);
            let r = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let bad = _mm256_movemask_ps(_mm256_cmp_ps(r, r, _CMP_UNORD_Q));
            if bad != 0 {
                for lane in 0..8 {
                    *dst.get_unchecked_mut(i + lane) =
                        quantize_one(*src.get_unchecked(i + lane), inv, clamp);
                }
            } else {
                // Clamping in the float domain is equivalent to the
                // reference's saturating `as i32` cast followed by the
                // integer clamp, because `clamp` is exactly representable.
                let c = _mm256_min_ps(_mm256_max_ps(r, lo), hi);
                let q = _mm256_cvttps_epi32(c);
                let p = _mm256_packs_epi32(q, q);
                let p = _mm256_permute4x64_epi64(p, 0x08);
                _mm_storeu_si128(
                    dst.as_mut_ptr().add(i) as *mut __m128i,
                    _mm256_castsi256_si128(p),
                );
            }
            i += 8;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = quantize_one(*src.get_unchecked(i), inv, clamp);
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// integer-GEMM epilogue
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx2,fma")]
fn scale_i32_add_avx2(dst: &mut [f32], acc: &[i32], sp: &[f32], bias: &[f32]) {
    let n = dst.len();
    debug_assert!(acc.len() >= n);
    debug_assert_eq!(n, sp.len());
    debug_assert_eq!(n, bias.len());
    unsafe {
        let mut i = 0usize;
        while i + 8 <= n {
            let a = _mm256_cvtepi32_ps(_mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i));
            let p = _mm256_mul_ps(a, _mm256_loadu_ps(sp.as_ptr().add(i)));
            let y = _mm256_add_ps(p, _mm256_loadu_ps(bias.as_ptr().add(i)));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), y);
            i += 8;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = crate::fma::mul_add2(
                *acc.get_unchecked(i) as f32,
                *sp.get_unchecked(i),
                *bias.get_unchecked(i),
            );
            i += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn scale_i32_add_inplace_avx2(dst: &mut [f32], acc: &[i32], sp: &[f32]) {
    let n = dst.len();
    debug_assert!(acc.len() >= n);
    debug_assert_eq!(n, sp.len());
    unsafe {
        let mut i = 0usize;
        while i + 8 <= n {
            let a = _mm256_cvtepi32_ps(_mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i));
            let p = _mm256_mul_ps(a, _mm256_loadu_ps(sp.as_ptr().add(i)));
            let y = _mm256_add_ps(p, _mm256_loadu_ps(dst.as_ptr().add(i)));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), y);
            i += 8;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = crate::fma::mul_add2(
                *acc.get_unchecked(i) as f32,
                *sp.get_unchecked(i),
                *dst.get_unchecked(i),
            );
            i += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn scale_i32_add_relu_avx2(dst: &mut [f32], acc: &[i32], sp: &[f32], bias: &[f32]) {
    let n = dst.len();
    debug_assert!(acc.len() >= n);
    debug_assert_eq!(n, sp.len());
    debug_assert_eq!(n, bias.len());
    let z = zero();
    unsafe {
        let mut i = 0usize;
        while i + 8 <= n {
            let a = _mm256_cvtepi32_ps(_mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i));
            let p = _mm256_mul_ps(a, _mm256_loadu_ps(sp.as_ptr().add(i)));
            let y = _mm256_add_ps(p, _mm256_loadu_ps(bias.as_ptr().add(i)));
            // `maxps(y, 0)` is `if y > 0 { y } else { 0 }`, including NaN and
            // negative zero (it returns the second operand).
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_max_ps(y, z));
            i += 8;
        }
        while i < n {
            let mut y = crate::fma::mul_add2(
                *acc.get_unchecked(i) as f32,
                *sp.get_unchecked(i),
                *bias.get_unchecked(i),
            );
            if !(y > 0.0) {
                y = 0.0;
            }
            *dst.get_unchecked_mut(i) = y;
            i += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn mul_add2_slices_avx2(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
    let n = dst.len();
    debug_assert_eq!(n, a.len());
    debug_assert_eq!(n, b.len());
    debug_assert_eq!(n, c.len());
    unsafe {
        let mut i = 0usize;
        while i + 8 <= n {
            let p = _mm256_mul_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
            );
            let y = _mm256_add_ps(p, _mm256_loadu_ps(c.as_ptr().add(i)));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), y);
            i += 8;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = crate::fma::mul_add2(
                *a.get_unchecked(i),
                *b.get_unchecked(i),
                *c.get_unchecked(i),
            );
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// activations
// ---------------------------------------------------------------------------

/// Vectorised `exp` range reduction + degree-8 polynomial, bit-identical to
/// [`crate::simd::exp_kernel`].
#[inline]
#[target_feature(enable = "avx2,fma")]
fn exp8(t: __m256, k: &ActConsts) -> __m256 {
    {
        let log2e = _mm256_set1_ps(k.log2e);
        let n = _mm256_round_ps(
            _mm256_mul_ps(t, log2e),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let ln2 = _mm256_set1_ps(k.ln2);
        let r = _mm256_fmadd_ps(n, ln2, t);
        let mut p = _mm256_fmadd_ps(
            r,
            _mm256_set1_ps(k.exp_poly[0]),
            _mm256_set1_ps(k.exp_poly[1]),
        );
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.exp_poly[2]));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.exp_poly[3]));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.exp_poly[4]));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.half));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.one));
        p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(k.one));
        let ni = _mm256_cvttps_epi32(n);
        let bits = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(EXP_BIAS)), 23);
        _mm256_mul_ps(p, _mm256_castsi256_ps(bits))
    }
}

/// Vectorised reciprocal model: 4096-entry table lookup via gather, then the
/// exact integer bit formula (including the zero/infinity/NaN/subnormal
/// branches).  The table itself is the same data the scalar path uses.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) fn rcp8(x: __m256) -> __m256 {
    unsafe {
        let bits = _mm256_castps_si256(x);
        let sign = _mm256_and_si256(bits, _mm256_set1_epi32(0x8000_0000u32 as i32));
        let exp = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xFF));
        let mant = _mm256_and_si256(bits, _mm256_set1_epi32(0x007F_FFFF));
        let idx = _mm256_and_si256(_mm256_srli_epi32(bits, 11), _mm256_set1_epi32(0xFFF));
        let k = _mm256_i32gather_epi32(RCP_TABLE_U32.as_ptr() as *const i32, idx, 4);

        let e = _mm256_sub_epi32(_mm256_set1_epi32(253), exp);
        let ebits = _mm256_slli_epi32(e, 23);
        let mbits = _mm256_slli_epi32(_mm256_sub_epi32(k, _mm256_set1_epi32(4096)), 11);
        let fast = _mm256_or_si256(sign, _mm256_or_si256(ebits, mbits));

        let zeroed = sign;
        let inf = _mm256_or_si256(sign, _mm256_set1_epi32(0x7F80_0000u32 as i32));
        let nan = _mm256_or_si256(_mm256_or_si256(inf, mant), _mm256_set1_epi32(0x0040_0000));
        let is_zero_exp = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());
        let is_255 = _mm256_cmpeq_epi32(exp, _mm256_set1_epi32(255));
        let ge253 = _mm256_cmpgt_epi32(exp, _mm256_set1_epi32(252));
        let mant_zero = _mm256_cmpeq_epi32(mant, _mm256_setzero_si256());
        let is_inf = _mm256_and_si256(is_255, mant_zero);
        let is_nan = _mm256_andnot_si256(mant_zero, is_255);

        let mut res = _mm256_blendv_epi8(fast, zeroed, ge253);
        res = _mm256_blendv_epi8(res, inf, is_zero_exp);
        res = _mm256_blendv_epi8(res, zeroed, is_inf);
        res = _mm256_blendv_epi8(res, nan, is_nan);
        _mm256_castsi256_ps(res)
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn sigmoid8(x: __m256, k: &ActConsts) -> __m256 {
    {
        let neg = _mm256_xor_ps(x, _mm256_set1_ps(-0.0));
        let t = _mm256_max_ps(
            _mm256_min_ps(neg, _mm256_set1_ps(k.exp_clamp)),
            _mm256_set1_ps(k.neg_exp_clamp),
        );
        let e = exp8(t, k);
        rcp8(_mm256_add_ps(e, _mm256_set1_ps(k.one)))
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn tanh_gate8(x: __m256, k: &ActConsts) -> __m256 {
    {
        let t = _mm256_max_ps(
            _mm256_min_ps(
                _mm256_mul_ps(x, _mm256_set1_ps(k.two)),
                _mm256_set1_ps(k.exp_clamp),
            ),
            _mm256_set1_ps(k.neg_exp_clamp),
        );
        let e = exp8(t, k);
        let one = _mm256_set1_ps(k.one);
        _mm256_mul_ps(_mm256_sub_ps(e, one), rcp8(_mm256_add_ps(e, one)))
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn tanh_cell8(x: __m256, k: &ActConsts) -> __m256 {
    {
        let v = _mm256_min_ps(
            _mm256_max_ps(x, _mm256_set1_ps(k.neg_four)),
            _mm256_set1_ps(k.four),
        );
        let x2 = _mm256_mul_ps(v, v);
        let mut p = _mm256_fmadd_ps(
            x2,
            _mm256_set1_ps(k.cell_poly[0]),
            _mm256_set1_ps(k.cell_poly[1]),
        );
        p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(k.cell_poly[2]));
        p = _mm256_fmadd_ps(x2, p, _mm256_set1_ps(k.cell_poly[3]));
        let q = _mm256_add_ps(x2, _mm256_set1_ps(k.cell_poly[4]));
        let s = _mm256_mul_ps(v, rcp8(p));
        _mm256_mul_ps(s, _mm256_fmadd_ps(x2, q, _mm256_set1_ps(k.cell_poly[3])))
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn lstm_activate_avx2(y: &mut [f32], h: usize, k: &ActConsts) {
    debug_assert_eq!(y.len(), 4 * h);
    unsafe {
        let mut t = 0usize;
        while t + 8 <= h {
            for block in 0..3 {
                let x = _mm256_loadu_ps(y.as_ptr().add(block * h + t));
                _mm256_storeu_ps(y.as_mut_ptr().add(block * h + t), sigmoid8(x, k));
            }
            let x = _mm256_loadu_ps(y.as_ptr().add(3 * h + t));
            _mm256_storeu_ps(y.as_mut_ptr().add(3 * h + t), tanh_gate8(x, k));
            t += 8;
        }
        while t < h {
            let i = sigmoid(y[t], k);
            let o = sigmoid(y[h + t], k);
            let f = sigmoid(y[2 * h + t], k);
            let g = tanh_gate(y[3 * h + t], k);
            y[t] = i;
            y[h + t] = o;
            y[2 * h + t] = f;
            y[3 * h + t] = g;
            t += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn lstm_cell_avx2(
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
    unsafe {
        let vinv = _mm256_set1_ps(inv_h);
        let hi = _mm256_set1_ps(clamp as f32);
        let lo = _mm256_set1_ps(-(clamp as f32));
        let mut t = 0usize;
        while t + 8 <= h {
            let iv = _mm256_loadu_ps(y.as_ptr().add(t));
            let ov = _mm256_loadu_ps(y.as_ptr().add(h + t));
            let fv = _mm256_loadu_ps(y.as_ptr().add(2 * h + t));
            let gv = _mm256_loadu_ps(y.as_ptr().add(3 * h + t));
            let c_prev = _mm256_loadu_ps(cell.as_ptr().add(t));
            let c_new = _mm256_fmadd_ps(gv, iv, _mm256_mul_ps(fv, c_prev));
            let tc = tanh_cell8(c_new, k);
            let h_new = _mm256_mul_ps(ov, tc);
            let bad = _mm256_movemask_ps(_mm256_or_ps(
                _mm256_cmp_ps(c_new, c_new, _CMP_UNORD_Q),
                _mm256_cmp_ps(h_new, h_new, _CMP_UNORD_Q),
            ));
            if bad != 0 {
                // NaN cannot occur for finite gate inputs, but the scalar
                // saturating-cast path differs for it, so fall back exactly.
                for lane in 0..8 {
                    let tt = t + lane;
                    let i = *y.get_unchecked(tt);
                    let o = *y.get_unchecked(h + tt);
                    let f = *y.get_unchecked(2 * h + tt);
                    let g = *y.get_unchecked(3 * h + tt);
                    let c_prev = *cell.get_unchecked(tt);
                    let c_new = crate::fma::fma(g, i, f * c_prev);
                    *cell.get_unchecked_mut(tt) = c_new;
                    let h_new = o * tanh_cell(c_new, k);
                    *qh.get_unchecked_mut(tt) = quantize_one(h_new, inv_h, clamp);
                }
            } else {
                _mm256_storeu_ps(cell.as_mut_ptr().add(t), c_new);
                let x = _mm256_mul_ps(h_new, vinv);
                let r = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                let c = _mm256_min_ps(_mm256_max_ps(r, lo), hi);
                let q = _mm256_cvttps_epi32(c);
                let p = _mm256_packs_epi32(q, q);
                let p = _mm256_permute4x64_epi64(p, 0x08);
                _mm_storeu_si128(
                    qh.as_mut_ptr().add(t) as *mut __m128i,
                    _mm256_castsi256_si128(p),
                );
            }
            t += 8;
        }
        while t < h {
            let i = *y.get_unchecked(t);
            let o = *y.get_unchecked(h + t);
            let f = *y.get_unchecked(2 * h + t);
            let g = *y.get_unchecked(3 * h + t);
            let c_prev = *cell.get_unchecked(t);
            let c_new = crate::fma::fma(g, i, f * c_prev);
            *cell.get_unchecked_mut(t) = c_new;
            let h_new = o * tanh_cell(c_new, k);
            *qh.get_unchecked_mut(t) = quantize_one(h_new, inv_h, clamp);
            t += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// trait implementation
// ---------------------------------------------------------------------------

impl Simd for Avx2Simd {
    #[inline]
    fn name(&self) -> &'static str {
        "avx2"
    }

    #[inline]
    fn fma_broadcast(acc: &mut [f32], x: f32, w: &[f32]) {
        debug_assert_eq!(acc.len(), w.len());
        // Small diagnostic helper; the hot kernels below do not use it.
        unsafe {
            let xv = _mm256_set1_ps(x);
            let mut i = 0;
            while i + 8 <= acc.len() {
                let a = _mm256_loadu_ps(acc.as_ptr().add(i));
                let wv = _mm256_loadu_ps(w.as_ptr().add(i));
                _mm256_storeu_ps(acc.as_mut_ptr().add(i), _mm256_fmadd_ps(xv, wv, a));
                i += 8;
            }
            while i < acc.len() {
                acc[i] = crate::fma::fma(x, w[i], acc[i]);
                i += 1;
            }
        }
    }

    #[inline]
    fn fma_tiled(acc: &mut [f32], x: &[f32], w: &[f32]) {
        debug_assert_eq!(acc.len(), x.len());
        let channels = w.len();
        debug_assert!(channels > 0 && acc.len().is_multiple_of(channels));
        if channels == CHANNELS {
            unsafe {
                let mut i = 0;
                while i + 32 <= acc.len() {
                    for lane in 0..4 {
                        let wv = _mm256_loadu_ps(w.as_ptr().add(lane * 8));
                        let at = i + lane * 8;
                        let a = _mm256_loadu_ps(acc.as_ptr().add(at));
                        let xv = _mm256_loadu_ps(x.as_ptr().add(at));
                        _mm256_storeu_ps(acc.as_mut_ptr().add(at), _mm256_fmadd_ps(xv, wv, a));
                    }
                    i += 32;
                }
            }
            return;
        }
        ScalarSimd::fma_tiled(acc, x, w);
    }

    #[inline]
    fn mul_add_tap(dst: &mut [f32], src: &[f32], k: f32) {
        unsafe { mul_add_tap_avx2(dst, src, k) }
    }

    #[inline]
    fn conv1x1_1(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
        unsafe { conv1x1_1_avx2(dst, src, w, bias) }
    }

    #[inline]
    fn conv1x1_32(dst: &mut [f32], src: &[f32], w: &[f32], bias: &[f32]) {
        unsafe { conv1x1_32_avx2(dst, src, w, bias) }
    }

    #[inline]
    fn depthwise_3x3(dst: &mut [f32], src: &[f32], w: &[f32], in_len: usize) {
        unsafe { depthwise_3x3_avx2(dst, src, w, in_len) }
    }

    #[inline]
    fn depthwise_1x3s2(dst: &mut [f32], src: &[f32], w: &[f32], pad_left: usize) {
        unsafe { depthwise_1x3s2_avx2(dst, src, w, pad_left) }
    }

    #[inline]
    fn maxpool1x3s2(dst: &mut [f32], src: &[f32], in_len: usize) {
        unsafe { maxpool1x3s2_avx2(dst, src, in_len) }
    }

    #[inline]
    fn gemm_i8(acc: &mut [i32], w: &[i8], zp: &[i8], qx: &[i16], k: usize, n: usize) {
        unsafe { gemm_i8_avx2(acc, w, zp, qx, k, n) }
    }

    #[inline]
    fn quantize_i16(dst: &mut [i16], src: &[f32], inv: f32, clamp: i32) {
        unsafe { quantize_i16_avx2(dst, src, inv, clamp) }
    }

    #[inline]
    fn scale_i32_add(dst: &mut [f32], acc: &[i32], sp: &[f32], bias: &[f32]) {
        unsafe { scale_i32_add_avx2(dst, acc, sp, bias) }
    }

    #[inline]
    fn scale_i32_add_inplace(dst: &mut [f32], acc: &[i32], sp: &[f32]) {
        unsafe { scale_i32_add_inplace_avx2(dst, acc, sp) }
    }

    #[inline]
    fn scale_i32_add_relu(dst: &mut [f32], acc: &[i32], sp: &[f32], bias: &[f32]) {
        unsafe { scale_i32_add_relu_avx2(dst, acc, sp, bias) }
    }

    #[inline]
    fn mul_add2_slices(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
        unsafe { mul_add2_slices_avx2(dst, a, b, c) }
    }

    #[inline]
    fn lstm_activate(y: &mut [f32], h: usize, k: &ActConsts) {
        unsafe { lstm_activate_avx2(y, h, k) }
    }

    #[inline]
    fn lstm_cell(
        cell: &mut [f32],
        qh: &mut [i16],
        y: &[f32],
        inv_h: f32,
        clamp: i32,
        k: &ActConsts,
    ) {
        unsafe { lstm_cell_avx2(cell, qh, y, inv_h, clamp, k) }
    }

    // -----------------------------------------------------------------------
    // echo-model kernels
    // -----------------------------------------------------------------------

    #[inline]
    fn echo_conv1x1(
        dst: &mut [f32],
        src: &[f32],
        w: &[f32],
        bias: &[f32],
        cin: usize,
        cout: usize,
    ) -> bool {
        unsafe { echo_conv1x1_avx2(dst, src, w, bias, cin, cout) }
    }

    #[inline]
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
        unsafe { echo_depthwise_avx2(dst, src, w, bias, channels, taps, fma_mode, bins) }
    }

    #[inline]
    fn echo_maxpool5(dst: &mut [f32], src: &[f32], channels: usize, bins: usize) -> bool {
        unsafe { echo_maxpool5_avx2(dst, src, channels, bins) }
    }

    #[inline]
    fn echo_gemm_i64(acc: &mut [i64], w: &[i8], zp: &[i8], qx: &[i16], k: usize, n: usize) -> bool {
        unsafe { echo_gemm_i64_avx2(acc, w, zp, qx, k, n) }
    }

    #[inline]
    fn echo_deconv(
        dst: &mut [f32],
        src: &[f32],
        w: &[f32],
        bias: &[f32],
        hin: usize,
        channels: usize,
    ) -> bool {
        unsafe { echo_deconv_avx2(dst, src, w, bias, hin, channels) }
    }

    #[inline]
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
        unsafe { echo_conv3_avx2(dst, src, w, bias, bins, cin, cout, leaky, slope) }
    }

    #[inline]
    fn echo_conv3_out1(
        dst: &mut [f32],
        src: &[f32],
        w: &[f32],
        bias: &[f32],
        bins: usize,
        cin: usize,
    ) -> bool {
        unsafe { echo_conv3_out1_avx2(dst, src, w, bias, bins, cin) }
    }

    #[inline]
    fn echo_mm33(dst: &mut [f32], src: &[f32], w: &[f32]) -> bool {
        unsafe { echo_mm33_avx2(dst, src, w) }
    }

    #[inline]
    fn echo_quantize_input(dst: &mut [i16], src: &[f32], quant_scale: f32) -> f32 {
        unsafe { echo_quantize_input_avx2(dst, src, quant_scale) }
    }

    #[inline]
    fn echo_quantize_hidden(
        dst: &mut [i16],
        src: &[f32],
        hidden_scale: f32,
        hidden_scale_wide: f32,
    ) -> f32 {
        unsafe { echo_quantize_hidden_avx2(dst, src, hidden_scale, hidden_scale_wide) }
    }

    #[inline]
    fn echo_gru_tail(yih: &mut [f32], yhh: &[f32], h: &mut [f32], k: &EchoActConsts) -> bool {
        unsafe { echo_gru_tail_avx2(yih, yhh, h, k) }
    }

    #[inline]
    fn echo_sigmoid88(dst: &mut [f32], src: &[f32], k: &EchoActConsts) -> bool {
        unsafe { echo_sigmoid88_avx2(dst, src, k) }
    }
}

// ---------------------------------------------------------------------------
// tests: the AVX2 kernels must be bit-identical to the scalar reference on
// adversarial integer and float inputs, not just on the model sequences.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::Backend;

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
        if !Backend::Avx2.is_available() {
            eprintln!("avx2 backend unavailable, skipping");
            return true;
        }
        false
    }

    #[test]
    fn gemm_matches_scalar_bit_for_bit() {
        if skip() {
            return;
        }
        let mut rng = Rng(0x1234_5678_9abc_def0);
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
            Avx2Simd::gemm_i8(&mut b, &w, &zp, &qx, k, n);
            assert_eq!(a, b, "gemm mismatch for k={k} n={n}");
        }
        // Sparse rows (the reference skips zero inputs) and the extreme bound.
        let (k, n) = (64usize, 96usize);
        let w: Vec<i8> = (0..k * n)
            .map(|i| if i % 3 == 0 { 127 } else { -128 })
            .collect();
        let zp = vec![0i8; n];
        let mut qx = vec![0i16; k];
        qx[1] = 8192;
        qx[k - 1] = -8192;
        let mut a = vec![0i32; n];
        let mut b = vec![0i32; n];
        ScalarSimd::gemm_i8(&mut a, &w, &zp, &qx, k, n);
        Avx2Simd::gemm_i8(&mut b, &w, &zp, &qx, k, n);
        assert_eq!(a, b);
    }

    #[test]
    fn convolutions_and_depthwise_match_scalar() {
        if skip() {
            return;
        }
        let mut rng = Rng(0xdead_beef_0bad_f00d);
        let positions = 37usize;
        let src: Vec<f32> = (0..positions * CHANNELS).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..CHANNELS * CHANNELS).map(|_| rng.f32() * 0.25).collect();
        let bias: Vec<f32> = (0..CHANNELS).map(|_| rng.f32()).collect();
        let mut a = vec![0.0f32; positions * CHANNELS];
        let mut b = vec![0.0f32; positions * CHANNELS];
        ScalarSimd::conv1x1_32(&mut a, &src, &w, &bias);
        Avx2Simd::conv1x1_32(&mut b, &src, &w, &bias);
        assert_eq!(a, b);

        let single: Vec<f32> = (0..positions).map(|_| rng.f32()).collect();
        let w1: Vec<f32> = (0..CHANNELS).map(|_| rng.f32() * 0.5).collect();
        ScalarSimd::conv1x1_1(&mut a, &single, &w1, &bias);
        Avx2Simd::conv1x1_1(&mut b, &single, &w1, &bias);
        assert_eq!(a, b);

        let in_len = 255usize;
        let src3: Vec<f32> = (0..3 * in_len * CHANNELS).map(|_| rng.f32()).collect();
        let w3: Vec<f32> = (0..9 * CHANNELS).map(|_| rng.f32()).collect();
        let mut a = vec![0.0f32; (in_len - 2) * CHANNELS];
        let mut b = vec![0.0f32; (in_len - 2) * CHANNELS];
        ScalarSimd::depthwise_3x3(&mut a, &src3, &w3, in_len);
        Avx2Simd::depthwise_3x3(&mut b, &src3, &w3, in_len);
        assert_eq!(a, b);

        for &(len, pad) in &[(126usize, 0usize), (63, 1), (32, 1)] {
            let src: Vec<f32> = (0..len * CHANNELS).map(|_| rng.f32()).collect();
            let w: Vec<f32> = (0..3 * CHANNELS).map(|_| rng.f32()).collect();
            let mut a = vec![0.0f32; len.div_ceil(2) * CHANNELS];
            let mut b = vec![0.0f32; len.div_ceil(2) * CHANNELS];
            ScalarSimd::depthwise_1x3s2(&mut a, &src, &w, pad);
            Avx2Simd::depthwise_1x3s2(&mut b, &src, &w, pad);
            assert_eq!(a, b);
        }

        // Pooling: include signed zeros and exact ties so the tie-selection
        // rule (`if a > b { a } else { b }`) is covered.
        let len = 63usize;
        let mut src: Vec<f32> = (0..3 * len * CHANNELS).map(|_| rng.f32()).collect();
        for chunk in src.chunks_mut(4) {
            chunk[0] = 0.0;
            if chunk.len() > 1 {
                chunk[1] = -0.0;
            }
        }
        let out_len = (len - 3) / 2 + 1;
        let mut a = vec![0.0f32; 3 * out_len * CHANNELS];
        let mut b = vec![0.0f32; 3 * out_len * CHANNELS];
        ScalarSimd::maxpool1x3s2(&mut a, &src, len);
        Avx2Simd::maxpool1x3s2(&mut b, &src, len);
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn quantiser_matches_scalar_on_extremes() {
        if skip() {
            return;
        }
        let mut rng = Rng(0x0f0f_1e1e_2d2d_3c3c);
        let mut src: Vec<f32> = (0..129).map(|_| rng.f32() * 5000.0).collect();
        src[0] = 0.0;
        src[1] = -0.0;
        src[2] = f32::INFINITY;
        src[3] = f32::NEG_INFINITY;
        src[4] = f32::NAN;
        src[5] = 8192.5;
        src[6] = -8192.5;
        src[7] = 1e30;
        src[8] = -1e30;
        src[9] = 16384.5;
        for &clamp in &[8192i32, 16384] {
            let inv = if clamp == 8192 {
                1.0 / 8192.0
            } else {
                1.0 / 16384.0
            };
            let mut a = vec![0i16; src.len()];
            let mut b = vec![0i16; src.len()];
            ScalarSimd::quantize_i16(&mut a, &src, inv, clamp);
            Avx2Simd::quantize_i16(&mut b, &src, inv, clamp);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn epilogues_match_scalar_on_extremes() {
        if skip() {
            return;
        }
        // NaN payloads may differ between the scalar and vector operations;
        // every other bit pattern must match exactly, and a NaN must stay a
        // NaN (or become +0 in the ReLU case).
        fn same(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len()
                && a.iter().zip(b).all(|(&x, &y)| {
                    x.to_bits() == y.to_bits()
                        || (x.is_nan() && y.is_nan())
                        || (x == y && x == 0.0 && x.is_sign_negative() == y.is_sign_negative())
                })
        }
        let mut rng = Rng(0x5a5a_1234_9876_4321);
        for &n in &[64usize, 65, 7, 1024, 1026] {
            let acc: Vec<i32> = (0..n)
                .map(|i| match i % 9 {
                    0 => i32::MIN,
                    1 => i32::MAX,
                    2 => 0,
                    3 => -1,
                    _ => rng.i16(16384) as i32,
                })
                .collect();
            let sp: Vec<f32> = (0..n)
                .map(|i| match i % 11 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::INFINITY,
                    3 => f32::NEG_INFINITY,
                    4 => f32::NAN,
                    _ => rng.f32(),
                })
                .collect();
            let bias: Vec<f32> = (0..n)
                .map(|i| match i % 13 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    3 => -0.0,
                    _ => rng.f32(),
                })
                .collect();

            let mut a = vec![0.0f32; n];
            let mut b = vec![0.0f32; n];
            ScalarSimd::scale_i32_add(&mut a, &acc, &sp, &bias);
            Avx2Simd::scale_i32_add(&mut b, &acc, &sp, &bias);
            assert!(same(&a, &b), "scale_i32_add n={n}");

            let base: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let mut a = base.clone();
            let mut b = base.clone();
            ScalarSimd::scale_i32_add_inplace(&mut a, &acc, &sp);
            Avx2Simd::scale_i32_add_inplace(&mut b, &acc, &sp);
            assert!(same(&a, &b), "scale_i32_add_inplace n={n}");

            let mut a = vec![0.0f32; n];
            let mut b = vec![0.0f32; n];
            ScalarSimd::scale_i32_add_relu(&mut a, &acc, &sp, &bias);
            Avx2Simd::scale_i32_add_relu(&mut b, &acc, &sp, &bias);
            assert!(same(&a, &b), "scale_i32_add_relu n={n}");

            let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let y: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let z: Vec<f32> = (0..n)
                .map(|i| match i % 7 {
                    0 => f32::NAN,
                    1 => -0.0,
                    _ => rng.f32(),
                })
                .collect();
            let mut a = vec![0.0f32; n];
            let mut b = vec![0.0f32; n];
            ScalarSimd::mul_add2_slices(&mut a, &x, &y, &z);
            Avx2Simd::mul_add2_slices(&mut b, &x, &y, &z);
            assert!(same(&a, &b), "mul_add2_slices n={n}");
        }
    }

    #[test]
    fn activations_match_scalar() {
        if skip() {
            return;
        }
        let k = ActConsts {
            log2e: std::f32::consts::LOG2_E,
            ln2: -std::f32::consts::LN_2,
            exp_poly: [
                1.0 / 40320.0,
                1.0 / 5040.0,
                1.0 / 720.0,
                1.0 / 120.0,
                1.0 / 24.0,
            ],
            half: 0.5,
            one: 1.0,
            two: 2.0,
            four: 4.0,
            neg_four: -4.0,
            exp_clamp: 88.0,
            neg_exp_clamp: -88.0,
            cell_poly: [1.0, 2.0, 3.0, 60.0, 4.0],
        };
        let mut rng = Rng(0xfeed_face_cafe_babe);
        let mut y: Vec<f32> = (0..4 * 256).map(|_| (rng.f32() - 0.0) * 6.0).collect();
        for v in y.iter_mut().take(16) {
            *v = 0.0;
        }
        y[16] = -0.0;
        y[17] = 1000.0;
        y[18] = -1000.0;
        y[19] = 88.0;
        y[20] = -88.0;
        let mut want = y.clone();
        ScalarSimd::lstm_activate(&mut want, 256, &k);
        Avx2Simd::lstm_activate(&mut y, 256, &k);
        assert_eq!(
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let mut cell: Vec<f32> = (0..256).map(|_| rng.f32()).collect();
        cell[0] = -0.0;
        cell[1] = 3.9;
        cell[2] = -3.9;
        cell[3] = 100.0;
        let mut want_cell = cell.clone();
        let mut want_qh = vec![0i16; 256];
        ScalarSimd::lstm_cell(&mut want_cell, &mut want_qh, &want, 1.0 / 8192.0, 8192, &k);
        let mut got_qh = vec![0i16; 256];
        Avx2Simd::lstm_cell(&mut cell, &mut got_qh, &want, 1.0 / 8192.0, 8192, &k);
        assert_eq!(
            want_cell.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            cell.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(want_qh, got_qh);
    }

    // -----------------------------------------------------------------------
    // echo kernels: every one must be bit-identical to the scalar reference
    // on adversarial random data, independent of the model sequences.
    // -----------------------------------------------------------------------
    fn f32_bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn echo_kernels_match_scalar() {
        if skip() {
            return;
        }
        let mut rng = Rng(0x5eed_ec40_1234_abcd);
        let act = EchoActConsts {
            one: 1.0,
            half: 0.5,
            eight: 8.0,
            neg_eight: -8.0,
            four: 4.0,
            neg_four: -4.0,
            sig: [1.0, 60.0, 2.0, 3.0, 0.1, 1.0, 0.05],
            tanh: [1.0, 2.0, 3.0, 60.0, 4.0],
            sig88_clamp: 88.37625885,
            sig88_neg_clamp: -88.37625885,
            exp_log2e: std::f32::consts::LOG2_E,
            exp_ln2: -std::f32::consts::LN_2,
            exp_c: [
                1.0 / 40320.0,
                1.0 / 5040.0,
                1.0 / 720.0,
                1.0 / 120.0,
                1.0 / 24.0,
            ],
        };

        // Quantisers: body RNE, half-away tail, clamp, all-zero branch.
        for &(n, zero) in &[(128usize, false), (513, false), (2112, false), (128, true)] {
            let mut src: Vec<f32> = (0..n).map(|_| rng.f32() * 2000.0).collect();
            if zero {
                src.fill(0.0);
            } else {
                src[0] = -3000.0;
                src[1] = 3000.0;
            }
            let mut a = vec![0i16; n];
            let mut b = vec![0i16; n];
            let sa = <ScalarSimd as Simd>::echo_quantize_input(&mut a, &src, 8192.0);
            let sb = <Avx2Simd as Simd>::echo_quantize_input(&mut b, &src, 8192.0);
            assert_eq!(a, b, "echo_quantize_input n={n} zero={zero}");
            assert_eq!(sa.to_bits(), sb.to_bits());
        }
        for &wide in &[false, true] {
            let mut src: Vec<f32> = (0..128)
                .map(|_| {
                    if wide {
                        rng.f32() * 2.0
                    } else {
                        rng.f32() * 0.25
                    }
                })
                .collect();
            src[3] = -0.0;
            let mut a = vec![0i16; 128];
            let mut b = vec![0i16; 128];
            let sa = <ScalarSimd as Simd>::echo_quantize_hidden(&mut a, &src, 127.0, 32767.0);
            let sb = <Avx2Simd as Simd>::echo_quantize_hidden(&mut b, &src, 127.0, 32767.0);
            assert_eq!(a, b, "echo_quantize_hidden wide={wide}");
            assert_eq!(sa.to_bits(), sb.to_bits());
        }

        // i64 GEMM used by the K = 2112 GRU input.
        {
            let (k, n) = (2112usize, 384usize);
            let w: Vec<i8> = (0..k * n).map(|_| rng.i8()).collect();
            let zp: Vec<i8> = (0..n).map(|_| rng.i8()).collect();
            let qx: Vec<i16> = (0..k).map(|_| rng.i16(8192)).collect();
            let mut a = vec![0i64; n];
            let mut b = vec![0i64; n];
            <ScalarSimd as Simd>::echo_gemm_i64(&mut a, &w, &zp, &qx, k, n);
            <Avx2Simd as Simd>::echo_gemm_i64(&mut b, &w, &zp, &qx, k, n);
            assert_eq!(a, b, "echo_gemm_i64");
        }

        // 1x1 convolutions (all four output widths).
        for &(cin, cout, pos) in &[
            (12usize, 16usize, 65usize),
            (16, 16, 33),
            (16, 32, 33),
            (32, 32, 33),
            (64, 64, 33),
            (64, 64, 3),
        ] {
            let src: Vec<f32> = (0..pos * cin).map(|_| rng.f32()).collect();
            let w: Vec<f32> = (0..cin * cout).map(|_| rng.f32() * 0.25).collect();
            let bias: Vec<f32> = (0..cout).map(|_| rng.f32()).collect();
            let mut a = vec![0.0f32; pos * cout];
            let mut b = vec![0.0f32; pos * cout];
            <ScalarSimd as Simd>::echo_conv1x1(&mut a, &src, &w, &bias, cin, cout);
            <Avx2Simd as Simd>::echo_conv1x1(&mut b, &src, &w, &bias, cin, cout);
            assert_eq!(f32_bits(&a), f32_bits(&b), "conv1x1 {cin}->{cout}");
        }

        // Depthwise 5x1 (both add modes) and 3x1.
        for &(ch, taps, fma_mode, bins) in &[
            (12usize, 5usize, false, 513usize),
            (16, 5, true, 257),
            (32, 5, true, 65),
            (64, 3, true, 33),
        ] {
            let src: Vec<f32> = (0..bins * ch).map(|_| rng.f32()).collect();
            let w: Vec<f32> = (0..ch * taps).map(|_| rng.f32()).collect();
            let bias: Vec<f32> = (0..ch).map(|_| rng.f32()).collect();
            let mut a = vec![0.0f32; bins * ch];
            let mut b = vec![0.0f32; bins * ch];
            <ScalarSimd as Simd>::echo_depthwise(&mut a, &src, &w, &bias, ch, taps, fma_mode, bins);
            <Avx2Simd as Simd>::echo_depthwise(&mut b, &src, &w, &bias, ch, taps, fma_mode, bins);
            assert_eq!(
                f32_bits(&a),
                f32_bits(&b),
                "depthwise ch={ch} taps={taps} fma={fma_mode} bins={bins}"
            );
        }

        // Max pool 5x1 stride 2, with signed zeros and exact ties.
        for &(ch, bins) in &[(16usize, 513usize), (32, 65)] {
            let mut src: Vec<f32> = (0..bins * ch).map(|_| rng.f32()).collect();
            for chunk in src.chunks_mut(7) {
                chunk[0] = 0.0;
                if chunk.len() > 1 {
                    chunk[1] = -0.0;
                }
                if chunk.len() > 2 {
                    chunk[2] = 1.5;
                    chunk[3 % chunk.len()] = 1.5;
                }
            }
            let out_bins = (bins - 1) / 2 + 1;
            let mut a = vec![0.0f32; out_bins * ch];
            let mut b = vec![0.0f32; out_bins * ch];
            <ScalarSimd as Simd>::echo_maxpool5(&mut a, &src, ch, bins);
            <Avx2Simd as Simd>::echo_maxpool5(&mut b, &src, ch, bins);
            assert_eq!(f32_bits(&a), f32_bits(&b), "maxpool ch={ch} bins={bins}");
        }

        // Depthwise deconvolution.
        for &(hin, ch) in &[(33usize, 32usize), (257, 16)] {
            let src: Vec<f32> = (0..hin * ch).map(|_| rng.f32()).collect();
            let w: Vec<f32> = (0..ch * 5).map(|_| rng.f32()).collect();
            let bias: Vec<f32> = (0..ch).map(|_| rng.f32()).collect();
            let hout = 2 * hin - 1;
            let mut a = vec![0.0f32; hout * ch];
            let mut b = vec![0.0f32; hout * ch];
            <ScalarSimd as Simd>::echo_deconv(&mut a, &src, &w, &bias, hin, ch);
            <Avx2Simd as Simd>::echo_deconv(&mut b, &src, &w, &bias, hin, ch);
            assert_eq!(f32_bits(&a), f32_bits(&b), "deconv hin={hin} ch={ch}");
        }

        // 3x1 convolutions (leaky) and the linear outC=1 form.
        for &(bins, cin, cout, leaky) in &[
            (65usize, 32usize, 32usize, true),
            (129, 32, 16, true),
            (257, 16, 16, true),
            (65, 32, 32, false),
        ] {
            let src: Vec<f32> = (0..bins * cin).map(|_| rng.f32()).collect();
            let w: Vec<f32> = (0..cout * cin * 3).map(|_| rng.f32()).collect();
            let bias: Vec<f32> = (0..cout).map(|_| rng.f32()).collect();
            let mut a = vec![0.0f32; bins * cout];
            let mut b = vec![0.0f32; bins * cout];
            <ScalarSimd as Simd>::echo_conv3(&mut a, &src, &w, &bias, bins, cin, cout, leaky, 0.01);
            <Avx2Simd as Simd>::echo_conv3(&mut b, &src, &w, &bias, bins, cin, cout, leaky, 0.01);
            assert_eq!(
                f32_bits(&a),
                f32_bits(&b),
                "conv3 bins={bins} cin={cin} cout={cout}"
            );
        }
        {
            let (bins, cin) = (513usize, 16usize);
            let src: Vec<f32> = (0..bins * cin).map(|_| rng.f32()).collect();
            let w: Vec<f32> = (0..3 * cin).map(|_| rng.f32()).collect();
            let bias = [rng.f32()];
            let mut a = vec![0.0f32; bins];
            let mut b = vec![0.0f32; bins];
            <ScalarSimd as Simd>::echo_conv3_out1(&mut a, &src, &w, &bias, bins, cin);
            <Avx2Simd as Simd>::echo_conv3_out1(&mut b, &src, &w, &bias, bins, cin);
            assert_eq!(f32_bits(&a), f32_bits(&b), "conv3_out1");
        }

        // 33x33 MatMul.
        {
            let src: Vec<f32> = (0..33 * 64).map(|_| rng.f32()).collect();
            let w: Vec<f32> = (0..33 * 33).map(|_| rng.f32()).collect();
            let mut a = vec![0.0f32; 33 * 64];
            let mut b = vec![0.0f32; 33 * 64];
            <ScalarSimd as Simd>::echo_mm33(&mut a, &src, &w);
            <Avx2Simd as Simd>::echo_mm33(&mut b, &src, &w);
            assert_eq!(f32_bits(&a), f32_bits(&b), "mm33");
        }

        // GRU tail: gate sums, sigmoid/tanh, and the no-FMA hidden update.
        {
            let hidden = 128usize;
            let yih: Vec<f32> = (0..3 * hidden).map(|_| rng.f32() * 8.0).collect();
            let yhh: Vec<f32> = (0..3 * hidden).map(|_| rng.f32() * 8.0).collect();
            let h: Vec<f32> = (0..hidden).map(|_| rng.f32()).collect();
            let mut a_y = yih.clone();
            let mut b_y = yih.clone();
            let mut a_h = h.clone();
            let mut b_h = h.clone();
            <ScalarSimd as Simd>::echo_gru_tail(&mut a_y, &yhh, &mut a_h, &act);
            <Avx2Simd as Simd>::echo_gru_tail(&mut b_y, &yhh, &mut b_h, &act);
            assert_eq!(f32_bits(&a_y), f32_bits(&b_y), "gru tail y");
            assert_eq!(f32_bits(&a_h), f32_bits(&b_h), "gru tail h");
        }

        // Final sigmoid, including the scalar tail element (element 512).
        {
            let n = 513usize;
            let mut src: Vec<f32> = (0..n).map(|_| rng.f32() * 30.0).collect();
            src[0] = -1000.0;
            src[1] = 1000.0;
            src[2] = -0.0;
            src[3] = 88.37625885;
            src[4] = -88.37625885;
            let mut a = vec![0.0f32; n];
            let mut b = vec![0.0f32; n];
            <ScalarSimd as Simd>::echo_sigmoid88(&mut a, &src, &act);
            <Avx2Simd as Simd>::echo_sigmoid88(&mut b, &src, &act);
            assert_eq!(f32_bits(&a), f32_bits(&b), "sigmoid88");
        }
    }
}
