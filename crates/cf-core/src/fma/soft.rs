//! Correctly-rounded `f32` FMA without a hardware FMA instruction.
//!
//! Portability (notably `wasm32`) has no deterministic hardware FMA, so the
//! `soft-fma` feature compiles a software implementation.  It still has to be
//! *correctly rounded*: every bit of the result must equal the hardware
//! `vfmadd`/`fmaf` result, because the wasm build is compared against a native
//! oracle bit for bit.
//!
//! # Algorithm
//!
//! The implementation uses `f64` as an exact intermediate, which is sound
//! because the operands are `f32`:
//!
//! 1. `p = (f64)a * (f64)b` is **exact**: a 24-bit by 24-bit significand
//!    product needs 48 bits, and `f64` has 53.  The exponent range of the
//!    product of two `f32` values (`2^-298 .. 2^256`) is comfortably inside
//!    the `f64` normal range, so no underflow or overflow occurs either.
//! 2. `s = p + (f64)c` is the round-to-nearest-even `f64` sum, and the
//!    TwoSum error-free transform gives the **exact** residual `err`, so
//!    `p + c = s + err` exactly.
//! 3. `s` is adjusted to *round-to-odd* precision (the last significand bit
//!    is forced to one in the direction of `err`) and then narrowed to `f32`
//!    with the normal round-to-nearest-even conversion.
//!
//! Rounding to odd at 53 bits and then to 24 bits is exactly equivalent to a
//! single rounding to 24 bits whenever the first precision is at least the
//! second plus two (53 >= 24 + 2).  That is the classical double-rounding
//! theorem; it also covers the `f32` subnormal target range because the
//! effective precision near the result only shrinks.  No host FPU helper is
//! used: only IEEE-754 `f64` multiply/add/subtract and a `f64 -> f32`
//! conversion, all of which WASM defines deterministically.
//!
//! The old integer-mantissa implementation is kept in the module history; the
//! `fma_check` suite (10M adversarial cases, midpoints, subnormals, overflow
//! boundaries) proves this one against the native correctly-rounded FMA.

use super::FmaOps;

const SIGN_BIT: u32 = 0x8000_0000;
const INF_BITS: u32 = 0x7F80_0000;
const QNAN_BITS: u32 = 0x7FC0_0000;

#[derive(Clone, Copy, Debug, Default)]
pub struct SoftFma;

impl FmaOps for SoftFma {
    const NAME: &'static str = "soft";

    #[inline]
    fn fma(self, a: f32, b: f32, c: f32) -> f32 {
        soft_fma(a, b, c)
    }
}

/// Correctly-rounded `f32` FMA (round to nearest, ties to even).
#[inline]
pub fn soft_fma(a: f32, b: f32, c: f32) -> f32 {
    let ba = a.to_bits();
    let bb = b.to_bits();
    let bc = c.to_bits();

    let (ea, fa) = ((ba >> 23) & 0xFF, ba & 0x7F_FFFF);
    let (eb, fb) = ((bb >> 23) & 0xFF, bb & 0x7F_FFFF);
    let (ec, fc) = ((bc >> 23) & 0xFF, bc & 0x7F_FFFF);

    // Any NaN input -> a quiet NaN.  The payload is deliberately canonical:
    // the model never produces NaN, and the verifier treats NaN == NaN.
    if (ea == 0xFF && fa != 0) || (eb == 0xFF && fb != 0) || (ec == 0xFF && fc != 0) {
        return f32::from_bits(QNAN_BITS);
    }

    let a_zero = ea == 0 && fa == 0;
    let b_zero = eb == 0 && fb == 0;
    let a_inf = ea == 0xFF;
    let b_inf = eb == 0xFF;
    let c_inf = ec == 0xFF;
    let product_sign = ((ba ^ bb) >> 31) != 0;

    if a_inf || b_inf {
        // inf * 0 is invalid.
        if a_zero || b_zero {
            return f32::from_bits(QNAN_BITS);
        }
        if c_inf && ((bc >> 31) != 0) != product_sign {
            return f32::from_bits(QNAN_BITS);
        }
        return f32::from_bits(if product_sign {
            SIGN_BIT | INF_BITS
        } else {
            INF_BITS
        });
    }
    if c_inf {
        return c;
    }

    // All three operands are finite.  Signed zeros are handled by the IEEE
    // rules of the `f64` add below (`(+0) + (-0) = +0`, equal signs kept), so
    // no separate zero branch is needed.
    let af = a as f64;
    let bf = b as f64;
    let cf = c as f64;

    let p = af * bf; // exact: 24 x 24 -> 48 significand bits
    let s = p + cf; // rounded to f64
    let bv = s - p;
    let err = (p - (s - bv)) + (cf - bv); // exact residual: p + cf == s + err

    let s_ro = round_to_odd(s, err);
    s_ro as f32
}

/// Round `s + err` to odd `f64` precision.
///
/// `s` is the round-to-nearest-even `f64` sum and `err` its exact residual, so
/// `|err| < ulp(s)/2` (or `err == 0`).  When the residual is non-zero the exact
/// value is not representable; the round-to-odd result is the neighbouring
/// `f64` whose last significand bit is one.
#[inline]
fn round_to_odd(s: f64, err: f64) -> f64 {
    let bits = s.to_bits();
    // Common case: the sum is already odd, or exact.
    if (bits & 1) != 0 || err == 0.0 {
        return s;
    }
    // `s` cannot be zero when `err != 0` (the exact sum would have to be a
    // non-zero multiple of 2^-298, which is representable as a normal `f64`).
    // For non-negative `s`, increasing the bit pattern moves toward +inf; for
    // negative `s` the sign-magnitude encoding makes it move toward -inf.
    let toward_greater = err > 0.0;
    let non_negative = bits >> 63 == 0;
    let next = if toward_greater == non_negative {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next)
}

#[cfg(test)]
mod tests {
    use super::soft_fma;

    #[test]
    fn special_values() {
        // inf * 0 is invalid.
        assert!(soft_fma(f32::INFINITY, 0.0, 1.0).is_nan());
        // inf handling keeps the sign.
        assert_eq!(soft_fma(f32::INFINITY, 2.0, 1.0), f32::INFINITY);
        assert_eq!(soft_fma(f32::NEG_INFINITY, 2.0, 1.0), f32::NEG_INFINITY);
        // inf - inf is invalid.
        assert!(soft_fma(f32::INFINITY, 1.0, f32::NEG_INFINITY).is_nan());
        // Signed-zero rules.
        assert_eq!(soft_fma(-1.0, 0.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(soft_fma(-1.0, 0.0, 0.0).to_bits(), 0);
        assert_eq!(soft_fma(0.0, -1.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(soft_fma(0.0, -1.0, 0.0).to_bits(), 0);
        // Exact cancellation yields +0 in round-to-nearest.
        assert_eq!(soft_fma(3.0, 0.5, -1.5).to_bits(), 0);
    }

    #[test]
    fn midpoint_subnormal_steering() {
        // 4097 * 4097 = 2^24 + 8193: exactly halfway between two f32 values.
        // The tie rounds to even (2^24 + 8192); a subnormal perturbation
        // steers it in the direction of its sign.
        let a = 4097.0f32;
        let b = 4097.0f32;
        assert_eq!(soft_fma(a, b, 0.0), 16_785_408.0);
        let above = f32::from_bits(1); // smallest positive subnormal
        assert_eq!(soft_fma(a, b, above), 16_785_410.0);
        assert_eq!(soft_fma(a, b, -above), 16_785_408.0);
    }

    #[test]
    fn underflow_and_overflow() {
        let min_sub = f32::from_bits(1);
        assert_eq!(soft_fma(min_sub, min_sub, 0.0).to_bits(), 0);
        assert_eq!(soft_fma(min_sub, 1.0, min_sub).to_bits(), 2);
        let max = f32::MAX;
        assert_eq!(soft_fma(max, 2.0, 0.0), f32::INFINITY);
        assert_eq!(soft_fma(max, 1.0, -max).to_bits(), 0);
    }

    #[test]
    fn cancellation_path_adds_the_product() {
        // The near-cancellation path must add `a*b` and `c` exactly, not `a`
        // and `c`.  These cases have `a*b` almost equal to `-c`, so the exact
        // result only has a few bits of magnitude.
        let cases = [
            (3.0f32, 0.1f32, -0.3f32),
            (-128.0f32, core::f32::consts::LN_2, 88.723_785f32),
            (1.0f32 + f32::EPSILON, 1.0f32, -1.0f32),
            (7.0f32, 1.0f32 / 3.0f32, -7.0f32 / 3.0f32),
            (1e20f32, 1.0f32, -1e20f32 + 1e13f32),
        ];
        for (a, b, c) in cases {
            assert_eq!(
                soft_fma(a, b, c).to_bits(),
                a.mul_add(b, c).to_bits(),
                "fma({a:e}, {b:e}, {c:e})"
            );
        }
    }

    #[test]
    fn cancellation_range_reduction_matches_native() {
        // Dense sweep over the exact cancellation pattern used by an
        // exponential range reduction: `fma(n, ln2, t)` with `t` near
        // `-n*ln2`, so the result is a small residual.
        let ln2 = core::f32::consts::LN_2;
        let mut checked = 0u64;
        for n in -128i32..=128 {
            for step in -64i32..=64 {
                let t = (n as f32) * ln2 + (step as f32) * (1.0 / 512.0);
                assert_eq!(
                    soft_fma(n as f32, ln2, t).to_bits(),
                    (n as f32).mul_add(ln2, t).to_bits(),
                    "n={n} step={step} t={t:e}"
                );
                checked += 1;
            }
        }
        assert!(checked > 16_000);
    }

    #[test]
    fn double_rounding_edges_match_native() {
        // Values just above/below an f32 rounding boundary, where a naive
        // "compute in f64 and narrow" implementation would double-round.
        let mut bad = 0u64;
        let mut checked = 0u64;
        for e in 100u32..160 {
            let base = f32::from_bits((e << 23) | 0x7F_FFFE);
            for delta in 0..8u32 {
                let c = f32::from_bits(base.to_bits().wrapping_add(delta));
                let a = 1.0f32 + (delta as f32) * f32::EPSILON;
                let b = -base;
                let want = a.mul_add(b, c);
                let got = soft_fma(a, b, c);
                checked += 1;
                if want.to_bits() != got.to_bits() {
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "double-rounding mismatch in {bad}/{checked} cases");
        assert!(checked >= 480);
    }
}
