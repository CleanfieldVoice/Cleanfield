//! Deterministic software exponential for the final sigmoid tail.
//!
//! The reference engine evaluates the one scalar tail element of the final
//! sigmoid as `1 / (f32(exp(-x)) + 1)` with a double-precision exponential and
//! a real `f32` division.  Native builds can call the platform `exp` for that,
//! but `wasm32-unknown-unknown` has no host libm and the platform exponential
//! is not part of the numeric contract.
//!
//! [`exp_f32`] is a portable software exponential built from integer/fixed
//! arithmetic and IEEE-754 `f64` primitive operations only (no libm, no host
//! call, no `f64::exp`, no `powi`):
//!
//! 1. range reduction `n = rne(x * log2(e))`, `r = x - n*ln2` with `ln2` split
//!    into a high part (exact product with the small integer `n`) and a low
//!    correction, so `r` is accurate well past `f32` precision;
//! 2. `exp(r)` by a degree-16 Taylor polynomial evaluated with `i128`
//!    fixed-point (`Q63`) Horner steps, `|r| <= ln2/2`;
//! 3. the `2^n` scale is applied while converting the fixed-point result to an
//!    `f64` with a round-to-nearest-even significand and a sticky bit; the
//!    final `f32` is the IEEE cast of that `f64`, which is exactly what the
//!    reference's `f32(exp_double(x))` does.
//!
//! The fixed-point step is deliberately taken in integer arithmetic so the
//! result cannot depend on the host's floating-point library.  The few `f64`
//! operations that remain are add/subtract/multiply and one integer-valued
//! conversion; IEEE-754 defines all of them bit for bit, including on wasm.

/// `log2(e)` as an `f64` (structural constant, not a model constant).
const LOG2E: f64 = f64::from_bits(0x3FF7_1547_652B_82FE);
/// High part of `ln2`; the low 20 significand bits are zero, so `n * LN2_HI`
/// is exact for the small integer `n` this module produces.
const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
/// Low correction: `ln2 - LN2_HI` rounded to an `f64`.
const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
/// `2^63`, exactly representable.
const TWO63: f64 = 9223372036854775808.0;
/// `2^52`, used by the round-to-nearest-even trick.
const TWO52: f64 = 4503599627370496.0;
/// Highest Taylor degree used by [`exp_f32`]; `ln2/2` to this power is far
/// below the `f32` precision.
const TAYLOR_DEGREE: usize = 16;

/// `1/k!` in `Q63` fixed point, rounded to nearest.
const fn inv_fact_q63(k: u32) -> i128 {
    let mut f: i128 = 1;
    let mut i: i128 = 2;
    while i <= k as i128 {
        f *= i;
        i += 1;
    }
    ((1i128 << 63) + f / 2) / f
}

const INV_FACT: [i128; TAYLOR_DEGREE + 1] = {
    let mut out = [0i128; TAYLOR_DEGREE + 1];
    let mut k = 0usize;
    while k <= TAYLOR_DEGREE {
        out[k] = inv_fact_q63(k as u32);
        k += 1;
    }
    out
};

/// Round to the nearest integer, ties to even, without a libm call.
///
/// The classic `+ 2^52 - 2^52` form: the addition rounds the value to an
/// integer using the current (nearest-even) rounding mode; the subtraction is
/// exact.  Valid for `|x| < 2^51`, which covers the range reduction here.
#[inline]
fn round_ties_even(x: f64) -> f64 {
    let biased = x + TWO52;
    biased - TWO52
}

/// Software `exp(x)` for the `f32` tail path.
///
/// Returns the same `f32` the native reference obtains from
/// `f32(exp_double(x))` for the tested input range.  Out-of-`f32`-range inputs
/// are saturated exactly the way the reference cast behaves: arguments above
/// the `f32` overflow threshold give `+inf`, and arguments below the `f32`
/// underflow threshold give `+0`.  The exact cut-offs are conservative
/// (`-120`/`90`) and lie outside the range where an `f32` result is non-zero
/// and finite, so the result bits are unaffected.
pub fn exp_f32(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    // f32 overflow threshold is ln(f32::MAX) ~= 88.72284; underflow of the
    // f32 cast happens below ~ -103.972.  Keep a wide margin so the fixed
    // conversion never has to build a subnormal f64.
    if x > 90.0 {
        return f32::INFINITY;
    }
    if x < -120.0 {
        return 0.0;
    }
    let xf = x as f64;

    // Range reduction: x = n*ln2 + r, |r| <= ln2/2.
    let n = round_ties_even(xf * LOG2E);
    let ni = n as i32;
    let nf = ni as f64;
    let r = (xf - nf * LN2_HI) - nf * LN2_LO;

    // exp(r) in Q63 fixed point: rq = r * 2^63 (the f64 product is already an
    // integer at this magnitude; any truncation is below 2^-63).
    let rq = (r * TWO63) as i128;
    let mut p = INV_FACT[TAYLOR_DEGREE];
    let mut k = TAYLOR_DEGREE;
    while k > 0 {
        k -= 1;
        // p = p*r + 1/k!, rounded to nearest at the Q63 boundary.
        p = ((p * rq + (1i128 << 62)) >> 63) + INV_FACT[k];
    }

    // p = exp(r) * 2^63.  Round it to a 53-bit significand (ties to even,
    // sticky bit), then apply 2^n while assembling the f64.
    let bits = 128 - p.leading_zeros() as i32; // significant bits of p (> 0)
    let mut shift = bits - 53;
    let mut q: u64;
    if shift > 0 {
        q = (p >> shift) as u64;
        let rem = p - ((q as i128) << shift);
        let half = 1i128 << (shift - 1);
        if rem > half || (rem == half && (q & 1) == 1) {
            q += 1;
            if q == (1u64 << 53) {
                q >>= 1;
                shift += 1;
            }
        }
    } else {
        q = (p << (-shift)) as u64;
    }
    // exp(x) = q * 2^(shift - 63 + n); q is in [2^52, 2^53).
    let e2 = shift - 63 + ni;
    let field = e2 + 52 + 1023;
    debug_assert!((1..=2046).contains(&field));
    let frac = q - (1u64 << 52);
    let value = f64::from_bits(((field as u64) << 52) | frac);
    value as f32
}

/// Software `sigmoid` tail: `1 / (f32(exp(-x)) + 1)`, the exact operation the
/// reference performs for the last `n % 8` element of layer 88.
#[inline]
pub fn sigmoid_tail_soft(x: f32) -> f32 {
    let e = exp_f32(-x);
    1.0f32 / (e + 1.0f32)
}

/// The sigmoid tail used by the compiled engine.
///
/// Native `x86_64` keeps the platform double-precision exponential so the
/// shipped native scalar path is unchanged and remains the verification
/// oracle; wasm (and any other target) routes through the deterministic
/// software exponential above.
#[inline]
pub fn sigmoid_tail(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        let e = (-(x as f64)).exp() as f32;
        1.0f32 / (e + 1.0f32)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        sigmoid_tail_soft(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
        }
    }

    #[test]
    fn special_values() {
        assert_eq!(exp_f32(0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(exp_f32(-0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(exp_f32(1.0).to_bits(), (1.0f64.exp() as f32).to_bits());
        assert_eq!(exp_f32(1000.0), f32::INFINITY);
        assert_eq!(exp_f32(-1000.0), 0.0);
        assert!(exp_f32(f32::NAN).is_nan());
    }

    #[test]
    fn matches_platform_exp_on_a_sweep() {
        let mut rng = Rng(0x0d15_ea5e_1234_5678);
        let mut mism = 0u32;
        let mut n = 0u32;
        for _ in 0..2_000_000 {
            // Uniform in [-88.5, 88.5] plus the extremes of the f32 sigmoid
            // tail range.
            let u = rng.next_u32();
            let x = (u as f32 / u32::MAX as f32) * 177.0 - 88.5;
            let want = (x as f64).exp() as f32;
            let got = exp_f32(x);
            n += 1;
            if got.to_bits() != want.to_bits() {
                mism += 1;
            }
        }
        assert_eq!(mism, 0, "{mism} of {n} exp f32 results differ");
    }
}
