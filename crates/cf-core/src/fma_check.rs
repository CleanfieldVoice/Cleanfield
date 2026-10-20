//! Cross-check suite: native `mul_add` vs the integer software FMA.
//!
//! The suite is intentionally adversarial:
//!
//! * random 32-bit patterns over the whole encoding space,
//! * random values in a realistic magnitude range,
//! * exact midpoints of the `f32` grid, including subnormal perturbations,
//! * subnormal / tiny inputs and results,
//! * overflow and underflow boundaries, and extreme exponent differences,
//! * a fixed matrix of special values (zeros, infinities, NaNs, extremes).
//!
//! A pair of results matches when the bit patterns are equal, or when both are
//! NaN (NaN payload propagation is not part of the contract; the model never
//! produces NaN in the first place).

use crate::fma::{soft_fma, FmaOps, NativeFma};

/// One category of the suite.
#[derive(Clone, Debug)]
pub struct CategoryReport {
    pub name: &'static str,
    pub cases: u64,
    pub mismatches: u64,
    pub both_nan: u64,
}

/// One observed mismatch (native bits, software bits).
#[derive(Clone, Copy, Debug)]
pub struct Mismatch {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub native: f32,
    pub soft: f32,
}

/// Aggregate result.
#[derive(Clone, Debug)]
pub struct FmaReport {
    pub categories: Vec<CategoryReport>,
    pub examples: Vec<Mismatch>,
    pub total: u64,
    pub mismatches: u64,
    pub both_nan: u64,
}

impl FmaReport {
    pub fn ok(&self) -> bool {
        self.mismatches == 0
    }
}

/// Run the suite over at least `cases` inputs (plus the fixed special matrix).
pub fn crosscheck(cases: u64, seed: u64) -> FmaReport {
    let budget = cases.max(1);
    let mut counts = [
        ("random-full-bits", budget * 55 / 100),
        ("random-realistic", budget * 20 / 100),
        ("f32-midpoints", budget * 10 / 100),
        ("subnormals", budget * 8 / 100),
        (
            "extremes",
            budget - (budget * 55 / 100 + budget * 20 / 100 + budget * 10 / 100 + budget * 8 / 100),
        ),
    ];
    // Never run a zero-case category.
    for c in counts.iter_mut() {
        if c.1 == 0 {
            c.1 = 1;
        }
    }

    let mut report = FmaReport {
        categories: Vec::with_capacity(counts.len() + 1),
        examples: Vec::new(),
        total: 0,
        mismatches: 0,
        both_nan: 0,
    };
    let mut rng = Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);

    for (name, n) in counts {
        let mut cat = CategoryReport {
            name,
            cases: 0,
            mismatches: 0,
            both_nan: 0,
        };
        let mut i = 0u64;
        while i < n {
            let (a, b, c) = match name {
                "random-full-bits" => gen_random(&mut rng),
                "random-realistic" => gen_realistic(&mut rng),
                "f32-midpoints" => gen_midpoint(&mut rng),
                "subnormals" => gen_subnormal(&mut rng),
                _ => gen_extreme(&mut rng),
            };
            eval(a, b, c, &mut cat, &mut report.examples);
            i += 1;
        }
        report.total += cat.cases;
        report.mismatches += cat.mismatches;
        report.both_nan += cat.both_nan;
        report.categories.push(cat);
    }

    // Fixed special-value matrix.
    let specials: [f32; 13] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        16_777_216.0,                      // 2^24
        f32::from_bits((103 + 127) << 23), // 2^103
    ];
    let mut cat = CategoryReport {
        name: "special-values",
        cases: 0,
        mismatches: 0,
        both_nan: 0,
    };
    for &a in &specials {
        for &b in &specials {
            for &c in &specials {
                eval(a, b, c, &mut cat, &mut report.examples);
            }
        }
    }
    report.total += cat.cases;
    report.mismatches += cat.mismatches;
    report.both_nan += cat.both_nan;
    report.categories.push(cat);

    report
}

#[inline]
fn eval(a: f32, b: f32, c: f32, cat: &mut CategoryReport, examples: &mut Vec<Mismatch>) {
    cat.cases += 1;
    let native = NativeFma.fma(a, b, c);
    let soft = soft_fma(a, b, c);
    if native.is_nan() && soft.is_nan() {
        cat.both_nan += 1;
        return;
    }
    if native.to_bits() != soft.to_bits() {
        cat.mismatches += 1;
        if examples.len() < 8 {
            examples.push(Mismatch {
                a,
                b,
                c,
                native,
                soft,
            });
        }
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(if seed == 0 {
            0x51ED_270B_9C3F_1A55
        } else {
            seed
        })
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    #[inline]
    fn u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    #[inline]
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }
    #[inline]
    fn flag(&mut self) -> bool {
        self.next_u64() & 1 != 0
    }
    #[inline]
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u32) as i32
    }
}

#[inline]
fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

#[inline]
fn with_sign(flag: bool, x: f32) -> f32 {
    if flag {
        -x
    } else {
        x
    }
}

fn gen_random(r: &mut Rng) -> (f32, f32, f32) {
    (
        f32::from_bits(r.u32()),
        f32::from_bits(r.u32()),
        f32::from_bits(r.u32()),
    )
}

fn gen_realistic(r: &mut Rng) -> (f32, f32, f32) {
    let one = |r: &mut Rng| {
        let sign = (r.flag() as u32) << 31;
        if r.below(20) == 0 {
            // subnormal
            f32::from_bits(sign | (r.u32() & 0x7F_FFFF))
        } else {
            let e = r.range(90, 160) as u32;
            f32::from_bits(sign | (e << 23) | (r.u32() & 0x7F_FFFF))
        }
    };
    let a = one(r);
    let b = one(r);
    // A meaningful fraction of zero addends exercises the product-only path.
    let c = if r.below(10) == 0 { 0.0 } else { one(r) };
    (a, b, c)
}

fn gen_midpoint(r: &mut Rng) -> (f32, f32, f32) {
    // 4097 * 4097 = 2^24 + 1: exactly halfway between two f32 values in the
    // [2^24, 2^25) binade.  Scaling both operands by powers of two moves the
    // midpoint to any normal binade.
    let p = r.range(-60, 60);
    let q = r.range(-60, 60);
    let a = f32::from_bits((((127 + 12 + p) as u32) << 23) | (1 << 11));
    let b = f32::from_bits((((127 + 12 + q) as u32) << 23) | (1 << 11));

    let c = match r.below(4) {
        0 => 0.0,
        1 => {
            // Subnormal perturbation: breaks the tie in either direction.
            let frac = r.u32() & 0x7F_FFFF | 1;
            with_sign(r.flag(), f32::from_bits(frac))
        }
        2 => {
            // Exactly half an ulp at the product's scale.
            with_sign(r.flag(), pow2(p + q))
        }
        _ => {
            // Small value around the ulp.
            let e = (p + q + 127 - 24).clamp(2, 254) as u32;
            with_sign(r.flag(), f32::from_bits((e << 23) | (r.u32() & 0x7F_FFFF)))
        }
    };
    (with_sign(r.flag(), a), with_sign(r.flag(), b), c)
}

fn gen_subnormal(r: &mut Rng) -> (f32, f32, f32) {
    let one = |r: &mut Rng| {
        let sign = (r.flag() as u32) << 31;
        match r.below(10) {
            0 => f32::from_bits(sign), // zero
            1 | 2 => {
                let e = r.range(1, 3) as u32;
                f32::from_bits(sign | (e << 23) | (r.u32() & 0x7F_FFFF))
            }
            _ => f32::from_bits(sign | (r.u32() & 0x7F_FFFF)),
        }
    };
    (one(r), one(r), one(r))
}

fn gen_extreme(r: &mut Rng) -> (f32, f32, f32) {
    match r.below(12) {
        // Overflow threshold: 2^128 - 2^103 is the exact midpoint between
        // f32::MAX and infinity; a tie rounds to even (infinity).
        0 => (pow2(64), pow2(64), -pow2(103)),
        1 => {
            // Just below the overflow midpoint: (2^103 + 2^80) is exact.
            let c = -(pow2(103) + pow2(80));
            (pow2(64), pow2(64), c)
        }
        2 => (pow2(64), pow2(64), pow2(103)),
        // The exact product 2 * f32::MAX minus f32::MAX is f32::MAX; a
        // premature intermediate overflow would wrongly produce infinity.
        3 => (f32::MAX, 2.0, -f32::MAX),
        4 => (f32::MAX, 1.0, with_sign(r.flag(), f32::from_bits(1))),
        // Exact subnormal boundary.
        5 => (pow2(-126), pow2(-23), 0.0),
        6 => (f32::from_bits(1), 0.5, f32::from_bits(1)),
        7 => (f32::from_bits(1), pow2(-23), 0.0),
        // Huge exponent difference, exact product.
        8 => (
            pow2(200),
            pow2(-200),
            with_sign(r.flag(), pow2(r.range(-60, 60))),
        ),
        // Exact cancellation of an exactly-representable product.
        9 => {
            let a = f32::from_bits(
                ((r.flag() as u32) << 31)
                    | ((r.range(1, 254) as u32) << 23)
                    | (r.u32() & 0x7F_FFFF),
            );
            (a, 1.0, -a)
        }
        // One-ulp difference after cancellation.
        10 => {
            let a = f32::from_bits(((r.range(120, 140) as u32) << 23) | (r.u32() & 0x7F_FFFF));
            (a, 1.0, -f32::from_bits(a.to_bits() - 1))
        }
        _ => {
            // Extreme exponent difference with a sticky tail.
            let a = f32::from_bits(
                ((r.flag() as u32) << 31) | ((r.range(1, 40) as u32) << 23) | (r.u32() & 0x7F_FFFF),
            );
            let b = f32::from_bits(((r.range(215, 254) as u32) << 23) | (r.u32() & 0x7F_FFFF));
            let c = f32::from_bits(
                ((r.flag() as u32) << 31)
                    | ((r.range(1, 254) as u32) << 23)
                    | (r.u32() & 0x7F_FFFF),
            );
            (a, b, c)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::crosscheck;

    #[test]
    fn soft_matches_native_over_ten_million_cases() {
        let report = crosscheck(10_000_000, 0xCF_00D1_5EED_2026);
        for c in &report.categories {
            eprintln!(
                "fma-check {:20} cases={:>9} mismatches={} both_nan={}",
                c.name, c.cases, c.mismatches, c.both_nan
            );
        }
        eprintln!(
            "fma-check TOTAL cases={} mismatches={} both_nan={}",
            report.total, report.mismatches, report.both_nan
        );
        assert_eq!(report.mismatches, 0, "examples: {:?}", report.examples);
        assert!(report.total >= 10_000_000);
    }
}
