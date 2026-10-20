//! Radix-2 complex FFT used by the partitioned filter and the delay estimator.
//!
//! The transform size is a const generic parameter: the filter uses
//! [`crate::FFT_SIZE`] (512) and the delay estimator uses a larger transform on
//! decimated input. A real input only needs the one-sided bins; the caller keeps
//! those and reconstructs the conjugate half before an inverse transform.
//! Twiddle factors are computed once in [`FftN::new`] and stored, so a transform
//! performs no allocation and no trigonometric calls.

use std::f32::consts::PI;

use crate::FFT_SIZE;

/// Pre-computed bit-reversal permutation and twiddle factors for one size.
pub(crate) struct FftN<const N: usize> {
    reverse: [u16; N],
    /// Twiddle tables. Only the first `N / 2` entries are used; the arrays are
    /// full length because a const generic parameter cannot appear in an array
    /// length expression in a struct field on the supported compiler.
    cos: [f32; N],
    sin: [f32; N],
}

/// Transform used by the partitioned adaptive filter.
pub(crate) type Fft = FftN<FFT_SIZE>;

impl<const N: usize> FftN<N> {
    /// Builds the tables. This is the only part of the transform that runs
    /// transcendental functions; it is called from `Aec::new`, not per frame.
    pub(crate) fn new() -> Self {
        assert!(N.is_power_of_two(), "FFT size must be a power of two");
        assert!(N >= 2, "FFT size must be at least two");
        let mut reverse = [0u16; N];
        let bits = N.trailing_zeros();
        for (i, slot) in reverse.iter_mut().enumerate() {
            *slot = (i as u16).reverse_bits() >> (16 - bits);
        }

        let mut cos = [0.0f32; N];
        let mut sin = [0.0f32; N];
        for k in 0..N / 2 {
            let phase = -2.0 * PI * (k as f32) / (N as f32);
            cos[k] = phase.cos();
            sin[k] = phase.sin();
        }

        Self { reverse, cos, sin }
    }

    /// In-place forward transform (`exp(-j 2 pi k n / N)` sign convention).
    pub(crate) fn forward(&self, re: &mut [f32; N], im: &mut [f32; N]) {
        self.transform(re, im, false);
    }

    /// In-place inverse transform, scaled by `1 / N`.
    pub(crate) fn inverse(&self, re: &mut [f32; N], im: &mut [f32; N]) {
        self.transform(re, im, true);
        let scale = 1.0 / (N as f32);
        for k in 0..N {
            re[k] *= scale;
            im[k] *= scale;
        }
    }

    fn transform(&self, re: &mut [f32; N], im: &mut [f32; N], inverse: bool) {
        for i in 0..N {
            let j = self.reverse[i] as usize;
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        // `sin` holds the forward exponent (negative phase); the inverse uses
        // the conjugate, so the sign flips.
        let sign = if inverse { -1.0f32 } else { 1.0f32 };
        let mut size = 2;
        while size <= N {
            let half = size / 2;
            let step = N / size;
            for start in (0..N).step_by(size) {
                for k in 0..half {
                    let t = k * step;
                    let wr = self.cos[t];
                    let wi = sign * self.sin[t];
                    let a = start + k;
                    let b = a + half;
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            size *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random<const N: usize>() -> [f32; N] {
        let mut out = [0.0f32; N];
        let mut state = 0x1234_5678u32;
        for v in out.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((state >> 9) as f32) * (1.0 / 4_194_304.0) - 1.0;
        }
        out
    }

    #[test]
    fn forward_matches_naive_dft() {
        const N: usize = FFT_SIZE;
        let input = pseudo_random::<N>();
        let fft = FftN::<N>::new();
        let mut re = input;
        let mut im = [0.0f32; N];
        fft.forward(&mut re, &mut im);

        let mut worst = 0.0f32;
        for k in 0..N {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;
            for (n, &x) in input.iter().enumerate() {
                let phase = -2.0 * PI * ((k * n) as f32) / (N as f32);
                sum_re += x * phase.cos();
                sum_im += x * phase.sin();
            }
            worst = worst.max((re[k] - sum_re).abs());
            worst = worst.max((im[k] - sum_im).abs());
        }
        // Both paths accumulate in f32; 512 terms of unit amplitude stay far
        // below this bound.
        assert!(worst < 2.0e-2, "naive-DFT difference {worst}");
    }

    #[test]
    fn inverse_round_trip_is_exact_enough() {
        const N: usize = FFT_SIZE;
        let input = pseudo_random::<N>();
        let fft = FftN::<N>::new();
        let mut re = input;
        let mut im = [0.0f32; N];
        fft.forward(&mut re, &mut im);
        fft.inverse(&mut re, &mut im);

        let mut worst = 0.0f32;
        for n in 0..N {
            worst = worst.max((re[n] - input[n]).abs());
            worst = worst.max(im[n].abs());
        }
        assert!(worst < 1.0e-5, "round-trip difference {worst}");
    }
}
