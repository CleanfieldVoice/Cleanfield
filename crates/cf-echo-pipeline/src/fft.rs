//! Fixed-size real-input FFT used by the echo front end.
//!
//! The echo path transforms 768-sample Hann-windowed analysis frames with a
//! 1024-point FFT and keeps the 513 non-negative bins. Synthesis is the
//! corresponding 1024-point inverse. The transform is plain radix-2 `f32` with
//! precomputed bit-reversal and twiddle tables; there is no allocation at run
//! time and no dependency outside `std`.

use std::f32::consts::PI;

/// FFT length.
pub const FFT_SIZE: usize = 1024;

/// Non-negative frequency bins (`0 ..= FFT_SIZE / 2`).
pub const BINS: usize = FFT_SIZE / 2 + 1;

/// Analysis/synthesis window length.
pub const WINDOW_LEN: usize = 768;

/// Precomputed FFT tables.
pub struct Fft {
    rev: [u16; FFT_SIZE],
    cos: [f32; FFT_SIZE / 2],
    sin: [f32; FFT_SIZE / 2],
}

impl Fft {
    /// Builds the bit-reversal and twiddle tables. No heap allocation.
    pub fn new() -> Self {
        let mut rev = [0u16; FFT_SIZE];
        for (index, slot) in rev.iter_mut().enumerate() {
            *slot = (index as u16).reverse_bits() >> 6;
        }
        let mut cos = [0.0f32; FFT_SIZE / 2];
        let mut sin = [0.0f32; FFT_SIZE / 2];
        for (k, (c, s)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
            let angle = 2.0 * PI * k as f32 / FFT_SIZE as f32;
            *c = angle.cos();
            *s = angle.sin();
        }
        Self { rev, cos, sin }
    }

    /// In-place complex forward transform.
    pub fn forward(&self, re: &mut [f32; FFT_SIZE], im: &mut [f32; FFT_SIZE]) {
        self.transform(re, im, false);
    }

    /// In-place complex inverse transform, including the `1 / FFT_SIZE` scale.
    pub fn inverse(&self, re: &mut [f32; FFT_SIZE], im: &mut [f32; FFT_SIZE]) {
        self.transform(re, im, true);
    }

    /// Real 768-sample frame -> 513 one-sided bins (zero-padded to 1024).
    pub fn real_spectrum(
        &self,
        frame: &[f32; WINDOW_LEN],
        out_re: &mut [f32; BINS],
        out_im: &mut [f32; BINS],
    ) {
        let mut re = [0.0f32; FFT_SIZE];
        let mut im = [0.0f32; FFT_SIZE];
        re[..WINDOW_LEN].copy_from_slice(frame);
        self.forward(&mut re, &mut im);
        out_re.copy_from_slice(&re[..BINS]);
        out_im.copy_from_slice(&im[..BINS]);
        // A real input has exactly zero imaginary DC and Nyquist bins.
        out_im[0] = 0.0;
        out_im[BINS - 1] = 0.0;
    }

    /// 513 one-sided bins -> real 768-sample frame, with conjugate symmetry.
    pub fn real_inverse(
        &self,
        spectrum_re: &[f32; BINS],
        spectrum_im: &[f32; BINS],
        out: &mut [f32; WINDOW_LEN],
    ) {
        let mut re = [0.0f32; FFT_SIZE];
        let mut im = [0.0f32; FFT_SIZE];
        re[..BINS].copy_from_slice(&spectrum_re[..BINS]);
        im[..BINS].copy_from_slice(&spectrum_im[..BINS]);
        for k in 1..BINS - 1 {
            re[FFT_SIZE - k] = spectrum_re[k];
            im[FFT_SIZE - k] = -spectrum_im[k];
        }
        im[0] = 0.0;
        im[BINS - 1] = 0.0;
        self.inverse(&mut re, &mut im);
        out.copy_from_slice(&re[..WINDOW_LEN]);
    }

    fn transform(&self, re: &mut [f32; FFT_SIZE], im: &mut [f32; FFT_SIZE], inverse: bool) {
        for i in 0..FFT_SIZE {
            let j = self.rev[i] as usize;
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        let mut len = 2;
        while len <= FFT_SIZE {
            let half = len / 2;
            let step = FFT_SIZE / len;
            let mut start = 0;
            while start < FFT_SIZE {
                for k in 0..half {
                    let (wr, wi) = if inverse {
                        (self.cos[k * step], self.sin[k * step])
                    } else {
                        (self.cos[k * step], -self.sin[k * step])
                    };
                    let a_re = re[start + k];
                    let a_im = im[start + k];
                    let b_re = re[start + k + half];
                    let b_im = im[start + k + half];
                    let t_re = b_re * wr - b_im * wi;
                    let t_im = b_re * wi + b_im * wr;
                    re[start + k] = a_re + t_re;
                    im[start + k] = a_im + t_im;
                    re[start + k + half] = a_re - t_re;
                    im[start + k + half] = a_im - t_im;
                }
                start += len;
            }
            len <<= 1;
        }

        if inverse {
            let scale = 1.0 / FFT_SIZE as f32;
            for i in 0..FFT_SIZE {
                re[i] *= scale;
                im[i] *= scale;
            }
        }
    }
}

impl Default for Fft {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_round_trip_reconstructs_the_window() {
        let fft = Fft::new();
        let mut frame = [0.0f32; WINDOW_LEN];
        for (n, value) in frame.iter_mut().enumerate() {
            *value = (0.01 * n as f32).sin() + 0.3 * (0.07 * n as f32).cos();
        }
        let mut re = [0.0f32; BINS];
        let mut im = [0.0f32; BINS];
        fft.real_spectrum(&frame, &mut re, &mut im);
        let mut back = [0.0f32; WINDOW_LEN];
        fft.real_inverse(&re, &im, &mut back);
        for (n, (&a, &b)) in frame.iter().zip(back.iter()).enumerate() {
            assert!((a - b).abs() < 1.0e-5, "sample {n}: {a} vs {b}");
        }
    }

    #[test]
    fn forward_impulse_has_a_flat_spectrum() {
        let fft = Fft::new();
        let mut frame = [0.0f32; WINDOW_LEN];
        frame[0] = 1.0;
        let mut re = [0.0f32; BINS];
        let mut im = [0.0f32; BINS];
        fft.real_spectrum(&frame, &mut re, &mut im);
        for k in 0..BINS {
            assert!((re[k] - 1.0).abs() < 1.0e-6);
            assert!(im[k].abs() < 1.0e-6);
        }
    }
}
