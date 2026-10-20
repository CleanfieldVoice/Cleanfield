//! Deterministic FFT convolution used only to synthesise the room responses.
//!
//! Direct convolution of a 10 s signal with a 4 096-tap hall response is
//! hundreds of millions of multiply-accumulates per scene. The report runs a
//! whole matrix, so the generator uses overlap-save instead. The transform is a
//! plain radix-2 `f32` FFT with precomputed tables; it lives in test code and is
//! deliberately independent of the FFT under test.

use std::f32::consts::PI;

/// One fixed-size complex FFT.
pub struct Fft {
    size: usize,
    reverse: Vec<u32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Fft {
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "FFT size must be a power of two");
        let bits = size.trailing_zeros();
        let mut reverse = vec![0u32; size];
        for (index, slot) in reverse.iter_mut().enumerate() {
            *slot = (index as u32).reverse_bits() >> (32 - bits);
        }
        let mut cos = vec![0.0f32; size / 2];
        let mut sin = vec![0.0f32; size / 2];
        for (k, (cosine, sine)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
            let phase = -2.0 * PI * (k as f32) / (size as f32);
            *cosine = phase.cos();
            *sine = phase.sin();
        }
        Self {
            size,
            reverse,
            cos,
            sin,
        }
    }

    pub fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        self.transform(re, im, false);
    }

    pub fn inverse(&self, re: &mut [f32], im: &mut [f32]) {
        self.transform(re, im, true);
        let scale = 1.0 / (self.size as f32);
        for (real, imaginary) in re.iter_mut().zip(im.iter_mut()) {
            *real *= scale;
            *imaginary *= scale;
        }
    }

    fn transform(&self, re: &mut [f32], im: &mut [f32], inverse: bool) {
        let size = self.size;
        for (index, &target) in self.reverse.iter().enumerate() {
            let target = target as usize;
            if target > index {
                re.swap(index, target);
                im.swap(index, target);
            }
        }

        let sign = if inverse { -1.0f32 } else { 1.0f32 };
        let mut block = 2;
        while block <= size {
            let half = block / 2;
            let step = size / block;
            for start in (0..size).step_by(block) {
                for k in 0..half {
                    let table = k * step;
                    let wr = self.cos[table];
                    let wi = sign * self.sin[table];
                    let a = start + k;
                    let b = a + half;
                    let br = re[b];
                    let bi = im[b];
                    let tr = br * wr - bi * wi;
                    let ti = br * wi + bi * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            block *= 2;
        }
    }
}

impl Default for Fft {
    fn default() -> Self {
        Self::new(256)
    }
}

/// Linear convolution, truncated to `out_len` samples, using overlap-save.
pub fn convolve(input: &[f32], kernel: &[f32], out_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_len];
    if input.is_empty() || kernel.is_empty() || out_len == 0 {
        return out;
    }

    // The hop is half the transform size, and a valid output block needs the
    // filter to fit in that half (plus one) for overlap-save.
    let size = (2 * (kernel.len() - 1))
        .next_power_of_two()
        .max(256)
        .max(input.len().min(256) * 2);
    let hop = size / 2;
    let fft = Fft::new(size);

    let mut kernel_re = vec![0.0f32; size];
    let mut kernel_im = vec![0.0f32; size];
    kernel_re[..kernel.len()].copy_from_slice(kernel);
    fft.forward(&mut kernel_re, &mut kernel_im);

    let mut window_re = vec![0.0f32; size];
    let mut window_im = vec![0.0f32; size];
    for start in (0..out_len).step_by(hop) {
        let end = (start + hop).min(out_len);
        // Window covers input indices `start - hop .. start - hop + size`.
        let offset = start as isize - hop as isize;
        for k in 0..size {
            let index = offset + k as isize;
            window_re[k] = if index >= 0 && (index as usize) < input.len() {
                input[index as usize]
            } else {
                0.0
            };
            window_im[k] = 0.0;
        }
        fft.forward(&mut window_re, &mut window_im);
        for k in 0..size {
            let re = window_re[k] * kernel_re[k] - window_im[k] * kernel_im[k];
            let im = window_re[k] * kernel_im[k] + window_im[k] * kernel_re[k];
            window_re[k] = re;
            window_im[k] = im;
        }
        fft.inverse(&mut window_re, &mut window_im);
        let length = end - start;
        out[start..end].copy_from_slice(&window_re[hop..hop + length]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_convolution_matches_direct_convolution() {
        let mut state = 0x1234_5678u32;
        let mut input = vec![0.0f32; 1_000];
        let mut kernel = vec![0.0f32; 90];
        for value in &mut input {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *value = ((state >> 9) as f32) * (1.0 / 4_194_304.0) - 1.0;
        }
        for value in &mut kernel {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *value = ((state >> 9) as f32) * (1.0 / 4_194_304.0) - 1.0;
        }
        let fast = convolve(&input, &kernel, input.len());
        let mut direct = vec![0.0f32; input.len()];
        for (n, value) in direct.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (k, &tap) in kernel.iter().enumerate() {
                if k > n {
                    break;
                }
                sum += tap * input[n - k];
            }
            *value = sum;
        }
        let worst = fast
            .iter()
            .zip(direct.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1.0e-5, "overlap-save error {worst}");
    }
}
