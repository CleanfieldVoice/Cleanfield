//! Demo-only wrapper around `cf-voice` with the documented streaming front end.
//!
//! This mirrors the reference denoiser front end: 16 kHz, 1024-point FFT, a
//! 768-sample periodic Hann window, a 256-sample hop, `ln(|X|^2 * scale +
//! 1e-20)` features, a two-hop mask look-ahead, the two-plane magnitude mask and
//! `8/9` overlap-add. It exists only so the README `both-after` clip can be
//! regenerated with this workspace's own voice model.

use cf_echo_pipeline::fft::{Fft, BINS, WINDOW_LEN};
use cf_echo_pipeline::{FRAME_LEN, SYNTHESIS_SCALE};
use cf_voice::VoiceModel;

/// Power multiplier folded from the model's int16 units.
const FEATURE_POWER_SCALE: f32 = 9.313_200_211_025_219e-10f32 * 1_073_741_824.0f32;
/// Feature floor.
const FEATURE_EPSILON: f32 = 1.0e-20;
/// Mask floor at the reference default of 80 dB maximum reduction.
const MASK_FLOOR: f32 = 1.0e-4;

/// Output latency of the voice stage, in samples.
pub const VOICE_LATENCY_SAMPLES: usize = WINDOW_LEN + 2 * FRAME_LEN;

pub struct VoiceDenoiser {
    model: VoiceModel,
    fft: Fft,
    window: [f32; WINDOW_LEN],
    analysis: [f32; WINDOW_LEN],
    overlap: [f32; WINDOW_LEN],
    spectra_re: [[f32; BINS]; 2],
    spectra_im: [[f32; BINS]; 2],
    head: usize,
    pending: [f32; FRAME_LEN],
}

impl VoiceDenoiser {
    pub fn new(container_bytes: &[u8]) -> Result<Self, String> {
        let model = VoiceModel::new(container_bytes).map_err(|e| format!("voice model: {e}"))?;
        let mut window = [0.0f32; WINDOW_LEN];
        for (n, value) in window.iter_mut().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * n as f64 / WINDOW_LEN as f64;
            *value = (0.5 - 0.5 * angle.cos()) as f32;
        }
        Ok(Self {
            model,
            fft: Fft::new(),
            window,
            analysis: [0.0; WINDOW_LEN],
            overlap: [0.0; WINDOW_LEN],
            spectra_re: [[0.0; BINS]; 2],
            spectra_im: [[0.0; BINS]; 2],
            head: 0,
            pending: [0.0; FRAME_LEN],
        })
    }

    /// Processes a whole number of 256-sample frames; output lags by
    /// [`VOICE_LATENCY_SAMPLES`].
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        for frame in input.chunks_exact(FRAME_LEN) {
            let block = self.process_frame(frame);
            out.extend_from_slice(&block);
        }
        out
    }

    fn process_frame(&mut self, frame: &[f32]) -> [f32; FRAME_LEN] {
        self.analysis.copy_within(FRAME_LEN.., 0);
        self.analysis[WINDOW_LEN - FRAME_LEN..].copy_from_slice(frame);

        let mut windowed = [0.0f32; WINDOW_LEN];
        for n in 0..WINDOW_LEN {
            windowed[n] = self.analysis[n] * self.window[n];
        }
        let mut re = [0.0f32; BINS];
        let mut im = [0.0f32; BINS];
        self.fft.real_spectrum(&windowed, &mut re, &mut im);

        let mut features = [0.0f32; BINS];
        for k in 0..BINS {
            let power = re[k] * re[k] + im[k] * im[k];
            features[k] = (power * FEATURE_POWER_SCALE + FEATURE_EPSILON).ln();
        }
        self.model
            .process(&features)
            .expect("the voice model frame length is fixed");
        let output = self.model.output();

        let slot = self.head;
        let mut processed_re = [0.0f32; BINS];
        let mut processed_im = [0.0f32; BINS];
        for k in 0..BINS {
            let real = output[k];
            let imag = output[BINS + k];
            let magnitude = (real * real + imag * imag + FEATURE_EPSILON).sqrt();
            let mask = magnitude.min(1.0).max(MASK_FLOOR);
            processed_re[k] = self.spectra_re[slot][k] * mask;
            processed_im[k] = self.spectra_im[slot][k] * mask;
        }
        self.spectra_re[slot] = re;
        self.spectra_im[slot] = im;
        self.head = (self.head + 1) % 2;

        let mut reconstruction = [0.0f32; WINDOW_LEN];
        self.fft
            .real_inverse(&processed_re, &processed_im, &mut reconstruction);
        for n in 0..WINDOW_LEN {
            self.overlap[n] += reconstruction[n] * self.window[n] * SYNTHESIS_SCALE;
        }

        let mut block = [0.0f32; FRAME_LEN];
        block.copy_from_slice(&self.overlap[..FRAME_LEN]);
        self.overlap.copy_within(FRAME_LEN.., 0);
        self.overlap[WINDOW_LEN - FRAME_LEN..].fill(0.0);

        let out = self.pending;
        self.pending = block;
        out
    }
}
