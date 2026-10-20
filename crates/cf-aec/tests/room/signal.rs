//! Deterministic signal material for the room-condition suite.
//!
//! Everything is integer-seeded and `f32`-only, so a scene is byte-for-byte
//! reproducible across machines. The far-end and near-end talkers use the same
//! source-filter generator: a jittered glottal pulse train, two drifting formant
//! resonators, a syllabic envelope and a slow phrase envelope. Noise is
//! independent of the talkers and has no random phase relationship with them.

use super::FS;

/// Small deterministic PRNG (xorshift32). `seed | 1` guarantees a non-zero state.
pub struct Rng {
    state: u32,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self { state: seed | 1 }
    }

    /// Uniform sample in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        ((x >> 9) as f32) * (1.0 / 4_194_304.0) - 1.0
    }

    /// Uniform sample in `[0, 1)`.
    pub fn next_unit(&mut self) -> f32 {
        0.5 * (self.next_f32() + 1.0)
    }
}

/// Two-pole resonator used to give a pulse train a formant-like spectrum.
struct Resonator {
    pole_1: f32,
    pole_2: f32,
    previous_1: f32,
    previous_2: f32,
}

impl Resonator {
    fn new(frequency: f32, radius: f32) -> Self {
        let angle = 2.0 * std::f32::consts::PI * frequency / FS;
        Self {
            pole_1: 2.0 * radius * angle.cos(),
            pole_2: -(radius * radius),
            previous_1: 0.0,
            previous_2: 0.0,
        }
    }

    fn step(&mut self, input: f32) -> f32 {
        let output = input + self.pole_1 * self.previous_1 + self.pole_2 * self.previous_2;
        self.previous_2 = self.previous_1;
        self.previous_1 = output;
        output
    }
}

/// Speech-like signal: a jittered glottal pulse train plus aspiration, shaped by
/// two drifting formant resonators and a syllabic envelope.
///
/// Different seeds change the pitch, the formant frequencies, the syllable rate
/// and the pulse positions, so two instances are effectively uncorrelated. This
/// is the same generator the functional validation suite uses; keeping it
/// identical means the room suite's baseline is directly comparable.
pub fn speech_like(len: usize, seed: u32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut voice = Rng::new(seed ^ 0x7F4A_7C15);
    let unit = |value: f32| 0.5 * (value + 1.0);
    let fundamental_base = 95.0 + 55.0 * unit(voice.next_f32());
    let fundamental_rate = 0.45 + 0.8 * unit(voice.next_f32());
    let fundamental_depth = 10.0 + 25.0 * unit(voice.next_f32());
    let syllable_rate = 2.2 + 2.4 * unit(voice.next_f32());
    let phrase_rate = 0.3 + 0.6 * unit(voice.next_f32());
    let formant_1_frequency = 450.0 + 450.0 * unit(voice.next_f32());
    let formant_2_frequency = 1_000.0 + 1_000.0 * unit(voice.next_f32());
    let aspiration = 0.02 + 0.08 * unit(voice.next_f32());

    let mut formant_1 = Resonator::new(formant_1_frequency, 0.975);
    let mut formant_2 = Resonator::new(formant_2_frequency, 0.965);
    let mut high_pass_input = 0.0f32;
    let mut high_pass_output = 0.0f32;
    let mut phase = 0.0f32;
    let mut out = vec![0.0f32; len];

    for (n, sample) in out.iter_mut().enumerate() {
        let t = (n as f32) / FS;
        let fundamental = fundamental_base
            + fundamental_depth * (2.0 * std::f32::consts::PI * fundamental_rate * t).sin();
        let period = FS / fundamental;
        phase += 1.0;
        let mut source = 0.0f32;
        if phase >= period {
            phase -= period;
            source = 1.0;
        }
        source += aspiration * rng.next_f32();
        let voiced = formant_2.step(formant_1.step(source));
        // One-pole DC block: the pulse train has a large DC term.
        high_pass_output = voiced - high_pass_input + 0.995 * high_pass_output;
        high_pass_input = voiced;
        let syllable = (0.5 + 0.5 * (2.0 * std::f32::consts::PI * syllable_rate * t).sin()).powi(2);
        let phrase = 0.55 + 0.45 * (2.0 * std::f32::consts::PI * phrase_rate * t).sin();
        *sample = high_pass_output * syllable * phrase;
    }

    normalize_rms(&mut out, 0.08);
    out
}

/// Background-noise family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseKind {
    White,
    Pink,
    Babble,
}

impl NoiseKind {
    pub const ALL: [Self; 3] = [Self::White, Self::Pink, Self::Babble];

    pub fn label(self) -> &'static str {
        match self {
            Self::White => "white",
            Self::Pink => "pink",
            Self::Babble => "babble",
        }
    }
}

/// Generates `len` samples of the requested noise, normalised to 0.08 RMS.
pub fn noise_like(len: usize, kind: NoiseKind, seed: u32) -> Vec<f32> {
    match kind {
        NoiseKind::White => {
            let mut rng = Rng::new(seed ^ 0x1111_2222);
            let mut out = vec![0.0f32; len];
            for value in &mut out {
                *value = rng.next_f32();
            }
            normalize_rms(&mut out, 0.08);
            out
        }
        NoiseKind::Pink => {
            // Paul Kellet's three-pole pink approximation, in f32. It is only a
            // spectral shape for the test, not a calibrated -3 dB/octave curve.
            let mut rng = Rng::new(seed ^ 0x3333_4444);
            let mut b0 = 0.0f32;
            let mut b1 = 0.0f32;
            let mut b2 = 0.0f32;
            let mut out = vec![0.0f32; len];
            for value in &mut out {
                let white = rng.next_f32();
                b0 = 0.997_65 * b0 + 0.099_046 * white;
                b1 = 0.963_00 * b1 + 0.296_516_4 * white;
                b2 = 0.570_00 * b2 + 1.052_691_3 * white;
                *value = b0 + b1 + b2 + 0.184_8 * white;
            }
            normalize_rms(&mut out, 0.08);
            out
        }
        NoiseKind::Babble => {
            // Four independent talkers, each at a similar level, summed with a
            // small deterministic level spread and circular time offset. This
            // is the conventional "babble" construction: many voices at once,
            // not one amplitude-modulated noise source.
            let mut rng = Rng::new(seed ^ 0x5555_6666);
            let mut out = vec![0.0f32; len];
            for voice in 0..4usize {
                let source_seed = seed ^ (0x9E37_79B9u32.wrapping_mul(voice as u32 + 1));
                let source = speech_like(len, source_seed);
                let gain = 0.55 + 0.45 * rng.next_unit();
                let offset = ((rng.next_unit() * 0.35 * FS) as usize).min(len / 2);
                for n in 0..len {
                    let index = (n + offset) % len;
                    out[n] += gain * source[index];
                }
            }
            normalize_rms(&mut out, 0.08);
            out
        }
    }
}

/// Linear-interpolated fractional delay: `out[n] = input[n - delay(n)]`.
///
/// A non-integer sample position is common when the delay drifts, and linear
/// interpolation is the least surprising deterministic choice available without
/// an external resampler. For an integer delay it reproduces the input exactly.
pub fn fractional_delay<F>(input: &[f32], delay_at: F) -> Vec<f32>
where
    F: Fn(usize) -> f32,
{
    let mut out = vec![0.0f32; input.len()];
    for (n, value) in out.iter_mut().enumerate() {
        let position = n as f32 - delay_at(n);
        if position < 0.0 {
            continue;
        }
        let index = position.floor() as usize;
        if index + 1 >= input.len() {
            *value = input[index];
            continue;
        }
        let fraction = position - index as f32;
        *value = input[index] * (1.0 - fraction) + input[index + 1] * fraction;
    }
    out
}

pub fn normalize_rms(signal: &mut [f32], target: f32) {
    let current = rms(signal);
    if current > 0.0 {
        let scale = target / current;
        for value in signal {
            *value *= scale;
        }
    }
}

pub fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|value| value * value).sum::<f32>() / signal.len() as f32).sqrt()
}

pub fn peak(signal: &[f32]) -> f32 {
    signal
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_signals_are_reproducible() {
        let a = speech_like(1_000, 7);
        let b = speech_like(1_000, 7);
        assert_eq!(a, b);
        let c = speech_like(1_000, 8);
        assert_ne!(a, c);
    }

    #[test]
    fn integer_fractional_delay_matches_an_integer_copy() {
        let input: Vec<f32> = (0..256).map(|n| n as f32 * 0.01).collect();
        let delayed = fractional_delay(&input, |_| 64.0);
        for n in 0..64 {
            assert_eq!(delayed[n], 0.0);
        }
        for n in 64..input.len() {
            assert_eq!(delayed[n], input[n - 64]);
        }
    }

    #[test]
    fn noise_families_have_the_requested_level() {
        for kind in NoiseKind::ALL {
            let value = noise_like(16_000, kind, 0x1234);
            assert!((rms(&value) - 0.08).abs() < 1.0e-3, "{kind:?}");
            assert!(value.iter().all(|sample| sample.is_finite()));
        }
    }
}
