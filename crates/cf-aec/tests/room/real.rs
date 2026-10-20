//! Bundled **measured** room impulse responses.
//!
//! Eighteen real responses from OpenSLR SLR28 (Apache-2.0, see
//! `real/PROVENANCE.md`) are compiled into the test binaries. They replace
//! nothing: the synthetic families in [`super::rir`] stay, because they are the
//! only way to cover a designed range of shapes. The real set is the check on
//! that design.
//!
//! The bundled files are mono 16 kHz 16-bit PCM WAV, trimmed to the measured
//! onset and kept whole up to 1.6 s. The loader is deliberately less strict
//! than the bundled set: the source archive declares 16 kHz but a replacement
//! or a re-extraction may be a 48 kHz or multi-channel measurement, so it
//! accepts 16 kHz or 48 kHz, any channel count, takes **channel 0**, resamples
//! 48 kHz down to 16 kHz with a fixed windowed-sinc low-pass, and rejects
//! anything else with a clear message. Channel 0 is the convention of
//! `real/extract_slr28.py`; for the binaural AIR and array RWCP entries it is
//! the left ear or the first array microphone.
//!
//! [`RealRir::impulse`] then applies the same unit-L2-energy policy as the
//! synthetic responses, so an ERLE number is not flattered or penalised by the
//! absolute gain of the recording. The acoustic metrics ([`rt20_s`],
//! [`c50_db`]) are measured from the stored response by this module; they are
//! not copied from the dataset's documentation.

use std::sync::OnceLock;

use super::rir::profile;
use super::FS;

/// One bundled measured response.
pub struct RealRir {
    /// Stable identifier used by [`super::SceneSpec::real_rir`] and the reports.
    pub id: &'static str,
    /// Measurement campaign the response comes from.
    pub dataset: &'static str,
    /// Room / microphone description from the dataset naming.
    pub room: &'static str,
    /// Name of the file inside the SLR28 archive.
    pub source_file: &'static str,
    /// The bundled WAV bytes.
    pub wav: &'static [u8],
}

macro_rules! rir {
    ($id:expr, $dataset:expr, $room:expr, $source:expr, $file:expr) => {
        RealRir {
            id: $id,
            dataset: $dataset,
            room: $room,
            source_file: $source,
            wav: include_bytes!(concat!("real/", $file)),
        }
    };
}

/// All bundled responses, in the order used by the reports.
///
/// The measured RT20 spread is 0.09-1.05 s and every entry was measured in a
/// real room. The C50 column in the report identifies the
/// strong-early-reflection cases. The order matches `PROVENANCE.md` and is not
/// re-sorted by RT20.
pub const ALL: [RealRir; 18] = [
    rir!(
        "air-booth",
        "AIR (Aachen)",
        "booth, 2.0 m",
        "air_type1_air_binaural_booth_1_2.wav",
        "air-booth.wav"
    ),
    rir!(
        "rvb-smallroom1-near",
        "REVERB 2014",
        "small room 1, near mic",
        "RVB2014_type1_rir_smallroom1_near_angla.wav",
        "rvb-smallroom1-near.wav"
    ),
    rir!(
        "rvb-smallroom2-near",
        "REVERB 2014",
        "small room 2, near mic",
        "RVB2014_type1_rir_smallroom2_near_angla.wav",
        "rvb-smallroom2-near.wav"
    ),
    rir!(
        "rvb-smallroom2-far",
        "REVERB 2014",
        "small room 2, far mic",
        "RVB2014_type1_rir_smallroom2_far_angla.wav",
        "rvb-smallroom2-far.wav"
    ),
    rir!(
        "air-meeting",
        "AIR (Aachen)",
        "meeting room, 1.0 m",
        "air_type1_air_binaural_meeting_0_1.wav",
        "air-meeting.wav"
    ),
    rir!(
        "rwcp-e2a",
        "RWCP",
        "room e2a, circular array ch 0",
        "RWCP_type1_rir_cirline_e2a_imp090.wav",
        "rwcp-e2a.wav"
    ),
    rir!(
        "rwcp-e1b",
        "RWCP",
        "room e1b, circular array ch 0",
        "RWCP_type1_rir_circle_e1b_imp090.wav",
        "rwcp-e1b.wav"
    ),
    rir!(
        "rwcp-e1c",
        "RWCP",
        "room e1c, circular array ch 0",
        "RWCP_type1_rir_circle_e1c_imp090.wav",
        "rwcp-e1c.wav"
    ),
    rir!(
        "rvb-mediumroom1-near",
        "REVERB 2014",
        "medium room 1, near mic",
        "RVB2014_type1_rir_mediumroom1_near_angla.wav",
        "rvb-mediumroom1-near.wav"
    ),
    rir!(
        "rvb-mediumroom2-near",
        "REVERB 2014",
        "medium room 2, near mic",
        "RVB2014_type1_rir_mediumroom2_near_angla.wav",
        "rvb-mediumroom2-near.wav"
    ),
    rir!(
        "air-office",
        "AIR (Aachen)",
        "office, 2.0 m",
        "air_type1_air_binaural_office_1_2.wav",
        "air-office.wav"
    ),
    rir!(
        "rvb-largeroom1-near",
        "REVERB 2014",
        "large room 1, near mic",
        "RVB2014_type1_rir_largeroom1_near_angla.wav",
        "rvb-largeroom1-near.wav"
    ),
    rir!(
        "rvb-mediumroom2-far",
        "REVERB 2014",
        "medium room 2, far mic",
        "RVB2014_type1_rir_mediumroom2_far_angla.wav",
        "rvb-mediumroom2-far.wav"
    ),
    rir!(
        "air-stairway",
        "AIR (Aachen)",
        "stairway, 1.0 m",
        "air_type1_air_binaural_stairway_1_1_0.wav",
        "air-stairway.wav"
    ),
    rir!(
        "rwcp-ofc",
        "RWCP",
        "room ofc, circular array ch 0",
        "RWCP_type1_rir_cirline_ofc_imp090.wav",
        "rwcp-ofc.wav"
    ),
    rir!(
        "air-lecture",
        "AIR (Aachen)",
        "lecture room, 2.0 m",
        "air_type1_air_binaural_lecture_0_2.wav",
        "air-lecture.wav"
    ),
    rir!(
        "rvb-largeroom2-near",
        "REVERB 2014",
        "large room 2, near mic",
        "RVB2014_type1_rir_largeroom2_near_angla.wav",
        "rvb-largeroom2-near.wav"
    ),
    rir!(
        "rvb-largeroom2-far",
        "REVERB 2014",
        "large room 2, far mic",
        "RVB2014_type1_rir_largeroom2_far_angla.wav",
        "rvb-largeroom2-far.wav"
    ),
];

/// Looks up a bundled response by id.
pub fn find(id: &str) -> &'static RealRir {
    ALL.iter()
        .find(|rir| rir.id == id)
        .unwrap_or_else(|| panic!("no bundled real RIR with id {id:?}"))
}

impl RealRir {
    /// The measured response as `f32` at 16 kHz, normalised to unit L2 energy.
    ///
    /// The WAV is parsed on every call; the files are 20-50 KB, and the scene
    /// generator calls this once per scene. The parser accepts 16 kHz or 48 kHz
    /// 16-bit PCM with any channel count, uses channel 0 and resamples 48 kHz to
    /// 16 kHz, so a re-extracted file cannot silently change the sample rate.
    pub fn impulse(&self) -> Vec<f32> {
        let mut impulse = parse_wav_channel0(self.wav, self.id);
        assert!(!impulse.is_empty(), "{}: empty response", self.id);
        let energy = impulse
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>();
        assert!(energy > 0.0, "{}: silent response", self.id);
        let scale = (1.0 / energy.sqrt()) as f32;
        for value in &mut impulse {
            *value *= scale;
        }
        impulse
    }

    /// Cached unit-energy response, for reports that read all eighteen.
    pub fn cached_impulse(&self) -> &'static [f32] {
        static CACHE: OnceLock<Vec<(&'static str, Vec<f32>)>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| {
            ALL.iter()
                .map(|rir| (rir.id, rir.impulse()))
                .collect::<Vec<_>>()
        });
        &cache
            .iter()
            .find(|(id, _)| *id == self.id)
            .expect("cached real RIR")
            .1
    }
}

/// Parses channel 0 of a 16-bit PCM WAV and returns it at 16 kHz.
///
/// Accepted inputs: 16-bit PCM, one or more channels, 16 000 Hz or 48 000 Hz.
/// A 48 kHz file is decimated by three with a 63-tap Blackman-windowed sinc
/// low-pass (cutoff 7.2 kHz). Multi-channel input takes channel 0 and ignores
/// the rest; the source measurements are binaural or array recordings whose
/// channel 0 is a complete response on its own. Any other rate, bit depth or
/// compression is rejected: silently treating 44.1 kHz as 16 kHz would change
/// every delay and every ERLE number in the report.
fn parse_wav_channel0(bytes: &[u8], id: &str) -> Vec<f32> {
    fn u16_at(bytes: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([bytes[at], bytes[at + 1]])
    }
    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }
    assert!(bytes.len() > 44, "{id}: truncated WAV");
    assert_eq!(&bytes[0..4], b"RIFF", "{id}: not a RIFF file");
    assert_eq!(&bytes[8..12], b"WAVE", "{id}: not a WAVE file");

    let mut format: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<(usize, usize)> = None;
    let mut at = 12usize;
    while at + 8 <= bytes.len() {
        let tag = &bytes[at..at + 4];
        let size = u32_at(bytes, at + 4) as usize;
        let body = at + 8;
        let end = body.checked_add(size).expect("chunk size overflow");
        assert!(end <= bytes.len(), "{id}: truncated {:?} chunk", tag);
        if tag == b"fmt " {
            assert!(size >= 16, "{id}: short fmt chunk");
            format = Some((
                u16_at(bytes, body),
                u16_at(bytes, body + 2),
                u32_at(bytes, body + 4),
                u16_at(bytes, body + 14),
            ));
        } else if tag == b"data" {
            data = Some((body, size));
        }
        at = end + (size & 1);
    }

    let (audio_format, channels, rate, bits) =
        format.unwrap_or_else(|| panic!("{id}: no fmt chunk"));
    assert_eq!(audio_format, 1, "{id}: not PCM");
    assert!(channels >= 1, "{id}: invalid channel count {channels}");
    assert_eq!(bits, 16, "{id}: expected 16-bit samples, found {bits}");
    assert!(
        rate == 16_000 || rate == 48_000,
        "{id}: expected 16 kHz or 48 kHz, found {rate} Hz"
    );
    let block_align = channels as usize * 2;
    let (offset, size) = data.unwrap_or_else(|| panic!("{id}: no data chunk"));
    assert_eq!(
        size % block_align,
        0,
        "{id}: data chunk is not a whole number of {channels}-channel frames"
    );

    let samples: Vec<f32> = bytes[offset..offset + size]
        .chunks_exact(block_align)
        .map(|frame| f32::from(i16::from_le_bytes([frame[0], frame[1]])) / 32768.0)
        .collect();
    if rate == 16_000 {
        samples
    } else {
        decimate_3(&samples)
    }
}

/// Number of taps in the fixed 48 kHz -> 16 kHz anti-alias filter.
const DECIMATION_TAPS: usize = 63;
/// Normalised cutoff of that filter, in cycles per 48 kHz input sample
/// (0.15 x 48 kHz = 7.2 kHz, below the 8 kHz output Nyquist).
const DECIMATION_CUTOFF: f64 = 0.15;

/// Decimates by exactly three with a Blackman-windowed sinc low-pass.
///
/// The kernel is built from `f64` constants at call time and normalised to unit
/// DC gain, so the filter is identical on every machine and does not change the
/// response's overall level. Edges are zero-padded; the bundled files are
/// trimmed to the measured onset, so the first few samples are not important,
/// and the tail is discarded by the 1.6 s trim anyway.
fn decimate_3(input: &[f32]) -> Vec<f32> {
    use std::f64::consts::PI;

    let half = (DECIMATION_TAPS / 2) as isize;
    let mut kernel = [0.0f64; DECIMATION_TAPS];
    let mut sum = 0.0f64;
    for (index, coefficient) in kernel.iter_mut().enumerate() {
        let offset = index as isize - half;
        let x = offset as f64;
        let sinc = if offset == 0 {
            2.0 * DECIMATION_CUTOFF
        } else {
            (2.0 * PI * DECIMATION_CUTOFF * x).sin() / (PI * x)
        };
        let phase = 2.0 * PI * index as f64 / (DECIMATION_TAPS - 1) as f64;
        let window = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
        *coefficient = sinc * window;
        sum += *coefficient;
    }
    assert!(sum.abs() > 1.0e-12, "degenerate decimation kernel");
    for coefficient in &mut kernel {
        *coefficient /= sum;
    }

    let mut out = Vec::with_capacity(input.len() / 3);
    for output_index in 0..input.len() / 3 {
        let centre = (output_index * 3) as isize;
        let mut accumulator = 0.0f64;
        for (index, &coefficient) in kernel.iter().enumerate() {
            let source = centre + index as isize - half;
            if source >= 0 && (source as usize) < input.len() {
                accumulator += coefficient * input[source as usize] as f64;
            }
        }
        out.push(accumulator as f32);
    }
    out
}

/// Energy decay time T20, in seconds.
///
/// Schroeder backward integration of `impulse^2`, with the -5 dB and -25 dB
/// crossings linearly interpolated in time and the T20 slope extrapolated to
/// 60 dB. Returns `None` when the response never decays 25 dB below its own
/// total energy (a truncated or noise-floor-limited measurement); the bundled
/// set is checked to return `Some` for every entry.
pub fn rt20_s(impulse: &[f32]) -> Option<f32> {
    let total = impulse
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>();
    if total <= 0.0 {
        return None;
    }
    let mut remaining = total;
    let mut previous_db = 0.0f64;
    let mut previous_time = 0.0f64;
    let mut t5 = None;
    let mut t25 = None;
    for (index, &value) in impulse.iter().enumerate() {
        let time = index as f64 / FS as f64;
        let db = 10.0 * (remaining / total).max(1.0e-30).log10();
        for (target, slot) in [(-5.0f64, &mut t5), (-25.0f64, &mut t25)] {
            if slot.is_none() && previous_db >= target && db < target {
                let span = previous_db - db;
                let fraction = if span > 0.0 {
                    (previous_db - target) / span
                } else {
                    0.0
                };
                *slot = Some(previous_time + fraction * (time - previous_time));
            }
        }
        remaining -= (value as f64) * (value as f64);
        previous_db = db;
        previous_time = time;
    }
    match (t5, t25) {
        (Some(t5), Some(t25)) if t25 > t5 => Some((60.0 * (t25 - t5) / 20.0) as f32),
        _ => None,
    }
}

/// Early-to-late energy ratio C50, in dB: energy in 0-50 ms over the rest.
pub fn c50_db(impulse: &[f32]) -> f32 {
    let split = (0.050 * FS) as usize;
    let early = impulse[..split.min(impulse.len())]
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>();
    let late = impulse[split.min(impulse.len())..]
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>();
    (10.0 * ((early + 1.0e-30) / (late + 1.0e-30)).log10()) as f32
}

/// Shape summary used by the report table.
#[derive(Clone, Copy, Debug)]
pub struct RealProfile {
    pub taps: usize,
    pub span_ms: f32,
    pub peak_index: usize,
    pub peak_ms: f32,
    pub centroid_ms: f32,
    pub energy_before_40ms: f32,
    pub rt20_s: f32,
    pub c50_db: f32,
}

/// Measures the shape and acoustic metrics of one response.
pub fn analyse(impulse: &[f32]) -> RealProfile {
    let shape = profile(impulse);
    RealProfile {
        taps: shape.taps,
        span_ms: shape.taps as f32 / (FS / 1000.0),
        peak_index: shape.peak_index,
        peak_ms: shape.peak_index as f32 / (FS / 1000.0),
        centroid_ms: shape.centroid_ms,
        energy_before_40ms: shape.energy_before_40ms,
        rt20_s: rt20_s(impulse).unwrap_or(f32::NAN),
        c50_db: c50_db(impulse),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bundled_response_parses_and_is_unit_energy() {
        for rir in &ALL {
            let impulse = rir.impulse();
            assert!(!impulse.is_empty(), "{}: empty", rir.id);
            assert!(
                impulse.iter().all(|value| value.is_finite()),
                "{}: non-finite sample",
                rir.id
            );
            let energy = impulse.iter().map(|value| value * value).sum::<f32>();
            assert!((energy - 1.0).abs() < 1.0e-4, "{}: energy {energy}", rir.id);
        }
    }

    #[test]
    fn the_set_spans_the_requested_reverberation_range() {
        let metrics: Vec<(f32, f32)> = ALL
            .iter()
            .map(|rir| {
                let impulse = rir.impulse();
                let rt = rt20_s(&impulse).unwrap_or_else(|| panic!("{}: no T20", rir.id));
                (rt, c50_db(&impulse))
            })
            .collect();

        let rt_min = metrics
            .iter()
            .map(|(rt, _)| *rt)
            .fold(f32::INFINITY, f32::min);
        let rt_max = metrics
            .iter()
            .map(|(rt, _)| *rt)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(rt_min <= 0.20, "shortest real RIR is {rt_min:.2} s");
        assert!(rt_max >= 1.00, "longest real RIR is {rt_max:.2} s");
        assert!(
            metrics.len() >= 10 && metrics.len() <= 20,
            "the requested set size is ~10-20, found {}",
            metrics.len()
        );

        // At least a couple of speech-relevant rooms where the early reflections
        // are strong enough that the direct path is not dominant.
        let strong_early = metrics.iter().filter(|(_, c50)| *c50 < 8.0).count();
        assert!(
            strong_early >= 4,
            "only {strong_early} responses have C50 below 8 dB"
        );
        let very_strong = metrics.iter().filter(|(_, c50)| *c50 < 5.0).count();
        assert!(
            very_strong >= 1,
            "no response has C50 below 5 dB (the strong early reflection case)"
        );
    }

    #[test]
    fn all_ids_are_unique_and_source_names_are_recorded() {
        let mut ids: Vec<&str> = ALL.iter().map(|rir| rir.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), ALL.len());
        for rir in &ALL {
            assert!(
                rir.source_file.ends_with(".wav"),
                "{}: source file name",
                rir.id
            );
            assert!(!rir.dataset.is_empty() && !rir.room.is_empty());
        }
    }

    #[test]
    fn a_bad_wav_header_is_rejected() {
        let mut bytes = ALL[0].wav.to_vec();
        // Corrupt the sample-rate field of the fmt chunk (mono, 16-bit at 44).
        let rate_at = 24;
        assert_eq!(&bytes[12..16], b"fmt ");
        bytes[rate_at..rate_at + 4].copy_from_slice(&8_000u32.to_le_bytes());
        let outcome = std::panic::catch_unwind(|| parse_wav_channel0(&bytes, "t"));
        assert!(outcome.is_err(), "an 8 kHz file was accepted");
    }

    /// Builds a minimal 16-bit PCM WAV with interleaved channels.
    fn wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        fn put(bytes: &mut Vec<u8>, value: &[u8]) {
            bytes.extend_from_slice(value);
        }
        let block_align = channels * 2;
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        put(&mut bytes, b"RIFF");
        put(&mut bytes, &(36 + data_len).to_le_bytes());
        put(&mut bytes, b"WAVE");
        put(&mut bytes, b"fmt ");
        put(&mut bytes, &16u32.to_le_bytes());
        put(&mut bytes, &1u16.to_le_bytes());
        put(&mut bytes, &channels.to_le_bytes());
        put(&mut bytes, &rate.to_le_bytes());
        put(&mut bytes, &(rate * block_align as u32).to_le_bytes());
        put(&mut bytes, &block_align.to_le_bytes());
        put(&mut bytes, &16u16.to_le_bytes());
        put(&mut bytes, b"data");
        put(&mut bytes, &data_len.to_le_bytes());
        for &sample in samples {
            put(&mut bytes, &sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn forty_eight_kilohertz_input_is_resampled_to_sixteen() {
        let rate = 48_000u32;
        let seconds = 0.25f32;
        let frames = (rate as f32 * seconds) as usize;
        let samples: Vec<i16> = (0..frames)
            .map(|n| {
                let phase = 2.0 * std::f32::consts::PI * 1_000.0 * n as f32 / rate as f32;
                (0.5 * phase.sin() * 32767.0) as i16
            })
            .collect();
        let parsed = parse_wav_channel0(&wav(rate, 1, &samples), "48k");
        assert_eq!(parsed.len(), frames / 3, "length after decimation");
        assert!(parsed.iter().all(|value| value.is_finite()));
        // Interior RMS of a 0.5-amplitude sine is 0.5/sqrt(2); the edges are
        // excluded because the zero-padded FIR shortens the first/last samples.
        let interior = &parsed[64..parsed.len() - 64];
        let rms = (interior.iter().map(|value| value * value).sum::<f32>() / interior.len() as f32)
            .sqrt();
        assert!(
            (rms - 0.5 / 2.0f32.sqrt()).abs() < 0.02,
            "resampled RMS {rms}"
        );
    }

    #[test]
    fn multi_channel_input_uses_channel_zero() {
        let frames = 400usize;
        let mut samples = Vec::with_capacity(frames * 2);
        for _ in 0..frames {
            samples.push(-16_384i16); // left: -0.5
            samples.push(16_384i16); // right: +0.5
        }
        let parsed = parse_wav_channel0(&wav(16_000, 2, &samples), "stereo");
        assert_eq!(parsed.len(), frames);
        assert!(
            parsed.iter().all(|value| (*value + 0.5).abs() < 1.0e-6),
            "channel 0 was not the left channel"
        );
    }

    #[test]
    fn forty_four_point_one_kilohertz_is_rejected() {
        let samples = vec![0i16; 441];
        let outcome =
            std::panic::catch_unwind(|| parse_wav_channel0(&wav(44_100, 1, &samples), "44k"));
        assert!(outcome.is_err(), "a 44.1 kHz file was accepted");
    }
}
