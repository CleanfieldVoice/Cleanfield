//! Minimal 16-bit mono PCM WAV reader/writer for the validation driver.

use std::fs;
use std::path::Path;

pub struct Wav {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

pub fn read(path: &Path) -> Result<Wav, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{} is not a RIFF/WAVE file", path.display()));
    }

    let mut format = None;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits = None;
    let mut data: Option<&[u8]> = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        let end = body.saturating_add(size).min(bytes.len());
        if id == b"fmt " && size >= 16 {
            format = Some(u16::from_le_bytes(
                bytes[body..body + 2].try_into().unwrap(),
            ));
            channels = Some(u16::from_le_bytes(
                bytes[body + 2..body + 4].try_into().unwrap(),
            ));
            sample_rate = Some(u32::from_le_bytes(
                bytes[body + 4..body + 8].try_into().unwrap(),
            ));
            bits = Some(u16::from_le_bytes(
                bytes[body + 14..body + 16].try_into().unwrap(),
            ));
        } else if id == b"data" {
            data = Some(&bytes[body..end]);
        }
        pos = body + size + (size & 1);
    }

    let (format, channels, sample_rate, bits) = match (format, channels, sample_rate, bits) {
        (Some(f), Some(c), Some(s), Some(b)) => (f, c, s, b),
        _ => return Err(format!("{} has no valid fmt chunk", path.display())),
    };
    if format != 1 || channels != 1 || bits != 16 {
        return Err(format!(
            "{} is format={format} channels={channels} bits={bits}; only mono 16-bit PCM is supported",
            path.display()
        ));
    }
    let data = data.ok_or_else(|| format!("{} has no data chunk", path.display()))?;
    let samples = data
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0)
        .collect();
    Ok(Wav {
        sample_rate,
        samples,
    })
}

pub fn to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&value| {
            let scaled = (value as f64 * 32768.0).round();
            scaled.clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

pub fn write(path: &Path, sample_rate: u32, samples: &[f32]) -> Result<(), String> {
    let pcm = to_i16(samples);
    let data_len = (pcm.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + pcm.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for value in &pcm {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}
