//! Reference-dump reader: `.npz` (ZIP + NPY) and the flat `CFRF` binary.
//!
//! `.npz` files are ZIP archives; NumPy's compressed variant uses raw DEFLATE
//! members, which the local [`crate::inflate`] decoder handles.  Only the
//! little-endian dtypes used by the reference dumps are accepted.
//!
//! `CFRF` is the optional flat fallback:
//!
//! ```text
//! magic "CFRF", u16 version (1), u16 count
//! per array: u8 name_len, name bytes, u8 dtype (0 = f32), u8 rank,
//!            u32 dims[4], u32 byte_len, payload
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::inflate::{inflate, InflateError};

/// Reader failure.
#[derive(Debug)]
pub struct RefError(pub String);

impl fmt::Display for RefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RefError {}

impl From<InflateError> for RefError {
    fn from(e: InflateError) -> Self {
        RefError(format!("deflate: {e}"))
    }
}

/// One array from a reference dump.
#[derive(Clone, Debug)]
pub struct Array {
    /// NumPy-style descriptor (`<f4`, `<i2`, ...).
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Array {
    pub fn itemsize(&self) -> usize {
        match self.descr.as_str() {
            "<f4" | "<i4" | "<u4" | "|f4" => 4,
            "<i2" | "<u2" => 2,
            "|i1" | "<i1" | "|u1" | "<u1" => 1,
            _ => 0,
        }
    }

    /// Little-endian f32 values, if this array is f32.
    pub fn as_f32(&self) -> Option<Vec<f32>> {
        if self.itemsize() != 4 || !self.descr.contains('f') {
            return None;
        }
        Some(
            self.data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    /// Little-endian signed 16-bit values, if this array is int16.
    pub fn as_i16(&self) -> Option<Vec<i16>> {
        if self.itemsize() != 2 || !self.descr.contains('i') {
            return None;
        }
        Some(
            self.data
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect(),
        )
    }

    /// Little-endian unsigned 32-bit values, if this array is uint32.
    pub fn as_u32(&self) -> Option<Vec<u32>> {
        if self.itemsize() != 4 || !self.descr.contains('u') {
            return None;
        }
        Some(
            self.data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }
}

/// A set of named arrays.
pub struct RefData {
    arrays: HashMap<String, Array>,
}

impl RefData {
    /// Parse `.npz` or `CFRF` bytes (auto-detected by magic).
    pub fn open(bytes: &[u8]) -> Result<RefData, RefError> {
        if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
            let members = read_zip(bytes)?;
            let mut arrays = HashMap::new();
            for (name, data) in members {
                if !name.ends_with(".npy") {
                    continue;
                }
                let stem = name
                    .rsplit('/')
                    .next()
                    .unwrap_or(&name)
                    .trim_end_matches(".npy")
                    .to_string();
                if let Some(array) = parse_npy(&data)? {
                    arrays.insert(stem, array);
                }
            }
            if arrays.is_empty() {
                return Err(RefError("archive has no .npy members".into()));
            }
            Ok(RefData { arrays })
        } else if bytes.starts_with(b"CFRF") {
            Ok(RefData {
                arrays: read_cfrf(bytes)?,
            })
        } else {
            Err(RefError(
                "unrecognised reference format (expected .npz or CFRF)".into(),
            ))
        }
    }

    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.arrays.get(name).map(|a| a.shape.clone())
    }

    pub fn f32(&self, name: &str) -> Option<Vec<f32>> {
        self.arrays.get(name).and_then(|a| a.as_f32())
    }

    pub fn i16(&self, name: &str) -> Option<Vec<i16>> {
        self.arrays.get(name).and_then(|a| a.as_i16())
    }

    pub fn u32(&self, name: &str) -> Option<Vec<u32>> {
        self.arrays.get(name).and_then(|a| a.as_u32())
    }
}

// ---------------------------------------------------------------------------
// ZIP
// ---------------------------------------------------------------------------
fn read_zip(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, RefError> {
    if bytes.len() < 22 {
        return Err(RefError("zip too short".into()));
    }
    let start = bytes.len().saturating_sub(22 + 0xFFFF);
    let mut eocd = None;
    for i in (start..=bytes.len() - 22).rev() {
        if &bytes[i..i + 4] == b"PK\x05\x06" {
            eocd = Some(i);
            break;
        }
    }
    let eocd = eocd.ok_or_else(|| RefError("zip end-of-central-directory not found".into()))?;
    let count = u16le(bytes, eocd + 10) as usize;
    let cd_offset = u32le(bytes, eocd + 16) as usize;
    if count == 0xFFFF || cd_offset == 0xFFFF_FFFF {
        return Err(RefError("zip64 is not supported".into()));
    }

    let mut out = Vec::with_capacity(count);
    let mut at = cd_offset;
    for _ in 0..count {
        if at + 46 > bytes.len() || &bytes[at..at + 4] != b"PK\x01\x02" {
            return Err(RefError(format!("bad central directory entry at {at:#x}")));
        }
        let method = u16le(bytes, at + 10);
        let comp_size = u32le(bytes, at + 20) as usize;
        let uncomp_size = u32le(bytes, at + 24) as usize;
        let name_len = u16le(bytes, at + 28) as usize;
        let extra_len = u16le(bytes, at + 30) as usize;
        let comment_len = u16le(bytes, at + 32) as usize;
        let local_off = u32le(bytes, at + 42) as usize;
        let name_end = at + 46 + name_len;
        if name_end > bytes.len() {
            return Err(RefError("zip name out of bounds".into()));
        }
        let name = String::from_utf8_lossy(&bytes[at + 46..name_end]).to_string();

        if local_off + 30 > bytes.len() || &bytes[local_off..local_off + 4] != b"PK\x03\x04" {
            return Err(RefError(format!("bad local header for {name}")));
        }
        let l_name_len = u16le(bytes, local_off + 26) as usize;
        let l_extra_len = u16le(bytes, local_off + 28) as usize;
        let data_off = local_off + 30 + l_name_len + l_extra_len;
        if data_off + comp_size > bytes.len() {
            return Err(RefError(format!("zip payload out of bounds for {name}")));
        }
        let compressed = &bytes[data_off..data_off + comp_size];
        let data = match method {
            0 => compressed.to_vec(),
            8 => inflate(compressed)?,
            other => {
                return Err(RefError(format!(
                    "unsupported zip method {other} for {name}"
                )))
            }
        };
        if data.len() != uncomp_size {
            return Err(RefError(format!(
                "zip size mismatch for {name}: got {}, expected {uncomp_size}",
                data.len()
            )));
        }
        out.push((name, data));
        at = name_end + extra_len + comment_len;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// NPY
// ---------------------------------------------------------------------------
fn parse_npy(bytes: &[u8]) -> Result<Option<Array>, RefError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(RefError("bad npy magic".into()));
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16le(bytes, 8) as usize, 10usize),
        2 | 3 => (u32le(bytes, 8) as usize, 12usize),
        other => return Err(RefError(format!("unsupported npy version {other}"))),
    };
    if header_start + header_len > bytes.len() {
        return Err(RefError("npy header out of bounds".into()));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|_| RefError("npy header is not utf-8".into()))?;

    let descr = extract_quoted(header, "'descr':")
        .ok_or_else(|| RefError(format!("npy header has no descr: {header}")))?;
    if header.contains("'fortran_order': True") {
        return Err(RefError(
            "Fortran-order npy arrays are not supported".into(),
        ));
    }
    let shape = extract_shape(header)
        .ok_or_else(|| RefError(format!("npy header has no shape: {header}")))?;

    let itemsize = match descr.as_str() {
        "<f4" | "<i4" | "<u4" | "|f4" => 4,
        "<i2" | "<u2" => 2,
        "|i1" | "<i1" | "|u1" | "<u1" => 1,
        _ => return Ok(None), // e.g. an object/string metadata array
    };
    let elements: usize = shape.iter().product();
    let data_start = header_start + header_len;
    let data_end = data_start + elements * itemsize;
    if data_end > bytes.len() {
        return Err(RefError("npy payload out of bounds".into()));
    }
    Ok(Some(Array {
        descr,
        shape,
        data: bytes[data_start..data_end].to_vec(),
    }))
}

fn extract_quoted(header: &str, key: &str) -> Option<String> {
    let at = header.find(key)? + key.len();
    let rest = &header[at..];
    let q1 = rest.find('\'')?;
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('\'')?;
    Some(rest[..q2].to_string())
}

fn extract_shape(header: &str) -> Option<Vec<usize>> {
    let at = header.find("'shape':")? + "'shape':".len();
    let rest = &header[at..];
    let open = rest.find('(')?;
    let close = rest[open..].find(')')? + open;
    let inner = &rest[open + 1..close];
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        shape.push(part.parse::<usize>().ok()?);
    }
    Some(shape)
}

// ---------------------------------------------------------------------------
// CFRF flat binary
// ---------------------------------------------------------------------------
fn read_cfrf(bytes: &[u8]) -> Result<HashMap<String, Array>, RefError> {
    if bytes.len() < 8 {
        return Err(RefError("CFRF too short".into()));
    }
    let version = u16le(bytes, 4);
    if version != 1 {
        return Err(RefError(format!("unsupported CFRF version {version}")));
    }
    let count = u16le(bytes, 6) as usize;
    let mut arrays = HashMap::new();
    let mut at = 8usize;
    for _ in 0..count {
        if at >= bytes.len() {
            return Err(RefError("CFRF truncated".into()));
        }
        let name_len = bytes[at] as usize;
        at += 1;
        if at + name_len + 2 + 16 + 4 > bytes.len() {
            return Err(RefError("CFRF entry truncated".into()));
        }
        let name = String::from_utf8_lossy(&bytes[at..at + name_len]).to_string();
        at += name_len;
        let dtype = bytes[at];
        let rank = bytes[at + 1] as usize;
        at += 2;
        let dims = [
            u32le(bytes, at) as usize,
            u32le(bytes, at + 4) as usize,
            u32le(bytes, at + 8) as usize,
            u32le(bytes, at + 12) as usize,
        ];
        at += 16;
        let byte_len = u32le(bytes, at) as usize;
        at += 4;
        if at + byte_len > bytes.len() {
            return Err(RefError("CFRF payload truncated".into()));
        }
        if dtype != 0 || rank > 4 {
            return Err(RefError(format!("unsupported CFRF dtype {dtype}")));
        }
        let descr = "<f4".to_string();
        arrays.insert(
            name,
            Array {
                descr,
                shape: dims[..rank].to_vec(),
                data: bytes[at..at + byte_len].to_vec(),
            },
        );
        at += byte_len;
    }
    Ok(arrays)
}

#[inline]
fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

#[inline]
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
