//! `.cfwm` container reader.
//!
//! The container is shared by every CleanField model.  It carries numeric
//! tags only: no names, no source offsets and no text.  The voice model uses
//! model id 1 and 37 tensors; the echo model uses model id 2 and 104 tensors,
//! which is why the table can hold up to [`MAX_TENSORS`] entries.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! header (32 bytes)
//!   0  magic "CFWM"
//!   4  u16 format version (2)
//!   6  u16 model id
//!   8  u16 tensor count
//!  10  u16 flags (0)
//!  12  u32 table offset (32)
//!  16  u32 data offset
//!  20  u32 data length
//!  24  u32 file length
//!  28  u32 reserved (0)
//! tensor table (32 bytes each)
//!   0  u16 tag
//!   2  u8  dtype (0 = f32, 1 = i8, 2 = i16, 3 = u32 bit pattern, 4 = f16)
//!   3  u8  rank
//!   4  u32 reserved
//!   8  u32 dims[4]
//!  24  u32 data offset (relative to the data region)
//!  28  u32 byte length
//! data region (tensors 64-byte aligned, zero padded)
//! ```
//!
//! The container stores numeric tags only: no names, no source offsets and no
//! text.  Parse once at load time; the inference path borrows the tensor
//! slices directly.
//!
//! `dtype = 3` holds the model's scalar constants as raw `u32` words.  A
//! constant that is an `f32` is stored as its bit pattern and must be restored
//! with `f32::from_bits`; writing it as a decimal literal in code is forbidden
//! (one earlier engine differed by one ULP for exactly that reason).
//!
//! `dtype = 4` (f16) is part of the format for containers whose kernels can
//! consume half precision directly; the current builders widen f16 source
//! tensors to f32 exactly at build time, so the shipped voice and echo
//! containers contain no f16 tensor.  A reader for it is provided so a future
//! container does not need a format change.

use std::borrow::Cow;
use std::fmt;

/// Container magic.
pub const MAGIC: &[u8; 4] = b"CFWM";
/// Supported container format version.
pub const FORMAT_VERSION: u16 = 2;

/// Maximum tensors in one container.
pub const MAX_TENSORS: usize = 128;
/// Tensor descriptor size.
pub const ENTRY_LEN: usize = 32;
/// Header size.
pub const HEADER_LEN: usize = 32;

/// Data type codes.
pub const DTYPE_F32: u8 = 0;
/// Signed 8-bit integers (quantised weights and zero points).
pub const DTYPE_I8: u8 = 1;
/// Signed 16-bit integers (quantised activations).
pub const DTYPE_I16: u8 = 2;
/// Raw 32-bit words (scalar constants as bit patterns).
pub const DTYPE_U32: u8 = 3;
/// IEEE-754 binary16 (half precision).
pub const DTYPE_F16: u8 = 4;

/// Element size for a dtype code, or `None` if unknown.
#[inline]
pub const fn dtype_size(dtype: u8) -> Option<usize> {
    match dtype {
        DTYPE_F32 | DTYPE_U32 => Some(4),
        DTYPE_I16 | DTYPE_F16 => Some(2),
        DTYPE_I8 => Some(1),
        _ => None,
    }
}

/// One tensor descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorDesc {
    pub tag: u16,
    pub dtype: u8,
    pub rank: u8,
    pub dims: [u32; 4],
    /// Offset in bytes relative to the data region.
    pub byte_offset: u32,
    /// Length in bytes.
    pub byte_len: u32,
}

impl TensorDesc {
    /// Number of elements, from the declared shape.
    pub fn elements(&self) -> usize {
        self.dims[..self.rank as usize]
            .iter()
            .map(|&d| d as usize)
            .product()
    }
}

/// Parse / validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CfwmError {
    TooShort,
    BadMagic,
    UnsupportedVersion(u16),
    BadModelId(u16),
    TooManyTensors(usize),
    BadHeader(&'static str),
    BadTensor(u16, &'static str),
    DuplicateTag(u16),
}

impl fmt::Display for CfwmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfwmError::TooShort => write!(f, "container shorter than the header"),
            CfwmError::BadMagic => write!(f, "bad container magic"),
            CfwmError::UnsupportedVersion(v) => write!(f, "unsupported container version {v}"),
            CfwmError::BadModelId(id) => write!(f, "unsupported model id {id}"),
            CfwmError::TooManyTensors(n) => write!(f, "too many tensors ({n})"),
            CfwmError::BadHeader(why) => write!(f, "bad container header: {why}"),
            CfwmError::BadTensor(tag, why) => write!(f, "bad tensor {tag}: {why}"),
            CfwmError::DuplicateTag(tag) => write!(f, "duplicate tensor tag {tag}"),
        }
    }
}

impl std::error::Error for CfwmError {}

/// Parsed, read-only container.
///
/// The whole image lives in one little-endian `u32` buffer so every tensor
/// view is 4-byte aligned.  `Owned` is the normal path (native file load and a
/// host-supplied wasm image); `Borrowed` lets the wasm build point at a
/// static, aligned model image without copying the 2.5 MB of weights.
pub struct Container {
    words: Cow<'static, [u32]>,
    format_version: u16,
    model_id: u16,
    data_offset: usize,
    data_len: usize,
    tensors: [TensorDesc; MAX_TENSORS],
    n_tensors: usize,
}

impl Container {
    /// Parse a container from bytes.  Copies once into an aligned word buffer
    /// (load time only).
    pub fn parse(bytes: &[u8]) -> Result<Container, CfwmError> {
        let mut words = Vec::with_capacity(bytes.len().div_ceil(4));
        for chunk in bytes.chunks(4) {
            let mut b = [0u8; 4];
            b[..chunk.len()].copy_from_slice(chunk);
            words.push(u32::from_le_bytes(b));
        }
        Container::parse_words(Cow::Owned(words), bytes.len())
    }

    /// Parse an image that is already a little-endian `u32` buffer.  No copy.
    ///
    /// This is the wasm host path: the caller allocates `ceil(len / 4)` words,
    /// copies the container bytes into them and hands ownership to the
    /// parser.  `len` must be a multiple of four.
    pub fn parse_owned_words(words: Vec<u32>) -> Result<Container, CfwmError> {
        let byte_len = words.len() * 4;
        Container::parse_words(Cow::Owned(words), byte_len)
    }

    /// Parse a container image held in static, 4-byte aligned memory.  No
    /// copy; the wasm build embeds the default container this way so the
    /// weights exist exactly once in the module.
    pub fn parse_static(bytes: &'static [u8]) -> Result<Container, CfwmError> {
        if !bytes.len().is_multiple_of(4) || bytes.len() < HEADER_LEN {
            return Err(CfwmError::TooShort);
        }
        if !(bytes.as_ptr() as usize).is_multiple_of(4) {
            return Err(CfwmError::BadHeader("static image is not 4-byte aligned"));
        }
        // SAFETY: `u32` and `u8` have identical size; the pointer is checked
        // 4-byte aligned and the length is a multiple of four, so the
        // reinterpretation is valid.  The input is `'static`, so the slice is.
        let words: &'static [u32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) };
        Container::parse_words(Cow::Borrowed(words), bytes.len())
    }

    fn parse_words(words: Cow<'static, [u32]>, bytes_len: usize) -> Result<Container, CfwmError> {
        if bytes_len < HEADER_LEN || words.len() * 4 < bytes_len {
            return Err(CfwmError::TooShort);
        }
        let w = words.as_ref();
        if &w[0].to_le_bytes() != MAGIC {
            return Err(CfwmError::BadMagic);
        }
        let format_version = rd16(w, 4);
        if format_version != FORMAT_VERSION {
            return Err(CfwmError::UnsupportedVersion(format_version));
        }
        let model_id = rd16(w, 6);
        let n_tensors = rd16(w, 8) as usize;
        let flags = rd16(w, 10);
        let table_offset = rd32(w, 12) as usize;
        let data_offset = rd32(w, 16) as usize;
        let data_len = rd32(w, 20) as usize;
        let file_len = rd32(w, 24) as usize;
        let reserved = rd32(w, 28);

        if flags != 0 || reserved != 0 {
            return Err(CfwmError::BadHeader("non-zero flags/reserved"));
        }
        if n_tensors > MAX_TENSORS {
            return Err(CfwmError::TooManyTensors(n_tensors));
        }
        if file_len != bytes_len || (data_offset.checked_add(data_len) != Some(file_len)) {
            return Err(CfwmError::BadHeader("length mismatch"));
        }
        let table_end = table_offset
            .checked_add(n_tensors.saturating_mul(ENTRY_LEN))
            .ok_or(CfwmError::BadHeader("table size overflow"))?;
        if table_offset < HEADER_LEN
            || !table_offset.is_multiple_of(4)
            || table_end > data_offset
            || !data_offset.is_multiple_of(4)
            || !data_len.is_multiple_of(4)
        {
            return Err(CfwmError::BadHeader("bad table/data offsets"));
        }

        let mut tensors = [TensorDesc {
            tag: 0,
            dtype: 0,
            rank: 0,
            dims: [0; 4],
            byte_offset: 0,
            byte_len: 0,
        }; MAX_TENSORS];

        for i in 0..n_tensors {
            let at = table_offset + i * ENTRY_LEN;
            let tag = rd16(w, at);
            let dtype = rd8(w, at + 2);
            let rank = rd8(w, at + 3);
            let entry_reserved = rd32(w, at + 4);
            let dims = [
                rd32(w, at + 8),
                rd32(w, at + 12),
                rd32(w, at + 16),
                rd32(w, at + 20),
            ];
            let byte_offset = rd32(w, at + 24);
            let byte_len = rd32(w, at + 28);

            if tag == 0 {
                return Err(CfwmError::BadTensor(tag, "zero tag"));
            }
            if tensors[..i].iter().any(|t| t.tag == tag) {
                return Err(CfwmError::DuplicateTag(tag));
            }
            let item = match dtype_size(dtype) {
                Some(n) => n,
                None => return Err(CfwmError::BadTensor(tag, "unsupported dtype")),
            };
            if rank > 4 {
                return Err(CfwmError::BadTensor(tag, "rank > 4"));
            }
            if entry_reserved != 0 {
                return Err(CfwmError::BadTensor(tag, "non-zero reserved field"));
            }
            let elements = dims[..rank as usize]
                .iter()
                .map(|&d| d as usize)
                .product::<usize>();
            let computed_len = elements * item;
            if byte_len as usize != computed_len {
                return Err(CfwmError::BadTensor(tag, "shape/byte length mismatch"));
            }
            if !(byte_offset as usize).is_multiple_of(item)
                || byte_offset as usize + byte_len as usize > data_len
            {
                return Err(CfwmError::BadTensor(tag, "data range out of bounds"));
            }
            if dims[rank as usize..].iter().any(|&d| d != 0) {
                return Err(CfwmError::BadTensor(tag, "non-zero trailing dimension"));
            }

            tensors[i] = TensorDesc {
                tag,
                dtype,
                rank,
                dims,
                byte_offset,
                byte_len,
            };
        }

        Ok(Container {
            words,
            format_version,
            model_id,
            data_offset,
            data_len,
            tensors,
            n_tensors,
        })
    }

    /// Container format version.
    pub fn format_version(&self) -> u16 {
        self.format_version
    }

    /// Declared model id.
    pub fn model_id(&self) -> u16 {
        self.model_id
    }

    /// Number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.n_tensors
    }

    /// Data region length in bytes.
    pub fn data_len(&self) -> usize {
        self.data_len
    }

    /// Total container length in bytes (header + table + data region).
    pub fn file_len(&self) -> usize {
        self.data_offset + self.data_len
    }

    /// Tensor descriptors in table order.
    pub fn tensors(&self) -> &[TensorDesc] {
        &self.tensors[..self.n_tensors]
    }

    /// Find a descriptor by tag.
    pub fn tensor(&self, tag: u16) -> Option<&TensorDesc> {
        self.tensors[..self.n_tensors].iter().find(|t| t.tag == tag)
    }

    /// Absolute byte offset of a tensor inside the container image.
    #[inline]
    fn absolute(&self, tag: u16, want: u8, align: usize) -> Option<(usize, usize)> {
        let desc = self.tensor(tag)?;
        if desc.dtype != want {
            return None;
        }
        let abs = self.data_offset + desc.byte_offset as usize;
        if !abs.is_multiple_of(align) {
            return None;
        }
        Some((abs, desc.byte_len as usize))
    }

    /// Byte view of a tensor, or `None` if absent.
    pub fn bytes(&self, tag: u16) -> Option<&[u8]> {
        let desc = self.tensor(tag)?;
        let abs = self.data_offset + desc.byte_offset as usize;
        let end = abs + desc.byte_len as usize;
        if end > self.words.len() * 4 {
            return None;
        }
        // SAFETY: the word buffer is a contiguous image of the whole file;
        // any byte range inside it is a valid `u8` slice.
        let all = unsafe {
            std::slice::from_raw_parts(self.words.as_ptr() as *const u8, self.words.len() * 4)
        };
        Some(&all[abs..end])
    }

    /// f32 view of a tensor, or `None` if absent / not f32.
    pub fn f32(&self, tag: u16) -> Option<&[f32]> {
        let (abs, len) = self.absolute(tag, DTYPE_F32, 4)?;
        let start = abs / 4;
        let count = len / 4;
        Some(f32_slice(&self.words[start..start + count]))
    }

    /// i8 view of a tensor, or `None` if absent / not i8.
    pub fn i8(&self, tag: u16) -> Option<&[i8]> {
        let (abs, len) = self.absolute(tag, DTYPE_I8, 1)?;
        let start = abs;
        let all = unsafe {
            std::slice::from_raw_parts(self.words.as_ptr() as *const i8, self.words.len() * 4)
        };
        Some(&all[start..start + len])
    }

    /// i16 view of a tensor, or `None` if absent / not i16.
    pub fn i16(&self, tag: u16) -> Option<&[i16]> {
        let (abs, len) = self.absolute(tag, DTYPE_I16, 2)?;
        let start = abs / 2;
        let count = len / 2;
        let all = unsafe {
            std::slice::from_raw_parts(self.words.as_ptr() as *const i16, self.words.len() * 2)
        };
        Some(&all[start..start + count])
    }

    /// u16 bit-pattern view of an f16 tensor, or `None` if absent / not f16.
    ///
    /// The raw IEEE-754 binary16 words are returned; the caller widens them
    /// with its own conversion (the widening itself is exact).
    pub fn f16_bits(&self, tag: u16) -> Option<&[u16]> {
        let (abs, len) = self.absolute(tag, DTYPE_F16, 2)?;
        let start = abs / 2;
        let count = len / 2;
        let all = unsafe {
            std::slice::from_raw_parts(self.words.as_ptr() as *const u16, self.words.len() * 2)
        };
        Some(&all[start..start + count])
    }

    /// u32 bit-pattern view of a tensor, or `None` if absent / not u32.
    pub fn u32_bits(&self, tag: u16) -> Option<&[u32]> {
        let (abs, len) = self.absolute(tag, DTYPE_U32, 4)?;
        let start = abs / 4;
        let count = len / 4;
        Some(&self.words[start..start + count])
    }
}

/// Reinterpret a word slice as f32.
///
/// SAFETY: `u32` and `f32` have identical size (4) and alignment (4), and the
/// words are the little-endian container image (from `u32::from_le_bytes`, a
/// host-supplied word buffer or the aligned static image), so the bit patterns
/// are exactly the stored little-endian f32 values.
#[inline]
fn f32_slice(words: &[u32]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const f32, words.len()) }
}

/// Read one byte from the little-endian word image at byte offset `at`.
#[inline]
fn rd8(words: &[u32], at: usize) -> u8 {
    (words[at / 4] >> (8 * (at % 4))) as u8
}

/// Read one little-endian `u16` word at byte offset `at` (which must be even).
#[inline]
fn rd16(words: &[u32], at: usize) -> u16 {
    (words[at / 4] >> (8 * (at % 4))) as u16
}

/// Read one little-endian `u32` word at byte offset `at` (4-byte aligned).
#[inline]
fn rd32(words: &[u32], at: usize) -> u32 {
    words[at / 4]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le16(b: &mut [u8], o: usize, v: u16) {
        b[o..o + 2].copy_from_slice(&v.to_le_bytes());
    }
    fn le32(b: &mut [u8], o: usize, v: u32) {
        b[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }

    #[test]
    fn rejects_truncated_and_bad_magic() {
        assert!(matches!(Container::parse(&[]), Err(CfwmError::TooShort)));
        let mut bytes = [0u8; 64];
        bytes[..4].copy_from_slice(b"XXXX");
        assert!(matches!(Container::parse(&bytes), Err(CfwmError::BadMagic)));
    }

    #[test]
    fn parses_minimal_container() {
        // Header + one tensor descriptor + one f32 payload.
        let mut bytes = vec![0u8; 32 + 32 + 4];
        bytes[..4].copy_from_slice(MAGIC);
        le16(&mut bytes, 4, FORMAT_VERSION);
        le16(&mut bytes, 6, 1);
        le16(&mut bytes, 8, 1);
        le32(&mut bytes, 12, 32);
        le32(&mut bytes, 16, 64);
        le32(&mut bytes, 20, 4);
        le32(&mut bytes, 24, 68);
        // tensor entry
        le16(&mut bytes, 32, 7);
        bytes[34] = DTYPE_F32;
        bytes[35] = 1;
        le32(&mut bytes, 40, 1);
        le32(&mut bytes, 56, 0);
        le32(&mut bytes, 60, 4);
        // payload
        bytes[64..68].copy_from_slice(&1.5f32.to_le_bytes());
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.tensor_count(), 1);
        assert_eq!(c.f32(7).unwrap(), &[1.5f32]);
        assert!(c.f32(8).is_none());
        assert!(c.i8(7).is_none());
    }

    #[test]
    fn parses_integer_and_bitpattern_tensors() {
        // Three tensors: i8 (3), i16 (2), u32 (2).  Data region starts at 128.
        let table = 32usize;
        let data_off = 32 + 3 * ENTRY_LEN; // 128, already 4-aligned
        let payload_i8 = [0x80u8, 0x7F, 0xFF];
        let payload_i16: [i16; 2] = [-8192, 8192];
        let payload_u32: [u32; 2] = [0x4270_0000, 0x0000_2000];
        let mut data = Vec::new();
        data.extend_from_slice(&payload_i8);
        data.resize(64, 0);
        for v in payload_i16 {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data.resize(128, 0);
        for v in payload_u32 {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let file_len = data_off + data.len();
        let mut bytes = vec![0u8; file_len];
        bytes[..4].copy_from_slice(MAGIC);
        le16(&mut bytes, 4, FORMAT_VERSION);
        le16(&mut bytes, 6, 1);
        le16(&mut bytes, 8, 3);
        le32(&mut bytes, 12, table as u32);
        le32(&mut bytes, 16, data_off as u32);
        le32(&mut bytes, 20, data.len() as u32);
        le32(&mut bytes, 24, file_len as u32);

        let entries: [(u16, u8, u32, u32, u32, u32); 3] = [
            // tag, dtype, dim0, offset, len, reserved
            (11, DTYPE_I8, 3, 0, 3, 0),
            (12, DTYPE_I16, 2, 64, 4, 0),
            (13, DTYPE_U32, 2, 128, 8, 0),
        ];
        for (i, &(tag, dtype, dim0, off, len, _)) in entries.iter().enumerate() {
            let at = table + i * ENTRY_LEN;
            le16(&mut bytes, at, tag);
            bytes[at + 2] = dtype;
            bytes[at + 3] = 1;
            le32(&mut bytes, at + 8, dim0);
            le32(&mut bytes, at + 24, off);
            le32(&mut bytes, at + 28, len);
        }
        bytes[data_off..data_off + data.len()].copy_from_slice(&data);

        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.i8(11).unwrap(), &[-128, 127, -1]);
        assert_eq!(c.i16(12).unwrap(), &[-8192, 8192]);
        assert_eq!(c.u32_bits(13).unwrap(), &payload_u32);
        assert!(c.f32(13).is_none());
    }

    #[test]
    fn parses_f16_tensor() {
        // One f16 tensor (2 bytes per element) in a 4-byte-padded data region.
        let data_off = 32 + ENTRY_LEN;
        let bits: [u16; 3] = [0x3C00, 0xC000, 0x7BFF]; // 1.0, -2.0, 65504
        let mut data = Vec::new();
        for v in bits {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data.resize(8, 0);
        let file_len = data_off + data.len();
        let mut bytes = vec![0u8; file_len];
        bytes[..4].copy_from_slice(MAGIC);
        le16(&mut bytes, 4, FORMAT_VERSION);
        le16(&mut bytes, 6, 2);
        le16(&mut bytes, 8, 1);
        le32(&mut bytes, 12, 32);
        le32(&mut bytes, 16, data_off as u32);
        le32(&mut bytes, 20, data.len() as u32);
        le32(&mut bytes, 24, file_len as u32);
        le16(&mut bytes, 32, 9);
        bytes[34] = DTYPE_F16;
        bytes[35] = 1;
        le32(&mut bytes, 40, 3);
        le32(&mut bytes, 56, 0);
        le32(&mut bytes, 60, 6);
        bytes[data_off..].copy_from_slice(&data);

        let c = Container::parse(&bytes).unwrap();
        assert_eq!(dtype_size(DTYPE_F16), Some(2));
        assert_eq!(c.f16_bits(9).unwrap(), &bits);
        assert!(c.f32(9).is_none());
        assert!(c.i16(9).is_none());
    }
}
