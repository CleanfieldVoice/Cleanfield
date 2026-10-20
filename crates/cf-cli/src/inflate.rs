//! Minimal raw DEFLATE (RFC 1951) decoder.
//!
//! The reference dumps are `.npz` archives, i.e. ZIP files using raw DEFLATE
//! members.  Rather than pull in a dependency, the CLI carries this small
//! decoder; it is a build/verify-time component and never runs in inference.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InflateError {
    Truncated,
    BadBlockType,
    BadStoredLength,
    BadCode,
    BadDistance,
    BadLengths,
    OverSubscribed,
}

impl fmt::Display for InflateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for InflateError {}

struct BitReader<'a> {
    src: &'a [u8],
    pos: usize,
    bits: u32,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    fn new(src: &'a [u8]) -> BitReader<'a> {
        BitReader {
            src,
            pos: 0,
            bits: 0,
            nbits: 0,
        }
    }

    #[inline]
    fn need(&mut self, n: u32) -> Result<u32, InflateError> {
        if n == 0 {
            return Ok(0);
        }
        debug_assert!(n <= 16);
        let mut val = self.bits;
        while self.nbits < n {
            let b = *self.src.get(self.pos).ok_or(InflateError::Truncated)?;
            self.pos += 1;
            val |= (b as u32) << self.nbits;
            self.nbits += 8;
        }
        self.bits = val >> n;
        self.nbits -= n;
        Ok(val & ((1u32 << n) - 1))
    }

    #[inline]
    fn align(&mut self) {
        self.bits = 0;
        self.nbits = 0;
    }
}

struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Huffman, InflateError> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if l > 15 {
                return Err(InflateError::BadLengths);
            }
            counts[l as usize] += 1;
        }
        counts[0] = 0;

        let mut left: i32 = 1;
        for len in 1..16 {
            left <<= 1;
            left -= counts[len] as i32;
            if left < 0 {
                return Err(InflateError::OverSubscribed);
            }
        }

        let mut offsets = [0u16; 16];
        for len in 1..15 {
            offsets[len + 1] = offsets[len] + counts[len];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offsets[l as usize] as usize] = sym as u16;
                offsets[l as usize] += 1;
            }
        }
        Ok(Huffman { counts, symbols })
    }

    #[inline]
    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16, InflateError> {
        let mut code = 0u32;
        let mut first = 0u32;
        let mut index = 0u32;
        for len in 1..=15u32 {
            code |= br.need(1)?;
            let count = self.counts[len as usize] as u32;
            if code >= first && code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(InflateError::BadCode)
    }
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

fn fixed_literal() -> Huffman {
    let mut lengths = [0u8; 288];
    lengths[..144].fill(8);
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..].fill(8);
    Huffman::new(&lengths).expect("fixed literal tree")
}

fn fixed_distance() -> Huffman {
    let lengths = [5u8; 32];
    Huffman::new(&lengths).expect("fixed distance tree")
}

/// Decompress a raw DEFLATE stream.
pub fn inflate(src: &[u8]) -> Result<Vec<u8>, InflateError> {
    let mut br = BitReader::new(src);
    let mut out = Vec::new();
    loop {
        let last = br.need(1)?;
        let kind = br.need(2)?;
        match kind {
            0 => stored(&mut br, &mut out)?,
            1 => {
                let lit = fixed_literal();
                let dist = fixed_distance();
                block(&mut br, &mut out, &lit, &dist)?;
            }
            2 => {
                let (lit, dist) = dynamic_tables(&mut br)?;
                block(&mut br, &mut out, &lit, &dist)?;
            }
            _ => return Err(InflateError::BadBlockType),
        }
        if last == 1 {
            break;
        }
    }
    Ok(out)
}

fn stored(br: &mut BitReader<'_>, out: &mut Vec<u8>) -> Result<(), InflateError> {
    br.align();
    let len = br.need(16)? as usize;
    let nlen = br.need(16)? as usize;
    if len != (!nlen & 0xFFFF) {
        return Err(InflateError::BadStoredLength);
    }
    out.reserve(len);
    for _ in 0..len {
        out.push(br.need(8)? as u8);
    }
    Ok(())
}

fn block(
    br: &mut BitReader<'_>,
    out: &mut Vec<u8>,
    lit: &Huffman,
    dist: &Huffman,
) -> Result<(), InflateError> {
    loop {
        let sym = lit.decode(br)?;
        match sym {
            0..=255 => out.push(sym as u8),
            256 => return Ok(()),
            257..=285 => {
                let idx = (sym - 257) as usize;
                let len = LENGTH_BASE[idx] as usize + br.need(LENGTH_EXTRA[idx] as u32)? as usize;
                let dsym = dist.decode(br)? as usize;
                if dsym >= 30 {
                    return Err(InflateError::BadDistance);
                }
                let distance =
                    DIST_BASE[dsym] as usize + br.need(DIST_EXTRA[dsym] as u32)? as usize;
                if distance > out.len() {
                    return Err(InflateError::BadDistance);
                }
                let start = out.len() - distance;
                for i in 0..len {
                    let b = out[start + i];
                    out.push(b);
                }
            }
            _ => return Err(InflateError::BadCode),
        }
    }
}

fn dynamic_tables(br: &mut BitReader<'_>) -> Result<(Huffman, Huffman), InflateError> {
    let hlit = br.need(5)? as usize + 257;
    let hdist = br.need(5)? as usize + 1;
    let hclen = br.need(4)? as usize + 4;
    if hlit > 286 || hdist > 32 {
        return Err(InflateError::BadLengths);
    }
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut cl_lengths = [0u8; 19];
    for &slot in ORDER.iter().take(hclen) {
        cl_lengths[slot] = br.need(3)? as u8;
    }
    let cl = Huffman::new(&cl_lengths)?;

    let mut lengths = vec![0u8; hlit + hdist];
    let mut i = 0usize;
    while i < lengths.len() {
        let sym = cl.decode(br)?;
        match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(InflateError::BadLengths);
                }
                let prev = lengths[i - 1];
                let rep = 3 + br.need(2)? as usize;
                if i + rep > lengths.len() {
                    return Err(InflateError::BadLengths);
                }
                lengths[i..i + rep].fill(prev);
                i += rep;
            }
            17 => {
                let rep = 3 + br.need(3)? as usize;
                if i + rep > lengths.len() {
                    return Err(InflateError::BadLengths);
                }
                i += rep;
            }
            18 => {
                let rep = 11 + br.need(7)? as usize;
                if i + rep > lengths.len() {
                    return Err(InflateError::BadLengths);
                }
                i += rep;
            }
            _ => return Err(InflateError::BadCode),
        }
    }
    if lengths[256] == 0 {
        return Err(InflateError::BadLengths);
    }
    let lit = Huffman::new(&lengths[..hlit])?;
    let dist = Huffman::new(&lengths[hlit..])?;
    Ok((lit, dist))
}

#[cfg(test)]
mod tests {
    use super::inflate;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn fixed_and_dynamic_vectors() {
        let cases: [(&str, Vec<u8>); 4] = [
            (
                "cb48cdc9c957c8cc4bcb492c49553030343236313533b7b00400",
                b"hello inflate 0123456789".to_vec(),
            ),
            (
                "73741c05a36014380e730000",
                vec![b'A'; 1000],
            ),
            (
                "6360646266616563e7e0e4e2e6e1e5e3171014121611151397909492969195935750545256515553d7d0d4d2d6d1d5d33730343236313533b7b0b4b2b6b1b5b37770747276717573f7f0f4f2f6f1f5f30f080c0a0e090d0b8f888c8a8e898d8b4f484c4a4e494d4bcfc8cccacec9cdcb2f282c2a2e292d2bafa8acaaaea9adab6f686c6a6e696d6befe8eceaeee9edeb9f3071d2e42953a74d9f3173d6ec3973e7cd5fb070d1e2254b972d5fb172d5ea356bd7addfb071d3e62d5bb76ddfb173d7ee3d7bf7ed3f70f0d0e123478f1d3f71f2d4e93367cf9dbf70f1d2e52b57af5dbf71f3d6ed3b77efdd7ff0f0d1e3274f9f3d7ff1f2d5eb376fdfbdfff0f1d3e72f5fbf7dfff1f3d7ef3f7ffffd6718f5ffa8ff5947aeff01",
                (0..=255u8).cycle().take(1024).collect(),
            ),
            (
                "0bc94855282ccd4cce56482aca2fcf5348cbaf50c82acd2d2856c82f4b2d5228c94855c849acaa5448c94fd7530819553caa7854f1a8e251c525288a01",
                b"The quick brown fox jumps over the lazy dog. ".repeat(30),
            ),
        ];
        for (hex, want) in cases {
            let got = inflate(&unhex(hex)).unwrap();
            assert_eq!(got, want);
        }
    }
}
