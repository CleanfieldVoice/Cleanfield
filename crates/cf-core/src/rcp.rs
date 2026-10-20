//! Portable reciprocal model used by the gate activations.
//!
//! The reference implementation returns a 12-bit-significand approximation of
//! `1/x` from a 4096-entry table indexed by the top 12 mantissa bits; the low
//! 11 mantissa bits are ignored and the exponent is an exact power-of-two
//! scale.  The table is *data* (the hardware ROM values are a coarse
//! approximation, not a rounding of `1/m`), so it is compiled in as a constant
//! and pinned by hash in `tests`.
//!
//! This module implements the integer bit formula exactly, including the
//! special-case branches (zero/subnormal inputs, infinities, NaN quieting and
//! subnormal results).

/// The captured 4096-entry table: `K[j] = 4096 + (12-bit significand)`.
pub use crate::rcp_table::RCP_TABLE;

/// The same table widened to `u32`, for 32-bit gather instructions.
///
/// The AVX2 backend uses `vpgatherdd`, which can only gather 32-bit elements;
/// this compile-time copy keeps the scalar and vector paths on the exact same
/// data with no run-time conversion.
pub static RCP_TABLE_U32: [u32; 4096] = widen_table(RCP_TABLE);

const fn widen_table(src: [u16; 4096]) -> [u32; 4096] {
    let mut out = [0u32; 4096];
    let mut i = 0;
    while i < 4096 {
        out[i] = src[i] as u32;
        i += 1;
    }
    out
}

/// Apply the reciprocal model to the bit pattern `u`.
#[inline]
pub fn rcp_ps_bits(u: u32) -> u32 {
    let sign = u & 0x8000_0000;
    let exp = (u >> 23) & 0xFF;
    let mant = u & 0x007F_FFFF;

    if exp == 0 {
        // +-0 and subnormal inputs are flushed; the reciprocal is +-inf.
        return sign | 0x7F80_0000;
    }
    if exp == 255 {
        if mant == 0 {
            // +-inf -> +-0.
            return sign;
        }
        // NaN: quiet it, keep the payload, force the quiet bit.
        return sign | 0x7F80_0000 | mant | 0x0040_0000;
    }
    if exp >= 253 {
        // Result would be subnormal: flushed to +-0.
        return sign;
    }
    let j = ((u >> 11) & 0xFFF) as usize;
    let k = RCP_TABLE[j] as u32;
    sign | ((253 - exp) << 23) | ((k - 4096) << 11)
}

/// Apply the reciprocal model to a float.
#[inline]
pub fn rcp_ps(x: f32) -> f32 {
    f32::from_bits(rcp_ps_bits(x.to_bits()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256;

    #[test]
    fn table_shape_and_range() {
        assert_eq!(RCP_TABLE.len(), 4096);
        assert_eq!(RCP_TABLE[0], 8190);
        assert_eq!(RCP_TABLE[4095], 4096);
        for w in RCP_TABLE.windows(2) {
            assert!(w[0] >= w[1], "table must be monotone non-increasing");
            assert!((4096..=8191).contains(&w[0]));
        }
        let distinct = {
            let mut v = RCP_TABLE.to_vec();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        assert_eq!(distinct, 3382);
    }

    #[test]
    fn special_cases() {
        // Zero / subnormal input -> signed infinity.
        assert_eq!(rcp_ps_bits(0x0000_0000), 0x7F80_0000);
        assert_eq!(rcp_ps_bits(0x8000_0000), 0xFF80_0000);
        assert_eq!(rcp_ps_bits(0x0000_0001), 0x7F80_0000);
        // Infinity -> signed zero.
        assert_eq!(rcp_ps_bits(0x7F80_0000), 0x0000_0000);
        assert_eq!(rcp_ps_bits(0xFF80_0000), 0x8000_0000);
        // NaN: quiet bit forced, payload preserved.
        assert_eq!(rcp_ps_bits(0x7F80_0001), 0x7FC0_0001);
        assert_eq!(rcp_ps_bits(0x7FBF_FFFF), 0x7FFF_FFFF);
        assert_eq!(rcp_ps_bits(0xFF80_0001), 0xFFC0_0001);
        // Subnormal result (x >= 2^126) is flushed.
        assert_eq!(rcp_ps_bits(0x7E80_0000), 0x0000_0000);
        assert_eq!(rcp_ps_bits(0x7E7F_FFFF), 0x0080_0000);
        // Normal reference points from the specification.
        assert_eq!(rcp_ps_bits(0x3F80_0000), 0x3F7F_F000);
    }

    #[test]
    fn result_is_close_to_reciprocal() {
        // Structural check only: the table is a coarse (12-bit) approximation,
        // so the relative error is bounded by 1.125 * 2^-12.
        for e in 1..=252u32 {
            for j in [0usize, 1, 2, 123, 2048, 4094, 4095] {
                let u = (e << 23) | ((j as u32) << 11);
                let r = rcp_ps_bits(u);
                assert_eq!((r >> 31) & 1, 0);
                assert_eq!((r >> 23) & 0xFF, 253 - e);
                let k = ((r >> 11) & 0xFFF) + 4096;
                assert!((4096..=8190).contains(&k));
            }
        }
    }

    #[test]
    fn matches_reference_table_file() {
        // The table is pinned by sha256 and compared value by value against
        // the captured .npy.  The test reads the source artifact rather than a
        // generated copy, so a bad conversion cannot hide.
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/rcpps_table.npy");
        let data = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("cannot read reference table {}: {e}", path.display()));
        let digest = sha256::digest(&data);
        assert_eq!(
            sha256::hex(&digest),
            "2e3bd841cd5cc51da94c416c8806dfd954b9018564acdee35a8eb77091f07683",
            "reference table hash changed"
        );
        assert_eq!(&data[..6], b"\x93NUMPY");
        let (header_len, data_off) = if data[6] == 1 {
            let n = u16::from_le_bytes([data[8], data[9]]) as usize;
            (n, 10 + n)
        } else {
            let n = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
            (n, 12 + n)
        };
        let header = std::str::from_utf8(&data[data_off - header_len..data_off]).unwrap();
        assert!(header.contains("'<u2'"), "unexpected dtype: {header}");
        assert!(header.contains("(4096,)"), "unexpected shape: {header}");
        let payload = &data[data_off..data_off + 4096 * 2];
        for (i, chunk) in payload.as_chunks::<2>().0.iter().enumerate() {
            let v = u16::from_le_bytes([chunk[0], chunk[1]]);
            assert_eq!(v, RCP_TABLE[i], "table mismatch at index {i}");
        }
    }
}
