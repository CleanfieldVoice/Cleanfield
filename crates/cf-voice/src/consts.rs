//! Scalar model constants.
//!
//! Every numeric constant the kernels need is stored in the `.cfwm` container
//! as a raw `u32` word and restored here with `f32::from_bits` (or as an
//! integer).  Nothing in the inference path spells a model constant as a
//! decimal float literal: a one-ULP transcription error of exactly that kind
//! is what made an earlier engine disagree on a rounding tie.
//!
//! The order of the words is the build-time constant table; it is verified
//! against the container length at load time and pinned by the tests below.

use cf_core::ActConsts;

use crate::cfwm::Container;
use crate::tags::TAG_CONSTANTS;

/// Number of `u32` words in the constant table.
pub const CONST_COUNT: usize = 32;
/// Number of leading words that are `f32` bit patterns.
const F32_COUNT: usize = 30;

// Indices into the constant table (build order).
const C_LOG2E: usize = 6;
const C_LN2: usize = 7;
const C_EXP_C0: usize = 8;
const C_EXP_C1: usize = 9;
const C_EXP_C2: usize = 10;
const C_EXP_C3: usize = 11;
const C_EXP_C4: usize = 12;
const C_HALF: usize = 13;
const C_ONE: usize = 14;
const C_TWO: usize = 15;
const C_FOUR: usize = 16;
const C_NEG_FOUR: usize = 17;
const C_EXP_CLAMP: usize = 18;
const C_NEG_EXP_CLAMP: usize = 19;
const C_CELL_C0: usize = 20;
const C_CELL_C1: usize = 21;
const C_CELL_C2: usize = 22;
const C_CELL_C3: usize = 23;
const C_CELL_C4: usize = 24;
const C_INV_IN15: usize = 25;
const C_INV_IN16: usize = 26;
const C_INV_H15: usize = 27;
const C_INV_H16: usize = 28;
const C_INV_MM18: usize = 29;
const C_CLAMP_LSTM: usize = 30;
const C_CLAMP_MM: usize = 31;

/// Restored scalar constants.  `Copy` so the hot loop can take it by value and
/// keep the container borrow independent of the workspace borrow.
#[derive(Clone, Copy)]
pub struct Consts {
    f: [f32; F32_COUNT],
    /// Quantisation clamp for the recurrent layers (`int16`).
    pub clamp_lstm: i32,
    /// Quantisation clamp for the output projections (`int16`).
    pub clamp_mm: i32,
}

impl Consts {
    /// Restore the constants from the container's constant tensor.
    ///
    /// Returns `None` if the tensor is missing or has the wrong length; the
    /// caller turns that into a load-time error.
    pub fn from_container(container: &Container) -> Option<Consts> {
        let raw = container.u32_bits(TAG_CONSTANTS)?;
        if raw.len() != CONST_COUNT {
            return None;
        }
        let mut f = [0.0f32; F32_COUNT];
        for (slot, &bits) in f.iter_mut().zip(raw.iter()) {
            *slot = f32::from_bits(bits);
        }
        Some(Consts {
            f,
            clamp_lstm: raw[C_CLAMP_LSTM] as i32,
            clamp_mm: raw[C_CLAMP_MM] as i32,
        })
    }

    /// Base-2 exponent multiplier for the exponential range reduction.
    #[inline]
    pub fn log2e(&self) -> f32 {
        self.f[C_LOG2E]
    }
    /// Negative natural log of two, for the range reduction residual.
    #[inline]
    pub fn ln2(&self) -> f32 {
        self.f[C_LN2]
    }
    #[inline]
    pub fn exp_c0(&self) -> f32 {
        self.f[C_EXP_C0]
    }
    #[inline]
    pub fn exp_c1(&self) -> f32 {
        self.f[C_EXP_C1]
    }
    #[inline]
    pub fn exp_c2(&self) -> f32 {
        self.f[C_EXP_C2]
    }
    #[inline]
    pub fn exp_c3(&self) -> f32 {
        self.f[C_EXP_C3]
    }
    #[inline]
    pub fn exp_c4(&self) -> f32 {
        self.f[C_EXP_C4]
    }
    #[inline]
    pub fn half(&self) -> f32 {
        self.f[C_HALF]
    }
    #[inline]
    pub fn one(&self) -> f32 {
        self.f[C_ONE]
    }
    #[inline]
    pub fn two(&self) -> f32 {
        self.f[C_TWO]
    }
    #[inline]
    pub fn four(&self) -> f32 {
        self.f[C_FOUR]
    }
    #[inline]
    pub fn neg_four(&self) -> f32 {
        self.f[C_NEG_FOUR]
    }
    #[inline]
    pub fn exp_clamp(&self) -> f32 {
        self.f[C_EXP_CLAMP]
    }
    #[inline]
    pub fn neg_exp_clamp(&self) -> f32 {
        self.f[C_NEG_EXP_CLAMP]
    }
    #[inline]
    pub fn cell_c0(&self) -> f32 {
        self.f[C_CELL_C0]
    }
    #[inline]
    pub fn cell_c1(&self) -> f32 {
        self.f[C_CELL_C1]
    }
    #[inline]
    pub fn cell_c2(&self) -> f32 {
        self.f[C_CELL_C2]
    }
    #[inline]
    pub fn cell_c3(&self) -> f32 {
        self.f[C_CELL_C3]
    }
    #[inline]
    pub fn cell_c4(&self) -> f32 {
        self.f[C_CELL_C4]
    }
    /// `1 / S_in` for the first recurrent layer.
    #[inline]
    pub fn inv_in15(&self) -> f32 {
        self.f[C_INV_IN15]
    }
    /// `1 / S_in` for the second recurrent layer (unused: its input is int16).
    #[inline]
    pub fn inv_in16(&self) -> f32 {
        self.f[C_INV_IN16]
    }
    /// `1 / S_h` for the first recurrent layer.
    #[inline]
    pub fn inv_h15(&self) -> f32 {
        self.f[C_INV_H15]
    }
    /// `1 / S_h` for the second recurrent layer.
    #[inline]
    pub fn inv_h16(&self) -> f32 {
        self.f[C_INV_H16]
    }
    /// `1 / A` for the final input quantiser.
    #[inline]
    pub fn inv_mm18(&self) -> f32 {
        self.f[C_INV_MM18]
    }

    /// Bundle of the constants the activation kernels need.
    ///
    /// Everything is copied by value; the backend never sees the container.
    #[inline]
    pub fn act(&self) -> ActConsts {
        ActConsts {
            log2e: self.f[C_LOG2E],
            ln2: self.f[C_LN2],
            exp_poly: [
                self.f[C_EXP_C0],
                self.f[C_EXP_C1],
                self.f[C_EXP_C2],
                self.f[C_EXP_C3],
                self.f[C_EXP_C4],
            ],
            half: self.f[C_HALF],
            one: self.f[C_ONE],
            two: self.f[C_TWO],
            four: self.f[C_FOUR],
            neg_four: self.f[C_NEG_FOUR],
            exp_clamp: self.f[C_EXP_CLAMP],
            neg_exp_clamp: self.f[C_NEG_EXP_CLAMP],
            cell_poly: [
                self.f[C_CELL_C0],
                self.f[C_CELL_C1],
                self.f[C_CELL_C2],
                self.f[C_CELL_C3],
                self.f[C_CELL_C4],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_container() -> Container {
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/voice-default.cfwm"
        ));
        Container::parse(bytes).expect("container")
    }

    #[test]
    fn restores_exact_bit_patterns() {
        let c = model_container();
        let k = Consts::from_container(&c).expect("constants");
        // The 60.0 ULP regression: the correct value is exactly 0x42700000.
        assert_eq!(k.cell_c4().to_bits(), 0x4270_0000);
        assert_eq!(k.cell_c3().to_bits(), 0x43F7_8000);
        assert_eq!(k.exp_c0().to_bits(), 0x3950_6967);
        assert_eq!(k.exp_c1().to_bits(), 0x3AB7_43CE);
        assert_eq!(k.exp_clamp().to_bits(), 0x42B0_C0A5);
        assert_eq!(k.inv_in15().to_bits(), 0x4588_E5FC);
        assert_eq!(k.inv_h15().to_bits(), 0x4600_0000);
        assert_eq!(k.inv_mm18().to_bits(), 0x4503_8C24);
        assert_eq!(k.one().to_bits(), 0x3F80_0000);
        assert_eq!(k.clamp_lstm, 8192);
        assert_eq!(k.clamp_mm, 16384);
    }

    #[test]
    fn rejects_wrong_length() {
        let c = model_container();
        assert_eq!(c.u32_bits(TAG_CONSTANTS).unwrap().len(), CONST_COUNT);
    }
}
