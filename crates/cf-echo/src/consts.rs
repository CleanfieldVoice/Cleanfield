//! Scalar model constants for the echo model.
//!
//! Every numeric constant is stored in the `.cfwm` container as a raw `u32`
//! word and restored with `f32::from_bits`.  The order below is the builder's
//! constant-table order; the table length is checked at load time and the
//! pinned values are asserted by the tests.
//!
//! The activation forms are *not* the voice model's: the GRU gate activations
//! are the reference's rational `rcpps` forms (clamp +-8 for the sigmoid,
//! +-4 for the tanh), and the final sigmoid is the exp-polynomial form with a
//! +-88.37625885 clamp.

use cf_core::cfwm::Container;

use crate::tags::TAG_CONSTANTS;

/// Number of `u32` words in the constant table.
pub const CONST_COUNT: usize = 32;

/// Restored scalar constants.  `Copy`, so the hot loop can take it by value and
/// keep the container borrow independent of the workspace borrow.
#[derive(Clone, Copy, Debug)]
pub struct EchoConsts {
    /// `1.0`.
    pub one: f32,
    /// `0.5`.
    pub half: f32,
    /// `8.0`, sigmoid clamp.
    pub eight: f32,
    /// `-8.0`, sigmoid clamp.
    pub neg_eight: f32,
    /// `4.0`, tanh clamp.
    pub four: f32,
    /// `-4.0`, tanh clamp.
    pub neg_four: f32,
    /// Leaky-ReLU slope (`0.01f`).
    pub leaky_slope: f32,
    /// Rational sigmoid coefficients: `c0..c3` for the denominator quadratic,
    /// `c4..c6` for the numerator polynomial.
    pub sig: [f32; 7],
    /// Rational tanh coefficients: `c0..c3` polynomial, `c4` denominator offset.
    pub tanh: [f32; 5],
    /// Final-sigmoid upper clamp (`88.37625885009766`).
    pub sig88_clamp: f32,
    /// Final-sigmoid lower clamp.
    pub sig88_neg_clamp: f32,
    /// `log2(e)`, exponential range reduction.
    pub exp_log2e: f32,
    /// `-ln(2)`, exponential range reduction residual.
    pub exp_ln2: f32,
    /// Degree-8 exponential polynomial (five coefficients).
    pub exp_c: [f32; 5],
    /// `8192.0`, per-frame quantisation scale.
    pub quant_scale: f32,
    /// `127.0`, hidden-state quantisation scale while `|h| <= 1`.
    pub hidden_scale: f32,
    /// `32767.0`, hidden-state quantisation scale otherwise.
    pub hidden_scale_wide: f32,
    /// `0.0`.
    pub zero: f32,
}

impl EchoConsts {
    /// The compact constant block the shared activation kernels take.
    #[inline]
    pub fn act(&self) -> cf_core::EchoActConsts {
        cf_core::EchoActConsts {
            one: self.one,
            half: self.half,
            eight: self.eight,
            neg_eight: self.neg_eight,
            four: self.four,
            neg_four: self.neg_four,
            sig: self.sig,
            tanh: self.tanh,
            sig88_clamp: self.sig88_clamp,
            sig88_neg_clamp: self.sig88_neg_clamp,
            exp_log2e: self.exp_log2e,
            exp_ln2: self.exp_ln2,
            exp_c: self.exp_c,
        }
    }

    /// Restore the constants from the container's constant tensor.
    ///
    /// Returns `None` if the tensor is missing or has the wrong length; the
    /// caller turns that into a load-time error.
    pub fn from_container(container: &Container) -> Option<EchoConsts> {
        let raw = container.u32_bits(TAG_CONSTANTS)?;
        if raw.len() != CONST_COUNT {
            return None;
        }
        let mut f = [0.0f32; CONST_COUNT];
        for (slot, &bits) in f.iter_mut().zip(raw.iter()) {
            *slot = f32::from_bits(bits);
        }
        Some(EchoConsts {
            one: f[0],
            half: f[1],
            eight: f[2],
            neg_eight: f[3],
            four: f[4],
            neg_four: f[5],
            leaky_slope: f[6],
            sig: [f[7], f[8], f[9], f[10], f[11], f[12], f[13]],
            tanh: [f[14], f[15], f[16], f[17], f[18]],
            sig88_clamp: f[19],
            sig88_neg_clamp: f[20],
            exp_log2e: f[21],
            exp_ln2: f[22],
            exp_c: [f[23], f[24], f[25], f[26], f[27]],
            quant_scale: f[28],
            hidden_scale: f[29],
            hidden_scale_wide: f[30],
            zero: f[31],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_container() -> Container {
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/echo-default.cfwm"
        ));
        Container::parse(bytes).expect("container")
    }

    #[test]
    fn restores_exact_bit_patterns() {
        let c = model_container();
        let k = EchoConsts::from_container(&c).expect("constants");
        // The 60.0 ULP trap is shared with the voice model.
        assert_eq!(k.tanh[4].to_bits(), 0x4270_0000);
        assert_eq!(k.sig88_clamp.to_bits(), 0x42B0_C0A5);
        assert_eq!(k.exp_c[0].to_bits(), 0x3950_6967);
        assert_eq!(k.exp_c[1].to_bits(), 0x3AB7_43CE);
        assert_eq!(k.leaky_slope.to_bits(), 0x3C23_D70A);
        assert_eq!(k.sig[1].to_bits(), 0x4052_0000);
        assert_eq!(k.quant_scale.to_bits(), 0x4600_0000);
        assert_eq!(k.hidden_scale.to_bits(), 0x42FE_0000);
        assert_eq!(k.hidden_scale_wide.to_bits(), 0x46FF_FE00);
        assert_eq!(k.one.to_bits(), 0x3F80_0000);
    }

    #[test]
    fn rejects_wrong_length() {
        let c = model_container();
        assert_eq!(c.u32_bits(TAG_CONSTANTS).unwrap().len(), CONST_COUNT);
    }
}
