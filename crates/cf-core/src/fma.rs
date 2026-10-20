//! `f32` multiply-add primitives.
//!
//! The model requires three numerically distinct operations, and they must not
//! be confused:
//!
//! * [`fma`] - a *single rounding* fused multiply-add.  This is the operation
//!   the reference kernels use for 1x1 convolutions, depthwise taps and the
//!   recurrent cell update.
//! * [`mul_add2`] - a `f32` multiply followed by a `f32` add (two roundings).
//!   Only the first convolution and the input staging use this.
//! * [`mul_add2_nofma`] - same as [`mul_add2`], explicit non-fused form used
//!   where the compiler must not contract the two operations.
//!
//! # Software FMA
//!
//! Portability (notably `wasm32`) has no deterministic hardware FMA, so the
//! `soft-fma` feature compiles an integer-mantissa implementation.  It is a
//! correctly-rounded `f32` FMA built from:
//!
//! 1. a 24x24 -> 48-bit exact product of the significands,
//! 2. an exact exponent alignment of the product and the addend with a sticky
//!    bit for everything shifted out,
//! 3. one round-to-nearest-even step at the 24-bit significand.
//!
//! It is not a `f64` shortcut and it does not depend on the host FPU.  The
//! `fma_check` suite (enabled with the same feature) proves it agrees with the
//! native path over randomised and adversarial inputs.

/// The result of an operation is a single `f32`; only the rounding differs.
pub trait FmaOps: Copy {
    /// Short stable name, used by the verifier output only.
    const NAME: &'static str;
    /// `a * b + c` with a single rounding.
    fn fma(self, a: f32, b: f32, c: f32) -> f32;
}

/// Hardware (or platform libm) fused multiply-add.  Correctly rounded on all
/// targets where the compiler's `mul_add` is an FMA or a conforming `fmaf`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeFma;

impl FmaOps for NativeFma {
    const NAME: &'static str = "native";

    #[inline]
    fn fma(self, a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
}

#[cfg(feature = "soft-fma")]
pub use soft::{soft_fma, SoftFma};

#[cfg(feature = "soft-fma")]
mod soft;

/// Whether the compiled [`fma`] is the software implementation.
///
/// This is a compile-time fact, not a runtime probe: the wasm export layer
/// uses it to prove that the module did not silently keep a hardware FMA path.
#[inline]
pub const fn is_soft() -> bool {
    cfg!(feature = "force-soft-fma")
}

/// The engine's FMA.  Native unless `force-soft-fma` is enabled, in which case
/// the integer implementation is compiled into every call site.
#[inline]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(feature = "force-soft-fma")]
    {
        soft::soft_fma(a, b, c)
    }
    #[cfg(not(feature = "force-soft-fma"))]
    {
        // On x86-64 the scalar `vfmadd231ss` instruction is a single-rounding
        // FMA identical to `mul_add`/`fmaf`, but without the libm call.  The
        // feature probe is cached by std, so the branch is predictable.
        #[cfg(target_arch = "x86_64")]
        {
            if crate::scalar_kernels::fma_available() {
                // SAFETY: guarded by the runtime feature check above.
                return unsafe { crate::scalar_kernels::hw_fma(a, b, c) };
            }
        }
        NativeFma.fma(a, b, c)
    }
}

/// Plain `f32` multiply followed by a plain `f32` add: two roundings.
///
/// This must not be turned into an FMA by the compiler.  Rust does not enable
/// floating-point contraction, so separate statements are enough; the helper
/// keeps the intent in one place.
#[inline]
#[allow(clippy::eq_op)]
pub fn mul_add2(a: f32, b: f32, c: f32) -> f32 {
    let product = a * b;
    c + product
}

/// Explicitly non-fused multiply-add; alias kept for call-site readability.
#[inline]
pub fn mul_add2_nofma(a: f32, b: f32, c: f32) -> f32 {
    mul_add2(a, b, c)
}
