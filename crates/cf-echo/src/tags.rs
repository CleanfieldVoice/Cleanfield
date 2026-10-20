//! Echo-container tensor tags.
//!
//! Tags are a per-model namespace.  The builder assigns them in this order:
//!
//! ```text
//!   1..3     staging scale / bias / history initializer (3 x 513 f32)
//!   4        scalar constants (u32 bit patterns)
//!   5..40    conv1x1 weights and biases (18 layers x 2)
//!  41..60    depthwise weights and biases (10 layers x 2)
//!  61..62    33x33 MatMul weights
//!  63..76    GRU tensors (2 layers x 7)
//!  77..88    int8 projection tensors (3 layers x 4)
//!  89..96    depthwise deconvolution (4 layers x 2)
//!  97..104   3x1 decoder convolutions (4 layers x 2)
//! ```

/// Input staging scale (3 x 513 f32).
pub const TAG_STAGE_SCALE: u16 = 1;
/// Input staging bias (3 x 513 f32).
pub const TAG_STAGE_BIAS: u16 = 2;
/// History initializer (3 x 513 f32), copied into all six rows at reset.
pub const TAG_HISTORY_INIT: u16 = 3;
/// Scalar constants as `u32` bit patterns.
pub const TAG_CONSTANTS: u16 = 4;

/// First conv1x1 tag (`[ic][oc]` weight, then bias).
pub const CONV1_BASE: u16 = 5;
/// First depthwise tag (`[c][tap]` weight, then bias).
pub const DEPTHWISE_BASE: u16 = 41;
/// 33x33 MatMul weight A.
pub const TAG_MM33_A: u16 = 61;
/// 33x33 MatMul weight B.
pub const TAG_MM33_B: u16 = 62;
/// First GRU tag (`w_in`, `zp_in`, `scale_in`, `w_rec`, `zp_rec`, `scale_rec`, `bias`).
pub const GRU_BASE: u16 = 63;
/// First projection tag (`weight`, `zp`, `scale`, `bias`).
pub const GEMM_BASE: u16 = 77;
/// First deconvolution tag (`weight`, `bias`).
pub const DECONV_BASE: u16 = 89;
/// First 3x1 decoder convolution tag (`weight`, `bias`).
pub const CONV3_BASE: u16 = 97;
/// Total tags in the default echo container.
pub const TAG_COUNT: u16 = 104;

/// `(weight, bias)` tags of one conv1x1 layer.
#[inline]
pub const fn conv1_tags(index: usize) -> (u16, u16) {
    let w = CONV1_BASE + 2 * index as u16;
    (w, w + 1)
}

/// `(weight, bias)` tags of one depthwise layer.
#[inline]
pub const fn depthwise_tags(index: usize) -> (u16, u16) {
    let w = DEPTHWISE_BASE + 2 * index as u16;
    (w, w + 1)
}

/// One GRU role.
pub mod gru_role {
    /// Input weights `[k][3H]` (i8).
    pub const W_IN: u16 = 0;
    /// Input zero points (i8).
    pub const ZP_IN: u16 = 1;
    /// Input per-channel scale (f32).
    pub const SCALE_IN: u16 = 2;
    /// Recurrent weights `[h][3H]` (i8).
    pub const W_REC: u16 = 3;
    /// Recurrent zero points (i8).
    pub const ZP_REC: u16 = 4;
    /// Recurrent per-channel scale (f32).
    pub const SCALE_REC: u16 = 5;
    /// `b_ih || b_hh` (f32, 768).
    pub const BIAS: u16 = 6;
}

/// Tag of one GRU tensor.
#[inline]
pub const fn gru_tag(group: usize, role: u16) -> u16 {
    GRU_BASE + 7 * group as u16 + role
}

/// One projection role.
pub mod gemm_role {
    /// Weights `[k][n]` (i8).
    pub const WEIGHT: u16 = 0;
    /// Zero points (i8).
    pub const ZP: u16 = 1;
    /// Per-channel scale (f32).
    pub const SCALE: u16 = 2;
    /// Bias (f32).
    pub const BIAS: u16 = 3;
}

/// Tag of one projection tensor.
#[inline]
pub const fn gemm_tag(group: usize, role: u16) -> u16 {
    GEMM_BASE + 4 * group as u16 + role
}

/// `(weight, bias)` tags of one deconvolution layer.
#[inline]
pub const fn deconv_tags(index: usize) -> (u16, u16) {
    let w = DECONV_BASE + 2 * index as u16;
    (w, w + 1)
}

/// `(weight, bias)` tags of one 3x1 decoder convolution.
#[inline]
pub const fn conv3_tags(index: usize) -> (u16, u16) {
    let w = CONV3_BASE + 2 * index as u16;
    (w, w + 1)
}
