//! Voice-container tensor tags.
//!
//! Tags are a per-model namespace: the container parser only requires that a
//! tag is unique within one image.  The echo model defines its own table in
//! `cf-echo`.

/// Input staging scale (513 f32).
pub const TAG_STAGE_SCALE: u16 = 1;
/// Input staging bias (513 f32).
pub const TAG_STAGE_BIAS: u16 = 2;

// --- convolution stack -----------------------------------------------------
/// First convolution taps (9 f32, `[kh][kw]`).
pub const TAG_L0_W: u16 = 3;
/// First 1x1 convolution kernel (32 f32, `[oc]`, one input channel).
pub const TAG_L1_W: u16 = 4;
/// First 1x1 convolution bias (32 f32).
pub const TAG_L1_B: u16 = 5;
/// Depthwise 3x3 kernel (288 f32, `[kh][kw][c]`).
pub const TAG_L4_W: u16 = 6;
/// Layer-5 1x1 kernel (1024 f32, `[ic][oc]`).
pub const TAG_L5_W: u16 = 7;
/// Layer-5 bias (32 f32).
pub const TAG_L5_B: u16 = 8;
/// Layer-8 depthwise 1x3 kernel (96 f32, `[tap][c]`).
pub const TAG_L8_W: u16 = 9;
/// Layer-9 1x1 kernel (1024 f32, `[ic][oc]`).
pub const TAG_L9_W: u16 = 10;
/// Layer-9 bias (32 f32).
pub const TAG_L9_B: u16 = 11;
/// Layer-11 depthwise 1x3 kernel (96 f32, `[tap][c]`).
pub const TAG_L11_W: u16 = 12;
/// Layer-12 1x1 kernel (1024 f32, `[ic][oc]`).
pub const TAG_L12_W: u16 = 13;
/// Layer-12 bias (32 f32).
pub const TAG_L12_B: u16 = 14;

// --- recurrent layers ------------------------------------------------------
/// Layer-15 input weights (1048576 i8, input-major `[k][j]`).
pub const TAG_L15_W_IN: u16 = 15;
/// Layer-15 input zero points (1024 i8).
pub const TAG_L15_ZP_IN: u16 = 16;
/// Layer-15 recurrent weights (262144 i8, input-major `[k][j]`).
pub const TAG_L15_W_REC: u16 = 17;
/// Layer-15 recurrent zero points (1024 i8).
pub const TAG_L15_ZP_REC: u16 = 18;
/// Layer-15 bias (1024 f32, `b_ih + b_hh` at build time).
pub const TAG_L15_BIAS: u16 = 19;
/// Layer-15 input per-channel scale (1024 f32, `sc_in * S_in`).
pub const TAG_L15_SP_IN: u16 = 20;
/// Layer-15 recurrent per-channel scale (1024 f32, `sc_rec * S_h`).
pub const TAG_L15_SP_REC: u16 = 21;
/// Layer-16 input weights (262144 i8, input-major `[k][j]`).
pub const TAG_L16_W_IN: u16 = 22;
/// Layer-16 input zero points (1024 i8).
pub const TAG_L16_ZP_IN: u16 = 23;
/// Layer-16 recurrent weights (262144 i8, input-major `[k][j]`).
pub const TAG_L16_W_REC: u16 = 24;
/// Layer-16 recurrent zero points (1024 i8).
pub const TAG_L16_ZP_REC: u16 = 25;
/// Layer-16 bias (1024 f32).
pub const TAG_L16_BIAS: u16 = 26;
/// Layer-16 input per-channel scale (1024 f32).
pub const TAG_L16_SP_IN: u16 = 27;
/// Layer-16 recurrent per-channel scale (1024 f32).
pub const TAG_L16_SP_REC: u16 = 28;

// --- output projection -----------------------------------------------------
/// Layer-17 int8 weights (131072 i8, `[k][j]`).
pub const TAG_L17_W: u16 = 29;
/// Layer-17 zero points (512 i8).
pub const TAG_L17_ZP: u16 = 30;
/// Layer-17 bias (512 f32).
pub const TAG_L17_BIAS: u16 = 31;
/// Layer-17 per-channel scale (512 f32, `A * wscale`).
pub const TAG_L17_SP: u16 = 32;
/// Layer-18 int8 weights (525312 i8, `[k][j]`).
pub const TAG_L18_W: u16 = 33;
/// Layer-18 zero points (1026 i8).
pub const TAG_L18_ZP: u16 = 34;
/// Layer-18 bias (1026 f32).
pub const TAG_L18_BIAS: u16 = 35;
/// Layer-18 per-channel scale (1026 f32).
pub const TAG_L18_SP: u16 = 36;
/// Scalar model constants as `u32` bit patterns.
pub const TAG_CONSTANTS: u16 = 37;
