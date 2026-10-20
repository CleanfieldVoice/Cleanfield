//! Voice model: input staging, the convolutional stack, the two recurrent
//! layers and the two output projections.
//!
//! Numeric contract (must not be "cleaned up"):
//!
//! * staging: `f32(f32(x[k] * A[k]) + B[k])` - two roundings, not an FMA;
//! * layer 0: 3x3 convolution, row-major taps `(kh, kw)`, multiply then add;
//! * layers 1/5/9/12: 1x1 convolution with the bias pre-filling the
//!   accumulator, input channels in ascending order, single-rounding FMA,
//!   fused ReLU;
//! * layers 2/6/10/13: identity (the ReLU is fused into the preceding 1x1);
//! * layers 3/7: max pool, kernel 3, stride 2 along the length axis;
//! * layer 4: depthwise 3x3, `[c][kh][kw]` tap order, FMA;
//! * layers 8/11: depthwise 1x3, stride 2, left padding 0 / 1, FMA;
//! * layer 14: reshape (identity on the same buffer);
//! * layers 15/16: int8 quantised gate GEMMs (input and recurrent), zero
//!   points subtracted in the integer domain, `f32(acc) * sp + bias`, gate
//!   order `[i, o, f, g]`, Padé/exp activations with the reciprocal table,
//!   `c_new = fma(g, i, f32(f * c))`, `h = f32(o * tanh_cell(c_new))`,
//!   `qh = clamp(rne(h * 1/S_h))` as int16; state persists across frames;
//! * layer 17/18: int8 MatMul, int16 input quantisation with round-to-nearest
//!   even, `f32(acc) * sp + bias`, optional ReLU, output clamp is implicit in
//!   the int16 quantiser.
//!
//! The five-row input history is written before the forward pass and shifted
//! after it, so frame `f` sees `[P(f-4) .. P(f)]` with `P(i<0) = B`.
//!
//! The scalar path is the reference implementation.  The AVX2 backend is the
//! same graph with wider kernels from [`cf_core::Simd`]; it is selected only
//! when the CPU reports AVX2 and FMA, and `simd-check` compares every
//! intermediate buffer of the two backends bit for bit.

use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use cf_core::Avx2Simd;
use cf_core::{ActConsts, Arena, Backend, ScalarSimd, Simd, Workspace};

use crate::cfwm::{CfwmError, Container};
use crate::consts::{Consts, CONST_COUNT};
use crate::tags::{
    TAG_CONSTANTS, TAG_L0_W, TAG_L11_W, TAG_L12_B, TAG_L12_W, TAG_L15_BIAS, TAG_L15_SP_IN,
    TAG_L15_SP_REC, TAG_L15_W_IN, TAG_L15_W_REC, TAG_L15_ZP_IN, TAG_L15_ZP_REC, TAG_L16_BIAS,
    TAG_L16_SP_IN, TAG_L16_SP_REC, TAG_L16_W_IN, TAG_L16_W_REC, TAG_L16_ZP_IN, TAG_L16_ZP_REC,
    TAG_L17_BIAS, TAG_L17_SP, TAG_L17_W, TAG_L17_ZP, TAG_L18_BIAS, TAG_L18_SP, TAG_L18_W,
    TAG_L18_ZP, TAG_L1_B, TAG_L1_W, TAG_L4_W, TAG_L5_B, TAG_L5_W, TAG_L8_W, TAG_L9_B, TAG_L9_W,
    TAG_STAGE_BIAS, TAG_STAGE_SCALE,
};

/// Stable model id for the default voice model.
pub const MODEL_VOICE_DEFAULT: u16 = 1;

/// Input feature length.
pub const INPUT_LEN: usize = 513;
/// Number of history rows.
pub const HISTORY_ROWS: usize = 5;
/// Staging history buffer length.
pub const HISTORY_LEN: usize = HISTORY_ROWS * INPUT_LEN;
/// Layer 0 output length: 3 x 511.
pub const L0_LEN: usize = 3 * 511;
/// Layers 1/2 output length: 3 x 511 x 32.
pub const L1_LEN: usize = L0_LEN * 32;
/// Layer 3 output length: 3 x 255 x 32.
pub const L3_LEN: usize = 3 * 255 * 32;
/// Layer 4 output length: 253 x 32.
pub const L4_LEN: usize = 253 * 32;
/// Layers 5/6 output length: 253 x 32.
pub const L5_LEN: usize = 253 * 32;
/// Layer 7 output length: 126 x 32.
pub const L7_LEN: usize = 126 * 32;
/// Layers 8 output length: 63 x 32.
pub const L8_LEN: usize = 63 * 32;
/// Layers 9/10 output length: 63 x 32.
pub const L9_LEN: usize = 63 * 32;
/// Layer 11 output length: 32 x 32.
pub const L11_LEN: usize = 32 * 32;
/// Layers 12/13/14 output length: 32 x 32.
pub const L12_LEN: usize = 32 * 32;
/// Recurrent hidden width.
pub const HIDDEN: usize = 256;
/// Gate block width (four gates).
pub const GATES: usize = 4 * HIDDEN;
/// Layer 17 output length.
pub const L17_LEN: usize = 512;
/// Layer 18 / final output length.
pub const L18_LEN: usize = 1026;

/// Every diagnostic f32 buffer, in layer order.
///
/// Fused/identity layers (`L2`, `L6`, `L10`, `L13`, `L14`) repeat the buffer of
/// the layer that produced them; `out` is the final output.  `simd-check`
/// walks this list and compares all of them bit for bit.
pub const DIAG_F32_SLOTS: &[&str] = &[
    "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14",
    "L15_y", "L15_c", "L16_y", "L16_c", "L17", "L18", "out", "history",
];

/// Every diagnostic `int16` buffer, in layer order.
pub const DIAG_I16_SLOTS: &[&str] = &["L15_qx", "L15_qh", "L16_qx", "L16_qh", "L17_qx", "L18_qx"];

/// Per-stage labels used by [`VoiceModel::process_profiled`].
pub const STAGE_LABELS: [&str; 18] = [
    "staging",
    "L0",
    "L1",
    "L3",
    "L4",
    "L5",
    "L7",
    "L8",
    "L9",
    "L11",
    "L12",
    "L15_quant",
    "L15_lstm",
    "L16_lstm",
    "L17_matmul",
    "L18_quant",
    "L18_matmul",
    "history",
];

/// Number of timed stages in one frame.
pub const STAGES: usize = STAGE_LABELS.len();

// Workspace slot order; resolved once in `new`.
const S_HISTORY: usize = 0;
const S_L0: usize = 1;
const S_L1: usize = 2;
const S_L3: usize = 3;
const S_L4: usize = 4;
const S_L5: usize = 5;
const S_L7: usize = 6;
const S_L8: usize = 7;
const S_L9: usize = 8;
const S_L11: usize = 9;
const S_L12: usize = 10;
const S_L15_Y: usize = 11;
const S_L15_C: usize = 12;
const S_L16_Y: usize = 13;
const S_L16_C: usize = 14;
const S_L17: usize = 15;
const S_L18: usize = 16;

// Integer arena slots.
const I_QX15: usize = 0;
const I_QH15: usize = 1;
const I_QH16: usize = 2;
const I_QX18: usize = 3;

const A_ACC: usize = 0;

const CHANNELS: usize = 32;

/// Load-time / call-time failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoiceError {
    Container(CfwmError),
    WrongModel(u16),
    MissingTensor(u16),
    TensorLen {
        tag: u16,
        expected: usize,
        got: usize,
    },
    BadFrameLength(usize),
    BackendUnavailable(&'static str),
}

impl std::fmt::Display for VoiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VoiceError::Container(e) => write!(f, "container: {e}"),
            VoiceError::WrongModel(id) => write!(f, "container model id {id} is not voice default"),
            VoiceError::MissingTensor(tag) => write!(f, "container is missing tensor {tag}"),
            VoiceError::TensorLen { tag, expected, got } => {
                write!(f, "tensor {tag} has {got} elements, expected {expected}")
            }
            VoiceError::BadFrameLength(n) => {
                write!(f, "frame has {n} features, expected {INPUT_LEN}")
            }
            VoiceError::BackendUnavailable(name) => {
                write!(f, "backend {name} is not available on this CPU")
            }
        }
    }
}

impl std::error::Error for VoiceError {}

impl From<CfwmError> for VoiceError {
    fn from(e: CfwmError) -> Self {
        VoiceError::Container(e)
    }
}

/// Callback used by [`VoiceModel::process_profiled`]; the plain inference path
/// uses a no-op implementation so it pays nothing.
pub trait LayerTimer {
    /// Called once after each completed stage.
    fn tick(&mut self);
}

struct NoTimer;

impl LayerTimer for NoTimer {
    #[inline(always)]
    fn tick(&mut self) {}
}

/// Voice model runtime.
pub struct VoiceModel {
    container: Container,
    consts: Consts,
    ws: Workspace,
    q: Arena<i16>,
    acc: Arena<i32>,
    backend: Backend,
}

impl VoiceModel {
    /// Parse the container and allocate the fixed workspaces, using the best
    /// backend the CPU supports.
    pub fn new(container_bytes: &[u8]) -> Result<VoiceModel, VoiceError> {
        VoiceModel::with_backend(container_bytes, Backend::detect())
    }

    /// Parse the container with an explicit backend.
    ///
    /// Fails with [`VoiceError::BackendUnavailable`] when the requested backend
    /// cannot run on this CPU; there is no silent fallback, so a benchmark or
    /// verification run always reports the backend that actually executed.
    pub fn with_backend(
        container_bytes: &[u8],
        backend: Backend,
    ) -> Result<VoiceModel, VoiceError> {
        if !backend.is_available() {
            return Err(VoiceError::BackendUnavailable(backend.name()));
        }
        let container = Container::parse(container_bytes)?;
        VoiceModel::from_container(container, backend)
    }

    /// Build a model from an already-parsed container.
    ///
    /// This is the wasm entry point: the host image is parsed without a copy
    /// (either borrowed from a static image or handed in as an owned word
    /// buffer), then the fixed workspaces are allocated exactly once.
    pub fn from_container(
        container: Container,
        backend: Backend,
    ) -> Result<VoiceModel, VoiceError> {
        if !backend.is_available() {
            return Err(VoiceError::BackendUnavailable(backend.name()));
        }
        if container.model_id() != MODEL_VOICE_DEFAULT {
            return Err(VoiceError::WrongModel(container.model_id()));
        }

        for &(tag, expected) in F32_TENSORS {
            let got = container
                .f32(tag)
                .ok_or(VoiceError::MissingTensor(tag))?
                .len();
            if got != expected {
                return Err(VoiceError::TensorLen { tag, expected, got });
            }
        }
        for &(tag, expected) in I8_TENSORS {
            let got = container
                .i8(tag)
                .ok_or(VoiceError::MissingTensor(tag))?
                .len();
            if got != expected {
                return Err(VoiceError::TensorLen { tag, expected, got });
            }
        }
        let consts = Consts::from_container(&container).ok_or_else(|| {
            let got = container
                .u32_bits(TAG_CONSTANTS)
                .map(|s| s.len())
                .unwrap_or(0);
            VoiceError::TensorLen {
                tag: TAG_CONSTANTS,
                expected: CONST_COUNT,
                got,
            }
        })?;

        let ws = Workspace::new(&[
            ("history", HISTORY_LEN),
            ("L0", L0_LEN),
            ("L1", L1_LEN),
            ("L3", L3_LEN),
            ("L4", L4_LEN),
            ("L5", L5_LEN),
            ("L7", L7_LEN),
            ("L8", L8_LEN),
            ("L9", L9_LEN),
            ("L11", L11_LEN),
            ("L12", L12_LEN),
            ("L15_y", GATES),
            ("L15_c", HIDDEN),
            ("L16_y", GATES),
            ("L16_c", HIDDEN),
            ("L17", L17_LEN),
            ("L18", L18_LEN),
        ]);
        debug_assert_eq!(ws.slot_name(S_HISTORY), "history");
        debug_assert_eq!(ws.slot_name(S_L18), "L18");

        let q = Arena::<i16>::new(&[
            ("L15_qx", GATES),
            ("L15_qh", HIDDEN),
            ("L16_qh", HIDDEN),
            ("L18_qx", L17_LEN),
        ]);
        debug_assert_eq!(q.slot_name(I_QX15), "L15_qx");

        let acc = Arena::<i32>::new(&[("acc", L18_LEN)]);

        let mut model = VoiceModel {
            container,
            consts,
            ws,
            q,
            acc,
            backend,
        };
        model.reset();
        Ok(model)
    }

    /// Backend in use.
    #[inline]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Backend name in use (`"scalar"` or `"avx2"`).
    #[inline]
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Reset cross-frame state: history rows equal the staging bias, cell state
    /// and quantised hidden state are zero.
    pub fn reset(&mut self) {
        let bias = self
            .container
            .f32(TAG_STAGE_BIAS)
            .expect("validated at load");
        {
            let hist = self.ws.slot_mut(S_HISTORY);
            for row in 0..HISTORY_ROWS {
                hist[row * INPUT_LEN..(row + 1) * INPUT_LEN].copy_from_slice(bias);
            }
        }
        for slot in [
            S_L0, S_L1, S_L3, S_L4, S_L5, S_L7, S_L8, S_L9, S_L11, S_L12, S_L15_Y, S_L15_C,
            S_L16_Y, S_L16_C, S_L17, S_L18,
        ] {
            self.ws.slot_mut(slot).fill(0.0);
        }
        self.q.clear();
        self.acc.clear();
    }

    /// Run one frame through the full model.  No allocation.
    pub fn process(&mut self, frame: &[f32]) -> Result<(), VoiceError> {
        let mut timer = NoTimer;
        self.process_timed(frame, &mut timer)
    }

    /// Run one frame and return the nanoseconds spent in each of [`STAGES`],
    /// in [`STAGE_LABELS`] order.  Diagnostics only: this is the same kernel
    /// sequence, with a clock read between stages, so the numeric result is
    /// identical to [`VoiceModel::process`].
    pub fn process_profiled(
        &mut self,
        frame: &[f32],
        out: &mut [u64; STAGES],
    ) -> Result<(), VoiceError> {
        struct Recorder<'a> {
            out: &'a mut [u64; STAGES],
            at: Instant,
            idx: usize,
        }
        impl LayerTimer for Recorder<'_> {
            #[inline]
            fn tick(&mut self) {
                let now = Instant::now();
                if self.idx < STAGES {
                    self.out[self.idx] = now.duration_since(self.at).as_nanos() as u64;
                }
                self.idx += 1;
                self.at = now;
            }
        }
        let mut rec = Recorder {
            out,
            at: Instant::now(),
            idx: 0,
        };
        self.process_timed(frame, &mut rec)
    }

    fn process_timed<T: LayerTimer>(
        &mut self,
        frame: &[f32],
        timer: &mut T,
    ) -> Result<(), VoiceError> {
        match self.backend {
            Backend::Scalar => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::arch::is_x86_feature_detected!("fma") {
                        return self.process_impl::<cf_core::ScalarFmaSimd, T>(frame, timer);
                    }
                }
                self.process_impl::<ScalarSimd, T>(frame, timer)
            }
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => self.process_impl::<Avx2Simd, T>(frame, timer),
            #[cfg(not(target_arch = "x86_64"))]
            Backend::Avx2 => Err(VoiceError::BackendUnavailable("avx2")),
        }
    }

    fn process_impl<S: Simd, T: LayerTimer>(
        &mut self,
        frame: &[f32],
        timer: &mut T,
    ) -> Result<(), VoiceError> {
        if frame.len() != INPUT_LEN {
            return Err(VoiceError::BadFrameLength(frame.len()));
        }
        let c = &self.container;
        let k = self.consts;
        let act = k.act();

        // Input staging: multiply and add with two separate roundings.
        {
            let scale = c.f32(TAG_STAGE_SCALE).expect("validated at load");
            let bias = c.f32(TAG_STAGE_BIAS).expect("validated at load");
            let hist = self.ws.slot_mut(S_HISTORY);
            let dst = &mut hist[(HISTORY_ROWS - 1) * INPUT_LEN..];
            S::mul_add2_slices(dst, frame, scale, bias);
        }
        timer.tick();

        // Layer 0: 3x3 convolution over the 5-row history.
        {
            let w0 = c.f32(TAG_L0_W).expect("validated at load");
            let (hist, l0) = self.ws.get2_mut(S_HISTORY, S_L0);
            l0.fill(0.0);
            for kh in 0..3 {
                for kw in 0..3 {
                    let tap = w0[kh * 3 + kw];
                    for r in 0..3 {
                        let src_off = (kh + r) * INPUT_LEN + kw;
                        let src = &hist[src_off..src_off + 511];
                        let dst = &mut l0[r * 511..(r + 1) * 511];
                        S::mul_add_tap(dst, src, tap);
                    }
                }
            }
        }
        timer.tick();

        // Layer 1: 1x1 convolution (one input channel) + bias + ReLU.
        {
            let w1 = c.f32(TAG_L1_W).expect("validated at load");
            let b1 = c.f32(TAG_L1_B).expect("validated at load");
            let (l0, l1) = self.ws.get2_mut(S_L0, S_L1);
            S::conv1x1_1(l1, l0, w1, b1);
        }
        timer.tick();

        // Layer 2 is the identity on layer 1's buffer.

        // Layer 3: max pool, kernel 3 / stride 2 along the length axis.
        {
            let (l1, l3) = self.ws.get2_mut(S_L1, S_L3);
            S::maxpool1x3s2(l3, l1, 511);
        }
        timer.tick();

        // Layer 4: depthwise 3x3, taps in [kh][kw] order, FMA.
        {
            let w4 = c.f32(TAG_L4_W).expect("validated at load");
            let (l3, l4) = self.ws.get2_mut(S_L3, S_L4);
            S::depthwise_3x3(l4, l3, w4, 255);
        }
        timer.tick();

        // Layer 5: 1x1 convolution + bias + ReLU.
        {
            let w = c.f32(TAG_L5_W).expect("validated at load");
            let b = c.f32(TAG_L5_B).expect("validated at load");
            let (src, dst) = self.ws.get2_mut(S_L4, S_L5);
            S::conv1x1_32(dst, src, w, b);
        }
        timer.tick();

        // Layer 7: max pool.
        {
            let (l5, l7) = self.ws.get2_mut(S_L5, S_L7);
            S::maxpool1x3s2(l7, l5, 253);
        }
        timer.tick();

        // Layer 8: depthwise 1x3 stride 2, left padding 0.
        {
            let w = c.f32(TAG_L8_W).expect("validated at load");
            let (src, dst) = self.ws.get2_mut(S_L7, S_L8);
            S::depthwise_1x3s2(dst, src, w, 0);
        }
        timer.tick();

        // Layer 9: 1x1 convolution + bias + ReLU.
        {
            let w = c.f32(TAG_L9_W).expect("validated at load");
            let b = c.f32(TAG_L9_B).expect("validated at load");
            let (src, dst) = self.ws.get2_mut(S_L8, S_L9);
            S::conv1x1_32(dst, src, w, b);
        }
        timer.tick();

        // Layer 11: depthwise 1x3 stride 2, left padding 1.
        {
            let w = c.f32(TAG_L11_W).expect("validated at load");
            let (src, dst) = self.ws.get2_mut(S_L9, S_L11);
            S::depthwise_1x3s2(dst, src, w, 1);
        }
        timer.tick();

        // Layer 12: 1x1 convolution + bias + ReLU.
        {
            let w = c.f32(TAG_L12_W).expect("validated at load");
            let b = c.f32(TAG_L12_B).expect("validated at load");
            let (src, dst) = self.ws.get2_mut(S_L11, S_L12);
            S::conv1x1_32(dst, src, w, b);
        }
        timer.tick();

        // Layers 13/14: ReLU fused into layer 12, reshape is the identity.

        // Layer 15: quantise the 1024 f32 features, then the int8 gate GEMMs.
        {
            let inv = k.inv_in15();
            S::quantize_i16(
                self.q.slot_mut(I_QX15),
                self.ws.slot(S_L12),
                inv,
                k.clamp_lstm,
            );
        }
        timer.tick();
        {
            let (y, cell) = self.ws.get2_mut(S_L15_Y, S_L15_C);
            let (qx, qh) = self.q.get2_mut(I_QX15, I_QH15);
            let acc = self.acc.slot_mut(A_ACC);
            lstm_layer::<S>(
                y,
                cell,
                qh,
                &*qx,
                c.i8(TAG_L15_W_IN).expect("validated at load"),
                c.i8(TAG_L15_ZP_IN).expect("validated at load"),
                c.f32(TAG_L15_SP_IN).expect("validated at load"),
                c.i8(TAG_L15_W_REC).expect("validated at load"),
                c.i8(TAG_L15_ZP_REC).expect("validated at load"),
                c.f32(TAG_L15_SP_REC).expect("validated at load"),
                c.f32(TAG_L15_BIAS).expect("validated at load"),
                L12_LEN,
                acc,
                k.inv_h15(),
                k.clamp_lstm,
                &act,
            );
        }
        timer.tick();

        // Layer 16: input is layer 15's int16 hidden state (same buffer).
        {
            let (y, cell) = self.ws.get2_mut(S_L16_Y, S_L16_C);
            let (qh15, qh16) = self.q.get2_mut(I_QH15, I_QH16);
            let acc = self.acc.slot_mut(A_ACC);
            lstm_layer::<S>(
                y,
                cell,
                qh16,
                &*qh15,
                c.i8(TAG_L16_W_IN).expect("validated at load"),
                c.i8(TAG_L16_ZP_IN).expect("validated at load"),
                c.f32(TAG_L16_SP_IN).expect("validated at load"),
                c.i8(TAG_L16_W_REC).expect("validated at load"),
                c.i8(TAG_L16_ZP_REC).expect("validated at load"),
                c.f32(TAG_L16_SP_REC).expect("validated at load"),
                c.f32(TAG_L16_BIAS).expect("validated at load"),
                HIDDEN,
                acc,
                k.inv_h16(),
                k.clamp_lstm,
                &act,
            );
        }
        timer.tick();

        // Layer 17: int8 MatMul with fused ReLU.
        {
            let dst = self.ws.slot_mut(S_L17);
            let qx = self.q.slot(I_QH16);
            let acc = self.acc.slot_mut(A_ACC);
            matmul::<S>(
                dst,
                acc,
                c.i8(TAG_L17_W).expect("validated at load"),
                c.i8(TAG_L17_ZP).expect("validated at load"),
                c.f32(TAG_L17_SP).expect("validated at load"),
                c.f32(TAG_L17_BIAS).expect("validated at load"),
                qx,
                256,
                L17_LEN,
                true,
            );
        }
        timer.tick();

        // Layer 18: quantise layer 17's f32 output, int8 MatMul, no ReLU.
        {
            S::quantize_i16(
                self.q.slot_mut(I_QX18),
                self.ws.slot(S_L17),
                k.inv_mm18(),
                k.clamp_mm,
            );
        }
        timer.tick();
        {
            let dst = self.ws.slot_mut(S_L18);
            let qx = self.q.slot(I_QX18);
            let acc = self.acc.slot_mut(A_ACC);
            matmul::<S>(
                dst,
                acc,
                c.i8(TAG_L18_W).expect("validated at load"),
                c.i8(TAG_L18_ZP).expect("validated at load"),
                c.f32(TAG_L18_SP).expect("validated at load"),
                c.f32(TAG_L18_BIAS).expect("validated at load"),
                qx,
                512,
                L18_LEN,
                false,
            );
        }
        timer.tick();

        // The shift happens after the forward pass.
        {
            let hist = self.ws.slot_mut(S_HISTORY);
            hist.copy_within(INPUT_LEN..HISTORY_LEN, 0);
        }
        timer.tick();
        Ok(())
    }

    /// Diagnostic view of an intermediate f32 buffer by name.
    ///
    /// Fused/identity layers are aliases: `L2` is `L1`, `L6` is `L5`, `L10` is
    /// `L9`, `L13`/`L14` are `L12`, `stage` is `history`, `out` is `L18`.
    pub fn slot(&self, name: &str) -> Option<&[f32]> {
        let key = match name {
            "L2" => "L1",
            "L6" => "L5",
            "L10" => "L9",
            "L13" | "L14" => "L12",
            "stage" => "history",
            "out" => "L18",
            other => other,
        };
        let idx = self.ws.slot_index(key)?;
        Some(self.ws.slot(idx))
    }

    /// Diagnostic view of a quantised (`int16`) buffer by name.
    ///
    /// `L16_qx` is layer 15's hidden state and `L17_qx` is layer 16's hidden
    /// state: the reference passes those buffers through unchanged.
    pub fn slot_i16(&self, name: &str) -> Option<&[i16]> {
        let idx = match name {
            "L15_qx" => I_QX15,
            "L15_qh" | "L16_qx" => I_QH15,
            "L16_qh" | "L17_qx" => I_QH16,
            "L18_qx" => I_QX18,
            _ => return None,
        };
        Some(self.q.slot(idx))
    }

    /// Final 1026 f32 outputs.
    pub fn output(&self) -> &[f32] {
        self.ws.slot(S_L18)
    }

    /// Container behind the model (tensor table / ids).
    pub fn container(&self) -> &Container {
        &self.container
    }

    /// Restored scalar constants (exposed for the regression test only).
    pub fn consts(&self) -> &Consts {
        &self.consts
    }

    /// Fixed f32 workspace size in elements.
    pub fn workspace_floats(&self) -> usize {
        self.ws.capacity_floats()
    }

    /// Fixed int16 arena size in elements.
    pub fn workspace_i16(&self) -> usize {
        self.q.capacity()
    }

    /// Fixed int32 arena size in elements.
    pub fn workspace_i32(&self) -> usize {
        self.acc.capacity()
    }
}

// ---------------------------------------------------------------------------
// tensor shape table
// ---------------------------------------------------------------------------
const F32_TENSORS: &[(u16, usize)] = &[
    (TAG_STAGE_SCALE, INPUT_LEN),
    (TAG_STAGE_BIAS, INPUT_LEN),
    (TAG_L0_W, 9),
    (TAG_L1_W, CHANNELS),
    (TAG_L1_B, CHANNELS),
    (TAG_L4_W, 9 * CHANNELS),
    (TAG_L5_W, CHANNELS * CHANNELS),
    (TAG_L5_B, CHANNELS),
    (TAG_L8_W, 3 * CHANNELS),
    (TAG_L9_W, CHANNELS * CHANNELS),
    (TAG_L9_B, CHANNELS),
    (TAG_L11_W, 3 * CHANNELS),
    (TAG_L12_W, CHANNELS * CHANNELS),
    (TAG_L12_B, CHANNELS),
    (TAG_L15_BIAS, GATES),
    (TAG_L15_SP_IN, GATES),
    (TAG_L15_SP_REC, GATES),
    (TAG_L16_BIAS, GATES),
    (TAG_L16_SP_IN, GATES),
    (TAG_L16_SP_REC, GATES),
    (TAG_L17_BIAS, L17_LEN),
    (TAG_L17_SP, L17_LEN),
    (TAG_L18_BIAS, L18_LEN),
    (TAG_L18_SP, L18_LEN),
];

const I8_TENSORS: &[(u16, usize)] = &[
    (TAG_L15_W_IN, GATES * L12_LEN),
    (TAG_L15_ZP_IN, GATES),
    (TAG_L15_W_REC, HIDDEN * GATES),
    (TAG_L15_ZP_REC, GATES),
    (TAG_L16_W_IN, HIDDEN * GATES),
    (TAG_L16_ZP_IN, GATES),
    (TAG_L16_W_REC, HIDDEN * GATES),
    (TAG_L16_ZP_REC, GATES),
    (TAG_L17_W, 256 * L17_LEN),
    (TAG_L17_ZP, L17_LEN),
    (TAG_L18_W, L17_LEN * L18_LEN),
    (TAG_L18_ZP, L18_LEN),
];

// ---------------------------------------------------------------------------
// layer bodies (backend independent; the kernels are trait methods)
// ---------------------------------------------------------------------------

/// One recurrent layer.
///
/// `y` receives the four gate blocks, `cell` is the persistent `f32` cell
/// state, `qh` is the persistent int16 hidden state (also the next frame's
/// recurrent input), `qx` is this frame's int16 input.
#[allow(clippy::too_many_arguments)]
fn lstm_layer<S: Simd>(
    y: &mut [f32],
    cell: &mut [f32],
    qh: &mut [i16],
    qx: &[i16],
    w_in: &[i8],
    zp_in: &[i8],
    sp_in: &[f32],
    w_rec: &[i8],
    zp_rec: &[i8],
    sp_rec: &[f32],
    bias: &[f32],
    k_in: usize,
    acc: &mut [i32],
    inv_h: f32,
    clamp: i32,
    k: &ActConsts,
) {
    let h = HIDDEN;
    let n = GATES;
    debug_assert_eq!(y.len(), n);
    debug_assert_eq!(cell.len(), h);
    debug_assert_eq!(qh.len(), h);

    // Input GEMM: y = f32(acc) * sp_in + bias.
    S::gemm_i8(acc, w_in, zp_in, qx, k_in, n);
    S::scale_i32_add(y, acc, sp_in, bias);
    // Recurrent GEMM: y = f32(acc) * sp_rec + y.
    S::gemm_i8(acc, w_rec, zp_rec, qh, h, n);
    S::scale_i32_add_inplace(y, acc, sp_rec);

    // Gate order is [i, o, f, g], each block HIDDEN wide.
    S::lstm_activate(y, h, k);
    S::lstm_cell(cell, qh, y, inv_h, clamp, k);
}

/// One int8 projection: `f32(acc) * sp + bias`, optional ReLU.
#[allow(clippy::too_many_arguments)]
fn matmul<S: Simd>(
    dst: &mut [f32],
    acc: &mut [i32],
    w: &[i8],
    zp: &[i8],
    sp: &[f32],
    bias: &[f32],
    qx: &[i16],
    k: usize,
    n: usize,
    apply_relu: bool,
) {
    debug_assert_eq!(dst.len(), n);
    S::gemm_i8(acc, w, zp, qx, k, n);
    if apply_relu {
        S::scale_i32_add_relu(dst, acc, sp, bias);
    } else {
        S::scale_i32_add(dst, acc, sp, bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_bytes() -> &'static [u8] {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/voice-default.cfwm"
        ))
    }

    #[test]
    fn loads_container_and_runs_frames() {
        let mut m = VoiceModel::new(model_bytes()).unwrap();
        assert_eq!(m.container().model_id(), MODEL_VOICE_DEFAULT);
        assert_eq!(m.container().tensor_count(), 37);
        assert!(m.backend().is_available());
        let mut frame = [0.0f32; INPUT_LEN];
        for (i, v) in frame.iter_mut().enumerate() {
            *v = ((i * 37 % 101) as f32) * 0.01 - 0.5;
        }
        m.process(&frame).unwrap();
        assert_eq!(m.slot("L0").unwrap().len(), L0_LEN);
        assert_eq!(m.slot("L1").unwrap().len(), L1_LEN);
        assert_eq!(m.slot("L2").unwrap().len(), L1_LEN);
        assert_eq!(m.slot("L3").unwrap().len(), L3_LEN);
        assert_eq!(m.slot("L4").unwrap().len(), L4_LEN);
        assert_eq!(m.slot("L5").unwrap().len(), L5_LEN);
        assert_eq!(m.slot("L6").unwrap().len(), L5_LEN);
        assert_eq!(m.slot("L7").unwrap().len(), L7_LEN);
        assert_eq!(m.slot("L8").unwrap().len(), L8_LEN);
        assert_eq!(m.slot("L9").unwrap().len(), L9_LEN);
        assert_eq!(m.slot("L10").unwrap().len(), L9_LEN);
        assert_eq!(m.slot("L11").unwrap().len(), L11_LEN);
        assert_eq!(m.slot("L12").unwrap().len(), L12_LEN);
        assert_eq!(m.slot("L13").unwrap().len(), L12_LEN);
        assert_eq!(m.slot("L14").unwrap().len(), L12_LEN);
        assert_eq!(m.slot("L15_y").unwrap().len(), GATES);
        assert_eq!(m.slot("L16_y").unwrap().len(), GATES);
        assert_eq!(m.slot("L17").unwrap().len(), L17_LEN);
        assert_eq!(m.slot("L18").unwrap().len(), L18_LEN);
        assert_eq!(m.output().len(), L18_LEN);
        assert_eq!(m.slot_i16("L15_qx").unwrap().len(), GATES);
        assert_eq!(m.slot_i16("L15_qh").unwrap().len(), HIDDEN);
        assert_eq!(m.slot_i16("L16_qx").unwrap().len(), HIDDEN);
        assert_eq!(m.slot_i16("L16_qh").unwrap().len(), HIDDEN);
        assert_eq!(m.slot_i16("L17_qx").unwrap().len(), HIDDEN);
        assert_eq!(m.slot_i16("L18_qx").unwrap().len(), L17_LEN);
        assert!(m.process(&[0.0f32; 3]).is_err());
    }

    #[test]
    fn state_persists_and_resets() {
        let mut m = VoiceModel::new(model_bytes()).unwrap();
        let mut frame = [0.0f32; INPUT_LEN];
        for (i, v) in frame.iter_mut().enumerate() {
            *v = ((i * 13 % 71) as f32) * 0.01 - 0.3;
        }
        m.process(&frame).unwrap();
        let after_one = m.output().to_vec();
        m.reset();
        m.process(&frame).unwrap();
        assert_eq!(after_one, m.output().to_vec());
        m.process(&frame).unwrap();
        assert_ne!(after_one, m.output().to_vec());
    }

    #[test]
    fn profiled_matches_plain_process() {
        let mut plain = VoiceModel::new(model_bytes()).unwrap();
        let mut profiled = VoiceModel::new(model_bytes()).unwrap();
        let mut frame = [0.0f32; INPUT_LEN];
        for (i, v) in frame.iter_mut().enumerate() {
            *v = ((i * 41 % 89) as f32) * 0.02 - 0.6;
        }
        let mut times = [0u64; STAGES];
        for _ in 0..3 {
            plain.process(&frame).unwrap();
            profiled.process_profiled(&frame, &mut times).unwrap();
            for name in DIAG_F32_SLOTS {
                assert_eq!(
                    plain.slot(name).unwrap(),
                    profiled.slot(name).unwrap(),
                    "stage mismatch in {name}"
                );
            }
        }
        assert!(times.iter().any(|&t| t > 0), "profiler recorded nothing");
    }

    #[test]
    fn scalar_backend_is_selectable() {
        let mut m = VoiceModel::with_backend(model_bytes(), Backend::Scalar).unwrap();
        assert_eq!(m.backend_name(), "scalar");
        let frame = [0.25f32; INPUT_LEN];
        m.process(&frame).unwrap();
        assert!(m.output().iter().all(|v| v.is_finite()));
    }
}
