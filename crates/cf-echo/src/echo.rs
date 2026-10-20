//! Echo-cancellation model: 89 layers from the shared `.cfwm` container.
//!
//! The graph is a Y-net: two encoder branches over a 6-deep staged feature
//! window, a 33-position neck with two 33x33 MatMuls and residual adds, two
//! GRU layers, an int8 projection, and a decoder built from depthwise
//! deconvolutions, 3x1 convolutions, skip 1x1 projections, two more int8
//! projections and a polynomial sigmoid.
//!
//! Numeric contract (load-bearing, do not "clean up"):
//!
//! * staging: `f32(f32(x * A) + B)`, two roundings, not an FMA;
//! * the 6-row history is shifted **before** the new frame is staged, so the
//!   window seen by the forward pass is `[t-5 .. t]` with the initializer for
//!   `t < 0`;
//! * the encoder stacks are `[ch0(6) | ch1(6)]` and `[ch1(6) | ch2(6)]`,
//!   position-major (`h * 12 + channel`);
//! * depthwise 5x1 taps are `h-4+2k` (dilated by 2); the 12-channel layers use
//!   multiply-then-add, the `C % 16 == 0` layers use FMA; out-of-range taps
//!   contribute an exact zero operand;
//! * 1x1 convolutions pre-fill the accumulator with the bias and apply one FMA
//!   per input channel in ascending order;
//! * max pool 5x1 stride 2 drops out-of-range taps and starts from `-inf`,
//!   so ties select the later operand;
//! * the 33x33 MatMul accumulates input positions ascending into
//!   `out[j*64+c]`;
//! * GRU: gate row order `[z, r, n]`, per-frame `A_in = maxabs(x)/8192`,
//!   `A_h = maxabs(h)/127` (or `/32767`), zero points subtracted in the integer
//!   domain, `y = f32(acc) * f32(A * scale) + bias`, sigmoid/tanh are the
//!   rational `rcpps` forms, `n = tanh(yhh*r + yih)`, and
//!   `h = (1-z)*n + z*h_prev` with two multiplies and one add (**no FMA**);
//! * int8 projections: per-frame `A = maxabs/8192`, round-to-nearest-even for
//!   the 8-wide body and half-away-from-zero for the `K % 8` tail, clamp
//!   `+-8192`; the input GEMM of the first GRU (`K = 2112`) accumulates in
//!   `i64` because its worst-case bound exceeds `i32`; the other GEMMs have
//!   proven `i32` bounds;
//! * depthwise deconvolution writes only even output rows,
//!   `out[2j] = bias + sum_{k=4..0} fma(x[j+2-k], w[c][k], acc)`; odd rows are
//!   the bias;
//! * 3x1 decoder convolutions are weights `[oc][ic][k]`, `ic` outer and `k`
//!   inner, FMA; 67/72/77 fuse leaky-ReLU, 82 is linear and uses the
//!   tap-0/1/2 nested-FMA form plus a sequential bias-first horizontal sum;
//! * the final sigmoid uses the exp polynomial and reciprocal table with a
//!   `+-88.37625885` clamp for elements `0..8*floor(n/8)`; the remaining
//!   `n % 8` elements use `1 / (f32(exp(-x)) + 1)` with a double-precision
//!   exponential and a real f32 division.
//!
//! The runtime is the scalar reference.  Every intermediate is exposed through
//! [`EchoObserver`] so `cf-cli check echo` can compare all 89 layers with the
//! reference engine without giving up the buffer reuse that keeps the
//! workspace small.  The ordinary [`EchoModel::process`] path uses a no-op
//! observer and performs no allocation.

use cf_core::cfwm::{CfwmError, Container};
use cf_core::fma::mul_add2;
#[cfg(target_arch = "x86_64")]
use cf_core::Avx2Simd;
use cf_core::{Arena, Backend, Workspace};
use cf_core::{EchoActConsts, ScalarSimd, Simd};

use crate::consts::{EchoConsts, CONST_COUNT};
use crate::tags::{
    conv1_tags, conv3_tags, deconv_tags, depthwise_tags, gemm_role, gemm_tag, gru_role, gru_tag,
    TAG_CONSTANTS, TAG_HISTORY_INIT, TAG_MM33_A, TAG_MM33_B, TAG_STAGE_BIAS, TAG_STAGE_SCALE,
};

/// Stable model id for the default echo model.
pub const MODEL_ECHO_DEFAULT: u16 = 2;
/// Input feature channels.
pub const INPUT_CHANNELS: usize = 3;
/// Frequency bins per feature channel.
pub const BINS: usize = 513;
/// One input frame (`3 x 513`).
pub const INPUT_LEN: usize = INPUT_CHANNELS * BINS;
/// Number of staged history rows.
pub const HISTORY_ROWS: usize = 6;
/// Staged history length (`6 x 3 x 513`), laid out `[row][channel][bin]`.
pub const HISTORY_LEN: usize = HISTORY_ROWS * INPUT_LEN;
/// Encoder input channels: two feature channels x six rows.
pub const STACK_CHANNELS: usize = 2 * HISTORY_ROWS;
/// GRU hidden width.
pub const HIDDEN: usize = 128;
/// GRU gate block width (`[z, r, n]`).
pub const GRU_GATES: usize = 3 * HIDDEN;
/// Final output length.
pub const OUTPUT_LEN: usize = 513;
/// Number of graph layers.
pub const LAYERS: usize = 89;

/// Width of every layer output, in graph order.
///
/// Fused/identity layers repeat the width of the buffer they alias.  The sum
/// is the per-frame layer-verification surface (247,351 values).
pub const LAYER_LEN: [usize; LAYERS] = [
    6156, 6156, 8208, 8208, 4112, 4112, 4112, 4112, 2064, 2064, 4128, 4128, 2080, 2080, 2080, 2080,
    1056, 6156, 6156, 8208, 8208, 4112, 4112, 4112, 4112, 2064, 2064, 4128, 4128, 2080, 2080, 2080,
    2080, 1056, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112,
    2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112, 128, 128, 1056, 1056, 1056,
    1056, 1056, 2080, 2080, 2080, 2080, 2080, 4128, 2064, 2064, 2064, 2064, 4112, 4112, 4112, 4112,
    4112, 8208, 513, 513, 513, 513, 513, 513, 513,
];

/// Conv1x1 layers in tag order: `(layer, in channels, out channels)`.
const CONV1_LAYERS: [(usize, usize, usize); 18] = [
    (2, 12, 16),
    (6, 16, 16),
    (10, 16, 32),
    (14, 32, 32),
    (19, 12, 16),
    (23, 16, 16),
    (27, 16, 32),
    (31, 32, 32),
    (35, 64, 64),
    (41, 64, 64),
    (44, 64, 64),
    (47, 64, 64),
    (53, 64, 64),
    (56, 64, 64),
    (64, 32, 32),
    (69, 32, 32),
    (74, 16, 16),
    (79, 16, 16),
];

/// Depthwise layers in tag order: `(layer, channels, taps)`.
const DEPTHWISE_LAYERS: [(usize, usize, usize); 10] = [
    (1, 12, 5),
    (5, 16, 5),
    (9, 16, 5),
    (13, 32, 5),
    (18, 12, 5),
    (22, 16, 5),
    (26, 16, 5),
    (30, 32, 5),
    (43, 64, 3),
    (55, 64, 3),
];

/// GRU layers in tag order: `(layer, input width)`.
const GRU_LAYERS: [(usize, usize); 2] = [(59, 2112), (60, 128)];

/// Int8 projection layers in tag order: `(layer, k, n)`.
const GEMM_LAYERS: [(usize, usize, usize); 3] = [(61, 128, 1056), (85, 513, 513), (87, 513, 513)];

/// Deconvolution layers in tag order: `(layer, channels)`.
const DECONV_LAYERS: [(usize, usize); 4] = [(66, 32), (71, 32), (76, 16), (81, 16)];

/// 3x1 decoder convolutions in tag order: `(layer, in channels, out channels)`.
const CONV3_LAYERS: [(usize, usize, usize); 4] =
    [(67, 32, 32), (72, 32, 16), (77, 16, 16), (82, 16, 1)];

// Workspace slot order; resolved once at load time.
const S_HISTORY: usize = 0;
const S_X: usize = 1;
const S_X2: usize = 2;
const S_T513_16: usize = 3;
const S_T513_16B: usize = 4;
const S_T257_16: usize = 5;
const S_T257_16B: usize = 6;
const S_T129_16: usize = 7;
const S_T129_16B: usize = 8;
const S_T129_32: usize = 9;
const S_T65_32: usize = 10;
const S_T65_32B: usize = 11;
const S_T65_32C: usize = 12;
const S_SKIP4: usize = 13;
const S_SKIP8: usize = 14;
const S_SKIP12: usize = 15;
const S_SKIP16: usize = 16;
const S_N33: usize = 17;
const S_NECK: usize = 18;
const S_T2112_A: usize = 19;
const S_T2112_B: usize = 20;
const S_T2112_C: usize = 21;
const S_H59: usize = 22;
const S_H60: usize = 23;
const S_Y: usize = 24;
const S_DEC: usize = 25;
const S_T1056: usize = 26;
const S_D65_32: usize = 27;
const S_OUT: usize = 28;
const S_SIG: usize = 29;

const WS_SPECS: &[(&str, usize)] = &[
    ("history", HISTORY_LEN),
    ("x", 513 * STACK_CHANNELS),
    ("x2", 513 * STACK_CHANNELS),
    ("t513_16", 513 * 16),
    ("t513_16b", 513 * 16),
    ("t257_16", 257 * 16),
    ("t257_16b", 257 * 16),
    ("t129_16", 129 * 16),
    ("t129_16b", 129 * 16),
    ("t129_32", 129 * 32),
    ("t65_32", 65 * 32),
    ("t65_32b", 65 * 32),
    ("t65_32c", 65 * 32),
    ("skip4", 257 * 16),
    ("skip8", 129 * 16),
    ("skip12", 65 * 32),
    ("skip16", 33 * 32),
    ("n33", 33 * 32),
    ("neck", 33 * 64),
    ("t2112a", 33 * 64),
    ("t2112b", 33 * 64),
    ("t2112c", 33 * 64),
    ("h59", HIDDEN),
    ("h60", HIDDEN),
    ("y", 2 * GRU_GATES),
    ("dec", 1056),
    ("t1056", 1056),
    ("d65_32", 65 * 32),
    ("out", OUTPUT_LEN),
    ("sig", OUTPUT_LEN),
];

// Integer arena slots.
const I_QX: usize = 0;
const I_QH: usize = 1;
const A_ACC32: usize = 0;
const A_ACC64: usize = 0;

/// Load-time / call-time failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EchoError {
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

impl std::fmt::Display for EchoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EchoError::Container(e) => write!(f, "container: {e}"),
            EchoError::WrongModel(id) => write!(f, "container model id {id} is not echo default"),
            EchoError::MissingTensor(tag) => write!(f, "container is missing tensor {tag}"),
            EchoError::TensorLen { tag, expected, got } => {
                write!(f, "tensor {tag} has {got} elements, expected {expected}")
            }
            EchoError::BadFrameLength(n) => {
                write!(f, "frame has {n} features, expected {INPUT_LEN}")
            }
            EchoError::BackendUnavailable(name) => {
                write!(f, "backend {name} is not available on this CPU")
            }
        }
    }
}

impl std::error::Error for EchoError {}

impl From<CfwmError> for EchoError {
    fn from(e: CfwmError) -> Self {
        EchoError::Container(e)
    }
}

/// Per-layer and per-buffer diagnostics, called as each value is produced.
///
/// The observer sees a buffer *after* the layer that writes it, so a checker
/// can compare every layer without the model keeping 89 live buffers.  The
/// plain [`EchoModel::process`] path uses [`NoObserver`], whose empty methods
/// are inlined away.
pub trait EchoObserver {
    /// One layer output, in graph order.
    fn layer(&mut self, _index: usize, _values: &[f32]) {}

    /// A named `f32` diagnostic buffer (staging, history, GRU gates, ...).
    fn diag_f32(&mut self, _name: &'static str, _values: &[f32]) {}

    /// A named quantised diagnostic buffer (`qx` / `qh`).
    fn diag_i16(&mut self, _name: &'static str, _values: &[i16]) {}
}

/// Observer that ignores everything.
pub struct NoObserver;

impl EchoObserver for NoObserver {}

/// Static diagnostic names for one GRU layer.
struct GruDiag {
    qx: &'static str,
    qh: &'static str,
    yih: &'static str,
    yhh: &'static str,
    z: &'static str,
    r: &'static str,
    n: &'static str,
    a_in: &'static str,
    a_h: &'static str,
}

const GRU59_DIAG: GruDiag = GruDiag {
    qx: "L59_qx",
    qh: "L59_qh",
    yih: "L59_yih",
    yhh: "L59_yhh",
    z: "L59_z",
    r: "L59_r",
    n: "L59_n",
    a_in: "L59_A_in",
    a_h: "L59_A_h",
};

const GRU60_DIAG: GruDiag = GruDiag {
    qx: "L60_qx",
    qh: "L60_qh",
    yih: "L60_yih",
    yhh: "L60_yhh",
    z: "L60_z",
    r: "L60_r",
    n: "L60_n",
    a_in: "L60_A_in",
    a_h: "L60_A_h",
};

/// Echo model runtime.
pub struct EchoModel {
    container: Container,
    consts: EchoConsts,
    backend: Backend,
    ws: Workspace,
    q: Arena<i16>,
    acc32: Arena<i32>,
    acc64: Arena<i64>,
}

impl EchoModel {
    /// Parse the container and allocate the fixed workspaces.
    ///
    /// The default constructor keeps the scalar backend: it is the verification
    /// oracle, and every command that compares against the reference dumps
    /// relies on it.  Use [`EchoModel::with_backend`] to select AVX2.
    pub fn new(container_bytes: &[u8]) -> Result<EchoModel, EchoError> {
        EchoModel::with_backend(container_bytes, Backend::Scalar)
    }

    /// Parse the container with an explicit backend.
    ///
    /// Fails with [`EchoError::BackendUnavailable`] when the requested backend
    /// is not usable on this CPU; there is no silent fallback, so a benchmark
    /// always reports the backend that actually executed.
    pub fn with_backend(container_bytes: &[u8], backend: Backend) -> Result<EchoModel, EchoError> {
        if !backend.is_available() {
            return Err(EchoError::BackendUnavailable(backend.name()));
        }
        let container = Container::parse(container_bytes)?;
        EchoModel::from_container_with(container, backend)
    }

    /// Build a model from an already-parsed container (scalar oracle).
    pub fn from_container(container: Container) -> Result<EchoModel, EchoError> {
        EchoModel::from_container_with(container, Backend::Scalar)
    }

    /// Build a model from an already-parsed container with an explicit backend.
    pub fn from_container_with(
        container: Container,
        backend: Backend,
    ) -> Result<EchoModel, EchoError> {
        if !backend.is_available() {
            return Err(EchoError::BackendUnavailable(backend.name()));
        }
        if container.model_id() != MODEL_ECHO_DEFAULT {
            return Err(EchoError::WrongModel(container.model_id()));
        }
        validate_tensors(&container)?;
        let consts = EchoConsts::from_container(&container).ok_or_else(|| {
            let got = container
                .u32_bits(TAG_CONSTANTS)
                .map(|s| s.len())
                .unwrap_or(0);
            EchoError::TensorLen {
                tag: TAG_CONSTANTS,
                expected: CONST_COUNT,
                got,
            }
        })?;

        let ws = Workspace::new(WS_SPECS);
        debug_assert_eq!(ws.slot_name(S_HISTORY), "history");
        debug_assert_eq!(ws.slot_name(S_SIG), "sig");
        debug_assert_eq!(ws.slot_len(S_T513_16), 513 * 16);
        debug_assert_eq!(ws.slot_len(S_Y), 2 * GRU_GATES);

        let q = Arena::<i16>::new(&[("qx", 2112), ("qh", HIDDEN)]);
        let acc32 = Arena::<i32>::new(&[("acc32", 1056)]);
        let acc64 = Arena::<i64>::new(&[("acc64", GRU_GATES)]);

        let mut model = EchoModel {
            container,
            consts,
            backend,
            ws,
            q,
            acc32,
            acc64,
        };
        model.reset();
        Ok(model)
    }

    /// Reset cross-frame state: the six history rows equal the initializer and
    /// both GRU hidden states are zero.
    pub fn reset(&mut self) {
        let init = self
            .container
            .f32(TAG_HISTORY_INIT)
            .expect("validated at load");
        self.ws.clear();
        {
            let hist = self.ws.slot_mut(S_HISTORY);
            for row in 0..HISTORY_ROWS {
                hist[row * INPUT_LEN..(row + 1) * INPUT_LEN].copy_from_slice(init);
            }
        }
        self.q.clear();
        self.acc32.clear();
        self.acc64.clear();
    }

    /// Run one frame through the full model.  No allocation.
    pub fn process(&mut self, frame: &[f32]) -> Result<(), EchoError> {
        let mut obs = NoObserver;
        self.process_with(frame, &mut obs)
    }

    /// Run one frame, reporting every layer output and the named internal
    /// buffers to `obs`.  Diagnostics only: the arithmetic is identical to
    /// [`EchoModel::process`].
    pub fn process_with<O: EchoObserver>(
        &mut self,
        frame: &[f32],
        obs: &mut O,
    ) -> Result<(), EchoError> {
        match self.backend {
            Backend::Scalar => self.process_impl::<ScalarSimd, O>(frame, obs),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => self.process_impl::<Avx2Simd, O>(frame, obs),
            #[cfg(not(target_arch = "x86_64"))]
            Backend::Avx2 => Err(EchoError::BackendUnavailable("avx2")),
        }
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

    fn process_impl<S: Simd, O: EchoObserver>(
        &mut self,
        frame: &[f32],
        obs: &mut O,
    ) -> Result<(), EchoError> {
        if frame.len() != INPUT_LEN {
            return Err(EchoError::BadFrameLength(frame.len()));
        }
        let c = &self.container;
        let k = self.consts;
        let act: EchoActConsts = k.act();

        // ---- staging: shift the window, then stage the new frame ----------
        {
            let hist = self.ws.slot_mut(S_HISTORY);
            hist.copy_within(INPUT_LEN..HISTORY_LEN, 0);
            let row = &mut hist[(HISTORY_ROWS - 1) * INPUT_LEN..];
            let a = f32_t(c, TAG_STAGE_SCALE);
            let b = f32_t(c, TAG_STAGE_BIAS);
            for i in 0..INPUT_LEN {
                row[i] = mul_add2(frame[i], a[i], b[i]);
            }
            obs.diag_f32("staged", row);
        }

        // ---- encoder branch 0: feature channels 0 and 1 -------------------
        {
            let (hist, x) = self.ws.get2_mut(S_HISTORY, S_X);
            stack_input(hist, x, 0, 1);
        }
        obs.layer(0, self.ws.slot(S_X));
        {
            let (w, b) = depthwise_tags(0);
            let (src, dst) = self.ws.get2_mut(S_X, S_X2);
            S::echo_depthwise(dst, src, f32_t(c, w), f32_t(c, b), 12, 5, false, BINS);
        }
        obs.layer(1, self.ws.slot(S_X2));
        {
            let (w, b) = conv1_tags(0);
            let (src, dst) = self.ws.get2_mut(S_X2, S_T513_16);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 12, 16);
        }
        obs.layer(2, self.ws.slot(S_T513_16));
        leaky_inplace(self.ws.slot_mut(S_T513_16), k.leaky_slope);
        obs.layer(3, self.ws.slot(S_T513_16));
        {
            let (src, dst) = self.ws.get2_mut(S_T513_16, S_SKIP4);
            S::echo_maxpool5(dst, src, 16, BINS);
        }
        obs.layer(4, self.ws.slot(S_SKIP4));
        {
            let (w, b) = depthwise_tags(1);
            let (src, dst) = self.ws.get2_mut(S_SKIP4, S_T257_16);
            S::echo_depthwise(dst, src, f32_t(c, w), f32_t(c, b), 16, 5, true, 257);
        }
        obs.layer(5, self.ws.slot(S_T257_16));
        {
            let (w, b) = conv1_tags(1);
            let (src, dst) = self.ws.get2_mut(S_T257_16, S_T257_16B);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 16, 16);
        }
        obs.layer(6, self.ws.slot(S_T257_16B));
        leaky_inplace(self.ws.slot_mut(S_T257_16B), k.leaky_slope);
        obs.layer(7, self.ws.slot(S_T257_16B));
        {
            let (src, dst) = self.ws.get2_mut(S_T257_16B, S_SKIP8);
            S::echo_maxpool5(dst, src, 16, 257);
        }
        obs.layer(8, self.ws.slot(S_SKIP8));
        {
            let (w, b) = depthwise_tags(2);
            let (src, dst) = self.ws.get2_mut(S_SKIP8, S_T129_16);
            S::echo_depthwise(dst, src, f32_t(c, w), f32_t(c, b), 16, 5, true, 129);
        }
        obs.layer(9, self.ws.slot(S_T129_16));
        {
            let (w, b) = conv1_tags(2);
            let (src, dst) = self.ws.get2_mut(S_T129_16, S_T129_32);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 16, 32);
        }
        obs.layer(10, self.ws.slot(S_T129_32));
        leaky_inplace(self.ws.slot_mut(S_T129_32), k.leaky_slope);
        obs.layer(11, self.ws.slot(S_T129_32));
        {
            let (src, dst) = self.ws.get2_mut(S_T129_32, S_SKIP12);
            S::echo_maxpool5(dst, src, 32, 129);
        }
        obs.layer(12, self.ws.slot(S_SKIP12));
        {
            let (w, b) = depthwise_tags(3);
            let (src, dst) = self.ws.get2_mut(S_SKIP12, S_T65_32);
            S::echo_depthwise(dst, src, f32_t(c, w), f32_t(c, b), 32, 5, true, 65);
        }
        obs.layer(13, self.ws.slot(S_T65_32));
        {
            let (w, b) = conv1_tags(3);
            let (src, dst) = self.ws.get2_mut(S_T65_32, S_T65_32B);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 32, 32);
        }
        obs.layer(14, self.ws.slot(S_T65_32B));
        leaky_inplace(self.ws.slot_mut(S_T65_32B), k.leaky_slope);
        obs.layer(15, self.ws.slot(S_T65_32B));
        {
            let (src, dst) = self.ws.get2_mut(S_T65_32B, S_SKIP16);
            S::echo_maxpool5(dst, src, 32, 65);
        }
        obs.layer(16, self.ws.slot(S_SKIP16));

        // ---- encoder branch 1: feature channels 1 and 2 -------------------
        {
            let (hist, x) = self.ws.get2_mut(S_HISTORY, S_X);
            stack_input(hist, x, 1, 2);
        }
        obs.layer(17, self.ws.slot(S_X));
        {
            let (w, b) = depthwise_tags(4);
            let (src, dst) = self.ws.get2_mut(S_X, S_X2);
            S::echo_depthwise(dst, src, f32_t(c, w), f32_t(c, b), 12, 5, false, BINS);
        }
        obs.layer(18, self.ws.slot(S_X2));
        {
            let (w, b) = conv1_tags(4);
            let (src, dst) = self.ws.get2_mut(S_X2, S_T513_16);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 12, 16);
        }
        obs.layer(19, self.ws.slot(S_T513_16));
        leaky_inplace(self.ws.slot_mut(S_T513_16), k.leaky_slope);
        obs.layer(20, self.ws.slot(S_T513_16));
        {
            let (src, dst) = self.ws.get2_mut(S_T513_16, S_T257_16);
            S::echo_maxpool5(dst, src, 16, BINS);
        }
        obs.layer(21, self.ws.slot(S_T257_16));
        {
            let (w, b) = depthwise_tags(5);
            let (src, dst) = self.ws.get2_mut(S_T257_16, S_T257_16B);
            S::echo_depthwise(dst, src, f32_t(c, w), f32_t(c, b), 16, 5, true, 257);
        }
        obs.layer(22, self.ws.slot(S_T257_16B));
        {
            let (w, b) = conv1_tags(5);
            let (src, dst) = self.ws.get2_mut(S_T257_16B, S_T257_16);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 16, 16);
        }
        obs.layer(23, self.ws.slot(S_T257_16));
        leaky_inplace(self.ws.slot_mut(S_T257_16), k.leaky_slope);
        obs.layer(24, self.ws.slot(S_T257_16));
        {
            let (src, dst) = self.ws.get2_mut(S_T257_16, S_T129_16);
            S::echo_maxpool5(dst, src, 16, 257);
        }
        obs.layer(25, self.ws.slot(S_T129_16));
        {
            let (w, b) = depthwise_tags(6);
            let (src, dst) = self.ws.get2_mut(S_T129_16, S_T129_16B);
            S::echo_depthwise(dst, src, f32_t(c, w), f32_t(c, b), 16, 5, true, 129);
        }
        obs.layer(26, self.ws.slot(S_T129_16B));
        {
            let (w, b) = conv1_tags(6);
            let (src, dst) = self.ws.get2_mut(S_T129_16B, S_T129_32);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 16, 32);
        }
        obs.layer(27, self.ws.slot(S_T129_32));
        leaky_inplace(self.ws.slot_mut(S_T129_32), k.leaky_slope);
        obs.layer(28, self.ws.slot(S_T129_32));
        {
            let (src, dst) = self.ws.get2_mut(S_T129_32, S_T65_32);
            S::echo_maxpool5(dst, src, 32, 129);
        }
        obs.layer(29, self.ws.slot(S_T65_32));
        {
            let (w, b) = depthwise_tags(7);
            let (src, dst) = self.ws.get2_mut(S_T65_32, S_T65_32B);
            S::echo_depthwise(dst, src, f32_t(c, w), f32_t(c, b), 32, 5, true, 65);
        }
        obs.layer(30, self.ws.slot(S_T65_32B));
        {
            let (w, b) = conv1_tags(7);
            let (src, dst) = self.ws.get2_mut(S_T65_32B, S_T65_32C);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 32, 32);
        }
        obs.layer(31, self.ws.slot(S_T65_32C));
        leaky_inplace(self.ws.slot_mut(S_T65_32C), k.leaky_slope);
        obs.layer(32, self.ws.slot(S_T65_32C));
        {
            let (src, dst) = self.ws.get2_mut(S_T65_32C, S_N33);
            S::echo_maxpool5(dst, src, 32, 65);
        }
        obs.layer(33, self.ws.slot(S_N33));

        // ---- neck: concat, two 33x33 MatMuls, residual adds ---------------
        {
            let (neck, a, b) = self.ws.get3_mut(S_NECK, S_SKIP16, S_N33);
            concat_neck(neck, a, b);
        }
        obs.layer(34, self.ws.slot(S_NECK));
        {
            let (w, b) = conv1_tags(8);
            let (src, dst) = self.ws.get2_mut(S_NECK, S_T2112_A);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 64, 64);
        }
        obs.layer(35, self.ws.slot(S_T2112_A));
        leaky_inplace(self.ws.slot_mut(S_T2112_A), k.leaky_slope);
        obs.layer(36, self.ws.slot(S_T2112_A));
        obs.layer(37, self.ws.slot(S_T2112_A));
        {
            let (src, dst) = self.ws.get2_mut(S_T2112_A, S_T2112_B);
            S::echo_mm33(dst, src, f32_t(c, TAG_MM33_A));
        }
        obs.layer(38, self.ws.slot(S_T2112_B));
        obs.layer(39, self.ws.slot(S_T2112_B));
        {
            let (dst, a, b) = self.ws.get3_mut(S_T2112_A, S_T2112_B, S_NECK);
            add3(dst, a, b);
        }
        obs.layer(40, self.ws.slot(S_T2112_A));
        {
            let (w, b) = conv1_tags(9);
            let (src, dst) = self.ws.get2_mut(S_T2112_A, S_T2112_B);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 64, 64);
        }
        obs.layer(41, self.ws.slot(S_T2112_B));
        leaky_inplace(self.ws.slot_mut(S_T2112_B), k.leaky_slope);
        obs.layer(42, self.ws.slot(S_T2112_B));
        {
            let (w, b) = depthwise_tags(8);
            let (src, dst) = self.ws.get2_mut(S_T2112_B, S_T2112_A);
            S::echo_depthwise(dst, src, f32_t(c, w), f32_t(c, b), 64, 3, true, 33);
        }
        obs.layer(43, self.ws.slot(S_T2112_A));
        {
            let (w, b) = conv1_tags(10);
            let (src, dst) = self.ws.get2_mut(S_T2112_A, S_T2112_B);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 64, 64);
        }
        obs.layer(44, self.ws.slot(S_T2112_B));
        leaky_inplace(self.ws.slot_mut(S_T2112_B), k.leaky_slope);
        obs.layer(45, self.ws.slot(S_T2112_B));
        {
            let (dst, a, b) = self.ws.get3_mut(S_T2112_A, S_NECK, S_T2112_B);
            add3(dst, a, b);
        }
        obs.layer(46, self.ws.slot(S_T2112_A));
        {
            let (w, b) = conv1_tags(11);
            let (src, dst) = self.ws.get2_mut(S_T2112_A, S_T2112_B);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 64, 64);
        }
        obs.layer(47, self.ws.slot(S_T2112_B));
        leaky_inplace(self.ws.slot_mut(S_T2112_B), k.leaky_slope);
        obs.layer(48, self.ws.slot(S_T2112_B));
        obs.layer(49, self.ws.slot(S_T2112_B));
        {
            let (src, dst) = self.ws.get2_mut(S_T2112_B, S_T2112_C);
            S::echo_mm33(dst, src, f32_t(c, TAG_MM33_B));
        }
        obs.layer(50, self.ws.slot(S_T2112_C));
        obs.layer(51, self.ws.slot(S_T2112_C));
        {
            let (dst, a, b) = self.ws.get3_mut(S_T2112_B, S_T2112_C, S_T2112_A);
            add3(dst, a, b);
        }
        obs.layer(52, self.ws.slot(S_T2112_B));
        {
            let (w, b) = conv1_tags(12);
            let (src, dst) = self.ws.get2_mut(S_T2112_B, S_T2112_A);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 64, 64);
        }
        obs.layer(53, self.ws.slot(S_T2112_A));
        leaky_inplace(self.ws.slot_mut(S_T2112_A), k.leaky_slope);
        obs.layer(54, self.ws.slot(S_T2112_A));
        {
            let (w, b) = depthwise_tags(9);
            let (src, dst) = self.ws.get2_mut(S_T2112_A, S_T2112_C);
            S::echo_depthwise(dst, src, f32_t(c, w), f32_t(c, b), 64, 3, true, 33);
        }
        obs.layer(55, self.ws.slot(S_T2112_C));
        {
            let (w, b) = conv1_tags(13);
            let (src, dst) = self.ws.get2_mut(S_T2112_C, S_T2112_A);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 64, 64);
        }
        obs.layer(56, self.ws.slot(S_T2112_A));
        relu_inplace(self.ws.slot_mut(S_T2112_A));
        obs.layer(57, self.ws.slot(S_T2112_A));
        obs.layer(58, self.ws.slot(S_T2112_A));

        // ---- GRU 59 (K = 2112) and GRU 60 (K = 128) ------------------------
        {
            let (h, y, x) = self.ws.get3_mut(S_H59, S_Y, S_T2112_A);
            let (qx, qh) = self.q.get2_mut(I_QX, I_QH);
            let acc32 = self.acc32.slot_mut(A_ACC32);
            let acc64 = self.acc64.slot_mut(A_ACC64);
            gru_step::<S, O>(
                h,
                y,
                &x[..2112],
                qx,
                qh,
                acc32,
                acc64,
                i8_t(c, gru_tag(0, gru_role::W_IN)),
                i8_t(c, gru_tag(0, gru_role::ZP_IN)),
                f32_t(c, gru_tag(0, gru_role::SCALE_IN)),
                i8_t(c, gru_tag(0, gru_role::W_REC)),
                i8_t(c, gru_tag(0, gru_role::ZP_REC)),
                f32_t(c, gru_tag(0, gru_role::SCALE_REC)),
                f32_t(c, gru_tag(0, gru_role::BIAS)),
                &k,
                &GRU59_DIAG,
                obs,
            );
        }
        obs.layer(59, self.ws.slot(S_H59));
        {
            let (h, y, x) = self.ws.get3_mut(S_H60, S_Y, S_H59);
            let (qx, qh) = self.q.get2_mut(I_QX, I_QH);
            let acc32 = self.acc32.slot_mut(A_ACC32);
            let acc64 = self.acc64.slot_mut(A_ACC64);
            gru_step::<S, O>(
                h,
                y,
                &x[..HIDDEN],
                qx,
                qh,
                acc32,
                acc64,
                i8_t(c, gru_tag(1, gru_role::W_IN)),
                i8_t(c, gru_tag(1, gru_role::ZP_IN)),
                f32_t(c, gru_tag(1, gru_role::SCALE_IN)),
                i8_t(c, gru_tag(1, gru_role::W_REC)),
                i8_t(c, gru_tag(1, gru_role::ZP_REC)),
                f32_t(c, gru_tag(1, gru_role::SCALE_REC)),
                f32_t(c, gru_tag(1, gru_role::BIAS)),
                &k,
                &GRU60_DIAG,
                obs,
            );
        }
        obs.layer(60, self.ws.slot(S_H60));

        // ---- int8 projection 61 and the decoder ---------------------------
        {
            let (dst, src) = self.ws.get2_mut(S_DEC, S_H60);
            let qx = self.q.slot_mut(I_QX);
            let a = S::echo_quantize_input(qx, src, k.quant_scale);
            obs.diag_i16("L61_qx", &qx[..128]);
            obs.diag_f32("L61_A", &[a]);
            let acc = self.acc32.slot_mut(A_ACC32);
            projection::<S>(
                dst,
                acc,
                i8_t(c, gemm_tag(0, gemm_role::WEIGHT)),
                i8_t(c, gemm_tag(0, gemm_role::ZP)),
                f32_t(c, gemm_tag(0, gemm_role::SCALE)),
                f32_t(c, gemm_tag(0, gemm_role::BIAS)),
                &qx[..128],
                128,
                1056,
                a,
            );
        }
        obs.layer(61, self.ws.slot(S_DEC));
        leaky_inplace(self.ws.slot_mut(S_DEC), k.leaky_slope);
        obs.layer(62, self.ws.slot(S_DEC));
        obs.layer(63, self.ws.slot(S_DEC));
        {
            let (w, b) = conv1_tags(14);
            let (src, dst) = self.ws.get2_mut(S_SKIP16, S_T1056);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 32, 32);
        }
        obs.layer(64, self.ws.slot(S_T1056));
        {
            let (dst, src) = self.ws.get2_mut(S_DEC, S_T1056);
            add_assign(dst, src);
        }
        obs.layer(65, self.ws.slot(S_DEC));
        {
            let (w, b) = deconv_tags(0);
            let (src, dst) = self.ws.get2_mut(S_DEC, S_D65_32);
            S::echo_deconv(dst, src, f32_t(c, w), f32_t(c, b), 33, 32);
        }
        obs.layer(66, self.ws.slot(S_D65_32));
        {
            let (w, b) = conv3_tags(0);
            let (src, dst) = self.ws.get2_mut(S_D65_32, S_T65_32);
            S::echo_conv3(
                dst,
                src,
                f32_t(c, w),
                f32_t(c, b),
                65,
                32,
                32,
                true,
                k.leaky_slope,
            );
        }
        obs.layer(67, self.ws.slot(S_T65_32));
        obs.layer(68, self.ws.slot(S_T65_32));
        {
            let (w, b) = conv1_tags(15);
            let (src, dst) = self.ws.get2_mut(S_SKIP12, S_D65_32);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 32, 32);
        }
        obs.layer(69, self.ws.slot(S_D65_32));
        {
            let (dst, src) = self.ws.get2_mut(S_T65_32, S_D65_32);
            add_assign(dst, src);
        }
        obs.layer(70, self.ws.slot(S_T65_32));
        {
            let (w, b) = deconv_tags(1);
            let (src, dst) = self.ws.get2_mut(S_T65_32, S_T129_32);
            S::echo_deconv(dst, src, f32_t(c, w), f32_t(c, b), 65, 32);
        }
        obs.layer(71, self.ws.slot(S_T129_32));
        {
            let (w, b) = conv3_tags(1);
            let (src, dst) = self.ws.get2_mut(S_T129_32, S_T129_16);
            S::echo_conv3(
                dst,
                src,
                f32_t(c, w),
                f32_t(c, b),
                129,
                32,
                16,
                true,
                k.leaky_slope,
            );
        }
        obs.layer(72, self.ws.slot(S_T129_16));
        obs.layer(73, self.ws.slot(S_T129_16));
        {
            let (w, b) = conv1_tags(16);
            let (src, dst) = self.ws.get2_mut(S_SKIP8, S_T129_16B);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 16, 16);
        }
        obs.layer(74, self.ws.slot(S_T129_16B));
        {
            let (dst, src) = self.ws.get2_mut(S_T129_16, S_T129_16B);
            add_assign(dst, src);
        }
        obs.layer(75, self.ws.slot(S_T129_16));
        {
            let (w, b) = deconv_tags(2);
            let (src, dst) = self.ws.get2_mut(S_T129_16, S_T257_16);
            S::echo_deconv(dst, src, f32_t(c, w), f32_t(c, b), 129, 16);
        }
        obs.layer(76, self.ws.slot(S_T257_16));
        {
            let (w, b) = conv3_tags(2);
            let (src, dst) = self.ws.get2_mut(S_T257_16, S_T513_16);
            S::echo_conv3(
                dst,
                src,
                f32_t(c, w),
                f32_t(c, b),
                257,
                16,
                16,
                true,
                k.leaky_slope,
            );
        }
        obs.layer(77, &self.ws.slot(S_T513_16)[..257 * 16]);
        obs.layer(78, &self.ws.slot(S_T513_16)[..257 * 16]);
        {
            let (w, b) = conv1_tags(17);
            let (src, dst) = self.ws.get2_mut(S_SKIP4, S_T129_32);
            S::echo_conv1x1(dst, src, f32_t(c, w), f32_t(c, b), 16, 16);
        }
        obs.layer(79, &self.ws.slot(S_T129_32)[..257 * 16]);
        {
            let (dst, src) = self.ws.get2_mut(S_T513_16, S_T129_32);
            add_assign(dst, src);
        }
        obs.layer(80, &self.ws.slot(S_T513_16)[..257 * 16]);
        {
            let (w, b) = deconv_tags(3);
            let (src, dst) = self.ws.get2_mut(S_T513_16, S_T513_16B);
            S::echo_deconv(dst, src, f32_t(c, w), f32_t(c, b), 257, 16);
        }
        obs.layer(81, self.ws.slot(S_T513_16B));
        {
            let (w, b) = conv3_tags(3);
            let (src, dst) = self.ws.get2_mut(S_T513_16B, S_OUT);
            S::echo_conv3_out1(dst, src, f32_t(c, w), f32_t(c, b), BINS, 16);
        }
        obs.layer(82, self.ws.slot(S_OUT));
        relu_inplace(self.ws.slot_mut(S_OUT));
        obs.layer(83, self.ws.slot(S_OUT));
        obs.layer(84, self.ws.slot(S_OUT));

        // ---- int8 projections 85/87 and the final sigmoid -----------------
        {
            let (dst, src) = self.ws.get2_mut(S_T1056, S_OUT);
            let src = &src[..BINS];
            let qx = self.q.slot_mut(I_QX);
            let a = S::echo_quantize_input(qx, src, k.quant_scale);
            obs.diag_i16("L85_qx", &qx[..BINS]);
            obs.diag_f32("L85_A", &[a]);
            let acc = self.acc32.slot_mut(A_ACC32);
            projection::<S>(
                dst,
                acc,
                i8_t(c, gemm_tag(1, gemm_role::WEIGHT)),
                i8_t(c, gemm_tag(1, gemm_role::ZP)),
                f32_t(c, gemm_tag(1, gemm_role::SCALE)),
                f32_t(c, gemm_tag(1, gemm_role::BIAS)),
                &qx[..BINS],
                BINS,
                BINS,
                a,
            );
        }
        obs.layer(85, &self.ws.slot(S_T1056)[..BINS]);
        relu_inplace(self.ws.slot_mut(S_T1056));
        obs.layer(86, &self.ws.slot(S_T1056)[..BINS]);
        {
            let (dst, src) = self.ws.get2_mut(S_OUT, S_T1056);
            let src = &src[..BINS];
            let qx = self.q.slot_mut(I_QX);
            let a = S::echo_quantize_input(qx, src, k.quant_scale);
            obs.diag_i16("L87_qx", &qx[..BINS]);
            obs.diag_f32("L87_A", &[a]);
            let acc = self.acc32.slot_mut(A_ACC32);
            projection::<S>(
                dst,
                acc,
                i8_t(c, gemm_tag(2, gemm_role::WEIGHT)),
                i8_t(c, gemm_tag(2, gemm_role::ZP)),
                f32_t(c, gemm_tag(2, gemm_role::SCALE)),
                f32_t(c, gemm_tag(2, gemm_role::BIAS)),
                &qx[..BINS],
                BINS,
                BINS,
                a,
            );
        }
        obs.layer(87, self.ws.slot(S_OUT));
        {
            let (src, dst) = self.ws.get2_mut(S_OUT, S_SIG);
            S::echo_sigmoid88(dst, src, &act);
        }
        obs.layer(88, self.ws.slot(S_SIG));
        obs.diag_f32("history", self.ws.slot(S_HISTORY));
        Ok(())
    }

    /// Diagnostics of a post-frame buffer that survives the graph.
    ///
    /// `history` is the 6-row window, `L59`/`L60` are the GRU hidden states,
    /// `L87` is the last projection and `L88`/`out` is the final mask.  Every
    /// other layer is exposed during the frame through [`EchoObserver`].
    pub fn slot(&self, name: &str) -> Option<&[f32]> {
        let idx = match name {
            "history" => S_HISTORY,
            "L59" => S_H59,
            "L60" => S_H60,
            "L87" => S_OUT,
            "L88" | "out" => S_SIG,
            _ => return None,
        };
        Some(self.ws.slot(idx))
    }

    /// Final 513 f32 mask.
    pub fn output(&self) -> &[f32] {
        self.ws.slot(S_SIG)
    }

    /// Container behind the model.
    pub fn container(&self) -> &Container {
        &self.container
    }

    /// Restored scalar constants.
    pub fn consts(&self) -> &EchoConsts {
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
        self.acc32.capacity()
    }

    /// Fixed int64 arena size in elements.
    pub fn workspace_i64(&self) -> usize {
        self.acc64.capacity()
    }
}

// ---------------------------------------------------------------------------
// load-time tensor validation
// ---------------------------------------------------------------------------
fn f32_t(c: &Container, tag: u16) -> &[f32] {
    c.f32(tag).expect("validated at load")
}

fn i8_t(c: &Container, tag: u16) -> &[i8] {
    c.i8(tag).expect("validated at load")
}

fn check_f32(c: &Container, tag: u16, expected: usize) -> Result<(), EchoError> {
    let got = c.f32(tag).ok_or(EchoError::MissingTensor(tag))?.len();
    if got != expected {
        return Err(EchoError::TensorLen { tag, expected, got });
    }
    Ok(())
}

fn check_i8(c: &Container, tag: u16, expected: usize) -> Result<(), EchoError> {
    let got = c.i8(tag).ok_or(EchoError::MissingTensor(tag))?.len();
    if got != expected {
        return Err(EchoError::TensorLen { tag, expected, got });
    }
    Ok(())
}

fn validate_tensors(c: &Container) -> Result<(), EchoError> {
    check_f32(c, TAG_STAGE_SCALE, INPUT_LEN)?;
    check_f32(c, TAG_STAGE_BIAS, INPUT_LEN)?;
    check_f32(c, TAG_HISTORY_INIT, INPUT_LEN)?;
    for (i, &(_, cin, cout)) in CONV1_LAYERS.iter().enumerate() {
        let (w, b) = conv1_tags(i);
        check_f32(c, w, cin * cout)?;
        check_f32(c, b, cout)?;
    }
    for (i, &(_, channels, taps)) in DEPTHWISE_LAYERS.iter().enumerate() {
        let (w, b) = depthwise_tags(i);
        check_f32(c, w, channels * taps)?;
        check_f32(c, b, channels)?;
    }
    check_f32(c, TAG_MM33_A, 33 * 33)?;
    check_f32(c, TAG_MM33_B, 33 * 33)?;
    for (g, &(_, k_in)) in GRU_LAYERS.iter().enumerate() {
        check_i8(c, gru_tag(g, gru_role::W_IN), k_in * GRU_GATES)?;
        check_i8(c, gru_tag(g, gru_role::ZP_IN), GRU_GATES)?;
        check_f32(c, gru_tag(g, gru_role::SCALE_IN), GRU_GATES)?;
        check_i8(c, gru_tag(g, gru_role::W_REC), HIDDEN * GRU_GATES)?;
        check_i8(c, gru_tag(g, gru_role::ZP_REC), GRU_GATES)?;
        check_f32(c, gru_tag(g, gru_role::SCALE_REC), GRU_GATES)?;
        check_f32(c, gru_tag(g, gru_role::BIAS), 2 * GRU_GATES)?;
    }
    for (g, &(_, k_in, n)) in GEMM_LAYERS.iter().enumerate() {
        check_i8(c, gemm_tag(g, gemm_role::WEIGHT), k_in * n)?;
        check_i8(c, gemm_tag(g, gemm_role::ZP), n)?;
        check_f32(c, gemm_tag(g, gemm_role::SCALE), n)?;
        check_f32(c, gemm_tag(g, gemm_role::BIAS), n)?;
    }
    for (i, &(_, channels)) in DECONV_LAYERS.iter().enumerate() {
        let (w, b) = deconv_tags(i);
        check_f32(c, w, channels * 5)?;
        check_f32(c, b, channels)?;
    }
    for (i, &(_, cin, cout)) in CONV3_LAYERS.iter().enumerate() {
        let (w, b) = conv3_tags(i);
        check_f32(c, w, cin * cout * 3)?;
        check_f32(c, b, cout)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// graph plumbing (backend independent: data movement and elementwise copies)
// ---------------------------------------------------------------------------
//
// Every arithmetic kernel lives in `cf_core::Simd`; this file only holds the
// graph, the workspace schedule and the data-movement helpers.

fn stack_input(hist: &[f32], x: &mut [f32], ch0: usize, ch1: usize) {
    for pos in 0..BINS {
        for row in 0..HISTORY_ROWS {
            x[pos * STACK_CHANNELS + row] = hist[(row * INPUT_CHANNELS + ch0) * BINS + pos];
            x[pos * STACK_CHANNELS + HISTORY_ROWS + row] =
                hist[(row * INPUT_CHANNELS + ch1) * BINS + pos];
        }
    }
}

fn leaky_inplace(buf: &mut [f32], slope: f32) {
    for v in buf.iter_mut() {
        if !(*v > 0.0) {
            *v *= slope;
        }
    }
}

fn relu_inplace(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        if !(*v > 0.0) {
            *v = 0.0;
        }
    }
}

fn concat_neck(dst: &mut [f32], a: &[f32], b: &[f32]) {
    // 33 positions x 32 channels each, concatenated on the channel axis.
    debug_assert_eq!(a.len(), 33 * 32);
    debug_assert_eq!(b.len(), 33 * 32);
    debug_assert_eq!(dst.len(), 33 * 64);
    for pos in 0..33 {
        dst[pos * 64..pos * 64 + 32].copy_from_slice(&a[pos * 32..(pos + 1) * 32]);
        dst[pos * 64 + 32..(pos + 1) * 64].copy_from_slice(&b[pos * 32..(pos + 1) * 32]);
    }
}

fn add_assign(dst: &mut [f32], src: &[f32]) {
    // Some layers write into a workspace slot that is larger than their output
    // (for example layer 80's 257 x 16 result lives in the 513 x 16 slot the
    // next deconvolution needs).  Only the source range is defined; the tail of
    // the destination is not part of the layer output.
    let n = dst.len().min(src.len());
    for (d, &s) in dst[..n].iter_mut().zip(src[..n].iter()) {
        *d += s;
    }
}

fn add3(dst: &mut [f32], a: &[f32], b: &[f32]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    for i in 0..dst.len() {
        dst[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// fused graph helpers (backend selected by `S`)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn projection<S: Simd>(
    dst: &mut [f32],
    acc: &mut [i32],
    w: &[i8],
    zp: &[i8],
    scale: &[f32],
    bias: &[f32],
    qx: &[i16],
    k_in: usize,
    n: usize,
    a: f32,
) {
    S::gemm_i8(acc, w, zp, qx, k_in, n);
    for j in 0..n {
        dst[j] = mul_add2(acc[j] as f32, a * scale[j], bias[j]);
    }
}

#[allow(clippy::too_many_arguments)]
fn gru_step<S: Simd, O: EchoObserver>(
    h: &mut [f32],
    y: &mut [f32],
    x: &[f32],
    qx: &mut [i16],
    qh: &mut [i16],
    acc32: &mut [i32],
    acc64: &mut [i64],
    w_in: &[i8],
    zp_in: &[i8],
    scale_in: &[f32],
    w_rec: &[i8],
    zp_rec: &[i8],
    scale_rec: &[f32],
    bias: &[f32],
    k: &EchoConsts,
    diag: &GruDiag,
    obs: &mut O,
) {
    let k_in = x.len();
    debug_assert_eq!(h.len(), HIDDEN);
    debug_assert_eq!(qh.len(), HIDDEN);
    debug_assert_eq!(y.len(), 2 * GRU_GATES);
    debug_assert_eq!(bias.len(), 2 * GRU_GATES);
    let (yih, yhh) = y.split_at_mut(GRU_GATES);

    // Per-frame input scale and quantisation.
    let a_in = S::echo_quantize_input(qx, x, k.quant_scale);
    obs.diag_i16(diag.qx, &qx[..k_in]);
    obs.diag_f32(diag.a_in, &[a_in]);

    // Hidden-state scale, derived from the previous frame's h.
    let a_h = S::echo_quantize_hidden(qh, h, k.hidden_scale, k.hidden_scale_wide);
    obs.diag_i16(diag.qh, qh);
    obs.diag_f32(diag.a_h, &[a_h]);

    // Gate pre-activations.  The first GRU's input width makes the worst-case
    // integer sum exceed i32, so that one GEMM accumulates in i64.
    if k_in > 1024 {
        S::echo_gemm_i64(acc64, w_in, zp_in, &qx[..k_in], k_in, GRU_GATES);
        for j in 0..GRU_GATES {
            yih[j] = mul_add2(acc64[j] as f32, a_in * scale_in[j], bias[j]);
        }
    } else {
        S::gemm_i8(acc32, w_in, zp_in, &qx[..k_in], k_in, GRU_GATES);
        for j in 0..GRU_GATES {
            yih[j] = mul_add2(acc32[j] as f32, a_in * scale_in[j], bias[j]);
        }
    }
    S::gemm_i8(acc32, w_rec, zp_rec, qh, HIDDEN, GRU_GATES);
    for j in 0..GRU_GATES {
        yhh[j] = mul_add2(acc32[j] as f32, a_h * scale_rec[j], bias[GRU_GATES + j]);
    }
    obs.diag_f32(diag.yih, yih);
    obs.diag_f32(diag.yhh, yhh);

    // z = sigmoid(yih[0:H] + yhh[0:H]); r likewise; n = tanh(yhh*r + yih);
    // h = (1 - z) * n + z * h_prev.  The whole tail is one shared kernel so
    // both backends perform the identical sequence per element.
    S::echo_gru_tail(yih, yhh, h, &k.act());
    obs.diag_f32(diag.z, &yih[..HIDDEN]);
    obs.diag_f32(diag.r, &yih[HIDDEN..2 * HIDDEN]);
    obs.diag_f32(diag.n, &yih[2 * HIDDEN..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_bytes() -> &'static [u8] {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/echo-default.cfwm"
        ))
    }

    fn test_frame() -> Vec<f32> {
        let mut frame = vec![0.0f32; INPUT_LEN];
        for (i, v) in frame.iter_mut().enumerate() {
            *v = ((i * 37 % 101) as f32) * 0.01 - 0.5;
        }
        frame
    }

    struct Collector {
        layers: Vec<(usize, usize)>,
        diag: Vec<(&'static str, usize)>,
        diag_i16: Vec<(&'static str, usize)>,
    }

    impl EchoObserver for Collector {
        fn layer(&mut self, index: usize, values: &[f32]) {
            self.layers.push((index, values.len()));
        }
        fn diag_f32(&mut self, name: &'static str, values: &[f32]) {
            self.diag.push((name, values.len()));
        }
        fn diag_i16(&mut self, name: &'static str, values: &[i16]) {
            self.diag_i16.push((name, values.len()));
        }
    }

    #[test]
    fn layer_widths_sum_to_the_verification_surface() {
        assert_eq!(LAYER_LEN.len(), LAYERS);
        assert_eq!(LAYER_LEN.iter().sum::<usize>(), 247_351);
        assert_eq!(LAYER_LEN[0], 513 * STACK_CHANNELS);
        assert_eq!(LAYER_LEN[88], OUTPUT_LEN);
    }

    #[test]
    fn loads_container_and_runs_frames() {
        let mut m = EchoModel::new(model_bytes()).unwrap();
        assert_eq!(m.container().model_id(), MODEL_ECHO_DEFAULT);
        assert_eq!(m.container().tensor_count(), 104);
        assert_eq!(m.output().len(), OUTPUT_LEN);
        m.process(&test_frame()).unwrap();
        assert!(m.output().iter().all(|v| v.is_finite()));
        assert_eq!(m.slot("L88").unwrap().len(), OUTPUT_LEN);
        assert_eq!(m.slot("history").unwrap().len(), HISTORY_LEN);
        assert!(m.process(&[0.0f32; 3]).is_err());
    }

    #[test]
    fn observer_sees_every_layer() {
        let mut m = EchoModel::new(model_bytes()).unwrap();
        let mut obs = Collector {
            layers: Vec::new(),
            diag: Vec::new(),
            diag_i16: Vec::new(),
        };
        m.process_with(&test_frame(), &mut obs).unwrap();
        assert_eq!(obs.layers.len(), LAYERS);
        for (i, &(index, len)) in obs.layers.iter().enumerate() {
            assert_eq!(index, i, "layer order");
            assert_eq!(len, LAYER_LEN[i], "layer {i} width");
        }
        assert!(obs.diag.iter().any(|&(n, _)| n == "staged"));
        assert!(obs.diag.iter().any(|&(n, _)| n == "history"));
        assert!(obs.diag.iter().any(|&(n, _)| n == "L59_yih"));
        assert!(obs.diag_i16.iter().any(|&(n, _)| n == "L59_qx"));
        assert!(obs.diag_i16.iter().any(|&(n, _)| n == "L87_qx"));
    }

    #[test]
    fn state_persists_and_resets() {
        let mut m = EchoModel::new(model_bytes()).unwrap();
        let frame = test_frame();
        m.process(&frame).unwrap();
        let after_one = m.output().to_vec();
        m.reset();
        m.process(&frame).unwrap();
        assert_eq!(after_one, m.output());
        m.process(&frame).unwrap();
        assert_ne!(after_one, m.output());
    }
}
