use std::ffi::c_char;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

use cf_core::cfwm::Container;
use cf_core::Backend;
use cf_echo::{EchoModel, INPUT_LEN as ECHO_INPUT_LEN, OUTPUT_LEN as ECHO_OUTPUT_LEN};
use cf_voice::{VoiceModel, INPUT_LEN as VOICE_INPUT_LEN, L18_LEN as VOICE_OUTPUT_LEN};

/// ABI revision reported by [`cf_abi_version`].  Incremented only for an
/// incompatible signature/semantic change.
pub const CF_ABI_VERSION: u32 = 1;

/// Model id of the voice (denoise) container.
pub const CF_VOICE_MODEL_ID: u32 = 1;
/// Model id of the echo (cancellation) container.
pub const CF_ECHO_MODEL_ID: u32 = 2;

/// Backend selector accepted by `*_new_with_backend`.
pub const CF_BACKEND_AUTO: u32 = 0;
/// Portable scalar reference backend.
pub const CF_BACKEND_SCALAR: u32 = 1;
/// AVX2 + FMA backend.
pub const CF_BACKEND_AVX2: u32 = 2;

/// Success.
pub const CF_OK: u32 = 0;
/// Bad argument: null model pointer, zero length, length not a multiple of 4,
/// or unknown backend code.
pub const CF_ERR_ARG: u32 = 1;
/// The model bytes are not a valid container.
pub const CF_ERR_CONTAINER: u32 = 2;
/// The container is not the model this constructor requires (wrong id),
/// or model validation failed.
pub const CF_ERR_MODEL: u32 = 3;
/// Null handle, null frame buffer, wrong handle type for a typed call, or a
/// failed frame (e.g. the caller's buffer was too small - see the lengths).
pub const CF_ERR_STATE: u32 = 4;
/// The requested backend is not available on this CPU.
pub const CF_ERR_BACKEND: u32 = 5;

/// Fixed workspace footprint of the voice model (f32 + i16 + i32 arenas).
pub const CF_VOICE_WORKSPACE_BYTES: u32 = 432_192 + 4_096 + 4_128;
/// Fixed workspace footprint of the echo model (f32 + i16 + i32 + i64 arenas).
pub const CF_ECHO_WORKSPACE_BYTES: u32 = 85_768 * 4 + 2_240 * 2 + 1_056 * 4 + 384 * 8;

const NAME_SCALAR: &[u8] = b"scalar\0";
const NAME_AVX2: &[u8] = b"avx2\0";

/// Last failure code, or 0.  Process-global (see the module docs).
static LAST_ERROR: AtomicU32 = AtomicU32::new(0);

#[inline]
fn fail(code: u32) -> *mut cf_model {
    LAST_ERROR.store(code, Ordering::Relaxed);
    ptr::null_mut()
}

#[inline]
fn fail_i32(code: u32) -> i32 {
    LAST_ERROR.store(code, Ordering::Relaxed);
    -(code as i32)
}

// ---------------------------------------------------------------------------
// handle
// ---------------------------------------------------------------------------

/// Opaque model handle (`typedef struct cf_model cf_model;` in the header).
///
/// Both the typed constructors (`cf_voice_new`, `cf_echo_new`) and the unified
/// constructor (`cf_model_new`) return this type.  The tag is checked by every
/// typed call.
#[repr(C)]
pub struct cf_model {
    tag: u32,
    inner: Inner,
}

enum Inner {
    Voice(Box<VoiceModel>),
    Echo(Box<EchoModel>),
}

impl cf_model {
    #[inline]
    fn input_len(&self) -> usize {
        match self.tag {
            CF_VOICE_MODEL_ID => VOICE_INPUT_LEN,
            _ => ECHO_INPUT_LEN,
        }
    }

    #[inline]
    fn output_len(&self) -> usize {
        match self.tag {
            CF_VOICE_MODEL_ID => VOICE_OUTPUT_LEN,
            _ => ECHO_OUTPUT_LEN,
        }
    }

    #[inline]
    fn backend(&self) -> Backend {
        match &self.inner {
            Inner::Voice(m) => m.backend(),
            Inner::Echo(m) => m.backend(),
        }
    }

    #[inline]
    fn workspace_bytes(&self) -> u32 {
        match &self.inner {
            Inner::Voice(m) => {
                (m.workspace_floats() * 4 + m.workspace_i16() * 2 + m.workspace_i32() * 4) as u32
            }
            Inner::Echo(m) => {
                (m.workspace_floats() * 4
                    + m.workspace_i16() * 2
                    + m.workspace_i32() * 4
                    + m.workspace_i64() * 8) as u32
            }
        }
    }

    #[inline]
    fn model_bytes(&self) -> u32 {
        match &self.inner {
            Inner::Voice(m) => m.container().file_len() as u32,
            Inner::Echo(m) => m.container().file_len() as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------

#[inline]
fn backend_from_code(code: u32) -> Result<Backend, u32> {
    match code {
        CF_BACKEND_AUTO => Ok(Backend::detect()),
        CF_BACKEND_SCALAR => Ok(Backend::Scalar),
        CF_BACKEND_AVX2 => Ok(Backend::Avx2),
        _ => Err(CF_ERR_ARG),
    }
}

/// Copy a host container image into an aligned word buffer and parse it.
unsafe fn container_from_memory(bytes: *const u8, len: usize) -> Result<Container, u32> {
    if bytes.is_null() || len == 0 || !len.is_multiple_of(4) {
        return Err(CF_ERR_ARG);
    }
    // SAFETY: the caller guarantees `bytes` is readable for `len` bytes.
    let src = unsafe { std::slice::from_raw_parts(bytes, len) };
    let mut words: Vec<u32> = Vec::with_capacity(len / 4);
    for chunk in src.as_chunks::<4>().0 {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Container::parse_owned_words(words).map_err(|_| CF_ERR_CONTAINER)
}

unsafe fn build(model_id: u32, bytes: *const u8, len: usize, backend_code: u32) -> *mut cf_model {
    LAST_ERROR.store(CF_OK, Ordering::Relaxed);
    if model_id != CF_VOICE_MODEL_ID && model_id != CF_ECHO_MODEL_ID {
        return fail(CF_ERR_ARG);
    }
    let backend = match backend_from_code(backend_code) {
        Ok(b) => b,
        Err(code) => return fail(code),
    };
    if !backend.is_available() {
        return fail(CF_ERR_BACKEND);
    }
    let container = match unsafe { container_from_memory(bytes, len) } {
        Ok(c) => c,
        Err(code) => return fail(code),
    };
    if u32::from(container.model_id()) != model_id {
        return fail(CF_ERR_MODEL);
    }
    let inner = match model_id {
        CF_VOICE_MODEL_ID => match VoiceModel::from_container(container, backend) {
            Ok(m) => Inner::Voice(Box::new(m)),
            Err(_) => return fail(CF_ERR_MODEL),
        },
        CF_ECHO_MODEL_ID => match EchoModel::from_container_with(container, backend) {
            Ok(m) => Inner::Echo(Box::new(m)),
            Err(_) => return fail(CF_ERR_MODEL),
        },
        _ => return fail(CF_ERR_ARG),
    };
    Box::into_raw(Box::new(cf_model {
        tag: model_id,
        inner,
    }))
}

/// ABI revision (currently 1).
#[no_mangle]
pub extern "C" fn cf_abi_version() -> u32 {
    CF_ABI_VERSION
}

/// Last error code set by a fallible call (0 = none).
#[no_mangle]
pub extern "C" fn cf_last_error() -> u32 {
    LAST_ERROR.load(Ordering::Relaxed)
}

/// Create a model selected by container model id (1 = voice, 2 = echo).
///
/// `model_bytes` must hold exactly `len` bytes of a `.cfwm` container; the
/// bytes are copied.  The backend is `CF_BACKEND_AUTO`.
///
/// Returns null on failure (see [`cf_last_error`]).
///
/// # Safety
///
/// `model_bytes` must be readable for `len` bytes.  It is not retained.
#[no_mangle]
pub unsafe extern "C" fn cf_model_new(
    model_id: u32,
    model_bytes: *const u8,
    len: usize,
) -> *mut cf_model {
    unsafe { build(model_id, model_bytes, len, CF_BACKEND_AUTO) }
}

/// Create a model with an explicit backend (`0` auto, `1` scalar, `2` avx2).
///
/// A requested backend that the CPU cannot run fails with code 5; there is no
/// silent fallback.
///
/// # Safety
///
/// Same contract as [`cf_model_new`].
#[no_mangle]
pub unsafe extern "C" fn cf_model_new_with_backend(
    model_id: u32,
    model_bytes: *const u8,
    len: usize,
    backend: u32,
) -> *mut cf_model {
    unsafe { build(model_id, model_bytes, len, backend) }
}

/// Create a voice model from container bytes (`CF_BACKEND_AUTO`).
///
/// # Safety
///
/// `model_bytes` must be readable for `len` bytes.  It is not retained.
#[no_mangle]
pub unsafe extern "C" fn cf_voice_new(model_bytes: *const u8, len: usize) -> *mut cf_model {
    unsafe { build(CF_VOICE_MODEL_ID, model_bytes, len, CF_BACKEND_AUTO) }
}

/// Create a voice model with an explicit backend.
///
/// # Safety
///
/// Same contract as [`cf_voice_new`].
#[no_mangle]
pub unsafe extern "C" fn cf_voice_new_with_backend(
    model_bytes: *const u8,
    len: usize,
    backend: u32,
) -> *mut cf_model {
    unsafe { build(CF_VOICE_MODEL_ID, model_bytes, len, backend) }
}

/// Create an echo model from container bytes (`CF_BACKEND_AUTO`).
///
/// # Safety
///
/// `model_bytes` must be readable for `len` bytes.  It is not retained.
#[no_mangle]
pub unsafe extern "C" fn cf_echo_new(model_bytes: *const u8, len: usize) -> *mut cf_model {
    unsafe { build(CF_ECHO_MODEL_ID, model_bytes, len, CF_BACKEND_AUTO) }
}

/// Create an echo model with an explicit backend.
///
/// # Safety
///
/// Same contract as [`cf_echo_new`].
#[no_mangle]
pub unsafe extern "C" fn cf_echo_new_with_backend(
    model_bytes: *const u8,
    len: usize,
    backend: u32,
) -> *mut cf_model {
    unsafe { build(CF_ECHO_MODEL_ID, model_bytes, len, backend) }
}

// ---------------------------------------------------------------------------
// processing
// ---------------------------------------------------------------------------

unsafe fn process_impl(
    handle: *mut cf_model,
    input: *const f32,
    output: *mut f32,
    expected_id: u32,
) -> i32 {
    LAST_ERROR.store(CF_OK, Ordering::Relaxed);
    if handle.is_null() || input.is_null() || output.is_null() {
        return fail_i32(CF_ERR_STATE);
    }
    let model = unsafe { &mut *handle };
    if expected_id != 0 && model.tag != expected_id {
        return fail_i32(CF_ERR_MODEL);
    }
    let in_len = model.input_len();
    let out_len = model.output_len();
    // SAFETY: the caller guarantees `input`/`output` are valid for the
    // documented lengths and do not overlap.
    let frame = unsafe { std::slice::from_raw_parts(input, in_len) };
    let dst = unsafe { std::slice::from_raw_parts_mut(output, out_len) };
    match &mut model.inner {
        Inner::Voice(m) => {
            if m.process(frame).is_err() {
                return fail_i32(CF_ERR_STATE);
            }
            dst.copy_from_slice(m.output());
        }
        Inner::Echo(m) => {
            if m.process(frame).is_err() {
                return fail_i32(CF_ERR_STATE);
            }
            dst.copy_from_slice(m.output());
        }
    }
    CF_OK as i32
}

/// Run one voice frame: 513 `f32` in, 1026 `f32` out.  No allocation.
///
/// Returns 0 on success, a negated error code otherwise.
///
/// # Safety
///
/// `handle` must come from `cf_voice_new*`, `input` must be readable for 513
/// floats and `output` writable for 1026 floats; the buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn cf_voice_process(
    handle: *mut cf_model,
    input: *const f32,
    output: *mut f32,
) -> i32 {
    unsafe { process_impl(handle, input, output, CF_VOICE_MODEL_ID) }
}

/// Run one echo frame: 1539 `f32` in, 513 `f32` out.  No allocation.
///
/// # Safety
///
/// `handle` must come from `cf_echo_new*`, `input` must be readable for 1539
/// floats and `output` writable for 513 floats; the buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn cf_echo_process(
    handle: *mut cf_model,
    input: *const f32,
    output: *mut f32,
) -> i32 {
    unsafe { process_impl(handle, input, output, CF_ECHO_MODEL_ID) }
}

/// Run one frame through a handle returned by any constructor.
///
/// # Safety
///
/// `input`/`output` must be valid for the lengths reported by
/// [`cf_model_input_len`] / [`cf_model_output_len`] and must not overlap.
#[no_mangle]
pub unsafe extern "C" fn cf_model_process(
    handle: *mut cf_model,
    input: *const f32,
    output: *mut f32,
) -> i32 {
    unsafe { process_impl(handle, input, output, 0) }
}

unsafe fn reset_impl(handle: *mut cf_model, expected_id: u32) -> i32 {
    LAST_ERROR.store(CF_OK, Ordering::Relaxed);
    if handle.is_null() {
        return fail_i32(CF_ERR_STATE);
    }
    let model = unsafe { &mut *handle };
    if expected_id != 0 && model.tag != expected_id {
        return fail_i32(CF_ERR_MODEL);
    }
    match &mut model.inner {
        Inner::Voice(m) => m.reset(),
        Inner::Echo(m) => m.reset(),
    }
    CF_OK as i32
}

/// Reset the voice cross-frame state (history, LSTM cells, quantised state).
#[no_mangle]
pub unsafe extern "C" fn cf_voice_reset(handle: *mut cf_model) -> i32 {
    unsafe { reset_impl(handle, CF_VOICE_MODEL_ID) }
}

/// Reset the echo cross-frame state (six history rows, GRU states, quantised
/// arenas).
#[no_mangle]
pub unsafe extern "C" fn cf_echo_reset(handle: *mut cf_model) -> i32 {
    unsafe { reset_impl(handle, CF_ECHO_MODEL_ID) }
}

/// Reset a handle returned by any constructor.
#[no_mangle]
pub unsafe extern "C" fn cf_model_reset(handle: *mut cf_model) -> i32 {
    unsafe { reset_impl(handle, 0) }
}

unsafe fn free_impl(handle: *mut cf_model) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Release a voice handle.
#[no_mangle]
pub unsafe extern "C" fn cf_voice_free(handle: *mut cf_model) {
    unsafe { free_impl(handle) }
}

/// Release an echo handle.
#[no_mangle]
pub unsafe extern "C" fn cf_echo_free(handle: *mut cf_model) {
    unsafe { free_impl(handle) }
}

/// Release a handle returned by any constructor.
#[no_mangle]
pub unsafe extern "C" fn cf_model_free(handle: *mut cf_model) {
    unsafe { free_impl(handle) }
}

// ---------------------------------------------------------------------------
// queries
// ---------------------------------------------------------------------------

/// Container model id of a handle (0 on null).
#[no_mangle]
pub unsafe extern "C" fn cf_model_id(handle: *const cf_model) -> u32 {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).tag }
}

/// Input feature count of a handle (0 on null).
#[no_mangle]
pub unsafe extern "C" fn cf_model_input_len(handle: *const cf_model) -> u32 {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).input_len() as u32 }
}

/// Output value count of a handle (0 on null).
#[no_mangle]
pub unsafe extern "C" fn cf_model_output_len(handle: *const cf_model) -> u32 {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).output_len() as u32 }
}

/// Live backend: 1 = scalar, 2 = avx2, 0 = null handle.
#[no_mangle]
pub unsafe extern "C" fn cf_model_backend(handle: *const cf_model) -> u32 {
    if handle.is_null() {
        return 0;
    }
    match unsafe { (*handle).backend() } {
        Backend::Scalar => CF_BACKEND_SCALAR,
        Backend::Avx2 => CF_BACKEND_AVX2,
    }
}

/// Live backend name as a NUL-terminated static string, or null.
#[no_mangle]
pub unsafe extern "C" fn cf_model_backend_name(handle: *const cf_model) -> *const c_char {
    if handle.is_null() {
        return ptr::null();
    }
    match unsafe { (*handle).backend() } {
        Backend::Scalar => NAME_SCALAR.as_ptr() as *const c_char,
        Backend::Avx2 => NAME_AVX2.as_ptr() as *const c_char,
    }
}

/// Resident container bytes of a handle (0 on null).
#[no_mangle]
pub unsafe extern "C" fn cf_model_model_bytes(handle: *const cf_model) -> u32 {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).model_bytes() }
}

/// Fixed workspace footprint in bytes (0 on null).
#[no_mangle]
pub unsafe extern "C" fn cf_model_workspace_bytes(handle: *const cf_model) -> u32 {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).workspace_bytes() }
}

/// Voice input feature count per frame (513).
#[no_mangle]
pub extern "C" fn cf_voice_input_len() -> u32 {
    VOICE_INPUT_LEN as u32
}

/// Voice output value count per frame (1026).
#[no_mangle]
pub extern "C" fn cf_voice_output_len() -> u32 {
    VOICE_OUTPUT_LEN as u32
}

/// Voice container model id (1).
#[no_mangle]
pub extern "C" fn cf_voice_model_id() -> u32 {
    CF_VOICE_MODEL_ID
}

/// Voice model workspace footprint in bytes.
#[no_mangle]
pub extern "C" fn cf_voice_workspace_bytes() -> u32 {
    CF_VOICE_WORKSPACE_BYTES
}

/// Resident container bytes of a voice handle (0 on null or wrong type).
#[no_mangle]
pub unsafe extern "C" fn cf_voice_model_bytes(handle: *const cf_model) -> u32 {
    if handle.is_null() || unsafe { (*handle).tag } != CF_VOICE_MODEL_ID {
        return 0;
    }
    unsafe { (*handle).model_bytes() }
}

/// Echo input feature count per frame (3 x 513 = 1539).
#[no_mangle]
pub extern "C" fn cf_echo_input_len() -> u32 {
    ECHO_INPUT_LEN as u32
}

/// Echo output value count per frame (513).
#[no_mangle]
pub extern "C" fn cf_echo_output_len() -> u32 {
    ECHO_OUTPUT_LEN as u32
}

/// Echo container model id (2).
#[no_mangle]
pub extern "C" fn cf_echo_model_id() -> u32 {
    CF_ECHO_MODEL_ID
}

/// Echo model workspace footprint in bytes.
#[no_mangle]
pub extern "C" fn cf_echo_workspace_bytes() -> u32 {
    CF_ECHO_WORKSPACE_BYTES
}

/// Resident container bytes of an echo handle (0 on null or wrong type).
#[no_mangle]
pub unsafe extern "C" fn cf_echo_model_bytes(handle: *const cf_model) -> u32 {
    if handle.is_null() || unsafe { (*handle).tag } != CF_ECHO_MODEL_ID {
        return 0;
    }
    unsafe { (*handle).model_bytes() }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    // LAST_ERROR is process-global, so the tests that inspect it must not run
    // concurrently with each other.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    use super::*;

    const VOICE_BYTES: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/voice-default.cfwm"
    ));
    const ECHO_BYTES: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/echo-default.cfwm"
    ));

    fn voice_frame() -> Vec<f32> {
        (0..VOICE_INPUT_LEN)
            .map(|i| ((i * 37 % 101) as f32) * 0.01 - 0.5)
            .collect()
    }

    fn echo_frame() -> Vec<f32> {
        (0..ECHO_INPUT_LEN)
            .map(|i| ((i * 37 % 211) as f32) * 0.01 - 1.0)
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn abi_version_and_static_lengths() {
        let _guard = TEST_LOCK.lock().unwrap();
        assert_eq!(cf_abi_version(), 1);
        assert_eq!(cf_voice_input_len(), 513);
        assert_eq!(cf_voice_output_len(), 1026);
        assert_eq!(cf_echo_input_len(), 1539);
        assert_eq!(cf_echo_output_len(), 513);
        assert_eq!(cf_voice_model_id(), 1);
        assert_eq!(cf_echo_model_id(), 2);
        assert_eq!(cf_voice_workspace_bytes(), CF_VOICE_WORKSPACE_BYTES);
        assert_eq!(cf_echo_workspace_bytes(), CF_ECHO_WORKSPACE_BYTES);
    }

    #[test]
    fn voice_typed_and_unified_paths_are_bit_identical() {
        let _guard = TEST_LOCK.lock().unwrap();
        let frame = voice_frame();
        let typed = unsafe { cf_voice_new(VOICE_BYTES.as_ptr(), VOICE_BYTES.len()) };
        assert!(!typed.is_null(), "last error {}", cf_last_error());
        let unified =
            unsafe { cf_model_new(CF_VOICE_MODEL_ID, VOICE_BYTES.as_ptr(), VOICE_BYTES.len()) };
        assert!(!unified.is_null(), "last error {}", cf_last_error());
        assert_eq!(unsafe { cf_model_id(typed) }, 1);
        assert_eq!(unsafe { cf_model_input_len(typed) }, 513);
        assert_eq!(unsafe { cf_model_output_len(typed) }, 1026);
        assert_eq!(
            unsafe { cf_model_workspace_bytes(typed) },
            CF_VOICE_WORKSPACE_BYTES
        );
        assert_eq!(
            unsafe { cf_model_model_bytes(typed) } as usize,
            VOICE_BYTES.len()
        );
        assert_eq!(
            unsafe { cf_model_backend(typed) },
            match Backend::detect() {
                Backend::Scalar => 1,
                Backend::Avx2 => 2,
            }
        );

        let mut out_typed = vec![0.0f32; VOICE_OUTPUT_LEN];
        let mut out_unified = vec![0.0f32; VOICE_OUTPUT_LEN];
        unsafe {
            assert_eq!(
                cf_voice_process(typed, frame.as_ptr(), out_typed.as_mut_ptr()),
                0
            );
            assert_eq!(
                cf_model_process(unified, frame.as_ptr(), out_unified.as_mut_ptr()),
                0
            );
            assert_eq!(bits(&out_typed), bits(&out_unified));
            assert_eq!(cf_voice_reset(typed), 0);
            assert_eq!(cf_model_reset(unified), 0);
            cf_voice_free(typed);
            cf_model_free(unified);
        }
    }

    #[test]
    fn voice_matches_the_direct_model_bits() {
        let _guard = TEST_LOCK.lock().unwrap();
        let frame = voice_frame();
        let mut reference =
            VoiceModel::with_backend(VOICE_BYTES, Backend::detect()).expect("voice model");
        reference.process(&frame).expect("process");
        let handle = unsafe { cf_voice_new(VOICE_BYTES.as_ptr(), VOICE_BYTES.len()) };
        assert!(!handle.is_null());
        let mut out = vec![0.0f32; VOICE_OUTPUT_LEN];
        unsafe {
            assert_eq!(
                cf_voice_process(handle, frame.as_ptr(), out.as_mut_ptr()),
                0
            );
            cf_voice_free(handle);
        }
        let want = bits(reference.output());
        assert_eq!(bits(&out), want);
    }

    #[test]
    fn echo_typed_and_unified_paths_are_bit_identical() {
        let _guard = TEST_LOCK.lock().unwrap();
        let frame = echo_frame();
        let typed = unsafe { cf_echo_new(ECHO_BYTES.as_ptr(), ECHO_BYTES.len()) };
        assert!(!typed.is_null(), "last error {}", cf_last_error());
        let unified =
            unsafe { cf_model_new(CF_ECHO_MODEL_ID, ECHO_BYTES.as_ptr(), ECHO_BYTES.len()) };
        assert!(!unified.is_null(), "last error {}", cf_last_error());
        assert_eq!(unsafe { cf_model_id(typed) }, 2);
        assert_eq!(unsafe { cf_model_input_len(typed) }, 1539);
        assert_eq!(unsafe { cf_model_output_len(typed) }, 513);
        assert_eq!(
            unsafe { cf_model_workspace_bytes(typed) },
            CF_ECHO_WORKSPACE_BYTES
        );

        let mut out_typed = vec![0.0f32; ECHO_OUTPUT_LEN];
        let mut out_unified = vec![0.0f32; ECHO_OUTPUT_LEN];
        unsafe {
            assert_eq!(
                cf_echo_process(typed, frame.as_ptr(), out_typed.as_mut_ptr()),
                0
            );
            assert_eq!(
                cf_model_process(unified, frame.as_ptr(), out_unified.as_mut_ptr()),
                0
            );
            assert_eq!(bits(&out_typed), bits(&out_unified));
            assert_eq!(cf_echo_reset(typed), 0);
            assert_eq!(cf_echo_reset(unified), 0);
            cf_echo_free(typed);
            cf_echo_free(unified);
        }
    }

    #[test]
    fn echo_matches_the_direct_avx2_model_bits() {
        let _guard = TEST_LOCK.lock().unwrap();
        let frame = echo_frame();
        let mut reference =
            EchoModel::with_backend(ECHO_BYTES, Backend::Avx2).expect("avx2 echo model");
        reference.process(&frame).expect("process");
        let handle = unsafe {
            cf_echo_new_with_backend(ECHO_BYTES.as_ptr(), ECHO_BYTES.len(), CF_BACKEND_AVX2)
        };
        assert!(!handle.is_null(), "last error {}", cf_last_error());
        assert_eq!(unsafe { cf_model_backend(handle) }, CF_BACKEND_AVX2);
        let name = unsafe { std::ffi::CStr::from_ptr(cf_model_backend_name(handle)) };
        assert_eq!(name.to_str().unwrap(), "avx2");
        let mut out = vec![0.0f32; ECHO_OUTPUT_LEN];
        unsafe {
            assert_eq!(cf_echo_process(handle, frame.as_ptr(), out.as_mut_ptr()), 0);
            cf_echo_free(handle);
        }
        assert_eq!(bits(&out), bits(reference.output()));
    }

    #[test]
    fn scalar_backend_selection_is_honoured() {
        let _guard = TEST_LOCK.lock().unwrap();
        let frame = voice_frame();
        let handle = unsafe {
            cf_voice_new_with_backend(VOICE_BYTES.as_ptr(), VOICE_BYTES.len(), CF_BACKEND_SCALAR)
        };
        assert!(!handle.is_null(), "last error {}", cf_last_error());
        assert_eq!(unsafe { cf_model_backend(handle) }, CF_BACKEND_SCALAR);
        let mut out = vec![0.0f32; VOICE_OUTPUT_LEN];
        unsafe {
            assert_eq!(
                cf_voice_process(handle, frame.as_ptr(), out.as_mut_ptr()),
                0
            );
            cf_voice_free(handle);
        }
    }

    #[test]
    fn rejects_bad_arguments_and_wrong_types() {
        let _guard = TEST_LOCK.lock().unwrap();
        assert!(unsafe { cf_voice_new(ptr::null(), 0) }.is_null());
        assert_eq!(cf_last_error(), CF_ERR_ARG);
        assert!(unsafe { cf_voice_new(VOICE_BYTES.as_ptr(), VOICE_BYTES.len() - 1) }.is_null());
        assert_eq!(cf_last_error(), CF_ERR_ARG);
        assert!(
            unsafe { cf_model_new(CF_VOICE_MODEL_ID, ECHO_BYTES.as_ptr(), ECHO_BYTES.len()) }
                .is_null()
        );
        assert_eq!(cf_last_error(), CF_ERR_MODEL);
        assert!(unsafe { cf_model_new(7, VOICE_BYTES.as_ptr(), VOICE_BYTES.len()) }.is_null());
        assert_eq!(cf_last_error(), CF_ERR_ARG);

        // A typed call on the other model's handle fails, and does not corrupt
        // the handle.
        let echo = unsafe { cf_echo_new(ECHO_BYTES.as_ptr(), ECHO_BYTES.len()) };
        assert!(!echo.is_null());
        let frame = echo_frame();
        let mut out = vec![0.0f32; ECHO_OUTPUT_LEN];
        assert_eq!(
            unsafe { cf_voice_process(echo, frame.as_ptr(), out.as_mut_ptr()) },
            -(CF_ERR_MODEL as i32)
        );
        assert_eq!(unsafe { cf_model_id(echo) }, 2);
        assert_eq!(
            unsafe { cf_model_process(ptr::null_mut(), ptr::null(), ptr::null_mut()) },
            -(CF_ERR_STATE as i32)
        );
        unsafe { cf_model_free(echo) };
    }
}
