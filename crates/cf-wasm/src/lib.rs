use std::alloc::{alloc, dealloc, Layout};
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

use cf_core::fma;
use cf_core::Backend;
use cf_voice::{Container, VoiceModel, DIAG_F32_SLOTS, DIAG_I16_SLOTS, INPUT_LEN, L18_LEN};

/// An embedded container, forced to 8-byte alignment so the `u32`/`f32`
/// tensor views are aligned and the `Container` can borrow it with no copy.
#[repr(C, align(8))]
struct AlignedImage<const N: usize>([u8; N]);

#[cfg(feature = "embedded-voice")]
static VOICE_IMAGE: AlignedImage<
    {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/voice-default.cfwm"
        ))
        .len()
    },
> = AlignedImage(*include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/voice-default.cfwm"
)));

/// The embedded voice container bytes (empty when not compiled in).
fn voice_image() -> &'static [u8] {
    #[cfg(feature = "embedded-voice")]
    {
        &VOICE_IMAGE.0
    }
    #[cfg(not(feature = "embedded-voice"))]
    {
        &[]
    }
}

/// Fixed workspace footprint in bytes (f32 arena + i16 arena + i32 arena).
///
/// `cf_voice_workspace_bytes` returns this value; the unit tests below assert
/// that a freshly loaded model still reports exactly these capacities.
const WORKSPACE_BYTES: u32 = 432_192 + 4_096 + 4_128;

/// Alignment of the workspace slot layout (8 floats).
const WORKSPACE_ALIGN: u32 = 32;

/// Last failure code, or 0 after a successful call.  Per module instance.
static LAST_ERROR: AtomicU32 = AtomicU32::new(0);

#[inline]
fn fail(code: u32) -> *mut c_void {
    LAST_ERROR.store(code, Ordering::Relaxed);
    ptr::null_mut()
}

#[inline]
fn fail_i32(code: u32) -> i32 {
    LAST_ERROR.store(code, Ordering::Relaxed);
    -(code as i32)
}

/// Last failure code from a fallible entry point.
///
/// [`cf_voice_new`] resets it to 0 on entry; other successful calls do not
/// clear a previous failure.  Read it immediately after a null/negative
/// return.
///
/// * 0 - no error
/// * 1 - bad model pointer / length
/// * 2 - container parse failure
/// * 3 - backend unavailable or model validation failure
/// * 4 - null handle / bad frame pointer
#[no_mangle]
pub extern "C" fn cf_last_error() -> u32 {
    LAST_ERROR.load(Ordering::Relaxed)
}

/// Container length of the embedded default model, in bytes.
#[no_mangle]
pub extern "C" fn cf_voice_model_image_len() -> u32 {
    voice_image().len() as u32
}

/// Workspace footprint in bytes (f32, i16 and i32 arenas).
#[no_mangle]
pub extern "C" fn cf_voice_workspace_bytes() -> u32 {
    WORKSPACE_BYTES
}

/// Recommended workspace alignment in bytes.
#[no_mangle]
pub extern "C" fn cf_voice_workspace_align() -> u32 {
    WORKSPACE_ALIGN
}

/// Input feature count per frame.
#[no_mangle]
pub extern "C" fn cf_voice_input_len() -> u32 {
    INPUT_LEN as u32
}

/// Output value count per frame.
#[no_mangle]
pub extern "C" fn cf_voice_output_len() -> u32 {
    L18_LEN as u32
}

/// The only backend the wasm build can run.
#[no_mangle]
pub extern "C" fn cf_voice_backend_is_scalar() -> u32 {
    1
}

/// Backend name of a live model (UTF-8, not NUL terminated).
#[no_mangle]
pub unsafe extern "C" fn cf_voice_backend_name_ptr(handle: *mut c_void) -> *const u8 {
    if handle.is_null() {
        return ptr::null();
    }
    let model = &*(handle as *const VoiceModel);
    model.backend_name().as_ptr()
}

/// Backend name length of a live model.
#[no_mangle]
pub unsafe extern "C" fn cf_voice_backend_name_len(handle: *mut c_void) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let model = &*(handle as *const VoiceModel);
    model.backend_name().len() as u32
}

/// Build a container from the embedded image or a host-owned word buffer.
///
/// Error codes match [`cf_last_error`]: 1 bad pointer/length, 2 parse failure,
/// 5 no image of that model was compiled into this module.
unsafe fn container_for(
    model_ptr: *mut u32,
    len: usize,
    image: &'static [u8],
) -> Result<Container, u32> {
    if len == 0 {
        if image.is_empty() {
            return Err(5);
        }
        return Container::parse_static(image).map_err(|_| 2);
    }
    if model_ptr.is_null() || !len.is_multiple_of(4) {
        return Err(1);
    }
    let count = len / 4;
    // SAFETY: documented ABI - the pointer comes from `cf_alloc_u32(count)`
    // and ownership of the exact allocation is transferred to the engine.
    let words: Vec<u32> = unsafe { Vec::from_raw_parts(model_ptr, count, count) };
    Container::parse_owned_words(words).map_err(|_| 2)
}

/// Create a model.
///
/// * `model_ptr = 0, len = 0`: use the embedded default container.
/// * `model_ptr != 0, len > 0`: take ownership of a buffer returned by
///   [`cf_alloc_u32`] holding `len` container bytes (`len` must be a multiple
///   of four).
///
/// Returns an opaque handle, or null on failure (see [`cf_last_error`]).
///
/// # Safety
///
/// With `len > 0`, `model_ptr` must point to `len / 4` writable, initialised
/// `u32` words allocated by [`cf_alloc_u32`] with exactly that count.  The
/// engine takes ownership of the allocation.
#[no_mangle]
pub unsafe extern "C" fn cf_voice_new(model_ptr: *mut u32, len: usize) -> *mut c_void {
    LAST_ERROR.store(0, Ordering::Relaxed);
    let container = match unsafe { container_for(model_ptr, len, voice_image()) } {
        Ok(c) => c,
        Err(code) => return fail(code),
    };
    match VoiceModel::from_container(container, Backend::Scalar) {
        Ok(model) => Box::into_raw(Box::new(model)) as *mut c_void,
        Err(_) => fail(3),
    }
}

/// Run one frame: 513 `f32` in, 1026 `f32` out.  No allocation.
///
/// Returns 0 on success, a negative error code otherwise.
#[no_mangle]
pub unsafe extern "C" fn cf_voice_process(
    handle: *mut c_void,
    input: *const f32,
    output: *mut f32,
) -> i32 {
    if handle.is_null() || input.is_null() || output.is_null() {
        return fail_i32(4);
    }
    let model = &mut *(handle as *mut VoiceModel);
    let frame = std::slice::from_raw_parts(input, INPUT_LEN);
    let dst = std::slice::from_raw_parts_mut(output, L18_LEN);
    match model.process(frame) {
        Ok(()) => {
            dst.copy_from_slice(model.output());
            0
        }
        Err(_) => fail_i32(4),
    }
}

/// Reset the cross-frame state (history, cell state, quantised hidden state).
#[no_mangle]
pub unsafe extern "C" fn cf_voice_reset(handle: *mut c_void) -> i32 {
    if handle.is_null() {
        return fail_i32(4);
    }
    let model = &mut *(handle as *mut VoiceModel);
    model.reset();
    0
}

/// Release a model created by [`cf_voice_new`].
#[no_mangle]
pub unsafe extern "C" fn cf_voice_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut VoiceModel));
    }
}

/// Container bytes resident for this model (embedded or host-supplied).
#[no_mangle]
pub unsafe extern "C" fn cf_voice_model_bytes(handle: *mut c_void) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let model = &*(handle as *const VoiceModel);
    model.container().file_len() as u32
}

// ---------------------------------------------------------------------------
// unified model-id selection
// ---------------------------------------------------------------------------

/// Create a model selected by container model id.  This module ships the voice
/// model only, so the only accepted id is 1.
///
/// `len == 0` uses the embedded voice image if it was compiled in (otherwise
/// the call fails with code 5).  A host buffer is parsed and its model id is
/// checked against `model_id`; mismatch fails with code 3.
#[no_mangle]
pub unsafe extern "C" fn cf_model_new(
    model_id: u32,
    model_ptr: *mut u32,
    len: usize,
) -> *mut c_void {
    LAST_ERROR.store(0, Ordering::Relaxed);
    if model_id != 1 {
        return fail(3);
    }
    let container = match unsafe { container_for(model_ptr, len, voice_image()) } {
        Ok(c) => c,
        Err(code) => return fail(code),
    };
    if container.model_id() != 1 {
        return fail(3);
    }
    match VoiceModel::from_container(container, Backend::Scalar) {
        Ok(model) => Box::into_raw(Box::new(model)) as *mut c_void,
        Err(_) => fail(3),
    }
}

/// Model id of a [`cf_model_new`] handle (0 on error).
#[no_mangle]
pub unsafe extern "C" fn cf_model_id(handle: *mut c_void) -> u32 {
    if handle.is_null() {
        0
    } else {
        1
    }
}

/// Input feature count of a [`cf_model_new`] handle (0 on error).
#[no_mangle]
pub unsafe extern "C" fn cf_model_input_len(handle: *mut c_void) -> u32 {
    if handle.is_null() {
        0
    } else {
        INPUT_LEN as u32
    }
}

/// Output value count of a [`cf_model_new`] handle (0 on error).
#[no_mangle]
pub unsafe extern "C" fn cf_model_output_len(handle: *mut c_void) -> u32 {
    if handle.is_null() {
        0
    } else {
        L18_LEN as u32
    }
}

/// Run one frame through a [`cf_model_new`] handle.
#[no_mangle]
pub unsafe extern "C" fn cf_model_process(
    handle: *mut c_void,
    input: *const f32,
    output: *mut f32,
) -> i32 {
    if handle.is_null() || input.is_null() || output.is_null() {
        return fail_i32(4);
    }
    let model = &mut *(handle as *mut VoiceModel);
    let frame = std::slice::from_raw_parts(input, INPUT_LEN);
    let dst = std::slice::from_raw_parts_mut(output, L18_LEN);
    match model.process(frame) {
        Ok(()) => {
            dst.copy_from_slice(model.output());
            0
        }
        Err(_) => fail_i32(4),
    }
}

/// Reset a [`cf_model_new`] handle.
#[no_mangle]
pub unsafe extern "C" fn cf_model_reset(handle: *mut c_void) -> i32 {
    if handle.is_null() {
        return fail_i32(4);
    }
    let model = &mut *(handle as *mut VoiceModel);
    model.reset();
    0
}

/// Release a [`cf_model_new`] handle.
#[no_mangle]
pub unsafe extern "C" fn cf_model_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut VoiceModel));
    }
}

// ---------------------------------------------------------------------------
// host allocator
// ---------------------------------------------------------------------------

/// Allocate `bytes` with alignment `align` (rounded up to a power of two).
///
/// The host uses this for input/output buffers.  Release with [`cf_free`] and
/// exactly the same arguments.
#[no_mangle]
pub extern "C" fn cf_alloc(bytes: usize, align: usize) -> *mut u8 {
    if bytes == 0 {
        return ptr::null_mut();
    }
    let align = align.max(1).next_power_of_two();
    match Layout::from_size_align(bytes, align) {
        Ok(layout) => unsafe { alloc(layout) },
        Err(_) => ptr::null_mut(),
    }
}

/// Release a buffer from [`cf_alloc`].
#[no_mangle]
pub unsafe extern "C" fn cf_free(ptr: *mut u8, bytes: usize, align: usize) {
    if ptr.is_null() || bytes == 0 {
        return;
    }
    let align = align.max(1).next_power_of_two();
    if let Ok(layout) = Layout::from_size_align(bytes, align) {
        dealloc(ptr, layout);
    }
}

/// Allocate `count` `u32` words for a host-supplied container image.
///
/// The host copies the container bytes into the returned buffer and passes
/// both the pointer and `count * 4` to [`cf_voice_new`], which takes ownership.
#[no_mangle]
pub extern "C" fn cf_alloc_u32(count: usize) -> *mut u32 {
    if count == 0 {
        return ptr::null_mut();
    }
    let mut words: Vec<u32> = Vec::with_capacity(count);
    let ptr = words.as_mut_ptr();
    std::mem::forget(words);
    ptr
}

/// Release a buffer from [`cf_alloc_u32`] that was not handed to
/// [`cf_voice_new`].
#[no_mangle]
pub unsafe extern "C" fn cf_free_u32(ptr: *mut u32, count: usize) {
    if !ptr.is_null() && count > 0 {
        drop(Vec::from_raw_parts(ptr, 0, count));
    }
}

// ---------------------------------------------------------------------------
// diagnostic views
// ---------------------------------------------------------------------------

/// Number of diagnostic buffers (f32 buffers, then i16 buffers).
#[no_mangle]
pub extern "C" fn cf_voice_diag_count() -> u32 {
    (DIAG_F32_SLOTS.len() + DIAG_I16_SLOTS.len()) as u32
}

fn diag_entry(index: usize) -> Option<(&'static str, u8)> {
    if let Some(&name) = DIAG_F32_SLOTS.get(index) {
        return Some((name, 0));
    }
    DIAG_I16_SLOTS
        .get(index - DIAG_F32_SLOTS.len())
        .map(|&name| (name, 1))
}

/// Buffer element kind: 0 = `f32`, 1 = `int16`.
#[no_mangle]
pub extern "C" fn cf_voice_diag_kind(index: u32) -> u32 {
    diag_entry(index as usize).map_or(2, |(_, kind)| kind as u32)
}

/// Stable buffer name pointer (UTF-8, not NUL terminated).
#[no_mangle]
pub extern "C" fn cf_voice_diag_name_ptr(index: u32) -> *const u8 {
    match diag_entry(index as usize) {
        Some((name, _)) => name.as_ptr(),
        None => ptr::null(),
    }
}

/// Stable buffer name length.
#[no_mangle]
pub extern "C" fn cf_voice_diag_name_len(index: u32) -> u32 {
    match diag_entry(index as usize) {
        Some((name, _)) => name.len() as u32,
        None => 0,
    }
}

/// Pointer to a diagnostic buffer for the current frame, or null on error.
#[no_mangle]
pub unsafe extern "C" fn cf_voice_diag_ptr(handle: *mut c_void, index: u32) -> *const u8 {
    if handle.is_null() {
        return ptr::null();
    }
    let model = &*(handle as *const VoiceModel);
    match diag_entry(index as usize) {
        Some((name, 0)) => model
            .slot(name)
            .map_or(ptr::null(), |s| s.as_ptr() as *const u8),
        Some((name, 1)) => model
            .slot_i16(name)
            .map_or(ptr::null(), |s| s.as_ptr() as *const u8),
        _ => ptr::null(),
    }
}

/// Element count of a diagnostic buffer, or 0 on error.
#[no_mangle]
pub unsafe extern "C" fn cf_voice_diag_len(handle: *mut c_void, index: u32) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let model = &*(handle as *const VoiceModel);
    match diag_entry(index as usize) {
        Some((name, 0)) => model.slot(name).map_or(0, |s| s.len() as u32),
        Some((name, 1)) => model.slot_i16(name).map_or(0, |s| s.len() as u32),
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// numeric-path self-test
// ---------------------------------------------------------------------------

/// 1 when the compiled `fma()` is the integer software implementation.
#[no_mangle]
pub extern "C" fn cf_fma_is_soft() -> u32 {
    fma::is_soft() as u32
}

/// Run the engine's `fma` on one triple and return the result bits.
#[no_mangle]
pub extern "C" fn cf_fma_probe(a: f32, b: f32, c: f32) -> u32 {
    fma::fma(a, b, c).to_bits()
}

/// Number of cases in [`cf_fma_self_test`].
#[no_mangle]
pub extern "C" fn cf_fma_self_test_count() -> u32 {
    FMA_CASES.len() as u32
}

/// Check the compiled FMA path against correctly-rounded expected bit
/// patterns, including near-total cancellation and the exponential
/// range-reduction pattern.
///
/// Returns 0 on success, `0xFFFF_FFFF` if the software path is not active, or
/// `1 + case_index` for the first mismatch.
#[no_mangle]
pub extern "C" fn cf_fma_self_test() -> u32 {
    if !fma::is_soft() {
        return 0xFFFF_FFFF;
    }
    for (i, &(a, b, c, want)) in FMA_CASES.iter().enumerate() {
        let got = fma::fma(f32::from_bits(a), f32::from_bits(b), f32::from_bits(c)).to_bits();
        if got != want {
            return i as u32 + 1;
        }
    }
    0
}

/// Hard cases with expected bits produced by the native hardware FMA (which is
/// correctly rounded).  Any silent change of the wasm numeric path shows up
/// here before the model runs.
static FMA_CASES: [(u32, u32, u32, u32); 42] = [
    (0x4040_0000, 0x3dcc_cccd, 0xbe99_999a, 0xb200_0000),
    (0xc300_0000, 0x3f31_7218, 0x42b1_7294, 0x3a78_0000),
    (0x4580_0800, 0x4580_0800, 0x0000_0000, 0x4b80_1000),
    (0x4580_0800, 0x4580_0800, 0x0000_0001, 0x4b80_1001),
    (0x4580_0800, 0x4580_0800, 0x8000_0001, 0x4b80_1000),
    (0x3f80_0001, 0x3f80_0000, 0xbf80_0000, 0x3400_0000),
    (0x40e0_0000, 0x3eaa_aaab, 0xc015_5555, 0x3420_0000),
    (0x60ad_78ec, 0x3f80_0000, 0xe0ad_78eb, 0x5500_0000),
    (0x0000_0001, 0x3f80_0000, 0x0000_0001, 0x0000_0002),
    (0x7f7f_ffff, 0x4000_0000, 0x0000_0000, 0x7f80_0000),
    (0x7f7f_ffff, 0x3f80_0000, 0xff7f_ffff, 0x0000_0000),
    (0x3f80_0000, 0x3f31_7218, 0xbf31_7218, 0x0000_0000),
    (0xc300_0000, 0x3f31_7218, 0x42b1_8218, 0x3d00_0000),
    (0xc2c2_0000, 0x3f31_7218, 0x4286_8496, 0x3cc1_fd00),
    (0xc280_0000, 0x3f31_7218, 0x4231_8218, 0x3c80_0000),
    (0xc214_0000, 0x3f31_7218, 0x41cd_3e6c, 0x3c14_0200),
    (0xc040_0000, 0x3f31_7218, 0x4005_2192, 0x3a40_0000),
    (0x3f80_0000, 0x3f31_7218, 0xbf31_8218, 0xb980_0000),
    (0x4188_0000, 0x3f31_7218, 0xc13c_9a3a, 0xbb88_0400),
    (0x4280_0000, 0x3f31_7218, 0xc231_8218, 0xbc80_0000),
    (0x42c2_0000, 0x3f31_7218, 0xc286_8496, 0xbcc1_fd00),
    (0x4300_0000, 0x3f31_7218, 0xc2b1_8218, 0xbd00_0000),
    (0x3f2c_eaee, 0x402a_8075, 0xbfe6_558d, 0x3225_ec60),
    (0x3f33_c490, 0x403c_69cf, 0xc004_4ea7, 0xb389_ff20),
    (0x3f37_7b2f, 0x4064_d815, 0xc024_0495, 0xb31c_f494),
    (0x3f76_6e10, 0x4010_69e6, 0xc00b_03da, 0xb3e9_1b40),
    (0x3f7b_e902, 0x4068_901b, 0xc064_d8f7, 0x33e5_666c),
    (0x3f72_caa7, 0x405a_55db, 0xc04f_11ff, 0x334b_3f74),
    (0x3f4c_88c0, 0x4020_c1ac, 0xc000_7035, 0xb3f6_be00),
    (0x3f1e_5c77, 0x4037_e836, 0xbfe3_877d, 0xb36e_9b98),
    (0x3f16_0fde, 0x4054_076a, 0xbff8_92f9, 0xb2b2_e0a0),
    (0x3f74_a4cb, 0x4075_4155, 0xc06a_6025, 0x33b2_84ce),
    (0x3f42_85a9, 0x3e8e_4d95, 0x3ffb_96da, 0x400b_4f90),
    (0x3f59_f321, 0x3efd_f64f, 0x3ff6_453a, 0x4016_297c),
    (0x3f12_191e, 0x3ed9_8959, 0x3fe8_518d, 0x4003_ad7c),
    (0x3f2a_7d7e, 0x3ecf_2735, 0x3f93_8c14, 0x3fb6_0979),
    (0x3f57_100d, 0x3eab_b1b0, 0x3fd6_e649, 0x3ffa_f589),
    (0x3f27_fdd0, 0x3ea9_5d23, 0x3fb9_76ba, 0x3fd5_3fa5),
    (0x3f5b_c5a7, 0x3ecb_cdeb, 0x3fa6_4803, 0x3fd2_05a4),
    (0x3f06_53e7, 0x3ecb_ad05, 0x3fa4_33bd, 0x3fbe_eb91),
    (0x3f51_c288, 0x3e8f_2179, 0x3f86_0b0f, 0x3fa3_5cd5),
    (0x3f4f_09bc, 0x3e87_1aba, 0x3fb5_51a6, 0x3fd0_a296),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn copy_into_wasm_style_buffer(image: &[u8]) -> (*mut u32, usize) {
        let words = image.len().div_ceil(4);
        let ptr = cf_alloc_u32(words);
        assert!(!ptr.is_null());
        // SAFETY: the allocation has `words` writable u32 slots.
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, words * 4) };
        dst[..image.len()].copy_from_slice(image);
        (ptr, image.len())
    }

    #[cfg(feature = "embedded-voice")]
    fn voice_frame() -> [f32; INPUT_LEN] {
        let mut frame = [0.0f32; INPUT_LEN];
        for (i, v) in frame.iter_mut().enumerate() {
            *v = ((i * 37 % 101) as f32) * 0.01 - 0.5;
        }
        frame
    }

    #[test]
    fn rejects_bad_model_pointers() {
        let handle = unsafe { cf_voice_new(ptr::null_mut(), 64) };
        assert!(handle.is_null());
        assert_eq!(cf_last_error(), 1);
    }

    // -----------------------------------------------------------------------
    // voice (embedded build only)
    // -----------------------------------------------------------------------
    #[cfg(feature = "embedded-voice")]
    fn voice_model_len() -> usize {
        voice_image().len()
    }

    #[cfg(feature = "embedded-voice")]
    #[test]
    fn embedded_voice_model_loads_and_processes() {
        let handle = unsafe { cf_voice_new(ptr::null_mut(), 0) };
        assert!(!handle.is_null(), "last error {}", cf_last_error());
        assert_eq!(cf_fma_is_soft(), 1);
        assert_eq!(cf_fma_self_test(), 0);
        let frame = voice_frame();
        let mut out = [0.0f32; L18_LEN];
        unsafe {
            assert_eq!(
                cf_voice_process(handle, frame.as_ptr(), out.as_mut_ptr()),
                0
            );
            assert_eq!(cf_voice_reset(handle), 0);
            assert_eq!(cf_voice_model_bytes(handle) as usize, voice_model_len());
            cf_voice_free(handle);
        }
        let mut reference =
            VoiceModel::with_backend(voice_image(), Backend::Scalar).expect("scalar model");
        reference.process(&frame).expect("process");
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference
                .output()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "embedded-voice")]
    #[test]
    fn host_supplied_voice_model_is_owned_without_copy() {
        let (ptr, len) = copy_into_wasm_style_buffer(voice_image());
        let handle = unsafe { cf_voice_new(ptr, len) };
        assert!(!handle.is_null(), "last error {}", cf_last_error());
        unsafe {
            assert_eq!(cf_voice_model_bytes(handle) as usize, len);
            cf_voice_free(handle);
        }
    }

    #[cfg(feature = "embedded-voice")]
    #[test]
    fn voice_workspace_footprint_matches_the_model() {
        let handle = unsafe { cf_voice_new(ptr::null_mut(), 0) };
        assert!(!handle.is_null());
        let model = unsafe { &*(handle as *const VoiceModel) };
        let actual =
            model.workspace_floats() * 4 + model.workspace_i16() * 2 + model.workspace_i32() * 4;
        assert_eq!(actual as u32, cf_voice_workspace_bytes());
        unsafe { cf_voice_free(handle) };
    }

    #[cfg(feature = "embedded-voice")]
    #[test]
    fn voice_diagnostic_views_cover_every_buffer() {
        let handle = unsafe { cf_voice_new(ptr::null_mut(), 0) };
        assert!(!handle.is_null());
        let mut frame = [0.0f32; INPUT_LEN];
        frame[0] = 0.25;
        let mut out = [0.0f32; L18_LEN];
        unsafe {
            assert_eq!(
                cf_voice_process(handle, frame.as_ptr(), out.as_mut_ptr()),
                0
            );
            assert_eq!(
                cf_voice_diag_count(),
                (DIAG_F32_SLOTS.len() + DIAG_I16_SLOTS.len()) as u32
            );
            let model = &*(handle as *const VoiceModel);
            for i in 0..cf_voice_diag_count() {
                let (name, kind) = diag_entry(i as usize).expect("valid diagnostic index");
                let expected_len = if kind == 0 {
                    model.slot(name).expect("f32 slot").len()
                } else {
                    model.slot_i16(name).expect("i16 slot").len()
                };
                assert_eq!(cf_voice_diag_kind(i), kind as u32);
                assert_eq!(cf_voice_diag_len(handle, i) as usize, expected_len);
                assert_eq!(cf_voice_diag_name_len(i) as usize, name.len());
                assert!(!cf_voice_diag_ptr(handle, i).is_null());
                assert!(!cf_voice_diag_name_ptr(i).is_null());
            }
            cf_voice_free(handle);
        }
    }

    // -----------------------------------------------------------------------
    // unified model-id selection
    // -----------------------------------------------------------------------
    #[cfg(feature = "embedded-voice")]
    #[test]
    fn model_id_selection_reaches_the_voice_model() {
        let (ptr, len) = copy_into_wasm_style_buffer(voice_image());
        let handle = unsafe { cf_model_new(1, ptr, len) };
        assert!(!handle.is_null(), "last error {}", cf_last_error());
        assert_eq!(unsafe { cf_model_id(handle) }, 1);
        assert_eq!(unsafe { cf_model_input_len(handle) }, INPUT_LEN as u32);
        assert_eq!(unsafe { cf_model_output_len(handle) }, L18_LEN as u32);
        let frame = voice_frame();
        let mut out = [0.0f32; L18_LEN];
        unsafe {
            assert_eq!(cf_model_reset(handle), 0);
            assert_eq!(
                cf_model_process(handle, frame.as_ptr(), out.as_mut_ptr()),
                0
            );
            cf_model_free(handle);
        }
        let mut reference =
            VoiceModel::with_backend(voice_image(), Backend::Scalar).expect("scalar model");
        reference.process(&frame).expect("process");
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference
                .output()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        // The echo model is not part of this module.
        let rejected = unsafe { cf_model_new(2, ptr::null_mut(), 0) };
        assert!(rejected.is_null());
        assert_eq!(cf_last_error(), 3);
    }
}
