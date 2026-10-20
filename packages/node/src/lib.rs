//! CleanField voice engine as a Node native addon.
//!
//! The JavaScript surface is deliberately small:
//!
//! ```js
//! const { createVoice } = require('@cleanfield/node');
//! const voice = createVoice({ modelPath });
//! const out = voice.process(frame);   // Float32Array(1026) <- Float32Array(513)
//! voice.reset();
//! voice.close();
//! voice.backend;                      // "scalar" | "avx2"
//! ```
//!
//! ## Why no `napi-rs`
//!
//! `napi`/`napi-derive` would add a crates.io dependency, a `build.rs` that
//! locates Node headers, and a link-time dependency on the host import library.
//! The release build host has no network access, and the engine crates are
//! dependency-free by design, so this addon instead:
//!
//! 1. links only `cf-ffi` (the hardened C ABI: validation, backend selection,
//!    error codes) from the workspace;
//! 2. declares the ~15 N-API entry points it needs by hand and resolves them
//!    from the host executable at run time (`src/napi.rs`);
//! 3. exports `napi_register_module_v1` directly, which Node's addon loader
//!    uses for N-API modules.
//!
//! The result is an ordinary Rust cdylib that is renamed to `cf-node.node`;
//! there is no node-gyp step, no Node headers, and no `.lib` to ship.
//!
//! ## Threading and memory
//!
//! One JS object owns one engine handle and one fixed workspace.  Calls must
//! not overlap for the same object; different objects (including objects in
//! different worker threads) are independent.  `process` allocates one
//! `ArrayBuffer` per call for the result; the engine itself does not allocate
//! per frame.  `close()` releases the engine immediately and is idempotent;
//! the small wrapper allocation is released by the JS garbage collector.
#![allow(clippy::missing_safety_doc)]

mod napi;

use std::ffi::c_void;
use std::ptr;

use cleanfield::{
    cf_last_error, cf_model, cf_model_backend_name, cf_model_free, cf_model_input_len,
    cf_model_new_with_backend, cf_model_output_len, cf_model_process, cf_model_reset,
    CF_BACKEND_AUTO, CF_BACKEND_AVX2, CF_BACKEND_SCALAR, CF_VOICE_MODEL_ID,
};

use napi::{Env, Info, Value, FLOAT32_ARRAY, OK, TYPE_OBJECT, TYPE_STRING};

/// Input feature count of the voice model (513).
const VOICE_INPUT_LEN: usize = 513;
/// Output value count of the voice model (1026).
const VOICE_OUTPUT_LEN: usize = 1026;

/// One engine handle plus the frame geometry reported by the C ABI.
struct Voice {
    /// Null after `close()`.
    model: *mut cf_model,
    input_len: usize,
    output_len: usize,
}

fn error_text(code: u32) -> &'static str {
    match code {
        1 => "invalid argument (bad model buffer or unknown backend)",
        2 => "the model bytes are not a valid .cfwm container",
        3 => "the container is not a voice model (wrong model id or failed validation)",
        4 => "internal state error (model handle or frame buffer)",
        5 => "the requested backend is not available on this CPU",
        _ => "unknown engine error",
    }
}

fn backend_code(name: &str) -> Option<u32> {
    match name {
        "auto" => Some(CF_BACKEND_AUTO),
        "scalar" => Some(CF_BACKEND_SCALAR),
        "avx2" => Some(CF_BACKEND_AVX2),
        _ => None,
    }
}

/// Read `argv` and the callback's data pointer.
unsafe fn callback_args<const N: usize>(env: Env, info: Info) -> (usize, [Value; N], *mut c_void) {
    let mut argv: [Value; N] = [ptr::null_mut(); N];
    let mut argc = N;
    let mut data: *mut c_void = ptr::null_mut();
    let Some(api) = napi::api() else {
        return (0, argv, data);
    };
    // SAFETY: argv has capacity N and is initialised; argc is in/out.
    let status = unsafe {
        (api.get_cb_info)(
            env,
            info,
            &mut argc,
            argv.as_mut_ptr(),
            ptr::null_mut(),
            &mut data,
        )
    };
    if status != OK {
        return (0, argv, data);
    }
    (argc.min(N), argv, data)
}

unsafe fn set_property(env: Env, object: Value, name: &'static [u8], value: Value) -> bool {
    napi::api().is_some_and(|api| unsafe {
        (api.set_named_property)(env, object, name.as_ptr(), value) == OK
    })
}

/// `createVoice({ modelPath, backend? })`
unsafe extern "C" fn create_voice(env: Env, info: Info) -> Value {
    let (argc, argv, _) = unsafe { callback_args::<1>(env, info) };
    if argc < 1 || unsafe { napi::type_of(env, argv[0]) } != TYPE_OBJECT {
        unsafe { napi::throw_type_error(env, "createVoice(options) expects an options object") };
        return ptr::null_mut();
    }
    let options = argv[0];
    let model_path = match unsafe { napi::get_property(env, options, b"modelPath\0") } {
        Some(value) if unsafe { napi::type_of(env, value) } == TYPE_STRING => {
            match unsafe { napi::get_string(env, value) } {
                Ok(path) => path,
                Err(()) => {
                    unsafe {
                        napi::throw_error(env, "createVoice: could not read options.modelPath")
                    };
                    return ptr::null_mut();
                }
            }
        }
        _ => {
            unsafe {
                napi::throw_type_error(env, "createVoice: options.modelPath must be a string")
            };
            return ptr::null_mut();
        }
    };

    let backend_name = match unsafe { napi::get_property(env, options, b"backend\0") } {
        None => "auto".to_string(),
        Some(value) if unsafe { napi::type_of(env, value) } == TYPE_STRING => {
            match unsafe { napi::get_string(env, value) } {
                Ok(name) => name,
                Err(()) => {
                    unsafe {
                        napi::throw_error(env, "createVoice: could not read options.backend")
                    };
                    return ptr::null_mut();
                }
            }
        }
        Some(_) => {
            unsafe { napi::throw_type_error(env, "createVoice: options.backend must be a string") };
            return ptr::null_mut();
        }
    };
    let Some(backend) = backend_code(&backend_name) else {
        unsafe {
            napi::throw_type_error(
                env,
                "createVoice: options.backend must be \"auto\", \"scalar\" or \"avx2\"",
            )
        };
        return ptr::null_mut();
    };

    let bytes = match std::fs::read(&model_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            unsafe {
                napi::throw_error(
                    env,
                    &format!("createVoice: could not read model file {model_path}: {error}"),
                )
            };
            return ptr::null_mut();
        }
    };
    if bytes.is_empty() {
        unsafe { napi::throw_error(env, "createVoice: the model file is empty") };
        return ptr::null_mut();
    }
    if bytes.len() % 4 != 0 {
        unsafe {
            napi::throw_error(
                env,
                "createVoice: the model file length is not a multiple of four bytes",
            )
        };
        return ptr::null_mut();
    }

    // SAFETY: `bytes` is readable for `bytes.len()` bytes for the duration of
    // the call; the C ABI copies the container into its own aligned buffer.
    let model = unsafe {
        cf_model_new_with_backend(CF_VOICE_MODEL_ID, bytes.as_ptr(), bytes.len(), backend)
    };
    if model.is_null() {
        let code = cf_last_error();
        unsafe {
            napi::throw_error(
                env,
                &format!(
                    "createVoice: could not construct the voice model: {} (engine error {code})",
                    error_text(code)
                ),
            )
        };
        return ptr::null_mut();
    }

    let input_len = unsafe { cf_model_input_len(model) } as usize;
    let output_len = unsafe { cf_model_output_len(model) } as usize;
    if output_len == 0 || input_len == 0 {
        unsafe { cf_model_free(model) };
        unsafe { napi::throw_error(env, "createVoice: the engine reported an empty frame size") };
        return ptr::null_mut();
    }

    let backend_live = unsafe {
        let ptr = cf_model_backend_name(model);
        if ptr.is_null() {
            "unknown".to_string()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    let Some(api) = napi::api() else {
        unsafe { cf_model_free(model) };
        return ptr::null_mut();
    };
    let mut object: Value = ptr::null_mut();
    if unsafe { (api.create_object)(env, &mut object) } != OK {
        unsafe { cf_model_free(model) };
        return ptr::null_mut();
    }
    let backend_value = unsafe { napi::make_string(env, &backend_live) };
    if !unsafe { set_property(env, object, b"backend\0", backend_value) } {
        unsafe { cf_model_free(model) };
        unsafe { napi::throw_error(env, "createVoice: could not define the backend property") };
        return ptr::null_mut();
    }

    let voice = Box::into_raw(Box::new(Voice {
        model,
        input_len,
        output_len,
    })) as *mut c_void;
    // SAFETY: `voice` stays alive until the finalizer or the JS object is
    // collected; the property loop above has released the model on failure.
    let functions_ok = unsafe {
        napi::define_function(env, object, b"process\0", process, voice)
            && napi::define_function(env, object, b"reset\0", reset, voice)
            && napi::define_function(env, object, b"close\0", close, voice)
    };
    if !functions_ok {
        // Reclaim the wrapper; no finalizer is registered yet.
        unsafe { drop(Box::from_raw(voice as *mut Voice)) };
        unsafe { napi::throw_error(env, "createVoice: could not define the voice methods") };
        return ptr::null_mut();
    }
    // SAFETY: the finalizer receives the same pointer once the JS object is
    // unreachable, and `Voice::model` is nulled by `close()`.
    let status = unsafe {
        (api.add_finalizer)(
            env,
            object,
            voice,
            Some(finalize_voice),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    if status != OK {
        unsafe { drop(Box::from_raw(voice as *mut Voice)) };
        unsafe { napi::throw_error(env, "createVoice: could not register the handle finalizer") };
        return ptr::null_mut();
    }
    object
}

unsafe extern "C" fn finalize_voice(_env: Env, data: *mut c_void, _hint: *mut c_void) {
    if data.is_null() {
        return;
    }
    // SAFETY: `data` is the Box<Voice> handed to napi_add_finalizer, and this
    // finalizer runs exactly once.
    let mut voice = unsafe { Box::from_raw(data as *mut Voice) };
    if !voice.model.is_null() {
        unsafe { cf_model_free(voice.model) };
        voice.model = ptr::null_mut();
    }
}

/// `voice.process(Float32Array) -> Float32Array`
unsafe extern "C" fn process(env: Env, info: Info) -> Value {
    let (argc, argv, data) = unsafe { callback_args::<1>(env, info) };
    if data.is_null() {
        unsafe { napi::throw_error(env, "process: the voice handle is invalid") };
        return ptr::null_mut();
    }
    // SAFETY: `data` is the Box<Voice> owned by the JS object for this call.
    let voice = unsafe { &mut *(data as *mut Voice) };
    if voice.model.is_null() {
        unsafe { napi::throw_error(env, "process: the voice handle is closed") };
        return ptr::null_mut();
    }
    if argc < 1 {
        unsafe { napi::throw_type_error(env, "process(frame) expects a Float32Array") };
        return ptr::null_mut();
    }

    let Some(api) = napi::api() else {
        return ptr::null_mut();
    };
    let mut array_type = -1i32;
    let mut length = 0usize;
    let mut input_data: *mut c_void = ptr::null_mut();
    let mut array_buffer: Value = ptr::null_mut();
    let mut byte_offset = 0usize;
    // SAFETY: all out-parameters are valid; a non-typed-array argument makes
    // the call fail and is reported as a TypeError below.
    let status = unsafe {
        (api.get_typedarray_info)(
            env,
            argv[0],
            &mut array_type,
            &mut length,
            &mut input_data,
            &mut array_buffer,
            &mut byte_offset,
        )
    };
    if status != OK || array_type != FLOAT32_ARRAY {
        unsafe { napi::throw_type_error(env, "process(frame): frame must be a Float32Array") };
        return ptr::null_mut();
    }
    if length != voice.input_len {
        unsafe {
            napi::throw_type_error(
                env,
                &format!(
                    "process(frame): expected {} samples, received {length}",
                    voice.input_len
                ),
            )
        };
        return ptr::null_mut();
    }

    // One output allocation per call, owned by the returned JS typed array.
    let mut output_data: *mut c_void = ptr::null_mut();
    let mut output_buffer: Value = ptr::null_mut();
    if unsafe {
        (api.create_arraybuffer)(
            env,
            voice.output_len * 4,
            &mut output_data,
            &mut output_buffer,
        )
    } != OK
        || output_data.is_null()
    {
        unsafe { napi::throw_error(env, "process: could not allocate the output buffer") };
        return ptr::null_mut();
    }
    let mut output_array: Value = ptr::null_mut();
    // SAFETY: `output_buffer` holds `output_len * 4` bytes and
    // `napi_float32_array` elements are four bytes wide.
    let status = unsafe {
        (api.create_typedarray)(
            env,
            FLOAT32_ARRAY,
            voice.output_len,
            output_buffer,
            0,
            &mut output_array,
        )
    };
    if status != OK {
        unsafe { napi::throw_error(env, "process: could not create the output typed array") };
        return ptr::null_mut();
    }

    // SAFETY: the caller's typed array is readable for `input_len` floats; the
    // output buffer is writable for `output_len` floats; the two do not alias.
    let code = unsafe {
        cf_model_process(
            voice.model,
            input_data as *const f32,
            output_data as *mut f32,
        )
    };
    if code != 0 {
        let last = cf_last_error();
        unsafe {
            napi::throw_error(
                env,
                &format!(
                    "process: the engine rejected the frame: {} (engine error {last})",
                    error_text(last)
                ),
            )
        };
        return ptr::null_mut();
    }
    output_array
}

/// `voice.reset()`
unsafe extern "C" fn reset(env: Env, info: Info) -> Value {
    let (_, _, data) = unsafe { callback_args::<0>(env, info) };
    if data.is_null() {
        unsafe { napi::throw_error(env, "reset: the voice handle is invalid") };
        return ptr::null_mut();
    }
    // SAFETY: `data` is the Box<Voice> owned by the JS object for this call.
    let voice = unsafe { &mut *(data as *mut Voice) };
    if voice.model.is_null() {
        unsafe { napi::throw_error(env, "reset: the voice handle is closed") };
        return ptr::null_mut();
    }
    let code = unsafe { cf_model_reset(voice.model) };
    if code != 0 {
        let last = cf_last_error();
        unsafe {
            napi::throw_error(
                env,
                &format!("reset: {}", error_text(if last != 0 { last } else { 4 })),
            )
        };
        return ptr::null_mut();
    }
    let mut undefined: Value = ptr::null_mut();
    if let Some(api) = napi::api() {
        unsafe {
            (api.get_undefined)(env, &mut undefined);
        }
    }
    undefined
}

/// `voice.close()`
unsafe extern "C" fn close(env: Env, info: Info) -> Value {
    let (_, _, data) = unsafe { callback_args::<0>(env, info) };
    if !data.is_null() {
        // SAFETY: `data` is the Box<Voice> owned by the JS object for this call.
        let voice = unsafe { &mut *(data as *mut Voice) };
        if !voice.model.is_null() {
            unsafe { cf_model_free(voice.model) };
            voice.model = ptr::null_mut();
        }
    }
    let mut undefined: Value = ptr::null_mut();
    if let Some(api) = napi::api() {
        unsafe {
            (api.get_undefined)(env, &mut undefined);
        }
    }
    undefined
}

/// Entry point used by Node's addon loader for N-API modules.
#[no_mangle]
pub unsafe extern "C" fn napi_register_module_v1(env: Env, exports: Value) -> Value {
    if napi::api().is_none() {
        // Not running inside a Node-compatible host: leave the exports empty
        // rather than aborting the process.
        return exports;
    }
    // SAFETY: the exports object is owned by the host for the duration of the
    // call; `define_function` only adds one property.
    unsafe {
        napi::define_function(
            env,
            exports,
            b"createVoice\0",
            create_voice,
            ptr::null_mut(),
        );
    }
    exports
}

/// Static frame sizes, also exported for tests and diagnostics.
pub const INPUT_LEN: usize = VOICE_INPUT_LEN;
/// Static frame sizes, also exported for tests and diagnostics.
pub const OUTPUT_LEN: usize = VOICE_OUTPUT_LEN;

#[cfg(test)]
mod tests {
    use super::*;

    // The N-API callbacks cannot run outside a Node host; the in-process engine
    // path is exercised here so the geometry assumptions are checked by
    // `cargo test` as well as by the Node smoke test.
    #[test]
    fn frame_geometry_matches_the_engine() {
        assert_eq!(cleanfield::cf_voice_input_len() as usize, INPUT_LEN);
        assert_eq!(cleanfield::cf_voice_output_len() as usize, OUTPUT_LEN);
        assert_eq!(cleanfield::cf_voice_model_id(), CF_VOICE_MODEL_ID);
    }

    #[test]
    fn backend_names_map_to_codes() {
        assert_eq!(backend_code("auto"), Some(CF_BACKEND_AUTO));
        assert_eq!(backend_code("scalar"), Some(CF_BACKEND_SCALAR));
        assert_eq!(backend_code("avx2"), Some(CF_BACKEND_AVX2));
        assert_eq!(backend_code("nope"), None);
    }

    #[test]
    fn error_codes_have_messages() {
        for code in 1..=5 {
            assert!(!error_text(code).is_empty());
        }
        assert!(!error_text(99).is_empty());
    }
}
