//! A dependency-free slice of the Node-API (N-API) C ABI.
//!
//! Normally a Node addon includes `node_api.h` through a package such as
//! `napi-sys` and links against `node.lib` (Windows) or the host executable
//! (Unix).  This build environment has no network access, no Node headers and
//! no import library, so the handful of entry points the addon needs are
//! resolved at run time from the host executable's export table:
//!
//! * Windows: `GetProcAddress(GetModuleHandleW(NULL), name)`
//! * Unix: `dlsym(RTLD_DEFAULT, name)`
//!
//! Node and Electron both export the N-API C symbols from their main
//! executable, which is why an addon can load without a link-time dependency.
//! The tradeoff is that a missing symbol surfaces as an empty addon rather than
//! a loader error; [`api`] therefore returns `None` and the registration
//! function degrades to returning the untouched exports object.
//!
//! Only the entry points used by this addon are declared.  Signatures follow
//! `node_api.h` (N-API version 8); pointers are opaque `*mut c_void`.

use std::ffi::c_void;
use std::sync::OnceLock;

pub type Env = *mut c_void;
pub type Value = *mut c_void;
pub type Info = *mut c_void;
pub type Status = i32;

/// `napi_ok`.
pub const OK: Status = 0;

/// `napi_float32_array` in `napi_typedarray_type`.
pub const FLOAT32_ARRAY: i32 = 7;
/// `napi_object` in `napi_valuetype`.
pub const TYPE_OBJECT: i32 = 6;
/// `napi_string` in `napi_valuetype`.
pub const TYPE_STRING: i32 = 4;
/// `napi_undefined` in `napi_valuetype`.
pub const TYPE_UNDEFINED: i32 = 0;

pub type Callback = unsafe extern "C" fn(Env, Info) -> Value;
pub type Finalize = unsafe extern "C" fn(Env, *mut c_void, *mut c_void);

type FnGetCbInfo =
    unsafe extern "C" fn(Env, Info, *mut usize, *mut Value, *mut Value, *mut *mut c_void) -> Status;
type FnCreateObject = unsafe extern "C" fn(Env, *mut Value) -> Status;
type FnCreateFunction = unsafe extern "C" fn(
    Env,
    *const u8,
    usize,
    Option<Callback>,
    *mut c_void,
    *mut Value,
) -> Status;
type FnSetNamed = unsafe extern "C" fn(Env, Value, *const u8, Value) -> Status;
type FnCreateString = unsafe extern "C" fn(Env, *const u8, usize, *mut Value) -> Status;
type FnGetString = unsafe extern "C" fn(Env, Value, *mut u8, usize, *mut usize) -> Status;
type FnTypeOf = unsafe extern "C" fn(Env, Value, *mut i32) -> Status;
type FnGetNamed = unsafe extern "C" fn(Env, Value, *const u8, *mut Value) -> Status;
type FnGetTypedArrayInfo = unsafe extern "C" fn(
    Env,
    Value,
    *mut i32,
    *mut usize,
    *mut *mut c_void,
    *mut Value,
    *mut usize,
) -> Status;
type FnCreateArrayBuffer = unsafe extern "C" fn(Env, usize, *mut *mut c_void, *mut Value) -> Status;
type FnCreateTypedArray = unsafe extern "C" fn(Env, i32, usize, Value, usize, *mut Value) -> Status;
type FnThrow = unsafe extern "C" fn(Env, *const u8, *const u8) -> Status;
type FnAddFinalizer = unsafe extern "C" fn(
    Env,
    Value,
    *mut c_void,
    Option<Finalize>,
    *mut c_void,
    *mut Value,
) -> Status;
type FnGetUndefined = unsafe extern "C" fn(Env, *mut Value) -> Status;

/// The resolved N-API entry points.
pub struct Api {
    pub get_cb_info: FnGetCbInfo,
    pub create_object: FnCreateObject,
    pub create_function: FnCreateFunction,
    pub set_named_property: FnSetNamed,
    pub create_string_utf8: FnCreateString,
    pub get_value_string_utf8: FnGetString,
    pub type_of: FnTypeOf,
    pub get_named_property: FnGetNamed,
    pub get_typedarray_info: FnGetTypedArrayInfo,
    pub create_arraybuffer: FnCreateArrayBuffer,
    pub create_typedarray: FnCreateTypedArray,
    pub throw_error: FnThrow,
    pub throw_type_error: FnThrow,
    pub add_finalizer: FnAddFinalizer,
    pub get_undefined: FnGetUndefined,
}

#[cfg(windows)]
fn sym(name: &[u8]) -> *mut c_void {
    extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }
    // SAFETY: a null name asks for the main module (node.exe / electron.exe).
    // The symbol names below are NUL-terminated literals.
    unsafe { GetProcAddress(GetModuleHandleW(std::ptr::null()), name.as_ptr()) }
}

#[cfg(all(unix, not(target_os = "macos")))]
#[link(name = "dl")]
extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
}

#[cfg(target_os = "macos")]
extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
}

/// `RTLD_DEFAULT` is `(void *)-2` on macOS and `NULL` elsewhere.
#[cfg(all(unix, target_os = "macos"))]
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;
#[cfg(all(unix, not(target_os = "macos")))]
const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

#[cfg(unix)]
fn sym(name: &[u8]) -> *mut c_void {
    // SAFETY: every caller passes a NUL-terminated literal.
    unsafe { dlsym(RTLD_DEFAULT, name.as_ptr() as *const std::ffi::c_char) }
}

#[cfg(not(any(windows, unix)))]
fn sym(_name: &[u8]) -> *mut c_void {
    std::ptr::null_mut()
}

impl Api {
    fn load() -> Option<Api> {
        macro_rules! load {
            ($name:literal, $ty:ty) => {{
                let p = sym(concat!($name, "\0").as_bytes());
                if p.is_null() {
                    return None;
                }
                // SAFETY: the symbol comes from the N-API C ABI and the
                // signature is declared to match `node_api.h`.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
            }};
        }
        Some(Api {
            get_cb_info: load!("napi_get_cb_info", FnGetCbInfo),
            create_object: load!("napi_create_object", FnCreateObject),
            create_function: load!("napi_create_function", FnCreateFunction),
            set_named_property: load!("napi_set_named_property", FnSetNamed),
            create_string_utf8: load!("napi_create_string_utf8", FnCreateString),
            get_value_string_utf8: load!("napi_get_value_string_utf8", FnGetString),
            type_of: load!("napi_typeof", FnTypeOf),
            get_named_property: load!("napi_get_named_property", FnGetNamed),
            get_typedarray_info: load!("napi_get_typedarray_info", FnGetTypedArrayInfo),
            create_arraybuffer: load!("napi_create_arraybuffer", FnCreateArrayBuffer),
            create_typedarray: load!("napi_create_typedarray", FnCreateTypedArray),
            throw_error: load!("napi_throw_error", FnThrow),
            throw_type_error: load!("napi_throw_type_error", FnThrow),
            add_finalizer: load!("napi_add_finalizer", FnAddFinalizer),
            get_undefined: load!("napi_get_undefined", FnGetUndefined),
        })
    }
}

/// The process-wide entry-point table, or `None` when not running inside a
/// Node-compatible host (for example `cargo test`).
pub fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(Api::load).as_ref()
}

/// NUL-terminated C string, with interior NULs removed so a JS exception
/// message can never fail to be thrown.
pub fn c_message(message: &str) -> Vec<u8> {
    let mut out: Vec<u8> = message
        .bytes()
        .map(|b| if b == 0 { b' ' } else { b })
        .collect();
    out.push(0);
    out
}

/// Throw a JS `Error` with `message` (a no-op if N-API is unavailable).
pub unsafe fn throw_error(env: Env, message: &str) {
    if let Some(api) = api() {
        let msg = c_message(message);
        // SAFETY: `msg` is NUL-terminated and outlives the call.
        unsafe {
            (api.throw_error)(env, std::ptr::null(), msg.as_ptr());
        }
    }
}

/// Throw a JS `TypeError` with `message`.
pub unsafe fn throw_type_error(env: Env, message: &str) {
    if let Some(api) = api() {
        let msg = c_message(message);
        // SAFETY: `msg` is NUL-terminated and outlives the call.
        unsafe {
            (api.throw_type_error)(env, std::ptr::null(), msg.as_ptr());
        }
    }
}

/// Read a JS string value as UTF-8.
pub unsafe fn get_string(env: Env, value: Value) -> Result<String, ()> {
    let Some(api) = api() else { return Err(()) };
    let mut len = 0usize;
    // SAFETY: the buffer is null with capacity 0, which asks for the length.
    if unsafe { (api.get_value_string_utf8)(env, value, std::ptr::null_mut(), 0, &mut len) } != OK {
        return Err(());
    }
    let mut buf = vec![0u8; len + 1];
    let mut written = 0usize;
    // SAFETY: `buf` has room for `len` bytes plus the NUL terminator.
    let status = unsafe {
        (api.get_value_string_utf8)(env, value, buf.as_mut_ptr(), buf.len(), &mut written)
    };
    if status != OK {
        return Err(());
    }
    buf.truncate(written);
    String::from_utf8(buf).map_err(|_| ())
}

/// Create a JS string from UTF-8.
pub unsafe fn make_string(env: Env, value: &str) -> Value {
    let Some(api) = api() else {
        return std::ptr::null_mut();
    };
    let mut out: Value = std::ptr::null_mut();
    // SAFETY: the pointer/length pair describes valid UTF-8 for the duration
    // of the call.
    unsafe {
        (api.create_string_utf8)(env, value.as_ptr(), value.len(), &mut out);
    }
    out
}

/// `typeof` of a JS value.
pub unsafe fn type_of(env: Env, value: Value) -> i32 {
    let Some(api) = api() else { return -1 };
    let mut out: i32 = -1;
    // SAFETY: `out` is a valid out-parameter.
    unsafe {
        (api.type_of)(env, value, &mut out);
    }
    out
}

/// Read a named property of a JS object; `None` when it is absent.
pub unsafe fn get_property(env: Env, object: Value, name: &'static [u8]) -> Option<Value> {
    let api = api()?;
    let mut out: Value = std::ptr::null_mut();
    // SAFETY: `name` is NUL-terminated by construction.
    if unsafe { (api.get_named_property)(env, object, name.as_ptr(), &mut out) } != OK {
        return None;
    }
    if unsafe { type_of(env, out) } == TYPE_UNDEFINED {
        return None;
    }
    Some(out)
}

/// Create a JS function and attach it to `object` under `name`.
pub unsafe fn define_function(
    env: Env,
    object: Value,
    name: &'static [u8],
    callback: Callback,
    data: *mut c_void,
) -> bool {
    let Some(api) = api() else { return false };
    let mut function: Value = std::ptr::null_mut();
    // SAFETY: `name` is NUL-terminated; the length excludes the terminator.
    let status = unsafe {
        (api.create_function)(
            env,
            name.as_ptr(),
            name.len() - 1,
            Some(callback),
            data,
            &mut function,
        )
    };
    if status != OK {
        return false;
    }
    // SAFETY: `function` is a live JS value in this env.
    unsafe { (api.set_named_property)(env, object, name.as_ptr(), function) == OK }
}
