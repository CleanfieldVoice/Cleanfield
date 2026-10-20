//! Tiny consumer of the native CleanField C ABI.
//!
//! This file deliberately declares the ABI by hand and uses only `std`: it is
//! a stand-in for a C host translation unit.  It is compiled twice against the
//! built library (see `scripts/verify_m6_link.ps1` or M6-REPORT.md):
//!
//! * dynamic: `rustc ... -L target/release -l dylib=cleanfield`
//! * static:  `rustc ... -L target/release -l static=cleanfield`
//!
//! (plus the platform system libraries for the static link).
//!
//! Usage: `embed [echo.cfwm] [voice.cfwm]` (defaults are the in-tree models).
//! For each model it loads the same container through the typed constructor and
//! through `cf_model_new`, runs a deterministic frame sequence through both,
//! checks that every output bit is identical, resets, replays, and prints a
//! checksum.  Exit code 0 means the ABI, backend selection, process, reset and
//! free paths all worked.

use std::ffi::CStr;
use std::fs;
use std::os::raw::c_char;
use std::process;

#[repr(C)]
struct cf_model {
    _opaque: [u8; 0],
}

extern "C" {
    fn cf_abi_version() -> u32;
    fn cf_last_error() -> u32;

    fn cf_model_new(model_id: u32, model_bytes: *const u8, len: usize) -> *mut cf_model;
    fn cf_model_process(m: *mut cf_model, input: *const f32, output: *mut f32) -> i32;
    fn cf_model_reset(m: *mut cf_model) -> i32;
    fn cf_model_free(m: *mut cf_model);
    fn cf_model_input_len(m: *const cf_model) -> u32;
    fn cf_model_output_len(m: *const cf_model) -> u32;
    fn cf_model_id(m: *const cf_model) -> u32;
    fn cf_model_backend_name(m: *const cf_model) -> *const c_char;
    fn cf_model_workspace_bytes(m: *const cf_model) -> u32;

    fn cf_voice_new(model_bytes: *const u8, len: usize) -> *mut cf_model;
    fn cf_voice_process(m: *mut cf_model, input: *const f32, output: *mut f32) -> i32;
    fn cf_voice_reset(m: *mut cf_model) -> i32;
    fn cf_voice_free(m: *mut cf_model);

    fn cf_echo_new(model_bytes: *const u8, len: usize) -> *mut cf_model;
    fn cf_echo_process(m: *mut cf_model, input: *const f32, output: *mut f32) -> i32;
    fn cf_echo_reset(m: *mut cf_model) -> i32;
    fn cf_echo_free(m: *mut cf_model);
}

fn fill(buf: &mut [f32], frame: usize) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ (frame as u64).wrapping_mul(0x1000_0000_1B3);
    for v in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = ((s >> 40) as f32) * (1.0 / 16_777_216.0);
    }
}

struct Run {
    name: &'static str,
    model_id: u32,
    frames: [usize; 2], // [in, out]
}

fn fail(msg: &str) -> ! {
    eprintln!("FAIL: {msg} (last_error={})", unsafe { cf_last_error() });
    process::exit(1);
}

fn run(model: &Run, bytes: &[u8]) {
    let (in_len, out_len) = (model.frames[0], model.frames[1]);

    let typed = match model.model_id {
        1 => unsafe { cf_voice_new(bytes.as_ptr(), bytes.len()) },
        2 => unsafe { cf_echo_new(bytes.as_ptr(), bytes.len()) },
        _ => unreachable!(),
    };
    if typed.is_null() {
        fail("typed constructor returned NULL");
    }
    let unified = unsafe { cf_model_new(model.model_id, bytes.as_ptr(), bytes.len()) };
    if unified.is_null() {
        fail("cf_model_new returned NULL");
    }

    let got_id = unsafe { cf_model_id(typed) };
    let got_in = unsafe { cf_model_input_len(typed) } as usize;
    let got_out = unsafe { cf_model_output_len(typed) } as usize;
    if got_id != model.model_id || got_in != in_len || got_out != out_len {
        fail("reported model id / lengths do not match the contract");
    }
    let backend = unsafe { CStr::from_ptr(cf_model_backend_name(typed)) }
        .to_string_lossy()
        .into_owned();
    let ws = unsafe { cf_model_workspace_bytes(typed) };

    let mut input = vec![0.0f32; in_len];
    let mut a = vec![0.0f32; out_len];
    let mut b = vec![0.0f32; out_len];
    let mut checksum = 0u64;
    let mut compared = 0u64;
    let mut nonzero = 0u64;

    for pass in 0..2 {
        if pass == 1 {
            let rc_typed = if model.model_id == 1 {
                unsafe { cf_voice_reset(typed) }
            } else {
                unsafe { cf_echo_reset(typed) }
            };
            if rc_typed != 0 || unsafe { cf_model_reset(unified) } != 0 {
                fail("reset failed");
            }
        }
        for f in 0..32 {
            fill(&mut input, f);
            let rc_a = if model.model_id == 1 {
                unsafe { cf_voice_process(typed, input.as_ptr(), a.as_mut_ptr()) }
            } else {
                unsafe { cf_echo_process(typed, input.as_ptr(), a.as_mut_ptr()) }
            };
            let rc_b = unsafe { cf_model_process(unified, input.as_ptr(), b.as_mut_ptr()) };
            if rc_a != 0 || rc_b != 0 {
                fail("process returned a non-zero code");
            }
            for (x, y) in a.iter().zip(b.iter()) {
                if x.to_bits() != y.to_bits() {
                    fail("typed and unified outputs differ");
                }
                checksum = (checksum ^ u64::from(x.to_bits()))
                    .wrapping_mul(0x0000_0100_0000_01B3);
                if x.to_bits() & 0x7fff_ffff != 0 {
                    nonzero += 1;
                }
                compared += 1;
            }
        }
    }

    println!(
        "{}: id={} in={} out={} backend={} workspace={} B, {} frames x 2 passes, \
         {} values compared ({} non-zero), checksum={:016x}",
        model.name,
        got_id,
        got_in,
        got_out,
        backend,
        ws,
        64,
        compared,
        nonzero,
        checksum
    );

    unsafe {
        if model.model_id == 1 {
            cf_voice_free(typed);
        } else {
            cf_echo_free(typed);
        }
        cf_model_free(unified);
    }
}

fn load(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        process::exit(2);
    })
}

fn main() {
    let mut args = std::env::args().skip(1);
    let echo_path = args.next().unwrap_or_else(|| "models/echo-default.cfwm".into());
    let voice_path = args.next().unwrap_or_else(|| "models/voice-default.cfwm".into());

    println!("cleanfield native ABI version {}", unsafe { cf_abi_version() });

    let echo = load(&echo_path);
    run(
        &Run {
            name: "echo",
            model_id: 2,
            frames: [1539, 513],
        },
        &echo,
    );

    let voice = load(&voice_path);
    run(
        &Run {
            name: "voice",
            model_id: 1,
            frames: [513, 1026],
        },
        &voice,
    );

    println!("RESULT: ABI link example passed");
}
