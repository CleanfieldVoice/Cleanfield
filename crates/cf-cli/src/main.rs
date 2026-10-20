mod inflate;
mod npz;
mod rt;

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use cf_core::fma::{FmaOps, NativeFma, SoftFma};
use cf_core::rcp::{rcp_ps_bits, RCP_TABLE};
use cf_core::sha256;
use cf_core::Backend;
use cf_echo::{EchoModel, EchoObserver, INPUT_LEN as ECHO_INPUT_LEN, LAYERS, LAYER_LEN};
use cf_voice::{
    VoiceModel, DIAG_F32_SLOTS, DIAG_I16_SLOTS, INPUT_LEN, L18_LEN, STAGES, STAGE_LABELS,
};

const DEFAULT_CFWM: &str = "models/voice-default.cfwm";
const DEFAULT_ECHO_CFWM: &str = "models/echo-default.cfwm";

// ---------------------------------------------------------------------------
// allocation counter (bench reports the steady-state delta)
// ---------------------------------------------------------------------------
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("check") => cmd_check(&args[1..]),
        Some("simd-check") => cmd_simd_check(&args[1..]),
        Some("bench") => cmd_bench(&args[1..]),
        Some("dump") => cmd_dump(&args[1..]),
        Some("fma-check") => cmd_fma_check(&args[1..]),
        Some("rcp-check") => cmd_rcp_check(&args[1..]),
        Some("exp-check") => cmd_exp_check(&args[1..]),
        Some("info") => cmd_info(&args[1..]),
        Some("rt") => rt::cmd_rt(&args[1..]),
        _ => {
            usage();
            2
        }
    };
    process::exit(code);
}

fn usage() {
    eprintln!("CleanField CLI");
    eprintln!(
        "  check voice <reference.npz|.cfrf> [--cfwm PATH] [--limit N] [--backend scalar|avx2]"
    );
    eprintln!(
        "  check echo <reference.cfrf|.npz> [--cfwm PATH] [--limit N] \
         [--backend auto|scalar|avx2]"
    );
    eprintln!("  simd-check voice|echo <reference.npz|.cfrf> [--cfwm PATH] [--limit N]");
    eprintln!("  bench voice <frames> [--cfwm PATH] [--backend scalar|avx2] [--per-layer]");
    eprintln!("  bench echo <frames> [--cfwm PATH] [--backend auto|scalar|avx2] [--per-layer]");
    eprintln!("  dump voice <reference.npz|.cfrf> --out PATH [--limit N] [--layers]");
    eprintln!("  dump echo <reference.cfrf|.npz> --out PATH [--limit N] [--layers]");
    eprintln!("  fma-check [--cases N] [--seed N]");
    eprintln!("  rcp-check [--npy PATH]");
    eprintln!("  exp-check echo <reference.cfrf|.npz> [--limit N] [--cases N] [--seed N]");
    eprintln!("  info voice|echo [--cfwm PATH]");
    eprintln!(
        "  rt voice|echo [--period-ms 16|10] [--soak MINUTES | --seconds S | --frames N] \
         [--backend auto|scalar|avx2]"
    );
}

fn is_echo(alias: &str) -> bool {
    matches!(alias, "echo" | "echo-default")
}

fn echo_model_path(args: &[String]) -> String {
    take_flag(args, "--cfwm")
        .or_else(|| take_flag(args, "--model"))
        .unwrap_or_else(|| DEFAULT_ECHO_CFWM.to_string())
}

fn fail(msg: impl AsRef<str>) -> i32 {
    eprintln!("error: {}", msg.as_ref());
    2
}

fn take_flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn positional(args: &[String]) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") {
            // Value-less switches (there are only two) consume one token;
            // every other flag consumes its value as well.
            i += if args[i] == "--layers" || args[i] == "--per-layer" {
                1
            } else {
                2
            };
        } else {
            out.push(args[i].as_str());
            i += 1;
        }
    }
    out
}

fn model_path(args: &[String]) -> String {
    take_flag(args, "--cfwm")
        .or_else(|| take_flag(args, "--model"))
        .unwrap_or_else(|| DEFAULT_CFWM.to_string())
}

fn check_alias(alias: &str) -> Result<(), String> {
    match alias {
        "voice" | "voice-default" => Ok(()),
        other => Err(format!("unknown model {other:?} (expected voice)")),
    }
}

fn load_model(path: &str) -> Result<VoiceModel, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    VoiceModel::new(&bytes).map_err(|e| format!("cannot load {path}: {e}"))
}

/// Parse `--backend auto|scalar|avx2`.  `None` means "let the model choose".
fn requested_backend(args: &[String]) -> Result<Option<Backend>, String> {
    match take_flag(args, "--backend").as_deref() {
        None | Some("auto") => Ok(None),
        Some("scalar") => Ok(Some(Backend::Scalar)),
        Some("avx2") => Ok(Some(Backend::Avx2)),
        Some(other) => Err(format!(
            "unknown backend {other:?} (expected auto, scalar or avx2)"
        )),
    }
}

fn load_model_with(path: &str, backend: Option<Backend>) -> Result<VoiceModel, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let model = match backend {
        Some(b) => VoiceModel::with_backend(&bytes, b),
        None => VoiceModel::new(&bytes),
    };
    model.map_err(|e| format!("cannot load {path}: {e}"))
}

/// Parse `--backend auto|scalar|avx2` for an echo command.
/// `None` means "use `default`".
fn echo_backend_arg(args: &[String], default: Backend) -> Result<Backend, String> {
    match take_flag(args, "--backend").as_deref() {
        None => Ok(default),
        Some("auto") => Ok(Backend::detect()),
        Some("scalar") => Ok(Backend::Scalar),
        Some("avx2") => Ok(Backend::Avx2),
        Some(other) => Err(format!(
            "unknown backend {other:?} (expected auto, scalar or avx2)"
        )),
    }
}

fn load_echo_with(path: &str, backend: Backend) -> Result<EchoModel, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    EchoModel::with_backend(&bytes, backend).map_err(|e| format!("cannot load {path}: {e}"))
}

fn dtype_name(dtype: u8) -> &'static str {
    match dtype {
        0 => "f32",
        1 => "i8",
        2 => "i16",
        3 => "u32",
        4 => "f16",
        _ => "?",
    }
}

// ---------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    F32,
    I16,
}

#[derive(Clone, Copy)]
struct RowSpec {
    label: &'static str,
    array: &'static str,
    kind: Kind,
    slot: &'static str,
}

/// Every array the reference dump can carry, in layer order.
///
/// Fused/identity layers are separate rows because the reference captures both
/// buffers; they alias the same runtime slot.
const ROWS: &[RowSpec] = &[
    RowSpec {
        label: "L0",
        array: "L0",
        kind: Kind::F32,
        slot: "L0",
    },
    RowSpec {
        label: "L1",
        array: "L1",
        kind: Kind::F32,
        slot: "L1",
    },
    RowSpec {
        label: "L2",
        array: "L2",
        kind: Kind::F32,
        slot: "L2",
    },
    RowSpec {
        label: "L3",
        array: "L3",
        kind: Kind::F32,
        slot: "L3",
    },
    RowSpec {
        label: "L4",
        array: "L4",
        kind: Kind::F32,
        slot: "L4",
    },
    RowSpec {
        label: "L5",
        array: "L5",
        kind: Kind::F32,
        slot: "L5",
    },
    RowSpec {
        label: "L6",
        array: "L6",
        kind: Kind::F32,
        slot: "L6",
    },
    RowSpec {
        label: "L7",
        array: "L7",
        kind: Kind::F32,
        slot: "L7",
    },
    RowSpec {
        label: "L8",
        array: "L8",
        kind: Kind::F32,
        slot: "L8",
    },
    RowSpec {
        label: "L9",
        array: "L9",
        kind: Kind::F32,
        slot: "L9",
    },
    RowSpec {
        label: "L10",
        array: "L10",
        kind: Kind::F32,
        slot: "L10",
    },
    RowSpec {
        label: "L11",
        array: "L11",
        kind: Kind::F32,
        slot: "L11",
    },
    RowSpec {
        label: "L12",
        array: "L12",
        kind: Kind::F32,
        slot: "L12",
    },
    RowSpec {
        label: "L13",
        array: "L13",
        kind: Kind::F32,
        slot: "L13",
    },
    RowSpec {
        label: "L14",
        array: "L14",
        kind: Kind::F32,
        slot: "L14",
    },
    RowSpec {
        label: "L15_x",
        array: "L15_x",
        kind: Kind::F32,
        slot: "L14",
    },
    RowSpec {
        label: "L15_qx",
        array: "L15_qx",
        kind: Kind::I16,
        slot: "L15_qx",
    },
    RowSpec {
        label: "L15_qh",
        array: "L15_qh",
        kind: Kind::I16,
        slot: "L15_qh",
    },
    RowSpec {
        label: "L15_c",
        array: "L15_c",
        kind: Kind::F32,
        slot: "L15_c",
    },
    RowSpec {
        label: "L16_qx",
        array: "L16_qx",
        kind: Kind::I16,
        slot: "L16_qx",
    },
    RowSpec {
        label: "L16_qh",
        array: "L16_qh",
        kind: Kind::I16,
        slot: "L16_qh",
    },
    RowSpec {
        label: "L16_c",
        array: "L16_c",
        kind: Kind::F32,
        slot: "L16_c",
    },
    RowSpec {
        label: "L17_qx",
        array: "L17_qx",
        kind: Kind::I16,
        slot: "L17_qx",
    },
    RowSpec {
        label: "L17",
        array: "L17",
        kind: Kind::F32,
        slot: "L17",
    },
    RowSpec {
        label: "L18_qx",
        array: "L18_qx",
        kind: Kind::I16,
        slot: "L18_qx",
    },
    RowSpec {
        label: "L18",
        array: "L18",
        kind: Kind::F32,
        slot: "L18",
    },
    RowSpec {
        label: "out",
        array: "out",
        kind: Kind::F32,
        slot: "L18",
    },
    RowSpec {
        label: "history",
        array: "history",
        kind: Kind::F32,
        slot: "history",
    },
];

enum RefValues {
    F32(Vec<f32>),
    I16(Vec<i16>),
}

impl RefValues {
    fn len(&self) -> usize {
        match self {
            RefValues::F32(v) => v.len(),
            RefValues::I16(v) => v.len(),
        }
    }
}

struct LayerStat {
    label: &'static str,
    total: u64,
    exact: u64,
    examples: Vec<String>,
}

fn cmd_check(args: &[String]) -> i32 {
    let pos = positional(args);
    if pos.len() < 2 {
        return fail("usage: check voice <reference.npz|.cfrf>");
    }
    if is_echo(pos[0]) {
        return cmd_check_echo(args);
    }
    if let Err(e) = check_alias(pos[0]) {
        return fail(e);
    }
    let ref_path = pos[1];
    let limit: Option<usize> = take_flag(args, "--limit").and_then(|s| s.parse().ok());
    let cfwm_path = model_path(args);
    let requested = match requested_backend(args) {
        Ok(b) => b,
        Err(e) => return fail(e),
    };

    let mut voice = match load_model_with(&cfwm_path, requested) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    let ref_bytes = match fs::read(ref_path) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {ref_path}: {e}")),
    };
    let reference = match npz::RefData::open(&ref_bytes) {
        Ok(r) => r,
        Err(e) => return fail(format!("cannot parse {ref_path}: {e}")),
    };

    println!(
        "model: id={} tensors={} backend={} workspace_floats={} workspace_i16={} workspace_i32={}",
        voice.container().model_id(),
        voice.container().tensor_count(),
        voice.backend_name(),
        voice.workspace_floats(),
        voice.workspace_i16(),
        voice.workspace_i32()
    );

    if reference.f32("frames").is_some() {
        check_layers(&mut voice, &reference, ref_path, limit)
    } else if reference.u32("frames").is_some() && reference.f32("features").is_some() {
        check_real_audio(&mut voice, &reference, ref_path, limit)
    } else {
        fail(format!(
            "{ref_path}: expected an f32 'frames' array (synthetic) or f32 'features' + u32 'frames' (real audio)"
        ))
    }
}

fn check_layers(
    voice: &mut VoiceModel,
    reference: &npz::RefData,
    ref_path: &str,
    limit: Option<usize>,
) -> i32 {
    let frames = match reference.f32("frames") {
        Some(v) => v,
        None => return fail("reference has no f32 'frames' array"),
    };
    let frame_shape = reference.shape("frames").unwrap_or_default();
    if frame_shape.len() != 2 || frame_shape[1] != INPUT_LEN {
        return fail(format!(
            "reference 'frames' shape {frame_shape:?} is not (n, {INPUT_LEN})"
        ));
    }
    let n_frames = frame_shape[0].min(limit.unwrap_or(usize::MAX));
    if frames.len() < n_frames * INPUT_LEN {
        return fail("reference 'frames' payload is shorter than its shape");
    }

    let mut rows: Vec<(RowSpec, usize, RefValues)> = Vec::new();
    let mut stats: Vec<LayerStat> = Vec::new();
    let mut missing: Vec<&'static str> = Vec::new();
    for spec in ROWS {
        let (width, values) = match spec.kind {
            Kind::F32 => match reference.f32(spec.array) {
                Some(v) => {
                    let shape = reference.shape(spec.array).unwrap_or_default();
                    let width = if shape.len() >= 2 {
                        shape[shape.len() - 1]
                    } else {
                        1
                    };
                    (width, RefValues::F32(v))
                }
                None => {
                    missing.push(spec.array);
                    continue;
                }
            },
            Kind::I16 => match reference.i16(spec.array) {
                Some(v) => {
                    let shape = reference.shape(spec.array).unwrap_or_default();
                    let width = if shape.len() >= 2 {
                        shape[shape.len() - 1]
                    } else {
                        1
                    };
                    (width, RefValues::I16(v))
                }
                None => {
                    missing.push(spec.array);
                    continue;
                }
            },
        };
        let runtime_len = match spec.kind {
            Kind::F32 => voice.slot(spec.slot).map(|s| s.len()).unwrap_or(0),
            Kind::I16 => voice.slot_i16(spec.slot).map(|s| s.len()).unwrap_or(0),
        };
        if runtime_len == 0 {
            return fail(format!("runtime has no slot {:?}", spec.slot));
        }
        if runtime_len != width {
            return fail(format!(
                "reference {:?} width {width} != runtime slot {:?} width {runtime_len}",
                spec.array, spec.slot
            ));
        }
        if values.len() < n_frames * width {
            return fail(format!(
                "reference {:?} has {} values, needs {}",
                spec.array,
                values.len(),
                n_frames * width
            ));
        }
        stats.push(LayerStat {
            label: spec.label,
            total: 0,
            exact: 0,
            examples: Vec::new(),
        });
        rows.push((*spec, width, values));
    }

    if !missing.is_empty() {
        println!(
            "note: {} reference array(s) absent and skipped: {}",
            missing.len(),
            missing.join(", ")
        );
    }

    for f in 0..n_frames {
        let frame = &frames[f * INPUT_LEN..(f + 1) * INPUT_LEN];
        if let Err(e) = voice.process(frame) {
            return fail(format!("frame {f}: {e}"));
        }
        for (ri, (spec, width, values)) in rows.iter().enumerate() {
            let base = f * width;
            let stat = &mut stats[ri];
            match values {
                RefValues::F32(want) => {
                    let got = voice.slot(spec.slot).expect("validated");
                    for k in 0..*width {
                        let g = got[k];
                        let w = want[base + k];
                        stat.total += 1;
                        if g.to_bits() == w.to_bits() {
                            stat.exact += 1;
                        } else if stat.examples.len() < 5 {
                            stat.examples.push(format!(
                                "frame {f} idx {k}: got 0x{:08x} ({g:e}) want 0x{:08x} ({w:e})",
                                g.to_bits(),
                                w.to_bits()
                            ));
                        }
                    }
                }
                RefValues::I16(want) => {
                    let got = voice.slot_i16(spec.slot).expect("validated");
                    for k in 0..*width {
                        let g = got[k];
                        let w = want[base + k];
                        stat.total += 1;
                        if g == w {
                            stat.exact += 1;
                        } else if stat.examples.len() < 5 {
                            stat.examples.push(format!(
                                "frame {f} idx {k}: got {g} want {w} (diff {})",
                                g as i32 - w as i32
                            ));
                        }
                    }
                }
            }
        }
    }

    println!("input: {ref_path} (synthetic frames, {n_frames} frames)");
    println!();
    println!(
        "{:<10} {:>12} {:>12} {:>10}  verdict",
        "layer", "compared", "exact", "mismatch"
    );
    let mut all_ok = true;
    let mut total = 0u64;
    let mut exact = 0u64;
    for s in &stats {
        let mism = s.total - s.exact;
        total += s.total;
        exact += s.exact;
        let verdict = if mism == 0 { "exact" } else { "MISMATCH" };
        println!(
            "{:<10} {:>12} {:>12} {:>10}  {}",
            s.label, s.total, s.exact, mism, verdict
        );
        if mism != 0 {
            all_ok = false;
            for e in &s.examples {
                println!("    {e}");
            }
        }
    }
    println!();
    println!("total: {exact}/{total} values compared");
    let complete = rows.len() == ROWS.len();
    if all_ok && complete {
        println!(
            "RESULT: u32-exact for all 19 layers ({} captured arrays), quantised state and input staging over {n_frames} frames",
            stats.len()
        );
        0
    } else if all_ok {
        println!(
            "RESULT: INCOMPLETE - {} of {} captured arrays present (missing: {})",
            rows.len(),
            ROWS.len(),
            missing.join(", ")
        );
        2
    } else {
        println!("RESULT: MISMATCH");
        1
    }
}

fn check_real_audio(
    voice: &mut VoiceModel,
    reference: &npz::RefData,
    ref_path: &str,
    limit: Option<usize>,
) -> i32 {
    let features = match reference.f32("features") {
        Some(v) => v,
        None => return fail("reference has no f32 'features' array"),
    };
    let fshape = reference.shape("features").unwrap_or_default();
    if fshape.len() != 2 || fshape[1] != INPUT_LEN {
        return fail(format!(
            "reference 'features' shape {fshape:?} is not (n, {INPUT_LEN})"
        ));
    }
    let expected = match reference.u32("frames") {
        Some(v) => v,
        None => return fail("reference has no u32 'frames' array"),
    };
    let eshape = reference.shape("frames").unwrap_or_default();
    if eshape.len() != 2 || eshape[1] != L18_LEN {
        return fail(format!(
            "reference 'frames' shape {eshape:?} is not (n, {L18_LEN})"
        ));
    }
    let n_frames = fshape[0].min(eshape[0]).min(limit.unwrap_or(usize::MAX));
    if features.len() < n_frames * INPUT_LEN || expected.len() < n_frames * L18_LEN {
        return fail("real-audio dump is shorter than its declared shape");
    }

    let mut total = 0u64;
    let mut exact = 0u64;
    let mut examples = Vec::new();
    for f in 0..n_frames {
        let frame = &features[f * INPUT_LEN..(f + 1) * INPUT_LEN];
        if let Err(e) = voice.process(frame) {
            return fail(format!("frame {f}: {e}"));
        }
        let got = voice.output();
        for k in 0..L18_LEN {
            let g = got[k];
            let w = expected[f * L18_LEN + k];
            total += 1;
            if g.to_bits() == w {
                exact += 1;
            } else if examples.len() < 5 {
                examples.push(format!(
                    "frame {f} idx {k}: got 0x{:08x} ({g:e}) want 0x{w:08x}",
                    g.to_bits()
                ));
            }
        }
    }

    println!("input: {ref_path} (real-audio features, {n_frames} frames)");
    println!();
    println!(
        "{:<10} {:>12} {:>12} {:>10}  verdict",
        "output", "compared", "exact", "mismatch"
    );
    let mism = total - exact;
    println!(
        "{:<10} {:>12} {:>12} {:>10}  {}",
        "out",
        total,
        exact,
        mism,
        if mism == 0 { "exact" } else { "MISMATCH" }
    );
    for e in &examples {
        println!("    {e}");
    }
    println!();
    if mism == 0 {
        println!("RESULT: u32-exact for the final {L18_LEN} outputs over {n_frames} frames");
        0
    } else {
        println!("RESULT: MISMATCH");
        1
    }
}

// ---------------------------------------------------------------------------
// echo: 89-layer model verification, benchmark and info
// ---------------------------------------------------------------------------
/// Named `f32` diagnostics the echo model reports during a frame.
const ECHO_F32_DIAGS: &[&str] = &[
    "staged", "history", "L59_yih", "L59_yhh", "L59_z", "L59_r", "L59_n", "L59_A_in", "L59_A_h",
    "L60_yih", "L60_yhh", "L60_z", "L60_r", "L60_n", "L60_A_in", "L60_A_h", "L61_A", "L85_A",
    "L87_A",
];

/// Named quantised diagnostics the echo model reports during a frame.
const ECHO_I16_DIAGS: &[&str] = &[
    "L59_qx", "L59_qh", "L60_qx", "L60_qh", "L61_qx", "L85_qx", "L87_qx",
];

struct EchoRow {
    label: String,
    want: Vec<f32>,
    width: usize,
    total: u64,
    exact: u64,
    examples: Vec<String>,
}

fn compare_f32(row: &mut EchoRow, frame: usize, got: &[f32]) {
    let base = frame * row.width;
    let n = row.width.min(got.len());
    for k in 0..n {
        let g = got[k];
        let w = row.want[base + k];
        row.total += 1;
        if g.to_bits() == w.to_bits() {
            row.exact += 1;
        } else if row.examples.len() < 5 {
            row.examples.push(format!(
                "frame {frame} idx {k}: got 0x{:08x} ({g:e}) want 0x{:08x} ({w:e})",
                g.to_bits(),
                w.to_bits()
            ));
        }
    }
}

fn compare_i16_f32(row: &mut EchoRow, frame: usize, got: &[i16]) {
    let base = frame * row.width;
    let n = row.width.min(got.len());
    for k in 0..n {
        let g = got[k] as f32;
        let w = row.want[base + k];
        row.total += 1;
        if g.to_bits() == w.to_bits() {
            row.exact += 1;
        } else if row.examples.len() < 5 {
            row.examples.push(format!(
                "frame {frame} idx {k}: got {} want {}",
                got[k], w as i32
            ));
        }
    }
}

struct EchoCheck<'a> {
    frame: usize,
    layer_row: &'a [Option<usize>],
    diag_row: &'a HashMap<&'static str, usize>,
    rows: &'a mut [EchoRow],
}

impl EchoObserver for EchoCheck<'_> {
    fn layer(&mut self, index: usize, values: &[f32]) {
        if let Some(ri) = self.layer_row.get(index).copied().flatten() {
            compare_f32(&mut self.rows[ri], self.frame, values);
        }
    }

    fn diag_f32(&mut self, name: &'static str, values: &[f32]) {
        if let Some(&ri) = self.diag_row.get(name) {
            compare_f32(&mut self.rows[ri], self.frame, values);
        }
    }

    fn diag_i16(&mut self, name: &'static str, values: &[i16]) {
        if let Some(&ri) = self.diag_row.get(name) {
            compare_i16_f32(&mut self.rows[ri], self.frame, values);
        }
    }
}

/// Read the 3 x 513 input frames of an echo reference dump.
fn echo_reference_frames(
    reference: &npz::RefData,
    ref_path: &str,
    limit: Option<usize>,
) -> Result<(Vec<f32>, usize), String> {
    let data = reference
        .f32("frames")
        .ok_or_else(|| format!("{ref_path}: no f32 'frames' array"))?;
    let shape = reference.shape("frames").unwrap_or_default();
    if shape.len() < 2 || shape[0] == 0 {
        return Err(format!(
            "{ref_path}: 'frames' shape {shape:?} is not (n, ...)"
        ));
    }
    let n_declared = shape[0];
    let per_frame: usize = shape[1..].iter().product();
    if per_frame != ECHO_INPUT_LEN {
        return Err(format!(
            "{ref_path}: 'frames' shape {shape:?} carries {per_frame} values per frame, \
             expected {ECHO_INPUT_LEN} (3 x 513)"
        ));
    }
    let n = n_declared.min(limit.unwrap_or(usize::MAX));
    if data.len() < n * ECHO_INPUT_LEN {
        return Err(format!(
            "{ref_path}: 'frames' payload is shorter than its shape"
        ));
    }
    Ok((data, n))
}

fn cmd_check_echo(args: &[String]) -> i32 {
    let pos = positional(args);
    if pos.len() < 2 {
        return fail("usage: check echo <reference.cfrf|.npz>");
    }
    let ref_path = pos[1];
    let limit: Option<usize> = take_flag(args, "--limit").and_then(|s| s.parse().ok());
    let cfwm_path = echo_model_path(args);
    let backend = match echo_backend_arg(args, Backend::Scalar) {
        Ok(b) => b,
        Err(e) => return fail(e),
    };

    let mut model = match load_echo_with(&cfwm_path, backend) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    let ref_bytes = match fs::read(ref_path) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {ref_path}: {e}")),
    };
    let reference = match npz::RefData::open(&ref_bytes) {
        Ok(r) => r,
        Err(e) => return fail(format!("cannot parse {ref_path}: {e}")),
    };
    let (frames, n_frames) = match echo_reference_frames(&reference, ref_path, limit) {
        Ok(v) => v,
        Err(e) => return fail(e),
    };

    println!("backend: {}", model.backend_name());
    println!(
        "model: id={} tensors={} workspace_floats={} workspace_i16={} workspace_i32={} workspace_i64={}",
        model.container().model_id(),
        model.container().tensor_count(),
        model.workspace_floats(),
        model.workspace_i16(),
        model.workspace_i32(),
        model.workspace_i64()
    );

    let mut rows: Vec<EchoRow> = Vec::new();
    let mut layer_row: Vec<Option<usize>> = vec![None; LAYERS];
    let mut missing_layers: Vec<String> = Vec::new();
    for (i, &width) in LAYER_LEN.iter().enumerate() {
        let name = format!("L{i}");
        match reference.f32(&name) {
            Some(v) => {
                if v.len() < n_frames * width {
                    return fail(format!(
                        "reference {name} has {} values, needs {}",
                        v.len(),
                        n_frames * width
                    ));
                }
                layer_row[i] = Some(rows.len());
                rows.push(EchoRow {
                    label: name,
                    want: v,
                    width,
                    total: 0,
                    exact: 0,
                    examples: Vec::new(),
                });
            }
            None => missing_layers.push(name),
        }
    }

    let mut diag_row: HashMap<&'static str, usize> = HashMap::new();
    let mut missing_diags: Vec<&'static str> = Vec::new();
    for &name in ECHO_F32_DIAGS.iter().chain(ECHO_I16_DIAGS.iter()) {
        match reference.f32(name) {
            Some(v) => {
                if v.len() % n_frames != 0 {
                    return fail(format!(
                        "reference {name} width is not a multiple of frames"
                    ));
                }
                let width = v.len() / n_frames;
                diag_row.insert(name, rows.len());
                rows.push(EchoRow {
                    label: name.to_string(),
                    want: v,
                    width,
                    total: 0,
                    exact: 0,
                    examples: Vec::new(),
                });
            }
            None => missing_diags.push(name),
        }
    }

    let out_row = match reference.f32("out") {
        Some(v) => {
            if v.len() < n_frames * 513 {
                return fail("reference 'out' is shorter than its shape");
            }
            let idx = rows.len();
            rows.push(EchoRow {
                label: "out".to_string(),
                want: v,
                width: 513,
                total: 0,
                exact: 0,
                examples: Vec::new(),
            });
            Some(idx)
        }
        None => None,
    };

    if !missing_layers.is_empty() {
        println!(
            "note: {} layer array(s) absent and skipped: {}",
            missing_layers.len(),
            missing_layers.join(", ")
        );
    }
    if !missing_diags.is_empty() {
        println!(
            "note: {} diagnostic array(s) absent (layers and outputs are still checked)",
            missing_diags.len()
        );
    }

    for f in 0..n_frames {
        let frame = &frames[f * ECHO_INPUT_LEN..(f + 1) * ECHO_INPUT_LEN];
        {
            let mut obs = EchoCheck {
                frame: f,
                layer_row: &layer_row,
                diag_row: &diag_row,
                rows: &mut rows,
            };
            if let Err(e) = model.process_with(frame, &mut obs) {
                return fail(format!("frame {f}: {e}"));
            }
        }
        if let Some(ri) = out_row {
            compare_f32(&mut rows[ri], f, model.output());
        }
    }

    println!("input: {ref_path} (3 x 513 frames, {n_frames} frames)");
    println!();
    println!(
        "{:<12} {:>12} {:>12} {:>10}  verdict",
        "layer", "compared", "exact", "mismatch"
    );
    let mut all_ok = true;
    let mut total = 0u64;
    let mut exact = 0u64;
    for row in &rows {
        let mism = row.total - row.exact;
        total += row.total;
        exact += row.exact;
        let verdict = if mism == 0 { "exact" } else { "MISMATCH" };
        println!(
            "{:<12} {:>12} {:>12} {:>10}  {}",
            row.label, row.total, row.exact, mism, verdict
        );
        if mism != 0 {
            all_ok = false;
            for e in &row.examples {
                println!("    {e}");
            }
        }
    }
    println!();
    println!("total: {exact}/{total} values compared");
    let layers_complete = layer_row.iter().all(|r| r.is_some());
    let complete = layers_complete && out_row.is_some();
    if all_ok && complete {
        let extras = rows.len() - LAYERS - usize::from(out_row.is_some());
        println!(
            "RESULT: u32-exact for all {LAYERS} layers and the final {n_frames} x 513 outputs \
             over {n_frames} frames ({extras} extra diagnostic arrays)"
        );
        0
    } else if all_ok {
        println!(
            "RESULT: INCOMPLETE - {} of {LAYERS} layer arrays present (missing: {})",
            LAYERS - missing_layers.len(),
            missing_layers.join(", ")
        );
        2
    } else {
        println!("RESULT: MISMATCH");
        1
    }
}

fn cmd_bench_echo(args: &[String]) -> i32 {
    let pos = positional(args);
    if pos.len() < 2 {
        return fail("usage: bench echo <frames>");
    }
    let frames: usize = match pos[1].parse() {
        Ok(n) if n > 0 => n,
        _ => return fail("frame count must be a positive integer"),
    };
    let cfwm_path = echo_model_path(args);
    let backend = match echo_backend_arg(args, Backend::detect()) {
        Ok(b) => b,
        Err(e) => return fail(e),
    };
    let per_layer = args.iter().any(|a| a == "--per-layer");
    let mut model = match load_echo_with(&cfwm_path, backend) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };

    // Deterministic 3 x 513 input frames (benchmark input, not a numeric result).
    let mut flat = vec![0.0f32; frames * ECHO_INPUT_LEN];
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    for v in flat.iter_mut() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let u = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32;
        *v = (u / (1u32 << 24) as f32) * 2.0 - 1.0;
    }

    let warmup = 100.min(frames);
    for i in 0..warmup {
        let _ = model.process(&flat[i * ECHO_INPUT_LEN..(i + 1) * ECHO_INPUT_LEN]);
    }

    let mut trials = [0.0f64; 3];
    let mut allocs = [0usize; 3];
    for (ti, t) in trials.iter_mut().enumerate() {
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        let start = Instant::now();
        for i in 0..frames {
            if model
                .process(&flat[i * ECHO_INPUT_LEN..(i + 1) * ECHO_INPUT_LEN])
                .is_err()
            {
                return fail("benchmark frame rejected");
            }
        }
        *t = start.elapsed().as_secs_f64();
        allocs[ti] = ALLOCATIONS.load(Ordering::Relaxed) - before;
    }
    let best = trials.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = trials.iter().sum::<f64>() / trials.len() as f64;
    let checksum: f32 = model.output().iter().copied().sum();

    println!("scope: 89-layer echo model (staging + encoders + neck + 2 GRU + decoder + sigmoid)");
    println!(
        "backend={} workspace_floats={} workspace_i16={} workspace_i32={} workspace_i64={} frames_per_trial={frames}",
        model.backend_name(),
        model.workspace_floats(),
        model.workspace_i16(),
        model.workspace_i32(),
        model.workspace_i64()
    );
    for (i, t) in trials.iter().enumerate() {
        println!(
            "trial {}: {:.3} ms total, {:.4} ms/frame, {:.1} fps, allocations={}",
            i + 1,
            t * 1000.0,
            t * 1000.0 / frames as f64,
            frames as f64 / t,
            allocs[i]
        );
    }
    println!(
        "best: {:.4} ms/frame, {:.1} fps (mean over trials: {:.4} ms/frame)",
        best * 1000.0 / frames as f64,
        frames as f64 / best,
        mean * 1000.0 / frames as f64
    );
    println!(
        "allocations: {} (steady-state inference must be 0; output checksum {checksum:e})",
        allocs.iter().sum::<usize>()
    );
    if per_layer {
        // The observer is called once per layer, after that layer's buffer is
        // written; the interval between two callbacks is the work of the
        // layer(s) in between.  The first interval includes staging and the
        // layer-0 stack copy, and GRU intervals include their quantisers,
        // both GEMMs and the gate activations.
        struct LayerClock {
            last: std::time::Instant,
            times: [u64; LAYERS],
        }
        impl EchoObserver for LayerClock {
            fn layer(&mut self, index: usize, _values: &[f32]) {
                let now = std::time::Instant::now();
                if index < LAYERS {
                    self.times[index] = now.duration_since(self.last).as_nanos() as u64;
                }
                self.last = now;
            }
        }
        let n = frames.min(200).max(1);
        let mut totals = [0u64; LAYERS];
        for i in 0..n {
            let mut clock = LayerClock {
                last: std::time::Instant::now(),
                times: [0; LAYERS],
            };
            if model
                .process_with(
                    &flat[i * ECHO_INPUT_LEN..(i + 1) * ECHO_INPUT_LEN],
                    &mut clock,
                )
                .is_err()
            {
                return fail("per-layer frame rejected");
            }
            for (s, &v) in clock.times.iter().enumerate() {
                totals[s] += v;
            }
        }
        println!();
        println!(
            "per-layer timings (backend={}, mean of {n} frames, first interval includes staging):",
            model.backend_name()
        );
        println!("{:<8} {:>10} {:>12}", "layer", "values", "us/frame");
        for (s, &t) in totals.iter().enumerate() {
            println!(
                "{:<8} {:>10} {:>12.3}",
                format!("L{s}"),
                LAYER_LEN[s],
                t as f64 / 1000.0 / n as f64
            );
        }
        println!(
            "{:<8} {:>10} {:>12.3}",
            "sum",
            LAYER_LEN.iter().sum::<usize>(),
            totals.iter().sum::<u64>() as f64 / 1000.0 / n as f64
        );
    }
    0
}

fn cmd_info_echo(args: &[String]) -> i32 {
    let path = echo_model_path(args);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {path}: {e}")),
    };
    let model = match EchoModel::new(&bytes) {
        Ok(m) => m,
        Err(e) => return fail(format!("cannot load {path}: {e}")),
    };
    println!(
        "container: version={} model_id={} tensors={} data_len={}",
        model.container().format_version(),
        model.container().model_id(),
        model.container().tensor_count(),
        model.container().data_len()
    );
    println!(
        "workspace_floats: {} ({}) i16: {} i32: {} i64: {}",
        model.workspace_floats(),
        model.workspace_floats() * 4,
        model.workspace_i16(),
        model.workspace_i32(),
        model.workspace_i64()
    );
    println!(
        "{:<6} {:<6} {:<6} {:>24} {:>10}",
        "tag", "dtype", "rank", "dims", "bytes"
    );
    for t in model.container().tensors() {
        println!(
            "{:<6} {:<6} {:<6} {:>24} {:>10}",
            t.tag,
            dtype_name(t.dtype),
            t.rank,
            format!("{:?}", &t.dims[..t.rank as usize]),
            t.byte_len
        );
    }
    println!("container bytes: {}", model.container().file_len());
    0
}

fn cmd_dump_echo(args: &[String]) -> i32 {
    let pos = positional(args);
    if pos.len() < 2 {
        return fail("usage: dump echo <reference.cfrf|.npz> --out PATH [--limit N] [--layers]");
    }
    let ref_path = pos[1];
    let out_path = match take_flag(args, "--out") {
        Some(p) => p,
        None => return fail("dump: --out PATH is required"),
    };
    let limit: Option<usize> = take_flag(args, "--limit").and_then(|s| s.parse().ok());
    let want_layers = args.iter().any(|a| a == "--layers");
    let cfwm_path = echo_model_path(args);

    let bytes = match fs::read(&cfwm_path) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {cfwm_path}: {e}")),
    };
    let mut model = match EchoModel::new(&bytes) {
        Ok(m) => m,
        Err(e) => return fail(format!("cannot load {cfwm_path}: {e}")),
    };
    if cf_core::fma::is_soft() {
        return fail(
            "dump: this build routes fma() through the software path; build cf-cli without the \
             workspace-wide force-soft-fma feature so the oracle is the native scalar path",
        );
    }
    let ref_bytes = match fs::read(ref_path) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {ref_path}: {e}")),
    };
    let reference = match npz::RefData::open(&ref_bytes) {
        Ok(r) => r,
        Err(e) => return fail(format!("cannot parse {ref_path}: {e}")),
    };
    let (frames, n_frames) = match echo_reference_frames(&reference, ref_path, limit) {
        Ok(v) => v,
        Err(e) => return fail(e),
    };
    const ECHO_DUMP_MAX_LAYER_FRAMES: usize = 64;
    if want_layers && n_frames > ECHO_DUMP_MAX_LAYER_FRAMES {
        return fail(format!(
            "--layers is limited to {ECHO_DUMP_MAX_LAYER_FRAMES} frames (got {n_frames}); use --limit"
        ));
    }

    // Layer table: 89 graph layers, then the named f32 and i16 diagnostics.
    struct DumpRow {
        name: String,
        kind: u8,
        len: usize,
    }
    let mut table: Vec<DumpRow> = Vec::new();
    let mut row_by_layer = vec![0usize; LAYERS];
    if want_layers {
        for (i, &len) in LAYER_LEN.iter().enumerate() {
            row_by_layer[i] = table.len();
            table.push(DumpRow {
                name: format!("L{i}"),
                kind: 0,
                len,
            });
        }
    }
    let diag_widths: &[(&str, u8, usize)] = &[
        ("staged", 0, ECHO_INPUT_LEN),
        ("history", 0, 6 * ECHO_INPUT_LEN),
        ("L59_yih", 0, 384),
        ("L59_yhh", 0, 384),
        ("L59_z", 0, 128),
        ("L59_r", 0, 128),
        ("L59_n", 0, 128),
        ("L59_A_in", 0, 1),
        ("L59_A_h", 0, 1),
        ("L60_yih", 0, 384),
        ("L60_yhh", 0, 384),
        ("L60_z", 0, 128),
        ("L60_r", 0, 128),
        ("L60_n", 0, 128),
        ("L60_A_in", 0, 1),
        ("L60_A_h", 0, 1),
        ("L61_A", 0, 1),
        ("L85_A", 0, 1),
        ("L87_A", 0, 1),
        ("L59_qx", 1, 2112),
        ("L59_qh", 1, 128),
        ("L60_qx", 1, 128),
        ("L60_qh", 1, 128),
        ("L61_qx", 1, 128),
        ("L85_qx", 1, 513),
        ("L87_qx", 1, 513),
    ];
    let mut row_by_name: HashMap<&'static str, usize> = HashMap::new();
    if want_layers {
        for &(name, kind, len) in diag_widths {
            let key = ECHO_F32_DIAGS
                .iter()
                .chain(ECHO_I16_DIAGS.iter())
                .find(|&&n| n == name)
                .copied()
                .expect("diagnostic name must be a model key");
            row_by_name.insert(key, table.len());
            table.push(DumpRow {
                name: name.to_string(),
                kind,
                len,
            });
        }
    }

    struct DumpObserver<'a> {
        row_by_layer: &'a [usize],
        row_by_name: &'a HashMap<&'static str, usize>,
        slots: &'a mut [Option<Vec<u8>>],
    }
    impl EchoObserver for DumpObserver<'_> {
        fn layer(&mut self, index: usize, values: &[f32]) {
            let row = self.row_by_layer[index];
            let slot = &mut self.slots[row];
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for v in values {
                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            *slot = Some(bytes);
        }
        fn diag_f32(&mut self, name: &'static str, values: &[f32]) {
            if let Some(&row) = self.row_by_name.get(name) {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for v in values {
                    bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                self.slots[row] = Some(bytes);
            }
        }
        fn diag_i16(&mut self, name: &'static str, values: &[i16]) {
            if let Some(&row) = self.row_by_name.get(name) {
                let mut bytes = Vec::with_capacity(values.len() * 2);
                for v in values {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                self.slots[row] = Some(bytes);
            }
        }
    }

    let mut outputs: Vec<u32> = Vec::with_capacity(n_frames * 513);
    let mut layer_bytes: Vec<u8> = Vec::new();
    for f in 0..n_frames {
        let frame = &frames[f * ECHO_INPUT_LEN..(f + 1) * ECHO_INPUT_LEN];
        if want_layers {
            let mut slots: Vec<Option<Vec<u8>>> = (0..table.len()).map(|_| None).collect();
            {
                let mut obs = DumpObserver {
                    row_by_layer: &row_by_layer,
                    row_by_name: &row_by_name,
                    slots: &mut slots,
                };
                if let Err(e) = model.process_with(frame, &mut obs) {
                    return fail(format!("frame {f}: {e}"));
                }
            }
            for (i, slot) in slots.iter_mut().enumerate() {
                match slot.take() {
                    Some(bytes) => {
                        if bytes.len() != table[i].len * if table[i].kind == 0 { 4 } else { 2 } {
                            return fail(format!(
                                "frame {f}: {} has {} bytes, table says {}",
                                table[i].name,
                                bytes.len(),
                                table[i].len
                            ));
                        }
                        layer_bytes.extend_from_slice(&bytes);
                    }
                    None => return fail(format!("frame {f}: {} was not captured", table[i].name)),
                }
            }
        } else if let Err(e) = model.process(frame) {
            return fail(format!("frame {f}: {e}"));
        }
        outputs.extend(model.output().iter().map(|v| v.to_bits()));
    }

    let file = match File::create(&out_path) {
        Ok(f) => f,
        Err(e) => return fail(format!("cannot create {out_path}: {e}")),
    };
    let mut w = BufWriter::new(file);
    let write_result = (|| -> std::io::Result<()> {
        w.write_all(DUMP_MAGIC)?;
        w.write_all(&DUMP_VERSION.to_le_bytes())?;
        let flags = if want_layers { DUMP_FLAG_LAYERS } else { 0 };
        w.write_all(&flags.to_le_bytes())?;
        w.write_all(&(n_frames as u32).to_le_bytes())?;
        w.write_all(&(ECHO_INPUT_LEN as u32).to_le_bytes())?;
        w.write_all(&(513u32).to_le_bytes())?;
        w.write_all(&(table.len() as u32).to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
        for row in &table {
            let name = row.name.as_bytes();
            w.write_all(&[row.kind, name.len() as u8])?;
            w.write_all(&0u16.to_le_bytes())?;
            w.write_all(&(row.len as u32).to_le_bytes())?;
            w.write_all(name)?;
            for _ in 0..((4 - name.len() % 4) % 4) {
                w.write_all(&[0u8])?;
            }
        }
        for &x in &frames[..n_frames * ECHO_INPUT_LEN] {
            w.write_all(&x.to_bits().to_le_bytes())?;
        }
        for &x in &outputs {
            w.write_all(&x.to_le_bytes())?;
        }
        w.write_all(&layer_bytes)?;
        w.flush()
    })();
    if let Err(e) = write_result {
        return fail(format!("cannot write {out_path}: {e}"));
    }
    let size = fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    let layer_values: usize = table.iter().map(|r| r.len).sum();

    println!("input: {ref_path} (3 x 513 frames, {n_frames} frames)");
    println!("backend: scalar (native oracle)");
    println!("fma: native (hardware)");
    println!(
        "layers: {}",
        if want_layers {
            format!("{} buffers per frame", table.len())
        } else {
            "not written".to_string()
        }
    );
    println!("wrote: {out_path} ({size} bytes, header {DUMP_HEADER_LEN} bytes)");
    println!(
        "oracle values: {} outputs{}",
        n_frames * 513,
        if want_layers {
            format!(", {} intermediate", n_frames * layer_values)
        } else {
            String::new()
        }
    );
    println!("RESULT: native scalar oracle dump complete");
    0
}

// ---------------------------------------------------------------------------
// simd-check: scalar vs AVX2, every intermediate, bit for bit
// ---------------------------------------------------------------------------
/// Pull the 513-wide input frames out of either reference dump format.
///
/// Returns the contiguous input payload, the frame count and the dump kind.
fn reference_frames(
    reference: &npz::RefData,
    ref_path: &str,
    limit: Option<usize>,
) -> Result<(Vec<f32>, usize, &'static str), String> {
    let (data, declared, kind) = if let Some(v) = reference.f32("frames") {
        let shape = reference.shape("frames").unwrap_or_default();
        if shape.len() != 2 || shape[1] != INPUT_LEN {
            return Err(format!(
                "{ref_path}: 'frames' shape {shape:?} is not (n, {INPUT_LEN})"
            ));
        }
        (v, shape[0], "synthetic")
    } else if let Some(v) = reference.f32("features") {
        let shape = reference.shape("features").unwrap_or_default();
        if shape.len() != 2 || shape[1] != INPUT_LEN {
            return Err(format!(
                "{ref_path}: 'features' shape {shape:?} is not (n, {INPUT_LEN})"
            ));
        }
        if reference.u32("frames").is_none() {
            return Err(format!("{ref_path}: real-audio dump has no u32 'frames'"));
        }
        (v, shape[0], "real-audio")
    } else {
        return Err(format!(
            "{ref_path}: expected f32 'frames' or f32 'features'"
        ));
    };
    let n = declared.min(limit.unwrap_or(usize::MAX));
    if data.len() < n * INPUT_LEN {
        return Err(format!(
            "{ref_path}: input payload is shorter than its shape"
        ));
    }
    Ok((data, n, kind))
}

fn cmd_simd_check(args: &[String]) -> i32 {
    let pos = positional(args);
    if pos.len() < 2 {
        return fail("usage: simd-check voice|echo <reference.npz|.cfrf>");
    }
    if is_echo(pos[0]) {
        return cmd_simd_check_echo(args);
    }
    if let Err(e) = check_alias(pos[0]) {
        return fail(e);
    }
    let ref_path = pos[1];
    let limit: Option<usize> = take_flag(args, "--limit").and_then(|s| s.parse().ok());
    let cfwm_path = model_path(args);

    if !Backend::Avx2.is_available() {
        return fail("avx2 backend is not available on this CPU; nothing to compare");
    }
    let mut scalar = match load_model_with(&cfwm_path, Some(Backend::Scalar)) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    let mut wide = match load_model_with(&cfwm_path, Some(Backend::Avx2)) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    let ref_bytes = match fs::read(ref_path) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {ref_path}: {e}")),
    };
    let reference = match npz::RefData::open(&ref_bytes) {
        Ok(r) => r,
        Err(e) => return fail(format!("cannot parse {ref_path}: {e}")),
    };
    let (input, n_frames, kind) = match reference_frames(&reference, ref_path, limit) {
        Ok(v) => v,
        Err(e) => return fail(e),
    };

    println!(
        "backends: scalar (reference) vs avx2; model id={} tensors={}",
        wide.container().model_id(),
        wide.container().tensor_count()
    );
    println!("input: {ref_path} ({kind}, {n_frames} frames)");
    println!();

    struct Stat {
        label: &'static str,
        total: u64,
        exact: u64,
        examples: Vec<String>,
    }
    let mut stats: Vec<Stat> = DIAG_F32_SLOTS
        .iter()
        .chain(DIAG_I16_SLOTS.iter())
        .map(|&label| Stat {
            label,
            total: 0,
            exact: 0,
            examples: Vec::new(),
        })
        .collect();

    let mut run = 0u64;
    let mut first_bad_frame: Option<usize> = None;
    for f in 0..n_frames {
        let frame = &input[f * INPUT_LEN..(f + 1) * INPUT_LEN];
        if let Err(e) = scalar.process(frame) {
            return fail(format!("scalar frame {f}: {e}"));
        }
        if let Err(e) = wide.process(frame) {
            return fail(format!("avx2 frame {f}: {e}"));
        }
        for (si, &name) in DIAG_F32_SLOTS.iter().enumerate() {
            let a = match scalar.slot(name) {
                Some(v) => v,
                None => return fail(format!("scalar has no slot {name:?}")),
            };
            let b = match wide.slot(name) {
                Some(v) => v,
                None => return fail(format!("avx2 has no slot {name:?}")),
            };
            if a.len() != b.len() {
                return fail(format!(
                    "slot {name:?}: scalar {} vs avx2 {}",
                    a.len(),
                    b.len()
                ));
            }
            let st = &mut stats[si];
            for k in 0..a.len() {
                st.total += 1;
                run += 1;
                if a[k].to_bits() == b[k].to_bits() {
                    st.exact += 1;
                } else {
                    first_bad_frame.get_or_insert(f);
                    if st.examples.len() < 3 {
                        st.examples.push(format!(
                            "frame {f} idx {k}: scalar 0x{:08x} ({:e}) avx2 0x{:08x} ({:e})",
                            a[k].to_bits(),
                            a[k],
                            b[k].to_bits(),
                            b[k]
                        ));
                    }
                }
            }
        }
        for (off, &name) in DIAG_I16_SLOTS.iter().enumerate() {
            let si = DIAG_F32_SLOTS.len() + off;
            let a = match scalar.slot_i16(name) {
                Some(v) => v,
                None => return fail(format!("scalar has no i16 slot {name:?}")),
            };
            let b = match wide.slot_i16(name) {
                Some(v) => v,
                None => return fail(format!("avx2 has no i16 slot {name:?}")),
            };
            if a.len() != b.len() {
                return fail(format!(
                    "i16 slot {name:?}: scalar {} vs avx2 {}",
                    a.len(),
                    b.len()
                ));
            }
            let st = &mut stats[si];
            for k in 0..a.len() {
                st.total += 1;
                run += 1;
                if a[k] == b[k] {
                    st.exact += 1;
                } else {
                    first_bad_frame.get_or_insert(f);
                    if st.examples.len() < 3 {
                        st.examples
                            .push(format!("frame {f} idx {k}: scalar {} avx2 {}", a[k], b[k]));
                    }
                }
            }
        }
    }

    println!(
        "{:<10} {:>12} {:>12} {:>10}  verdict",
        "buffer", "compared", "exact", "mismatch"
    );
    let mut all_ok = true;
    for s in &stats {
        let mism = s.total - s.exact;
        println!(
            "{:<10} {:>12} {:>12} {:>10}  {}",
            s.label,
            s.total,
            s.exact,
            mism,
            if mism == 0 { "exact" } else { "MISMATCH" }
        );
        if mism != 0 {
            all_ok = false;
            for e in &s.examples {
                println!("    {e}");
            }
        }
    }
    println!();
    println!(
        "total: {run} values compared across {} buffers",
        stats.len()
    );
    if all_ok {
        println!(
            "RESULT: both backends produce identical bits for every intermediate over {n_frames} frames"
        );
        0
    } else {
        if let Some(f) = first_bad_frame {
            println!("first differing frame: {f}");
        }
        println!("RESULT: MISMATCH");
        1
    }
}

// ---------------------------------------------------------------------------
// simd-check echo: scalar vs AVX2, every layer and diagnostic buffer
// ---------------------------------------------------------------------------
struct EchoCapture {
    layers: Vec<Vec<u32>>,
    f32_diag: HashMap<&'static str, Vec<u32>>,
    i16_diag: HashMap<&'static str, Vec<i16>>,
}

impl EchoCapture {
    fn new() -> EchoCapture {
        EchoCapture {
            layers: vec![Vec::new(); LAYERS],
            f32_diag: HashMap::new(),
            i16_diag: HashMap::new(),
        }
    }
}

impl EchoObserver for EchoCapture {
    fn layer(&mut self, index: usize, values: &[f32]) {
        if index < LAYERS {
            self.layers[index] = values.iter().map(|v| v.to_bits()).collect();
        }
    }
    fn diag_f32(&mut self, name: &'static str, values: &[f32]) {
        self.f32_diag
            .insert(name, values.iter().map(|v| v.to_bits()).collect());
    }
    fn diag_i16(&mut self, name: &'static str, values: &[i16]) {
        self.i16_diag.insert(name, values.to_vec());
    }
}

fn cmd_simd_check_echo(args: &[String]) -> i32 {
    let pos = positional(args);
    if pos.len() < 2 {
        return fail("usage: simd-check echo <reference.cfrf|.npz>");
    }
    let ref_path = pos[1];
    let limit: Option<usize> = take_flag(args, "--limit").and_then(|s| s.parse().ok());
    let cfwm_path = echo_model_path(args);

    if !Backend::Avx2.is_available() {
        return fail("avx2 backend is not available on this CPU; nothing to compare");
    }
    let mut scalar = match load_echo_with(&cfwm_path, Backend::Scalar) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    let mut wide = match load_echo_with(&cfwm_path, Backend::Avx2) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    let ref_bytes = match fs::read(ref_path) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {ref_path}: {e}")),
    };
    let reference = match npz::RefData::open(&ref_bytes) {
        Ok(r) => r,
        Err(e) => return fail(format!("cannot parse {ref_path}: {e}")),
    };
    let (input, n_frames) = match echo_reference_frames(&reference, ref_path, limit) {
        Ok(v) => v,
        Err(e) => return fail(e),
    };

    println!(
        "backends: {} (reference) vs {}; model id={} tensors={}",
        scalar.backend_name(),
        wide.backend_name(),
        wide.container().model_id(),
        wide.container().tensor_count()
    );
    println!("input: {ref_path} (3 x 513 frames, {n_frames} frames)");
    println!();

    struct Stat {
        label: String,
        total: u64,
        exact: u64,
        examples: Vec<String>,
    }
    let mut stats: Vec<Stat> = (0..LAYERS)
        .map(|i| Stat {
            label: format!("L{i}"),
            total: 0,
            exact: 0,
            examples: Vec::new(),
        })
        .chain(ECHO_F32_DIAGS.iter().map(|&name| Stat {
            label: name.to_string(),
            total: 0,
            exact: 0,
            examples: Vec::new(),
        }))
        .chain(ECHO_I16_DIAGS.iter().map(|&name| Stat {
            label: name.to_string(),
            total: 0,
            exact: 0,
            examples: Vec::new(),
        }))
        .collect();

    let mut total = 0u64;
    let mut exact = 0u64;
    let mut first_bad_frame: Option<usize> = None;
    for f in 0..n_frames {
        let frame = &input[f * ECHO_INPUT_LEN..(f + 1) * ECHO_INPUT_LEN];
        let mut a = EchoCapture::new();
        let mut b = EchoCapture::new();
        if let Err(e) = scalar.process_with(frame, &mut a) {
            return fail(format!("scalar frame {f}: {e}"));
        }
        if let Err(e) = wide.process_with(frame, &mut b) {
            return fail(format!("avx2 frame {f}: {e}"));
        }

        let mut cmp_f32 = |st: &mut Stat, want: &[u32], got: &[u32], label: &str| {
            if want.len() != got.len() {
                return Err(format!(
                    "{label}: scalar {} vs avx2 {}",
                    want.len(),
                    got.len()
                ));
            }
            for k in 0..want.len() {
                st.total += 1;
                total += 1;
                if want[k] == got[k] {
                    st.exact += 1;
                    exact += 1;
                } else {
                    first_bad_frame.get_or_insert(f);
                    if st.examples.len() < 3 {
                        st.examples.push(format!(
                            "frame {f} {label}[{k}]: scalar 0x{:08x} avx2 0x{:08x}",
                            want[k], got[k]
                        ));
                    }
                }
            }
            Ok(())
        };

        for i in 0..LAYERS {
            let av = a.layers[i].as_slice();
            let bv = b.layers[i].as_slice();
            if av.is_empty() || bv.is_empty() {
                return fail(format!("frame {f}: layer L{i} was not captured"));
            }
            if av.len() != LAYER_LEN[i] || bv.len() != LAYER_LEN[i] {
                return fail(format!(
                    "frame {f}: L{i} widths scalar {} avx2 {} expected {}",
                    av.len(),
                    bv.len(),
                    LAYER_LEN[i]
                ));
            }
            if let Err(e) = cmp_f32(&mut stats[i], av, bv, &format!("L{i}")) {
                return fail(format!("frame {f}: {e}"));
            }
        }
        for (di, &name) in ECHO_F32_DIAGS.iter().enumerate() {
            let si = LAYERS + di;
            let (Some(av), Some(bv)) = (a.f32_diag.get(name), b.f32_diag.get(name)) else {
                return fail(format!("frame {f}: diagnostic {name} was not captured"));
            };
            if let Err(e) = cmp_f32(&mut stats[si], av, bv, name) {
                return fail(format!("frame {f}: {e}"));
            }
        }
        for (di, &name) in ECHO_I16_DIAGS.iter().enumerate() {
            let si = LAYERS + ECHO_F32_DIAGS.len() + di;
            let (Some(av), Some(bv)) = (a.i16_diag.get(name), b.i16_diag.get(name)) else {
                return fail(format!("frame {f}: diagnostic {name} was not captured"));
            };
            if av.len() != bv.len() {
                return fail(format!(
                    "frame {f}: {name}: scalar {} vs avx2 {}",
                    av.len(),
                    bv.len()
                ));
            }
            for k in 0..av.len() {
                stats[si].total += 1;
                total += 1;
                if av[k] == bv[k] {
                    stats[si].exact += 1;
                    exact += 1;
                } else {
                    first_bad_frame.get_or_insert(f);
                    if stats[si].examples.len() < 3 {
                        stats[si].examples.push(format!(
                            "frame {f} {name}[{k}]: scalar {} avx2 {}",
                            av[k], bv[k]
                        ));
                    }
                }
            }
        }
    }

    println!(
        "{:<10} {:>12} {:>12} {:>10}  verdict",
        "buffer", "compared", "exact", "mismatch"
    );
    let mut all_ok = true;
    for s in &stats {
        let mism = s.total - s.exact;
        println!(
            "{:<10} {:>12} {:>12} {:>10}  {}",
            s.label,
            s.total,
            s.exact,
            mism,
            if mism == 0 { "exact" } else { "MISMATCH" }
        );
        if mism != 0 {
            all_ok = false;
            for e in &s.examples {
                println!("    {e}");
            }
        }
    }
    println!();
    println!(
        "total: {exact}/{total} values compared across {} buffers",
        stats.len()
    );
    if all_ok {
        println!(
            "RESULT: both backends produce identical bits for every layer and diagnostic over \
             {n_frames} frames"
        );
        0
    } else {
        if let Some(f) = first_bad_frame {
            println!("first differing frame: {f}");
        }
        println!("RESULT: MISMATCH");
        1
    }
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------
fn cmd_bench(args: &[String]) -> i32 {
    let pos = positional(args);
    if pos.len() < 2 {
        return fail("usage: bench voice <frames>");
    }
    if is_echo(pos[0]) {
        return cmd_bench_echo(args);
    }
    if let Err(e) = check_alias(pos[0]) {
        return fail(e);
    }
    let frames: usize = match pos[1].parse() {
        Ok(n) if n > 0 => n,
        _ => return fail("frame count must be a positive integer"),
    };
    let cfwm_path = model_path(args);
    let requested = match requested_backend(args) {
        Ok(b) => b,
        Err(e) => return fail(e),
    };
    let per_layer = args.iter().any(|a| a == "--per-layer");
    let mut voice = match load_model_with(&cfwm_path, requested) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };

    // Deterministic synthetic frames (benchmark input, not a numeric result).
    let mut flat = vec![0.0f32; frames * INPUT_LEN];
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    for v in flat.iter_mut() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let u = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32;
        *v = (u / (1u32 << 24) as f32) * 2.0 - 1.0;
    }

    let warmup = 100.min(frames);
    for i in 0..warmup {
        let _ = voice.process(&flat[i * INPUT_LEN..(i + 1) * INPUT_LEN]);
    }

    let mut trials = [0.0f64; 3];
    let mut allocs = [0usize; 3];
    for (ti, t) in trials.iter_mut().enumerate() {
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        let start = Instant::now();
        for i in 0..frames {
            if voice
                .process(&flat[i * INPUT_LEN..(i + 1) * INPUT_LEN])
                .is_err()
            {
                return fail("benchmark frame rejected");
            }
        }
        *t = start.elapsed().as_secs_f64();
        allocs[ti] = ALLOCATIONS.load(Ordering::Relaxed) - before;
    }
    let best = trials.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = trials.iter().sum::<f64>() / trials.len() as f64;

    // Keep the last output observable so the loop cannot be optimised away.
    let checksum: f32 = voice.output().iter().copied().sum();

    println!("scope: input staging + layers 0-18 (full model, 19 layers)");
    println!(
        "backend={} workspace_floats={} workspace_i16={} workspace_i32={} frames_per_trial={frames}",
        voice.backend_name(),
        voice.workspace_floats(),
        voice.workspace_i16(),
        voice.workspace_i32()
    );
    for (i, t) in trials.iter().enumerate() {
        println!(
            "trial {}: {:.3} ms total, {:.4} ms/frame, {:.1} fps, allocations={}",
            i + 1,
            t * 1000.0,
            t * 1000.0 / frames as f64,
            frames as f64 / t,
            allocs[i]
        );
    }
    println!(
        "best: {:.4} ms/frame, {:.1} fps (mean over trials: {:.4} ms/frame)",
        best * 1000.0 / frames as f64,
        frames as f64 / best,
        mean * 1000.0 / frames as f64
    );
    println!(
        "allocations: {} (steady-state inference must be 0; output checksum {checksum:e})",
        allocs.iter().sum::<usize>()
    );
    if per_layer {
        let n = frames.min(200).max(1);
        let mut totals = [0u64; STAGES];
        for i in 0..n {
            let mut t = [0u64; STAGES];
            if voice
                .process_profiled(&flat[i * INPUT_LEN..(i + 1) * INPUT_LEN], &mut t)
                .is_err()
            {
                return fail("per-layer frame rejected");
            }
            for (s, &v) in t.iter().enumerate() {
                totals[s] += v;
            }
        }
        println!();
        println!(
            "per-layer timings (backend={}, mean of {n} frames):",
            voice.backend_name()
        );
        println!("{:<12} {:>12}", "stage", "us/frame");
        for (s, &label) in STAGE_LABELS.iter().enumerate() {
            println!(
                "{:<12} {:>12.3}",
                label,
                totals[s] as f64 / 1000.0 / n as f64
            );
        }
        println!(
            "{:<12} {:>12.3}",
            "sum",
            totals.iter().sum::<u64>() as f64 / 1000.0 / n as f64
        );
    }
    0
}

// ---------------------------------------------------------------------------
// dump: native scalar replay -> flat oracle file for the wasm harness
// ---------------------------------------------------------------------------
/// Flat oracle dump magic.
const DUMP_MAGIC: &[u8; 4] = b"CFD1";
/// Flat oracle dump version.
const DUMP_VERSION: u16 = 1;
/// Header flag: per-frame intermediate buffers follow the outputs.
const DUMP_FLAG_LAYERS: u16 = 1;
/// Header length in bytes.
const DUMP_HEADER_LEN: usize = 32;
/// Safety cap for `--layers` (per-frame buffers are ~0.7 MB for this model).
const DUMP_MAX_LAYER_FRAMES: usize = 256;

/// One intermediate buffer written by `dump --layers`.
struct DumpLayer {
    name: &'static str,
    /// 0 = f32 bits, 1 = int16.
    kind: u8,
    len: usize,
}

fn cmd_dump(args: &[String]) -> i32 {
    let pos = positional(args);
    if pos.len() < 2 {
        return fail("usage: dump voice <reference.npz|.cfrf> --out PATH [--limit N] [--layers]");
    }
    if is_echo(pos[0]) {
        return cmd_dump_echo(args);
    }
    if let Err(e) = check_alias(pos[0]) {
        return fail(e);
    }
    let ref_path = pos[1];
    let out_path = match take_flag(args, "--out") {
        Some(p) => p,
        None => return fail("dump: --out PATH is required"),
    };
    let limit: Option<usize> = take_flag(args, "--limit").and_then(|s| s.parse().ok());
    let want_layers = args.iter().any(|a| a == "--layers");
    let cfwm_path = model_path(args);

    let mut voice = match load_model_with(&cfwm_path, Some(Backend::Scalar)) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    if cf_core::fma::is_soft() {
        return fail(
            "dump: this build routes fma() through the software path; build cf-cli \
             without the workspace-wide force-soft-fma feature (cargo build --release -p cf-cli) \
             so the oracle is the native hardware-FMA scalar path",
        );
    }
    let ref_bytes = match fs::read(ref_path) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {ref_path}: {e}")),
    };
    let reference = match npz::RefData::open(&ref_bytes) {
        Ok(r) => r,
        Err(e) => return fail(format!("cannot parse {ref_path}: {e}")),
    };
    let (input, n_frames, kind) = match reference_frames(&reference, ref_path, limit) {
        Ok(v) => v,
        Err(e) => return fail(e),
    };
    if want_layers && n_frames > DUMP_MAX_LAYER_FRAMES {
        return fail(format!(
            "--layers is limited to {DUMP_MAX_LAYER_FRAMES} frames (got {n_frames}); use --limit"
        ));
    }

    let mut layers: Vec<DumpLayer> = Vec::new();
    if want_layers {
        for &name in DIAG_F32_SLOTS {
            match voice.slot(name) {
                Some(s) => layers.push(DumpLayer {
                    name,
                    kind: 0,
                    len: s.len(),
                }),
                None => return fail(format!("runtime has no slot {name:?}")),
            }
        }
        for &name in DIAG_I16_SLOTS {
            match voice.slot_i16(name) {
                Some(s) => layers.push(DumpLayer {
                    name,
                    kind: 1,
                    len: s.len(),
                }),
                None => return fail(format!("runtime has no i16 slot {name:?}")),
            }
        }
    }

    // Replay through the native scalar backend (the oracle), collecting the
    // final outputs as raw bit patterns and, optionally, every intermediate.
    let mut outputs: Vec<u32> = Vec::with_capacity(n_frames * L18_LEN);
    let mut layer_bytes: Vec<u8> = Vec::new();
    for f in 0..n_frames {
        let frame = &input[f * INPUT_LEN..(f + 1) * INPUT_LEN];
        if let Err(e) = voice.process(frame) {
            return fail(format!("frame {f}: {e}"));
        }
        outputs.extend(voice.output().iter().map(|v| v.to_bits()));
        for layer in &layers {
            match layer.kind {
                0 => {
                    let src = voice.slot(layer.name).expect("validated above");
                    for v in src {
                        layer_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                }
                _ => {
                    let src = voice.slot_i16(layer.name).expect("validated above");
                    for v in src {
                        layer_bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
    }

    let file = match File::create(&out_path) {
        Ok(f) => f,
        Err(e) => return fail(format!("cannot create {out_path}: {e}")),
    };
    let mut w = BufWriter::new(file);
    let write_result = (|| -> std::io::Result<()> {
        w.write_all(DUMP_MAGIC)?;
        w.write_all(&DUMP_VERSION.to_le_bytes())?;
        let flags = if want_layers { DUMP_FLAG_LAYERS } else { 0 };
        w.write_all(&flags.to_le_bytes())?;
        w.write_all(&(n_frames as u32).to_le_bytes())?;
        w.write_all(&(INPUT_LEN as u32).to_le_bytes())?;
        w.write_all(&(L18_LEN as u32).to_le_bytes())?;
        w.write_all(&(layers.len() as u32).to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
        for layer in &layers {
            let name = layer.name.as_bytes();
            w.write_all(&[layer.kind, name.len() as u8])?;
            w.write_all(&0u16.to_le_bytes())?;
            w.write_all(&(layer.len as u32).to_le_bytes())?;
            w.write_all(name)?;
            for _ in 0..((4 - name.len() % 4) % 4) {
                w.write_all(&[0u8])?;
            }
        }
        for &x in &input[..n_frames * INPUT_LEN] {
            w.write_all(&x.to_bits().to_le_bytes())?;
        }
        for &x in &outputs {
            w.write_all(&x.to_le_bytes())?;
        }
        w.write_all(&layer_bytes)?;
        w.flush()
    })();
    if let Err(e) = write_result {
        return fail(format!("cannot write {out_path}: {e}"));
    }
    let size = fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);

    println!("input: {ref_path} ({kind}, {n_frames} frames)");
    println!("backend: scalar (native oracle)");
    println!("fma: native (hardware)");
    println!(
        "layers: {}",
        if want_layers {
            format!("{} buffers per frame", layers.len())
        } else {
            "not written".to_string()
        }
    );
    println!("wrote: {out_path} ({size} bytes, header {DUMP_HEADER_LEN} bytes)");
    println!(
        "oracle values: {} outputs{}",
        n_frames * L18_LEN,
        if want_layers {
            format!(", {} intermediate", n_frames * DumpLayer::values(&layers))
        } else {
            String::new()
        }
    );
    println!("RESULT: native scalar oracle dump complete");
    0
}

impl DumpLayer {
    fn values(layers: &[DumpLayer]) -> usize {
        layers.iter().map(|l| l.len).sum()
    }
}

// ---------------------------------------------------------------------------
// exp-check: deterministic software exp vs the native reference
// ---------------------------------------------------------------------------
fn exp_bits(x: f32) -> (u32, u32) {
    let e_native = (-(x as f64)).exp() as f32;
    let s_native = 1.0f32 / (e_native + 1.0f32);
    let e_soft = cf_core::dexp::exp_f32(-x);
    let s_soft = cf_core::dexp::sigmoid_tail_soft(x);
    let e_bad = u32::from(e_native.to_bits() != e_soft.to_bits());
    let s_bad = u32::from(s_native.to_bits() != s_soft.to_bits());
    (e_bad | (s_bad << 1), s_soft.to_bits())
}

fn cmd_exp_check(args: &[String]) -> i32 {
    let pos = positional(args);
    if pos.len() < 2 || !is_echo(pos[0]) {
        return fail(
            "usage: exp-check echo <reference.cfrf|.npz> [--limit N] [--cases N] [--seed N]",
        );
    }
    let ref_path = pos[1];
    let limit: Option<usize> = take_flag(args, "--limit").and_then(|s| s.parse().ok());
    let cases: u64 = take_flag(args, "--cases")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000_000);
    let seed: u64 = take_flag(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xE7_00D1_5EED_2026);
    let cfwm_path = echo_model_path(args);

    let mut model = match load_echo_with(&cfwm_path, Backend::Scalar) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    let ref_bytes = match fs::read(ref_path) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {ref_path}: {e}")),
    };
    let reference = match npz::RefData::open(&ref_bytes) {
        Ok(r) => r,
        Err(e) => return fail(format!("cannot parse {ref_path}: {e}")),
    };
    let (input, n_frames) = match echo_reference_frames(&reference, ref_path, limit) {
        Ok(v) => v,
        Err(e) => return fail(e),
    };

    // Element 512 of layer 87 is the pre-sigmoid input of the scalar tail.
    struct TailCapture {
        x: f32,
        seen: bool,
    }
    impl EchoObserver for TailCapture {
        fn layer(&mut self, index: usize, values: &[f32]) {
            if index == 87 && values.len() > 512 {
                self.x = values[512];
                self.seen = true;
            }
        }
    }
    let mut xs: Vec<f32> = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let frame = &input[f * ECHO_INPUT_LEN..(f + 1) * ECHO_INPUT_LEN];
        let mut obs = TailCapture {
            x: 0.0,
            seen: false,
        };
        if let Err(e) = model.process_with(frame, &mut obs) {
            return fail(format!("frame {f}: {e}"));
        }
        if !obs.seen {
            return fail(format!("frame {f}: layer 87 was not observed"));
        }
        xs.push(obs.x);
    }
    let mut xmin = f32::INFINITY;
    let mut xmax = f32::NEG_INFINITY;
    for &x in &xs {
        if x < xmin {
            xmin = x;
        }
        if x > xmax {
            xmax = x;
        }
    }

    println!("model: {} (scalar reference)", model.backend_name());
    println!("capture: {ref_path} ({n_frames} frames)");
    println!(
        "layer 87 element 512 (tail input): min {xmin} (0x{:08x}), max {xmax} (0x{:08x}); \
         exp argument range [{}, {}]",
        xmin.to_bits(),
        xmax.to_bits(),
        -xmax,
        -xmin
    );

    let mut cap_e = 0u64;
    let mut cap_s = 0u64;
    let mut cap_examples: Vec<String> = Vec::new();
    for (i, &x) in xs.iter().enumerate() {
        let (bad, _) = exp_bits(x);
        if bad & 1 != 0 {
            cap_e += 1;
        }
        if bad & 2 != 0 {
            cap_s += 1;
            if cap_examples.len() < 5 {
                let e_native = (-(x as f64)).exp() as f32;
                let s_native = 1.0f32 / (e_native + 1.0f32);
                cap_examples.push(format!(
                    "frame {i} x 0x{:08x}: exp native 0x{:08x} soft 0x{:08x}; \
                     sigmoid native 0x{:08x} soft 0x{:08x}",
                    x.to_bits(),
                    e_native.to_bits(),
                    cf_core::dexp::exp_f32(-x).to_bits(),
                    s_native.to_bits(),
                    cf_core::dexp::sigmoid_tail_soft(x).to_bits()
                ));
            }
        }
    }
    println!(
        "capture tail: {n_frames} values, exp f32 mismatches {cap_e}, sigmoid-tail mismatches {cap_s}"
    );
    for e in &cap_examples {
        println!("    {e}");
    }

    // Deterministic random sweep of the observed range (with margin) and of
    // the full finite tail argument range the sigmoid can see.
    struct Rng(u64);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
        }
    }
    let mut rng = Rng(seed);
    let margin = 4.0f32;
    let lo = (xmin - margin).min(-8.0);
    let hi = (xmax + margin).max(8.0);
    let mut sweep = |label: &str, lo: f32, hi: f32, n: u64| -> (u64, u64, u64, Vec<String>) {
        let mut e_bad = 0u64;
        let mut s_bad = 0u64;
        let mut examples = Vec::new();
        let span = hi - lo;
        for _ in 0..n {
            let u = (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
            let x = lo + span * u;
            let (bad, _) = exp_bits(x);
            if bad & 1 != 0 {
                e_bad += 1;
            }
            if bad & 2 != 0 {
                s_bad += 1;
                if examples.len() < 5 {
                    let e_native = (-(x as f64)).exp() as f32;
                    let s_native = 1.0f32 / (e_native + 1.0f32);
                    examples.push(format!(
                        "x 0x{:08x}: exp native 0x{:08x} soft 0x{:08x}; \
                         sigmoid native 0x{:08x} soft 0x{:08x}",
                        x.to_bits(),
                        e_native.to_bits(),
                        cf_core::dexp::exp_f32(-x).to_bits(),
                        s_native.to_bits(),
                        cf_core::dexp::sigmoid_tail_soft(x).to_bits()
                    ));
                }
            }
        }
        println!(
            "sweep {label}: {n} values in [{lo}, {hi}], exp f32 mismatches {e_bad}, \
             sigmoid-tail mismatches {s_bad}"
        );
        for e in &examples {
            println!("    {e}");
        }
        (n, e_bad, s_bad, examples)
    };
    let half = cases / 2;
    let (n1, e1, s1, _) = sweep("observed+margin", lo, hi, half);
    let (n2, e2, s2, _) = sweep("wide", -120.0, 90.0, cases - half);

    let total = n1 + n2;
    let e_total = e1 + e2;
    let s_total = s1 + s2;
    println!();
    println!(
        "total sweep: {total} values, exp f32 mismatches {e_total}, sigmoid-tail mismatches {s_total}"
    );
    if cap_e == 0 && cap_s == 0 && e_total == 0 && s_total == 0 {
        println!(
            "RESULT: software exp matches the native reference on the capture and the random sweep"
        );
        0
    } else {
        println!("RESULT: MISMATCH");
        1
    }
}

// ---------------------------------------------------------------------------
// fma-check
// ---------------------------------------------------------------------------
fn cmd_fma_check(args: &[String]) -> i32 {
    let cases: u64 = take_flag(args, "--cases")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);
    let seed: u64 = take_flag(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xCF_00D1_5EED_2026);

    println!(
        "native={} soft={} requested_cases={cases} seed=0x{seed:016x}",
        NativeFma::NAME,
        soft_fma_name(),
    );
    let report = cf_core::fma_check::crosscheck(cases, seed);
    println!(
        "{:<20} {:>12} {:>12} {:>10}",
        "category", "cases", "mismatches", "both_nan"
    );
    for c in &report.categories {
        println!(
            "{:<20} {:>12} {:>12} {:>10}",
            c.name, c.cases, c.mismatches, c.both_nan
        );
    }
    println!(
        "{:<20} {:>12} {:>12} {:>10}",
        "TOTAL", report.total, report.mismatches, report.both_nan
    );
    for e in &report.examples {
        println!(
            "  mismatch a={:e} b={:e} c={:e} native=0x{:08x} soft=0x{:08x}",
            e.a,
            e.b,
            e.c,
            e.native.to_bits(),
            e.soft.to_bits()
        );
    }
    if report.mismatches == 0 {
        println!("RESULT: 0 mismatches over {} cases", report.total);
        0
    } else {
        println!("RESULT: {} mismatches", report.mismatches);
        1
    }
}

fn soft_fma_name() -> &'static str {
    SoftFma::NAME
}

// ---------------------------------------------------------------------------
// rcp-check
// ---------------------------------------------------------------------------
fn cmd_rcp_check(args: &[String]) -> i32 {
    let path = take_flag(args, "--npy").unwrap_or_else(|| "../tools/rcpps_table.npy".to_string());
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {path}: {e}")),
    };
    let digest = sha256::hex(&sha256::digest(&bytes));
    let expected = "2e3bd841cd5cc51da94c416c8806dfd954b9018564acdee35a8eb77091f07683";
    println!("file: {path}");
    println!("sha256: {digest}");
    if digest != expected {
        println!("RESULT: table sha256 mismatch (expected {expected})");
        return 1;
    }
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return fail("not a .npy file");
    }
    let (header_len, header_start, data_start) = if bytes[6] == 1 {
        let n = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (n, 10usize, 10 + n)
    } else {
        let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (n, 12usize, 12 + n)
    };
    let header = match std::str::from_utf8(&bytes[header_start..header_start + header_len]) {
        Ok(h) => h,
        Err(_) => return fail("npy header is not utf-8"),
    };
    if !header.contains("'<u2'") || !header.contains("(4096,)") {
        return fail(format!("unexpected table header: {header}"));
    }
    if data_start + 4096 * 2 > bytes.len() {
        return fail("table payload is truncated");
    }
    let mut mismatches = 0u64;
    for i in 0..4096 {
        let at = data_start + i * 2;
        let v = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        if v != RCP_TABLE[i] {
            mismatches += 1;
            if mismatches <= 5 {
                println!("  table[{i}]: npy={v} rust={}", RCP_TABLE[i]);
            }
        }
    }
    println!("entries compared: 4096, mismatches: {mismatches}");
    // A handful of structural/special-case checks on the integer formula.
    let specials: [(u32, u32); 10] = [
        (0x0000_0000, 0x7F80_0000),
        (0x8000_0000, 0xFF80_0000),
        (0x0000_0001, 0x7F80_0000),
        (0x7F80_0000, 0x0000_0000),
        (0xFF80_0000, 0x8000_0000),
        (0x7F80_0001, 0x7FC0_0001),
        (0x7FBF_FFFF, 0x7FFF_FFFF),
        (0xFF80_0001, 0xFFC0_0001),
        (0x7E80_0000, 0x0000_0000),
        (0x3F80_0000, 0x3F7F_F000),
    ];
    let mut special_bad = 0;
    for (input, want) in specials {
        let got = rcp_ps_bits(input);
        if got != want {
            special_bad += 1;
            println!("  special 0x{input:08x}: got 0x{got:08x} want 0x{want:08x}");
        }
    }
    println!(
        "special cases: {} compared, {special_bad} mismatches",
        specials.len()
    );
    if mismatches == 0 && special_bad == 0 {
        println!("RESULT: u16-exact against reference table");
        0
    } else {
        1
    }
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------
fn cmd_info(args: &[String]) -> i32 {
    let pos = positional(args);
    if pos.is_empty() {
        return fail("usage: info voice|echo");
    }
    if is_echo(pos[0]) {
        return cmd_info_echo(args);
    }
    if let Err(e) = check_alias(pos[0]) {
        return fail(e);
    }
    let path = model_path(args);
    let model = match load_model(&path) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    println!(
        "container: version={} model_id={} tensors={} data_len={}",
        model.container().format_version(),
        model.container().model_id(),
        model.container().tensor_count(),
        model.container().data_len()
    );
    println!(
        "workspace_floats: {} ({}) i16: {} i32: {}",
        model.workspace_floats(),
        model.workspace_floats() * 4,
        model.workspace_i16(),
        model.workspace_i32()
    );
    println!(
        "{:<6} {:<6} {:<6} {:>24} {:>10}",
        "tag", "dtype", "rank", "dims", "bytes"
    );
    for t in model.container().tensors() {
        println!(
            "{:<6} {:<6} {:<6} {:>24} {:>10}",
            t.tag,
            dtype_name(t.dtype),
            t.rank,
            format!("{:?}", &t.dims[..t.rank as usize]),
            t.byte_len
        );
    }
    println!("container bytes: {}", model.container().file_len());
    0
}
