//! Echo scalar-vs-AVX2 equivalence on every layer and diagnostic buffer.
//!
//! This is the in-tree counterpart of `cf-cli simd-check echo`: it runs the
//! same three frames through both backends and compares all 89 layer outputs
//! and the 26 named diagnostics as raw bits.  `cargo test --workspace` builds
//! the scalar path with the software FMA (the wasm-destined arithmetic), so a
//! pass also proves hardware-FMA AVX2 and soft-FMA scalar agree bit for bit.

use cf_core::Backend;
use cf_echo::{EchoModel, EchoObserver, INPUT_LEN, LAYERS, LAYER_LEN};

const ECHO_F32_DIAGS: &[&str] = &[
    "staged", "history", "L59_yih", "L59_yhh", "L59_z", "L59_r", "L59_n", "L59_A_in", "L59_A_h",
    "L60_yih", "L60_yhh", "L60_z", "L60_r", "L60_n", "L60_A_in", "L60_A_h", "L61_A", "L85_A",
    "L87_A",
];

const ECHO_I16_DIAGS: &[&str] = &[
    "L59_qx", "L59_qh", "L60_qx", "L60_qh", "L61_qx", "L85_qx", "L87_qx",
];

struct Capture {
    layers: Vec<Vec<u32>>,
    f32_diag: Vec<(&'static str, Vec<u32>)>,
    i16_diag: Vec<(&'static str, Vec<i16>)>,
}

impl Capture {
    fn new() -> Capture {
        Capture {
            layers: vec![Vec::new(); LAYERS],
            f32_diag: Vec::new(),
            i16_diag: Vec::new(),
        }
    }
}

impl EchoObserver for Capture {
    fn layer(&mut self, index: usize, values: &[f32]) {
        self.layers[index] = values.iter().map(|v| v.to_bits()).collect();
    }
    fn diag_f32(&mut self, name: &'static str, values: &[f32]) {
        self.f32_diag
            .push((name, values.iter().map(|v| v.to_bits()).collect()));
    }
    fn diag_i16(&mut self, name: &'static str, values: &[i16]) {
        self.i16_diag.push((name, values.to_vec()));
    }
}

fn frame(seed: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; INPUT_LEN];
    for (i, v) in out.iter_mut().enumerate() {
        // Deterministic, out-of-distribution-ish content per frame.
        let n = (i * 37 + seed * 977) % 211;
        *v = (n as f32) * 0.01 - 1.0;
    }
    // A few high-amplitude bins exercise the quantiser clamp.
    for k in 0..15 {
        out[k * 97] = if k % 2 == 0 { 3.25 } else { -3.25 };
    }
    out
}

#[test]
fn scalar_and_avx2_match_on_every_echo_buffer() {
    if !Backend::Avx2.is_available() {
        eprintln!("avx2 backend unavailable, skipping");
        return;
    }
    let bytes: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/echo-default.cfwm"
    ));
    let mut scalar = EchoModel::with_backend(bytes, Backend::Scalar).expect("scalar model");
    let mut wide = EchoModel::with_backend(bytes, Backend::Avx2).expect("avx2 model");

    let mut layers = 0u64;
    let mut diags = 0u64;
    for f in 0..3 {
        let input = frame(f);
        let mut a = Capture::new();
        let mut b = Capture::new();
        scalar.process_with(&input, &mut a).expect("scalar frame");
        wide.process_with(&input, &mut b).expect("avx2 frame");

        for i in 0..LAYERS {
            assert_eq!(
                a.layers[i].len(),
                LAYER_LEN[i],
                "frame {f} L{i} scalar width"
            );
            assert_eq!(b.layers[i].len(), LAYER_LEN[i], "frame {f} L{i} avx2 width");
            for (k, (&x, &y)) in a.layers[i].iter().zip(b.layers[i].iter()).enumerate() {
                layers += 1;
                assert_eq!(x, y, "frame {f} L{i}[{k}]: scalar 0x{x:08x} avx2 0x{y:08x}");
            }
        }
        assert_eq!(a.f32_diag.len(), ECHO_F32_DIAGS.len(), "frame {f}");
        assert_eq!(a.i16_diag.len(), ECHO_I16_DIAGS.len(), "frame {f}");
        for (name, av) in &a.f32_diag {
            let (_, bv) = b
                .f32_diag
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("frame {f}: avx2 missing {name}"));
            assert_eq!(av.len(), bv.len(), "frame {f} {name} width");
            for (k, (&x, &y)) in av.iter().zip(bv.iter()).enumerate() {
                diags += 1;
                assert_eq!(x, y, "frame {f} {name}[{k}]: 0x{x:08x} vs 0x{y:08x}");
            }
        }
        for (name, av) in &a.i16_diag {
            let (_, bv) = b
                .i16_diag
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("frame {f}: avx2 missing {name}"));
            assert_eq!(av.len(), bv.len(), "frame {f} {name} width");
            for (k, (&x, &y)) in av.iter().zip(bv.iter()).enumerate() {
                diags += 1;
                assert_eq!(x, y, "frame {f} {name}[{k}]: {x} vs {y}");
            }
        }
    }
    assert_eq!(layers, 3 * LAYER_LEN.iter().sum::<usize>() as u64);
    // 3 frames of layer values must be exactly covered; diagnostics are
    // counted separately and must be non-zero.
    assert!(diags > 0);
}
