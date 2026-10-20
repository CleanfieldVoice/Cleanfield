//! AVX2 / scalar equivalence test.
//!
//! Builds the same container twice, once with each backend, replays a short
//! deterministic sequence and compares every captured intermediate buffer bit
//! for bit.  This is the in-tree twin of `cf-cli simd-check`.

use cf_core::Backend;
use cf_voice::{VoiceModel, DIAG_F32_SLOTS, DIAG_I16_SLOTS, INPUT_LEN};

fn model_bytes() -> &'static [u8] {
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/voice-default.cfwm"
    ))
}

#[test]
fn avx2_matches_scalar_on_every_intermediate() {
    if !Backend::Avx2.is_available() {
        eprintln!("avx2 backend unavailable on this CPU; equivalence test skipped");
        return;
    }
    let mut scalar =
        VoiceModel::with_backend(model_bytes(), Backend::Scalar).expect("scalar model");
    let mut wide = VoiceModel::with_backend(model_bytes(), Backend::Avx2).expect("avx2 model");
    assert_eq!(scalar.backend_name(), "scalar");
    assert_eq!(wide.backend_name(), "avx2");

    let mut frames = vec![[0.0f32; INPUT_LEN]; 3];
    for (f, frame) in frames.iter_mut().enumerate() {
        for (i, v) in frame.iter_mut().enumerate() {
            let x = (i * 37 + f * 101) % 211;
            *v = (x as f32) * 0.011 - 1.05;
        }
    }

    for (fi, frame) in frames.iter().enumerate() {
        scalar.process(frame).expect("scalar process");
        wide.process(frame).expect("avx2 process");

        for name in DIAG_F32_SLOTS {
            let a = scalar.slot(name).expect("scalar slot");
            let b = wide.slot(name).expect("avx2 slot");
            assert_eq!(a.len(), b.len(), "frame {fi} {name}: length");
            for k in 0..a.len() {
                assert_eq!(
                    a[k].to_bits(),
                    b[k].to_bits(),
                    "frame {fi} {name}[{k}]: scalar 0x{:08x} avx2 0x{:08x}",
                    a[k].to_bits(),
                    b[k].to_bits()
                );
            }
        }
        for name in DIAG_I16_SLOTS {
            let a = scalar.slot_i16(name).expect("scalar slot");
            let b = wide.slot_i16(name).expect("avx2 slot");
            assert_eq!(a, b, "frame {fi} {name}");
        }
    }
}
