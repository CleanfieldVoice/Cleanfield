//! Steady-state allocation test.
//!
//! `VoiceModel::new` allocates the container copy and the fixed workspace.
//! After that, `process` must not allocate at all.  A counting global
//! allocator measures the delta across a long steady-state loop.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use cf_voice::{VoiceModel, INPUT_LEN};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn steady_state_inference_does_not_allocate() {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/voice-default.cfwm"
    ));
    let mut model = VoiceModel::new(bytes).expect("model container");
    let mut frame = [0.0f32; INPUT_LEN];
    for (i, v) in frame.iter_mut().enumerate() {
        *v = ((i * 29 % 97) as f32) * 0.02 - 0.9;
    }

    // Warm up (state is already allocated by `new`).
    for _ in 0..16 {
        model.process(&frame).expect("frame");
    }

    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..256 {
        model.process(&frame).expect("frame");
    }
    let after = ALLOCS.load(Ordering::Relaxed);

    assert_eq!(
        after - before,
        0,
        "steady-state inference performed {} allocation(s)",
        after - before
    );
    // Keep the state observable so the loop cannot be optimised away.
    let sum: f32 = model.slot("L4").unwrap().iter().sum();
    assert!(sum.is_finite());
}
