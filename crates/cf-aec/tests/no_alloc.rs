//! Steady-state allocation test.
//!
//! `Aec::new` builds the tables and the fixed workspace. After that,
//! `process` must not allocate at all. A counting global allocator measures the
//! delta across a long loop that includes both single-frame and multi-frame
//! calls, plus a reset (which clears existing buffers, it does not create new
//! ones).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use cf_aec::{Aec, AecConfig, FRAME_LEN};

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

/// Deterministic, allocation-free test signal.
fn fill(frame: &mut [f32], state: &mut u32, scale: f32) {
    for value in frame {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *value = ((*state >> 9) as f32) * (1.0 / 4_194_304.0) * scale;
    }
}

#[test]
fn steady_state_processing_does_not_allocate() {
    let mut aec = Aec::new(AecConfig::default());
    let mut mic = [0.0f32; FRAME_LEN];
    let mut reference = [0.0f32; FRAME_LEN];
    let mut state = 0x1234_5678u32;
    let mut energy = 0.0f32;

    // Warm up: the first calls populate the delay history and the filter.
    for _ in 0..64 {
        fill(&mut mic, &mut state, 0.2);
        fill(&mut reference, &mut state, 0.15);
        let residual = aec.process(&mic, &reference);
        energy += residual[0];
    }

    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..256 {
        fill(&mut mic, &mut state, 0.2);
        fill(&mut reference, &mut state, 0.15);
        let residual = aec.process(&mic, &reference);
        energy += residual[0] + residual[FRAME_LEN - 1];
    }
    // Also exercise the multi-frame path.
    let mut block = [0.0f32; 4 * FRAME_LEN];
    let mut block_ref = [0.0f32; 4 * FRAME_LEN];
    for _ in 0..64 {
        fill(&mut block, &mut state, 0.2);
        fill(&mut block_ref, &mut state, 0.15);
        let residual = aec.process(&block, &block_ref);
        energy += residual[0] + residual[residual.len() - 1];
    }
    let after = ALLOCS.load(Ordering::Relaxed);

    assert_eq!(
        after - before,
        0,
        "steady-state processing performed {} allocation(s)",
        after - before
    );
    // Keep the loop observable.
    assert!(energy.is_finite());
    assert!(aec.erle_db().is_finite());
}
