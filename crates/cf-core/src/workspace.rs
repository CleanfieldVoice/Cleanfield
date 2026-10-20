//! Fixed-size, named-slot arena.
//!
//! The arena is allocated exactly once.  After that, every buffer access is a
//! slice into the same allocation; the inference path never grows, pushes or
//! reallocates.  Slot families are addressed by index resolved at model load
//! time, so the steady-state path does not even do a name lookup.

/// Maximum number of named slots in one workspace.
pub const MAX_SLOTS: usize = 48;

/// One named region of the arena.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    /// Diagnostic name (never part of the model container).
    pub name: &'static str,
    /// Offset in `f32` elements from the arena base.
    pub offset: usize,
    /// Length in `f32` elements.
    pub len: usize,
}

/// Preallocated arena with named slots.
pub struct Workspace {
    buf: Box<[f32]>,
    slots: [Slot; MAX_SLOTS],
    n_slots: usize,
}

impl Workspace {
    /// Build the arena from `(name, len)` pairs.
    ///
    /// Slot offsets are 8-float aligned so later SIMD backends can use
    /// aligned loads if they want to.  This is the only allocating call.
    pub fn new(specs: &[(&'static str, usize)]) -> Workspace {
        assert!(specs.len() <= MAX_SLOTS, "too many workspace slots");
        let mut slots = [Slot {
            name: "",
            offset: 0,
            len: 0,
        }; MAX_SLOTS];
        let mut total = 0usize;
        for (i, &(name, len)) in specs.iter().enumerate() {
            total = align_up(total, 8);
            assert!(!name.is_empty(), "workspace slot needs a name");
            slots[i] = Slot {
                name,
                offset: total,
                len,
            };
            total += len;
        }
        total = align_up(total, 8);
        Workspace {
            buf: vec![0.0f32; total].into_boxed_slice(),
            slots,
            n_slots: specs.len(),
        }
    }

    /// Total arena size in `f32` elements.
    #[inline]
    pub fn capacity_floats(&self) -> usize {
        self.buf.len()
    }

    /// Number of slots.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.n_slots
    }

    /// Diagnostic name of a slot.
    #[inline]
    pub fn slot_name(&self, index: usize) -> &'static str {
        self.slots[index].name
    }

    /// Resolve a name to a slot index once, at model-load time.
    pub fn slot_index(&self, name: &str) -> Option<usize> {
        self.slots[..self.n_slots]
            .iter()
            .position(|s| s.name == name)
    }

    /// Slot length in floats.
    #[inline]
    pub fn slot_len(&self, index: usize) -> usize {
        self.slots[index].len
    }

    /// Immutable view of one slot.
    #[inline]
    pub fn slot(&self, index: usize) -> &[f32] {
        let s = self.slots[index];
        &self.buf[s.offset..s.offset + s.len]
    }

    /// Mutable view of one slot.
    #[inline]
    pub fn slot_mut(&mut self, index: usize) -> &mut [f32] {
        let s = self.slots[index];
        &mut self.buf[s.offset..s.offset + s.len]
    }

    /// Two disjoint mutable slots.  Panics on equal or out-of-range indices.
    ///
    /// This is the workhorse of the layer pipeline: each layer reads one slot
    /// and writes another without allocating or copying.
    #[inline]
    pub fn get2_mut(&mut self, a: usize, b: usize) -> (&mut [f32], &mut [f32]) {
        assert!(a < self.n_slots && b < self.n_slots && a != b);
        let sa = self.slots[a];
        let sb = self.slots[b];
        if sa.offset < sb.offset {
            let (lo, hi) = self.buf.split_at_mut(sb.offset);
            (&mut lo[sa.offset..sa.offset + sa.len], &mut hi[..sb.len])
        } else {
            let (lo, hi) = self.buf.split_at_mut(sa.offset);
            (&mut hi[..sa.len], &mut lo[sb.offset..sb.offset + sb.len])
        }
    }

    /// Three disjoint mutable slots.
    ///
    /// The layer pipeline needs at most three operands at once (concatenation,
    /// a three-input add, and the recurrent step with its input, hidden state
    /// and gate buffer).  The slots are laid out once by [`Workspace::new`] and
    /// never overlap, so the returned slices are disjoint.
    #[inline]
    pub fn get3_mut(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
    ) -> (&mut [f32], &mut [f32], &mut [f32]) {
        assert!(a < self.n_slots && b < self.n_slots && c < self.n_slots);
        assert!(a != b && b != c && a != c, "get3_mut needs distinct slots");
        let sa = self.slots[a];
        let sb = self.slots[b];
        let sc = self.slots[c];
        let ptr = self.buf.as_mut_ptr();
        // SAFETY: the three slot ranges are pairwise disjoint by construction
        // and lie inside the arena allocation; `get3_mut` rejects equal
        // indices, so no two returned slices alias.
        unsafe {
            let pa = std::slice::from_raw_parts_mut(ptr.add(sa.offset), sa.len);
            let pb = std::slice::from_raw_parts_mut(ptr.add(sb.offset), sb.len);
            let pc = std::slice::from_raw_parts_mut(ptr.add(sc.offset), sc.len);
            (pa, pb, pc)
        }
    }

    /// Zero every slot (used by reset paths, not by the hot loop).
    pub fn clear(&mut self) {
        self.buf.fill(0.0);
    }
}

#[inline]
fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

/// Fixed-size arena of integers, built the same way as [`Workspace`].
///
/// The quantised state (`int16` hidden state, `int32` GEMM accumulators) needs
/// the same "allocate once, borrow slices" shape as the float buffers.  The
/// element type is deliberately not `f32`: the integer path must stay integer.
pub struct Arena<T> {
    buf: Box<[T]>,
    slots: [Slot; MAX_SLOTS],
    n_slots: usize,
}

impl<T: Copy + Default> Arena<T> {
    /// Build the arena from `(name, len)` pairs.  Allocates once.
    pub fn new(specs: &[(&'static str, usize)]) -> Arena<T> {
        assert!(specs.len() <= MAX_SLOTS, "too many arena slots");
        let mut slots = [Slot {
            name: "",
            offset: 0,
            len: 0,
        }; MAX_SLOTS];
        let mut total = 0usize;
        for (i, &(name, len)) in specs.iter().enumerate() {
            total = align_up(total, 8);
            assert!(!name.is_empty(), "arena slot needs a name");
            slots[i] = Slot {
                name,
                offset: total,
                len,
            };
            total += len;
        }
        total = align_up(total, 8);
        Arena {
            buf: vec![T::default(); total].into_boxed_slice(),
            slots,
            n_slots: specs.len(),
        }
    }

    /// Total arena length in elements.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Resolve a name to a slot index once, at model-load time.
    pub fn slot_index(&self, name: &str) -> Option<usize> {
        self.slots[..self.n_slots]
            .iter()
            .position(|s| s.name == name)
    }

    /// Diagnostic name of a slot.
    #[inline]
    pub fn slot_name(&self, index: usize) -> &'static str {
        self.slots[index].name
    }

    /// Slot length in elements.
    #[inline]
    pub fn slot_len(&self, index: usize) -> usize {
        self.slots[index].len
    }

    /// Immutable view of one slot.
    #[inline]
    pub fn slot(&self, index: usize) -> &[T] {
        let s = self.slots[index];
        &self.buf[s.offset..s.offset + s.len]
    }

    /// Mutable view of one slot.
    #[inline]
    pub fn slot_mut(&mut self, index: usize) -> &mut [T] {
        let s = self.slots[index];
        &mut self.buf[s.offset..s.offset + s.len]
    }

    /// Two disjoint mutable slots.
    #[inline]
    pub fn get2_mut(&mut self, a: usize, b: usize) -> (&mut [T], &mut [T]) {
        assert!(a < self.n_slots && b < self.n_slots && a != b);
        let sa = self.slots[a];
        let sb = self.slots[b];
        if sa.offset < sb.offset {
            let (lo, hi) = self.buf.split_at_mut(sb.offset);
            (&mut lo[sa.offset..sa.offset + sa.len], &mut hi[..sb.len])
        } else {
            let (lo, hi) = self.buf.split_at_mut(sa.offset);
            (&mut hi[..sa.len], &mut lo[sb.offset..sb.offset + sb.len])
        }
    }

    /// Zero every slot (reset paths only, never the hot loop).
    pub fn clear(&mut self) {
        for v in self.buf.iter_mut() {
            *v = T::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_distinct_and_aligned() {
        let mut ws = Workspace::new(&[("a", 3), ("b", 5), ("c", 7)]);
        assert!(ws.capacity_floats() >= 15);
        assert_eq!(ws.slot_index("b"), Some(1));
        assert_eq!(ws.slot_index("missing"), None);
        let a = ws.slot_mut(0);
        a.fill(1.0);
        assert_eq!(ws.slot(0), &[1.0, 1.0, 1.0]);
        assert_eq!(ws.slot(1), &[0.0; 5]);
        let (x, y) = ws.get2_mut(1, 2);
        x.fill(2.0);
        y.fill(3.0);
        assert_eq!(ws.slot(1), &[2.0; 5]);
        assert_eq!(ws.slot(2), &[3.0; 7]);
    }

    #[test]
    fn three_slots_are_disjoint() {
        let mut ws = Workspace::new(&[("a", 3), ("b", 5), ("c", 7)]);
        let (a, b, c) = ws.get3_mut(2, 0, 1);
        a.fill(1.0);
        b.fill(2.0);
        c.fill(3.0);
        assert_eq!(ws.slot(0), &[2.0; 3]);
        assert_eq!(ws.slot(1), &[3.0; 5]);
        assert_eq!(ws.slot(2), &[1.0; 7]);
    }

    #[test]
    fn int_arena_slots_are_distinct() {
        let mut a: Arena<i16> = Arena::new(&[("x", 3), ("y", 4)]);
        assert_eq!(a.slot_index("y"), Some(1));
        a.slot_mut(0).copy_from_slice(&[1, -2, 3]);
        a.slot_mut(1).copy_from_slice(&[4, 5, 6, 7]);
        assert_eq!(a.slot(0), &[1, -2, 3]);
        assert_eq!(a.slot(1), &[4, 5, 6, 7]);
        let (x, y) = a.get2_mut(0, 1);
        assert_eq!(x, &[1, -2, 3]);
        assert_eq!(y, &[4, 5, 6, 7]);
        a.clear();
        assert!(a.slot(0).iter().all(|&v| v == 0));

        let mut b: Arena<i32> = Arena::new(&[("acc", 8)]);
        b.slot_mut(0)[7] = -5;
        assert_eq!(b.slot(0)[7], -5);
        assert_eq!(b.slot_len(0), 8);
    }
}
