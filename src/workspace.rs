// Copyright (c) rgmin developers
//! Buffers a solver reuses across iterations.
//!
//! The two-loop, a line-search trial and a curvature pair are length-n
//! vectors. Building a fresh one on every trial is a heap allocation
//! even when the arithmetic writes every entry. The first use at a
//! length allocates; a later use at that length overwrites the same
//! storage. The values written are the ones the allocating code wrote.

use ndarray::{Array1, ArrayView1};

/// Grow `buf` to `n` when it is not already that length.
pub(crate) fn fit(buf: &mut Array1<f64>, n: usize) {
    if buf.len() != n {
        *buf = Array1::zeros(n);
    }
}

/// Copy `src` into `buf`, allocating only when the length changes.
pub(crate) fn assign_fit(buf: &mut Array1<f64>, src: ArrayView1<'_, f64>) {
    let _phase = crate::phase::Guard::copy();
    if buf.len() != src.len() {
        *buf = src.to_owned();
    } else if let (Some(dst), Some(src)) = (buf.as_slice_mut(), src.as_slice()) {
        crate::vecops::copy_slice(dst, src);
    } else {
        buf.assign(&src);
    }
}

/// `out[i] = src[i]` clipped to `[low, high]`, the same expression as
/// `eindir_core::Bounds::clip`.
pub(crate) fn clip_into(
    low: &[f64],
    high: &[f64],
    src: ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
) {
    fit(out, src.len());
    for (i, &xi) in src.iter().enumerate() {
        out[i] = xi.max(low[i]).min(high[i]);
    }
}

/// Line-search scratch that outlives one search.
///
/// `anchor` is `x(0)`, `trial` is the current probe, `best` is the
/// best probe so far, and `clipped` is the point after bounds and
/// `maxmove`. `delta` holds `clipped - x` when a tied Wolfe check
/// needs that difference.
#[derive(Clone, Debug)]
pub(crate) struct SearchBuffers {
    pub(crate) anchor: Array1<f64>,
    pub(crate) trial: Array1<f64>,
    pub(crate) best: Array1<f64>,
    pub(crate) clipped: Array1<f64>,
    pub(crate) delta: Array1<f64>,
    /// `0` anchor, `1` best, `2` trial.
    pub(crate) chosen: u8,
}

impl SearchBuffers {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            anchor: Array1::zeros(n),
            trial: Array1::zeros(n),
            best: Array1::zeros(n),
            clipped: Array1::zeros(n),
            delta: Array1::zeros(n),
            chosen: 0,
        }
    }

    pub(crate) fn copy_anchor_to_best(&mut self) {
        let SearchBuffers { anchor, best, .. } = self;
        assign_fit(best, anchor.view());
    }

    pub(crate) fn copy_trial_to_best(&mut self) {
        let SearchBuffers { trial, best, .. } = self;
        assign_fit(best, trial.view());
    }

    pub(crate) fn result(&self) -> ArrayView1<'_, f64> {
        match self.chosen {
            1 => self.best.view(),
            2 => self.trial.view(),
            _ => self.anchor.view(),
        }
    }
}

impl Default for SearchBuffers {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Copy `src` into `dst` when the lengths match. A length change takes
/// ownership of `src`.
pub(crate) fn absorb(dst: &mut Array1<f64>, src: Array1<f64>) {
    if dst.len() == src.len() {
        assign_fit(dst, src.view());
        recycle_array(src);
    } else {
        *dst = src;
    }
}

struct Pool {
    bufs: Vec<Array1<f64>>,
    /// How many buffers this thread retains. L-BFGS raises it to
    /// `2 * max_pairs`, the pair history, when that exceeds the default.
    keep: usize,
}

thread_local! {
    static POOL: std::cell::RefCell<Pool> = std::cell::RefCell::new(Pool {
        bufs: Vec::with_capacity(32),
        keep: 32,
    });
}

/// Remember that the caller holds `count` length-n history buffers.
///
/// The pool keeps at least that many recycled buffers. A smaller cap
/// would drop a pair and allocate it again on the next take.
pub fn note_buffer_need(count: usize) {
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if count > pool.keep {
            pool.keep = count;
            pool.bufs.reserve(count);
        }
    });
}

/// How many recycled buffers this thread will retain.
pub fn buffer_keep() -> usize {
    POOL.with(|pool| pool.borrow().keep)
}

/// A length-n vector, reused after a previous owner recycled it.
pub fn take_array(n: usize) -> Array1<f64> {
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if let Some(i) = pool.bufs.iter().position(|a| a.len() == n) {
            pool.bufs.swap_remove(i)
        } else {
            drop(pool);
            Array1::zeros(n)
        }
    })
}

/// Keep `array` for a later [`take_array`] of the same length.
pub fn recycle_array(array: Array1<f64>) {
    if array.is_empty() {
        return;
    }
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let n = array.len();
        let held = pool.bufs.iter().filter(|buf| buf.len() == n).count();
        let keep = pool.keep;
        if held < keep {
            pool.bufs.push(array);
        }
    });
}
