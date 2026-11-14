// Copyright (c) rgmin developers
//! Inclusive wall-time counters for a scaling profile.
//!
//! The counters stay at zero until [`enable`]. A region that calls
//! another counted region includes that time: the line search includes
//! the two-loop, and the two-loop includes the dots and axpys it makes.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static ON: AtomicBool = AtomicBool::new(false);

/// `true` after [`enable`].
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Start recording. Later regions add into the process-wide counters.
pub fn enable() {
    ON.store(true, Ordering::Relaxed);
}

/// Zero every counter. Does not disable recording.
pub fn reset() {
    TWO_LOOP.reset();
    LINE_SEARCH.reset();
    DOT.reset();
    AXPY.reset();
    RAYON.reset();
    DOT_CALLS.reset();
    AXPY_CALLS.reset();
    RAYON_CALLS.reset();
    COPY.reset();
    ELEM.reset();
}

struct Counter(AtomicU64);

impl Counter {
    const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    fn add(&self, n: u64) {
        if enabled() {
            self.0.fetch_add(n, Ordering::Relaxed);
        }
    }

    fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    fn reset(&self) {
        self.0.store(0, Ordering::Relaxed);
    }
}

static TWO_LOOP: Counter = Counter::new();
static LINE_SEARCH: Counter = Counter::new();
static DOT: Counter = Counter::new();
static AXPY: Counter = Counter::new();
static RAYON: Counter = Counter::new();
static DOT_CALLS: Counter = Counter::new();
static AXPY_CALLS: Counter = Counter::new();
static RAYON_CALLS: Counter = Counter::new();
static COPY: Counter = Counter::new();
static ELEM: Counter = Counter::new();

pub(crate) struct Guard {
    start: Instant,
    slot: &'static Counter,
}

impl Guard {
    fn enter(slot: &'static Counter) -> Option<Self> {
        if enabled() {
            Some(Self {
                start: Instant::now(),
                slot,
            })
        } else {
            None
        }
    }

    pub(crate) fn two_loop() -> Option<Self> {
        Self::enter(&TWO_LOOP)
    }

    pub(crate) fn line_search() -> Option<Self> {
        Self::enter(&LINE_SEARCH)
    }

    pub(crate) fn dot() -> Option<Self> {
        DOT_CALLS.add(1);
        Self::enter(&DOT)
    }

    pub(crate) fn axpy() -> Option<Self> {
        AXPY_CALLS.add(1);
        Self::enter(&AXPY)
    }

    #[cfg(feature = "par")]
    pub(crate) fn rayon() -> Option<Self> {
        RAYON_CALLS.add(1);
        Self::enter(&RAYON)
    }

    pub(crate) fn copy() -> Option<Self> {
        Self::enter(&COPY)
    }

    pub(crate) fn elem() -> Option<Self> {
        Self::enter(&ELEM)
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.slot
            .add(u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX));
    }
}

/// Inclusive nanoseconds and call counts since the last [`reset`].
#[derive(Clone, Copy, Debug)]
pub struct PhaseTimes {
    /// `Lbfgs::apply_two_loop`, including its dots and axpys.
    pub two_loop_ns: u64,
    /// `Lbfgs::step_objective`, including the two-loop and the
    /// objective evaluations the line search makes.
    pub line_search_ns: u64,
    /// Every `vecops::dot` and `vecops::nrm2`.
    pub dot_ns: u64,
    /// Every `vecops::axpy`.
    pub axpy_ns: u64,
    /// The parallel body of `dot`, `axpy`, and `sum` (the rayon join).
    pub rayon_ns: u64,
    /// How many dots were timed.
    pub dot_calls: u64,
    /// How many axpys were timed.
    pub axpy_calls: u64,
    /// How many rayon joins in `vecops` were timed.
    pub rayon_calls: u64,
    /// `assign_fit` copies.
    pub copy_ns: u64,
    /// Serial element sweeps: clip, project, negate, add, finite.
    pub elem_ns: u64,
}

/// Snapshot the counters.
pub fn times() -> PhaseTimes {
    PhaseTimes {
        two_loop_ns: TWO_LOOP.get(),
        line_search_ns: LINE_SEARCH.get(),
        dot_ns: DOT.get(),
        axpy_ns: AXPY.get(),
        rayon_ns: RAYON.get(),
        dot_calls: DOT_CALLS.get(),
        axpy_calls: AXPY_CALLS.get(),
        rayon_calls: RAYON_CALLS.get(),
        copy_ns: COPY.get(),
        elem_ns: ELEM.get(),
    }
}
