//! One libm for results that must match across instruction sets.
//!
//! `f64::exp` and `f64::sin` call the host math library. x86_64 and
//! aarch64 do not agree on those bits. The Exp pair weight and the
//! sine probe that estimates μ go through these functions.

/// `exp` from the `libm` crate.
#[must_use]
pub(crate) fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// `sin` from the `libm` crate.
#[must_use]
pub(crate) fn sin(x: f64) -> f64 {
    libm::sin(x)
}
