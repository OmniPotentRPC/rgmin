//! Closed DLA-Future range tokens. These compile without a linked
//! DLA-Future so a host can name the partial-spectrum window.

/// Typed DLA-Future extras. `begin` is the first eigenpair index.
/// The waist only accepts `begin == 0` (`il = 1` in 1-based PDSYEVR).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DlaFutureParams {
    /// First index of the requested window. Must be 0.
    pub begin: usize,
}

impl Default for DlaFutureParams {
    fn default() -> Self {
        Self { begin: 0 }
    }
}
