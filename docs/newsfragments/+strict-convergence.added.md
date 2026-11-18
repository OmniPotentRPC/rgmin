Added opt-in strict convergence checks through
`Control::with_require_convergence(bool)`.

When strict convergence is on, drivers now return
`Error::NotConverged(Box<Report>)` on non-converged budget and stalled
exits, and `Report` carries a `converged` flag so callers can inspect
convergence without recomputing the stop test.

The C ABI now exposes the same report flag.
