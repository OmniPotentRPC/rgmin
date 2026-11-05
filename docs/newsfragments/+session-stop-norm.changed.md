`Control` and `Report` each gain a public `stop_norm` field, and `StopNorm` gains the `Rms` variant.
A `Control` struct literal that names every field needs `stop_norm` or `..Default::default()`.
A `Report` struct literal needs `stop_norm`, and an exhaustive `match` on `StopNorm` needs an `Rms` arm.
