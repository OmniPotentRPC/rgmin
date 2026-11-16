Release builds on Rust 1.93 and 1.94 no longer crash rustc.
The LLVM 21 those toolchains ship segfaulted vectorizing the step cap of the FIRE loop in `Solver::run_fill`, so `cargo build --release` and the Meson build failed with them.
The cap now walks the atoms with `as_chunks` and writes the same bits as before.
CI builds and tests in release on 1.93.0 and 1.94.1.
