# Benchmarks

Force calls and median wall time for L-BFGS, FIRE and conjugate gradients on the extended Rosenbrock function, Lennard-Jones clusters and the periodic Morse platinum cluster. `rgmin-cg` is the eOn conjugate-gradient step. The eOn cap, FIRE and that CG fill the gradient buffer in place. What was not adopted is `notes.md`.

Raw rows are `raw/rgmin.csv`, `raw/ase.csv`, `raw/scipy.csv` and `raw/eon.csv`. The table is `results.md`. The larger set is `examples/bench_algo.rs`, with counts in `raw/algo.csv` and the eOn step log in `raw/steps.csv`. `notes.md` records what was not adopted.

`eon_ref.cpp` is the eOn client step from TheochemUI/eOn commit `579738a9387e8d0ea55496f45dd057812cfab892`: `LBFGS::getStep`, `FIRE::step` and `ConjugateGradients::single_step`, with that commit's default parameters.

```
env -i PATH HOME LANG cargo run --release --example bench_suite -- bench/raw/rgmin.csv
g++ -O2 -std=c++20 -o /tmp/eon_ref bench/eon_ref.cpp
env -i PATH HOME LANG /tmp/eon_ref "$PWD" bench/raw/eon.csv
python3 -m venv /tmp/rgmin-bench-venv
/tmp/rgmin-bench-venv/bin/pip install ase scipy numpy
env -i PATH="/tmp/rgmin-bench-venv/bin:$PATH" HOME="$HOME" LANG="${LANG:-C.UTF-8}" python bench/compare.py
```

`compare.py` rewrites `results.md` from `rgmin.csv`, `eon.csv`, `ase.csv` and `scipy.csv`.

`examples/large_n.rs` times the capped L-BFGS step, FIRE 1.0 and the eOn conjugate-gradient step on Lennard-Jones and Morse clusters and slabs at 500, 5 000 and 50 000 atoms. The overhead column is wall time minus the force call. The CSV is not checked in; the example writes it.

```
cargo run --release --example large_n -- bench/raw/large_n.csv
g++ -O2 -std=c++20 -o /tmp/large_n_eon bench/large_n_eon.cpp
/tmp/large_n_eon bench/raw/large_n_eon.csv
```

`large_n_eon.cpp` is the L-BFGS and FIRE step from `eon_ref.cpp` on those same geometries. The full eOn client is a separate build.
