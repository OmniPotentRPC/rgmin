`benchmarks/minimize.py` is an airspeed velocity (asv) suite that times one `rgmin_minimize` call on a convex quadratic in 16 and 64 dimensions.
`snakemake -c 1` calibrates an upper bound for each benchmark from repeated runs of `develop`.
`snakemake -c 1 pr_gate` exits non-zero when a benchmark is slower than that bound, and `benchflow/README.md` describes the workflow.
