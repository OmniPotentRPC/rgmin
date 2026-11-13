# asv suite

Timings come from asv.
Memory comes from memray through the `asv_bench_memray` plugin (`ray_*` methods).
Snakemake is the only driver.
Tool versions are pinned in `benchflow/requirements.txt` and repeated in `asv.conf.json`.

```sh
python -m pip install -r benchflow/requirements.txt
snakemake -c 1
python -m unittest discover -s benchflow
```

`benchflow/config.yaml` sets `rounds` (10) and `runs` (10).
Those runs of `develop` on one process measure the noise.
Each run uses `--interleave-rounds` and does not pass `--quick` or `--skip-existing`.
Two runs of the same commit give a calibration ratio: the median of the second run over the median of the first.
With MAD for the median absolute deviation of the calibration ratios, the relative threshold is

```text
max(0.05, 3 * MAD, the 99th percentile of |ratio - 1|, the maximum |ratio - 1|)
```

so the upper bound is at least the widest calibration ratio.
It is also at least five percent above one.
Each benchmark keeps its own upper bound.

`time_minimize` sets `number = 2000`.
Each recorded sample is the mean of 2000 fresh minimizations, divided back to one call.
A single call is a few tens of microseconds, which is not enough work for one sample.

Each run writes asv results under `benchflow/out/results/<run>`.
From those results, the exporter takes the one result for the run's commit, machine, and environment.
It fails when no result or more than one result matches.

Every run uses the machine name `bench-runner`.
Its `os`, `arch`, `cpu`, `num_cpu`, and `ram` fields are fixed generic values.
That record is in `benchflow/out/asv/.asv-machine.json`, the asv machine file.
The machine check reads only the `bench-runner` record and ignores every other record in that file.

`snakemake -c 1 pr_compare` compares two distinct revisions, `base` and `head`.
It runs one `asv continuous --interleave-rounds` for them and writes the table to `benchflow/out/pr_compare.txt`.
A row in that table fails when its head/base median ratio is above its own upper bound.
The result goes to `benchflow/out/pr_compare.status`, and `pr_compare` itself exits zero.
Equal revisions do not run a comparison, and their status is non-zero.
`pr_gate` exits with that status, so it fails on a slowdown.
Continuous integration should run `pr_gate`.

The timed call is `rgmin_minimize` in the installed `librgmin.so`.
Meson configures that library with `--buildtype=release` and `-Drgmin_features=capi`.
The project default buildtype is debugoptimized, and Meson's own default is debug, so a configure that omits the flag is not this result.
Cargo's release profile is selected when the buildtype is not debug.
The build tree is the commit cache.
`meson install` puts `librgmin.so` and the thin `librgmin_asv` helper into the environment `lib` directory.
The helper is off unless `-Dwith_asv_helper=true`.
Benchmarks load those installed files and fail if either is missing.
`meson introspect --buildoptions` is written beside the build tree.
Each timed call allocates its own start vector and borrows it for that call.
`warmup_time` uses the same entry.
Those calls are not samples.
