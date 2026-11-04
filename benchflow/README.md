# The benchmark suite

This suite times the C minimize entry of rgmin.
It reports a slowdown between two revisions.
Timings come from airspeed velocity (asv).
Memory comes from memray and its `asv_bench_memray` plugin (`ray_*` methods).
Snakemake is the only driver.
Tool versions are pinned in `benchflow/requirements.txt`.
`asv.conf.json` repeats the same versions.
The suite uses three commands:

```sh
python -m pip install -r benchflow/requirements.txt
snakemake -c 1
python -m unittest discover -s benchflow
```

## Noise calibration

This section explains how the suite turns measured noise into an upper bound.
`benchflow/config.yaml` sets `rounds` (10) and `runs` (10).
Those runs of `develop` on one process measure the noise.
Each run uses `--interleave-rounds` without `--quick` or `--skip-existing`.
Two runs of the same commit give a calibration ratio.
That ratio is the median of the second run over the median of the first.
The relative threshold uses the median absolute deviation (MAD) of those ratios.
The threshold is the largest of these four values:

```text
max(0.05, 3 * MAD, the 99th percentile of |ratio - 1|, the maximum |ratio - 1|)
```

Each benchmark's upper bound is one plus that threshold.
So the upper bound is at least the widest calibration ratio.
It is also at least five percent above one.

## Samples

This section explains why each sample covers many calls.
`time_minimize` sets `number = 2000`.
Each recorded sample is the mean of 2000 fresh minimizations, divided back to one call.
One call takes a few tens of microseconds.
That is not enough work for one sample.

## Run output and the machine record

This section explains what those timing runs leave behind.
Each run writes asv results under `benchflow/out/results/<run>`.
The exporter takes the result for the run's commit, machine, and environment.
It fails when no result or more than one result matches.

For the machine, every run uses the name `bench-runner`.
Its `os`, `arch`, `cpu`, `num_cpu`, and `ram` fields have fixed generic values.
That record is in `benchflow/out/home/.asv-machine.json`, the asv machine file.
The machine check reads only the `bench-runner` record.
It ignores every other record in that file.

## Revision comparison

This section explains how the suite compares revisions and reports a slowdown.
`snakemake -c 1 pr_compare` compares two distinct revisions, `base` and `head`.
It runs one `asv continuous --interleave-rounds` for them.
It writes the table to `benchflow/out/pr_compare.txt`.
A table row fails when its head/base median ratio is above its own upper bound.
The result goes to `benchflow/out/pr_compare.status`.
`pr_compare` itself exits zero.
Equal revisions do not run a comparison, and their status is non-zero.
`pr_gate` exits with that status, so it fails on a slowdown.
Continuous integration should run `pr_gate`.

## The library build

This section describes the library that the timings call and how it is built.
The timed call is `rgmin_minimize` in the installed `librgmin.so`.
Meson configures that library with `--buildtype=release` and `-Drgmin_features=capi`.
The project default buildtype is debugoptimized.
Meson's own default is debug.
A configure without the release flag would time a different build.
The build selects Cargo's release profile when the buildtype is not debug.
The build tree is the commit cache.
The output of `meson introspect --buildoptions` goes beside that tree.

After the build, `meson install` puts two files into the environment `lib` directory.
They are `librgmin.so` and the thin `librgmin_asv` helper.
The helper is off unless `-Dwith_asv_helper=true`.
Benchmarks load those installed files and fail if either is missing.
Once those files are loaded, each timed call allocates its own start vector and borrows it for that call.
`warmup_time` uses the same entry.
Those calls are not samples.
