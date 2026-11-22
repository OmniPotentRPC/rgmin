# GROMACS 2023.3 reference build

The reference transcripts were generated from the upstream
`gromacs-2023.3.tar.gz` source archive:

- URL: `https://ftp.gromacs.org/gromacs/gromacs-2023.3.tar.gz`
- SHA-256: `4ec8f8d0c7af76b13f8fd16db8e2c120e749de439ae9554d9f653f812d78d1cb`
- C compiler: `/usr/bin/gcc`, GCC 13.3.0
- C++ compiler: `/usr/bin/g++`, GCC 13.3.0
- CMake: 3.28.3

Both builds used these CMake options:

```text
-DGMX_MPI=OFF
-DGMX_THREAD_MPI=ON
-DGMX_GPU=OFF
-DGMX_SIMD=AVX2_256
-DGMX_BUILD_OWN_FFTW=ON
-DBUILD_TESTING=OFF
-DREGRESSIONTEST_DOWNLOAD=OFF
-DCMAKE_BUILD_TYPE=Release
```

The double build additionally used `-DGMX_DOUBLE=ON`; the mixed-precision
build used `-DGMX_DOUBLE=OFF`. The complete compiler and dependency details
reported by each executable are in `gromacs_cg_reference/gmx_d-version.txt`
and `gromacs_cg_reference/gmx-version.txt`.

The transcripts were generated with:

```text
python3 tests/data/gromacs_cg_reference.py \
  --gmx-double /tmp/rgmin-gromacs-2023.3/install-double/bin/gmx_d \
  --gmx-single /tmp/rgmin-gromacs-2023.3/install-single/bin/gmx
```

Each case directory contains the exact MDP, coordinates, topology,
`grompp` terminal transcript, `mdrun -v` terminal transcript, full `em.log`,
and final coordinates. `summary.tsv` is parsed directly from those files.
