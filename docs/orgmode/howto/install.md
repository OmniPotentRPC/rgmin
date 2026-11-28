

# Install rgmin

rgmin 0.8.0 is a Rust crate with a C ABI.
It needs Rust 1.88 or newer, the `rust-version` in `Cargo.toml`.
This page covers the crate, the C library from cargo-c or Meson, the two Python packages and the LAMMPS plugin.


## From crates.io

Add rgmin, the `ndarray` it takes, and `eindir-core` for the objective traits:

```bash
cargo add rgmin@0.8 ndarray@0.17 eindir-core@0.6
```

rgmin takes `eindir-core` 0.6 from crates.io too, so Cargo builds one copy of it.


### Features

`simd` is on by default.
The others are off until you name them.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Feature</th>
<th scope="col" class="org-left">Adds</th>
<th scope="col" class="org-left">Needs</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>simd</code></td>
<td class="org-left">single instruction, multiple data (SIMD) versions of the vector loops, chosen for the CPU at run time</td>
<td class="org-left">nothing more</td>
</tr>

<tr>
<td class="org-left"><code>capi</code></td>
<td class="org-left">the C symbols behind <code>include/rgmin.h</code></td>
<td class="org-left">nothing more</td>
</tr>

<tr>
<td class="org-left"><code>par</code></td>
<td class="org-left">rayon threads for long dot products and vector updates</td>
<td class="org-left">nothing more</td>
</tr>

<tr>
<td class="org-left"><code>highs</code></td>
<td class="org-left">the HiGHS step of </td>
<td class="org-left">CMake, a C++ compiler and libclang, since <code>highs-sys</code> builds HiGHS from source</td>
</tr>

<tr>
<td class="org-left"><code>slepc</code></td>
<td class="org-left"><code>EigensolverKind::Slepc</code></td>
<td class="org-left">PETSc and SLEPc, and a C compiler</td>
</tr>

<tr>
<td class="org-left"><code>primme</code></td>
<td class="org-left"><code>EigensolverKind::Primme</code></td>
<td class="org-left">PRIMME 3.0 or newer, and a C compiler</td>
</tr>

<tr>
<td class="org-left"><code>chase</code></td>
<td class="org-left"><code>EigensolverKind::Chase</code></td>
<td class="org-left">the compiled ChASE C interface, and a C compiler</td>
</tr>

<tr>
<td class="org-left"><code>libkrylov</code></td>
<td class="org-left"><code>EigensolverKind::Libkrylov</code></td>
<td class="org-left">libkrylov, the gfortran runtime, a Basic Linear Algebra Subprograms (BLAS) library, and a C compiler</td>
</tr>

<tr>
<td class="org-left"><code>readcon</code></td>
<td class="org-left"><code>read_checkpoint</code> and <code>write_checkpoint</code> for CON geometry files</td>
<td class="org-left">nothing more, since it pulls in <code>readcon-core</code></td>
</tr>
</tbody>
</table>

For example:

```bash
cargo add rgmin@0.8 --features highs
```

The `slepc`, `primme`, `chase` and `libkrylov` features each look for a library.
They try pkg-config first.
Next they try a directory variable.
That is `PETSC_DIR` and `SLEPC_DIR`, `PRIMME_DIR`, `CHASE_DIR` or `LIBKRYLOV_DIR`.
Last they try `CONDA_PREFIX` and then `/usr`.
If the library is missing, the build still passes.
That eigensolver kind then returns `Error::EigenUnavailable` at run time.
Cargo prints the build warning for a dependency only with `cargo build -vv`.

The `periodic-product` feature adds `periodic_product::periodic_product_spectrum`, which multiplies short transfer matrices and finds the multipliers of the product at a chosen binary precision.
It pulls in `rug`, which links `gmp-mpfr-sys`.
Both crates are LGPL-3.0 or later, unlike the MIT licence of rgmin, and a binary that links either one statically must meet the LGPL terms for it.
`gmp-mpfr-sys` builds GMP, MPFR and MPC from source with autotools, so it needs `m4`, `make`, `diffutils` and a C compiler.
The crate always builds a static library. When `capi` is also enabled, including under `--all-features`, the module is left out so that library does not link GMP. `--features periodic-product` is the build that includes it.


## From git

To build a commit that is not on crates.io, such as the tip of `develop`, depend on the repository:

```bash
cargo add rgmin --git https://github.com/OmniPotentRPC/rgmin --branch develop
cargo add ndarray@0.17 eindir-core@0.6
```

`develop` also takes `eindir-core` 0.6 from crates.io, so this builds one copy of it as well.
Releases are tagged `vX.Y.Z`; `--tag` in place of `--branch develop` pins one.


## Check the install

```bash
cargo build
```

is a first program.


## From a checkout

A clone starts on `main`.
These pages describe `develop`:

```bash
git clone https://github.com/OmniPotentRPC/rgmin.git
cd rgmin
git switch develop
cargo test --features capi
```

With `--features highs`, some tests check HiGHS steps against SciPy through `tests/highs_scipy_gold.py`.
They run `python3`, or the interpreter named in `PYTHON`, and that interpreter needs SciPy:

```bash
PYTHON=python3.12 cargo test --features highs
```


## For C and C++ with cargo-c

cargo-c builds and installs the C library, the headers and a pkg-config file from a checkout.
rgmin needs cargo-c 0.10.17 or newer, the `min_version` in `Cargo.toml`.
Each cargo-c release needs a recent Rust: 0.10.17 builds with Rust 1.89, and 0.10.26 needs Rust 1.97.
On Linux cargo-c also needs pkg-config and the OpenSSL headers, such as `libssl-dev` on Debian and Ubuntu.
This is the cargo-c that CI installs:

```bash
cargo +1.89.0 install cargo-c --locked --version 0.10.17+cargo-0.92.0
```

With Rust 1.97 or newer, `cargo install cargo-c --locked` installs the newest release instead.
Then, in the checkout:

```bash
cargo cinstall --prefix=/usr/local --features capi
```

`cargo cinstall` builds in release mode.
It installs `librgmin` as a shared and a static library, `rgmin.h`, the C++ headers `rgmin/optimize.hpp` and `rgbind/rgbind.hpp`, the `xts` headers, `dlpack/dlpack.h` and the `rgmin` pkg-config file.
On Debian and Ubuntu the library and the pkg-config file go under the multiarch directory of the prefix, such as `lib/x86_64-linux-gnu`.
Compile a C program against the install with pkg-config:

```bash
cc $(pkg-config --cflags rgmin) prog.c $(pkg-config --libs rgmin) -o prog
```

If the dynamic loader does not search that library directory, run the program with it in `LD_LIBRARY_PATH`.

walks through the C calls.
`tests/cinstall/smoke.sh` is the check that CI runs: it installs into `target/rgmin-cinstall` and compiles a C and a C++ program.


## For C and C++ with Meson

The repository is also a Meson project.
It needs Meson 1.3 or newer, Ninja, a C and a C++ compiler, Python 3, and Cargo with Rust 1.88 or newer.
Meson runs Cargo to build `librgmin` and declares `rgmin_dep`: the include directory, the static library, the system libraries it needs, and the define `USE_RGMIN`.

To use rgmin as a subproject, add `subprojects/rgmin.wrap` to your project:

```ini
[wrap-git]
url = https://github.com/OmniPotentRPC/rgmin.git
revision = develop
depth = 1

[provide]
dependency_names = rgmin
```

A release tag in `revision` pins that release.
Then ask for the dependency in your `meson.build`:

```meson
rgmin_dep = dependency('rgmin')
```

The `rgmin_features` option holds the Cargo features, `capi,highs` by default.
With `highs`, the static library holds HiGHS, which is C++.
A C++ target that links `rgmin_dep` gets the C++ standard library from its linker, and a C target does not and fails to link.
`highs` also needs a libclang new enough for the HiGHS headers.
For a C program, or where libclang is missing, build only `capi`:

```bash
meson setup build -Drgmin:rgmin_features=capi
meson compile -C build
```

A build of the repository itself makes `librgmin.so` and `librgmin.a`:

```bash
meson setup build
meson compile -C build
meson install -C build
```

`meson install` installs the shared library only, with no headers or pkg-config file.
Use cargo-c for a full install.


## Python

PyPI has the nanobind package: the free-threaded module `rgmin._core` and the ASE optimizers in `rgmin.ase`.
The `ase` extra adds ASE, which `rgmin.ase` needs and `import rgmin` does not load:

```bash
python -m pip install rgmin
python -m pip install 'rgmin[ase]'
```

There are wheels for Linux with glibc 2.28 or newer (`manylinux_2_28`) on `x86_64` and `aarch64`, macOS 14 or newer on `arm64` and `x86_64`, and Windows on `x86_64`.
CPython 3.10 and 3.11 each have a wheel, one abi3 wheel serves 3.12 and newer, and the free-threaded 3.14t and 3.15t each have a wheel.
On any other platform or interpreter, pip builds the sdist, which has the requirements of a checkout below, the Corrosion download included.

The repository root builds the same package with scikit-build-core.
It needs a C++17 compiler, CMake 3.15 or newer, and Rust 1.88 or newer.
CMake downloads Corrosion from GitHub while it configures.
From the checkout:

```bash
python -m pip install '.[ase]'
```

`python/` builds the PyO3 module with maturin.
It is not on PyPI.
It adds `rgmin.optimize`, the ASE drop-in of , and needs Rust 1.88 or newer:

```bash
python -m pip install './python[ase]'
```

Both install the import package `rgmin`, so an environment holds one of them.
shows both.


## LAMMPS plugin

`lammps/` builds `min_style rgmin`, a plugin for LAMMPS 22Jul2025 or newer built with CMake and the `PLUGIN` package.
gives the build commands.
