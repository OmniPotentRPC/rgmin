#!/usr/bin/env bash
# Native ThreadSanitizer leg for the Python binding. No container.
#
# Builds a free-threaded CPython with TSan, NumPy from source with the same
# flags, then rgmin: the Rust staticlib with -Zsanitizer=thread and the
# nanobind module with -fsanitize=thread. Runs the ASE contract and threaded
# tests with halt_on_error=1, so the first race fails the leg.
#
# Environment:
#   TSAN_WORK         work tree (default: $PWD/.tsan-work)
#   CPYTHON_VERSION   CPython release to build (default: 3.14.10)
#   RUST_TOOLCHAIN    dated nightly that carries -Zsanitizer (default below)
#   CC, CXX           clang and clang++ (default: clang, clang++)
#   SYMBOLIZER        llvm-symbolizer path (default: first one on PATH)
#   TSAN_LOG          file that receives the full pytest and TSan output
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
work=${TSAN_WORK:-$PWD/.tsan-work}
version=${CPYTHON_VERSION:-3.14.10}
toolchain=${RUST_TOOLCHAIN:-nightly-2026-01-20}
log=${TSAN_LOG:-$work/tsan.log}
prefix=$work/python
venv=$work/venv
build=$work/rust-tsan-build
suppressions=$root/ci/tsan_suppressions.txt

mkdir -p "$work"
export CC=${CC:-clang}
export CXX=${CXX:-clang++}
sanitize="-fsanitize=thread -fno-omit-frame-pointer -g -ffp-contract=off"

symbolizer=${SYMBOLIZER:-}
if [ -z "$symbolizer" ]; then
    symbolizer=$(command -v llvm-symbolizer || true)
fi
if [ -z "$symbolizer" ]; then
    for candidate in /usr/lib/llvm-*/bin/llvm-symbolizer; do
        [ -x "$candidate" ] && symbolizer=$candidate
    done
fi

clang_major=$("$CXX" -dM -E -x c /dev/null | awk '/^#define __clang_major__ / { print $3 }')
echo "clang LLVM major ${clang_major}"

if [ ! -x "$prefix/bin/python3.14t" ]; then
    archive=$work/Python-$version.tgz
    [ -f "$archive" ] || curl -fsSL -o "$archive" "https://www.python.org/ftp/python/$version/Python-$version.tgz"
    rm -rf "$work/Python-$version"
    tar -C "$work" -xzf "$archive"
    mkdir -p "$work/Python-$version/build"
    (
        cd "$work/Python-$version/build"
        ../configure --prefix="$prefix" --disable-gil --with-thread-sanitizer
        make -j"$(nproc)"
        make install
    )
fi
py=$prefix/bin/python3.14t
"$py" -c 'import sys; print(sys.version); assert not sys._is_gil_enabled()'

if [ ! -x "$venv/bin/python" ]; then
    "$py" -m venv "$venv"
fi
py=$venv/bin/python
"$py" -m pip install -U pip
"$py" -m pip install nanobind scikit-build-core cmake ninja pytest pytest-timeout \
    pytest-run-parallel hypothesis "ase==3.23.0"

# An uninstrumented NumPy takes locks and atomics that TSan cannot see, so
# its own caches report races that are not there. Build it with the same
# flags as the extension. The wheel cache keys on the sdist, not on the
# flags, so it is bypassed.
numpy_instrumented() {
    "$py" - <<'PY'
import glob, subprocess, sys
try:
    import numpy
except ImportError:
    sys.exit(1)
lib = glob.glob(numpy.__path__[0] + "/_core/_multiarray_umath*.so")[0]
out = subprocess.run(["nm", "-D", lib], capture_output=True, text=True).stdout
sys.exit(0 if "__tsan" in out else 1)
PY
}
if ! numpy_instrumented; then
    CFLAGS="$sanitize" CXXFLAGS="$sanitize" LDFLAGS="-fsanitize=thread" \
        "$py" -m pip install 'numpy>=2' --no-binary=numpy --no-cache-dir --force-reinstall --no-deps
fi
numpy_instrumented || { echo "NumPy is not built with the thread sanitizer" >&2; exit 1; }

if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi
rustup toolchain install "$toolchain" --component rust-src --profile minimal
echo "rustc LLVM: $(rustup run "$toolchain" rustc -vV | awk '/^LLVM version:/ { print $3 }')"

rm -rf "$build"
(
    cd "$root"
    CFLAGS="$sanitize" CXXFLAGS="$sanitize" LDFLAGS="-fsanitize=thread" \
    CMAKE_ARGS="-DRGMIN_TSAN=ON -DRust_TOOLCHAIN=$toolchain" \
    RUSTUP_TOOLCHAIN=$toolchain \
        "$py" -m pip install --no-build-isolation --no-deps --force-reinstall \
        --config-settings=build-dir="$build" .
)

# The staticlib must carry the instrumentation, or the run proves nothing.
archive=$(find "$build" -name librust_core_demo.a -type f | head -n 1)
[ -n "$archive" ] || { echo "instrumented staticlib not found under $build" >&2; exit 1; }
entries=$(nm -A "$archive" | grep -c __tsan_func_entry || true)
echo "__tsan_func_entry count $entries in $archive"
[ "$entries" -gt 0 ] || { echo "Rust staticlib has no __tsan_func_entry" >&2; exit 1; }

options="halt_on_error=1 allocator_may_return_null=1 suppressions=$suppressions"
[ -n "$symbolizer" ] && options="$options external_symbolizer_path=$symbolizer"
export TSAN_OPTIONS=$options
echo "TSAN_OPTIONS=$TSAN_OPTIONS"

tests=(
    tests/python/test_ase.py
    tests/python/test_ase_stops.py
    tests/python/test_ase_upstream.py
    tests/python/test_threads.py
)
cd "$root"
{
    echo "### serial"
    "$py" -m pytest -s "${tests[@]}" --timeout=900 -o faulthandler_timeout=600 -p no:cacheprovider
    echo "### parallel-threads=4 iterations=5"
    "$py" -m pytest -s "${tests[@]}" --parallel-threads=4 --iterations=5 \
        --timeout=900 -o faulthandler_timeout=600 -p no:cacheprovider
} 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
races=$(grep -c 'WARNING: ThreadSanitizer' "$log" || true)
echo "ThreadSanitizer reports: $races"
exit "$status"
