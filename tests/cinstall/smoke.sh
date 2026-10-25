#!/bin/sh
# Install rgmin with cargo cinstall and compile C and C++ against that prefix.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$root"
prefix=${1:-${TMPDIR:-/tmp}/rgmin-cinstall-$$}
rm -rf "$prefix"
mkdir -p "$prefix"

cargo cinstall --locked --debug --prefix "$prefix" --features capi

pc=$(find "$prefix" -name rgmin.pc -print)
if [ -z "$pc" ]; then
    echo "rgmin.pc was not installed under $prefix" >&2
    find "$prefix" -print >&2
    exit 1
fi
export PKG_CONFIG_PATH=$(dirname "$pc")${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
pkg-config --exists rgmin

header=$prefix/include/rgmin.h
dlpack=$prefix/include/dlpack/dlpack.h
test -f "$header"
test -f "$dlpack"
cmp "$header" include/rgmin.h
cmp "$dlpack" include/dlpack/dlpack.h

if grep -q 'HighsInt' "$header"; then
    echo "installed rgmin.h names HighsInt" >&2
    exit 1
fi
if grep -E -q 'namespace[[:space:]]+[A-Za-z_]' "$header"; then
    echo "installed rgmin.h opens a C++ namespace" >&2
    exit 1
fi
if grep -q 'RGMIN\.H_' "$header"; then
    echo "installed rgmin.h has a dotted version macro" >&2
    exit 1
fi
if ! grep -q '#define RGMIN_H_MAJOR ' "$header"; then
    echo "installed rgmin.h lacks RGMIN_H_MAJOR" >&2
    exit 1
fi

cflags=$(pkg-config --cflags rgmin)
libs=$(pkg-config --libs rgmin)
libdir=$(pkg-config --variable=libdir rgmin)
cc=${CC:-cc}
cxx=""
for bin in ${CXX:-} c++ g++ clang++; do
    if [ -z "$bin" ]; then
        continue
    fi
    if printf '#include <cstdio>\nint main(){return 0;}\n' | "$bin" -std=c++17 -x c++ - -c -o /dev/null >/dev/null 2>&1; then
        cxx=$bin
        break
    fi
done
if [ -z "$cxx" ]; then
    echo "no C++ compiler can include <cstdio>" >&2
    exit 1
fi

# pkg-config flags are split on spaces. The prefix path has none.
# shellcheck disable=SC2086
"$cc" -std=c11 -Wall -Wextra -Werror $cflags \
    tests/cinstall/smoke.c $libs -Wl,-rpath,"$libdir" -o "$prefix/smoke_c"
# shellcheck disable=SC2086
"$cxx" -std=c++17 -Wall -Wextra -Werror $cflags \
    tests/cinstall/smoke.cpp $libs -Wl,-rpath,"$libdir" -o "$prefix/smoke_cxx"

export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
out_c=$("$prefix/smoke_c")
out_cxx=$("$prefix/smoke_cxx")
mod=$(pkg-config --modversion rgmin)
expected="$mod $mod"
if [ "$out_c" != "$expected" ]; then
    echo "C smoke: expected '$expected', got '$out_c'" >&2
    exit 1
fi
if [ "$out_cxx" != "$expected" ]; then
    echo "C++ smoke: expected '$expected', got '$out_cxx'" >&2
    exit 1
fi
printf 'cinstall smoke ok: %s\n' "$out_c"
