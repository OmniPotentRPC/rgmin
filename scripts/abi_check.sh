#!/bin/sh
# Compare the C ABI of this build with abi/rgmin-symbols.txt and abi/rgmin.xml.
# RGMIN_ABI_RECORD=1 writes both from the current library.
# rustc already passes an anonymous version script, so a second map is
# rejected by ld. objcopy localizes every dynamic symbol that is not rgmin_*.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
cargo build --locked --features capi
lib=$root/target/debug/librgmin.so
test -f "$lib"
symbols=$root/target/rgmin-symbols.txt
nm -D --defined-only "$lib" | awk '$3 ~ /^rgmin_/ { print $3 }' | sort -u > "$symbols"
if [ "${RGMIN_ABI_RECORD:-0}" = 1 ]; then
    cp "$symbols" "$root/abi/rgmin-symbols.txt"
fi
test -f "$root/abi/rgmin-symbols.txt"
diff -u "$root/abi/rgmin-symbols.txt" "$symbols"
filtered=$root/target/librgmin-cabi.so
cp "$lib" "$filtered"
objcopy --keep-global-symbols="$root/abi/rgmin-symbols.txt" "$filtered"
current=$root/target/rgmin-abi.xml
abidw --no-corpus-path "$filtered" \
    | sed "s/ comp-dir-path='[^']*'//g" > "$current"
if [ "${RGMIN_ABI_RECORD:-0}" = 1 ]; then
    cp "$current" "$root/abi/rgmin.xml"
    exit 0
fi
test -f "$root/abi/rgmin.xml"
abidiff --no-added-syms "$root/abi/rgmin.xml" "$current"
