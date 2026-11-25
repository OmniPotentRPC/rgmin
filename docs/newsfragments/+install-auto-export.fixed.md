The shared and static libraries no longer export the internal `install_auto`
helper behind `RGMIN_PRECON_AUTO` as an unprefixed C symbol. CI now fails when
either library exports a name outside the `rgmin_` namespace, other than the
eindir-core C ABI and toolchain names, or an `rgmin_` name missing from
`include/rgmin.h` and the ABI baseline.
