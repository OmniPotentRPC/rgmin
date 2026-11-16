`cargo cinstall` installs the hand-written `include/rgmin.h` and `dlpack/dlpack.h`.
The header compiles as C and as C++.
Version macros are `RGMIN_H_MAJOR`, `RGMIN_H_MINOR`, and `RGMIN_H_PATCH`.
`HighsInt` stays out of the public header.
Continuous integration (CI) installs into a prefix and compiles a C program and a C++ program against that header and the `rgmin` pkg-config file.
