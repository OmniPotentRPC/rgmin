The docs index and `reference/architecture` define the hourglass and its waist, the C ABI that C and C++ callers go through.
A figure that `docs/figures/architecture/draw_hourglass.py` draws from the headers and `src/ffi.rs` replaces the old text diagram.
The page gives the ABI version from `include/rgmin.h`, 1.29 with layout revision 4, instead of 1.9 with revision 2.
