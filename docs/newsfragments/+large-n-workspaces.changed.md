The eOn conjugate-gradient step fuses the Polak-Ribiere products and swaps its direction buffers instead of copying them.
`examples/large_n.rs` reports the per-step overhead with the force call subtracted, and the force calls to a fixed fmax, on Lennard-Jones and Morse clusters and slabs.
The capped L-BFGS step and FIRE are unchanged. The strong Wolfe search stays opt-in.
