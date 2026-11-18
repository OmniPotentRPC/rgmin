A zero-budget `gromacs-steep` report now fails when a per-atom gradient norm is not finite. The previous reduction kept a finite atom through `std::max` and dropped the NaN.
