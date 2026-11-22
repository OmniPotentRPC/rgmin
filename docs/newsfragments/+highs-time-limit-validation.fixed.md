`HighsOptions::time_limit` overrides that are negative, `NaN`, or infinite are
rejected with `Error::HighsTimeLimit` before reaching HiGHS on both the
projection and feasible step QP solves.
