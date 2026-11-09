`Solver::set_highs_trust` stores the radius on the HiGHS step
that `set_trust` already uses. A build without HiGHS reports
that it did not store the radius.
