`Solver::set_highs_time_limit` sets `HighsOptions::time_limit` on a session.
It refuses a value that is NaN, infinite or not positive, keeps the limit as
it was, and returns false in a build without the `highs` feature.

A HiGHS solve that reaches a wall-clock limit is now `Error::HighsTimedOut`,
which carries the limit the solve ran under, instead of a generic status
string. A projection QP that uses up its fixed iteration budget is
`Error::HighsIterationLimit`. The projection QP still has no hidden
wall-clock cap. The Hessian step used to discard a timeout and fall through to
the shifted Newton, RFO or L-BFGS step without a trace. It still falls
through, since the step can go on without the QP, and `Solver::highs_timeouts`
counts each fallback. A step the box, trust radius or equalities must shape
returns the error.
