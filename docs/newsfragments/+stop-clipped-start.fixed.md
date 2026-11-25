Python `minimize` now measures a no-move L-BFGS run from the point after
projection onto objective bounds. It previously copied `x0` before the
projection, so a run that never left the projected point reported
`Stop.BUDGET` instead of `Stop.LINE_SEARCH`.
