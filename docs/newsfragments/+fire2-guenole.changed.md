FIRE 2.0 now follows Guénolé et al.'s semi-implicit update order, half-step-back uphill correction with a fresh corrected-point force, initial time-step delay, 2,000-step non-positive-power limit, and cap-consistent velocity.
An installed exponential preconditioner uses `P` as the metric: power is `F · v`, and the kick and the mix use `P^{-1} F`.
