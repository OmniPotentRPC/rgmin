Session SR2 under `Accept::Energy` or `Accept::Nonmonotone` starts as BFGS and SR1 do.
It takes `istep` along `B = I`, sets `B = I y·y / s·y` before the first update, and resets `B` to `I` after a refused step.
On boxed Rosenbrock 2D from `(-1.2, 1.0)` it used to stay at `f = 24.2`.
`Accept::None` is unchanged.
