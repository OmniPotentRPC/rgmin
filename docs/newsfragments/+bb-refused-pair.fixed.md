Session Barzilai-Borwein drops its stored `(s, y)` pair after `Accept::Energy` or `Accept::Nonmonotone` refuses a step, so the next call starts from the `istep` length.
The refused spectral length used to be retried from the same point.
On boxed Rosenbrock 2D from `(-1.2, 1.0)` the energy stayed at `f = 0.116646`.
`Accept::None` is unchanged.
