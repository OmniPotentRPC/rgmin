Session BFGS and SR1 under `Accept::Energy` or `Accept::Nonmonotone` take `istep` along the starting identity and rescale it by `s·y / y·y` (Nocedal-Wright 6.20) before the first update, as L-BFGS does with an empty memory.
The raw `-g` step rose on all ten halvings, so on boxed Rosenbrock 2D from `(-1.2, 1.0)` they stayed at `f = 24.2`.
A refused step resets the matrix to that identity, so SR1 no longer retries an uphill direction from the same point.
`Accept::None` is unchanged.
