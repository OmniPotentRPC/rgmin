`ManifoldKind::MultinomialDoublyStochastic` (`multinomialdoublystochastic`, `RGMIN_MANIFOLD_MULTINOMIAL_DS` = 18): manopt `multinomialdoublystochasticfactory`, the doubly-stochastic matrices.
Packed row-major `n^2`, `n >= 2`.
Fisher-orthogonal projection solves the `[I X; X^T I]` saddle, retraction is `X ⊙ exp(V ⊘ X)` then Sinkhorn, transport is projection at arrival.
A 3N cluster is rejected.
Tokens 7-10 stay reserved.
`abi_minor` 17.
