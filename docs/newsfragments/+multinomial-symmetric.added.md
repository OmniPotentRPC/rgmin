`ManifoldKind::MultinomialSymmetric` (`multinomialsymmetric`, `RGMIN_MANIFOLD_MULTINOMIAL_SYM` = 19): manopt `multinomialsymmetricfactory`, the symmetric doubly-stochastic matrices.
Packed row-major `n^2`, `n >= 2`.
Fisher-orthogonal projection solves `(I + X) alpha = V 1`, retraction is `X ⊙ exp(V ⊘ X)` then Sinkhorn then symmetrize, transport is projection at arrival.
A 3N cluster is rejected.
Tokens 7-10 stay reserved.
`abi_minor` 18.
