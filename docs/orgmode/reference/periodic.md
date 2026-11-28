

# Periodic block determinants

`rgmin::periodic` computes $\ln|\det A|$ of a bordered cyclic block matrix, and factors it so that linear systems with $A$ can be solved afterwards.
The matrix is what multiple shooting builds for a periodic orbit $x_{k+1} = T_k x_k$, $k = 0, \dots, b-1$, whose monodromy is $M = T_{b-1} \cdots T_0$.
The determinant $\det(I - M)$ equals the determinant of the cyclic block matrix without ever forming $M$, so the product of the period never overflows or drops the decaying multipliers.


## The matrix

Row block $i$ holds $-T_i$ in block column $i$ and the identity in block column $(i + 1) \bmod b$.
With $b = 1$ the two coincide and the matrix is $I - T_0$.
Each $T_i$ is a row-major $m \times m$ block.

An optional border of $p$ columns and $p$ rows turns the singular matrix of a periodic orbit into a regular one.
Column $s$ of the border has $m$ entries in the rows of block $b - 1$, and row $s$ of the border has $m$ entries in the columns of block 0.
The trailing $p \times p$ block is zero.
The order of the bordered matrix is $b m + p$.


## Factoring

```rust
use rgmin::periodic::bordered_cyclic_log_abs;

// Free particle: T = [[1, h], [0, 1]] has the Jordan block at the
// multiplier one. Bordering the velocity row and the position column
// leaves |det| equal to the period.
let beta = 7.0;
let b = 4;
let h = beta / b as f64;
let transfers: Vec<Vec<f64>> = (0..b).map(|_| vec![1.0, h, 0.0, 1.0]).collect();
let column = [vec![0.0, 1.0]];
let row = [vec![1.0, 0.0]];

let factor = bordered_cyclic_log_abs(&transfers, 2, &column, &row)?;
assert!((factor.log_abs - beta.ln()).abs() < 1e-12);
let x = factor.solve(&vec![1.0; factor.order()])?;
```

`bordered_cyclic_log_abs(transfers, m, column_border, row_border)` takes one slice of `m * m` entries per block and two lists of `p` border vectors of length `m`.
It returns a `CyclicFactor` with these public items:

-   `log_abs`, the natural logarithm of the absolute value of $\det A$;
-   `norm1`, the 1-norm of $A$, its largest column sum of absolute values;
-   `inverse_norm1`, the Hager-Higham estimate of the 1-norm of $A^{-1}$, usually within a factor of three;
-   `order()`, $b m + p$;
-   `solve(y)`, the vector $x$ with $A x = y$;
-   `solve_transpose(w)`, the vector $z$ with $A^T z = w$.

The product `norm1 * inverse_norm1` is a 1-norm condition estimate of $A$.
Both solves reuse the stored factors and form no new factorization.


## Method

Block column $0, 1, \dots, b - 1$ is eliminated in turn with Householder reflectors on the block rows it couples.
Each new block row is stacked above the running one, and the fill stays in block column $b - 1$ and in the border.
Every transformation is orthogonal, so no intermediate entry grows, and the total work is $O(b m^3)$ instead of the $O((b m)^3)$ of a dense factorization.

Rows and columns are first scaled by powers of two so that each has unit largest entry.
Householder QR is backward stable column by column only, and shooting blocks mix positions and momenta whose rows differ by many decades.
The scaling is exact and is undone in `log_abs` and in both solves.

For one block the scaled matrix is formed from the entries of $I - T_0$ directly, not from $T_0$ and the identity scaled apart.
A unit diagonal entry of $T_0$ makes the matrix entry exactly zero, and scaling the two terms separately by large powers of two would overflow both and leave $\infty - \infty$.


## Errors

`CyclicError` is `#[non_exhaustive]` and has four variants:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Variant</th>
<th scope="col" class="org-left">Raised when</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>Shape(why)</code></td>
<td class="org-left">there are no blocks, <code>m</code> is zero, <code>p</code> exceeds <code>m</code>, a block is not <code>m * m</code>, a border vector is not length <code>m</code>, the border lists differ in length, a dimension overflows, or a right-hand side has the wrong length</td>
</tr>

<tr>
<td class="org-left"><code>NonFinite</code></td>
<td class="org-left">an input entry is NaN or infinite, or a triangular diagonal produced by the factorization is not finite</td>
</tr>

<tr>
<td class="org-left"><code>NonFiniteNormEstimate</code></td>
<td class="org-left">the 1-norm of the matrix or the estimate of the 1-norm of its inverse overflows or becomes NaN</td>
</tr>

<tr>
<td class="org-left"><code>Singular</code></td>
<td class="org-left">a triangular diagonal entry is exactly zero</td>
</tr>
</tbody>
</table>

A singular matrix is an expected outcome: the identity transfers of the unbordered periodic problem, for example, have $\det(I - M) = 0$ and are reported as `Singular`.
A matrix that is finite and nonsingular but whose inverse has entries beyond the range of `f64` is reported as `NonFiniteNormEstimate`, not as `Singular`: for one block $T_0 = [[1, \epsilon], [\epsilon, 2]]$ the inverse of $I - T_0$ is of order $\epsilon^{-2}$, so `epsilon = 1e-150` factors and `epsilon = 1e-160` and `epsilon = 1e-200` return `NonFiniteNormEstimate`.
No error is turned into a default value.
