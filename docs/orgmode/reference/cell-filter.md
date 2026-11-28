

# Cell filter

`CellFilter` relaxes Cartesian positions and the periodic cell as one Euclidean objective.
L-BFGS, FIRE, FIRE 2.0, and conjugate gradients minimize it through the same `DifferentiableObjective` handle they already use.
It is opt-in.
Existing constructors keep their defaults, and the Python ASE optimizers still call ASE's own cell filters.

The callback receives the Cartesian positions and the row-major cell, after a mask or a hydrostatic projection, and returns a `CellSample`.
`CellSample` holds the potential energy, the Cartesian forces, and the Cauchy stress.
A force is the negative gradient of the energy with respect to the Cartesian position.
The stress is row-major and uses the sign of ASE `get_stress(voigt=False)`.


## Coordinates

Lattice vectors are rows of the cell.
The deformation gradient $F$ satisfies $\mathrm{cell} = \mathrm{orig\_cell}\, F^{T}$: each lattice vector is $F$ times the reference lattice vector, and a Cartesian position is $r_i = F q_i$.
The optimizer stores the reference positions $q_i$ and a $3 \times 3$ cell block.
The vector has length $3(N + 3)$, with $N$ the number of atoms: the $q_i$ first, then the cell block in row-major order.

-   `CellCoordinates::Deformation` stores `cell_factor * F`, ASE's `UnitCellFilter`. `cell_factor` defaults to $N$.
-   `CellCoordinates::Frechet` stores `exp_cell_factor * log(cell_factor * F)`, the principal logarithm of that block. `exp_cell_factor` defaults to $N$. At the reference cell $F = I$, so the block is `exp_cell_factor * log(cell_factor) * I`. With both factors equal to $N$ the block is $N \ln N$ times the identity.

The formulas are those of ASE 3.23.0 `UnitCellFilter` and `FrechetCellFilter` ([ase/filters.py](https://gitlab.com/ase/ase/-/blob/3.23.0/ase/filters.py)).
ASE 3.29 and 3.30 produce the same coordinates and the same generalized force on the unmasked, non-hydrostatic map when `cell_factor` is the atom count.
ASE UnitCellFilter multiplies the virial by the mask after the $F^{-T}$ pullback, as this filter does on both maps, while FrechetCellFilter multiplies it before, so masked Frechet forces differ from ASE.
`coordinates` returns the unprojected vector.
`geometry` and every objective evaluation apply the mask.


## Enthalpy and gradient

The value is the enthalpy $E + P V$, with $V = \det(\mathrm{cell})$.
$P$ is an external scalar pressure in energy per volume and defaults to zero.
The gradient is the derivative of that enthalpy with respect to the stored vector.
It is the negative of the generalized force ASE's filter returns from `get_forces` on the unmasked, non-hydrostatic map.
ASE UnitCellFilter multiplies the virial by the mask after the $F^{-T}$ pullback, as this filter does on both maps, while FrechetCellFilter multiplies it before, so masked Frechet forces differ from ASE.

The virial is $W = -V (\mathrm{stress} + P I)$.
For an unmasked filter the atom block of the gradient is $-F^{T} f$, with $f$ the Cartesian forces, and the deformation-map cell block is $-(W F^{-T}) / \mathrm{cell\_factor}$.
On the Frechet map that cell force is pulled back through $F = \exp(U / \mathrm{exp\_cell\_factor}) / \mathrm{cell\_factor}$, where $U$ is the stored block, by the Frechet derivative of the matrix exponential.

A central difference of the enthalpy reproduces this gradient.
`coordinates` on the Frechet map returns `Logarithm` when the current cell has no real logarithm.
A reflected cell is that error, and the message is `the deformation gradient has no principal logarithm`.
A half-turn has a real logarithm, so `coordinates` accepts a cell that `geometry` produced from that deformation.
`geometry` returns `Dim` for a vector of the wrong length, `NonFinite` for a non-finite entry, and `Decode` when the vector does not become a cell of positive volume.
The objective value and the gradient are NaN for those vectors, and when the sample is not finite.
A solver that refuses non-finite numbers stops.


## Mask and hydrostatic strain

`with_mask` takes six booleans in the Voigt order `xx, yy, zz, yz, xz, xy`, which is ASE's strain-mask order.
`false` holds that entry of $F$ at the identity.
The cell is $\mathrm{orig\_cell}\, F^{T}$, so a free entry of $F$ can move a masked Cartesian entry when the reference cell is skewed.
`true` leaves it free.
The default is all free.

`with_hydrostatic_strain(true)` replaces $F$ by $(\mathrm{tr}\, F / 3) I$ before the mask, so the cell stays a scalar multiple of the reference cell.
The mask can still pin a diagonal entry of that isotropic deformation.

The gradient is the derivative of the enthalpy of the projected geometry.
A masked entry of $F$ stays at the identity, and a finite difference of the value matches the gradient.
ASE 3.23 zeroes the corresponding virial entries and writes the proposed cell unchanged.
This wrapper projects $F$.
It does not hold the matching Cartesian entries of a skewed cell.


## Units

Positions, the reference positions $q_i$, and the cell entries share the length unit of the callback.
$F$, `cell_factor`, and `exp_cell_factor` are dimensionless.
Energy is the callback's energy.
Stress and $P$ are energy per volume.
The atom gradient is energy per length.
The cell-block gradient is energy, because the cell block is dimensionless.
The reported bounds are $\pm 10^{20}$ in these coordinates, wide enough that a real cell step stays inside them.


## Calling it

```rust
use rgmin::{
    CellFilter, CellSample, Control, LineSearch, Method, minimize_method,
};

let filter = CellFilter::frechet(natoms, cell, sample)
    .expect("reference cell")
    .with_scalar_pressure(pressure)
    .expect("pressure");
let start = filter.coordinates(&positions, &cell).expect("coordinates");
let report = minimize_method(
    &filter,
    start,
    &Control::default(),
    Method::lbfgs(),
    LineSearch::wolfe(1.0e-4, 0.9, 20),
)
.expect("minimize");
let (positions, cell) = filter.geometry(report.coords.view()).expect("geometry");
```

`sample` is `Fn(&[f64], &[f64; 9]) -> CellSample + Send + Sync`.
`CellFilter::deformation` selects the `UnitCellFilter` block.
`with_cell_factor` and `with_exp_cell_factor` replace the two positive scales.
`evaluations` counts one sample for the value, the gradient, or the fused pair, and `reset_evaluations` sets that counter back to zero.


## Errors

`CellFilterError` is `#[non_exhaustive]`.

-   `Empty`: `natoms` is zero.
-   `SingularCell`: the reference cell is non-finite or has no inverse, or the current cell has no inverse.
-   `Orientation`: the reference cell has a negative determinant.
-   `Factor`: `cell_factor` or `exp_cell_factor` is not finite and positive.
-   `Pressure`: the external pressure is not finite.
-   `Dim`: a position vector or a generalized coordinate vector has the wrong length.
-   `NonFinite`: a position, cell, or coordinate entry is NaN or infinite.
-   `Decode`: the generalized vector does not decode to a cell of positive volume. A left-handed deformation takes this path.
-   `Logarithm`: `log(cell_factor * F)` is not a real logarithm. A reflected cell takes this path. A half-turn has a real logarithm and is not this error.
