

Variable-cell coordinate
------------------------

A periodic relaxation has two kinds of coordinate. The atoms move
inside the cell, and the cell itself stretches. Those coordinates
do not share a unit. A strain is a pure number. An atom coordinate
is a length. A stepper that caps the longest atom move, or a FIRE
step that treats every entry of the vector alike, needs them on
one scale. The solid-state dimer that searches a shear-sensitive
barrier has to use the same scale, or the minimizer and the dimer
are not in the same space.

The chart this crate uses
~~~~~~~~~~~~~~~~~~~~~~~~~

Sheppard, Xiao, and Henkelman, J. Chem. Phys. 136, 074103 (2012)
(`10.1063/1.3684549 <https://doi.org/10.1063/1.3684549>`_) put the atoms and the cell in one vector for
the generalized solid-state nudged elastic band. The cell part is
a strain, multiplied by the Jacobian

::

    J = V^(1/3) * N^(1/6)

:math:`V` is a cell volume and :math:`N` is the number of atoms. :math:`J` is a
length. A step of 0.2 in a scaled strain is a strain of :math:`0.2/J`,
the same cap the eOn L-BFGS step and ASE's ``maxstep`` use for an
atom.

This crate fixes :math:`J` and the reference cell at the start of the
relaxation. The deformation is :math:`F = I + \varepsilon`, measured
from that cell. Lattice vectors are rows of :math:`H`. With :math:`q` the
atomic coordinates the stepper stores,

::

    r = F q
    H = H_0 F^T

At zero strain, :math:`q` is the Cartesian position. The stepper's
vector is :math:`q`, then :math:`J` times each free component of
:math:`\varepsilon`.

The potential returns the virial
:math:`W_{ij} = \partial E/\partial\delta_{ij}` for an incremental
strain at fixed fractional coordinates, and the stress
:math:`\sigma = W/V`. The enthalpy is

::

    Phi = E + P V - V_ref * sigma_ext : epsilon

:math:`P` is the external pressure, positive when it compresses.
:math:`\sigma_{\mathrm{ext}}` is a dead-load stress. At the reference
cell, :math:`F = I` and :math:`V = V_{\mathrm{ref}}`, the derivative with
respect to the scaled strain is

::

    d Phi / d (J epsilon) = (V / J) (sigma + P I - sigma_ext)

The generalized force in the 2012 paper is the negative of that
derivative,

::

    -(V / J) (sigma - sigma_applied)

with :math:`\sigma_{\mathrm{applied}} = \sigma_{\mathrm{ext}} - P I`.
Away from the reference the derivative keeps the chain rule
through :math:`F`: :math:`\partial E/\partial F = W F^{-\mathsf T}`, and
:math:`\partial(PV)/\partial\varepsilon = P V F^{-\mathsf T}`. A
finite difference of :math:`\Phi` along the chart matches that
gradient. Dropping the :math:`F^{-\mathsf T}` factor matches the
published force only at the reference cell.

The elastic dipole stored on the minimum is :math:`P_{ij} = V\sigma_{ij}
= W_{ij}`. That is :math:`\partial E/\partial\delta_{ij}`.

Why not the Fréchet cell filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ASE's ``UnitCellFilter`` uses the same deformation gradient. Tadmor,
Smith, Bernstein, and Kaxiras, Phys. Rev. B 59, 235 (1999)
(`10.1103/PhysRevB.59.235 <https://doi.org/10.1103/PhysRevB.59.235>`_) is the reference ASE cites for that
coordinate. The cell block of the filter is ``cell_factor * F``. The
default ``cell_factor`` is the number of atoms, a pure number, so
the cell entries are not lengths. The force on :math:`F` is the virial
pushed through :math:`F^{-\mathsf T}`, divided by ``cell_factor``. At
:math:`F = I` that is the 2012 force with :math:`J` replaced by
``cell_factor``.

``FrechetCellFilter``, added in ASE 3.23, stores
:math:`U = \log F` instead of :math:`F`. The cell force is the Fréchet
derivative of the matrix exponential applied to the virial
(Al-Mohy and Higham, SIAM J. Matrix Anal. Appl. 30, 1639 (2009),
`10.1137/080716426 <https://doi.org/10.1137/080716426>`_). Shinohara's note
(`cell\ :sub:`grad.pdf`\ <https://github.com/lan496/lan496.github.io/blob/main/notes/cell_grad.pdf>`_)
writes the derivative ASE implements. The exponential keeps the
cell in :math:`\mathrm{GL}^+(3)`, and the force is the true gradient
at large strain. ASE deprecates the older ``ExpCellFilter``, whose
cell force was not that derivative, and the ``UnitCellFilter``
notes say the Fréchet filter will probably perform better.

At :math:`U = 0` the Fréchet derivative is the identity, so the two
forces agree. A vacancy or a :math:`\langle 100\rangle` split
interstitial in copper sits close to a reference cell that is
already about right. The strain is small. The correction that
makes the Fréchet force the better large-strain minimizer is
second order there.

The other half of this minimizer is the solid-state dimer.
Xiao, Sheppard, Rogal, and Henkelman, J. Chem. Phys. 140, 174104
(2014) (`10.1063/1.4873437 <https://doi.org/10.1063/1.4873437>`_) search a solid-solid saddle in the
2012 coordinate, not in :math:`\log F`. A barrier that moves under
shear is a saddle in that vector. The minimizer that polishes the
minima on either side has to be the same vector, with the same
Jacobian, or the dimer tangent and the relaxation step are not
comparable. That is the chart ``relax_cell`` uses. The Fréchet
filter remains the better pure minimizer when the cell change is
large and no dimer will read the coordinate.

What the mask removes
~~~~~~~~~~~~~~~~~~~~~

A frozen mask stores no cell coordinate. The objective is the
potential energy and the Cartesian force, and the stepper is the
fixed-cell one. Hydrostatic strain keeps a single :math:`s` with
:math:`\varepsilon = s I`. A component mask drops the frozen entries
of :math:`\varepsilon` from the vector. A lattice-component mask is
that component mask transposed, and only when the reference cell
is diagonal, so each flag is one entry of :math:`H`.
