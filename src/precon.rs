//! Geometry preconditioners for atomistic descent.
//!
//! [`Exp`] is the universal exponential metric. [`Lindh`] is the model
//! Hessian of Lindh et al. 1995. [`Ff`] is the bonded reduced Hessian
//! of Makri, Ortner, and Kermode 2019. All three are sparse symmetric
//! positive definite matrices, solved by the same Jacobi-preconditioned
//! conjugate gradients, and held between rebuilds by [`needs_rebuild`].
//! [`select_precon`] documents which one applies.
//!
//! Pair assembly is a cell list. Occupied bins are visited in
//! lexicographic order and the pairs are sorted by atom index. A
//! matrix-vector product writes each row from that row's own sum. Every
//! dot product in the solve folds fixed chunks of 1024 in chunk order,
//! starting from `0.0`, in every build and at every length. The `par`
//! feature only decides whether those chunks run concurrently. Each
//! kernel has its own atom-count gate, and the gate does not read the
//! thread count.
//!
//! Packwood, Kermode, Mones, Bernstein, Woolley, Gould, Ortner, and
//! Csányi, *A universal preconditioner for simulating condensed phase
//! materials*, J. Chem. Phys. 144, 164109 (2016),
//! <https://doi.org/10.1063/1.4947024>.
//!
//! The caller supplies positions and a neighbour list (linkcell, vesin,
//! or any other producer). The matrix is the stabilized form of the
//! paper's equation (16): neighbour weight
//! `c_ij = exp(-A (r_ij / r_nn - 1))`, off-diagonal blocks `-μ c_ij I_3`
//! (C order, xyz per atom), and diagonal
//! `μ (C_stab + sum_j c_ij) I_3`. `r_nn` is the maximum of the per-atom
//! nearest-neighbour distances. `μ` is equation (14), the Rayleigh
//! quotient of the sine probe (15) against this matrix at `μ = 1`.
//!
//! `P` is symmetric positive definite for `μ > 0` and `C_stab > 0`.
//! The linear solve is Jacobi-preconditioned conjugate gradients.

mod ff;
mod geom;
mod lindh;
mod select;

pub(crate) use ff::FfState;
pub use ff::{AngleTerm, BondTerm, DihedralTerm, Ff, FfParams, FfTerms, MorseTerm, bonded_terms};
pub(crate) use lindh::LindhState;
pub use lindh::{Ghost, Lindh, LindhParams};
pub use select::{GeometryKind, N_LINDH, OptimizerFamily, PreconChoice, select_precon};

/// One geometry metric for [`crate::Solver::set_precon`].
///
/// Morse bonds and harmonic torsions travel in [`PreconSpec::Ff`].
/// The C structure carries harmonic bonds, harmonic angles, and the
/// Lindh torsion flag. Morse and FF torsions stay on this enum.
pub enum PreconSpec<'a> {
    /// Packwood exponential metric.
    Exp {
        /// Coordinates, length `3 N`, C order.
        positions: ArrayView1<'a, f64>,
        /// Caller neighbour list. Indices lie in `0..N`.
        neighbours: &'a Neighbours,
        /// `μ` `None` estimates the scale on the next step.
        params: &'a ExpParams,
    },
    /// Lindh model Hessian.
    Lindh {
        /// Coordinates, length `3 N`, C order.
        positions: ArrayView1<'a, f64>,
        /// One atomic number per free atom.
        z: &'a [i32],
        /// Frozen atoms. Their indices are not free-atom indices.
        ghosts: &'a [Ghost],
        /// `μ` `None` estimates the scale on the next step.
        params: &'a LindhParams,
    },
    /// Bonded metric. Unit constants are a geometric metric. The
    /// Makri field is this same matrix with that paper's curvatures.
    Ff {
        /// Coordinates, length `3 N`, C order.
        positions: ArrayView1<'a, f64>,
        /// Harmonic bonds and angles, Morse bonds, harmonic torsions.
        terms: FfTerms<'a>,
        /// `μ` `None` estimates the scale on the next step.
        params: &'a FfParams,
    },
}

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::{Error, Result};

/// One undirected neighbour list over atoms, plus bonds to atoms that
/// are not degrees of freedom.
///
/// Pairs are atom indices into a length-`3 N` coordinate vector in C
/// order. A list that stores both `(i, j)` and `(j, i)` is stored once.
/// An empty `r` means Cartesian distances from the positions passed to
/// [`Exp::assemble`]. `anchor` is a bond from a free atom to an atom
/// with no coordinate in the vector: it contributes to that atom's
/// diagonal only.
#[derive(Clone, Debug, Default)]
pub struct Neighbours {
    /// First atom of each pair.
    pub i: Vec<usize>,
    /// Second atom of each pair.
    pub j: Vec<usize>,
    /// Pair distances. Empty selects Cartesian distances.
    pub r: Vec<f64>,
    /// `(atom, distance)` bonds to atoms outside the coordinate vector.
    pub anchor: Vec<(usize, f64)>,
}

/// Parameters of the exponential preconditioner.
///
/// Defaults are the paper's generic choice: `A = 3`, `C_stab = 0.1`,
/// `μ` and `r_nn` computed from the geometry, and no extra cutoff
/// beyond the supplied list.
#[derive(Clone, Debug)]
pub struct ExpParams {
    /// Exponent `A` in `exp(-A (r / r_nn - 1))`.
    pub a: f64,
    /// Diagonal stabilization `C_stab`. Must be positive.
    pub c_stab: f64,
    /// Energy scale `μ`. `None` estimates it from the objective.
    pub mu: Option<f64>,
    /// Nearest-neighbour distance. `None` uses [`nearest_neighbour_distance`].
    pub r_nn: Option<f64>,
    /// Drop pairs with `r >= r_cut`. `None` keeps every supplied pair.
    pub r_cut: Option<f64>,
    /// Reestimate `μ` whenever the matrix is rebuilt.
    pub recalc_mu: bool,
    /// Lattice lengths `(Lx, Ly, Lz)` for the sine probe.
    /// `None` uses the span of the supplied positions. A nonpositive
    /// component falls back to that span.
    pub lengths: Option<[f64; 3]>,
}

impl Default for ExpParams {
    fn default() -> Self {
        Self {
            a: 3.0,
            c_stab: 0.1,
            mu: None,
            r_nn: None,
            r_cut: None,
            recalc_mu: false,
            lengths: None,
        }
    }
}

#[derive(Clone, Debug)]
struct Csr {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    val: Vec<f64>,
    diag: Vec<f64>,
    /// Atom count at which `apply` may use the `par` pool.
    apply_gate: usize,
    /// Atom count at which the solve's products may use the `par` pool.
    solve_gate: usize,
}

/// Sparse SPD metric. `apply` is `P v`. `solve` is `P x = rhs`.
#[derive(Clone, Debug)]
pub(crate) struct Metric {
    csr: Arc<Csr>,
}

impl Metric {
    fn from_csr(csr: Csr) -> Self {
        Self { csr: Arc::new(csr) }
    }

    pub(crate) fn dim(&self) -> usize {
        self.csr.n
    }

    pub(crate) fn apply(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let mut y = vec![0.0; self.csr.n];
        if v.len() == self.csr.n {
            matvec(
                &self.csr,
                v.as_slice().unwrap_or(&v.to_vec()),
                &mut y,
                self.csr.apply_gate,
            );
        }
        Array1::from_vec(y)
    }

    pub(crate) fn solve(&self, rhs: ArrayView1<f64>) -> Option<Array1<f64>> {
        self.solve_counted(rhs).map(|(x, _)| x)
    }

    pub(crate) fn solve_counted(&self, rhs: ArrayView1<f64>) -> Option<(Array1<f64>, usize)> {
        if rhs.len() != self.csr.n {
            return None;
        }
        let rhs_vec = rhs
            .as_slice()
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| rhs.to_vec());
        solve_pcg(&self.csr, &rhs_vec).map(|(x, iters)| (Array1::from_vec(x), iters))
    }

    pub(crate) fn to_dense(&self) -> Array2<f64> {
        let n = self.csr.n;
        let mut a = Array2::zeros((n, n));
        for i in 0..n {
            for k in self.csr.row_ptr[i]..self.csr.row_ptr[i + 1] {
                a[(i, self.csr.col_idx[k])] = self.csr.val[k];
            }
        }
        a
    }

    /// `μ P`. Lindh and FF assemble the model at scale 1 and apply the
    /// estimated energy scale here, the same place Exp folds `μ` into
    /// the matrix.
    pub(crate) fn scaled(&self, mu: f64) -> Self {
        let mut csr = (*self.csr).clone();
        for v in &mut csr.val {
            *v *= mu;
        }
        for d in &mut csr.diag {
            *d *= mu;
        }
        Self::from_csr(csr)
    }
}

/// Assembled exponential preconditioner `P`.
#[derive(Clone, Debug)]
pub struct Exp {
    mu: f64,
    r_nn: f64,
    metric: Metric,
}

/// Session copy of the neighbour list and the current factor.
#[derive(Clone, Debug)]
pub(crate) struct ExpState {
    neighbours: Neighbours,
    params: ExpParams,
    mu: Option<f64>,
    r_nn: f64,
    matrix: Option<Exp>,
    /// `r_nn` is filled from the first real geometry.
    r_nn_pending: bool,
    /// Positions of the last assembly. The matrix is held while the
    /// largest atomic move stays within `r_nn / 2`.
    built_at: Option<Array1<f64>>,
}

impl Exp {
    /// Build `P` at a fixed `μ`.
    ///
    /// `positions` has length `3 N`. Pair indices lie in `0..N`.
    pub fn assemble(
        positions: ArrayView1<f64>,
        neighbours: &Neighbours,
        params: &ExpParams,
        mu: f64,
        r_nn: f64,
    ) -> Result<Self> {
        if !(mu.is_finite() && mu > 0.0) {
            return Err(Error::Precon {
                what: "mu is not positive",
            });
        }
        if !(r_nn.is_finite() && r_nn > 0.0) {
            return Err(Error::Precon {
                what: "r_nn is not positive",
            });
        }
        if !(params.c_stab.is_finite() && params.c_stab > 0.0) {
            return Err(Error::Precon {
                what: "c_stab is not positive",
            });
        }
        if !params.a.is_finite() {
            return Err(Error::Precon {
                what: "A is not finite",
            });
        }
        let natoms = atom_count(positions)?;
        let (pairs, anchors) = bonds(natoms, positions, neighbours, params.r_cut)?;
        let csr = build_csr(natoms, &pairs, &anchors, mu, params.a, r_nn, params.c_stab);
        Ok(Self {
            mu,
            r_nn,
            metric: Metric::from_csr(csr),
        })
    }

    pub(crate) fn metric(&self) -> &Metric {
        &self.metric
    }

    /// Diagonal SPD matrix. Jacobi PCG inverts it in one iteration.
    #[cfg(test)]
    pub(crate) fn diagonal(diag: &[f64]) -> Self {
        let n = diag.len();
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::with_capacity(n);
        let mut val = Vec::with_capacity(n);
        row_ptr.push(0);
        for (i, &d) in diag.iter().enumerate() {
            col_idx.push(i);
            val.push(d);
            row_ptr.push(i + 1);
        }
        Self {
            mu: 1.0,
            r_nn: 1.0e9,
            metric: Metric::from_csr(Csr {
                n,
                row_ptr,
                col_idx,
                val,
                diag: diag.to_vec(),
                apply_gate: N_PAR_EXP_APPLY,
                solve_gate: N_PAR_EXP_SOLVE,
            }),
        }
    }

    /// Energy scale used to build this matrix.
    #[must_use]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Nearest-neighbour distance used to build this matrix.
    #[must_use]
    pub fn r_nn(&self) -> f64 {
        self.r_nn
    }

    /// Matrix order, `3 N`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.metric.dim()
    }

    /// `P v`.
    #[must_use]
    pub fn apply(&self, v: ArrayView1<f64>) -> Array1<f64> {
        self.metric.apply(v)
    }

    /// Solve `P x = rhs`. `None` when the iterate is not a descent-quality
    /// solution (the matrix is not SPD on the Krylov subspace, or the
    /// residual stalls).
    #[must_use]
    pub fn solve(&self, rhs: ArrayView1<f64>) -> Option<Array1<f64>> {
        self.metric.solve(rhs)
    }

    /// Jacobi PCG iterations for one solve. The count is the number of
    /// matrix-vector products.
    #[must_use]
    pub fn solve_iterations(&self, rhs: ArrayView1<f64>) -> Option<usize> {
        self.metric.solve_counted(rhs).map(|(_, n)| n)
    }

    /// Dense copy, for tests and for a dense Cholesky reference.
    #[must_use]
    pub fn to_dense(&self) -> Array2<f64> {
        self.metric.to_dense()
    }
}

impl ExpState {
    pub(crate) fn new(
        positions: ArrayView1<f64>,
        neighbours: &Neighbours,
        params: &ExpParams,
    ) -> Result<Self> {
        let natoms = atom_count(positions)?;
        let _ = natoms;
        if neighbours.i.len() != neighbours.j.len() {
            return Err(Error::Precon {
                what: "neighbour index arrays differ in length",
            });
        }
        if !neighbours.r.is_empty() && neighbours.r.len() != neighbours.i.len() {
            return Err(Error::Precon {
                what: "neighbour distances do not match the pair list",
            });
        }
        let r_nn = resolve_r_nn(positions, neighbours, params)?;
        let mut state = Self {
            neighbours: neighbours.clone(),
            params: params.clone(),
            mu: None,
            r_nn,
            matrix: None,
            r_nn_pending: false,
            built_at: None,
        };
        if let Some(mu) = params.mu {
            state.matrix = Some(Exp::assemble(positions, neighbours, params, mu, r_nn)?);
            state.mu = Some(state.matrix.as_ref().unwrap().mu());
            state.built_at = Some(positions.to_owned());
        }
        Ok(state)
    }

    /// Neighbour list without a geometry. The next refresh places `r_nn`
    /// and builds `P`. Used by the C ABI, whose coordinates arrive with
    /// the step.
    #[cfg(feature = "capi")]
    pub(crate) fn deferred(neighbours: Neighbours, params: ExpParams) -> Result<Self> {
        if neighbours.i.len() != neighbours.j.len() {
            return Err(Error::Precon {
                what: "neighbour index arrays differ in length",
            });
        }
        if !neighbours.r.is_empty() && neighbours.r.len() != neighbours.i.len() {
            return Err(Error::Precon {
                what: "neighbour distances do not match the pair list",
            });
        }
        if !(params.a.is_finite() && params.c_stab.is_finite() && params.c_stab > 0.0) {
            return Err(Error::Precon {
                what: "A or c_stab is not usable",
            });
        }
        if let Some(mu) = params.mu {
            if !(mu.is_finite() && mu > 0.0) {
                return Err(Error::Precon {
                    what: "mu is not positive",
                });
            }
        }
        let r_nn_pending = params.r_nn.is_none();
        let r_nn = params.r_nn.unwrap_or(1.0);
        if !(r_nn.is_finite() && r_nn > 0.0) {
            return Err(Error::Precon {
                what: "r_nn is not positive",
            });
        }
        let mu = params.mu;
        Ok(Self {
            neighbours,
            params,
            mu,
            r_nn,
            matrix: None,
            r_nn_pending,
            built_at: None,
        })
    }

    pub(crate) fn mu(&self) -> Option<f64> {
        self.mu
    }

    pub(crate) fn matrix(&self) -> Option<&Exp> {
        self.matrix.as_ref()
    }

    /// Rebuild `P` at `positions` when the hold has lapsed.
    ///
    /// Estimates `μ` when it is unset or [`ExpParams::recalc_mu`] is set.
    /// `grad_here` is `∇E(positions)`. `Ok(true)` means the matrix was
    /// replaced. `Ok(false)` means the previous matrix is still in use.
    pub(crate) fn refresh<G>(
        &mut self,
        positions: ArrayView1<f64>,
        grad_here: &Array1<f64>,
        grad: G,
    ) -> Result<bool>
    where
        G: Fn(ArrayView1<f64>) -> Array1<f64>,
    {
        if self.r_nn_pending
            && let Some(r) = nearest_neighbour_distance(positions, &self.neighbours)
        {
            self.r_nn = r;
            self.r_nn_pending = false;
        }
        if self.hold(positions) {
            return Ok(false);
        }
        let need_mu = self.mu.is_none() || self.params.recalc_mu;
        if need_mu {
            match estimate_mu_with(
                positions,
                &self.neighbours,
                &self.params,
                self.r_nn,
                Some(grad_here),
                grad,
            ) {
                Ok(mu) => self.mu = Some(mu),
                Err(_) => {
                    if self.mu.is_none() {
                        self.mu = Some(1.0);
                    }
                }
            }
        }
        let mu = self.mu.unwrap_or(1.0);
        self.matrix = Some(Exp::assemble(
            positions,
            &self.neighbours,
            &self.params,
            mu,
            self.r_nn,
        )?);
        self.built_at = Some(positions.to_owned());
        Ok(true)
    }

    fn hold(&self, positions: ArrayView1<f64>) -> bool {
        if self.r_nn_pending || self.matrix.is_none() || self.mu.is_none() || self.params.recalc_mu
        {
            return false;
        }
        self.built_at.as_ref().is_some_and(|prev| {
            !needs_rebuild(max_atom_displacement(prev.view(), positions), self.r_nn)
        })
    }

    /// A diagonal metric that refresh will not replace. For the
    /// preconditioned-CG unit test.
    #[cfg(test)]
    pub(crate) fn holding_diagonal(diag: &[f64], positions: ArrayView1<f64>) -> Self {
        Self {
            neighbours: Neighbours::default(),
            params: ExpParams {
                mu: Some(1.0),
                r_nn: Some(1.0e9),
                recalc_mu: false,
                ..ExpParams::default()
            },
            mu: Some(1.0),
            r_nn: 1.0e9,
            matrix: Some(Exp::diagonal(diag)),
            r_nn_pending: false,
            built_at: Some(positions.to_owned()),
        }
    }
}

/// Largest per-atom displacement between two `3 N` coordinate vectors.
///
/// A length mismatch returns infinity, so [`needs_rebuild`] treats it
/// as a stale matrix.
#[must_use]
pub fn max_atom_displacement(previous: ArrayView1<f64>, current: ArrayView1<f64>) -> f64 {
    if previous.len() != current.len() || !previous.len().is_multiple_of(3) {
        return f64::INFINITY;
    }
    let mut best = 0.0_f64;
    for i in (0..previous.len()).step_by(3) {
        let mut s = 0.0;
        for k in 0..3 {
            let d = previous[i + k] - current[i + k];
            s += d * d;
        }
        best = best.max(s.sqrt());
    }
    best
}

/// Whether `P` should be reassembled.
///
/// Packwood et al. 2016 rebuild the exponential metric when the largest
/// atomic move since the last assembly exceeds half the nearest-neighbour
/// distance. Lindh and FF use the same rule. `scale` is that distance.
/// A nonpositive or nonfinite scale, or a nonfinite displacement, rebuilds.
#[must_use]
pub fn needs_rebuild(max_atom_displacement: f64, scale: f64) -> bool {
    if !(scale.is_finite() && scale > 0.0 && max_atom_displacement.is_finite()) {
        return true;
    }
    max_atom_displacement > 0.5 * scale
}

/// Installed geometry metric. One slot, replaced by the next setter.
pub(crate) enum Slot {
    Exp(ExpState),
    Lindh(lindh::LindhState),
    Ff(ff::FfState),
}

impl Slot {
    pub(crate) fn metric(&self) -> Option<&Metric> {
        match self {
            Self::Exp(state) => state.matrix().map(Exp::metric),
            Self::Lindh(state) => state.metric(),
            Self::Ff(state) => state.metric(),
        }
    }

    /// `Ok(true)` when the installed matrix was replaced.
    pub(crate) fn refresh<G>(
        &mut self,
        positions: ArrayView1<f64>,
        grad_here: &Array1<f64>,
        grad: G,
    ) -> Result<bool>
    where
        G: Fn(ArrayView1<f64>) -> Array1<f64>,
    {
        match self {
            Self::Exp(state) => state.refresh(positions, grad_here, grad),
            Self::Lindh(state) => state.refresh(positions, grad_here, grad),
            Self::Ff(state) => state.refresh(positions, grad_here, grad),
        }
    }

    pub(crate) fn scale(&self) -> Option<f64> {
        match self {
            Self::Exp(state) => state.mu(),
            Self::Lindh(state) => state.mu(),
            Self::Ff(state) => state.mu(),
        }
    }
}

/// Paper's `r_nn`: the maximum, over atoms that have a neighbour, of
/// that atom's nearest-neighbour distance.
///
/// Distances come from `neighbours.r` when it is populated, otherwise
/// from Cartesian positions. Anchor bonds count.
#[must_use]
pub fn nearest_neighbour_distance(
    positions: ArrayView1<f64>,
    neighbours: &Neighbours,
) -> Option<f64> {
    let natoms = positions.len() / 3;
    if natoms == 0 || positions.len() != 3 * natoms {
        return None;
    }
    let mut min_d = vec![f64::INFINITY; natoms];
    let cartesian = neighbours.r.is_empty();
    for k in 0..neighbours.i.len() {
        let i = neighbours.i[k];
        let j = neighbours.j[k];
        if i >= natoms || j >= natoms || i == j {
            continue;
        }
        let r = if cartesian {
            cartesian_distance(positions, i, j)
        } else {
            neighbours.r.get(k).copied().unwrap_or(f64::NAN)
        };
        if r.is_finite() && r > 0.0 {
            min_d[i] = min_d[i].min(r);
            min_d[j] = min_d[j].min(r);
        }
    }
    for &(atom, r) in &neighbours.anchor {
        if atom < natoms && r.is_finite() && r > 0.0 {
            min_d[atom] = min_d[atom].min(r);
        }
    }
    let mut acc = 0.0_f64;
    let mut any = false;
    for d in min_d {
        if d.is_finite() {
            acc = acc.max(d);
            any = true;
        }
    }
    any.then_some(acc)
}

/// Sine probe `v` of equation (15).
///
/// `v_i = M (sin(x_i/Lx), sin(y_i/Ly), sin(z_i/Lz))` with
/// `M = 10^{-2} r_nn I`. A vanishing length zeroes that column of `M`.
#[must_use]
pub fn sine_probe(positions: ArrayView1<f64>, r_nn: f64, lengths: Option<[f64; 3]>) -> Array1<f64> {
    let n = positions.len();
    let mut v = Array1::zeros(n);
    if !n.is_multiple_of(3) || !(r_nn.is_finite() && r_nn > 0.0) {
        return v;
    }
    let span = position_span(positions);
    let mut len = lengths.unwrap_or(span);
    for a in 0..3 {
        if !(len[a].is_finite() && len[a] > 0.0) {
            len[a] = span[a];
        }
    }
    let scale = 1e-2 * r_nn;
    let natoms = n / 3;
    for i in 0..natoms {
        for a in 0..3 {
            if !(len[a].is_finite() && len[a] > 0.0) {
                continue;
            }
            let s = (positions[3 * i + a] / len[a]).sin();
            v[3 * i + a] = scale * s;
        }
    }
    v
}

/// Equation (14): `μ = vᵀ (∇E(x+v) - ∇E(x)) / vᵀ P_{μ=1} v`.
///
/// `grad` is `∇E`. The denominator is this preconditioner at `μ = 1`,
/// including `C_stab`. A nonpositive or nonfinite ratio is an error;
/// the session then keeps the previous scale, or `1` if it has none.
pub fn estimate_mu<G>(
    positions: ArrayView1<f64>,
    neighbours: &Neighbours,
    params: &ExpParams,
    grad: G,
) -> Result<f64>
where
    G: Fn(ArrayView1<f64>) -> Array1<f64>,
{
    let r_nn = resolve_r_nn(positions, neighbours, params)?;
    estimate_mu_with(positions, neighbours, params, r_nn, None, grad)
}

fn estimate_mu_with<G>(
    positions: ArrayView1<f64>,
    neighbours: &Neighbours,
    params: &ExpParams,
    r_nn: f64,
    grad_here: Option<&Array1<f64>>,
    grad: G,
) -> Result<f64>
where
    G: Fn(ArrayView1<f64>) -> Array1<f64>,
{
    let p1 = Exp::assemble(positions, neighbours, params, 1.0, r_nn)?;
    let v = sine_probe(positions, r_nn, params.lengths);
    let g0 = match grad_here {
        Some(g) if g.len() == positions.len() => g.clone(),
        _ => grad(positions),
    };
    if g0.len() != positions.len() {
        return Err(Error::Precon {
            what: "gradient length does not match positions",
        });
    }
    let mut shifted = positions.to_owned();
    shifted += &v;
    let g1 = grad(shifted.view());
    if g1.len() != positions.len() {
        return Err(Error::Precon {
            what: "gradient length does not match positions",
        });
    }
    let mut num = 0.0;
    for i in 0..positions.len() {
        num += (g1[i] - g0[i]) * v[i];
    }
    let pv = p1.apply(v.view());
    let den = v.dot(&pv);
    if !(den.is_finite() && den.abs() > 1e-30 && num.is_finite()) {
        return Err(Error::Precon {
            what: "mu estimate denominator vanished",
        });
    }
    let mu = num / den;
    if !(mu.is_finite() && mu > 0.0) {
        return Err(Error::Precon {
            what: "mu estimate is not positive",
        });
    }
    Ok(mu)
}

/// `μ = vᵀ (∇E(x+v) - ∇E(x)) / vᵀ P v` for a model already at scale 1.
///
/// The probe is [`sine_probe`]. A nonpositive ratio is an error. The
/// session then keeps the previous scale, or `1` if it has none.
pub(crate) fn model_scale<G>(
    metric: &Metric,
    positions: ArrayView1<f64>,
    r_nn: f64,
    lengths: Option<[f64; 3]>,
    grad_here: &Array1<f64>,
    grad: G,
) -> Result<f64>
where
    G: Fn(ArrayView1<f64>) -> Array1<f64>,
{
    if grad_here.len() != positions.len() {
        return Err(Error::Precon {
            what: "gradient length does not match positions",
        });
    }
    let v = sine_probe(positions, r_nn, lengths);
    let mut shifted = positions.to_owned();
    shifted += &v;
    let g1 = grad(shifted.view());
    if g1.len() != positions.len() {
        return Err(Error::Precon {
            what: "gradient length does not match positions",
        });
    }
    let mut num = 0.0;
    for i in 0..positions.len() {
        num += (g1[i] - grad_here[i]) * v[i];
    }
    let den = v.dot(&metric.apply(v.view()));
    if !(den.is_finite() && den.abs() > 1e-30 && num.is_finite()) {
        return Err(Error::Precon {
            what: "mu estimate denominator vanished",
        });
    }
    let mu = num / den;
    if !(mu.is_finite() && mu > 0.0) {
        return Err(Error::Precon {
            what: "mu estimate is not positive",
        });
    }
    Ok(mu)
}

fn resolve_r_nn(
    positions: ArrayView1<f64>,
    neighbours: &Neighbours,
    params: &ExpParams,
) -> Result<f64> {
    if let Some(r) = params.r_nn {
        if !(r.is_finite() && r > 0.0) {
            return Err(Error::Precon {
                what: "r_nn is not positive",
            });
        }
        return Ok(r);
    }
    Ok(nearest_neighbour_distance(positions, neighbours).unwrap_or(1.0))
}

fn atom_count(positions: ArrayView1<f64>) -> Result<usize> {
    if !positions.len().is_multiple_of(3) || positions.is_empty() {
        return Err(Error::Precon {
            what: "positions length is not a positive multiple of 3",
        });
    }
    Ok(positions.len() / 3)
}

fn position_span(positions: ArrayView1<f64>) -> [f64; 3] {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let natoms = positions.len() / 3;
    for i in 0..natoms {
        for a in 0..3 {
            let c = positions[3 * i + a];
            lo[a] = lo[a].min(c);
            hi[a] = hi[a].max(c);
        }
    }
    [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
}

fn cartesian_distance(positions: ArrayView1<f64>, i: usize, j: usize) -> f64 {
    let mut s = 0.0;
    for a in 0..3 {
        let d = positions[3 * i + a] - positions[3 * j + a];
        s += d * d;
    }
    s.sqrt()
}

type BondRow = (usize, usize, f64);
type AnchorRow = (usize, f64);

fn bonds(
    natoms: usize,
    positions: ArrayView1<f64>,
    neighbours: &Neighbours,
    r_cut: Option<f64>,
) -> Result<(Vec<BondRow>, Vec<AnchorRow>)> {
    if neighbours.i.len() != neighbours.j.len() {
        return Err(Error::Precon {
            what: "neighbour index arrays differ in length",
        });
    }
    if !neighbours.r.is_empty() && neighbours.r.len() != neighbours.i.len() {
        return Err(Error::Precon {
            what: "neighbour distances do not match the pair list",
        });
    }
    let cartesian = neighbours.r.is_empty();
    let mut seen = std::collections::HashSet::new();
    let mut pairs = Vec::new();
    for k in 0..neighbours.i.len() {
        let i = neighbours.i[k];
        let j = neighbours.j[k];
        if i >= natoms || j >= natoms {
            return Err(Error::Precon {
                what: "neighbour index is out of range",
            });
        }
        if i == j {
            continue;
        }
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        if !seen.insert((a, b)) {
            continue;
        }
        let r = if cartesian {
            cartesian_distance(positions, a, b)
        } else {
            neighbours.r[k]
        };
        if !(r.is_finite() && r > 0.0) {
            continue;
        }
        if r_cut.is_some_and(|cut| r >= cut) {
            continue;
        }
        pairs.push((a, b, r));
    }
    let mut anchors = Vec::new();
    for &(atom, r) in &neighbours.anchor {
        if atom >= natoms {
            return Err(Error::Precon {
                what: "anchor index is out of range",
            });
        }
        if !(r.is_finite() && r > 0.0) {
            continue;
        }
        if r_cut.is_some_and(|cut| r >= cut) {
            continue;
        }
        anchors.push((atom, r));
    }
    Ok((pairs, anchors))
}

fn weight(mu: f64, a: f64, r: f64, r_nn: f64) -> f64 {
    mu * (-a * (r / r_nn - 1.0)).exp()
}

fn build_csr(
    natoms: usize,
    pairs: &[(usize, usize, f64)],
    anchors: &[(usize, f64)],
    mu: f64,
    a: f64,
    r_nn: f64,
    c_stab: f64,
) -> Csr {
    let n = 3 * natoms;
    let mut diag = vec![mu * c_stab; n];
    let mut trips: Vec<(usize, usize, f64)> = Vec::with_capacity(pairs.len() * 6 + n);
    for &(i, j, r) in pairs {
        let c = weight(mu, a, r, r_nn);
        for d in 0..3 {
            let ii = 3 * i + d;
            let jj = 3 * j + d;
            diag[ii] += c;
            diag[jj] += c;
            trips.push((ii, jj, -c));
            trips.push((jj, ii, -c));
        }
    }
    for &(atom, r) in anchors {
        let c = weight(mu, a, r, r_nn);
        for d in 0..3 {
            diag[3 * atom + d] += c;
        }
    }
    for i in 0..n {
        trips.push((i, i, diag[i]));
    }
    csr_from_coo(n, trips, N_PAR_EXP_APPLY, N_PAR_EXP_SOLVE)
}

fn csr_from_coo(
    n: usize,
    mut trips: Vec<(usize, usize, f64)>,
    apply_gate: usize,
    solve_gate: usize,
) -> Csr {
    trips.sort_unstable_by_key(|p| (p.0, p.1));
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(trips.len());
    let mut val = Vec::with_capacity(trips.len());
    let mut diag = vec![0.0; n];
    row_ptr.push(0);
    let mut cursor = 0;
    while cursor < trips.len() {
        let row = trips[cursor].0;
        while row_ptr.len() <= row {
            row_ptr.push(col_idx.len());
        }
        let col = trips[cursor].1;
        let mut acc = 0.0;
        while cursor < trips.len() && trips[cursor].0 == row && trips[cursor].1 == col {
            acc += trips[cursor].2;
            cursor += 1;
        }
        col_idx.push(col);
        val.push(acc);
        if row == col {
            diag[row] = acc;
        }
    }
    while row_ptr.len() <= n {
        row_ptr.push(col_idx.len());
    }
    Csr {
        n,
        row_ptr,
        col_idx,
        val,
        diag,
        apply_gate,
        solve_gate,
    }
}

/// Record which free atoms share a 3×3 block, then scatter into that
/// pattern. Memory is the stored nonzeros. Each row is summed from the
/// terms that touch it, in the order those terms are presented, so a
/// later parallel fill of the rows does not change the sum.
struct BlockAccum {
    n_atoms: usize,
    partners: Vec<Vec<usize>>,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    val: Vec<f64>,
}

impl BlockAccum {
    /// `partners[i]` lists other free atoms. `i` itself is added here.
    fn new(n_atoms: usize, mut partners: Vec<Vec<usize>>) -> Self {
        for (i, p) in partners.iter_mut().enumerate() {
            p.push(i);
            p.sort_unstable();
            p.dedup();
        }
        let mut row_ptr = Vec::with_capacity(3 * n_atoms + 1);
        let mut col_idx = Vec::new();
        row_ptr.push(0);
        for atom in 0..n_atoms {
            let cols: Vec<usize> = partners[atom]
                .iter()
                .flat_map(|&b| [3 * b, 3 * b + 1, 3 * b + 2])
                .collect();
            for _d in 0..3 {
                col_idx.extend_from_slice(&cols);
                row_ptr.push(col_idx.len());
            }
        }
        let nnz = col_idx.len();
        Self {
            n_atoms,
            partners,
            row_ptr,
            col_idx,
            val: vec![0.0; nnz],
        }
    }

    /// Add `scale * g g^T` on the free atoms. A `None` index is a ghost.
    fn add_rank1(&mut self, atoms: &[(Option<usize>, [f64; 3])], scale: f64) {
        if !(scale.is_finite() && scale != 0.0) {
            return;
        }
        let free: Vec<(usize, [f64; 3])> = atoms
            .iter()
            .copied()
            .filter_map(|(i, g)| i.map(|i| (i, g)))
            .collect();
        let partners = &self.partners;
        let row_ptr = &self.row_ptr;
        let val = &mut self.val;
        for &(ia, ga) in &free {
            let plist = &partners[ia];
            for d in 0..3 {
                let start = row_ptr[3 * ia + d];
                for &(ib, gb) in &free {
                    let p = plist
                        .binary_search(&ib)
                        .expect("rank-1 atom is missing from the sparsity pattern");
                    let base = start + 3 * p;
                    for e in 0..3 {
                        val[base + e] += scale * ga[d] * gb[e];
                    }
                }
            }
        }
    }

    /// Add `c` to every diagonal after the rank-1 terms.
    fn add_stabilization(&mut self, c: f64) {
        for atom in 0..self.n_atoms {
            let p = self.partners[atom]
                .binary_search(&atom)
                .expect("diagonal block");
            for d in 0..3 {
                let base = self.row_ptr[3 * atom + d] + 3 * p;
                self.val[base + d] += c;
            }
        }
    }

    fn finish(self, apply_gate: usize, solve_gate: usize) -> Csr {
        let n = 3 * self.n_atoms;
        let mut diag = vec![0.0; n];
        for i in 0..n {
            let s = self.row_ptr[i];
            let e = self.row_ptr[i + 1];
            if let Ok(k) = self.col_idx[s..e].binary_search(&i) {
                diag[i] = self.val[s + k];
            }
        }
        Csr {
            n,
            row_ptr: self.row_ptr,
            col_idx: self.col_idx,
            val: self.val,
            diag,
            apply_gate,
            solve_gate,
        }
    }
}

/// Push every free atom of a rank-1 term onto the partner lists.
fn note_block(partners: &mut [Vec<usize>], atoms: &[(Option<usize>, [f64; 3])], scale: f64) {
    if !(scale.is_finite() && scale != 0.0) {
        return;
    }
    let ids: Vec<usize> = atoms.iter().filter_map(|a| a.0).collect();
    for &i in &ids {
        for &j in &ids {
            partners[i].push(j);
        }
    }
}

/// Atom counts for the `par` loops. Each gate is an atom count.
/// A kernel stays serial below its gate in every build.
///
/// Measured on bulk diamond Si, 4 Å cutoff, release, five interleaved
/// runs, threads 1/2/4/8 (`tests/precon_scaling.rs`). A gate is the
/// smallest size in {160, 2_000, 8_000, 32_000, 100_000} where two
/// threads beat one by more than the wider run-to-run range.
/// `usize::MAX` means that bar was never cleared, so the kernel stays
/// serial. The gate does not read the thread count.
///
/// Lindh and bonded-metric matvec, eight applies.
/// 160 atoms: 1 thread 0.493 ms, 2 threads 0.361 ms, gap 0.132 ms,
/// range 0.344 ms, no.
/// 2,000 atoms: 1 thread 8.360 ms, 2 threads 4.751 ms, gap 3.609 ms,
/// range 1.052 ms, yes.
pub const N_PAR_LINDH_APPLY: usize = 2_000;
/// Lindh and bonded-metric Jacobi PCG, eight solves.
/// 160 atoms: 1 thread 31.355 ms, 2 threads 16.752 ms, gap 14.602 ms,
/// range 17.571 ms, no.
/// 2,000 atoms: 1 thread 0.914 s, 2 threads 0.467 s, gap 0.447 s,
/// range 0.035 s, yes.
pub const N_PAR_LINDH_SOLVE: usize = 2_000;
/// Exp matvec, eight applies.
/// 8,000 atoms: 1 thread 2.607 ms, 2 threads 1.939 ms, gap 0.668 ms,
/// range 0.961 ms, no.
/// 32,000 atoms: 1 thread 11.770 ms, 2 threads 7.745 ms, gap 4.025 ms,
/// range 3.247 ms, yes.
pub const N_PAR_EXP_APPLY: usize = 32_000;
/// Exp Jacobi PCG, eight solves.
/// 8,000 atoms: 1 thread 0.161 s, 2 threads 0.146 s, gap 0.015 s,
/// range 0.059 s, no.
/// 32,000 atoms: 1 thread 1.026 s, 2 threads 0.681 s, gap 0.345 s,
/// range 0.144 s, yes.
pub const N_PAR_EXP_SOLVE: usize = 32_000;
/// Pair-list assembly. The value fill is a serial scatter into the
/// stored nonzeros, so this gate covers only the cell-list search.
/// Two threads never beat one by more than the run-to-run range.
/// The closest row is Exp at 100,000 atoms: 1 thread 0.388 s,
/// 2 threads 0.362 s, gap 0.026 s, range 0.027 s, no.
/// Lindh at 100,000 atoms: 1 thread 6.339 s, 2 threads 6.401 s,
/// gap -0.062 s, range 0.709 s, no.
pub const N_PAR_ASSEMBLE: usize = usize::MAX;

/// Fixed length of one dot-product chunk. The same chunks are used
/// with and without the `par` feature.
const DOT_CHUNK: usize = 1024;

/// Map `0..n` and keep index order.
///
/// With the `par` feature and `atoms >= gate` the chunks run on the
/// current rayon pool. `collect` on an indexed iterator concatenates
/// in index order. Below the gate, and in a build without `par`, the
/// map is a serial loop. The gate is an atom count.
pub(super) fn ordered_map<T, F>(n: usize, atoms: usize, gate: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    let parallel = gate < usize::MAX && atoms >= gate && n >= 2;
    #[cfg(not(feature = "par"))]
    let _ = parallel;
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        if parallel {
            return (0..n).into_par_iter().map(f).collect();
        }
    }
    (0..n).map(f).collect()
}

fn row_dot(a: &Csr, x: &[f64], i: usize) -> f64 {
    let mut s = 0.0;
    for k in a.row_ptr[i]..a.row_ptr[i + 1] {
        s += a.val[k] * x[a.col_idx[k]];
    }
    s
}

fn matvec(a: &Csr, x: &[f64], y: &mut [f64], gate: usize) {
    let parallel = gate < usize::MAX && a.n / 3 >= gate && a.n >= 2;
    #[cfg(not(feature = "par"))]
    let _ = parallel;
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        // Each row is an independent product. Writing `y[i]` from that
        // row does not reorder any sum, so the vector matches the
        // serial loop at every thread count.
        if parallel {
            y.par_iter_mut().enumerate().for_each(|(i, yi)| {
                *yi = row_dot(a, x, i);
            });
            return;
        }
    }
    for i in 0..a.n {
        y[i] = row_dot(a, x, i);
    }
}

/// Dot product. Chunks of [`DOT_CHUNK`] are summed left to right from
/// `0.0` in every build. The `par` feature evaluates the chunks
/// concurrently when `atoms >= gate`; the fold of the partials stays
/// in chunk order.
fn dot(a: &[f64], b: &[f64], gate: usize) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let nchunks = a.len().div_ceil(DOT_CHUNK);
    let atoms = a.len() / 3;
    let partials = ordered_map(nchunks, atoms, gate, |c| {
        let start = c * DOT_CHUNK;
        let end = (start + DOT_CHUNK).min(a.len());
        let mut acc = 0.0;
        for i in start..end {
            acc += a[i] * b[i];
        }
        acc
    });
    partials.into_iter().fold(0.0, |acc, v| acc + v)
}

fn nrm2(a: &[f64], gate: usize) -> f64 {
    dot(a, a, gate).sqrt()
}

fn solve_pcg(a: &Csr, b: &[f64]) -> Option<(Vec<f64>, usize)> {
    let n = a.n;
    let gate = a.solve_gate;
    if b.len() != n {
        return None;
    }
    let bnorm = nrm2(b, gate);
    if bnorm == 0.0 {
        return Some((vec![0.0; n], 0));
    }
    if a.diag.iter().any(|d| !(d.is_finite() && *d > 0.0)) {
        return None;
    }
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];
    jacobi(a, &r, &mut z);
    p.copy_from_slice(&z);
    let mut rz = dot(&r, &z, gate);
    if !(rz.is_finite() && rz > 0.0) {
        return None;
    }
    let tol = 1e-12 * bnorm;
    let maxit = (8 * n).max(64);
    let mut iters = 0usize;
    for _ in 0..maxit {
        if nrm2(&r, gate) <= tol {
            return finite_vec(x).map(|v| (v, iters));
        }
        iters += 1;
        matvec(a, &p, &mut ap, gate);
        let pap = dot(&p, &ap, gate);
        if !(pap.is_finite() && pap > 0.0) {
            return None;
        }
        let alpha = rz / pap;
        if !alpha.is_finite() {
            return None;
        }
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if nrm2(&r, gate) <= tol {
            return finite_vec(x).map(|v| (v, iters));
        }
        jacobi(a, &r, &mut z);
        let rz_new = dot(&r, &z, gate);
        if !(rz_new.is_finite() && rz_new > 0.0) {
            return None;
        }
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    if nrm2(&r, gate) <= 1e-8 * bnorm {
        return finite_vec(x).map(|v| (v, iters));
    }
    None
}

fn jacobi(a: &Csr, r: &[f64], z: &mut [f64]) {
    for i in 0..a.n {
        z[i] = r[i] / a.diag[i];
    }
}

fn finite_vec(x: Vec<f64>) -> Option<Vec<f64>> {
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn two_atoms() -> (Array1<f64>, Neighbours) {
        let pos = array![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let neigh = Neighbours {
            i: vec![0],
            j: vec![1],
            r: vec![1.5],
            anchor: vec![],
        };
        (pos, neigh)
    }

    #[test]
    fn the_quadratic_form_matches_the_paper() {
        let (pos, neigh) = two_atoms();
        let params = ExpParams::default();
        let p = Exp::assemble(pos.view(), &neigh, &params, 1.0, 1.5).unwrap();
        // c = exp(0) = 1. One bond: u^T P u = |u0 - u1|^2 + 0.1 |u|^2.
        let u = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let pu = p.apply(u.view());
        let form = u.dot(&pu);
        assert!((form - 1.1).abs() < 1e-12, "form {form}");
        let dense = p.to_dense();
        for i in 0..6 {
            for j in 0..i {
                assert!(
                    (dense[(i, j)] - dense[(j, i)]).abs() < 1e-12,
                    "asymmetry at {i},{j}"
                );
            }
        }
        let solved = crate::qn::solve_spd(&dense, &u);
        assert!(solved.is_some(), "dense Cholesky rejected P");
    }

    #[test]
    fn r_nn_is_the_max_of_per_atom_minima() {
        // Distances 1 (0-1), 2 (1-2), 3 (0-2). Per-atom minima are 1, 1, 2.
        let pos = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let neigh = Neighbours {
            i: vec![0, 1, 0],
            j: vec![1, 2, 2],
            r: vec![1.0, 2.0, 3.0],
            anchor: vec![],
        };
        let r = nearest_neighbour_distance(pos.view(), &neigh).unwrap();
        assert!((r - 2.0).abs() < 1e-12, "r_nn {r}");
    }

    #[test]
    fn both_directions_are_stored_once() {
        let (pos, _) = two_atoms();
        let neigh = Neighbours {
            i: vec![0, 1],
            j: vec![1, 0],
            r: vec![1.5, 1.5],
            anchor: vec![],
        };
        let params = ExpParams::default();
        let p = Exp::assemble(pos.view(), &neigh, &params, 1.0, 1.5).unwrap();
        let u = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let form = u.dot(&p.apply(u.view()));
        assert!(
            (form - 1.1).abs() < 1e-12,
            "double-counted bond, form {form}"
        );
    }

    #[test]
    fn an_anchor_adds_only_to_the_diagonal() {
        let (pos, mut neigh) = two_atoms();
        neigh.anchor.push((0, 1.5));
        let params = ExpParams::default();
        let p = Exp::assemble(pos.view(), &neigh, &params, 1.0, 1.5).unwrap();
        let dense = p.to_dense();
        // Free-free block is unchanged off diagonal. Atom 0's diagonal
        // gains the anchor weight 1.
        assert!((dense[(0, 3)] + 1.0).abs() < 1e-12);
        assert!(
            (dense[(0, 0)] - 2.1).abs() < 1e-12,
            "diag {}",
            dense[(0, 0)]
        );
    }

    #[test]
    fn the_sparse_solve_matches_a_dense_factor() {
        let (pos, neigh) = two_atoms();
        let params = ExpParams::default();
        let p = Exp::assemble(pos.view(), &neigh, &params, 2.5, 1.5).unwrap();
        let dense = p.to_dense();
        let rhs = array![0.2, -0.4, 0.1, 0.3, 0.0, -0.2];
        let sparse = p.solve(rhs.view()).expect("pcg");
        let dense_sol = crate::qn::solve_spd(&dense, &rhs).expect("chol");
        let err = (&sparse - &dense_sol)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(err < 1e-8, "sparse vs dense {err}");
    }

    #[test]
    fn mu_matches_the_paper_on_a_quadratic_metric() {
        let (pos, neigh) = two_atoms();
        let params = ExpParams {
            lengths: Some([4.0, 4.0, 4.0]),
            ..ExpParams::default()
        };
        let mu_true = 4.2;
        let p1 = Exp::assemble(pos.view(), &neigh, &params, 1.0, 1.5).unwrap();
        let grad = |x: ArrayView1<f64>| mu_true * p1.apply(x);
        let mu = estimate_mu(pos.view(), &neigh, &params, grad).unwrap();
        assert!((mu - mu_true).abs() < 1e-8, "mu {mu}");

        // Independent transcription of (14) and (15).
        let r_nn = 1.5;
        let scale = 1e-2 * r_nn;
        let mut v = Array1::zeros(pos.len());
        for i in 0..2 {
            for a in 0..3 {
                v[3 * i + a] = scale * (pos[3 * i + a] / 4.0).sin();
            }
        }
        let g0 = grad(pos.view());
        let g1 = grad((&pos + &v).view());
        let num = (&g1 - &g0).dot(&v);
        let den = v.dot(&p1.apply(v.view()));
        assert!((mu - num / den).abs() < 1e-12, "formula drift");
    }

    /// FNV-1a over the little-endian bits of the solution. The same
    /// digest is required with and without the `par` feature.
    fn fnv1a(x: &[f64]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for v in x {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    #[test]
    fn pcg_solution_hash_is_the_same_in_every_build() {
        // 512 atoms, several chunks of 1024, a banded SPD matrix.
        let n = 1536;
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut val = Vec::new();
        let mut diag = vec![0.0; n];
        for i in 0..n {
            let mut cols = Vec::new();
            if i >= 5 {
                cols.push((i - 5, -0.25));
            }
            if i > 0 {
                cols.push((i - 1, -1.0));
            }
            let d = 4.0 + (i % 7) as f64;
            cols.push((i, d));
            diag[i] = d;
            if i + 1 < n {
                cols.push((i + 1, -1.0));
            }
            if i + 5 < n {
                cols.push((i + 5, -0.25));
            }
            for (c, v) in cols {
                col_idx.push(c);
                val.push(v);
            }
            row_ptr.push(col_idx.len());
        }
        let csr = Csr {
            n,
            row_ptr,
            col_idx,
            val,
            diag,
            apply_gate: usize::MAX,
            solve_gate: usize::MAX,
        };
        let rhs: Vec<f64> = (0..n).map(|i| ((i * 17) % 11) as f64 - 5.0).collect();
        let (sol, iters) = solve_pcg(&csr, &rhs).expect("pcg");
        let hash = fnv1a(&sol);
        assert!(
            iters > 1 && iters <= n,
            "iters {iters} should take more than one step and at most n"
        );
        assert_eq!(
            hash, 0xf9a8d91302c9cb63,
            "committed hash drifted: iters {iters} hash {hash:#x}"
        );
    }
}
