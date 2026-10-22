//! Universal exponential preconditioner.
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
//! The session holds one assembly until the largest atomic move since
//! that assembly exceeds `r_nn / 2` (Packwood et al., the neighbour
//! cutoff used by their exponential factor). A rebuild estimates `μ`
//! again only when the caller asked for it.

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
}

/// Assembled exponential preconditioner `P`.
#[derive(Clone, Debug)]
pub struct Exp {
    mu: f64,
    r_nn: f64,
    csr: Arc<Csr>,
}

/// Session copy of the neighbour list and the factor last built.
#[derive(Clone, Debug)]
pub(crate) struct ExpState {
    neighbours: Neighbours,
    params: ExpParams,
    mu: Option<f64>,
    r_nn: f64,
    matrix: Option<Exp>,
    /// `r_nn` is filled from the first real geometry.
    r_nn_pending: bool,
    /// Positions of the assembly currently held. `None` until the first
    /// matrix this session will reuse.
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
        let bonds = bonds(natoms, positions, neighbours, params.r_cut)?;
        let csr = build_csr(
            natoms,
            &bonds.pairs,
            &bonds.anchors,
            mu,
            params.a,
            r_nn,
            params.c_stab,
        );
        Ok(Self {
            mu,
            r_nn,
            csr: Arc::new(csr),
        })
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
        self.csr.n
    }

    /// `P v`.
    #[must_use]
    pub fn apply(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let mut y = vec![0.0; self.csr.n];
        if v.len() == self.csr.n {
            matvec(&self.csr, v.as_slice().unwrap_or(&v.to_vec()), &mut y);
        }
        Array1::from_vec(y)
    }

    /// Solve `P x = rhs`. `None` when the iterate is not a descent-quality
    /// solution (the matrix is not SPD on the Krylov subspace, or the
    /// residual stalls).
    #[must_use]
    pub fn solve(&self, rhs: ArrayView1<f64>) -> Option<Array1<f64>> {
        if rhs.len() != self.csr.n {
            return None;
        }
        let rhs_vec = rhs
            .as_slice()
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| rhs.to_vec());
        solve_pcg(&self.csr, &rhs_vec).map(Array1::from_vec)
    }

    /// Dense copy, for tests and for a dense Cholesky reference.
    #[must_use]
    pub fn to_dense(&self) -> Array2<f64> {
        let n = self.csr.n;
        let mut a = Array2::zeros((n, n));
        for i in 0..n {
            for k in self.csr.row_ptr[i]..self.csr.row_ptr[i + 1] {
                a[(i, self.csr.col_idx[k])] = self.csr.val[k];
            }
        }
        a
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
            let matrix = Exp::assemble(positions, neighbours, params, mu, r_nn)?;
            state.mu = Some(matrix.mu());
            state.matrix = Some(matrix);
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
        if !params.a.is_finite() || !params.c_stab.is_finite() || params.c_stab <= 0.0 {
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
        if self.r_nn_pending {
            if let Some(r) = nearest_neighbour_distance(positions, &self.neighbours) {
                self.r_nn = r;
                self.r_nn_pending = false;
            }
        }
        if self.held(positions) {
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

    /// True when the assembly at [`Self::built_at`] is still the one to use.
    fn held(&self, positions: ArrayView1<f64>) -> bool {
        if self.r_nn_pending || self.matrix.is_none() || self.mu.is_none() || self.params.recalc_mu
        {
            return false;
        }
        let Some(prev) = &self.built_at else {
            return false;
        };
        if prev.len() != positions.len() || !(self.r_nn.is_finite() && self.r_nn > 0.0) {
            return false;
        }
        let delta = &positions.to_owned() - prev;
        let disp = crate::step::max_atom_displacement(&delta);
        disp.is_finite() && disp <= 0.5 * self.r_nn
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
    if n % 3 != 0 || !(r_nn.is_finite() && r_nn > 0.0) {
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
    if positions.len() % 3 != 0 || positions.is_empty() {
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

struct BondSet {
    pairs: Vec<(usize, usize, f64)>,
    anchors: Vec<(usize, f64)>,
}

fn bonds(
    natoms: usize,
    positions: ArrayView1<f64>,
    neighbours: &Neighbours,
    r_cut: Option<f64>,
) -> Result<BondSet> {
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
    Ok(BondSet { pairs, anchors })
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
    for (i, diag_i) in diag.iter().enumerate() {
        trips.push((i, i, *diag_i));
    }
    trips.sort_unstable_by(|p, q| (p.0, p.1).cmp(&(q.0, q.1)));
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(trips.len());
    let mut val = Vec::with_capacity(trips.len());
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
    }
}

fn matvec(a: &Csr, x: &[f64], y: &mut [f64]) {
    for (i, slot) in y.iter_mut().enumerate() {
        let mut s = 0.0;
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            s += a.val[k] * x[a.col_idx[k]];
        }
        *slot = s;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn nrm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn solve_pcg(a: &Csr, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.n;
    if b.len() != n {
        return None;
    }
    let bnorm = nrm2(b);
    if bnorm == 0.0 {
        return Some(vec![0.0; n]);
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
    let mut rz = dot(&r, &z);
    if !(rz.is_finite() && rz > 0.0) {
        return None;
    }
    let tol = 1e-12 * bnorm;
    let maxit = (8 * n).max(64);
    for _ in 0..maxit {
        if nrm2(&r) <= tol {
            return finite_vec(x);
        }
        matvec(a, &p, &mut ap);
        let pap = dot(&p, &ap);
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
        jacobi(a, &r, &mut z);
        let rz_new = dot(&r, &z);
        if !(rz_new.is_finite() && rz_new > 0.0) {
            return None;
        }
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    if nrm2(&r) <= 1e-8 * bnorm {
        return finite_vec(x);
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

    /// Cholesky solve. `None` when a leading minor is not positive.
    fn chol_solve(a: &Array2<f64>, rhs: &Array1<f64>) -> Option<Array1<f64>> {
        let n = rhs.len();
        if a.nrows() != n || a.ncols() != n {
            return None;
        }
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[(i, j)];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if sum.partial_cmp(&1e-14) != Some(std::cmp::Ordering::Greater) {
                        return None;
                    }
                    l[i * n + j] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = rhs[i];
            for k in 0..i {
                sum -= l[i * n + k] * y[k];
            }
            y[i] = sum / l[i * n + i];
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..n {
                sum -= l[k * n + i] * x[k];
            }
            x[i] = sum / l[i * n + i];
        }
        Some(Array1::from_vec(x))
    }

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
        let solved = chol_solve(&dense, &u);
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
        let dense_sol = chol_solve(&dense, &rhs).expect("chol");
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

    #[test]
    fn the_matrix_is_held_until_an_atom_moves_half_of_r_nn() {
        let mut pos = array![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let neigh = Neighbours {
            i: vec![0],
            j: vec![1],
            r: vec![],
            anchor: vec![],
        };
        let params = ExpParams {
            mu: Some(1.0),
            r_nn: Some(1.5),
            ..ExpParams::default()
        };
        let mut state = ExpState::new(pos.view(), &neigh, &params).unwrap();
        let grad = Array1::zeros(pos.len());
        let grad_fn = |p: ArrayView1<f64>| Array1::zeros(p.len());
        assert!(
            !state.refresh(pos.view(), &grad, grad_fn).unwrap(),
            "the install assembly should be reused"
        );
        let held = state.matrix().unwrap().to_dense();
        pos[3] += 0.1;
        assert!(
            !state.refresh(pos.view(), &grad, grad_fn).unwrap(),
            "0.1 is under r_nn / 2"
        );
        let still = state.matrix().unwrap().to_dense();
        assert_eq!(held, still);
        pos[3] += 1.0;
        assert!(
            state.refresh(pos.view(), &grad, grad_fn).unwrap(),
            "the atom has moved past r_nn / 2"
        );
        let rebuilt = state.matrix().unwrap().to_dense();
        assert!(
            (rebuilt[(0, 3)] - held[(0, 3)]).abs() > 1e-3,
            "distance changed, weight did not"
        );
    }
}
