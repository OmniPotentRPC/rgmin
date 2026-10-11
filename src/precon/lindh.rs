//! Lindh model Hessian as a preconditioner.
//!
//! Lindh, Bernhardsson, Karlström, and Malmqvist, *On the use of a
//! Hessian model function in molecular geometry optimizations*,
//! Chem. Phys. Lett. 241, 423 (1995),
//! <https://doi.org/10.1016/0009-2614(95)00646-L>.
//!
//! `P` is the Cartesian reduced Hessian of that model. A pair inside
//! the cutoff contributes `k_r ρ (∂r/∂x)(∂r/∂x)^T`. An angle whose two
//! legs are both inside the cutoff contributes
//! `k_θ ρ_ij ρ_jk (∂θ/∂x)(∂θ/∂x)^T`. The diagonal carries `c_stab`.
//! `ρ = exp(α (r_ref² - r²))` uses the paper's six `α` and six `r_ref`
//! values, indexed by element row (H/He, Li-Ne, Na and heavier). `α`
//! and `r_ref` are converted from Bohr to Ångström. The stretch
//! constant `0.45` is converted from Hartree/Bohr² to eV/Å². The angle
//! constant `0.15` and the torsion constant `0.005` are converted from
//! Hartree/rad² to eV/rad². A torsion is added only when
//! [`LindhParams::dihedrals`] is set.
//!
//! [`Lindh::assemble`] returns that model at scale 1. The session
//! multiplies it by `μ`. [`LindhParams::mu`] `None` estimates `μ` with
//! the same sine-probe Rayleigh quotient as [`super::Exp`]. `P` is
//! symmetric positive definite for `c_stab > 0` and `μ > 0`. The solve
//! is the Jacobi conjugate gradient shared with [`super::Exp`]. The
//! session holds `P` until the largest atomic move exceeds half the
//! nearest-neighbour distance. Pairs come from a cell list whose bins
//! are visited in lexicographic order and then sorted by atom index,
//! so the matrix does not depend on the thread count.

use ndarray::{Array1, ArrayView1};

use super::geom::{angle_grad, dihedral_grad, mic, norm3, pos_of, require_atoms};
use super::{
    BlockAccum, Metric, N_PAR_LINDH_APPLY, N_PAR_LINDH_SOLVE, max_atom_displacement, needs_rebuild,
    note_block,
};
use crate::error::{Error, Result};

const BOHR: f64 = 0.529177210903;
const HARTREE: f64 = 27.211386245988;
const BOHR2: f64 = BOHR * BOHR;
const RHO_MIN: f64 = 1e-3;

const ALPHA_RAW: [[f64; 3]; 3] = [
    [1.0000, 0.3949, 0.3949],
    [0.3949, 0.2800, 0.2800],
    [0.3949, 0.2800, 0.2800],
];
const RREF_RAW: [[f64; 3]; 3] = [[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]];

fn row_of(z: i32) -> usize {
    if z <= 2 {
        0
    } else if z <= 10 {
        1
    } else {
        2
    }
}

fn alpha(i: usize, j: usize) -> f64 {
    ALPHA_RAW[i][j] / BOHR2
}

fn r_ref(i: usize, j: usize) -> f64 {
    RREF_RAW[i][j] * BOHR
}

fn k_stretch() -> f64 {
    0.45 * HARTREE / BOHR2
}

fn k_bend() -> f64 {
    0.15 * HARTREE
}

fn k_torsion() -> f64 {
    0.005 * HARTREE
}

fn rho(z1: i32, z2: i32, r: f64) -> f64 {
    let i = row_of(z1);
    let j = row_of(z2);
    let a = alpha(i, j);
    let rr = r_ref(i, j);
    (a * (rr * rr - r * r)).exp()
}

fn default_search() -> f64 {
    let mut max_r = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let a = alpha(i, j);
            let rr = r_ref(i, j);
            let r2 = rr * rr - RHO_MIN.ln() / a;
            if r2 > 0.0 {
                max_r = max_r.max(r2.sqrt());
            }
        }
    }
    max_r
}

/// An atom with no coordinate in the free vector.
#[derive(Clone, Copy, Debug)]
pub struct Ghost {
    /// Atomic number.
    pub z: i32,
    /// Cartesian position, same frame as the free coordinates.
    pub position: [f64; 3],
}

/// Parameters of the Lindh preconditioner.
///
/// `c_stab` defaults to `0.1` eV/Å². `r_cut` `None` keeps a pair whose
/// Lindh weight is at least `1e-3`. `angles` defaults on. `dihedrals`
/// defaults off.
#[derive(Clone, Debug)]
pub struct LindhParams {
    /// Diagonal shift, eV/Å². Must be positive.
    pub c_stab: f64,
    /// Hard cutoff, Å. `None` uses the weight threshold.
    pub r_cut: Option<f64>,
    /// Include angle terms.
    pub angles: bool,
    /// Include the torsion term.
    pub dihedrals: bool,
    /// Orthorhombic cell lengths for the minimum image.
    /// A nonpositive component leaves that axis open.
    pub lengths: Option<[f64; 3]>,
    /// Energy scale. `None` estimates it on the step that builds `P`.
    pub mu: Option<f64>,
    /// Reestimate `μ` whenever the matrix is rebuilt.
    pub recalc_mu: bool,
}

impl Default for LindhParams {
    fn default() -> Self {
        Self {
            c_stab: 0.1,
            r_cut: None,
            angles: true,
            dihedrals: false,
            lengths: None,
            mu: None,
            recalc_mu: false,
        }
    }
}

struct Site {
    z: i32,
    pos: [f64; 3],
    free: Option<usize>,
}

#[derive(Clone)]
struct Leg {
    site: usize,
    rho: f64,
}

/// Assembled Lindh preconditioner.
#[derive(Clone, Debug)]
pub struct Lindh {
    r_nn: f64,
    metric: Metric,
}

/// Session copy. The matrix is rebuilt from the current positions.
#[derive(Clone, Debug)]
pub(crate) struct LindhState {
    z: Vec<i32>,
    ghosts: Vec<Ghost>,
    params: LindhParams,
    metric: Option<Metric>,
    mu: Option<f64>,
    r_nn: f64,
    built_at: Option<Array1<f64>>,
}

impl Lindh {
    /// Build `P` at `positions`.
    ///
    /// `positions` has length `3 N` in C order. `z` has length `N`.
    /// `ghosts` are frozen atoms. A bond to a ghost contributes only
    /// the free atom's block.
    pub fn assemble(
        positions: ArrayView1<f64>,
        z: &[i32],
        ghosts: &[Ghost],
        params: &LindhParams,
    ) -> Result<Self> {
        let natoms = require_atoms(positions.len())?;
        if z.len() != natoms {
            return Err(Error::Precon {
                what: "atomic numbers do not match the coordinate vector",
            });
        }
        if !(params.c_stab.is_finite() && params.c_stab > 0.0) {
            return Err(Error::Precon {
                what: "c_stab is not positive",
            });
        }
        if let Some(cut) = params.r_cut {
            if !(cut.is_finite() && cut > 0.0) {
                return Err(Error::Precon {
                    what: "r_cut is not positive",
                });
            }
        }
        if z.iter().any(|a| *a <= 0) || ghosts.iter().any(|g| g.z <= 0) {
            return Err(Error::Precon {
                what: "atomic number is not positive",
            });
        }
        let coords = positions
            .as_slice()
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| positions.to_vec());
        if coords.iter().any(|v| !v.is_finite())
            || ghosts
                .iter()
                .any(|g| g.position.iter().any(|v| !v.is_finite()))
        {
            return Err(Error::Precon {
                what: "a coordinate is not finite",
            });
        }
        let (metric, r_nn) = assemble_metric(&coords, z, ghosts, params)?;
        Ok(Self { r_nn, metric })
    }

    /// Nearest-neighbour distance of this assembly.
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

    /// Solve `P x = rhs`. `None` when the iterate is not usable.
    #[must_use]
    pub fn solve(&self, rhs: ArrayView1<f64>) -> Option<Array1<f64>> {
        self.metric.solve(rhs)
    }

    /// Jacobi PCG iterations for one solve.
    #[must_use]
    pub fn solve_iterations(&self, rhs: ArrayView1<f64>) -> Option<usize> {
        self.metric.solve_counted(rhs).map(|(_, n)| n)
    }

    /// Dense copy, for tests and for a dense Cholesky reference.
    #[must_use]
    pub fn to_dense(&self) -> ndarray::Array2<f64> {
        self.metric.to_dense()
    }
}

impl LindhState {
    pub(crate) fn new(
        positions: ArrayView1<f64>,
        z: &[i32],
        ghosts: &[Ghost],
        params: &LindhParams,
    ) -> Result<Self> {
        let built = Lindh::assemble(positions, z, ghosts, params)?;
        let mu = usable_mu(params.mu)?;
        let metric = mu.map(|m| built.metric.scaled(m));
        let built_at = metric.as_ref().map(|_| positions.to_owned());
        Ok(Self {
            z: z.to_vec(),
            ghosts: ghosts.to_vec(),
            params: params.clone(),
            r_nn: built.r_nn(),
            metric,
            mu,
            built_at,
        })
    }

    #[cfg(feature = "capi")]
    pub(crate) fn deferred(z: Vec<i32>, ghosts: Vec<Ghost>, params: LindhParams) -> Result<Self> {
        if z.is_empty() || z.iter().any(|a| *a <= 0) || ghosts.iter().any(|g| g.z <= 0) {
            return Err(Error::Precon {
                what: "atomic number is not positive",
            });
        }
        if !(params.c_stab.is_finite() && params.c_stab > 0.0) {
            return Err(Error::Precon {
                what: "c_stab is not positive",
            });
        }
        let mu = usable_mu(params.mu)?;
        Ok(Self {
            z,
            ghosts,
            params,
            metric: None,
            mu,
            r_nn: 1.0,
            built_at: None,
        })
    }

    pub(crate) fn metric(&self) -> Option<&Metric> {
        self.metric.as_ref()
    }

    pub(crate) fn mu(&self) -> Option<f64> {
        self.mu
    }

    pub(crate) fn refresh<G>(
        &mut self,
        positions: ArrayView1<f64>,
        grad_here: &Array1<f64>,
        grad: G,
    ) -> Result<bool>
    where
        G: Fn(ArrayView1<f64>) -> Array1<f64>,
    {
        if self.metric.is_some()
            && self.mu.is_some()
            && !self.params.recalc_mu
            && self.built_at.as_ref().is_some_and(|prev| {
                !needs_rebuild(max_atom_displacement(prev.view(), positions), self.r_nn)
            })
        {
            return Ok(false);
        }
        let built = Lindh::assemble(positions, &self.z, &self.ghosts, &self.params)?;
        self.r_nn = built.r_nn();
        if self.mu.is_none() || self.params.recalc_mu {
            match super::model_scale(
                &built.metric,
                positions,
                self.r_nn,
                self.params.lengths,
                grad_here,
                &grad,
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
        self.metric = Some(built.metric.scaled(mu));
        self.built_at = Some(positions.to_owned());
        Ok(true)
    }
}

fn usable_mu(mu: Option<f64>) -> Result<Option<f64>> {
    if let Some(mu) = mu {
        if !(mu.is_finite() && mu > 0.0) {
            return Err(Error::Precon {
                what: "mu is not positive",
            });
        }
    }
    Ok(mu)
}

/// One free-atom pair with its distance and Lindh weight.
type WeightedPair = (usize, usize, f64, f64);

/// Pairs inside the cutoff, from a cell list, sorted by site index.
fn gather_pairs(
    sites: &[Site],
    n_free: usize,
    search: f64,
    params: &LindhParams,
) -> (Vec<WeightedPair>, Vec<f64>) {
    let coords: Vec<[f64; 3]> = sites.iter().map(|s| s.pos).collect();
    let found = super::geom::pairs_in_range(&coords, search, params.lengths);
    let mut pairs = Vec::new();
    let mut min_d = vec![f64::INFINITY; n_free];
    for (a, b, r) in found {
        if sites[a].free.is_none() && sites[b].free.is_none() {
            continue;
        }
        let w = rho(sites[a].z, sites[b].z, r);
        let keep = match params.r_cut {
            Some(cut) => r < cut,
            None => w >= RHO_MIN,
        };
        if !keep || !w.is_finite() {
            continue;
        }
        if let Some(i) = sites[a].free {
            min_d[i] = min_d[i].min(r);
        }
        if let Some(j) = sites[b].free {
            min_d[j] = min_d[j].min(r);
        }
        pairs.push((a, b, r, w));
    }
    (pairs, min_d)
}

fn assemble_metric(
    positions: &[f64],
    z: &[i32],
    ghosts: &[Ghost],
    params: &LindhParams,
) -> Result<(Metric, f64)> {
    let n_free = z.len();
    let mut sites = Vec::with_capacity(n_free + ghosts.len());
    for i in 0..n_free {
        sites.push(Site {
            z: z[i],
            pos: pos_of(positions, i),
            free: Some(i),
        });
    }
    for g in ghosts {
        sites.push(Site {
            z: g.z,
            pos: g.position,
            free: None,
        });
    }
    let search = params.r_cut.unwrap_or_else(default_search);
    let (pairs, min_d) = gather_pairs(&sites, n_free, search, params);
    let mut r_nn = 0.0_f64;
    let mut any = false;
    for d in min_d {
        if d.is_finite() {
            r_nn = r_nn.max(d);
            any = true;
        }
    }
    if !any {
        r_nn = 1.0;
    }

    let mut adj: Vec<Vec<Leg>> = vec![Vec::new(); sites.len()];
    for &(a, b, _, w) in &pairs {
        adj[a].push(Leg { site: b, rho: w });
        adj[b].push(Leg { site: a, rho: w });
    }

    let mut partners = vec![Vec::new(); n_free];
    emit_terms(&sites, &pairs, &adj, params, |atoms, scale| {
        note_block(&mut partners, atoms, scale);
    });
    let mut acc = BlockAccum::new(n_free, partners);
    emit_terms(&sites, &pairs, &adj, params, |atoms, scale| {
        acc.add_rank1(atoms, scale);
    });
    acc.add_stabilization(params.c_stab);
    Ok((
        Metric::from_csr(acc.finish(N_PAR_LINDH_APPLY, N_PAR_LINDH_SOLVE)),
        r_nn,
    ))
}

/// Present each stretch, angle, and torsion in a fixed order.
fn emit_terms(
    sites: &[Site],
    pairs: &[(usize, usize, f64, f64)],
    adj: &[Vec<Leg>],
    params: &LindhParams,
    mut emit: impl FnMut(&[(Option<usize>, [f64; 3])], f64),
) {
    let ks = k_stretch();
    for &(a, b, _r, w) in pairs {
        let d = mic(sites[a].pos, sites[b].pos, params.lengths);
        let r = norm3(d);
        if r < 1e-12 {
            continue;
        }
        let e = [d[0] / r, d[1] / r, d[2] / r];
        let neg = [-e[0], -e[1], -e[2]];
        emit(&[(sites[a].free, e), (sites[b].free, neg)], ks * w);
    }
    if params.angles {
        let kb = k_bend();
        for v in 0..sites.len() {
            let legs = &adj[v];
            for i in 0..legs.len() {
                for j in (i + 1)..legs.len() {
                    let si = legs[i].site;
                    let sk = legs[j].site;
                    let Some((gi, gj, gk)) =
                        angle_grad(sites[si].pos, sites[v].pos, sites[sk].pos, params.lengths)
                    else {
                        continue;
                    };
                    emit(
                        &[
                            (sites[si].free, gi),
                            (sites[v].free, gj),
                            (sites[sk].free, gk),
                        ],
                        kb * legs[i].rho * legs[j].rho,
                    );
                }
            }
        }
    }
    if params.dihedrals {
        let kt = k_torsion();
        for &(j, k, _r, rho_jk) in pairs {
            if j >= k {
                continue;
            }
            for leg_i in &adj[j] {
                if leg_i.site == k {
                    continue;
                }
                for leg_l in &adj[k] {
                    if leg_l.site == j || leg_l.site == leg_i.site || leg_i.site >= leg_l.site {
                        continue;
                    }
                    let i = leg_i.site;
                    let l = leg_l.site;
                    let Some(g) = dihedral_grad(
                        sites[i].pos,
                        sites[j].pos,
                        sites[k].pos,
                        sites[l].pos,
                        params.lengths,
                    ) else {
                        continue;
                    };
                    emit(
                        &[
                            (sites[i].free, g[0]),
                            (sites[j].free, g[1]),
                            (sites[k].free, g[2]),
                            (sites[l].free, g[3]),
                        ],
                        kt * leg_i.rho * rho_jk * leg_l.rho,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn constants(z1: i32, z2: i32, r: f64) -> (f64, f64) {
        (k_stretch() * rho(z1, z2, r), 0.1)
    }

    #[test]
    fn a_stretch_matches_the_reduced_bond() {
        let pos = array![0.74, 0.0, 0.0, 0.0, 0.0, 0.0];
        let z = [1, 1];
        let p = Lindh::assemble(pos.view(), &z, &[], &LindhParams::default()).unwrap();
        let (kr, c) = constants(1, 1, 0.74);
        let u = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let form = u.dot(&p.apply(u.view()));
        assert!((form - (kr + c)).abs() < 1e-8, "form {form} kr {kr}");
        let t = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let trans = t.dot(&p.apply(t.view()));
        assert!((trans - 2.0 * c).abs() < 1e-8, "translation {trans}");
    }

    #[test]
    fn the_matrix_is_symmetric_positive_definite() {
        let pos = array![
            0.0,
            0.0,
            0.0,
            0.96,
            0.0,
            0.0,
            0.96 * 104.5_f64.to_radians().cos(),
            0.96 * 104.5_f64.to_radians().sin(),
            0.0
        ];
        let z = [8, 1, 1];
        let p = Lindh::assemble(pos.view(), &z, &[], &LindhParams::default()).unwrap();
        let dense = p.to_dense();
        for i in 0..9 {
            for j in 0..i {
                assert!((dense[(i, j)] - dense[(j, i)]).abs() < 1e-9);
            }
        }
        let rhs = array![0.2, -0.1, 0.05, 0.0, 0.3, -0.2, -0.1, 0.0, 0.15];
        let sparse = p.solve(rhs.view()).expect("pcg");
        let dense_sol = crate::qn::solve_spd(&dense, &rhs).expect("chol");
        let err = (&sparse - &dense_sol)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(err < 1e-6, "sparse vs dense {err}");
    }

    #[test]
    fn a_ghost_bond_touches_only_the_free_atom() {
        let pos = array![0.0, 0.0, 0.0];
        let z = [1];
        let ghosts = [Ghost {
            z: 1,
            position: [0.74, 0.0, 0.0],
        }];
        let p = Lindh::assemble(pos.view(), &z, &ghosts, &LindhParams::default()).unwrap();
        assert_eq!(p.dim(), 3);
        let (kr, c) = constants(1, 1, 0.74);
        let u = array![1.0, 0.0, 0.0];
        let form = u.dot(&p.apply(u.view()));
        assert!((form - (kr + c)).abs() < 1e-8, "anchor form {form}");
    }
}
