//! Bonded force-field preconditioner.
//!
//! Makri, Ortner, and Kermode, *A preconditioning scheme for minimum
//! energy path finding methods*, J. Chem. Phys. 150, 094109 (2019),
//! <https://doi.org/10.1063/1.5064465>.
//!
//! `P` is the sum of reduced internal terms
//! `|∂²V/∂q²| (∂q/∂x)(∂q/∂x)^T`, plus `c_stab` on the diagonal. A
//! harmonic bond uses `k` as that curvature. A harmonic angle uses
//! `k_θ`. A Morse bond uses the absolute second derivative
//! `|2 D α² exp(-α(r-r0)) (2 exp(-α(r-r0)) - 1)|`, which is the ASE
//! reduced Morse term, so the tail does not make `P` indefinite. A
//! harmonic torsion uses `k_φ (∂φ/∂x)(∂φ/∂x)^T`. Terms longer than
//! `r_cut` are dropped. Harmonic curvatures enter as absolute values,
//! the same convention as the Morse term, so a signed constant cannot
//! make `P` indefinite.
//!
//! [`Ff::assemble`] returns that sum at scale 1. The session multiplies
//! it by `μ`. [`FfParams::mu`] `None` estimates `μ` with the sine-probe
//! Rayleigh quotient shared with [`super::Exp`]. The matrix is
//! symmetric positive definite for `c_stab > 0` and `μ > 0`. The solve
//! and the rebuild rule are those of [`super::Exp`].

use ndarray::{Array1, ArrayView1};

use super::geom::{
    add_rank1, angle_grad, dihedral_grad, mic, norm3, pos_of, require_atoms, scale3,
};
use super::{Metric, csr_from_coo, max_atom_displacement, needs_rebuild};
use crate::error::{Error, Result};

/// Harmonic bond. `k` is the curvature, energy / Å².
#[derive(Clone, Copy, Debug)]
pub struct BondTerm {
    /// First atom.
    pub i: usize,
    /// Second atom.
    pub j: usize,
    /// Curvature.
    pub k: f64,
}

/// Harmonic angle. `k_theta` is energy / rad². The vertex is `vertex`.
#[derive(Clone, Copy, Debug)]
pub struct AngleTerm {
    /// First end.
    pub i: usize,
    /// Vertex.
    pub vertex: usize,
    /// Second end.
    pub k: usize,
    /// Curvature.
    pub k_theta: f64,
}

/// Morse bond. The reduced term uses the absolute curvature.
#[derive(Clone, Copy, Debug)]
pub struct MorseTerm {
    /// First atom.
    pub i: usize,
    /// Second atom.
    pub j: usize,
    /// Well depth.
    pub d: f64,
    /// Width `α`.
    pub alpha: f64,
    /// Equilibrium distance.
    pub r0: f64,
}

/// Harmonic torsion `(i, j, k, l)`. `k_phi` is energy / rad².
#[derive(Clone, Copy, Debug)]
pub struct DihedralTerm {
    /// First end.
    pub i: usize,
    /// Second atom, start of the central bond.
    pub j: usize,
    /// Third atom, end of the central bond.
    pub k: usize,
    /// Fourth atom.
    pub l: usize,
    /// Curvature.
    pub k_phi: f64,
}

/// Borrowed bonded terms. Empty slices contribute nothing.
#[derive(Clone, Copy, Debug)]
pub struct FfTerms<'a> {
    /// Harmonic bonds.
    pub bonds: &'a [BondTerm],
    /// Harmonic angles.
    pub angles: &'a [AngleTerm],
    /// Morse bonds.
    pub morses: &'a [MorseTerm],
    /// Harmonic torsions.
    pub dihedrals: &'a [DihedralTerm],
}

impl Default for FfTerms<'static> {
    fn default() -> Self {
        Self {
            bonds: &[],
            angles: &[],
            morses: &[],
            dihedrals: &[],
        }
    }
}

/// Parameters of the FF preconditioner.
#[derive(Clone, Debug)]
pub struct FfParams {
    /// Diagonal shift. Must be positive. Default `0.1`.
    pub c_stab: f64,
    /// Drop a term whose leg is at least this long. `None` keeps all.
    pub r_cut: Option<f64>,
    /// Orthorhombic cell lengths for the minimum image.
    pub lengths: Option<[f64; 3]>,
    /// Energy scale. `None` estimates it on the step that builds `P`.
    pub mu: Option<f64>,
    /// Reestimate `μ` whenever the matrix is rebuilt.
    pub recalc_mu: bool,
}

impl Default for FfParams {
    fn default() -> Self {
        Self {
            c_stab: 0.1,
            r_cut: None,
            lengths: None,
            mu: None,
            recalc_mu: false,
        }
    }
}

/// Assembled FF preconditioner.
#[derive(Clone, Debug)]
pub struct Ff {
    r_nn: f64,
    metric: Metric,
}

/// Session copy of the bonded list.
#[derive(Clone, Debug)]
pub(crate) struct FfState {
    bonds: Vec<BondTerm>,
    angles: Vec<AngleTerm>,
    morses: Vec<MorseTerm>,
    dihedrals: Vec<DihedralTerm>,
    params: FfParams,
    metric: Option<Metric>,
    mu: Option<f64>,
    r_nn: f64,
    built_at: Option<Array1<f64>>,
}

impl Ff {
    /// Build `P` from explicit bonded terms.
    ///
    /// At least one term is required. Indices lie in `0..N` for a
    /// coordinate vector of length `3 N`.
    pub fn assemble(
        positions: ArrayView1<f64>,
        terms: FfTerms<'_>,
        params: &FfParams,
    ) -> Result<Self> {
        let natoms = require_atoms(positions.len())?;
        if !(params.c_stab.is_finite() && params.c_stab > 0.0) {
            return Err(Error::Precon {
                what: "c_stab is not positive",
            });
        }
        if terms.bonds.is_empty()
            && terms.angles.is_empty()
            && terms.morses.is_empty()
            && terms.dihedrals.is_empty()
        {
            return Err(Error::Precon {
                what: "ff has no bonded terms",
            });
        }
        let coords = positions
            .as_slice()
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| positions.to_vec());
        if coords.iter().any(|v| !v.is_finite()) {
            return Err(Error::Precon {
                what: "a coordinate is not finite",
            });
        }
        let (metric, r_nn) = assemble_metric(natoms, &coords, terms, params)?;
        Ok(Self { r_nn, metric })
    }

    /// Nearest-neighbour distance among the terms that entered `P`.
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

    /// Dense copy, for tests and for a dense Cholesky reference.
    #[must_use]
    pub fn to_dense(&self) -> ndarray::Array2<f64> {
        self.metric.to_dense()
    }
}

impl FfState {
    pub(crate) fn new(
        positions: ArrayView1<f64>,
        terms: FfTerms<'_>,
        params: &FfParams,
    ) -> Result<Self> {
        let built = Ff::assemble(positions, terms, params)?;
        let mu = usable_mu(params.mu)?;
        let metric = mu.map(|m| built.metric.scaled(m));
        let built_at = metric.as_ref().map(|_| positions.to_owned());
        Ok(Self {
            bonds: terms.bonds.to_vec(),
            angles: terms.angles.to_vec(),
            morses: terms.morses.to_vec(),
            dihedrals: terms.dihedrals.to_vec(),
            params: params.clone(),
            r_nn: built.r_nn(),
            metric,
            mu,
            built_at,
        })
    }

    #[cfg(feature = "capi")]
    pub(crate) fn deferred(
        bonds: Vec<BondTerm>,
        angles: Vec<AngleTerm>,
        params: FfParams,
    ) -> Result<Self> {
        if bonds.is_empty() && angles.is_empty() {
            return Err(Error::Precon {
                what: "ff has no bonded terms",
            });
        }
        if !(params.c_stab.is_finite() && params.c_stab > 0.0) {
            return Err(Error::Precon {
                what: "c_stab is not positive",
            });
        }
        let mu = usable_mu(params.mu)?;
        Ok(Self {
            bonds,
            angles,
            morses: Vec::new(),
            dihedrals: Vec::new(),
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
        let terms = FfTerms {
            bonds: &self.bonds,
            angles: &self.angles,
            morses: &self.morses,
            dihedrals: &self.dihedrals,
        };
        let built = Ff::assemble(positions, terms, &self.params)?;
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

fn in_range(atom: usize, natoms: usize) -> Result<()> {
    if atom >= natoms {
        Err(Error::Precon {
            what: "bonded index is out of range",
        })
    } else {
        Ok(())
    }
}

fn leg_ok(r: f64, cut: Option<f64>) -> bool {
    r.is_finite() && r > 1e-8 && cut.is_none_or(|c| r < c)
}

fn assemble_metric(
    natoms: usize,
    positions: &[f64],
    terms: FfTerms<'_>,
    params: &FfParams,
) -> Result<(Metric, f64)> {
    for b in terms.bonds {
        in_range(b.i, natoms)?;
        in_range(b.j, natoms)?;
        if b.i == b.j || !b.k.is_finite() {
            return Err(Error::Precon {
                what: "a bond term is not usable",
            });
        }
    }
    for a in terms.angles {
        in_range(a.i, natoms)?;
        in_range(a.vertex, natoms)?;
        in_range(a.k, natoms)?;
        if a.i == a.vertex || a.k == a.vertex || a.i == a.k || !a.k_theta.is_finite() {
            return Err(Error::Precon {
                what: "an angle term is not usable",
            });
        }
    }
    for m in terms.morses {
        in_range(m.i, natoms)?;
        in_range(m.j, natoms)?;
        if m.i == m.j
            || !m.d.is_finite()
            || !m.alpha.is_finite()
            || !(m.r0.is_finite() && m.r0 > 0.0)
        {
            return Err(Error::Precon {
                what: "a morse term is not usable",
            });
        }
    }
    for d in terms.dihedrals {
        in_range(d.i, natoms)?;
        in_range(d.j, natoms)?;
        in_range(d.k, natoms)?;
        in_range(d.l, natoms)?;
        let ids = [d.i, d.j, d.k, d.l];
        if ids.iter().collect::<std::collections::HashSet<_>>().len() != 4 || !d.k_phi.is_finite() {
            return Err(Error::Precon {
                what: "a dihedral term is not usable",
            });
        }
    }
    let mut min_d = vec![f64::INFINITY; natoms];
    let mut trips = Vec::new();
    let note = |min_d: &mut [f64], i: usize, j: usize, r: f64| {
        min_d[i] = min_d[i].min(r);
        min_d[j] = min_d[j].min(r);
    };
    for b in terms.bonds {
        let pi = pos_of(positions, b.i);
        let pj = pos_of(positions, b.j);
        let delta = mic(pi, pj, params.lengths);
        let r = norm3(delta);
        if !leg_ok(r, params.r_cut) {
            continue;
        }
        let e = scale3(delta, 1.0 / r);
        let neg = scale3(e, -1.0);
        add_rank1(&mut trips, &[(Some(b.i), e), (Some(b.j), neg)], b.k.abs());
        note(&mut min_d, b.i, b.j, r);
    }
    for m in terms.morses {
        let pi = pos_of(positions, m.i);
        let pj = pos_of(positions, m.j);
        let delta = mic(pi, pj, params.lengths);
        let r = norm3(delta);
        if !leg_ok(r, params.r_cut) {
            continue;
        }
        let efold = (-m.alpha * (r - m.r0)).exp();
        let curv = (2.0 * m.d * m.alpha * m.alpha * efold * (2.0 * efold - 1.0)).abs();
        let e = scale3(delta, 1.0 / r);
        let neg = scale3(e, -1.0);
        add_rank1(&mut trips, &[(Some(m.i), e), (Some(m.j), neg)], curv);
        note(&mut min_d, m.i, m.j, r);
    }
    for a in terms.angles {
        let pi = pos_of(positions, a.i);
        let pv = pos_of(positions, a.vertex);
        let pk = pos_of(positions, a.k);
        let ri = norm3(mic(pi, pv, params.lengths));
        let rk = norm3(mic(pk, pv, params.lengths));
        if !leg_ok(ri, params.r_cut) || !leg_ok(rk, params.r_cut) {
            continue;
        }
        let Some((gi, gj, gk)) = angle_grad(pi, pv, pk, params.lengths) else {
            continue;
        };
        add_rank1(
            &mut trips,
            &[(Some(a.i), gi), (Some(a.vertex), gj), (Some(a.k), gk)],
            a.k_theta.abs(),
        );
        note(&mut min_d, a.i, a.vertex, ri);
        note(&mut min_d, a.k, a.vertex, rk);
    }
    for d in terms.dihedrals {
        let pi = pos_of(positions, d.i);
        let pj = pos_of(positions, d.j);
        let pk = pos_of(positions, d.k);
        let pl = pos_of(positions, d.l);
        let rij = norm3(mic(pi, pj, params.lengths));
        let rjk = norm3(mic(pj, pk, params.lengths));
        let rkl = norm3(mic(pk, pl, params.lengths));
        if !leg_ok(rij, params.r_cut) || !leg_ok(rjk, params.r_cut) || !leg_ok(rkl, params.r_cut) {
            continue;
        }
        let Some(g) = dihedral_grad(pi, pj, pk, pl, params.lengths) else {
            continue;
        };
        add_rank1(
            &mut trips,
            &[
                (Some(d.i), g[0]),
                (Some(d.j), g[1]),
                (Some(d.k), g[2]),
                (Some(d.l), g[3]),
            ],
            d.k_phi.abs(),
        );
    }
    let n = 3 * natoms;
    for i in 0..n {
        trips.push((i, i, params.c_stab));
    }
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
    Ok((Metric::from_csr(csr_from_coo(n, trips)), r_nn))
}

/// Bonds for every pair inside `r_cut`, and an angle at each atom for
/// every pair of its neighbours. Force constants are the caller's.
pub fn bonded_terms(
    positions: ArrayView1<f64>,
    r_cut: f64,
    k_bond: f64,
    k_angle: f64,
    lengths: Option<[f64; 3]>,
) -> Result<(Vec<BondTerm>, Vec<AngleTerm>)> {
    let natoms = require_atoms(positions.len())?;
    if !(r_cut.is_finite() && r_cut > 0.0) || !k_bond.is_finite() || !k_angle.is_finite() {
        return Err(Error::Precon {
            what: "bonded cutoff or force constant is not usable",
        });
    }
    let coords = positions
        .as_slice()
        .map(<[f64]>::to_vec)
        .unwrap_or_else(|| positions.to_vec());
    let mut bonds = Vec::new();
    let mut adj = vec![Vec::new(); natoms];
    for i in 0..natoms {
        for j in (i + 1)..natoms {
            let r = norm3(mic(pos_of(&coords, i), pos_of(&coords, j), lengths));
            if r > 1e-8 && r < r_cut {
                bonds.push(BondTerm { i, j, k: k_bond });
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    let mut angles = Vec::new();
    for v in 0..natoms {
        for a in 0..adj[v].len() {
            for b in (a + 1)..adj[v].len() {
                angles.push(AngleTerm {
                    i: adj[v][a],
                    vertex: v,
                    k: adj[v][b],
                    k_theta: k_angle,
                });
            }
        }
    }
    Ok((bonds, angles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn a_harmonic_bond_is_the_outer_product_of_its_gradient() {
        let pos = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let bonds = [BondTerm { i: 0, j: 1, k: 2.5 }];
        let terms = FfTerms {
            bonds: &bonds,
            ..FfTerms::default()
        };
        let mut params = FfParams::default();
        params.c_stab = 0.1;
        let p = Ff::assemble(pos.view(), terms, &params).unwrap();
        let u = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let form = u.dot(&p.apply(u.view()));
        assert!((form - 2.6).abs() < 1e-12, "form {form}");
        let t = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let trans = t.dot(&p.apply(t.view()));
        assert!((trans - 0.2).abs() < 1e-12, "translation {trans}");
        let dense = p.to_dense();
        assert!(crate::qn::solve_spd(&dense, &u).is_some());
    }

    #[test]
    fn an_angle_term_matches_the_wilson_gradient() {
        let pos = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let angles = [AngleTerm {
            i: 0,
            vertex: 1,
            k: 2,
            k_theta: 3.0,
        }];
        let terms = FfTerms {
            angles: &angles,
            ..FfTerms::default()
        };
        let mut params = FfParams::default();
        params.c_stab = 0.2;
        let p = Ff::assemble(pos.view(), terms, &params).unwrap();
        let (gi, gj, gk) =
            angle_grad([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], None).unwrap();
        let mut g = Array1::zeros(9);
        for d in 0..3 {
            g[d] = gi[d];
            g[3 + d] = gj[d];
            g[6 + d] = gk[d];
        }
        let u = array![0.3, -0.1, 0.2, 0.0, 0.4, -0.2, -0.3, 0.1, 0.05];
        let expect = 3.0 * g.dot(&u).powi(2) + 0.2 * u.dot(&u);
        let form = u.dot(&p.apply(u.view()));
        assert!((form - expect).abs() < 1e-8, "form {form} expect {expect}");
    }

    #[test]
    fn the_morse_tail_stays_positive() {
        let pos = array![3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let morses = [MorseTerm {
            i: 0,
            j: 1,
            d: 1.0,
            alpha: 1.5,
            r0: 1.0,
        }];
        let terms = FfTerms {
            morses: &morses,
            ..FfTerms::default()
        };
        let p = Ff::assemble(pos.view(), terms, &FfParams::default()).unwrap();
        let u = array![1.0, 0.2, -0.1, -0.4, 0.0, 0.3];
        let form = u.dot(&p.apply(u.view()));
        assert!(form > 0.0, "form {form}");
        let e = (-1.5_f64 * (3.0 - 1.0)).exp();
        let curv = (2.0 * 1.0 * 1.5 * 1.5 * e * (2.0 * e - 1.0)).abs();
        let stretch = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let along = stretch.dot(&p.apply(stretch.view()));
        assert!(
            (along - (curv + 0.1)).abs() < 1e-8,
            "along {along} curv {curv}"
        );
    }
}
