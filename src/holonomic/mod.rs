//! Holonomic interatomic distances for minimisation.
//!
//! A [`DistanceConstraints`] fixes `|x_i - x_j| = d` for a list of atom
//! pairs, with optional per-atom masses that set the inverse-mass metric
//! `W = M^-1`. Unit masses are the default.
//!
//! Trial points return to the set from the point they left (the anchor)
//! by corrections along the anchor bonds, mass weighted: SHAKE
//! (Ryckaert, Ciccotti and Berendsen 1977,
//! <https://doi.org/10.1016/0021-9991(77)90098-5>), LINCS (Hess, Bekker,
//! Berendsen and Fraaije 1997,
//! <https://doi.org/10.1002/(SICI)1096-987X(199709)18:12%3C1463::AID-JCC4%3E3.0.CO;2-H>;
//! Hess 2008, <https://doi.org/10.1021/ct700200b>) or, for a rigid
//! triangle such as three-site water, the analytic SETTLE (Miyamoto and
//! Kollman 1992, <https://doi.org/10.1002/jcc.540130805>). All three
//! solve the same equations, so they define one retraction. Vectors are
//! projected onto the tangent space with the inverse-mass projector
//! `Pi = I - W J^T (J W J^T)^-1 J`, as RATTLE (Andersen 1983,
//! <https://doi.org/10.1016/0021-9991(83)90014-1>) does for velocities,
//! and gradients with its transpose.
//!
//! Each block of coupled constraints (a connected component of the
//! constraint graph) is solved on its own. The GROMACS sources
//! `src/gromacs/mdlib/shake.cpp`, `lincs.cpp`, `settle.cpp` and
//! `constr.cpp` were the working reference; this module is written for
//! minimisation and its structure differs.

mod linalg;
mod lincs;
mod settle;
mod shake;

use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};

/// Unit roundoff.
pub(crate) const U: f64 = f64::EPSILON / 2.0;

/// `gamma(n) = n U / (1 - n U)`, Higham (2002) Lemma 3.1: the relative
/// error bound of `n` rounded operations.
const fn gamma(n: f64) -> f64 {
    n * U / (1.0 - n * U)
}

/// Six rounded operations: a three-term dot product of differences.
pub(crate) const GAMMA6: f64 = gamma(6.0);

/// Which iterative solver returns a block to the set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConstraintSolver {
    /// LINCS where its expansion converges at least as fast per pass as
    /// the Gauss-Seidel bound of SHAKE, SHAKE elsewhere.
    #[default]
    Auto,
    /// Gauss-Seidel SHAKE on every block.
    Shake,
    /// LINCS on every block. A block whose expansion diverges is a
    /// [`ConstraintError::LincsUnsafe`].
    Lincs,
}

/// The algorithm that ran or failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConstraintAlgorithm {
    /// Gauss-Seidel SHAKE on positions.
    Shake,
    /// LINCS on positions.
    Lincs,
    /// Analytic SETTLE on a rigid triangle.
    Settle,
    /// Tangent projection of a vector or gradient.
    Rattle,
}

impl ConstraintAlgorithm {
    /// Lower-case token.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shake => "shake",
            Self::Lincs => "lincs",
            Self::Settle => "settle",
            Self::Rattle => "rattle",
        }
    }
}

impl std::fmt::Display for ConstraintAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a constraint set was refused or could not be met.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ConstraintError {
    /// The pairs, lengths or options do not describe a constraint set.
    #[error("malformed set: {what}")]
    Malformed {
        /// What is wrong.
        what: &'static str,
    },
    /// A pair names an atom past the end of the coordinates.
    #[error("pair {pair:?} is past the last of {atoms} atoms")]
    OutOfRange {
        /// The offending pair.
        pair: (usize, usize),
        /// Atoms in the coordinates.
        atoms: usize,
    },
    /// A mass is not positive and finite.
    #[error("mass of atom {atom} is not positive and finite")]
    Mass {
        /// Atom index.
        atom: usize,
    },
    /// The session cannot carry the set in this configuration.
    #[error("not supported: {what}")]
    Unsupported {
        /// The refused combination.
        what: &'static str,
    },
    /// The solver spent its derived iteration budget.
    #[error(
        "{algorithm} left relative residual {residual:e} on pair {pair:?} after {iterations} iterations"
    )]
    NotConverged {
        /// The solver that ran.
        algorithm: ConstraintAlgorithm,
        /// The pair with the largest residual.
        pair: (usize, usize),
        /// Its residual: `|d^2 - r^2| / (2 d^2)` on positions, the
        /// relative tangent residual on vectors.
        residual: f64,
        /// Iterations spent.
        iterations: usize,
    },
    /// A bond turned too far from its anchor direction for the solver:
    /// a right angle for SHAKE and the tangent solve, `|r| > sqrt(2) d`
    /// or a reversed projection for LINCS.
    #[error("{algorithm}: pair {pair:?} turned too far from its anchor direction")]
    Reversed {
        /// The solver that ran.
        algorithm: ConstraintAlgorithm,
        /// The pair.
        pair: (usize, usize),
    },
    /// The constraints of a block are linearly dependent at the anchor
    /// to within the rounding of their projection.
    #[error("the block with pair {pair:?} is linearly dependent at the anchor")]
    Dependent {
        /// First pair of the block.
        pair: (usize, usize),
    },
    /// LINCS was requested on a block whose expansion diverges.
    #[error("LINCS diverges on the block with pair {pair:?}: spectral radius {rho}")]
    LincsUnsafe {
        /// First pair of the block.
        pair: (usize, usize),
        /// Spectral radius of the coupling the expansion sums.
        rho: f64,
    },
    /// The trial moved a rigid triangle so far that SETTLE has no
    /// solution.
    #[error("SETTLE has no solution for the triangle on atoms {atoms:?}")]
    SettleDegenerate {
        /// Its atoms.
        atoms: [usize; 3],
    },
    /// A constrained atom reached a coordinate bound. Constrained
    /// coordinates are never clipped, so the box and the set act on
    /// different coordinates; a trial that takes a constrained atom to
    /// its box is refused instead.
    #[error("atom {atom} of a constraint reached its coordinate bounds")]
    Bound {
        /// The atom.
        atom: usize,
    },
    /// A coordinate, mass or residual is not finite.
    #[error("a coordinate or residual is not finite")]
    NonFinite,
}

/// What one return to the set did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConstraintReport {
    /// Blocks solved by SHAKE.
    pub shake: usize,
    /// Blocks solved by LINCS.
    pub lincs: usize,
    /// Blocks solved by SETTLE.
    pub settle: usize,
    /// Largest iteration count over the blocks.
    pub iterations: usize,
}

/// A rigid triangle: atoms `a`, `b`, `c` and the indices of the
/// constraints `ab`, `ac`, `bc`.
#[derive(Clone, Debug, PartialEq)]
struct Triangle {
    atoms: [usize; 3],
    edges: [usize; 3],
}

#[derive(Clone, Debug, PartialEq)]
struct Block {
    members: Vec<usize>,
    triangle: Option<Triangle>,
}

/// Distances `|x_i - x_j| = d` between atoms of packed `3 N` coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct DistanceConstraints {
    pairs: Vec<(usize, usize)>,
    lengths: Vec<f64>,
    masses: Option<Vec<f64>>,
    solver: ConstraintSolver,
    settle: bool,
    tolerance: f64,
    lincs_order: Option<usize>,
    blocks: Vec<Block>,
}

fn malformed(what: &'static str) -> Error {
    ConstraintError::Malformed { what }.into()
}

pub(crate) fn unsupported(what: &'static str) -> Error {
    ConstraintError::Unsupported { what }.into()
}

pub(crate) fn bond(x: ArrayView1<f64>, (i, j): (usize, usize)) -> [f64; 3] {
    [0, 1, 2].map(|c| x[3 * i + c] - x[3 * j + c])
}

pub(crate) fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `|| |v_i| + |v_j| ||_2`. Storing `v` moves `v_i - v_j` along a unit
/// vector by at most `U` times this.
fn reach(v: ArrayView1<f64>, (i, j): (usize, usize)) -> f64 {
    let s = [0, 1, 2].map(|c| v[3 * i + c].abs() + v[3 * j + c].abs());
    dot3(&s, &s).sqrt()
}

impl DistanceConstraints {
    /// Fix each pair at the given length, unit masses, [`ConstraintSolver::Auto`],
    /// SETTLE on rigid triangles, and a tolerance at the rounding floor.
    pub fn new(pairs: Vec<(usize, usize)>, lengths: Vec<f64>) -> Result<Self> {
        if pairs.len() != lengths.len() {
            return Err(malformed("one length per pair"));
        }
        if pairs.iter().any(|&(i, j)| i == j) {
            return Err(malformed("a pair needs two atoms"));
        }
        if lengths.iter().any(|d| !(d.is_finite() && *d > 0.0)) {
            return Err(malformed("lengths must be positive and finite"));
        }
        let mut keys: Vec<_> = pairs.iter().map(|&(i, j)| (i.min(j), i.max(j))).collect();
        keys.sort_unstable();
        if keys.windows(2).any(|w| w[0] == w[1]) {
            return Err(malformed("a pair appears twice"));
        }
        let blocks = blocks(&pairs);
        for t in blocks.iter().filter_map(|b| b.triangle.as_ref()) {
            let [ab, ac, bc] = t.edges.map(|k| lengths[k]);
            let longest = ab.max(ac).max(bc);
            if 2.0 * longest >= ab + ac + bc {
                return Err(malformed("triangle lengths break the triangle inequality"));
            }
        }
        Ok(Self {
            pairs,
            lengths,
            masses: None,
            solver: ConstraintSolver::Auto,
            settle: true,
            tolerance: 0.0,
            lincs_order: None,
            blocks,
        })
    }

    /// Fix each pair at its length in `x`.
    pub fn at(pairs: Vec<(usize, usize)>, x: ArrayView1<f64>) -> Result<Self> {
        let atoms = x.len() / 3;
        if let Some(&pair) = pairs.iter().find(|&&(i, j)| i.max(j) >= atoms) {
            return Err(ConstraintError::OutOfRange { pair, atoms }.into());
        }
        let lengths = pairs.iter().map(|&p| dot3(&bond(x, p), &bond(x, p)).sqrt());
        let lengths = lengths.collect();
        Self::new(pairs, lengths)
    }

    /// Per-atom masses, one per atom of the coordinates the set is used
    /// with. Corrections and projections then use the metric `M^-1`.
    pub fn with_masses(mut self, masses: Vec<f64>) -> Result<Self> {
        if let Some(atom) = masses.iter().position(|m| !(m.is_finite() && *m > 0.0)) {
            return Err(ConstraintError::Mass { atom }.into());
        }
        self.masses = Some(masses);
        Ok(self)
    }

    /// Solver for blocks that SETTLE does not take.
    pub fn with_solver(mut self, solver: ConstraintSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Solve rigid triangles analytically with SETTLE (the default) or
    /// leave them to the solver.
    pub fn with_settle(mut self, settle: bool) -> Self {
        self.settle = settle;
        self
    }

    /// Relative tolerance on `|d^2 - r^2| / (2 d^2)`, the GROMACS
    /// `shake-tol` measure. Zero (the default) asks for the rounding floor
    /// of each constraint; a tolerance below the floor is raised to it.
    pub fn with_tolerance(mut self, tolerance: f64) -> Result<Self> {
        if !(tolerance.is_finite() && tolerance >= 0.0) {
            return Err(malformed("tolerance must be finite and not negative"));
        }
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Fix the LINCS expansion order (GROMACS `lincs-order`). `None`, the
    /// default, derives it per block from the coupling and the tolerance.
    pub fn with_lincs_order(mut self, order: Option<usize>) -> Self {
        self.lincs_order = order;
        self
    }

    /// Constrained atom pairs.
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Target length of each pair.
    pub fn lengths(&self) -> &[f64] {
        &self.lengths
    }

    /// Per-atom masses, `None` for unit masses.
    pub fn masses(&self) -> Option<&[f64]> {
        self.masses.as_deref()
    }

    /// Solver choice.
    pub fn solver(&self) -> ConstraintSolver {
        self.solver
    }

    /// Whether rigid triangles go to SETTLE.
    pub fn settle(&self) -> bool {
        self.settle
    }

    /// Requested relative tolerance; zero is the rounding floor.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Fixed LINCS order, `None` when derived.
    pub fn lincs_order(&self) -> Option<usize> {
        self.lincs_order
    }

    /// Whether the set fits `dim` packed coordinates.
    pub fn fits(&self, dim: usize) -> Result<()> {
        if !dim.is_multiple_of(3) {
            return Err(malformed("coordinates must be 3 N"));
        }
        let atoms = dim / 3;
        if let Some(&pair) = self.pairs.iter().find(|&&(i, j)| i.max(j) >= atoms) {
            return Err(ConstraintError::OutOfRange { pair, atoms }.into());
        }
        match &self.masses {
            Some(m) if m.len() != atoms => Err(malformed("one mass per atom")),
            _ => Ok(()),
        }
    }

    /// Atoms that appear in a pair.
    pub(crate) fn constrained_atoms(&self) -> Vec<usize> {
        let mut atoms: Vec<_> = self.pairs.iter().flat_map(|&(i, j)| [i, j]).collect();
        atoms.sort_unstable();
        atoms.dedup();
        atoms
    }

    fn inverse_mass(&self, atom: usize) -> f64 {
        self.masses.as_ref().map_or(1.0, |m| 1.0 / m[atom])
    }

    /// Largest relative residual `|d^2 - r^2| / (2 d^2)`, NaN if any is.
    pub fn residual(&self, x: ArrayView1<f64>) -> f64 {
        let e = (0..self.pairs.len()).map(|k| self.relative(x, k));
        e.fold(0.0, |m, e| if e > m || e.is_nan() { e } else { m })
    }

    /// Largest ratio of the relative residual to its tolerance, NaN if a
    /// residual is. At most one when every constraint is met.
    pub fn excess(&self, x: ArrayView1<f64>) -> f64 {
        let e = (0..self.pairs.len()).map(|k| self.relative(x, k) / self.tol(x, k));
        e.fold(0.0, |m, e| if e > m || e.is_nan() { e } else { m })
    }

    fn relative(&self, x: ArrayView1<f64>, k: usize) -> f64 {
        let r = bond(x, self.pairs[k]);
        let d = self.lengths[k];
        ((d * d - dot3(&r, &r)) / (2.0 * d * d)).abs()
    }

    /// The relative residual of constraint `k` costs `gamma(6) / 2` to
    /// evaluate (`gamma(5)` for `r^2`, `U` for `d^2`, a subtraction exact
    /// near the set) and `U reach / d` to store the coordinates. The
    /// tolerance is the larger of that floor and the requested one.
    fn tol(&self, x: ArrayView1<f64>, k: usize) -> f64 {
        let floor = GAMMA6 / 2.0 + U * reach(x, self.pairs[k]) / self.lengths[k];
        floor.max(self.tolerance)
    }

    /// Return `x` to the set from `anchor`, the point `x` was stepped
    /// from. Corrections run along the anchor bonds. On an error `x` is
    /// left as it was.
    pub fn constrain(
        &self,
        anchor: ArrayView1<f64>,
        x: &mut Array1<f64>,
    ) -> Result<ConstraintReport> {
        self.fits(x.len())?;
        Tangent::new(self, anchor)?.constrain(x)
    }

    /// Project the vector `v` at `at` onto the tangent space with the
    /// inverse-mass projector.
    pub fn project_vector(&self, at: ArrayView1<f64>, v: &mut Array1<f64>) -> Result<()> {
        self.fits(at.len())?;
        Tangent::new(self, at)?.vector(v)
    }

    /// Project the gradient `g` at `at`: `g - J^T lambda`, the covector
    /// whose `M^-1` image is tangent. Returns the multipliers `lambda`.
    pub fn project_gradient(&self, at: ArrayView1<f64>, g: &mut Array1<f64>) -> Result<Vec<f64>> {
        self.fits(at.len())?;
        Tangent::new(self, at)?.covector(g)
    }
}

/// Connected components of the constraint graph, members in input order.
fn blocks(pairs: &[(usize, usize)]) -> Vec<Block> {
    let atoms = pairs.iter().map(|&(i, j)| i.max(j) + 1).max().unwrap_or(0);
    let mut parent: Vec<usize> = (0..atoms).collect();
    fn root(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    for &(i, j) in pairs {
        let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
        parent[ri.max(rj)] = ri.min(rj);
    }
    let mut order: Vec<usize> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for (k, &(i, _)) in pairs.iter().enumerate() {
        let r = root(&mut parent, i);
        match order.iter().position(|&o| o == r) {
            Some(b) => members[b].push(k),
            None => {
                order.push(r);
                members.push(vec![k]);
            }
        }
    }
    members
        .into_iter()
        .map(|m| {
            let triangle = triangle(pairs, &m);
            Block {
                members: m,
                triangle,
            }
        })
        .collect()
}

/// Three constraints on three atoms.
fn triangle(pairs: &[(usize, usize)], members: &[usize]) -> Option<Triangle> {
    let [k0, k1, k2] = *members else {
        return None;
    };
    let (a, b) = pairs[k0];
    let touches = |k: usize, u: usize| pairs[k].0 == u || pairs[k].1 == u;
    let other = |k: usize| {
        if touches(k, a) {
            pairs[k].0 + pairs[k].1 - a
        } else {
            usize::MAX
        }
    };
    let c = other(k1);
    let c = if c == b || c == usize::MAX {
        other(k2)
    } else {
        c
    };
    if c == usize::MAX || c == a || c == b {
        return None;
    }
    let find = |u: usize, w: usize| {
        [k1, k2]
            .into_iter()
            .find(|&k| touches(k, u) && touches(k, w))
    };
    Some(Triangle {
        atoms: [a, b, c],
        edges: [k0, find(a, c)?, find(b, c)?],
    })
}

/// Per-block analysis at one point: the bonds there, the scaled coupling
/// `S = D^-1/2 J W J^T D^-1/2` (unit diagonal), its extreme eigenvalues
/// and Cholesky factor.
struct Plan {
    /// `sqrt(diag(J W J^T))`.
    scale: Vec<f64>,
    s: Vec<f64>,
    chol: Vec<f64>,
    lambda_min: f64,
    lambda_max: f64,
    /// Bits one projection solve gains: `log2(1 / (b kappa gamma(6)))`.
    bits: f64,
}

impl Plan {
    /// Spectral radius of `I - S`, the matrix the LINCS expansion sums.
    fn rho_lincs(&self) -> f64 {
        (1.0 - self.lambda_min).max(self.lambda_max - 1.0)
    }

    /// Gauss-Seidel contraction in the `S` norm for the unit-diagonal
    /// splitting `S = (I + L) + L^T`:
    /// `rho^2 <= 1 - lambda_min / ||I + L||_2^2` (the Xu-Zikatanov
    /// identity bounded through `sigma_min((I + L)^-1) = 1 / ||I + L||_2`),
    /// with the Frobenius norm as the computable bound of `||I + L||_2`.
    fn rho_gauss_seidel(&self) -> f64 {
        let b = self.scale.len();
        let mut frob2 = b as f64;
        for k in 0..b {
            for l in 0..k {
                frob2 += self.s[k * b + l] * self.s[k * b + l];
            }
        }
        (1.0 - self.lambda_min / frob2).max(0.0).sqrt()
    }
}

/// The set analysed at one point. Serves as the anchor of trial
/// corrections and as the tangent space of projections.
pub(crate) struct Tangent<'a> {
    set: &'a DistanceConstraints,
    at: Array1<f64>,
    bonds: Vec<[f64; 3]>,
    plans: Vec<Plan>,
}

impl<'a> Tangent<'a> {
    pub(crate) fn new(set: &'a DistanceConstraints, at: ArrayView1<f64>) -> Result<Self> {
        if !at.iter().all(|v| v.is_finite()) {
            return Err(ConstraintError::NonFinite.into());
        }
        let bonds: Vec<_> = set.pairs.iter().map(|&p| bond(at, p)).collect();
        let plans = set
            .blocks
            .iter()
            .map(|block| Self::plan(set, &bonds, block))
            .collect::<Result<_>>()?;
        Ok(Self {
            set,
            at: at.to_owned(),
            bonds,
            plans,
        })
    }

    /// `(J W J^T)_kl`: the inverse masses of the atoms pairs `k` and `l`
    /// share, signed by the ends they share them at, times `b_k . b_l`.
    fn coupling(set: &DistanceConstraints, k: usize, l: usize) -> f64 {
        let (p, q) = (set.pairs[k], set.pairs[l]);
        let mut w = 0.0;
        for (a, sa) in [(p.0, 1.0), (p.1, -1.0)] {
            for (c, sc) in [(q.0, 1.0), (q.1, -1.0)] {
                if a == c {
                    w += sa * sc * set.inverse_mass(a);
                }
            }
        }
        w
    }

    fn plan(set: &DistanceConstraints, bonds: &[[f64; 3]], block: &Block) -> Result<Plan> {
        let m = &block.members;
        let b = m.len();
        let first = set.pairs[m[0]];
        let mut a = vec![0.0; b * b];
        for (r, &k) in m.iter().enumerate() {
            for (c, &l) in m.iter().enumerate() {
                a[r * b + c] = Self::coupling(set, k, l) * dot3(&bonds[k], &bonds[l]);
            }
        }
        let scale: Vec<f64> = (0..b).map(|r| a[r * b + r].sqrt()).collect();
        if !scale.iter().all(|s| s.is_finite() && *s > 0.0) {
            return Err(ConstraintError::Dependent { pair: first }.into());
        }
        let s: Vec<f64> = (0..b * b)
            .map(|rc| a[rc] / (scale[rc / b] * scale[rc % b]))
            .collect();
        let (lambda_min, lambda_max) = linalg::extreme_eigenvalues(s.clone(), b);
        // A backward-stable solve with S gains log2(1 / (b kappa gamma(6)))
        // bits; a block where that is under one bit cannot be projected.
        let bits = (lambda_min / (b as f64 * lambda_max * GAMMA6)).log2();
        let chol = linalg::cholesky(&s, b);
        match chol {
            Some(chol) if bits >= 1.0 => Ok(Plan {
                scale,
                s,
                chol,
                lambda_min,
                lambda_max,
                bits,
            }),
            _ => Err(ConstraintError::Dependent { pair: first }.into()),
        }
    }

    pub(crate) fn point(&self) -> &Array1<f64> {
        &self.at
    }

    /// Project `v` with `Pi` (a vector) or `g` with `Pi^T` (a covector).
    /// Each pass solves `J W J^T lambda = J u` for `u = v` or `u = W g`
    /// and subtracts `W J^T lambda` or `J^T lambda`. A pass leaves a
    /// tangent residual of `b kappa gamma(6)` times the previous one, so
    /// refinement reaches the floor of the stored result within the bits
    /// to gain over the bits per pass.
    fn projection(&self, v: &mut Array1<f64>, covector: bool) -> Result<Vec<f64>> {
        let set = self.set;
        let w = |atom: usize| {
            if covector {
                set.inverse_mass(atom)
            } else {
                1.0
            }
        };
        let image = |v: &Array1<f64>| -> Array1<f64> {
            let mut u = v.clone();
            for a in 0..u.len() / 3 {
                let wa = w(a);
                (0..3).for_each(|c| u[3 * a + c] *= wa);
            }
            u
        };
        let mut lambda = vec![0.0; set.pairs.len()];
        let mut u = image(v);
        let mut mag = u.mapv(f64::abs);
        let mut budget: Option<usize> = None;
        let mut pass = 0;
        loop {
            let mut worst = (0.0_f64, 0);
            for (k, &p) in set.pairs.iter().enumerate() {
                let bk = &self.bonds[k];
                let along = dot3(&bond(u.view(), p), bk).abs() / dot3(bk, bk).sqrt();
                let ratio = along / (GAMMA6 * reach(mag.view(), p));
                if !ratio.is_finite() && along != 0.0 {
                    return Err(ConstraintError::NonFinite.into());
                }
                if ratio > worst.0 {
                    worst = (ratio, k);
                }
            }
            if worst.0 <= 1.0 {
                return Ok(lambda);
            }
            let bits = self
                .plans
                .iter()
                .map(|p| p.bits)
                .fold(f64::INFINITY, f64::min);
            let limit = *budget.get_or_insert((worst.0.log2() / bits).ceil() as usize + 1);
            if pass >= limit {
                return Err(ConstraintError::NotConverged {
                    algorithm: ConstraintAlgorithm::Rattle,
                    pair: set.pairs[worst.1],
                    residual: worst.0 * GAMMA6,
                    iterations: pass,
                }
                .into());
            }
            let mut correction = Array1::<f64>::zeros(v.len());
            for (block, plan) in set.blocks.iter().zip(&self.plans) {
                let m = &block.members;
                let b = m.len();
                let mut y: Vec<f64> = m
                    .iter()
                    .zip(&plan.scale)
                    .map(|(&k, s)| dot3(&bond(u.view(), set.pairs[k]), &self.bonds[k]) / s)
                    .collect();
                linalg::cholesky_solve(&plan.chol, b, &mut y);
                for ((&k, s), y) in m.iter().zip(&plan.scale).zip(&y) {
                    let mu = y / s;
                    lambda[k] += mu;
                    let (i, j) = set.pairs[k];
                    let wi = if covector { 1.0 } else { set.inverse_mass(i) };
                    let wj = if covector { 1.0 } else { set.inverse_mass(j) };
                    for c in 0..3 {
                        correction[3 * i + c] += wi * mu * self.bonds[k][c];
                        correction[3 * j + c] -= wj * mu * self.bonds[k][c];
                    }
                }
            }
            *v -= &correction;
            let du = image(&correction);
            mag += &du.mapv(f64::abs);
            u = image(v);
            pass += 1;
        }
    }

    /// `v <- Pi v`.
    pub(crate) fn vector(&self, v: &mut Array1<f64>) -> Result<()> {
        self.projection(v, false).map(|_| ())
    }

    /// `g <- Pi^T g`, returning the multipliers.
    pub(crate) fn covector(&self, g: &mut Array1<f64>) -> Result<Vec<f64>> {
        self.projection(g, true)
    }

    /// Return `x`, a trial stepped from this point, to the set. The
    /// blocks are solved on a copy, and `x` takes the copy only when
    /// every block is met, so an error leaves `x` as it was.
    pub(crate) fn constrain(&self, x: &mut Array1<f64>) -> Result<ConstraintReport> {
        if !x.iter().all(|v| v.is_finite()) {
            return Err(ConstraintError::NonFinite.into());
        }
        let mut trial = x.clone();
        let mut report = ConstraintReport::default();
        for (block, plan) in self.set.blocks.iter().zip(&self.plans) {
            let iterations = match (&block.triangle, self.set.settle) {
                (Some(t), true) => {
                    report.settle += 1;
                    self.settle(t, &mut trial)?;
                    self.shake(block, plan, &mut trial)?
                }
                _ => match self.choose(block, plan)? {
                    ConstraintAlgorithm::Lincs => {
                        report.lincs += 1;
                        self.lincs(block, plan, &mut trial)?
                    }
                    _ => {
                        report.shake += 1;
                        self.shake(block, plan, &mut trial)?
                    }
                },
            };
            report.iterations = report.iterations.max(iterations);
        }
        *x = trial;
        Ok(report)
    }

    fn choose(&self, block: &Block, plan: &Plan) -> Result<ConstraintAlgorithm> {
        let rho = plan.rho_lincs();
        match self.set.solver {
            ConstraintSolver::Shake => Ok(ConstraintAlgorithm::Shake),
            ConstraintSolver::Lincs if rho >= 1.0 => Err(ConstraintError::LincsUnsafe {
                pair: self.set.pairs[block.members[0]],
                rho,
            }
            .into()),
            ConstraintSolver::Lincs => Ok(ConstraintAlgorithm::Lincs),
            ConstraintSolver::Auto if rho < 1.0 && rho <= plan.rho_gauss_seidel() => {
                Ok(ConstraintAlgorithm::Lincs)
            }
            ConstraintSolver::Auto => Ok(ConstraintAlgorithm::Shake),
        }
    }

    /// Largest `relative / tol` over a block and the constraint where it
    /// is, NaN propagated.
    fn block_excess(&self, block: &Block, x: ArrayView1<f64>) -> (f64, usize) {
        let mut worst = (0.0, block.members[0]);
        for &k in &block.members {
            let e = self.set.relative(x, k) / self.set.tol(x, k);
            if e > worst.0 || e.is_nan() {
                worst = (e, k);
            }
        }
        worst
    }

    fn not_converged(
        &self,
        algorithm: ConstraintAlgorithm,
        x: ArrayView1<f64>,
        k: usize,
        iterations: usize,
    ) -> Error {
        ConstraintError::NotConverged {
            algorithm,
            pair: self.set.pairs[k],
            residual: self.set.relative(x, k),
            iterations,
        }
        .into()
    }

    /// Velocity of the curve `R(t) = constrain(x + t d)` at the trial
    /// `r = R(t)`: `R' = d - W J_x^T mu` with `J_r R' = 0`, an oblique
    /// block solve `(J_r W J_x^T) mu = J_r d`.
    pub(crate) fn curve_velocity(
        &self,
        r: ArrayView1<f64>,
        d: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let set = self.set;
        let mut out = d.clone();
        for block in &set.blocks {
            let m = &block.members;
            let b = m.len();
            let rb: Vec<_> = m.iter().map(|&k| bond(r, set.pairs[k])).collect();
            let mut c = vec![0.0; b * b];
            for (row, &k) in m.iter().enumerate() {
                for (col, &l) in m.iter().enumerate() {
                    c[row * b + col] =
                        Tangent::coupling(set, k, l) * dot3(&rb[row], &self.bonds[l]);
                }
            }
            let rhs = m
                .iter()
                .zip(&rb)
                .map(|(&k, r)| dot3(&bond(d.view(), set.pairs[k]), r))
                .collect();
            let reversed = || ConstraintError::Reversed {
                algorithm: ConstraintAlgorithm::Rattle,
                pair: set.pairs[m[0]],
            };
            let mu = linalg::lu_solve(c, rhs).ok_or_else(reversed)?;
            for (&k, mu) in m.iter().zip(&mu) {
                let (i, j) = set.pairs[k];
                let (wi, wj) = (set.inverse_mass(i), set.inverse_mass(j));
                for c in 0..3 {
                    out[3 * i + c] -= wi * mu * self.bonds[k][c];
                    out[3 * j + c] += wj * mu * self.bonds[k][c];
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
