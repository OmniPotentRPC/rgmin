//! Geometry preconditioners for atomistic descent.
//!
//! [`Exp`] is the universal exponential metric of Packwood et al. 2016.
//! [`Lindh`] is the model Hessian of Lindh et al. 1995. [`Ff`] is the
//! bonded reduced Hessian of Makri, Ortner, and Kermode 2019. All three
//! are sparse symmetric positive definite matrices, inverted by
//! preconditioned conjugate gradients and held between rebuilds by
//! [`needs_rebuild`]. [`select_precon`] documents which one applies.
//!
//! Because [`Exp`] has Kronecker form `P = A ⊗ I_3` over an `N × N`
//! scalar symmetric M-matrix `A`, `A` is stored once in scalar CSR form
//! and applied simultaneously to the three Cartesian components of each
//! atom, cutting CSR storage and matrix-vector bandwidth by 3×. Its PCG
//! inner preconditioner defaults to zero-fill incomplete Cholesky
//! (Meijerink and van der Vorst 1977) on fixed diagonal atom blocks of
//! `A`, with `A`-orthogonal Rayleigh-projected warm starts across
//! consecutive solves and stopping tolerance relative to the force norm
//! ([`DEFAULT_PRECON_RTOL`]). [`Lindh`] and [`Ff`] precondition PCG with
//! exact `3 × 3` per-atom symmetric diagonal blocks.
//!
//! Pair assembly is a cell list. Occupied bins are visited in
//! lexicographic order and the pairs are sorted by atom index. A
//! matrix-vector product writes each row from that row's own sum. Every
//! dot product in the solve reduces fixed chunks of 1024 along one
//! deterministic chunk-aligned binary tree with zero heap allocations.
//!
//! Packwood, Kermode, Mones, Bernstein, Woolley, Gould, Ortner, and
//! Csányi, *A universal preconditioner for simulating condensed phase
//! materials*, J. Chem. Phys. 144, 164109 (2016),
//! <https://doi.org/10.1063/1.4947024>.
//!
//! Makri, Ortner, and Kermode, *A preconditioning scheme for minimum
//! energy path finding methods*, J. Chem. Phys. 150, 094109 (2019),
//! <https://doi.org/10.1063/1.5064465>.
//!
//! Meijerink and van der Vorst, *An iterative solution method for linear
//! systems of which the coefficient matrix is a symmetric M-matrix*,
//! Math. Comp. 31, 148 (1977),
//! <https://doi.org/10.1090/S0025-5718-1977-0438681-4>.
//!
//! The caller supplies positions and a neighbour list (linkcell, vesin,
//! or any other producer). The matrix is the stabilized form of
//! Packwood et al. equation (16): neighbour weight
//! `c_ij = exp(-A (r_ij / r_nn - 1))`, off-diagonal blocks `-μ c_ij I_3`
//! (C order, xyz per atom), and diagonal
//! `μ (C_stab + sum_j c_ij) I_3`. `r_nn` is the maximum of the per-atom
//! nearest-neighbour distances. `μ` is equation (14), the Rayleigh
//! quotient of the sine probe (15) against this matrix at `μ = 1`.
//!
//! `P` is symmetric positive definite for `μ > 0` and `C_stab > 0`.

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

use std::mem::size_of_val;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::{Error, Result};

/// Default relative residual tolerance `||r||_2 / ||b||_2` for preconditioner
/// PCG solves (Packwood et al. 2016; Makri, Ortner, and Kermode 2019).
pub const DEFAULT_PRECON_RTOL: f64 = 1.0e-8;

/// Relaxed relative residual accepted when PCG reaches its iteration cap.
const FALLBACK_PRECON_RTOL: f64 = 1.0e-6;

/// Largest `||r_0||_2 / ||b||_2` ratio at which a Rayleigh-projected warm
/// start is retained over a cold zero start.
const WARM_START_MAX_RATIO: f64 = 0.95;

/// Sine-probe amplitude in units of `r_nn` (Packwood et al. 2016, Eq. 15).
const SINE_PROBE_SCALE: f64 = 1.0e-2;

/// Fixed diagonal atom-block size for [`ExpPcKind::BlockIc0`].
const IC_BLOCK_ATOMS: usize = 1024;

/// Smallest atom count that runs multi-threaded when the caller explicitly sets `threads >= 2`.
const EXPLICIT_PAR_MIN_ATOMS: usize = 32;

/// Atom chunk width for parallel matrix-vector and block-Jacobi rows (`64` atoms = `192` floats).
const MATVEC_ATOM_CHUNK: usize = 64;

/// Pair chunk width for parallel bond-distance and exponential-weight evaluation.
const PAIR_PAR_CHUNK: usize = 512;

#[cfg(test)]
const TEST_HOLD_R_NN: f64 = 1_000_000_000.0;

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

/// Inner preconditioner used by PCG when inverting [`Exp`].
///
/// Because the scalar graph matrix `A` of [`Exp`] (`P = A ⊗ I_3`) has
/// positive diagonal entries, nonpositive off-diagonal entries
/// `-μ c_ij`, and strict diagonal dominance `μ C_stab > 0`, `A` is a
/// symmetric M-matrix and every principal block of `A` admits a
/// breakdown-free zero-fill incomplete Cholesky factor (Meijerink and
/// van der Vorst 1977).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExpPcKind {
    /// Zero-fill incomplete Cholesky on fixed diagonal atom blocks of `A`.
    #[default]
    BlockIc0,
    /// Diagonal Jacobi scaling on `A`.
    Jacobi,
}

/// Parameters of the exponential preconditioner.
///
/// Defaults are the paper's generic choice: `A = 3`, `C_stab = 0.1`,
/// `μ` and `r_nn` computed from the geometry, no extra cutoff beyond
/// the supplied list, [`DEFAULT_PRECON_RTOL`] relative PCG tolerance,
/// warm starts enabled, and [`ExpPcKind::BlockIc0`] preconditioning.
#[non_exhaustive]
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
    /// Relative residual tolerance `||r||_2 / ||b||_2` of the PCG solve.
    pub rtol: f64,
    /// Reuse the previous PCG solution via Rayleigh projection when helpful.
    pub warm_start: bool,
    /// Inner preconditioner for the PCG solve on `P = A ⊗ I_3`.
    pub pc: ExpPcKind,
    /// Worker threads for preconditioner assembly and PCG. `None` is serial
    /// (`1` thread) unless called inside a multi-thread Rayon pool; `Some(1)`
    /// forces serial execution.
    pub threads: Option<usize>,
}

with_setters!(ExpParams {
    with_a => a: val f64,
    with_c_stab => c_stab: val f64,
    with_mu => mu: opt f64,
    with_r_nn => r_nn: opt f64,
    with_r_cut => r_cut: opt f64,
    with_recalc_mu => recalc_mu: val bool,
    with_lengths => lengths: opt [f64; 3],
    with_rtol => rtol: val f64,
    with_warm_start => warm_start: val bool,
    with_pc => pc: val ExpPcKind,
    with_threads => threads: opt usize,
});

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
            rtol: DEFAULT_PRECON_RTOL,
            warm_start: true,
            pc: ExpPcKind::BlockIc0,
            threads: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct Exec {
    threads: Option<usize>,
    #[cfg(feature = "par")]
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl Exec {
    pub(super) fn new(threads: Option<usize>) -> Result<Self> {
        if matches!(threads, Some(0)) {
            return Err(Error::Precon {
                what: "thread count is zero",
            });
        }
        #[cfg(feature = "par")]
        let pool = match threads {
            Some(t) if t >= 2 => {
                if rayon::current_thread_index().is_some() && rayon::current_num_threads() == t {
                    None
                } else {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(t)
                        .build()
                        .ok()
                        .map(Arc::new)
                }
            }
            _ => None,
        };
        Ok(Self {
            threads,
            #[cfg(feature = "par")]
            pool,
        })
    }

    #[inline]
    pub(super) fn install<R>(&self, f: impl FnOnce() -> R + Send) -> R
    where
        R: Send,
    {
        #[cfg(feature = "par")]
        if let Some(pool) = &self.pool {
            return pool.install(f);
        }
        f()
    }

    /// Worker count for a kernel over `atoms` atoms with ambient gate `gate`.
    /// Returns `1` whenever execution stays serial.
    #[inline]
    pub(super) fn workers(&self, atoms: usize, gate: usize) -> usize {
        match self.threads {
            Some(t) if t <= 1 => 1,
            Some(t) => {
                if atoms >= EXPLICIT_PAR_MIN_ATOMS {
                    t
                } else {
                    1
                }
            }
            None => {
                if gate == usize::MAX || atoms < gate {
                    return 1;
                }
                #[cfg(feature = "par")]
                {
                    if rayon::current_thread_index().is_some() {
                        let t = rayon::current_num_threads();
                        if t >= 2 {
                            return t;
                        }
                    }
                }
                1
            }
        }
    }

    /// Worker count for an outer multi-image batch, independent of atom gate.
    #[inline]
    fn batch_workers(&self) -> usize {
        match self.threads {
            Some(t) if t <= 1 => 1,
            Some(t) => t,
            None => {
                #[cfg(feature = "par")]
                {
                    if rayon::current_thread_index().is_some() {
                        let t = rayon::current_num_threads();
                        if t >= 2 {
                            return t;
                        }
                    }
                }
                1
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Ic0Subblock {
    atoms: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<u16>,
    val: Vec<f32>,
    inv_d: Vec<f64>,
}

#[derive(Clone, Debug)]
struct BlockIc0 {
    blocks: Vec<Ic0Subblock>,
}

/// Scalar `N × N` CSR graph representing `P = A ⊗ I_3` on `3 N` coordinates.
#[derive(Clone, Debug)]
struct ScalarCsr {
    n_atoms: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    val: Vec<f64>,
    diag: Vec<f64>,
    ic0: Option<BlockIc0>,
    apply_gate: usize,
    solve_gate: usize,
}

#[derive(Clone, Debug)]
struct Csr {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    val: Vec<f64>,
    diag: Vec<f64>,
    /// Optional per-atom symmetric `3 × 3` inverse `[m00, m01, m02, m11, m12, m22]`.
    inv_block3: Option<Vec<[f64; 6]>>,
    /// Atom count at which `apply` may use the `par` pool.
    apply_gate: usize,
    /// Atom count at which the solve's products may use the `par` pool.
    solve_gate: usize,
}

#[derive(Clone, Debug)]
enum SparseOp {
    Iso3(ScalarCsr),
    General(Csr),
}

#[derive(Default, Debug)]
struct PcgWork {
    x: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    last_x: Vec<f64>,
    has_last: bool,
}

/// Sparse SPD metric. `apply` is `P v`. `solve` is `P x = rhs`.
#[derive(Clone, Debug)]
pub(crate) struct Metric {
    op: Arc<SparseOp>,
    work: Arc<Mutex<PcgWork>>,
    rtol: f64,
    warm_start: bool,
    exec: Exec,
}

impl Metric {
    fn from_scalar_csr(csr: ScalarCsr, rtol: f64, warm_start: bool, exec: Exec) -> Self {
        Self {
            op: Arc::new(SparseOp::Iso3(csr)),
            work: Arc::new(Mutex::new(PcgWork::default())),
            rtol,
            warm_start,
            exec,
        }
    }

    fn from_general_csr(csr: Csr, rtol: f64, warm_start: bool, exec: Exec) -> Self {
        Self {
            op: Arc::new(SparseOp::General(csr)),
            work: Arc::new(Mutex::new(PcgWork::default())),
            rtol,
            warm_start,
            exec,
        }
    }

    pub(crate) fn set_exec(&mut self, exec: Exec) {
        self.exec = exec;
    }

    pub(crate) fn dim(&self) -> usize {
        self.op.dim()
    }

    pub(crate) fn stored_nonzeros(&self) -> usize {
        self.op.stored_nonzeros()
    }

    pub(crate) fn csr_bytes(&self) -> usize {
        self.op.csr_bytes()
    }

    pub(crate) fn apply(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let n = self.dim();
        let mut y = vec![0.0; n];
        if v.len() == n {
            let owned;
            let slice = match v.as_slice() {
                Some(s) => s,
                None => {
                    owned = v.to_vec();
                    &owned
                }
            };
            self.exec.install(|| {
                self.op.matvec(slice, &mut y, &self.exec);
            });
        }
        Array1::from_vec(y)
    }

    pub(crate) fn solve(&self, rhs: ArrayView1<f64>) -> Option<Array1<f64>> {
        self.solve_counted(rhs).map(|(x, _)| x)
    }

    pub(crate) fn solve_counted(&self, rhs: ArrayView1<f64>) -> Option<(Array1<f64>, usize)> {
        if rhs.len() != self.dim() {
            return None;
        }
        let owned;
        let rhs_slice = match rhs.as_slice() {
            Some(s) => s,
            None => {
                owned = rhs.to_vec();
                &owned
            }
        };
        let exec = &self.exec;
        let res = exec.install(|| match self.work.try_lock() {
            Ok(mut guard) => solve_pcg_op(
                &self.op,
                rhs_slice,
                self.rtol,
                self.warm_start,
                self.warm_start,
                exec,
                &mut guard,
            ),
            Err(_) => {
                let mut local = PcgWork::default();
                solve_pcg_op(
                    &self.op, rhs_slice, self.rtol, false, false, exec, &mut local,
                )
            }
        });
        res.map(|(x, iters)| (Array1::from_vec(x), iters))
    }

    pub(crate) fn solve_iterations(&self, rhs: ArrayView1<f64>) -> Option<usize> {
        if rhs.len() != self.dim() {
            return None;
        }
        let owned;
        let rhs_slice = match rhs.as_slice() {
            Some(s) => s,
            None => {
                owned = rhs.to_vec();
                &owned
            }
        };
        let exec = &self.exec;
        exec.install(|| {
            let mut local = PcgWork::default();
            solve_pcg_op(
                &self.op, rhs_slice, self.rtol, false, false, exec, &mut local,
            )
            .map(|(_, n)| n)
        })
    }

    pub(crate) fn to_dense(&self) -> Array2<f64> {
        self.op.to_dense()
    }

    /// `μ P`. Lindh and FF assemble the model at scale 1 and apply the
    /// estimated energy scale here, the same place Exp folds `μ` into
    /// the matrix.
    pub(crate) fn scaled(&self, mu: f64) -> Self {
        let op = match &*self.op {
            SparseOp::Iso3(csr) => {
                let mut next = csr.clone();
                for v in &mut next.val {
                    *v *= mu;
                }
                for d in &mut next.diag {
                    *d *= mu;
                }
                if let Some(ic0) = &mut next.ic0 {
                    let root = mu.sqrt();
                    let root_f32 = root as f32;
                    for block in &mut ic0.blocks {
                        for lv in &mut block.val {
                            *lv *= root_f32;
                        }
                        for inv in &mut block.inv_d {
                            *inv /= root;
                        }
                    }
                }
                SparseOp::Iso3(next)
            }
            SparseOp::General(csr) => {
                let mut next = csr.clone();
                for v in &mut next.val {
                    *v *= mu;
                }
                for d in &mut next.diag {
                    *d *= mu;
                }
                if let Some(blocks) = &mut next.inv_block3 {
                    let inv_mu = 1.0 / mu;
                    for b in blocks {
                        for entry in b {
                            *entry *= inv_mu;
                        }
                    }
                }
                SparseOp::General(next)
            }
        };
        Self {
            op: Arc::new(op),
            work: Arc::new(Mutex::new(PcgWork::default())),
            rtol: self.rtol,
            warm_start: self.warm_start,
            exec: self.exec.clone(),
        }
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
        if !(params.rtol.is_finite() && params.rtol > 0.0) {
            return Err(Error::Precon {
                what: "rtol is not positive",
            });
        }
        let exec = Exec::new(params.threads)?;
        let natoms = atom_count(positions)?;
        if u32::try_from(natoms).is_err() {
            return Err(Error::Precon {
                what: "atom count exceeds u32 index range",
            });
        }
        let csr = exec.install(|| -> Result<ScalarCsr> {
            let (pairs, anchors) = bonds(natoms, positions, neighbours, params.r_cut, &exec)?;
            Ok(build_scalar_csr(
                natoms,
                &pairs,
                &anchors,
                mu,
                params.a,
                r_nn,
                params.c_stab,
                params.pc,
                &exec,
            ))
        })?;
        Ok(Self {
            mu,
            r_nn,
            metric: Metric::from_scalar_csr(csr, params.rtol, params.warm_start, exec),
        })
    }

    /// Assemble one [`Exp`] metric per image in deterministic slice order.
    ///
    /// Uses [`ExpParams::threads`] (or the ambient Rayon pool when unset,
    /// staying serial at one thread) to assemble independent images
    /// concurrently without nested thread-pool oversubscription.
    pub fn assemble_batch(
        images: &[(ArrayView1<f64>, &Neighbours)],
        params: &ExpParams,
        mu: f64,
        r_nn: f64,
    ) -> Result<Vec<Self>> {
        let outer = Exec::new(params.threads)?;
        let workers = outer.batch_workers();
        if workers <= 1 || images.len() < 2 {
            let mut out = Vec::with_capacity(images.len());
            for &(pos, neigh) in images {
                out.push(Self::assemble(pos, neigh, params, mu, r_nn)?);
            }
            return Ok(out);
        }
        let mut inner = params.clone();
        inner.threads = Some(1);
        let built = ordered_map_batch(&outer, images.len(), workers, |idx| {
            let (pos, neigh) = images[idx];
            Self::assemble(pos, neigh, &inner, mu, r_nn)
        });
        let mut out = Vec::with_capacity(built.len());
        for item in built {
            out.push(item?);
        }
        Ok(out)
    }

    /// Solve `P_m x_m = rhs_m` across a slice of per-image metrics in
    /// deterministic slice order.
    ///
    /// `threads` `None` stays serial unless called inside a multi-thread
    /// Rayon pool; `Some(1)` forces serial execution; `Some(0)` is an error.
    pub fn solve_batch(
        metrics: &[Self],
        rhs: &[ArrayView1<f64>],
        threads: Option<usize>,
    ) -> Result<Option<Vec<Array1<f64>>>> {
        if metrics.len() != rhs.len() {
            return Err(Error::Precon {
                what: "batch metric and rhs counts differ",
            });
        }
        let outer = Exec::new(threads)?;
        let workers = outer.batch_workers();
        if workers <= 1 || metrics.len() < 2 {
            let mut out = Vec::with_capacity(metrics.len());
            for (m, &b) in metrics.iter().zip(rhs.iter()) {
                match m.solve(b) {
                    Some(x) => out.push(x),
                    None => return Ok(None),
                }
            }
            return Ok(Some(out));
        }
        let solved = ordered_map_batch(&outer, metrics.len(), workers, |idx| {
            metrics[idx].solve(rhs[idx])
        });
        let mut out = Vec::with_capacity(solved.len());
        for item in solved {
            match item {
                Some(x) => out.push(x),
                None => return Ok(None),
            }
        }
        Ok(Some(out))
    }

    pub(crate) fn metric(&self) -> &Metric {
        &self.metric
    }

    pub(crate) fn set_exec(&mut self, exec: Exec) {
        self.metric.set_exec(exec);
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
            r_nn: TEST_HOLD_R_NN,
            metric: Metric::from_general_csr(
                Csr {
                    n,
                    row_ptr,
                    col_idx,
                    val,
                    diag: diag.to_vec(),
                    inv_block3: None,
                    apply_gate: N_PAR_EXP_APPLY,
                    solve_gate: N_PAR_EXP_SOLVE,
                },
                DEFAULT_PRECON_RTOL,
                true,
                Exec::default(),
            ),
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

    /// Number of stored scalar-graph CSR entries (`2 * n_bonds + N`).
    #[must_use]
    pub fn stored_nonzeros(&self) -> usize {
        self.metric.stored_nonzeros()
    }

    /// Heap bytes occupied by the scalar-graph CSR arrays.
    #[must_use]
    pub fn csr_bytes(&self) -> usize {
        self.metric.csr_bytes()
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

    /// Solve `P x = rhs`, returning both the solution and the number of
    /// Krylov iterations taken (reflecting any cached warm start).
    #[must_use]
    pub fn solve_counted(&self, rhs: ArrayView1<f64>) -> Option<(Array1<f64>, usize)> {
        self.metric.solve_counted(rhs)
    }

    /// Cold-start PCG iterations for one solve. Does not mutate the warm-start cache.
    #[must_use]
    pub fn solve_iterations(&self, rhs: ArrayView1<f64>) -> Option<usize> {
        self.metric.solve_iterations(rhs)
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
        let _ = Exec::new(params.threads)?;
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
        let _ = Exec::new(params.threads)?;
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
        if !(params.rtol.is_finite() && params.rtol > 0.0) {
            return Err(Error::Precon {
                what: "rtol is not positive",
            });
        }
        if let Some(mu) = params.mu
            && !(mu.is_finite() && mu > 0.0)
        {
            return Err(Error::Precon {
                what: "mu is not positive",
            });
        }
        let (r_nn, r_nn_pending) = match params.r_nn {
            Some(r) => (r, false),
            None => (1.0, true),
        };
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

    pub(crate) fn set_threads(&mut self, threads: Option<usize>) -> Result<()> {
        let exec = Exec::new(threads)?;
        self.params.threads = threads;
        if let Some(matrix) = &mut self.matrix {
            matrix.set_exec(exec);
        }
        Ok(())
    }

    #[cfg(feature = "capi")]
    pub(crate) fn threads(&self) -> Option<usize> {
        self.params.threads
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
        let mu = *self.mu.get_or_insert(1.0);
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
                r_nn: Some(TEST_HOLD_R_NN),
                recalc_mu: false,
                ..ExpParams::default()
            },
            mu: Some(1.0),
            r_nn: TEST_HOLD_R_NN,
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

    pub(crate) fn set_threads(&mut self, threads: Option<usize>) -> Result<()> {
        match self {
            Self::Exp(state) => state.set_threads(threads),
            Self::Lindh(state) => state.set_threads(threads),
            Self::Ff(state) => state.set_threads(threads),
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
            match neighbours.r.get(k) {
                Some(&val) => val,
                None => f64::NAN,
            }
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
    let usable = n.is_multiple_of(3) && r_nn.is_finite() && r_nn > 0.0;
    if !usable {
        return v;
    }
    let span = position_span(positions);
    let mut len = match lengths {
        Some(l) => l,
        None => span,
    };
    for a in 0..3 {
        if !(len[a].is_finite() && len[a] > 0.0) {
            len[a] = span[a];
        }
    }
    let scale = SINE_PROBE_SCALE * r_nn;
    let natoms = n / 3;
    for i in 0..natoms {
        for a in 0..3 {
            if !(len[a].is_finite() && len[a] > 0.0) {
                continue;
            }
            let s = crate::portable::sin(positions[3 * i + a] / len[a]);
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
    // Reject cancellation at machine precision relative to the dot-product terms.
    let den_scale = v.iter().zip(&pv).map(|(a, b)| (a * b).abs()).sum::<f64>();
    if !(den.is_finite()
        && den_scale.is_finite()
        && den.abs() > f64::EPSILON * den_scale
        && num.is_finite())
    {
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
    let pv = metric.apply(v.view());
    let den = v.dot(&pv);
    // Reject cancellation at machine precision relative to the dot-product terms.
    let den_scale = v.iter().zip(&pv).map(|(a, b)| (a * b).abs()).sum::<f64>();
    if !(den.is_finite()
        && den_scale.is_finite()
        && den.abs() > f64::EPSILON * den_scale
        && num.is_finite())
    {
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
    if let Some(r) = nearest_neighbour_distance(positions, neighbours) {
        return Ok(r);
    }
    Ok(1.0)
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
    exec: &Exec,
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
    let mut raw: Vec<(usize, usize, usize)> = Vec::with_capacity(neighbours.i.len());
    let mut strictly_sorted = true;
    let mut prev = (0usize, 0usize);
    let mut have_prev = false;
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
        if have_prev && (a, b) <= prev {
            strictly_sorted = false;
        }
        prev = (a, b);
        have_prev = true;
        raw.push((a, b, k));
    }
    if !strictly_sorted {
        raw.sort_unstable();
        raw.dedup_by(|x, y| x.0 == y.0 && x.1 == y.1);
    }
    let pairs = if exec.workers(natoms, N_PAR_ASSEMBLE) >= 2 && raw.len() >= 2 * PAIR_PAR_CHUNK {
        let nchunks = raw.len().div_ceil(PAIR_PAR_CHUNK);
        let parts = ordered_map_exec(exec, nchunks, natoms, N_PAR_ASSEMBLE, |c| {
            let start = c * PAIR_PAR_CHUNK;
            let end = (start + PAIR_PAR_CHUNK).min(raw.len());
            let mut local = Vec::with_capacity(end - start);
            for &(a, b, k) in &raw[start..end] {
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
                local.push((a, b, r));
            }
            local
        });
        let total: usize = parts.iter().map(Vec::len).sum();
        let mut merged = Vec::with_capacity(total);
        for part in parts {
            merged.extend(part);
        }
        merged
    } else {
        let mut pairs = Vec::with_capacity(raw.len());
        for (a, b, k) in raw {
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
        pairs
    };
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
    mu * crate::portable::exp(-a * (r / r_nn - 1.0))
}

/// Assemble the scalar `N × N` graph CSR `A` (`P = A ⊗ I_3`) directly from
/// lexicographically sorted unique pairs `(i < j)`.
fn build_scalar_csr(
    natoms: usize,
    pairs: &[(usize, usize, f64)],
    anchors: &[(usize, f64)],
    mu: f64,
    a: f64,
    r_nn: f64,
    c_stab: f64,
    pc: ExpPcKind,
    exec: &Exec,
) -> ScalarCsr {
    let mut diag = vec![mu * c_stab; natoms];
    let mut lower_deg = vec![0usize; natoms];
    let mut upper_deg = vec![0usize; natoms];
    let mut weights = vec![0.0; pairs.len()];
    par_chunks_mut_exec(
        exec,
        &mut weights,
        PAIR_PAR_CHUNK,
        natoms,
        N_PAR_ASSEMBLE,
        |c, out| {
            let base = c * PAIR_PAR_CHUNK;
            for (offset, slot) in out.iter_mut().enumerate() {
                let (_, _, r) = pairs[base + offset];
                *slot = weight(mu, a, r, r_nn);
            }
        },
    );
    for (&(i, j, _), &c) in pairs.iter().zip(weights.iter()) {
        diag[i] += c;
        diag[j] += c;
        upper_deg[i] += 1;
        lower_deg[j] += 1;
    }
    for &(atom, r) in anchors {
        diag[atom] += weight(mu, a, r, r_nn);
    }
    let nnz = 2 * pairs.len() + natoms;
    let mut row_ptr = Vec::with_capacity(natoms + 1);
    row_ptr.push(0);
    for i in 0..natoms {
        let deg = lower_deg[i] + 1 + upper_deg[i];
        row_ptr.push(row_ptr[i] + deg);
    }
    let mut col_idx = vec![0u32; nnz];
    let mut val = vec![0.0; nnz];
    let mut lower_cur = vec![0usize; natoms];
    let mut upper_cur = vec![0usize; natoms];
    let mut diag_pos = vec![0usize; natoms];
    for i in 0..natoms {
        lower_cur[i] = row_ptr[i];
        let dp = row_ptr[i] + lower_deg[i];
        diag_pos[i] = dp;
        col_idx[dp] = i as u32;
        val[dp] = diag[i];
        upper_cur[i] = dp + 1;
    }
    for (&(i, j, _), &c) in pairs.iter().zip(weights.iter()) {
        let up = upper_cur[i];
        col_idx[up] = j as u32;
        val[up] = -c;
        upper_cur[i] = up + 1;

        let lo = lower_cur[j];
        col_idx[lo] = i as u32;
        val[lo] = -c;
        lower_cur[j] = lo + 1;
    }
    let ic0 = match pc {
        ExpPcKind::BlockIc0 => {
            build_block_ic0(natoms, &row_ptr, &diag_pos, &col_idx, &val, &diag, exec)
        }
        ExpPcKind::Jacobi => None,
    };
    ScalarCsr {
        n_atoms: natoms,
        row_ptr,
        col_idx,
        val,
        diag,
        ic0,
        apply_gate: N_PAR_EXP_APPLY,
        solve_gate: N_PAR_EXP_SOLVE,
    }
}

fn factor_ic0_subblock(
    b_start: usize,
    b_end: usize,
    row_ptr: &[usize],
    diag_pos: &[usize],
    col_idx: &[u32],
    val: &[f64],
    diag: &[f64],
) -> Option<Ic0Subblock> {
    let m = b_end - b_start;
    if u16::try_from(m).is_err() {
        return None;
    }
    let mut l_row_ptr = Vec::with_capacity(m + 1);
    l_row_ptr.push(0);
    let mut cap = 0usize;
    for i in b_start..b_end {
        for &c in &col_idx[row_ptr[i]..diag_pos[i]] {
            if (c as usize) >= b_start {
                cap += 1;
            }
        }
    }
    let mut l_col: Vec<u16> = Vec::with_capacity(cap);
    let mut l_val: Vec<f32> = Vec::with_capacity(cap);
    let mut inv_d = vec![0.0_f64; m];
    let mut work = vec![0.0_f64; m];

    for li in 0..m {
        let i = b_start + li;
        let row_start = l_col.len();
        for k in row_ptr[i]..diag_pos[i] {
            let j = col_idx[k] as usize;
            if j >= b_start {
                let lj = j - b_start;
                work[lj] = val[k];
                l_col.push(lj as u16);
                l_val.push(0.0);
            }
        }
        let row_end = l_col.len();
        l_row_ptr.push(row_end);

        for pos in row_start..row_end {
            let lj = usize::from(l_col[pos]);
            let mut dot_lj = 0.0_f64;
            for p in l_row_ptr[lj]..l_row_ptr[lj + 1] {
                let lk = usize::from(l_col[p]);
                dot_lj += work[lk] * f64::from(l_val[p]);
            }
            let lij = (work[lj] - dot_lj) * inv_d[lj];
            if !lij.is_finite() {
                return None;
            }
            let lij_f32 = lij as f32;
            l_val[pos] = lij_f32;
            work[lj] = f64::from(lij_f32);
        }

        let mut diag_sq = diag[i];
        for pos in row_start..row_end {
            let lj = usize::from(l_col[pos]);
            let w = work[lj];
            diag_sq -= w * w;
            work[lj] = 0.0;
        }
        if !(diag_sq.is_finite() && diag_sq > 0.0) {
            return None;
        }
        inv_d[li] = 1.0 / diag_sq.sqrt();
    }

    Some(Ic0Subblock {
        atoms: m,
        row_ptr: l_row_ptr,
        col_idx: l_col,
        val: l_val,
        inv_d,
    })
}

fn build_block_ic0(
    natoms: usize,
    row_ptr: &[usize],
    diag_pos: &[usize],
    col_idx: &[u32],
    val: &[f64],
    diag: &[f64],
    exec: &Exec,
) -> Option<BlockIc0> {
    let nblocks = natoms.div_ceil(IC_BLOCK_ATOMS);
    let parts = ordered_map_exec(exec, nblocks, natoms, N_PAR_ASSEMBLE, |b| {
        let b_start = b * IC_BLOCK_ATOMS;
        let b_end = (b_start + IC_BLOCK_ATOMS).min(natoms);
        factor_ic0_subblock(b_start, b_end, row_ptr, diag_pos, col_idx, val, diag)
    });
    let mut blocks = Vec::with_capacity(nblocks);
    for item in parts {
        blocks.push(item?);
    }
    Some(BlockIc0 { blocks })
}

#[inline]
fn invert_spd3(d00: f64, d01: f64, d02: f64, d11: f64, d12: f64, d22: f64) -> Option<[f64; 6]> {
    if !(d00.is_finite() && d00 > 0.0) {
        return None;
    }
    let minor2 = d00 * d11 - d01 * d01;
    if !(minor2.is_finite() && minor2 > 0.0) {
        return None;
    }
    let c00 = d11 * d22 - d12 * d12;
    let c01 = d02 * d12 - d01 * d22;
    let c02 = d01 * d12 - d02 * d11;
    let c11 = d00 * d22 - d02 * d02;
    let c12 = d01 * d02 - d00 * d12;
    let c22 = minor2;
    let det = d00 * c00 + d01 * c01 + d02 * c02;
    // A determinant scales cubically, so compare it to epsilon at the block's scale.
    let scale = d00
        .abs()
        .max(d01.abs())
        .max(d02.abs())
        .max(d11.abs())
        .max(d12.abs())
        .max(d22.abs());
    let det_floor = f64::EPSILON * scale * scale * scale;
    if !(det.is_finite() && det > det_floor) {
        return None;
    }
    let inv = 1.0 / det;
    let out = [
        c00 * inv,
        c01 * inv,
        c02 * inv,
        c11 * inv,
        c12 * inv,
        c22 * inv,
    ];
    out.iter().all(|v| v.is_finite()).then_some(out)
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

    fn finish(self, apply_gate: usize, solve_gate: usize, exec: &Exec) -> Csr {
        let n = 3 * self.n_atoms;
        let mut diag = vec![0.0; n];
        for i in 0..n {
            let s = self.row_ptr[i];
            let e = self.row_ptr[i + 1];
            if let Ok(k) = self.col_idx[s..e].binary_search(&i) {
                diag[i] = self.val[s + k];
            }
        }
        let partners = &self.partners;
        let row_ptr = &self.row_ptr;
        let val = &self.val;
        let nchunks = self.n_atoms.div_ceil(MATVEC_ATOM_CHUNK);
        let parts = ordered_map_exec(exec, nchunks, self.n_atoms, N_PAR_ASSEMBLE, |c| {
            let start = c * MATVEC_ATOM_CHUNK;
            let end = (start + MATVEC_ATOM_CHUNK).min(self.n_atoms);
            let mut local = Vec::with_capacity(end - start);
            for atom in start..end {
                let Ok(p) = partners[atom].binary_search(&atom) else {
                    return None;
                };
                let r0 = row_ptr[3 * atom] + 3 * p;
                let r1 = row_ptr[3 * atom + 1] + 3 * p;
                let r2 = row_ptr[3 * atom + 2] + 3 * p;
                local.push(invert_spd3(
                    val[r0],
                    val[r0 + 1],
                    val[r0 + 2],
                    val[r1 + 1],
                    val[r1 + 2],
                    val[r2 + 2],
                )?);
            }
            Some(local)
        });
        let mut inv_blocks = Vec::with_capacity(self.n_atoms);
        let mut all_spd = true;
        for part in parts {
            match part {
                Some(v) => inv_blocks.extend(v),
                None => {
                    all_spd = false;
                    break;
                }
            }
        }
        let inv_block3 = all_spd.then_some(inv_blocks);
        Csr {
            n,
            row_ptr: self.row_ptr,
            col_idx: self.col_idx,
            val: self.val,
            diag,
            inv_block3,
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

/// Atom counts for the ambient `par` loops when `threads` is unset.
///
/// Serial (`threads = 1`) is the default outside an installed multi-thread
/// Rayon pool, and `Some(1)` forces serial execution inside one. When
/// `threads >= 2` is passed explicitly, multi-threaded kernels engage at
/// [`EXPLICIT_PAR_MIN_ATOMS`].
///
/// Lindh and bonded-metric matvec ambient atom gate.
pub const N_PAR_LINDH_APPLY: usize = 512;
/// Lindh and bonded-metric PCG solve ambient atom gate.
pub const N_PAR_LINDH_SOLVE: usize = 512;
/// Exp scalar-graph matvec ambient atom gate.
pub const N_PAR_EXP_APPLY: usize = 512;
/// Exp block-IC(0) PCG solve ambient atom gate.
pub const N_PAR_EXP_SOLVE: usize = 512;
/// Preconditioner assembly ambient atom gate.
pub const N_PAR_ASSEMBLE: usize = 512;

/// Fixed length of one dot-product chunk. The same chunks are used
/// with and without the `par` feature.
const DOT_CHUNK: usize = 1024;
/// Smallest chunk count split across parallel tasks (`4 * 1024 = 4096` elements).
const DOT_PAR_MIN_CHUNKS: usize = 4;

/// Map `0..n` and keep index order under `exec`.
pub(super) fn ordered_map_exec<T, F>(
    exec: &Exec,
    n: usize,
    atoms: usize,
    gate: usize,
    f: F,
) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    let workers = exec.workers(atoms, gate);
    if workers <= 1 || n < 2 {
        return (0..n).map(f).collect();
    }
    ordered_map_batch(exec, n, workers, f)
}

fn ordered_map_batch<T, F>(exec: &Exec, n: usize, workers: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    if workers <= 1 || n < 2 {
        return (0..n).map(f).collect();
    }
    #[cfg(feature = "par")]
    {
        exec.install(|| {
            use rayon::prelude::*;
            (0..n).into_par_iter().map(f).collect()
        })
    }
    #[cfg(not(feature = "par"))]
    {
        let _ = exec;
        let nw = workers.min(n);
        let span = n.div_ceil(nw);
        let parts = std::thread::scope(|scope| {
            let f_ref = &f;
            let mut handles = Vec::with_capacity(nw);
            for w in 0..nw {
                let start = w * span;
                let end = (start + span).min(n);
                if start >= end {
                    break;
                }
                handles.push(scope.spawn(move || (start..end).map(f_ref).collect::<Vec<T>>()));
            }
            let mut joined = Vec::with_capacity(handles.len());
            for h in handles {
                match h.join() {
                    Ok(v) => joined.push(v),
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            }
            joined
        });
        let mut out = Vec::with_capacity(n);
        for part in parts {
            out.extend(part);
        }
        out
    }
}

fn par_chunks_mut_exec<T, F>(
    exec: &Exec,
    slice: &mut [T],
    chunk_len: usize,
    atoms: usize,
    gate: usize,
    f: F,
) where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    if slice.is_empty() {
        return;
    }
    let nchunks = slice.len().div_ceil(chunk_len);
    let workers = exec.workers(atoms, gate);
    if workers <= 1 || nchunks < 2 {
        for (c, sub) in slice.chunks_mut(chunk_len).enumerate() {
            f(c, sub);
        }
        return;
    }
    #[cfg(feature = "par")]
    {
        exec.install(|| {
            use rayon::prelude::*;
            slice
                .par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(c, sub)| f(c, sub));
        });
    }
    #[cfg(not(feature = "par"))]
    {
        let _ = exec;
        let nw = workers.min(nchunks);
        let chunks_per = nchunks.div_ceil(nw);
        std::thread::scope(|scope| {
            let f_ref = &f;
            let mut rest = slice;
            let mut base_chunk = 0usize;
            while !rest.is_empty() {
                let take_elems = (chunks_per * chunk_len).min(rest.len());
                let (head, tail) = rest.split_at_mut(take_elems);
                let chunk0 = base_chunk;
                scope.spawn(move || {
                    for (local, sub) in head.chunks_mut(chunk_len).enumerate() {
                        f_ref(chunk0 + local, sub);
                    }
                });
                base_chunk += take_elems.div_ceil(chunk_len);
                rest = tail;
            }
        });
    }
}

#[inline]
fn scalar_row_apply(a: &ScalarCsr, x_atoms: &[[f64; 3]], atom: usize) -> [f64; 3] {
    let start = a.row_ptr[atom];
    let end = a.row_ptr[atom + 1];
    let cols = &a.col_idx[start..end];
    let vals = &a.val[start..end];
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for (&col_u32, &v) in cols.iter().zip(vals.iter()) {
        if let Some(xj) = x_atoms.get(col_u32 as usize) {
            s0 += v * xj[0];
            s1 += v * xj[1];
            s2 += v * xj[2];
        }
    }
    [s0, s1, s2]
}

fn matvec_scalar(a: &ScalarCsr, x: &[f64], y: &mut [f64], gate: usize, exec: &Exec) {
    let (x_atoms, _) = x.as_chunks::<3>();
    let (y_atoms, _) = y.as_chunks_mut::<3>();
    par_chunks_mut_exec(
        exec,
        y_atoms,
        MATVEC_ATOM_CHUNK,
        a.n_atoms,
        gate,
        |c, chunk| {
            let atom0 = c * MATVEC_ATOM_CHUNK;
            for (offset, out) in chunk.iter_mut().enumerate() {
                *out = scalar_row_apply(a, x_atoms, atom0 + offset);
            }
        },
    );
}

fn apply_ic0_subblock(sub: &Ic0Subblock, r_sub: &[f64], z_sub: &mut [f64]) {
    let m = sub.atoms;
    for li in 0..m {
        let r_off = 3 * li;
        let mut acc0 = r_sub[r_off];
        let mut acc1 = r_sub[r_off + 1];
        let mut acc2 = r_sub[r_off + 2];
        for p in sub.row_ptr[li]..sub.row_ptr[li + 1] {
            let lj_off = 3 * usize::from(sub.col_idx[p]);
            let lv = f64::from(sub.val[p]);
            acc0 -= lv * z_sub[lj_off];
            acc1 -= lv * z_sub[lj_off + 1];
            acc2 -= lv * z_sub[lj_off + 2];
        }
        let inv = sub.inv_d[li];
        z_sub[r_off] = acc0 * inv;
        z_sub[r_off + 1] = acc1 * inv;
        z_sub[r_off + 2] = acc2 * inv;
    }
    for li in (0..m).rev() {
        let z_off = 3 * li;
        let inv = sub.inv_d[li];
        let zi0 = z_sub[z_off] * inv;
        let zi1 = z_sub[z_off + 1] * inv;
        let zi2 = z_sub[z_off + 2] * inv;
        z_sub[z_off] = zi0;
        z_sub[z_off + 1] = zi1;
        z_sub[z_off + 2] = zi2;
        for p in sub.row_ptr[li]..sub.row_ptr[li + 1] {
            let lj_off = 3 * usize::from(sub.col_idx[p]);
            let lv = f64::from(sub.val[p]);
            z_sub[lj_off] -= lv * zi0;
            z_sub[lj_off + 1] -= lv * zi1;
            z_sub[lj_off + 2] -= lv * zi2;
        }
    }
}

fn precond_scalar(a: &ScalarCsr, r: &[f64], z: &mut [f64], gate: usize, exec: &Exec) {
    if let Some(ic0) = &a.ic0 {
        let block_coords = 3 * IC_BLOCK_ATOMS;
        par_chunks_mut_exec(exec, z, block_coords, a.n_atoms, gate, |b, z_sub| {
            let start = b * block_coords;
            let end = start + z_sub.len();
            apply_ic0_subblock(&ic0.blocks[b], &r[start..end], z_sub);
        });
        return;
    }
    let (z_atoms, _) = z.as_chunks_mut::<3>();
    let (r_atoms, _) = r.as_chunks::<3>();
    par_chunks_mut_exec(
        exec,
        z_atoms,
        MATVEC_ATOM_CHUNK,
        a.n_atoms,
        gate,
        |c, z_chunk| {
            let atom0 = c * MATVEC_ATOM_CHUNK;
            for (offset, z_atom) in z_chunk.iter_mut().enumerate() {
                let atom = atom0 + offset;
                let inv = 1.0 / a.diag[atom];
                let r_atom = r_atoms[atom];
                *z_atom = [r_atom[0] * inv, r_atom[1] * inv, r_atom[2] * inv];
            }
        },
    );
}

fn row_dot(a: &Csr, x: &[f64], i: usize) -> f64 {
    let mut s = 0.0;
    for k in a.row_ptr[i]..a.row_ptr[i + 1] {
        s += a.val[k] * x[a.col_idx[k]];
    }
    s
}

fn matvec(a: &Csr, x: &[f64], y: &mut [f64], gate: usize, exec: &Exec) {
    let atoms = a.n / 3;
    let row_chunk = 3 * MATVEC_ATOM_CHUNK;
    par_chunks_mut_exec(exec, y, row_chunk, atoms, gate, |c, y_chunk| {
        let row0 = c * row_chunk;
        for (offset, yi) in y_chunk.iter_mut().enumerate() {
            *yi = row_dot(a, x, row0 + offset);
        }
    });
}

fn precond_general(a: &Csr, r: &[f64], z: &mut [f64], gate: usize, exec: &Exec) {
    let atoms = a.n / 3;
    if let Some(blocks) = &a.inv_block3 {
        let (z_atoms, _) = z.as_chunks_mut::<3>();
        let (r_atoms, _) = r.as_chunks::<3>();
        par_chunks_mut_exec(
            exec,
            z_atoms,
            MATVEC_ATOM_CHUNK,
            atoms,
            gate,
            |c, z_chunk| {
                let atom0 = c * MATVEC_ATOM_CHUNK;
                for (offset, z_atom) in z_chunk.iter_mut().enumerate() {
                    let atom = atom0 + offset;
                    let [r0, r1, r2] = r_atoms[atom];
                    let inv = &blocks[atom];
                    *z_atom = [
                        inv[0] * r0 + inv[1] * r1 + inv[2] * r2,
                        inv[1] * r0 + inv[3] * r1 + inv[4] * r2,
                        inv[2] * r0 + inv[4] * r1 + inv[5] * r2,
                    ];
                }
            },
        );
        return;
    }
    par_chunks_mut_exec(exec, z, DOT_CHUNK, atoms, gate, |c, z_chunk| {
        let i0 = c * DOT_CHUNK;
        for (offset, zi) in z_chunk.iter_mut().enumerate() {
            let i = i0 + offset;
            *zi = r[i] / a.diag[i];
        }
    });
}

impl SparseOp {
    fn dim(&self) -> usize {
        match self {
            Self::Iso3(s) => 3 * s.n_atoms,
            Self::General(g) => g.n,
        }
    }

    fn solve_gate(&self) -> usize {
        match self {
            Self::Iso3(s) => s.solve_gate,
            Self::General(g) => g.solve_gate,
        }
    }

    fn stored_nonzeros(&self) -> usize {
        match self {
            Self::Iso3(s) => s.val.len(),
            Self::General(g) => g.val.len(),
        }
    }

    fn csr_bytes(&self) -> usize {
        match self {
            Self::Iso3(s) => {
                size_of_val(s.row_ptr.as_slice())
                    + size_of_val(s.col_idx.as_slice())
                    + size_of_val(s.val.as_slice())
                    + size_of_val(s.diag.as_slice())
            }
            Self::General(g) => {
                size_of_val(g.row_ptr.as_slice())
                    + size_of_val(g.col_idx.as_slice())
                    + size_of_val(g.val.as_slice())
                    + size_of_val(g.diag.as_slice())
            }
        }
    }

    fn valid_diag(&self) -> bool {
        match self {
            Self::Iso3(s) => s.diag.iter().all(|d| d.is_finite() && *d > 0.0),
            Self::General(g) => g.diag.iter().all(|d| d.is_finite() && *d > 0.0),
        }
    }

    fn matvec(&self, x: &[f64], y: &mut [f64], exec: &Exec) {
        match self {
            Self::Iso3(s) => matvec_scalar(s, x, y, s.apply_gate, exec),
            Self::General(g) => matvec(g, x, y, g.apply_gate, exec),
        }
    }

    fn matvec_for_solve(&self, x: &[f64], y: &mut [f64], exec: &Exec) {
        match self {
            Self::Iso3(s) => matvec_scalar(s, x, y, s.solve_gate, exec),
            Self::General(g) => matvec(g, x, y, g.solve_gate, exec),
        }
    }

    fn precond(&self, r: &[f64], z: &mut [f64], exec: &Exec) {
        match self {
            Self::Iso3(s) => precond_scalar(s, r, z, s.solve_gate, exec),
            Self::General(g) => precond_general(g, r, z, g.solve_gate, exec),
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Iso3(s) => {
                let n = 3 * s.n_atoms;
                let mut a = Array2::zeros((n, n));
                for i in 0..s.n_atoms {
                    for k in s.row_ptr[i]..s.row_ptr[i + 1] {
                        let j = s.col_idx[k] as usize;
                        let v = s.val[k];
                        for d in 0..3 {
                            a[(3 * i + d, 3 * j + d)] = v;
                        }
                    }
                }
                a
            }
            Self::General(g) => {
                let mut a = Array2::zeros((g.n, g.n));
                for i in 0..g.n {
                    for k in g.row_ptr[i]..g.row_ptr[i + 1] {
                        a[(i, g.col_idx[k])] = g.val[k];
                    }
                }
                a
            }
        }
    }
}

#[inline]
fn dot_leaf(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let blocks = n / 8;
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut s4 = 0.0;
    let mut s5 = 0.0;
    let mut s6 = 0.0;
    let mut s7 = 0.0;
    for k in 0..blocks {
        let base = 8 * k;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
        s4 += a[base + 4] * b[base + 4];
        s5 += a[base + 5] * b[base + 5];
        s6 += a[base + 6] * b[base + 6];
        s7 += a[base + 7] * b[base + 7];
    }
    let mut acc = ((s0 + s4) + (s1 + s5)) + ((s2 + s6) + (s3 + s7));
    for i in (8 * blocks)..n {
        acc += a[i] * b[i];
    }
    acc
}

fn dot_tree_serial(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    if len <= DOT_CHUNK {
        return dot_leaf(a, b);
    }
    let nchunks = len.div_ceil(DOT_CHUNK);
    let mid = (nchunks / 2) * DOT_CHUNK;
    dot_tree_serial(&a[..mid], &b[..mid]) + dot_tree_serial(&a[mid..], &b[mid..])
}

#[cfg(feature = "par")]
fn dot_tree_rayon(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    if len <= DOT_CHUNK {
        return dot_leaf(a, b);
    }
    let nchunks = len.div_ceil(DOT_CHUNK);
    if nchunks < DOT_PAR_MIN_CHUNKS {
        return dot_tree_serial(a, b);
    }
    let mid = (nchunks / 2) * DOT_CHUNK;
    let (left, right) = rayon::join(
        || dot_tree_rayon(&a[..mid], &b[..mid]),
        || dot_tree_rayon(&a[mid..], &b[mid..]),
    );
    left + right
}

#[cfg(not(feature = "par"))]
fn dot_tree_scoped(a: &[f64], b: &[f64], budget: usize) -> f64 {
    let len = a.len();
    if len <= DOT_CHUNK {
        return dot_leaf(a, b);
    }
    let nchunks = len.div_ceil(DOT_CHUNK);
    if budget <= 1 || nchunks < DOT_PAR_MIN_CHUNKS {
        return dot_tree_serial(a, b);
    }
    let mid = (nchunks / 2) * DOT_CHUNK;
    let right_budget = budget / 2;
    let left_budget = budget - right_budget;
    std::thread::scope(|scope| {
        let right_handle = scope.spawn(|| dot_tree_scoped(&a[mid..], &b[mid..], right_budget));
        let left = dot_tree_scoped(&a[..mid], &b[..mid], left_budget);
        let right = match right_handle.join() {
            Ok(v) => v,
            Err(payload) => std::panic::resume_unwind(payload),
        };
        left + right
    })
}

/// Allocation-free deterministic dot product used by preconditioner PCG.
///
/// Leaves of length `1024` are reduced with an eight-lane accumulator tree
/// and combined along a fixed chunk-aligned binary split, so serial and
/// multi-threaded `par` builds produce identical bits with zero heap
/// allocations. Mismatched or empty slices return `0.0`.
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let exec = Exec::default();
    dot_exec(&exec, a, b, N_PAR_EXP_SOLVE)
}

fn dot_exec(exec: &Exec, a: &[f64], b: &[f64], gate: usize) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let nchunks = a.len().div_ceil(DOT_CHUNK);
    let atoms = a.len() / 3;
    let workers = exec.workers(atoms, gate);
    if workers <= 1 || nchunks < 2 {
        return dot_tree_serial(a, b);
    }
    #[cfg(feature = "par")]
    {
        exec.install(|| dot_tree_rayon(a, b))
    }
    #[cfg(not(feature = "par"))]
    {
        dot_tree_scoped(a, b, workers)
    }
}

fn nrm2_exec(exec: &Exec, a: &[f64], gate: usize) -> f64 {
    dot_exec(exec, a, a, gate).sqrt()
}

fn pcg_axpy2(
    exec: &Exec,
    atoms: usize,
    gate: usize,
    alpha: f64,
    p: &[f64],
    ap: &[f64],
    x: &mut [f64],
    r: &mut [f64],
) {
    let nchunks = x.len().div_ceil(DOT_CHUNK);
    let workers = exec.workers(atoms, gate);
    if workers <= 1 || nchunks < 2 {
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        return;
    }
    #[cfg(feature = "par")]
    {
        exec.install(|| {
            use rayon::prelude::*;
            x.par_chunks_mut(DOT_CHUNK)
                .zip(r.par_chunks_mut(DOT_CHUNK))
                .zip(p.par_chunks(DOT_CHUNK))
                .zip(ap.par_chunks(DOT_CHUNK))
                .for_each(|(((xc, rc), pc), apc)| {
                    for i in 0..xc.len() {
                        xc[i] += alpha * pc[i];
                        rc[i] -= alpha * apc[i];
                    }
                });
        });
    }
    #[cfg(not(feature = "par"))]
    {
        let nw = workers.min(nchunks);
        let elems_per = nchunks.div_ceil(nw) * DOT_CHUNK;
        std::thread::scope(|scope| {
            let mut xr = x;
            let mut rr = r;
            let mut off = 0usize;
            while !xr.is_empty() {
                let take = elems_per.min(xr.len());
                let (xh, xt) = xr.split_at_mut(take);
                let (rh, rt) = rr.split_at_mut(take);
                let ps = &p[off..off + take];
                let aps = &ap[off..off + take];
                scope.spawn(move || {
                    for i in 0..xh.len() {
                        xh[i] += alpha * ps[i];
                        rh[i] -= alpha * aps[i];
                    }
                });
                off += take;
                xr = xt;
                rr = rt;
            }
        });
    }
}

fn pcg_xpay(exec: &Exec, atoms: usize, gate: usize, beta: f64, z: &[f64], p: &mut [f64]) {
    par_chunks_mut_exec(exec, p, DOT_CHUNK, atoms, gate, |c, pc| {
        let base = c * DOT_CHUNK;
        let zc = &z[base..base + pc.len()];
        for i in 0..pc.len() {
            pc[i] = zc[i] + beta * pc[i];
        }
    });
}

fn record_warm(work: &mut PcgWork, record_last: bool) {
    if record_last {
        if work.last_x.len() != work.x.len() {
            work.last_x.resize(work.x.len(), 0.0);
        }
        work.last_x.copy_from_slice(&work.x);
        work.has_last = true;
    }
}

fn solve_pcg_op(
    op: &SparseOp,
    b: &[f64],
    rtol: f64,
    use_warm: bool,
    record_last: bool,
    exec: &Exec,
    work: &mut PcgWork,
) -> Option<(Vec<f64>, usize)> {
    let n = op.dim();
    let atoms = n / 3;
    let gate = op.solve_gate();
    if b.len() != n {
        return None;
    }
    let bnorm = nrm2_exec(exec, b, gate);
    if bnorm == 0.0 {
        if record_last {
            work.has_last = false;
        }
        return Some((vec![0.0; n], 0));
    }
    if !op.valid_diag() {
        return None;
    }

    work.x.resize(n, 0.0);
    work.r.resize(n, 0.0);
    work.z.resize(n, 0.0);
    work.p.resize(n, 0.0);
    work.ap.resize(n, 0.0);

    let eff_rtol = if rtol.is_finite() && rtol > 0.0 {
        rtol
    } else {
        DEFAULT_PRECON_RTOL
    };
    let tol = eff_rtol * bnorm;

    let mut rnorm = bnorm;
    let mut warmed = false;
    if use_warm && work.has_last && work.last_x.len() == n {
        op.matvec_for_solve(&work.last_x, &mut work.ap, exec);
        let u_au = dot_exec(exec, &work.last_x, &work.ap, gate);
        let b_u = dot_exec(exec, b, &work.last_x, gate);
        if u_au.is_finite() && u_au > 0.0 && b_u.is_finite() {
            let gamma = b_u / u_au;
            if gamma.is_finite() {
                for i in 0..n {
                    work.x[i] = gamma * work.last_x[i];
                    work.r[i] = b[i] - gamma * work.ap[i];
                }
                let rnorm0 = nrm2_exec(exec, &work.r, gate);
                if rnorm0.is_finite() && rnorm0 <= tol {
                    if !work.x.iter().all(|v| v.is_finite()) {
                        return None;
                    }
                    let out = work.x.clone();
                    record_warm(work, record_last);
                    return Some((out, 0));
                }
                if rnorm0.is_finite() && rnorm0 < WARM_START_MAX_RATIO * bnorm {
                    rnorm = rnorm0;
                    warmed = true;
                }
            }
        }
    }
    if !warmed {
        work.x.fill(0.0);
        work.r.copy_from_slice(b);
    }

    if rnorm <= tol {
        let out = work.x.clone();
        record_warm(work, record_last);
        return finite_vec(out).map(|v| (v, 0));
    }

    op.precond(&work.r, &mut work.z, exec);
    work.p.copy_from_slice(&work.z);
    let mut rz = dot_exec(exec, &work.r, &work.z, gate);
    if !(rz.is_finite() && rz > 0.0) {
        return None;
    }

    let maxit = (8 * n).max(64);
    let mut iters = 0usize;
    for _ in 0..maxit {
        iters += 1;
        op.matvec_for_solve(&work.p, &mut work.ap, exec);
        let pap = dot_exec(exec, &work.p, &work.ap, gate);
        if !(pap.is_finite() && pap > 0.0) {
            return None;
        }
        let alpha = rz / pap;
        if !alpha.is_finite() {
            return None;
        }
        pcg_axpy2(
            exec,
            atoms,
            gate,
            alpha,
            &work.p,
            &work.ap,
            &mut work.x,
            &mut work.r,
        );
        rnorm = nrm2_exec(exec, &work.r, gate);
        if rnorm <= tol {
            if !work.x.iter().all(|v| v.is_finite()) {
                return None;
            }
            let out = work.x.clone();
            record_warm(work, record_last);
            return Some((out, iters));
        }
        op.precond(&work.r, &mut work.z, exec);
        let rz_new = dot_exec(exec, &work.r, &work.z, gate);
        if !(rz_new.is_finite() && rz_new > 0.0) {
            return None;
        }
        let beta = rz_new / rz;
        pcg_xpay(exec, atoms, gate, beta, &work.z, &mut work.p);
        rz = rz_new;
    }

    let fallback_tol = FALLBACK_PRECON_RTOL.max(eff_rtol) * bnorm;
    if rnorm <= fallback_tol && work.x.iter().all(|v| v.is_finite()) {
        let out = work.x.clone();
        record_warm(work, record_last);
        return Some((out, iters));
    }
    None
}

#[cfg(test)]
fn solve_pcg(a: &Csr, b: &[f64]) -> Option<(Vec<f64>, usize)> {
    let op = SparseOp::General(a.clone());
    let exec = Exec::default();
    let mut work = PcgWork::default();
    solve_pcg_op(&op, b, 1.0e-12, false, false, &exec, &mut work)
}

fn finite_vec(x: Vec<f64>) -> Option<Vec<f64>> {
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Spread of two directional curvatures, `max/min`.
///
/// A non-positive or non-finite curvature returns infinity so the gate
/// stays shut. Water's bond Rayleigh quotient sits far above the
/// intermolecular one; a condensed metal or Lennard-Jones cluster does
/// not.
#[must_use]
pub fn curvature_spread(a: f64, b: f64) -> f64 {
    if !(a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0) {
        return f64::INFINITY;
    }
    a.max(b) / a.min(b)
}

/// Starting gate. A spread above this turns the exponential metric off.
/// Chosen so a stiff bond against a soft mode (ratios of tens) falls
/// back, and a ratio near one stays on. Not fit to the held-out set.
pub const EXP_GATE_MAX_SPREAD: f64 = 6.0;

/// `true` when the exponential preconditioner is allowed.
///
/// The safe answer is `false`: a failed probe, a non-positive
/// curvature, or a spread above `max_spread` leaves the scalar
/// initial Hessian in place.
#[must_use]
pub fn exp_gate_open(c_force: f64, c_bond: f64, max_spread: f64) -> bool {
    if !(max_spread.is_finite() && max_spread > 1.0) {
        return false;
    }
    curvature_spread(c_force, c_bond) <= max_spread
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
                v[3 * i + a] = scale * crate::portable::sin(pos[3 * i + a] / 4.0);
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
            inv_block3: None,
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
            hash, 0x75ec5f2a232e41f3,
            "committed hash drifted: iters {iters} hash {hash:#x}"
        );
    }

    #[test]
    fn exp_weight_and_sine_probe_hash_the_portable_libm() {
        // Scalar libm. The digest does not depend on the thread count
        // or on AVX2 versus AVX-512, because those kernels are not used.
        let (pos, neigh) = two_atoms();
        let params = ExpParams {
            lengths: Some([4.0, 4.0, 4.0]),
            ..ExpParams::default()
        };
        let matrix = Exp::assemble(pos.view(), &neigh, &params, 1.7, 1.5).unwrap();
        let probe = sine_probe(pos.view(), 1.5, Some([4.0, 4.0, 4.0]));
        let mut bits = matrix.apply(pos.view()).to_vec();
        bits.extend(probe.iter().copied());
        bits.push(weight(1.7, 3.0, 1.2, 1.5));
        let hash = fnv1a(&bits);
        assert_eq!(hash, 0x33a0_3fc0_17cf_ffab, "hash {hash:#x}");
    }

    #[test]
    fn the_gate_shuts_on_a_stiff_bond_and_on_a_failed_probe() {
        assert!(!exp_gate_open(2.0, 80.0, EXP_GATE_MAX_SPREAD));
        assert!(exp_gate_open(4.0, 5.5, EXP_GATE_MAX_SPREAD));
        assert!(!exp_gate_open(f64::NAN, 1.0, EXP_GATE_MAX_SPREAD));
        assert!(!exp_gate_open(-1.0, 2.0, EXP_GATE_MAX_SPREAD));
        assert!((curvature_spread(2.0, 8.0) - 4.0).abs() < 1e-15);
    }
}
