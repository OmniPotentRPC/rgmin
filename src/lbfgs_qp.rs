//! L-BFGS direction, HiGHS only for the feasible set.
//!
//! The two-loop recursion is the unconstrained minimizer of the L-BFGS
//! quadratic model (Nocedal-Wright 7.4). Forming the dense compact
//! Hessian and handing it to HiGHS is slower and, on a tiny accepted
//! step, indefinite: `Highs_run` does not return.
//!
//! Constrained step: keep the two-loop direction and scale it (eOn
//! `maxAtomMotionAppliedV`, same idea as a CFL limit) so it fits the
//! trust region and box. Packed layouts also subtract the per-axis
//! mean. Arbitrary equalities still go to HiGHS as
//! `min 1/2 ||p - d||^2` with `Q = I`.
//!
//! Huangfu and Hall, *Parallelizing the dual revised simplex method*,
//! <https://doi.org/10.1007/s12532-017-0130-5>.

use highs::{HighsModelStatus, RowProblem, Sense};
use highs_sys::{Highs_passHessian, HighsInt, STATUS_OK};
use ndarray::{Array1, Array2, ArrayView1};

use crate::error::{Error, Result};
use crate::lbfgs::Lbfgs;

/// Pin OpenMP to one thread exactly once, before any HiGHS solve.
///
/// The setter mutated the process environment from inside library calls,
/// which is a data race the moment two threads reach a solve together:
/// getenv in one thread against setenv in another is undefined behaviour
/// in glibc. One write through a `Once`, ordered before the first solve
/// on any thread, and the option calls below keep HiGHS itself serial
/// regardless.
fn serialise_openmp_once() {
    static OMP: std::sync::Once = std::sync::Once::new();
    OMP.call_once(|| unsafe {
        std::env::set_var("OMP_NUM_THREADS", "1");
    });
}

/// One side of a coordinate box. Length 1 is uniform; length `n` is
/// per-coordinate. `None` on [`HighsStep::lo`] / [`HighsStep::hi`] is
/// unbounded on that side.
pub fn box_side_at(side: &Option<Vec<f64>>, k: usize) -> Option<f64> {
    match side.as_ref() {
        None => None,
        Some(v) if v.len() == 1 => Some(v[0]),
        Some(v) => v.get(k).copied(),
    }
}

/// Bounds and equalities on one L-BFGS model step.
#[derive(Clone, Debug, Default)]
pub struct HighsStep {
    /// L_inf trust radius on the step. `None` is unbounded.
    pub trust: Option<f64>,
    /// Box lower bound on coordinates of `x + p`.
    /// Length 1 is uniform; length `n` is per-coordinate. `None` is unbounded.
    pub lo: Option<Vec<f64>>,
    /// Box upper bound on coordinates of `x + p`.
    /// Length 1 is uniform; length `n` is per-coordinate. `None` is unbounded.
    pub hi: Option<Vec<f64>>,
    /// Linear equalities `a · p = rhs`.
    pub equalities: Vec<(Vec<(usize, f64)>, f64)>,
    /// Packed `(n_atoms, dim)`: enforce `sum_i p[i * dim + h] = 0` per axis.
    /// This is a mean subtract, not a QP.
    pub center_axes: Option<(usize, usize)>,
}

impl HighsStep {
    fn has_box(&self) -> bool {
        self.trust.is_some() || self.lo.is_some() || self.hi.is_some()
    }

    fn needs_qp(&self) -> bool {
        !self.equalities.is_empty()
    }
}

impl Lbfgs {
    /// L-BFGS direction at `x` with gradient `g`, projected if needed.
    pub fn highs_step(&self, x: ArrayView1<f64>, g: ArrayView1<f64>) -> Result<Array1<f64>> {
        let opts = self.highs.as_ref().ok_or_else(|| {
            Error::Highs("Lbfgs.highs is None; set HighsStep before highs_step".into())
        })?;
        if x.len() != g.len() {
            return Err(Error::Dim {
                got: x.len(),
                dim: g.len(),
            });
        }
        let d = self.direction(g);
        if !opts.has_box() && !opts.needs_qp() && opts.center_axes.is_none() {
            return Ok(d);
        }
        let mut p = d;
        if let Some((n_atoms, dim)) = opts.center_axes {
            project_center_scale(&mut p, x, opts, n_atoms, dim);
        } else if opts.has_box() {
            scale_to_bounds(&mut p, x, opts);
        }
        if !opts.needs_qp() {
            return Ok(p);
        }
        match project_qp(&p, x, opts) {
            Ok(q) => Ok(q),
            Err(_) => Ok(p),
        }
    }
}

fn project_qp(d: &Array1<f64>, x: ArrayView1<f64>, opts: &HighsStep) -> Result<Array1<f64>> {
    let n = d.len();
    let mut pb = RowProblem::default();
    let mut cols = Vec::with_capacity(n);
    for k in 0..n {
        let (lo, hi) = column_bounds(k, x, opts);
        // min 1/2 ||p - d||^2  <=>  min -d · p + 1/2 p^T p
        cols.push(pb.add_column(-d[k], lo..=hi));
    }
    for (coeffs, rhs) in &opts.equalities {
        let row: Vec<_> = coeffs.iter().map(|(i, a)| (cols[*i], *a)).collect();
        pb.add_row(*rhs..=*rhs, &row);
    }

    let mut model = pb
        .try_optimise(Sense::Minimise)
        .map_err(|e| Error::Highs(format!("pass LP {e:?}")))?;
    model.make_quiet();
    serialise_openmp_once();
    model
        .try_set_option("parallel", "off")
        .map_err(|_| Error::Highs("cannot set parallel=off".into()))?;
    model
        .try_set_option("threads", 1_i32)
        .map_err(|_| Error::Highs("cannot set threads=1".into()))?;
    model
        .try_set_option("time_limit", 0.05_f64)
        .map_err(|_| Error::Highs("cannot set time_limit".into()))?;

    let (q_start, q_index, q_value) = identity_csc(n);
    let st = unsafe {
        Highs_passHessian(
            model.as_mut_ptr(),
            n as HighsInt,
            n as HighsInt,
            1,
            q_start.as_ptr(),
            q_index.as_ptr(),
            q_value.as_ptr(),
        )
    };
    if st != STATUS_OK {
        return Err(Error::Highs(format!("pass Hessian status {st}")));
    }

    let solved = model
        .try_solve()
        .map_err(|e| Error::Highs(format!("solve {e:?}")))?;
    if solved.status() != HighsModelStatus::Optimal {
        return Err(Error::Highs(format!("status {:?}", solved.status())));
    }
    let sol = solved.get_solution();
    let p = sol.columns();
    if p.len() != n {
        return Err(Error::Highs(format!("column count {} != {n}", p.len())));
    }
    Ok(Array1::from(p.to_vec()))
}

fn column_bounds(k: usize, x: ArrayView1<f64>, opts: &HighsStep) -> (f64, f64) {
    let mut lo = opts.trust.map(|t| -t).unwrap_or(f64::NEG_INFINITY);
    let mut hi = opts.trust.map(|t| t).unwrap_or(f64::INFINITY);
    if let Some(b0) = box_side_at(&opts.lo, k) {
        lo = lo.max(b0 - x[k]);
    }
    if let Some(b1) = box_side_at(&opts.hi, k) {
        hi = hi.min(b1 - x[k]);
    }
    if lo > hi {
        lo = hi;
    }
    (lo, hi)
}

/// Center, then scale the whole increment (eOn `maxAtomMotionAppliedV`).
/// Component clamps bend the two-loop direction; a single scale does not.
fn project_center_scale(
    d: &mut Array1<f64>,
    x: ArrayView1<f64>,
    opts: &HighsStep,
    n_atoms: usize,
    dim: usize,
) {
    if n_atoms == 0 || dim == 0 || d.len() != n_atoms * dim {
        scale_to_bounds(d, x, opts);
        return;
    }
    for _ in 0..8 {
        center_axes(d, n_atoms, dim);
        scale_site_motion(d, n_atoms, dim, opts.trust);
        scale_to_bounds(d, x, opts);
    }
    center_axes(d, n_atoms, dim);
}

fn scale_site_motion(d: &mut Array1<f64>, n_atoms: usize, dim: usize, trust: Option<f64>) {
    let Some(tmax) = trust else {
        return;
    };
    if !(tmax > 0.0) {
        return;
    }
    let mut max_mot = 0.0_f64;
    for i in 0..n_atoms {
        let mut n2 = 0.0;
        for h in 0..dim {
            let v = d[i * dim + h];
            n2 += v * v;
        }
        max_mot = max_mot.max(n2.sqrt());
    }
    if max_mot > tmax {
        let s = tmax / max_mot;
        for v in d.iter_mut() {
            *v *= s;
        }
    }
}

fn center_axes(d: &mut Array1<f64>, n_atoms: usize, dim: usize) {
    let n = n_atoms as f64;
    for h in 0..dim {
        let mut sum = 0.0;
        for i in 0..n_atoms {
            sum += d[i * dim + h];
        }
        let mean = sum / n;
        for i in 0..n_atoms {
            d[i * dim + h] -= mean;
        }
    }
}

fn scale_to_bounds(d: &mut Array1<f64>, x: ArrayView1<f64>, opts: &HighsStep) {
    let mut s = 1.0_f64;
    for k in 0..d.len() {
        let (lo, hi) = column_bounds(k, x, opts);
        let dk = d[k];
        if dk > 1e-16 {
            s = s.min(hi / dk);
        } else if dk < -1e-16 {
            s = s.min(lo / dk);
        }
    }
    if s < 1.0 && s > 0.0 {
        for v in d.iter_mut() {
            *v *= s;
        }
    } else if s <= 0.0 {
        for k in 0..d.len() {
            let (lo, hi) = column_bounds(k, x, opts);
            d[k] = d[k].clamp(lo, hi);
        }
    }
}

/// Newton QP on a PSD host Hessian, or `Q = I` projection of `d`.
///
/// `min 1/2 p^T Q p + c^T p` with the box, trust, and equalities on
/// `opts`. Unconstrained (no box, no centering, no equalities) skips
/// HiGHS.
pub fn highs_feasible_step(
    direction: Option<&Array1<f64>>,
    hess: Option<&Array2<f64>>,
    grad: &Array1<f64>,
    x: ArrayView1<f64>,
    opts: &HighsStep,
) -> Result<Array1<f64>> {
    let n = grad.len();
    if x.len() != n {
        return Err(Error::Dim {
            got: x.len(),
            dim: n,
        });
    }
    if !opts.has_box() && !opts.needs_qp() && opts.center_axes.is_none() {
        if let Some(h) = hess {
            return Ok(crate::newton::shifted_newton(h, grad));
        }
        if let Some(d) = direction {
            return Ok(d.clone());
        }
        return Ok(grad.mapv(|v| -v));
    }

    let (c, q) = if let Some(h) = hess {
        if h.nrows() != n || h.ncols() != n {
            return Err(Error::Dim {
                got: h.nrows(),
                dim: n,
            });
        }
        (grad.clone(), Some(h))
    } else if let Some(d) = direction {
        (d.mapv(|v| -v), None)
    } else {
        (grad.clone(), None)
    };

    let mut pb = RowProblem::default();
    let mut cols = Vec::with_capacity(n);
    for k in 0..n {
        let (lo, hi) = column_bounds(k, x, opts);
        cols.push(pb.add_column(c[k], lo..=hi));
    }
    if let Some((n_atoms, dim)) = opts.center_axes {
        if n_atoms * dim == n && n_atoms > 0 {
            for h in 0..dim {
                let row: Vec<_> = (0..n_atoms).map(|i| (cols[i * dim + h], 1.0)).collect();
                pb.add_row(0.0..=0.0, &row);
            }
        }
    }
    for (coeffs, rhs) in &opts.equalities {
        let row: Vec<_> = coeffs
            .iter()
            .filter(|(i, _)| *i < n)
            .map(|(i, a)| (cols[*i], *a))
            .collect();
        if !row.is_empty() {
            pb.add_row(*rhs..=*rhs, &row);
        }
    }

    let mut model = pb
        .try_optimise(Sense::Minimise)
        .map_err(|e| Error::Highs(format!("pass LP {e:?}")))?;
    model.make_quiet();
    serialise_openmp_once();
    let _ = model.try_set_option("parallel", "off");
    let _ = model.try_set_option("threads", 1_i32);
    let _ = model.try_set_option("time_limit", 1.0_f64);

    let (q_start, q_index, q_value) = match q {
        Some(h) => dense_csc(h),
        None => identity_csc(n),
    };
    let st = unsafe {
        Highs_passHessian(
            model.as_mut_ptr(),
            n as HighsInt,
            q_value.len() as HighsInt,
            1,
            q_start.as_ptr(),
            q_index.as_ptr(),
            q_value.as_ptr(),
        )
    };
    if st != STATUS_OK {
        return Err(Error::Highs(format!("pass Hessian status {st}")));
    }
    let solved = model
        .try_solve()
        .map_err(|e| Error::Highs(format!("solve {e:?}")))?;
    if solved.status() != HighsModelStatus::Optimal {
        return Err(Error::Highs(format!("status {:?}", solved.status())));
    }
    let sol = solved.get_solution();
    let p = sol.columns();
    if p.len() != n {
        return Err(Error::Highs(format!("column count {} != {n}", p.len())));
    }
    Ok(Array1::from(p.to_vec()))
}

fn dense_csc(h: &Array2<f64>) -> (Vec<HighsInt>, Vec<HighsInt>, Vec<f64>) {
    let n = h.nrows();
    let mut start = Vec::with_capacity(n + 1);
    let mut index = Vec::new();
    let mut value = Vec::new();
    start.push(0);
    for j in 0..n {
        for i in 0..n {
            let v = h[(i, j)];
            if v.abs() > 1e-16 {
                index.push(i as HighsInt);
                value.push(v);
            }
        }
        start.push(value.len() as HighsInt);
    }
    if value.is_empty() {
        return identity_csc(n);
    }
    (start, index, value)
}

fn identity_csc(n: usize) -> (Vec<HighsInt>, Vec<HighsInt>, Vec<f64>) {
    let mut start: Vec<HighsInt> = (0..n).map(|j| j as HighsInt).collect();
    start.push(n as HighsInt);
    let index: Vec<HighsInt> = (0..n).map(|j| j as HighsInt).collect();
    let value = vec![1.0; n];
    (start, index, value)
}

#[cfg(test)]
mod tests {
    use super::identity_csc;

    #[test]
    fn identity_hessian_has_terminal_column_pointer() {
        let (start, index, value) = identity_csc(32);
        assert_eq!(start.len(), 33);
        assert_eq!(start.last(), Some(&(value.len() as _)));
        assert_eq!(index.len(), value.len());
    }
}
