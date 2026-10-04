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
use crate::{HighsCallback, HighsOptions};

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

/// Bounds and equalities on one L-BFGS model step.
#[derive(Clone, Debug, Default)]
pub struct HighsStep {
    /// L_inf trust radius on the step. `None` is unbounded.
    pub trust: Option<f64>,
    /// Uniform box lower bound on coordinates of `x + p`.
    pub lo: Option<f64>,
    /// Uniform box upper bound on coordinates of `x + p`.
    pub hi: Option<f64>,
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

unsafe extern "C" {
    fn Highs_setCallback(
        highs: *mut std::os::raw::c_void,
        callback: Option<
            unsafe extern "C" fn(
                i32,
                *const std::os::raw::c_char,
                *const std::os::raw::c_void,
                *mut std::os::raw::c_void,
                *mut std::os::raw::c_void,
            ),
        >,
        user: *mut std::os::raw::c_void,
    ) -> HighsInt;
    fn Highs_startCallback(highs: *mut std::os::raw::c_void, kind: i32) -> HighsInt;
}

#[repr(C)]
struct HighsCallbackDataIn {
    user_interrupt: std::os::raw::c_int,
}

unsafe extern "C" fn highs_callback(
    kind: i32,
    message: *const std::os::raw::c_char,
    _data_out: *const std::os::raw::c_void,
    data_in: *mut std::os::raw::c_void,
    user: *mut std::os::raw::c_void,
) {
    let binding = unsafe { &*user.cast::<HighsCallback>() };
    let mut interrupt = 0;
    unsafe { (binding.function)(kind, message, &mut interrupt, binding.user as *mut _) };
    if !data_in.is_null() && interrupt != 0 {
        unsafe { (*data_in.cast::<HighsCallbackDataIn>()).user_interrupt = 1 };
    }
}

fn apply_engine(model: &mut highs::Model, options: &HighsOptions) -> Result<()> {
    if let Some(name) = options.solver.as_highs() {
        model
            .try_set_option("solver", name)
            .map_err(|_| Error::Highs(format!("cannot set solver={name}")))?;
    }
    if let Some(name) = options.crossover.as_highs() {
        model
            .try_set_option("run_crossover", name)
            .map_err(|_| Error::Highs(format!("cannot set run_crossover={name}")))?;
    }
    Ok(())
}

fn bind_callback(
    model: &mut highs::Model,
    options: &HighsOptions,
    owner: &mut Option<Box<HighsCallback>>,
) -> Result<()> {
    let Some(binding) = options.callback else {
        return Ok(());
    };
    // Model construction disables output, including the logging callback.
    // Callback delivery needs output enabled; console logging stays off.
    model
        .try_set_option("output_flag", true)
        .map_err(|_| Error::Highs("cannot enable callback output".into()))?;
    *owner = Some(Box::new(binding));
    let ptr = model.as_mut_ptr();
    let context = owner.as_mut().unwrap().as_mut() as *mut HighsCallback;
    let status = unsafe { Highs_setCallback(ptr, Some(highs_callback), context.cast()) };
    if status != STATUS_OK {
        return Err(Error::Highs(format!("set callback: status {status}")));
    }
    for kind in [0, 1, 2] {
        let status = unsafe { Highs_startCallback(ptr, kind) };
        if status != STATUS_OK {
            return Err(Error::Highs(format!(
                "start callback {kind}: status {status}"
            )));
        }
    }
    Ok(())
}

impl Lbfgs {
    /// L-BFGS direction with explicit coordinate bounds, intersected with
    /// the uniform bounds and step trust radius in [`HighsStep`].
    ///
    /// A missing or empty side is unbounded; one entry broadcasts to every
    /// coordinate. Other sides have the same length as `x`. The starting
    /// point must be finite and inside the coordinate box. Without linear
    /// constraints, component projection preserves free coordinates and
    /// falls back to projected steepest descent for a non-descent step.
    /// Linear equalities and centering use the configured HiGHS projection;
    /// their residual tolerance is `1e-7` in the units of each row.
    pub fn highs_step_boxed(
        &self,
        x: ArrayView1<f64>,
        g: ArrayView1<f64>,
        lo: Option<&[f64]>,
        hi: Option<&[f64]>,
    ) -> Result<Array1<f64>> {
        let opts = self.highs.as_ref().ok_or_else(|| {
            Error::Highs("Lbfgs.highs is None; set HighsStep before highs_step_boxed".into())
        })?;
        let n = g.len();
        if x.len() != n {
            return Err(Error::Dim {
                got: x.len(),
                dim: n,
            });
        }
        for side in [lo, hi].into_iter().flatten() {
            if !side.is_empty() && side.len() != 1 && side.len() != n {
                return Err(Error::Dim {
                    got: side.len(),
                    dim: n,
                });
            }
        }
        if x.iter().chain(g.iter()).any(|value| !value.is_finite())
            || opts.lo.is_some_and(f64::is_nan)
            || opts.hi.is_some_and(f64::is_nan)
            || opts
                .trust
                .is_some_and(|radius| !radius.is_finite() || radius < 0.0)
            || opts.center_axes.is_some_and(|(atoms, dim)| {
                atoms == 0 || dim == 0 || atoms.checked_mul(dim) != Some(n)
            })
        {
            return Err(Error::Highs("invalid bounded-step data".into()));
        }
        let mut lower = Array1::zeros(n);
        let mut upper = Array1::zeros(n);
        for k in 0..n {
            let requested_low = crate::box_objective::side_at(lo, k).unwrap_or(f64::NEG_INFINITY);
            let requested_high = crate::box_objective::side_at(hi, k).unwrap_or(f64::INFINITY);
            if requested_low.is_nan() || requested_high.is_nan() {
                return Err(Error::Highs("invalid coordinate box".into()));
            }
            let low = requested_low.max(opts.lo.unwrap_or(f64::NEG_INFINITY));
            let high = requested_high.min(opts.hi.unwrap_or(f64::INFINITY));
            if !(low <= x[k] && x[k] <= high) {
                return Err(Error::Highs("point outside coordinate box".into()));
            }
            lower[k] = (low - x[k]).max(opts.trust.map_or(f64::NEG_INFINITY, |r| -r));
            upper[k] = (high - x[k]).min(opts.trust.unwrap_or(f64::INFINITY));
        }
        let project = |direction: &mut Array1<f64>| {
            for k in 0..n {
                direction[k] = direction[k].clamp(lower[k], upper[k]);
            }
        };
        let mut direction = self.direction(g);
        if direction.iter().any(|value| !value.is_finite()) {
            return Err(Error::Highs("non-finite L-BFGS direction".into()));
        }
        if !opts.needs_qp() && opts.center_axes.is_none() {
            project(&mut direction);
            if g.dot(&direction) >= 0.0 {
                direction = g.mapv(|value| -value);
                project(&mut direction);
            }
            return Ok(direction);
        }
        let bounds = eindir_core::Bounds::new(lower.clone(), upper.clone(), 0.0);
        let mut step = highs_feasible_step_boxed(
            Some(&direction),
            None,
            &g.to_owned(),
            None,
            None,
            opts.center_axes,
            Some((Array1::zeros(n).view(), &bounds)),
            &opts.equalities,
            &self.highs_options,
        )?;
        if step.iter().any(|value| !value.is_finite()) {
            return Err(Error::Highs("non-finite bounded step".into()));
        }
        project(&mut step);
        for (row, rhs) in &opts.equalities {
            let residual = row.iter().map(|(k, a)| a * step[*k]).sum::<f64>() - rhs;
            if !residual.is_finite() || residual.abs() > EQUALITY_FEASIBILITY_TOLERANCE {
                return Err(Error::Highs(
                    "bounded step violates a linear equality".into(),
                ));
            }
        }
        if let Some((atoms, dim)) = opts.center_axes {
            for axis in 0..dim {
                let residual = (0..atoms).map(|atom| step[atom * dim + axis]).sum::<f64>();
                if !residual.is_finite() || residual.abs() > EQUALITY_FEASIBILITY_TOLERANCE {
                    return Err(Error::Highs("bounded step violates centering".into()));
                }
            }
        }
        Ok(step)
    }

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
        project_qp(&p, x, opts, &self.highs_options)
    }
}

/// Project the supplied direction onto a feasible set by minimizing
/// `||p - direction||²/2`, without scaling the direction first.
pub fn highs_projected_step(
    direction: &Array1<f64>,
    point: ArrayView1<f64>,
    constraints: &HighsStep,
    options: &HighsOptions,
) -> Result<Array1<f64>> {
    if let Some((atoms, dim)) = constraints.center_axes {
        if atoms == 0 || dim == 0 || atoms.checked_mul(dim) != Some(direction.len()) {
            return Err(Error::Highs("invalid packed projection shape".into()));
        }
        let mut centered = constraints.clone();
        centered.center_axes = None;
        for axis in 0..dim {
            centered.equalities.push((
                (0..atoms).map(|atom| (atom * dim + axis, 1.0)).collect(),
                0.0,
            ));
        }
        project_qp(direction, point, &centered, options)
    } else {
        project_qp(direction, point, constraints, options)
    }
}

fn project_qp(
    d: &Array1<f64>,
    x: ArrayView1<f64>,
    opts: &HighsStep,
    options: &HighsOptions,
) -> Result<Array1<f64>> {
    let n = d.len();
    if x.len() != n {
        return Err(Error::Dim {
            got: x.len(),
            dim: n,
        });
    }
    if d.iter().chain(x.iter()).any(|v| !v.is_finite())
        || opts.trust.is_some_and(|t| !t.is_finite() || t < 0.0)
        || opts.lo.is_some_and(f64::is_nan)
        || opts.hi.is_some_and(f64::is_nan)
        || opts.equalities.iter().any(|(row, rhs)| {
            !rhs.is_finite() || row.iter().any(|(i, a)| *i >= n || !a.is_finite())
        })
    {
        return Err(Error::Highs("invalid projection data".into()));
    }
    for k in 0..n {
        let lo = opts
            .trust
            .map_or(f64::NEG_INFINITY, |t| -t)
            .max(opts.lo.map_or(f64::NEG_INFINITY, |v| v - x[k]));
        let hi = opts
            .trust
            .unwrap_or(f64::INFINITY)
            .min(opts.hi.map_or(f64::INFINITY, |v| v - x[k]));
        if lo > hi {
            return Err(Error::Highs("infeasible projection box".into()));
        }
    }
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

    // The model and solved model drop before their callback storage.
    let mut callback_owner = None;
    let mut model = pb
        .try_optimise(Sense::Minimise)
        .map_err(|e| Error::Highs(format!("pass LP {e:?}")))?;
    if options.callback.is_none() {
        model.make_quiet();
    }
    apply_engine(&mut model, options)?;
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

    bind_callback(&mut model, options, &mut callback_owner)?;
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
    let mut hi = opts.trust.unwrap_or(f64::INFINITY);
    if let Some(b0) = opts.lo {
        lo = lo.max(b0 - x[k]);
    }
    if let Some(b1) = opts.hi {
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
    if tmax.is_nan() || tmax <= 0.0 {
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
/// `min 1/2 p^T Q p + c^T p` with per-coordinate boxes from
/// `atom_maxmove`. Unconstrained (no box, no centering) skips HiGHS.
pub fn highs_feasible_step(
    direction: Option<&Array1<f64>>,
    hess: Option<&Array2<f64>>,
    grad: &Array1<f64>,
    atom_maxmove: Option<f64>,
    trust: Option<f64>,
    center_axes: Option<(usize, usize)>,
) -> Result<Array1<f64>> {
    highs_feasible_step_with_options(
        direction,
        hess,
        grad,
        atom_maxmove,
        trust,
        center_axes,
        &HighsOptions::default(),
    )
}

/// [`highs_feasible_step`] with explicit engine and callback policies.
pub fn highs_feasible_step_with_options(
    direction: Option<&Array1<f64>>,
    hess: Option<&Array2<f64>>,
    grad: &Array1<f64>,
    atom_maxmove: Option<f64>,
    trust: Option<f64>,
    center_axes: Option<(usize, usize)>,
    options: &HighsOptions,
) -> Result<Array1<f64>> {
    highs_feasible_step_boxed(
        direction,
        hess,
        grad,
        atom_maxmove,
        trust,
        center_axes,
        None,
        &[],
        options,
    )
}

pub(crate) const EQUALITY_FEASIBILITY_TOLERANCE: f64 = 1e-7;

pub(crate) fn highs_feasible_step_boxed(
    direction: Option<&Array1<f64>>,
    hess: Option<&Array2<f64>>,
    grad: &Array1<f64>,
    atom_maxmove: Option<f64>,
    trust: Option<f64>,
    center_axes: Option<(usize, usize)>,
    coordinate_box: Option<(ArrayView1<'_, f64>, &eindir_core::Bounds<f64>)>,
    equalities: &[(Vec<(usize, f64)>, f64)],
    options: &HighsOptions,
) -> Result<Array1<f64>> {
    let n = grad.len();
    let boxed = atom_maxmove.is_some_and(|c| c > 0.0) || trust.is_some_and(|c| c > 0.0);
    if !boxed && center_axes.is_none() && coordinate_box.is_none() && equalities.is_empty() {
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
        let (mut lo, mut hi) = coord_bounds(k, atom_maxmove, trust);
        if let Some((x, bounds)) = coordinate_box {
            lo = lo.max(bounds.low[k] - x[k]);
            hi = hi.min(bounds.high[k] - x[k]);
            if !(lo <= hi) {
                return Err(Error::Highs("invalid coordinate box".into()));
            }
        }
        cols.push(pb.add_column(c[k], lo..=hi));
    }
    for (coefficients, rhs) in equalities {
        if !rhs.is_finite() || coefficients.iter().any(|(k, a)| *k >= n || !a.is_finite()) {
            return Err(Error::Highs("invalid linear equality".into()));
        }
        let row: Vec<_> = coefficients.iter().map(|(k, a)| (cols[*k], *a)).collect();
        pb.add_row(*rhs..=*rhs, &row);
    }
    if let Some((n_atoms, dim)) = center_axes
        && n_atoms * dim == n
        && n_atoms > 0
    {
        for h in 0..dim {
            let row: Vec<_> = (0..n_atoms).map(|i| (cols[i * dim + h], 1.0)).collect();
            pb.add_row(0.0..=0.0, &row);
        }
    }

    // The model and solved model drop before their callback storage.
    let mut callback_owner = None;
    let mut model = pb
        .try_optimise(Sense::Minimise)
        .map_err(|e| Error::Highs(format!("pass LP {e:?}")))?;
    if options.callback.is_none() {
        model.make_quiet();
    }
    apply_engine(&mut model, options)?;
    serialise_openmp_once();
    let _ = model.try_set_option("parallel", "off");
    let _ = model.try_set_option("threads", 1_i32);
    let _ = model.try_set_option("time_limit", 1.0_f64);
    if !equalities.is_empty() {
        model
            .try_set_option(
                "primal_feasibility_tolerance",
                EQUALITY_FEASIBILITY_TOLERANCE,
            )
            .map_err(|_| Error::Highs("cannot set equality feasibility tolerance".into()))?;
    }

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
    bind_callback(&mut model, options, &mut callback_owner)?;
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

fn coord_bounds(k: usize, atom_maxmove: Option<f64>, trust: Option<f64>) -> (f64, f64) {
    let _ = k;
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    if let Some(t) = trust
        && t > 0.0
    {
        lo = -t;
        hi = t;
    }
    if let Some(a) = atom_maxmove
        && a > 0.0
    {
        lo = lo.max(-a);
        hi = hi.min(a);
    }
    if lo > hi {
        lo = hi;
    }
    (lo, hi)
}

fn dense_csc(h: &Array2<f64>) -> (Vec<HighsInt>, Vec<HighsInt>, Vec<f64>) {
    let n = h.nrows();
    let mut start = Vec::with_capacity(n + 1);
    let mut index = Vec::new();
    let mut value = Vec::new();
    start.push(0);
    for j in 0..n {
        for i in j..n {
            let v = h[(i, j)];
            if v.abs() > 1e-16 {
                index.push(i as HighsInt);
                value.push(v);
            }
        }
        start.push(value.len() as HighsInt);
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
