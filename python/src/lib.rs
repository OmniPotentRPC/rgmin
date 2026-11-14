//! One step of an rgmin session, called from Python.
//!
//! The host owns the loop. ASE's `Optimizer.run` is that loop: each
//! `step` evaluates the atoms through the callback and advances the
//! session. L-BFGS defaults here to the ASE length control
//! ([`rgmin::LbfgsStep::Scaled`]). `eon_cap` selects the crate default,
//! the line-search-free eOn cap. `wolfe` is the strong Wolfe search.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rgmin::{
    Accept, ApplyBlockPreconditioner, ApplyHessian, Control, Error, FireConfig, FireKind, LbfgsH0,
    LbfgsPairs, LbfgsStep, LineSearch, ManifoldFactor, Method, Oracle, RiemannianManifold,
    SearchStep, Solver, SpdMetric, StiefelRetraction, StopNorm, SuppliedSearch,
    lowest_eigenpairs as solve_lowest_eigenpairs, lowest_eigenpairs_precond,
};

#[pyclass]
struct Session {
    solver: Solver,
}

fn control(dt: f64) -> Control {
    Control {
        maxiter: 1_000_000,
        gtol: 1e-12,
        istep: if dt.is_finite() && dt > 0.0 { dt } else { 1.0 },
        maxmove: None,
        stop_norm: StopNorm::MaxAtom,
        ..Control::default()
    }
}

#[pymethods]
impl Session {
    /// `kind` is `lbfgs`, `bfgs`, `fire`, `fire2`, or `cg`.
    #[new]
    #[pyo3(signature = (
        kind,
        n,
        *,
        max_atom=0.2,
        wolfe=false,
        eon_cap=false,
        memory=8,
        dt=0.1,
        dt_max=1.0,
        n_min=5,
        f_inc=1.1,
        f_dec=0.5,
        alpha=0.1,
        f_alpha=0.99,
        downhill=false,
    ))]
    fn new(
        kind: &str,
        n: usize,
        max_atom: f64,
        wolfe: bool,
        eon_cap: bool,
        memory: usize,
        dt: f64,
        dt_max: f64,
        n_min: usize,
        f_inc: f64,
        f_dec: f64,
        alpha: f64,
        f_alpha: f64,
        downhill: bool,
    ) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("dimension must be positive"));
        }
        let method = match kind {
            "lbfgs" => Method::Lbfgs {
                memory: memory.max(1),
            },
            "bfgs" => Method::Bfgs,
            "fire" => Method::Fire { kind: FireKind::V1 },
            "fire2" => Method::Fire { kind: FireKind::V2 },
            "cg" => Method::polak_ribiere_plus(),
            other => {
                return Err(PyValueError::new_err(format!("unknown method {other}")));
            }
        };
        let istep = if kind.starts_with("fire") { dt } else { 1.0 };
        let mut solver = Solver::new(method, control(istep), n);
        match kind {
            "lbfgs" => {
                let step = if wolfe {
                    LbfgsStep::Wolfe
                } else if eon_cap {
                    LbfgsStep::Capped { max_atom }
                } else {
                    LbfgsStep::Scaled { max_atom }
                };
                solver
                    .set_lbfgs_step(step)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            "fire" | "fire2" => {
                let mut cfg = FireConfig::ase();
                cfg.dt = dt;
                cfg.dt_max = dt_max;
                cfg.n_min = n_min;
                cfg.f_inc = f_inc;
                cfg.f_dec = f_dec;
                cfg.alpha = alpha;
                cfg.f_alpha = f_alpha;
                solver
                    .set_fire_config(cfg)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                if downhill {
                    solver.set_accept(Accept::Energy);
                }
            }
            "cg" => solver.set_linesearch(LineSearch::Wolfe {
                c1: 1e-4,
                c2: 0.1,
                maxiter: 20,
            }),
            _ => {}
        }
        if max_atom.is_finite() && max_atom > 0.0 {
            solver.set_atom_maxmove(max_atom);
        }
        Ok(Self { solver })
    }

    /// One outer iteration.
    ///
    /// `x` is the current point. `callback(x) -> (value, gradient)` is
    /// the objective. The gradient must have the same length as `x`.
    /// Returns the accepted point, its value, and the gradient norm.
    fn step(
        &mut self,
        py: Python<'_>,
        x: Vec<f64>,
        callback: PyObject,
    ) -> PyResult<(Vec<f64>, f64, f64)> {
        if x.len() != self.solver.dim() {
            return Err(PyValueError::new_err(format!(
                "x length {} != session dimension {}",
                x.len(),
                self.solver.dim()
            )));
        }
        let mut pos = Array1::from_vec(x);
        let caught: Arc<Mutex<Option<PyErr>>> = Arc::new(Mutex::new(None));
        let slot = caught.clone();
        let cb = callback.clone_ref(py);
        let oracle = Oracle::unbounded(pos.len(), move |v: ArrayView1<f64>| {
            match call_objective(&cb, v) {
                Ok(pair) => pair,
                Err(err) => {
                    if let Ok(mut guard) = slot.lock() {
                        *guard = Some(err);
                    }
                    (f64::NAN, Array1::from_elem(v.len(), f64::NAN))
                }
            }
        });
        let report = py.allow_threads(|| self.solver.step(&oracle, &mut pos));
        if let Some(err) = caught.lock().ok().and_then(|mut guard| guard.take()) {
            return Err(err);
        }
        match report {
            Ok(report) => Ok((pos.to_vec(), report.value, report.grad_norm)),
            Err(err) => Err(PyValueError::new_err(err.to_string())),
        }
    }
}

/// A line search that takes the caller's energy and gradient.
///
/// The first `offer` is the energy and gradient at `x`. Each later
/// `offer` is the energy and gradient at the trial the previous call
/// returned. The search does not call an objective.
#[pyclass(name = "LineSearch")]
struct LineSearchPy {
    search: SuppliedSearch,
}

#[pymethods]
impl LineSearchPy {
    /// `kind` is `brent`, `backtracking`, `goldstein`, or `wolfe`.
    #[new]
    #[pyo3(signature = (kind, x, direction, *, istep=1.0, c1=1e-4, c2=0.9, beta=0.5, maxiter=40, tol=1e-10))]
    fn new(
        kind: &str,
        x: Vec<f64>,
        direction: Vec<f64>,
        istep: f64,
        c1: f64,
        c2: f64,
        beta: f64,
        maxiter: usize,
        tol: f64,
    ) -> PyResult<Self> {
        if x.len() != direction.len() || x.is_empty() {
            return Err(PyValueError::new_err(
                "x and direction must have the same positive length",
            ));
        }
        let rule = match kind {
            "brent" => LineSearch::Brent { maxiter, tol },
            "backtracking" => LineSearch::Backtracking {
                c: c1,
                beta,
                maxiter,
            },
            "goldstein" => LineSearch::Goldstein {
                c: c1,
                beta,
                maxiter,
            },
            "wolfe" => LineSearch::Wolfe { c1, c2, maxiter },
            _ => return Err(PyValueError::new_err(format!("unknown line search {kind}"))),
        };
        Ok(Self {
            search: SuppliedSearch::new(
                rule,
                ArrayView1::from(x.as_slice()),
                ArrayView1::from(direction.as_slice()),
                istep,
            ),
        })
    }

    /// Return `(x, alpha, value, done)`. `done` is false when `x` is the next trial.
    fn offer(&mut self, f: f64, gradient: Vec<f64>) -> PyResult<(Vec<f64>, f64, f64, bool)> {
        let step = self.search.offer(f, ArrayView1::from(gradient.as_slice()));
        Ok(match step {
            SearchStep::Evaluate { alpha, x } => (x.to_vec(), alpha, 0.0, false),
            SearchStep::Accepted { x, value, alpha } => (x.to_vec(), alpha, value, true),
        })
    }
}

/// Stored `(s, y)` pairs for one L-BFGS direction.
///
/// `direction` returns `-H g` and does not evaluate an objective.
/// `h0` is a float (`H0 = γ I`), a one-dimensional NumPy array or
/// sequence (the diagonal of `H0`), or a callable `v -> H0 v`.
#[pyclass(name = "LbfgsPairs")]
struct LbfgsPairsPy {
    pairs: LbfgsPairs,
}

fn as_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(values) = obj.extract::<Vec<f64>>() {
        return Ok(values);
    }
    if obj.hasattr("tolist").unwrap_or(false) {
        let listed = obj.call_method0("tolist")?;
        return listed
            .extract::<Vec<f64>>()
            .map_err(|_| PyValueError::new_err("H0 must be a one-dimensional sequence of floats"));
    }
    Err(PyValueError::new_err("expected a sequence of floats"))
}

fn direction_error(err: Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pymethods]
impl LbfgsPairsPy {
    /// `capacity` 0 keeps ten pairs.
    #[new]
    #[pyo3(signature = (n, capacity=10))]
    fn new(n: usize, capacity: usize) -> PyResult<Self> {
        LbfgsPairs::new(n, capacity)
            .map(|pairs| Self { pairs })
            .map_err(direction_error)
    }

    fn __len__(&self) -> usize {
        self.pairs.len()
    }

    fn dim(&self) -> usize {
        self.pairs.dim()
    }

    /// Store one pair. Returns false when the curvature gate declines it.
    fn push(&mut self, s: Vec<f64>, y: Vec<f64>) -> PyResult<bool> {
        self.pairs
            .push(
                ArrayView1::from(s.as_slice()),
                ArrayView1::from(y.as_slice()),
            )
            .map_err(direction_error)
    }

    fn clear(&mut self) {
        self.pairs.clear();
    }

    /// Direction `-H g`. `h0` is a float, a NumPy array, or a callable.
    fn direction(&self, gradient: Vec<f64>, h0: Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let g = ArrayView1::from(gradient.as_slice());
        let numpy = h0.hasattr("ndim").unwrap_or(false);
        let computed = if h0.is_callable() && !numpy {
            let callback = h0.clone().unbind();
            let failed = RefCell::new(None);
            let result = self.pairs.direction(
                g,
                LbfgsH0::Apply(&|v| {
                    Python::with_gil(|py| match callback.call1(py, (v.to_vec(),)) {
                        Ok(out) => match as_f64_vec(out.bind(py)) {
                            Ok(values) => Ok(Array1::from_vec(values)),
                            Err(err) => {
                                *failed.borrow_mut() = Some(err);
                                Err(Error::LbfgsDirection {
                                    what: "H0 callback failed",
                                })
                            }
                        },
                        Err(err) => {
                            *failed.borrow_mut() = Some(err);
                            Err(Error::LbfgsDirection {
                                what: "H0 callback failed",
                            })
                        }
                    })
                }),
            );
            if let Some(err) = failed.into_inner() {
                return Err(err);
            }
            result
        } else if numpy {
            let ndim: usize = h0.getattr("ndim")?.extract()?;
            if ndim == 0 {
                let gamma: f64 = h0.call_method0("item")?.extract()?;
                self.pairs.direction(g, LbfgsH0::Scale(gamma))
            } else {
                let diag = Array1::from_vec(as_f64_vec(&h0)?);
                self.pairs.direction(g, LbfgsH0::Diagonal(diag.view()))
            }
        } else if let Ok(gamma) = h0.extract::<f64>() {
            self.pairs.direction(g, LbfgsH0::Scale(gamma))
        } else {
            let diag = Array1::from_vec(as_f64_vec(&h0)?);
            self.pairs.direction(g, LbfgsH0::Diagonal(diag.view()))
        };
        computed.map(|d| d.to_vec()).map_err(direction_error)
    }
}

fn call_objective(callback: &PyObject, x: ArrayView1<f64>) -> PyResult<(f64, Array1<f64>)> {
    Python::with_gil(|py| {
        let coords: Vec<f64> = x.iter().copied().collect();
        let out = callback.call1(py, (coords,))?;
        let (value, gradient): (f64, Vec<f64>) = out.extract(py)?;
        if gradient.len() != x.len() {
            return Err(PyValueError::new_err(format!(
                "gradient length {} != {}",
                gradient.len(),
                x.len()
            )));
        }
        Ok((value, Array1::from_vec(gradient)))
    })
}

fn call_precond(callback: &PyObject, block: ArrayView2<f64>) -> PyResult<Array2<f64>> {
    Python::with_gil(|py| {
        let (n, m) = block.dim();
        let rows: Vec<Vec<f64>> = (0..n).map(|i| block.row(i).to_vec()).collect();
        let out = callback.call1(py, (rows,))?;
        let rows: Vec<Vec<f64>> = out.extract(py)?;
        if rows.len() != n || rows.iter().any(|row| row.len() != m) {
            return Err(PyValueError::new_err(
                "preconditioner must return shape (n, m)",
            ));
        }
        let mut image = Array2::zeros((n, m));
        for (i, row) in rows.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                image[(i, j)] = *value;
            }
        }
        Ok(image)
    })
}

fn call_hvp(callback: &PyObject, x: ArrayView1<f64>, v: ArrayView1<f64>) -> PyResult<Array1<f64>> {
    Python::with_gil(|py| {
        let point: Vec<f64> = x.iter().copied().collect();
        let vector: Vec<f64> = v.iter().copied().collect();
        let out = callback.call1(py, (point, vector))?;
        let product: Vec<f64> = out.extract(py)?;
        if product.len() != v.len() {
            return Err(PyValueError::new_err(format!(
                "hessian product length {} != {}",
                product.len(),
                v.len()
            )));
        }
        Ok(Array1::from_vec(product))
    })
}

struct PyHvp {
    callback: PyObject,
    caught: Arc<Mutex<Option<PyErr>>>,
}

struct PyPrecond {
    callback: PyObject,
    caught: Arc<Mutex<Option<PyErr>>>,
}

impl ApplyBlockPreconditioner for PyPrecond {
    fn apply_block_preconditioner(&self, block: ArrayView2<f64>) -> Array2<f64> {
        match call_precond(&self.callback, block) {
            Ok(image) => image,
            Err(err) => {
                if let Ok(mut guard) = self.caught.lock() {
                    if guard.is_none() {
                        *guard = Some(err);
                    }
                }
                Array2::from_elem(block.dim(), f64::NAN)
            }
        }
    }
}

impl ApplyHessian for PyHvp {
    fn apply_hessian(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        match call_hvp(&self.callback, x, v) {
            Ok(product) => product,
            Err(err) => {
                if let Ok(mut guard) = self.caught.lock() {
                    if guard.is_none() {
                        *guard = Some(err);
                    }
                }
                Array1::from_elem(v.len(), f64::NAN)
            }
        }
    }
}

type SolvedPairs = (Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<bool>);

/// Lowest eigenpairs of a symmetric product.
///
/// `hvp(x, v)` returns `H(x) v`. `x0_block` is a list of rows, shape
/// `(n, k)`. Values come back ascending. Each vector is one column.
/// The third list is `||Hv − θv||` and the fourth is the converged flag
/// of each pair. `precond(block)` applies `T` to a row-major `(n, m)`
/// block. Omitted, `T` is the identity. This function does not raise
/// when a pair misses `tol`; the flag says so.
#[pyfunction]
#[pyo3(signature = (hvp, x0_block, x=None, *, tol=1e-8, max_iter=None, precond=None))]
fn lowest_eigenpairs(
    py: Python<'_>,
    hvp: PyObject,
    x0_block: Vec<Vec<f64>>,
    x: Option<Vec<f64>>,
    tol: f64,
    max_iter: Option<usize>,
    precond: Option<PyObject>,
) -> PyResult<SolvedPairs> {
    let n = x0_block.len();
    let k = x0_block.first().map(|row| row.len()).unwrap_or(0);
    if n == 0 || k == 0 || x0_block.iter().any(|row| row.len() != k) {
        return Err(PyValueError::new_err(
            "x0_block must have shape (n, k) with n and k positive",
        ));
    }
    let point = match x {
        Some(point) if point.len() == n => point,
        Some(point) => {
            return Err(PyValueError::new_err(format!(
                "x length {} != {}",
                point.len(),
                n
            )));
        }
        None => vec![0.0; n],
    };
    let mut seed = ndarray::Array2::zeros((n, k));
    for (i, row) in x0_block.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            seed[(i, j)] = *value;
        }
    }
    let iterations = max_iter.unwrap_or_else(|| n.max(8));
    let caught: Arc<Mutex<Option<PyErr>>> = Arc::new(Mutex::new(None));
    let apply = PyHvp {
        callback: hvp.clone_ref(py),
        caught: caught.clone(),
    };
    let pre_apply = precond.map(|callback| PyPrecond {
        callback,
        caught: caught.clone(),
    });
    let x_arr = Array1::from_vec(point);
    let solved = py.allow_threads(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Some(op) = pre_apply.as_ref() {
                lowest_eigenpairs_precond(
                    &apply,
                    Some(op),
                    x_arr.view(),
                    seed.view(),
                    tol,
                    iterations,
                )
            } else {
                solve_lowest_eigenpairs(&apply, x_arr.view(), seed.view(), tol, iterations)
            }
        }))
    });
    if let Some(err) = caught.lock().ok().and_then(|mut guard| guard.take()) {
        return Err(err);
    }
    let found = match solved {
        Ok(Ok(found)) => found,
        Ok(Err(err)) => return Err(PyValueError::new_err(err.to_string())),
        Err(_) => return Err(PyRuntimeError::new_err("lowest eigenpairs panicked")),
    };
    let values = found.pairs.iter().map(|pair| pair.value).collect();
    let columns = found
        .pairs
        .iter()
        .map(|pair| pair.vector.to_vec())
        .collect();
    let residuals = found.pairs.iter().map(|pair| pair.residual).collect();
    let converged = found.pairs.iter().map(|pair| pair.converged).collect();
    Ok((values, columns, residuals, converged))
}

/// Stiefel, SPD or product manifold over flat column-major lists.
///
/// `rgmin.manifold` wraps this class with NumPy arrays.
#[pyclass(name = "RiemannianManifold", frozen)]
struct ManifoldPy {
    inner: RiemannianManifold,
}

fn manifold_error(err: Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

impl ManifoldPy {
    fn unary(
        &self,
        a: &[f64],
        b: &[f64],
        op: impl Fn(&RiemannianManifold, &[f64], &[f64], &mut [f64]) -> rgmin::Result<()>,
    ) -> PyResult<Vec<f64>> {
        let mut out = vec![0.0; self.inner.packed_len()];
        op(&self.inner, a, b, &mut out).map_err(manifold_error)?;
        Ok(out)
    }
}

#[pymethods]
impl ManifoldPy {
    /// St(n, p). `retraction` is `qr` or `polar`.
    #[staticmethod]
    #[pyo3(signature = (n, p, retraction="qr"))]
    fn stiefel(n: usize, p: usize, retraction: &str) -> PyResult<Self> {
        let retraction = match retraction {
            "qr" => StiefelRetraction::Qr,
            "polar" => StiefelRetraction::Polar,
            other => {
                return Err(PyValueError::new_err(format!(
                    "retraction must be 'qr' or 'polar', got '{other}'"
                )));
            }
        };
        RiemannianManifold::stiefel(n, p, retraction)
            .map(|inner| Self { inner })
            .map_err(manifold_error)
    }

    /// SPD(p). `metric` is `affine_invariant` or `log_euclidean`.
    #[staticmethod]
    #[pyo3(signature = (p, metric="affine_invariant"))]
    fn spd(p: usize, metric: &str) -> PyResult<Self> {
        let metric = match metric {
            "affine_invariant" => SpdMetric::AffineInvariant,
            "log_euclidean" => SpdMetric::LogEuclidean,
            other => {
                return Err(PyValueError::new_err(format!(
                    "metric must be 'affine_invariant' or 'log_euclidean', got '{other}'"
                )));
            }
        };
        RiemannianManifold::spd(p, metric)
            .map(|inner| Self { inner })
            .map_err(manifold_error)
    }

    /// Product of the factors in order.
    #[staticmethod]
    fn product(factors: Vec<PyRef<'_, Self>>) -> PyResult<Self> {
        RiemannianManifold::product(factors.iter().map(|f| f.inner.clone()))
            .map(|inner| Self { inner })
            .map_err(manifold_error)
    }

    fn __len__(&self) -> usize {
        self.inner.packed_len()
    }

    /// `(kind, rows, columns)` of each factor; kind is `stiefel` or `spd`.
    fn factors(&self) -> Vec<(&'static str, usize, usize)> {
        self.inner
            .factors()
            .iter()
            .map(|f| {
                let (rows, cols) = f.shape();
                let kind = match f {
                    ManifoldFactor::Stiefel { .. } => "stiefel",
                    ManifoldFactor::Spd { .. } => "spd",
                };
                (kind, rows, cols)
            })
            .collect()
    }

    fn project(&self, x: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<f64>> {
        self.unary(&x, &v, RiemannianManifold::project)
    }

    fn egrad_to_rgrad(&self, x: Vec<f64>, egrad: Vec<f64>) -> PyResult<Vec<f64>> {
        self.unary(&x, &egrad, RiemannianManifold::egrad_to_rgrad)
    }

    fn retract(&self, x: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<f64>> {
        self.unary(&x, &v, RiemannianManifold::retract)
    }

    fn exp(&self, x: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<f64>> {
        self.unary(&x, &v, RiemannianManifold::exp)
    }

    fn log(&self, x: Vec<f64>, y: Vec<f64>) -> PyResult<Vec<f64>> {
        self.unary(&x, &y, RiemannianManifold::log)
    }

    fn transport(&self, x_from: Vec<f64>, x_to: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut out = vec![0.0; self.inner.packed_len()];
        self.inner
            .transport(&x_from, &x_to, &v, &mut out)
            .map_err(manifold_error)?;
        Ok(out)
    }

    fn inner(&self, x: Vec<f64>, u: Vec<f64>, v: Vec<f64>) -> PyResult<f64> {
        self.inner.inner(&x, &u, &v).map_err(manifold_error)
    }

    /// `(rgrad, x_next)` for one step of size `t`.
    fn gradient_step(
        &self,
        x: Vec<f64>,
        egrad: Vec<f64>,
        t: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let len = self.inner.packed_len();
        let (mut rgrad, mut next) = (vec![0.0; len], vec![0.0; len]);
        self.inner
            .gradient_step(&x, &egrad, t, &mut rgrad, &mut next)
            .map_err(manifold_error)?;
        Ok((rgrad, next))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Session>()?;
    m.add_class::<LineSearchPy>()?;
    m.add_class::<LbfgsPairsPy>()?;
    m.add_class::<ManifoldPy>()?;
    m.add_function(wrap_pyfunction!(lowest_eigenpairs, m)?)?;
    m.add("version", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
