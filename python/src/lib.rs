//! One step of an rgmin session, called from Python.
//!
//! The host owns the loop. ASE's `Optimizer.run` is that loop: each
//! `step` evaluates the atoms through the callback and advances the
//! session. L-BFGS defaults here to the ASE length control
//! ([`rgmin::LbfgsStep::Scaled`]). `eon_cap` selects the crate default,
//! the line-search-free eOn cap. `wolfe` is the strong Wolfe search.

use std::sync::{Arc, Mutex};

use ndarray::{Array1, ArrayView1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rgmin::{
    Accept, ApplyHessian, Control, FireConfig, FireKind, LbfgsStep, LineSearch, Method, Oracle,
    SearchStep, Solver, StopNorm, SuppliedSearch, lowest_eigenpairs as solve_lowest_eigenpairs,
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

/// Lowest eigenpairs of a symmetric product.
///
/// `hvp(x, v)` returns `H(x) v`. `x0_block` is a list of rows, shape
/// `(n, k)`. Values come back ascending. Each vector is one column.
/// A pair that misses `tol` raises `RuntimeError`.
#[pyfunction]
#[pyo3(signature = (hvp, x0_block, x=None, *, tol=1e-8, max_iter=None))]
fn lowest_eigenpairs(
    py: Python<'_>,
    hvp: PyObject,
    x0_block: Vec<Vec<f64>>,
    x: Option<Vec<f64>>,
    tol: f64,
    max_iter: Option<usize>,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
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
    let x_arr = Array1::from_vec(point);
    let solved = py.allow_threads(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve_lowest_eigenpairs(&apply, x_arr.view(), seed.view(), tol, iterations)
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
    if found.pairs.iter().any(|pair| !pair.converged) {
        return Err(PyRuntimeError::new_err(
            "lowest eigenpairs did not meet the residual tolerance",
        ));
    }
    let values = found.pairs.iter().map(|pair| pair.value).collect();
    let columns = found
        .pairs
        .iter()
        .map(|pair| pair.vector.to_vec())
        .collect();
    Ok((values, columns))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Session>()?;
    m.add_class::<LineSearchPy>()?;
    m.add_function(wrap_pyfunction!(lowest_eigenpairs, m)?)?;
    m.add("version", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
