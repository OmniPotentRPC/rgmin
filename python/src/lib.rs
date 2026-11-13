//! Python bindings. Arrays come in as NumPy views and leave as new
//! NumPy arrays. A callback must not keep the position array it is shown.

use ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rgmin::{EonCg, GradNorm, LbfgsStep};

fn owned(x: PyReadonlyArray1<'_, f64>) -> PyResult<Array1<f64>> {
    if let Ok(slice) = x.as_slice() {
        return Ok(Array1::from(slice.to_vec()));
    }
    Ok(Array1::from_iter(x.as_array().iter().copied()))
}

fn step_of(name: &str, max_atom: f64) -> PyResult<LbfgsStep> {
    match name {
        "capped" => Ok(LbfgsStep::Capped { max_atom }),
        "scaled" => Ok(LbfgsStep::Scaled { max_atom }),
        "wolfe" => Ok(LbfgsStep::Wolfe),
        _ => Err(PyValueError::new_err("step is capped, scaled, or wolfe")),
    }
}

fn norm_of(name: &str) -> PyResult<GradNorm> {
    match name {
        "infinity" | "inf" => Ok(GradNorm::Infinity),
        "euclidean" | "2" => Ok(GradNorm::Euclidean),
        _ => Err(PyValueError::new_err("norm is infinity or euclidean")),
    }
}

/// Persistent L-BFGS. `minimize` borrows a C-contiguous vector and
/// returns a new one.
#[pyclass(name = "Lbfgs")]
struct PyLbfgs {
    inner: rgmin::Lbfgs,
}

#[pymethods]
impl PyLbfgs {
    /// `step` is `capped` (the default), `scaled`, or `wolfe`.
    #[new]
    #[pyo3(signature = (memory=8, gtol=1e-6, ftol=0.0, step="capped", max_atom=0.2, norm="infinity"))]
    fn new(
        memory: usize,
        gtol: f64,
        ftol: f64,
        step: &str,
        max_atom: f64,
        norm: &str,
    ) -> PyResult<Self> {
        let mut inner = rgmin::Lbfgs::with_capacity(memory);
        inner.gtol = gtol;
        inner.ftol = ftol;
        inner.step = step_of(step, max_atom)?;
        inner.norm = norm_of(norm)?;
        Ok(Self { inner })
    }

    fn set_gtol(&mut self, gtol: f64) {
        self.inner.gtol = gtol;
    }

    fn set_ftol(&mut self, ftol: f64) {
        self.inner.ftol = ftol;
    }

    fn set_step(&mut self, step: &str, max_atom: f64) -> PyResult<()> {
        self.inner.step = step_of(step, max_atom)?;
        Ok(())
    }

    fn set_norm(&mut self, norm: &str) -> PyResult<()> {
        self.inner.norm = norm_of(norm)?;
        Ok(())
    }

    /// Relax `x`. `fg(x)` returns `(value, gradient)` or `None` to stop.
    ///
    /// A C-contiguous `x` is read without an extra temporary. The
    /// position array passed into `fg` is a view of the solver's
    /// buffer and is invalid after `fg` returns.
    fn minimize<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        max_iter: usize,
        fg: &Bound<'py, PyAny>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>, usize)> {
        let start = owned(x)?;
        let fg = fg.clone();
        let mut py_err: Option<PyErr> = None;
        let (value, out, evals) = self.inner.minimize(start.view(), max_iter, |xv| {
            if py_err.is_some() {
                return None;
            }
            call_fg(py, &fg, xv, &mut py_err)
        });
        if let Some(err) = py_err {
            return Err(err);
        }
        let py_x = PyArray1::from_vec(py, out.to_vec());
        Ok((value, py_x, evals))
    }
}

fn call_fg(
    py: Python<'_>,
    fg: &Bound<'_, PyAny>,
    xv: ArrayView1<f64>,
    py_err: &mut Option<PyErr>,
) -> Option<(f64, Array1<f64>)> {
    let slice = xv.as_slice()?;
    let arr = PyArray1::from_slice(py, slice);
    let result = match fg.call1((arr,)) {
        Ok(v) => v,
        Err(err) => {
            *py_err = Some(err);
            return None;
        }
    };
    if result.is_none() {
        return None;
    }
    let (value, grad) = match result.extract::<(f64, PyReadonlyArray1<f64>)>() {
        Ok(pair) => pair,
        Err(err) => {
            *py_err = Some(err);
            return None;
        }
    };
    let g = match grad.as_slice() {
        Ok(s) => Array1::from(s.to_vec()),
        Err(_) => match grad.as_array().iter().copied().collect::<Vec<_>>() {
            v => Array1::from(v),
        },
    };
    Some((value, g))
}

/// eOn conjugate gradients. `x` is updated. Returns energy, gradient
/// norm, evaluations, and whether the gradient test was met.
#[pyfunction]
#[pyo3(signature = (x, fill, max_move=0.2, gtol=1e-5, max_iter=20_000, cluster=false, atom_only=false))]
fn minimize_eon_cg<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    fill: &Bound<'py, PyAny>,
    max_move: f64,
    gtol: f64,
    max_iter: usize,
    cluster: bool,
    atom_only: bool,
) -> PyResult<(f64, f64, usize, bool, Bound<'py, PyArray1<f64>>)> {
    let mut state = owned(x)?;
    let opts = EonCg {
        max_move,
        gtol,
        max_iter,
        cluster,
        atom_only,
    };
    let fill = fill.clone();
    let mut py_err: Option<PyErr> = None;
    let report = rgmin::minimize_eon_cg(
        state.as_slice_mut().expect("owned vector"),
        |pos, grad| {
            if py_err.is_some() {
                return f64::NAN;
            }
            let arr = PyArray1::from_slice(py, pos);
            let result = match fill.call1((arr,)) {
                Ok(v) => v,
                Err(err) => {
                    py_err = Some(err);
                    return f64::NAN;
                }
            };
            let (energy, g) = match result.extract::<(f64, PyReadonlyArray1<f64>)>() {
                Ok(pair) => pair,
                Err(err) => {
                    py_err = Some(err);
                    return f64::NAN;
                }
            };
            let src = match g.as_slice() {
                Ok(s) => s,
                Err(_) => return f64::NAN,
            };
            let n = grad.len().min(src.len());
            grad[..n].copy_from_slice(&src[..n]);
            energy
        },
        &opts,
    );
    if let Some(err) = py_err {
        return Err(err);
    }
    let py_x = PyArray1::from_vec(py, state.to_vec());
    Ok((
        report.energy,
        report.grad,
        report.evals,
        report.converged,
        py_x,
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLbfgs>()?;
    m.add_function(wrap_pyfunction!(minimize_eon_cg, m)?)?;
    Ok(())
}
