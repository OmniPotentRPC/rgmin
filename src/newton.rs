//! Shifted Newton and Banerjee RFO on a caller-supplied dense Hessian.
//!
//! This is not L-BFGS. When the Hessian is cheap (analytic pair potential,
//! model Hessian), Nocedal and Wright take a Newton step. Geometry codes
//! stabilize that step with a Levenberg shift or with rational function
//! optimization.
//!
//! Banerjee, Adams, Simons, Shepard, *Search for stationary points on
//! surfaces*, <https://doi.org/10.1021/j100247a015>.
//! Baker, *An algorithm for the location of transition states*,
//! <https://doi.org/10.1002/jcc.540070402>.
//! Nocedal and Wright, *Numerical Optimization*,
//! <https://doi.org/10.1007/978-0-387-40065-5>.

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, Array2, ArrayView1};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::report::Report;
use crate::step::{l2, scale_step};

const ENERGY_RISE: f64 = 1.0e-8;
const EIG_FLOOR: f64 = 1.0e-8;
const LDLT_PIVOT: f64 = 1.0e-18;

/// Dense Hessian at a point. Row-major `n × n`, symmetric.
pub trait HessianObjective: DifferentiableObjective<f64> {
    /// Hessian `∇²f(x)`.
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64>;
}

/// How the Newton direction is formed from `H` and `g`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NewtonKind {
    /// `d = -(H + μ I)^{-1} g` with `μ = max(0, ε - λ̂)` via LDLT.
    Shifted,
    /// Lowest mode of the Banerjee augmented Hessian.
    Rfo,
}

/// Closure adapter: fused `(f, g)` plus a dense Hessian.
pub struct HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    f: F,
    hess: H,
    bounds: eindir_core::Bounds<f64>,
}

impl<F, H> HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    /// Wide-box oracle of dimension `dim`.
    pub fn unbounded(dim: usize, f: F, hess: H) -> Self {
        const LO: f64 = -1e12;
        const HI: f64 = 1e12;
        Self {
            f,
            hess,
            bounds: eindir_core::Bounds::new(
                Array1::from_elem(dim, LO),
                Array1::from_elem(dim, HI),
                0.0,
            ),
        }
    }
}

impl<F, H> eindir_core::Objective<f64> for HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (self.f)(x).0
    }
}

impl<F, H> eindir_core::Gradient<f64> for HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        (self.f)(x).1
    }
}

impl<F, H> DifferentiableObjective<f64> for HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.f)(x)
    }
}

impl<F, H> HessianObjective for HessianOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>) -> Array2<f64> + Send + Sync,
{
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64> {
        (self.hess)(x)
    }
}

/// Minimize with a Newton or RFO direction on the supplied Hessian.
pub fn minimize_newton<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    kind: NewtonKind,
) -> Result<Report>
where
    O: HessianObjective + ?Sized,
{
    let mut pos = init.into();
    if pos.len() != eindir_core::Objective::dim(obj) {
        return Err(Error::Dim {
            got: pos.len(),
            dim: eindir_core::Objective::dim(obj),
        });
    }
    pos = obj.bounds().clip(pos.view());
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(Report {
                value,
                coords: pos,
                steps: step,
                grad_norm: gnorm,
            });
        }
        let hess = obj.hessian(pos.view());
        let dir = match kind {
            NewtonKind::Shifted => shifted_newton(&hess, &grad),
            NewtonKind::Rfo => rfo_direction(&hess, &grad),
        };
        let (npos, nval, ngrad, moved) = energy_backtrack(obj, &pos, value, &dir, control);
        if !moved {
            let sd = grad.mapv(|g| -g);
            let (spos, sval, sgrad, sok) = energy_backtrack(obj, &pos, value, &sd, control);
            if sok {
                pos = spos;
                value = sval;
                grad = sgrad;
            }
        } else {
            pos = npos;
            value = nval;
            grad = ngrad;
        }
    }
    Ok(Report {
        value,
        coords: pos,
        steps: control.maxiter,
        grad_norm: l2(&grad),
    })
}

pub(crate) fn energy_backtrack<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    dir: &Array1<f64>,
    control: &Control,
) -> (Array1<f64>, f64, Array1<f64>, bool)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut alpha = 1.0;
    for _ in 0..10 {
        let mut trial = pos + &(dir * alpha);
        if let Some(cap) = control.maxmove {
            scale_step(pos, &mut trial, cap);
        }
        trial = obj.bounds().clip(trial.view());
        let (ft, gt) = obj.value_and_gradient(trial.view());
        if ft - value <= ENERGY_RISE {
            return (trial, ft, gt, true);
        }
        alpha *= 0.5;
    }
    (pos.clone(), value, Array1::zeros(pos.len()), false)
}

pub(crate) fn shifted_newton(h: &Array2<f64>, g: &Array1<f64>) -> Array1<f64> {
    let n = g.len();
    let mut mu = 0.0;
    for _ in 0..24 {
        let mut a = h.clone();
        for i in 0..n {
            a[(i, i)] += mu;
        }
        if let Some(d) = ldlt_solve(&a, g, false) {
            return -d;
        }
        mu = if mu == 0.0 { EIG_FLOOR } else { mu * 10.0 };
    }
    g.mapv(|v| -v)
}

pub(crate) fn rfo_direction(h: &Array2<f64>, g: &Array1<f64>) -> Array1<f64> {
    // Minimization selects the lowest eigenpair of [H g; g^T 0].
    // The stationary equation also has higher roots with uphill steps.
    crate::sella_step::rfo_get_s(h, g, 0, 1.0)
}

/// LDLT solve `A x = b`. `allow_indefinite` keeps negative pivots (RFO).
fn ldlt_solve(a: &Array2<f64>, b: &Array1<f64>, allow_indefinite: bool) -> Option<Array1<f64>> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    let mut d = Array1::<f64>::zeros(n);
    for j in 0..n {
        let mut dj = a[(j, j)];
        for k in 0..j {
            dj -= l[(j, k)] * l[(j, k)] * d[k];
        }
        if dj.abs() < LDLT_PIVOT {
            return None;
        }
        if !allow_indefinite && dj <= 0.0 {
            return None;
        }
        d[j] = dj;
        l[(j, j)] = 1.0;
        for i in (j + 1)..n {
            let mut lij = a[(i, j)];
            for k in 0..j {
                lij -= l[(i, k)] * l[(j, k)] * d[k];
            }
            l[(i, j)] = lij / dj;
        }
    }
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[(i, k)] * y[k];
        }
        y[i] = s;
    }
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        z[i] = y[i] / d[i];
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[(k, i)] * x[k];
        }
        x[i] = s;
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ldlt_recovers_identity() {
        let a = Array2::<f64>::eye(3);
        let b = array![1.0, 2.0, 3.0];
        let x = ldlt_solve(&a, &b, false).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }
}
