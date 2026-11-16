//! Dense Sella trust restriction over the shared signed QN family.

use ndarray::{Array1, Array2};

use crate::error::{Error, Result};
use crate::vecops::{dot, nrm2};

const CONS_FLOOR: f64 = 1e-12;

/// Sella `TrustRegion`: `cons(s) = ||s||`, target radius `delta`.
///
/// Distinct from IRCTrustRegion, which constrains
/// `||(s + d1) * sqrt(m)||`. This type never mass-weights the step
/// and never shifts it by an IRC tangent.
#[derive(Clone, Debug)]
pub struct TrustRegion {
    /// Trust radius. The accepted step satisfies `||s|| <= delta`.
    pub delta: f64,
    /// Number of Hessian modes whose curvature sign is flipped
    /// (0 = minimum, 1 = first-order saddle).
    pub order: usize,
    /// Absolute tolerance on `||s|| - delta`.
    pub tol: f64,
    /// Newton / bisection iterations.
    pub maxiter: usize,
}

/// Accepted restricted step.
///
/// `cons` is `||s||` when the unconstrained QN step is inside the
/// region, and `delta` when the solver sat on the bound.
#[derive(Clone, Debug)]
pub struct RestrictedStep {
    /// Displacement `s`.
    pub step: Array1<f64>,
    /// Constraint value Sella `get_s` reports: `||s||` or `delta`.
    pub cons: f64,
}

impl RestrictedStep {
    /// Euclidean length of the step (vecops `nrm2`).
    pub fn nrm2(&self) -> f64 {
        nrm2(self.step.view())
    }
}

impl TrustRegion {
    /// Radius `delta`, minimum-mode QN (`order = 0`).
    pub fn new(delta: f64) -> Self {
        Self {
            delta: if delta.is_finite() {
                delta.max(0.0)
            } else {
                delta
            },
            order: 0,
            tol: 1e-10,
            maxiter: 1000,
        }
    }

    /// Flip the lowest `order` Hessian modes, Sella `order`.
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Restrict the Sella quasi-Newton family `s(α)` to the trust sphere.
    ///
    /// `α = 0` is the unconstrained QN step. When that step is longer
    /// than `delta`, a Newton-bisection hybrid solves `||s(α)|| = delta`
    /// for `α > 0` (Baker 1986 / Sella `TrustRegion.get_s`).
    pub fn restrict_qn(&self, hess: &Array2<f64>, grad: &Array1<f64>) -> Result<RestrictedStep> {
        let n = grad.len();
        if hess.nrows() != n || hess.ncols() != n {
            return Err(Error::Dim {
                got: hess.nrows(),
                dim: n,
            });
        }
        if !self.delta.is_finite()
            || self.delta < 0.0
            || !self.tol.is_finite()
            || self.tol < 0.0
            || hess.iter().chain(grad.iter()).any(|x| !x.is_finite())
        {
            return Err(Error::RestrictedStep);
        }
        if self.delta == 0.0 {
            return Ok(RestrictedStep {
                step: Array1::zeros(n),
                cons: 0.0,
            });
        }
        let (values, vectors) = crate::hvp::sym_eig_jacobi(hess.clone());
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::zeros((n, n));
        for (column, &index) in indices.iter().enumerate() {
            eigenvalues[column] = values[index];
            eigenvectors
                .column_mut(column)
                .assign(&vectors.column(index));
        }
        let vg = eigenvectors.t().dot(grad);
        let (step, cons) = restrict(
            |alpha| {
                let (s, dsda) = crate::qn_irc::qn_get_s_proj(
                    &eigenvalues,
                    &eigenvectors,
                    vg.view(),
                    self.order,
                    alpha,
                );
                let (val, dval) = trust_cons(&s, &dsda);
                (s, val, dval)
            },
            RestrictParams {
                alpha0: 0.0,
                alphamin: 0.0,
                alphamax: f64::INFINITY,
                slope: -1.0,
                newton_safe: true,
                delta: self.delta,
                tol: self.tol,
                maxiter: self.maxiter,
            },
        )?;
        let mut step = step;
        let norm = nrm2(step.view());
        if !norm.is_finite() {
            return Err(Error::RestrictedStep);
        }
        if norm > self.delta {
            // Root tolerance does not relax the geometric radius. Rounding
            // the projection scale inward leaves room for its final multiply.
            let scale = (self.delta / norm).next_down().max(0.0);
            step.mapv_inplace(|x| x * scale);
        }
        let norm = nrm2(step.view());
        if !norm.is_finite() || norm > self.delta {
            return Err(Error::RestrictedStep);
        }
        Ok(RestrictedStep { step, cons })
    }
}

/// Sella `TrustRegion.cons`: `||s||` and `d||s||/dα = (ds/dα · s) / ||s||`.
fn trust_cons(s: &Array1<f64>, dsda: &Array1<f64>) -> (f64, f64) {
    let val = nrm2(s.view());
    let dval = dot(dsda.view(), s.view()) / val.max(CONS_FLOOR);
    (val, dval)
}

struct RestrictParams {
    alpha0: f64,
    alphamin: f64,
    alphamax: f64,
    slope: f64,
    newton_safe: bool,
    delta: f64,
    tol: f64,
    maxiter: usize,
}

/// Sella `BaseRestrictedStep.get_s` on a one-parameter family.
fn restrict(
    mut eval: impl FnMut(f64) -> (Array1<f64>, f64, f64),
    p: RestrictParams,
) -> Result<(Array1<f64>, f64)> {
    let mut lower = p.alphamin;
    let mut upper = p.alphamax;
    let mut alpha = p.alpha0;
    let (mut s, mut val, mut dval) = eval(alpha);
    if !val.is_finite() {
        return Err(Error::RestrictedStep);
    }
    if val < p.delta {
        return Ok((s, val));
    }
    let mut err = val - p.delta;
    for niter in 0..p.maxiter {
        if err.abs() <= p.tol {
            return Ok((s, p.delta));
        }
        if lower.next_up() >= upper {
            return Ok((s, p.delta));
        }
        if err * p.slope > 0.0 {
            upper = alpha;
        } else {
            lower = alpha;
        }
        let a1 = alpha - err / dval;
        alpha = if !a1.is_finite() || a1 <= lower || a1 >= upper || (niter > 4 && !p.newton_safe) {
            let a2 = 0.5 * (lower + upper);
            if a2.is_infinite() {
                alpha + 1.0_f64.max(0.5 * alpha) * a2.signum()
            } else {
                a2
            }
        } else {
            a1
        };
        let next = eval(alpha);
        s = next.0;
        val = next.1;
        dval = next.2;
        if !val.is_finite() {
            return Err(Error::RestrictedStep);
        }
        err = val - p.delta;
    }
    Err(Error::RestrictedStep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn qn_step_inside_a_large_radius_is_unconstrained() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let acc = TrustRegion::new(10.0).restrict_qn(&h, &g).unwrap();
        assert!((acc.step[0] + 1.0).abs() < 1e-10);
        assert!(acc.step[1].abs() < 1e-10);
        assert!(acc.nrm2() <= 10.0);
        assert!(acc.nrm2() < 10.0 - 1e-6);
        assert!((acc.cons - acc.nrm2()).abs() < 1e-12);
    }

    #[test]
    fn qn_step_sits_on_the_bound_when_newton_is_longer() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let delta = 0.1;
        let acc = TrustRegion::new(delta).restrict_qn(&h, &g).unwrap();
        let n = acc.nrm2();
        assert!(n <= delta + 1e-10);
        assert!((n - delta).abs() < 1e-9);
        assert!((acc.cons - delta).abs() < 1e-14);
        assert!(acc.step[0] < 0.0);
    }

    #[test]
    fn retract_of_the_restricted_step_stays_on_the_trust_ball() {
        use crate::manifold::{Euclidean, Manifold};
        let h = Array2::<f64>::eye(3) * 2.0;
        let g = array![2.0, -1.0, 0.5];
        let delta = 0.2;
        let x = array![1.0, -2.0, 0.5];
        let acc = TrustRegion::new(delta).restrict_qn(&h, &g).unwrap();
        assert!(acc.nrm2() <= delta + 1e-10);
        let y = Euclidean.retract(&x, &acc.step);
        let v = Euclidean.project(&x, &(&y - &x));
        assert!(nrm2(v.view()) <= delta + 1e-10);
        let t = Euclidean.transport(&x, &y, &acc.step);
        assert!((nrm2(t.view()) - acc.nrm2()).abs() < 1e-12);
    }

    #[test]
    fn cons_is_euclidean_norm_not_mass_weighted_irc() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let acc = TrustRegion::new(0.1).restrict_qn(&h, &g).unwrap();
        assert!((acc.nrm2() - 0.1).abs() < 1e-9);
        let irc = (acc.step[0] * 10.0).hypot(acc.step[1]);
        assert!((irc - 0.1).abs() > 0.5);
    }

    #[test]
    fn saddle_order_flips_the_lowest_mode() {
        let h = array![[-2.0, 0.0], [0.0, 2.0]];
        let g = array![2.0, 2.0];
        let free = TrustRegion::new(10.0)
            .with_order(1)
            .restrict_qn(&h, &g)
            .unwrap();
        assert!((free.step[0] - 1.0).abs() < 1e-8);
        assert!((free.step[1] + 1.0).abs() < 1e-8);
        let bound = TrustRegion::new(0.2)
            .with_order(1)
            .restrict_qn(&h, &g)
            .unwrap();
        assert!(bound.nrm2() <= 0.2 + 1e-10);
        assert!((bound.nrm2() - 0.2).abs() < 1e-8);
    }

    #[test]
    fn a_stationary_point_returns_the_zero_step() {
        let h = Array2::<f64>::eye(2);
        let g = array![0.0, 0.0];
        let acc = TrustRegion::new(0.5).restrict_qn(&h, &g).unwrap();
        assert!(acc.nrm2() < 1e-14);
    }

    #[test]
    fn hessian_dimension_mismatch_is_dim() {
        let h = Array2::<f64>::eye(2);
        let g = array![1.0, 2.0, 3.0];
        match TrustRegion::new(1.0).restrict_qn(&h, &g).unwrap_err() {
            Error::Dim { got: 2, dim: 3 } => {}
            other => panic!("{other:?}"),
        }
    }

    fn same_bits(a: &Array1<f64>, b: &Array1<f64>) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "[{i}]");
        }
    }

    /// On `H = c I` the trust step is `-g/c`, or that direction scaled onto the sphere.
    fn isotropic_step(g: &Array1<f64>, curvature: f64, delta: f64) -> Array1<f64> {
        let newton: Array1<f64> = g.mapv(|v| -v / curvature);
        let length = nrm2(newton.view());
        if length <= delta {
            newton
        } else {
            newton.mapv(|v| v * (delta / length))
        }
    }

    #[test]
    fn a_diagonal_trust_step_matches_the_closed_form() {
        for n in [1usize, 3] {
            let h = Array2::<f64>::eye(n) * 2.0;
            let g = Array1::from_shape_fn(n, |i| if i == 0 { 2.0 } else { 0.0 });
            let inside = TrustRegion::new(10.0).restrict_qn(&h, &g).unwrap();
            let bound = TrustRegion::new(0.1).restrict_qn(&h, &g).unwrap();
            let again = TrustRegion::new(0.1).restrict_qn(&h, &g).unwrap();
            let want_in = isotropic_step(&g, 2.0, 10.0);
            let want_out = isotropic_step(&g, 2.0, 0.1);
            for i in 0..n {
                assert!(
                    (inside.step[i] - want_in[i]).abs() <= 1e-12,
                    "inside {n} [{i}] {} vs {}",
                    inside.step[i],
                    want_in[i]
                );
                assert!(
                    (bound.step[i] - want_out[i]).abs() <= 1e-12,
                    "bound {n} [{i}] {} vs {}",
                    bound.step[i],
                    want_out[i]
                );
            }
            same_bits(&bound.step, &again.step);
            assert_eq!(bound.cons.to_bits(), again.cons.to_bits());
        }
        let zero = TrustRegion::new(0.0)
            .restrict_qn(&Array2::<f64>::eye(1), &array![4.0])
            .unwrap();
        assert_eq!(zero.step[0].to_bits(), 0.0f64.to_bits());
        let empty_h = Array2::<f64>::zeros((0, 0));
        let empty_g = Array1::<f64>::zeros(0);
        let empty = TrustRegion::new(1.0).restrict_qn(&empty_h, &empty_g);
        assert!(empty.is_ok());
        let bad = array![f64::NAN, 1.0];
        assert!(
            TrustRegion::new(1.0)
                .restrict_qn(&Array2::eye(2), &bad)
                .is_err()
        );
    }
}
