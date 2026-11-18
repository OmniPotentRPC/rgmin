//! Level-set constraint \(g(x) = c\) from objective value and gradient.

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};

/// Configuration of a level-set constraint \(g(x) = c\).
///
/// Non-exhaustive: start from [`Self::default`] (the level set at 0) or
/// [`Self::new`] and chain [`Self::with_target`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LevelSetConstraint {
    /// Target value \(c\). The default is 0.
    pub target: f64,
}

with_setters!(LevelSetConstraint {
    with_target => target: val f64,
});

impl LevelSetConstraint {
    /// Constraint at `target`.
    pub fn new(target: f64) -> Self {
        Self { target }
    }

    pub(crate) fn validate(self) -> Result<Self> {
        if self.target.is_finite() {
            Ok(self)
        } else {
            Err(Error::LevelSet {
                what: "target is not finite",
            })
        }
    }
}

/// Project `v` onto the tangent of the level set at a point with normal `grad`.
pub(crate) fn project_tangent(grad: ArrayView1<'_, f64>, v: &Array1<f64>) -> Result<Array1<f64>> {
    if grad.len() != v.len() {
        return Err(Error::LevelSet {
            what: "gradient length does not match vector length",
        });
    }
    let gn2 = crate::vecops::dot(grad, grad);
    if !(gn2.is_finite() && gn2 > 0.0) {
        return Err(Error::LevelSet {
            what: "level-set gradient is not finite and nonzero",
        });
    }
    let gv = crate::vecops::dot(grad, v.view());
    let scale = gv / gn2;
    Ok(Array1::from_iter(
        v.iter().zip(grad.iter()).map(|(vi, gi)| vi - scale * gi),
    ))
}

/// Absolute tolerance on `|g(x) - c|`, in the units of `g`.
///
/// `sqrt(eps) * (|c| + |g|)` is the value's rounding, loosened so the
/// last Newton correction is not required to undercut an ulp the oracle
/// cannot resolve. `||∇g|| * eps * (||x|| + 1)` is the change in `g`
/// from a one-ulp move in `x`: relative to `||x||`, plus one absolute
/// coordinate unit. `eps` is the absolute floor of a unit-scale energy.
pub(crate) fn correction_tolerance(
    target: f64,
    value: f64,
    grad: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
) -> f64 {
    let eps = f64::EPSILON;
    let sqrt_eps = eps.sqrt();
    let value_scale = target.abs() + value.abs();
    let grad_scale = crate::vecops::nrm2(grad);
    let x_scale = crate::vecops::nrm2(x);
    sqrt_eps * value_scale + grad_scale * eps * (x_scale + 1.0) + eps
}

#[derive(Debug)]
pub(crate) struct Retraction {
    pub(crate) point: Array1<f64>,
    pub(crate) value: f64,
    pub(crate) gradient: Array1<f64>,
}

#[allow(dead_code)]
pub(crate) fn retract_to_level_set<O>(
    obj: &O,
    cfg: LevelSetConstraint,
    start: &Array1<f64>,
) -> Result<Retraction>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    retract_to_level_set_fallible(cfg, start, |x| Ok(obj.value_and_gradient(x)))
}

pub(crate) fn retract_to_level_set_fallible<F>(
    cfg: LevelSetConstraint,
    start: &Array1<f64>,
    mut value_and_gradient: F,
) -> Result<Retraction>
where
    F: FnMut(ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>)>,
{
    let cfg = cfg.validate()?;
    let mut x = start.clone();
    let mut iter = 0usize;
    let mut previous_abs: Option<f64> = None;
    loop {
        let (value, gradient) = value_and_gradient(x.view())?;
        if !value.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(Error::LevelSet {
                what: "non-finite value or gradient during level-set retraction",
            });
        }
        let residual = value - cfg.target;
        let tol = correction_tolerance(cfg.target, value, gradient.view(), x.view());
        let residual_abs = residual.abs();
        if residual_abs <= tol {
            return Ok(Retraction {
                point: x,
                value,
                gradient,
            });
        }
        if let Some(previous) = previous_abs {
            // A strict contraction is progress even when its drop is
            // smaller than the current tolerance. Only compare evaluations
            // once a real previous residual exists.
            if residual_abs >= previous {
                return Err(Error::LevelSetStalled {
                    iterations: iter,
                    residual,
                    tolerance: tol,
                    previous_residual: previous,
                });
            }
        }
        let gn2 = crate::vecops::dot(gradient.view(), gradient.view());
        if !(gn2.is_finite() && gn2 > 0.0) {
            return Err(Error::LevelSet {
                what: "zero or non-finite level-set gradient during retraction",
            });
        }
        let step = residual / gn2;
        for (xi, gi) in x.iter_mut().zip(gradient.iter()) {
            *xi -= step * gi;
        }
        // If no coordinate bits changed, reevaluate the same point. That
        // gives the stall predicate a real previous residual; it also avoids
        // calling a strict contraction a stall merely because its next
        // correction rounded away.
        previous_abs = Some(residual_abs);
        iter += 1;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::{Array1, ArrayView1, array};

    use super::{LevelSetConstraint, correction_tolerance, retract_to_level_set};
    use crate::Error;

    struct Circle {
        log: Mutex<Vec<f64>>,
    }

    impl Objective<f64> for Circle {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static BOUNDS: OnceLock<Bounds<f64>> = OnceLock::new();
            BOUNDS.get_or_init(|| {
                Bounds::new(
                    Array1::from_elem(2, -f64::INFINITY),
                    Array1::from_elem(2, f64::INFINITY),
                    0.0,
                )
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.dot(&x)
        }
    }

    impl Gradient<f64> for Circle {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            x.mapv(|v| 2.0 * v)
        }
    }

    impl DifferentiableObjective<f64> for Circle {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            let value = self.eval(x);
            self.log.lock().expect("log").push((value - 1.0).abs());
            (value, self.grad(x))
        }
    }

    #[test]
    fn retraction_converges_quadratically_on_a_circle() {
        let objective = Circle {
            log: Mutex::new(Vec::new()),
        };
        let start = array![2.0, 0.0];
        let out = retract_to_level_set(&objective, LevelSetConstraint::new(1.0), &start).unwrap();
        let residual = (out.value - 1.0).abs();
        assert!(residual <= f64::EPSILON.sqrt() * (1.0 + out.value.abs()));
        let log = objective.log.lock().expect("log");
        let tail: Vec<f64> = log
            .iter()
            .copied()
            .filter(|r| r.is_finite() && *r < 1.0)
            .collect();
        assert!(tail.len() >= 3, "insufficient residual history: {tail:?}");
        for window in tail.windows(3) {
            let prev = window[0];
            let curr = window[1];
            let next = window[2];
            if prev > 0.0 && curr > 0.0 {
                let factor = next * prev / (curr * curr);
                assert!(
                    factor.is_finite() && factor < 0.5,
                    "not quadratic: {window:?}"
                );
            }
        }
    }

    struct Stuck;

    impl Objective<f64> for Stuck {
        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static BOUNDS: OnceLock<Bounds<f64>> = OnceLock::new();
            BOUNDS.get_or_init(|| Bounds::new(array![-f64::INFINITY], array![f64::INFINITY], 0.0))
        }
        fn eval(&self, _x: ArrayView1<f64>) -> f64 {
            1.0
        }
    }

    impl Gradient<f64> for Stuck {
        fn dim(&self) -> usize {
            1
        }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
            array![f64::EPSILON.sqrt()]
        }
    }

    impl DifferentiableObjective<f64> for Stuck {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    #[test]
    fn failed_retraction_returns_stalled_error() {
        let err =
            retract_to_level_set(&Stuck, LevelSetConstraint::new(0.0), &array![0.0]).unwrap_err();
        assert!(matches!(err, Error::LevelSetStalled { .. }), "{err:?}");
    }

    struct ScaledLinear {
        slope: f64,
    }

    impl Objective<f64> for ScaledLinear {
        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static BOUNDS: OnceLock<Bounds<f64>> = OnceLock::new();
            BOUNDS.get_or_init(|| Bounds::new(array![-f64::INFINITY], array![f64::INFINITY], 0.0))
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.slope * x[0]
        }
    }

    impl Gradient<f64> for ScaledLinear {
        fn dim(&self) -> usize {
            1
        }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
            array![self.slope]
        }
    }

    impl DifferentiableObjective<f64> for ScaledLinear {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    #[test]
    fn large_gradient_does_not_count_as_convergence() {
        // `2^40 * 2^{-30} = 2^10`. A tolerance that adds `||∇g||` in
        // energy units is about `10^7` here and would accept residual 1024.
        let slope = 2.0_f64.powi(40);
        let start = array![2.0_f64.powi(-30)];
        let out = retract_to_level_set(
            &ScaledLinear { slope },
            LevelSetConstraint::default(),
            &start,
        )
        .unwrap();
        // Retraction accepts an objective residual up to its derived
        // objective-unit tolerance. For g(x) = slope*x, dividing that
        // tolerance by |slope| gives the matching coordinate tolerance.
        let residual_tolerance =
            correction_tolerance(0.0, out.value, out.gradient.view(), out.point.view());
        let coordinate_tolerance = residual_tolerance / slope.abs();
        assert!(
            out.value.abs() <= residual_tolerance,
            "residual {} at {:?}",
            out.value,
            out.point
        );
        assert!(
            out.point[0].abs() <= coordinate_tolerance,
            "point {:?}",
            out.point
        );
    }

    struct Scripted {
        values: Mutex<Vec<f64>>,
    }

    impl Objective<f64> for Scripted {
        fn dim(&self) -> usize {
            1
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static BOUNDS: OnceLock<Bounds<f64>> = OnceLock::new();
            BOUNDS.get_or_init(|| Bounds::new(array![-f64::INFINITY], array![f64::INFINITY], 0.0))
        }
        fn eval(&self, _x: ArrayView1<f64>) -> f64 {
            let mut values = self.values.lock().expect("values");
            values.remove(0)
        }
    }

    impl Gradient<f64> for Scripted {
        fn dim(&self) -> usize {
            1
        }
        fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
            array![1.0]
        }
    }

    impl DifferentiableObjective<f64> for Scripted {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }

    #[test]
    fn near_set_contraction_is_not_a_stall() {
        // The second residual is still above the derived tolerance but
        // contracts by less than one tolerance. That strict decrease is
        // real Newton progress and must not be reported as a stall.
        let eps = f64::EPSILON;
        let objective = Scripted {
            values: Mutex::new(vec![8.0 * eps, 7.0 * eps, 0.0]),
        };
        let out =
            retract_to_level_set(&objective, LevelSetConstraint::new(0.0), &array![0.0]).unwrap();
        assert_eq!(out.value, 0.0);
    }
}
