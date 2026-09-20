//! L-BFGS on the unit sphere with a great-circle line search.

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, array};

use crate::control::Control;
use crate::lbfgs::Lbfgs;
use crate::linesearch::LineSearch;
use crate::manifold::{Manifold, Sphere};
use crate::vecops::nrm2;

/// Parallel transport rotates the component along the geodesic and
/// preserves every component orthogonal to its plane.
fn transport(direction: &Array1<f64>, velocity: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    v + v.dot(direction) * (velocity - direction)
}

/// The scalar line search follows exp(origin, alpha * direction). Its
/// derivative uses the parallel-transported direction, so the accepted
/// displacement and gradient difference obey the same parameterization.
pub(crate) fn step<O>(
    objective: &O,
    origin: &Array1<f64>,
    value: f64,
    gradient: &Array1<f64>,
    solver: &mut Lbfgs,
    linesearch: LineSearch,
    control: &Control,
) -> (Array1<f64>, f64, Array1<f64>)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut direction = Sphere.project(
        origin,
        &solver.search_direction(origin.view(), gradient.view()),
    );
    if direction.dot(gradient) >= 0.0 {
        solver.forget();
        direction = -gradient;
    }
    let mut trial_step = if solver.is_empty() {
        control.istep
    } else {
        1.0
    };
    loop {
        let speed = nrm2(direction.view());
        if !speed.is_finite() || speed <= f64::MIN_POSITIVE {
            return (origin.clone(), value, gradient.clone());
        }
        let unit = &direction / speed;
        // With no curvature information, istep is an arc length. Scaling
        // the objective must not change the first geometric trial.
        let initial_parameter = if solver.is_empty() {
            trial_step / speed
        } else {
            trial_step
        };
        let geodesic = |alpha: f64| {
            let (sine, cosine) = (alpha * speed).sin_cos();
            (
                cosine * origin + sine * &unit,
                -sine * origin + cosine * &unit,
            )
        };
        let mut last = (0.0, origin.clone(), value, gradient.clone());
        let (parameter, _, _) = linesearch.search(
            |z| {
                let alpha = z[0];
                if alpha == 0.0 {
                    return (value, array![gradient.dot(&direction)]);
                }
                let angle = alpha * speed;
                // The exponential chart ends at the sphere's cut locus.
                if !angle.is_finite() || angle.abs() >= std::f64::consts::PI {
                    return (f64::INFINITY, array![f64::NAN]);
                }
                let (point, velocity) = geodesic(alpha);
                let bounded = objective.bounds().clip(point.view());
                let outside_step = control
                    .maxmove
                    .is_some_and(|cap| nrm2((&point - origin).view()) > cap);
                if bounded != point || outside_step {
                    return (f64::INFINITY, array![f64::NAN]);
                }
                let (f, eg) = objective.value_and_gradient(point.view());
                let rg = Sphere.project(&point, &eg);
                let derivative = speed * rg.dot(&velocity);
                last = (alpha, point, f, rg);
                (f, array![derivative])
            },
            array![0.0].view(),
            array![1.0].view(),
            initial_parameter,
        );
        let alpha = parameter[0];
        if alpha == 0.0 {
            if solver.is_empty() {
                return (origin.clone(), value, gradient.clone());
            }
            solver.forget();
            direction = -gradient;
            trial_step = control.istep;
            continue;
        }
        let (candidate, velocity) = geodesic(alpha);
        let (point, f, rg) = if alpha == last.0 {
            (last.1, last.2, last.3)
        } else {
            let point = candidate;
            let (f, eg) = objective.value_and_gradient(point.view());
            let rg = Sphere.project(&point, &eg);
            (point, f, rg)
        };
        if !f.is_finite() || !rg.iter().all(|g| g.is_finite()) || f > value {
            return (origin.clone(), value, gradient.clone());
        }
        solver.transport(|vector| transport(&unit, &velocity, vector));
        let displacement = (alpha * speed) * &velocity;
        let change = &rg - &transport(&unit, &velocity, gradient);
        solver.push_pair(displacement, change, Some(nrm2(rg.view())));
        return (point, f, rg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn transported_history_preserves_tangent_inner_products() {
        let point = array![0.6, 0.8, 0.0];
        let direction = array![0.0, 1.0, 0.0];
        let velocity = array![-0.8, 0.6, 0.0];
        let s = array![0.0, 3.0, 4.0];
        let y = array![0.0, -2.0, 5.0];
        let moved_s = transport(&direction, &velocity, &s);
        let moved_y = transport(&direction, &velocity, &y);
        assert!(point.dot(&moved_s).abs() < 1e-14);
        assert!(point.dot(&moved_y).abs() < 1e-14);
        assert!((moved_s.dot(&moved_s) - s.dot(&s)).abs() < 1e-14);
        assert!((moved_s.dot(&moved_y) - s.dot(&y)).abs() < 1e-14);
    }
}
