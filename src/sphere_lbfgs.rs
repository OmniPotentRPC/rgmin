//! L-BFGS on the unit sphere with a differentiated normalization retraction.

use eindir_core::DifferentiableObjective;
use ndarray::Array1;

use crate::control::Control;
use crate::lbfgs::Lbfgs;
use crate::linesearch::LineSearch;
use crate::manifold::{Manifold, Sphere};
use crate::vecops::nrm2;

/// Parallel transport along the short great-circle arc. A normalization
/// retraction from a tangent step stays in the origin's open hemisphere.
fn transport(origin: &Array1<f64>, point: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    v - (v.dot(point) / (1.0 + origin.dot(point))) * (origin + point)
}

/// The line search sees f(z / |z|), whose gradient is P grad(f) / |z|.
/// Every physical objective call is on the sphere. Curvature pairs live
/// in the accepted point's tangent space, including the retained pairs.
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
        let mut last = (origin.clone(), origin.clone(), value, gradient.clone());
        let (raw, _, _) = linesearch.search(
            |z| {
                if z == origin.view() {
                    return (value, gradient.clone());
                }
                let norm = nrm2(z);
                if !norm.is_finite() || norm <= f64::MIN_POSITIVE {
                    return (f64::INFINITY, Array1::from_elem(z.len(), f64::NAN));
                }
                let point = &z / norm;
                let bounded = objective.bounds().clip(point.view());
                let outside_step = control
                    .maxmove
                    .is_some_and(|cap| nrm2((&point - origin).view()) > cap);
                if bounded != point || outside_step {
                    return (f64::INFINITY, Array1::from_elem(z.len(), f64::NAN));
                }
                let (f, eg) = objective.value_and_gradient(point.view());
                let rg = Sphere.project(&point, &eg);
                last = (z.to_owned(), point, f, rg.clone());
                (f, rg / norm)
            },
            origin.view(),
            direction.view(),
            trial_step,
        );
        if raw == *origin {
            if solver.is_empty() {
                return (origin.clone(), value, gradient.clone());
            }
            solver.forget();
            direction = -gradient;
            trial_step = control.istep;
            continue;
        }
        let (point, f, rg) = if raw == last.0 {
            (last.1, last.2, last.3)
        } else {
            let point = &raw / nrm2(raw.view());
            let (f, eg) = objective.value_and_gradient(point.view());
            let rg = Sphere.project(&point, &eg);
            (point, f, rg)
        };
        if !f.is_finite() || !rg.iter().all(|g| g.is_finite()) || f > value {
            return (origin.clone(), value, gradient.clone());
        }
        solver.transport(|vector| transport(origin, &point, vector));
        let cosine = origin.dot(&point);
        let backwards = origin - cosine * &point;
        let sine = nrm2(backwards.view());
        let displacement = if sine > 0.0 {
            -(sine.atan2(cosine) / sine) * backwards
        } else {
            Array1::zeros(point.len())
        };
        let change = &rg - &transport(origin, &point, gradient);
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
        let origin = array![1.0, 0.0, 0.0];
        let point = array![0.6, 0.8, 0.0];
        let s = array![0.0, 3.0, 4.0];
        let y = array![0.0, -2.0, 5.0];
        let moved_s = transport(&origin, &point, &s);
        let moved_y = transport(&origin, &point, &y);
        assert!(point.dot(&moved_s).abs() < 1e-14);
        assert!(point.dot(&moved_y).abs() < 1e-14);
        assert!((moved_s.dot(&moved_s) - s.dot(&s)).abs() < 1e-14);
        assert!((moved_s.dot(&moved_y) - s.dot(&y)).abs() < 1e-14);
    }
}
