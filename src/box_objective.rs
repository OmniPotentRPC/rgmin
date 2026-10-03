//! Extend an objective by infinity outside a session's coordinate box.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};

use crate::error::{Error, Result};

use crate::newton::HessianObjective;

pub(crate) struct BoxObjective<'a, O: ?Sized> {
    inner: &'a O,
    bounds: Bounds<f64>,
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> BoxObjective<'a, O> {
    pub(crate) fn new(inner: &'a O, lo: Option<Vec<f64>>, hi: Option<Vec<f64>>) -> Result<Self> {
        let dim = Objective::dim(inner);
        for side in [&lo, &hi].into_iter().flatten() {
            if !side.is_empty() && side.len() != 1 && side.len() != dim {
                return Err(Error::Dim {
                    got: side.len(),
                    dim,
                });
            }
        }
        let mut low = inner.bounds().low.clone();
        let mut high = inner.bounds().high.clone();
        if low.len() != dim || high.len() != dim {
            return Err(Error::Dim {
                got: low.len(),
                dim,
            });
        }
        for k in 0..dim {
            let requested_low = side_at(lo.as_deref(), k).unwrap_or(f64::NEG_INFINITY);
            let requested_high = side_at(hi.as_deref(), k).unwrap_or(f64::INFINITY);
            if !(requested_low <= requested_high) || !(low[k] <= high[k]) {
                return Err(Error::Highs("invalid coordinate box".into()));
            }
            low[k] = low[k].max(requested_low);
            high[k] = high[k].min(requested_high);
            if !(low[k] <= high[k]) || low[k] == f64::INFINITY || high[k] == f64::NEG_INFINITY {
                return Err(Error::Highs("invalid coordinate box".into()));
            }
        }
        Ok(Self {
            inner,
            bounds: Bounds::new(low, high, 0.0),
        })
    }

    fn limits(&self, k: usize) -> (f64, f64) {
        (self.bounds.low[k], self.bounds.high[k])
    }

    fn contains(&self, x: ArrayView1<f64>) -> bool {
        x.len() == Objective::dim(self.inner)
            && x.iter().enumerate().all(|(k, &value)| {
                let (lo, hi) = self.limits(k);
                value.is_finite() && lo <= value && value <= hi
            })
    }

    pub(crate) fn clip_start(&self, x: &mut Array1<f64>) -> Result<bool> {
        let dim = Objective::dim(self.inner);
        if x.len() != dim {
            return Err(Error::Dim { got: x.len(), dim });
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(Error::Highs("non-finite point in coordinate box".into()));
        }
        let mut changed = false;
        for (k, value) in x.iter_mut().enumerate() {
            let (lo, hi) = self.limits(k);
            let projected = value.clamp(lo, hi);
            changed |= projected != *value;
            *value = projected;
        }
        Ok(changed)
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> Objective<f64> for BoxObjective<'_, O> {
    fn dim(&self) -> usize {
        Objective::dim(self.inner)
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        if self.contains(x) {
            self.inner.eval(x)
        } else {
            f64::INFINITY
        }
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> Gradient<f64> for BoxObjective<'_, O> {
    fn dim(&self) -> usize {
        Objective::dim(self.inner)
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        if self.contains(x) {
            self.inner.grad(x)
        } else {
            Array1::from_elem(x.len(), f64::NAN)
        }
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> DifferentiableObjective<f64>
    for BoxObjective<'_, O>
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        if self.contains(x) {
            self.inner.value_and_gradient(x)
        } else {
            (f64::INFINITY, Array1::from_elem(x.len(), f64::NAN))
        }
    }
}

impl<O: HessianObjective + ?Sized> HessianObjective for BoxObjective<'_, O> {
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64> {
        if self.contains(x) {
            self.inner.hessian(x)
        } else {
            Array2::from_elem((x.len(), x.len()), f64::NAN)
        }
    }
}

/// A missing or empty side is unbounded; length one broadcasts.
pub(crate) fn side_at(side: Option<&[f64]>, k: usize) -> Option<f64> {
    match side {
        Some([value]) => Some(*value),
        Some(values) => values.get(k).copied(),
        None => None,
    }
}

/// Project a direction into the displacement box, retaining free coordinates.
pub(crate) fn project_direction(
    bounds: &Bounds<f64>,
    x: ArrayView1<f64>,
    grad: ArrayView1<f64>,
    mut direction: Array1<f64>,
) -> Array1<f64> {
    let project = |p: &mut Array1<f64>| {
        for k in 0..p.len() {
            p[k] = p[k].clamp(bounds.low[k] - x[k], bounds.high[k] - x[k]);
        }
    };
    project(&mut direction);
    if grad.dot(&direction) >= 0.0 {
        direction = grad.mapv(|value| -value);
        project(&mut direction);
    }
    direction
}

/// Remove outward wall reactions from a force or velocity.
pub(crate) fn project_tangent(bounds: &Bounds<f64>, x: &Array1<f64>, v: &mut Array1<f64>) {
    for k in 0..v.len() {
        if (x[k] <= bounds.low[k] && v[k] < 0.0) || (x[k] >= bounds.high[k] && v[k] > 0.0) {
            v[k] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn a_wall_preserves_the_free_direction_and_descent() {
        let bounds = Bounds::new(
            array![f64::NEG_INFINITY, 0.0],
            array![f64::INFINITY, f64::INFINITY],
            0.0,
        );
        for distance in [1e-8, 1e-12] {
            let x = array![3.0, distance];
            let gradient = array![3.0, 1.0 + distance];
            let step = project_direction(&bounds, x.view(), gradient.view(), -&gradient);
            assert!((step[0] + 3.0).abs() < 1e-14);
            assert!((step[1] + distance).abs() < 1e-16);
            assert!(gradient.dot(&step) < 0.0);
        }
        let mut model = crate::Lbfgs::default();
        model.record(array![1.0, 2.0], array![1.0, 0.0]);
        let bounds = Bounds::new(
            array![f64::NEG_INFINITY, f64::NEG_INFINITY],
            array![f64::INFINITY, 0.0],
            0.0,
        );
        let x = array![0.0, 0.0];
        let gradient = array![1.0, -1.0];
        let step = project_direction(
            &bounds,
            x.view(),
            gradient.view(),
            model.two_loop(gradient.view()),
        );
        assert!(gradient.dot(&step) < 0.0);
        assert!((step[0] + 1.0).abs() < 1e-14);
        assert_eq!(step[1], 0.0);
    }

    #[test]
    fn invalid_boxes_and_points_never_reach_the_callback() {
        let calls = AtomicUsize::new(0);
        let objective = crate::Oracle::unbounded(2, |x| {
            calls.fetch_add(1, Ordering::Relaxed);
            (x.dot(&x), 2.0 * &x)
        });
        for lo in [
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![1.0, 2.0, 3.0],
            vec![1e13],
        ] {
            assert!(BoxObjective::new(&objective, Some(lo), None).is_err());
        }
        let guarded = BoxObjective::new(&objective, Some(vec![0.0]), Some(vec![1.0])).unwrap();
        for x in [array![-1.0, 0.0], array![0.0, 2.0], array![f64::NAN, 0.0]] {
            assert_eq!(guarded.eval(x.view()), f64::INFINITY);
            assert!(guarded.grad(x.view()).iter().all(|v| v.is_nan()));
            let (value, gradient) = guarded.value_and_gradient(x.view());
            assert_eq!(value, f64::INFINITY);
            assert!(gradient.iter().all(|v| v.is_nan()));
        }
        assert!(guarded.clip_start(&mut array![f64::NAN, 0.0]).is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let mut start = array![-1.0, 2.0];
        assert!(guarded.clip_start(&mut start).unwrap());
        assert_eq!(start, array![0.0, 1.0]);
        assert_eq!(guarded.eval(start.view()), 1.0);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
