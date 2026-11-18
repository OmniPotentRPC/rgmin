//! Unit sphere \(S^{n-1}\). manopt_cpp `Sphere`.
//!
//! Projection \(v - (x\cdot v)x\). Retraction \((x+v)/\|x+v\|\).
//! Transport is projection at the arrival point.
//!
//! The geodesic quantities are manopt `spherefactory`: the exponential
//! \(x\cos\|v\| + v\,\mathrm{sinc}\|v\|\), the distance
//! \(2\arcsin(\|x-y\|/2)\), parallel transport along the great circle,
//! and the Hessian \(P_x(\nabla^2 f\,v - (x^\top \nabla f)\,v)\). The
//! free functions act on one unit vector so that the oblique manifold,
//! the complex circle and the complex sphere reuse them blockwise.
//!
//! The Hessian keeps the Weingarten term \(-(x^\top\nabla f)\,v\)
//! explicit, so it holds for any smooth extension of the cost off the
//! sphere. If \(h\) vanishes on the sphere then \(\nabla h = \mu x\)
//! there, and \(P_x\nabla^2 h\,v = \mu v\) for tangent \(v\), which the
//! term removes. The degree-zero extension \(f(x/\|x\|)\) has
//! \(x^\top\nabla f = 0\) on the sphere, where the projected Hessian
//! alone is exact. The session's dogleg and Newton-CG steps model the
//! Hessian that way, without the term; [`crate::rtr::minimize_rtr`]
//! calls this one.

use ndarray::{Array1, ArrayView1};

use super::Manifold;

/// Unit sphere in the ambient Euclidean metric.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sphere;

fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(u, v)| u * v).sum()
}

fn nrm(a: &Array1<f64>) -> f64 {
    dot(a, a).sqrt()
}

fn vdot(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(u, v)| u * v).sum()
}

/// \(\sin t / t\), with the series below the cancellation threshold.
pub(crate) fn sinc(t: f64) -> f64 {
    if t.abs() < 1e-4 {
        1.0 - t * t / 6.0
    } else {
        t.sin() / t
    }
}

/// \(P_x v = v - (x^\top v) x\) on one unit vector.
pub(crate) fn project_unit(x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
    let s = vdot(x, v);
    Array1::from_iter(x.iter().zip(v.iter()).map(|(xi, vi)| vi - s * xi))
}

/// Great-circle exponential \(x\cos\|v\| + v\,\mathrm{sinc}\|v\|\).
pub(crate) fn exp_unit(x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
    let t = vdot(v, v).sqrt();
    let (c, s) = (t.cos(), sinc(t));
    Array1::from_iter(x.iter().zip(v.iter()).map(|(xi, vi)| c * xi + s * vi))
}

/// Arc length \(2\arcsin(\|x-y\|/2)\), accurate for near and far points.
pub(crate) fn dist_unit(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    let chord: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    2.0 * (0.5 * chord).min(1.0).asin()
}

/// Tangent at `x` pointing to `y` with length [`dist_unit`].
pub(crate) fn log_unit(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Array1<f64> {
    let diff = &y - &x;
    let mut v = project_unit(x, diff.view());
    let nv = vdot(v.view(), v.view()).sqrt();
    if nv > 0.0 {
        v *= dist_unit(x, y) / nv;
    }
    v
}

/// Parallel transport of `v` along \(t \mapsto \exp_x(t d)\) to `t = 1`.
///
/// The component along \(u = d/\|d\|\) turns with the great circle and
/// the rest is unchanged: manopt `paralleltransp` with
/// \(\theta = \|d\|\).
pub(crate) fn transport_along_unit(
    x: ArrayView1<f64>,
    d: ArrayView1<f64>,
    v: ArrayView1<f64>,
) -> Array1<f64> {
    let theta = vdot(d, d).sqrt();
    if theta == 0.0 {
        return v.to_owned();
    }
    let along = vdot(d, v) / theta;
    let (sine, cosine) = theta.sin_cos();
    let du = (cosine - 1.0) * along / theta;
    let dx = sine * along;
    Array1::from_iter(
        v.iter()
            .zip(d.iter().zip(x.iter()))
            .map(|(vi, (di, xi))| vi + du * di - dx * xi),
    )
}

/// \(P_x(\nabla^2 f\,v - (x^\top \nabla f)\,v)\), manopt `ehess2rhess`.
pub(crate) fn hess_unit(
    x: ArrayView1<f64>,
    egrad: ArrayView1<f64>,
    ehess: ArrayView1<f64>,
    v: ArrayView1<f64>,
) -> Array1<f64> {
    let normal = vdot(x, egrad);
    let w = Array1::from_iter(ehess.iter().zip(v.iter()).map(|(h, vi)| h - normal * vi));
    project_unit(x, w.view())
}

fn same_len(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    !a.is_empty() && a.len() == b.len()
}

impl Manifold for Sphere {
    fn euclidean_normals(&self, x: &Array1<f64>) -> Option<ndarray::Array2<f64>> {
        if x.is_empty() || !x.iter().all(|value| value.is_finite()) {
            return None;
        }
        let norm_sq = x.iter().map(|value| value * value).sum::<f64>();
        if !(norm_sq.is_finite() && norm_sq > 0.0) {
            return None;
        }
        let inv = 1.0 / norm_sq.sqrt();
        let mut normals = ndarray::Array2::zeros((x.len(), 1));
        for (row, value) in x.iter().enumerate() {
            normals[(row, 0)] = value * inv;
        }
        Some(normals)
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        #[cfg(feature = "par")]
        if x.len() >= crate::vecops::PAR_MIN_LEN && x.len() == v.len() {
            let projection = crate::vecops::dot(x.view(), v.view());
            let mut out = v.clone();
            crate::vecops::axpy(-projection, x.view(), &mut out);
            return out;
        }
        let s = dot(x, v);
        Array1::from_iter(x.iter().zip(v.iter()).map(|(xi, vi)| vi - s * xi))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        #[cfg(feature = "par")]
        if x.len() >= crate::vecops::PAR_MIN_LEN && x.len() == v.len() {
            let mut y = x.clone();
            crate::vecops::axpy(1.0, v.view(), &mut y);
            let norm = crate::vecops::nrm2(y.view());
            if norm <= 1e-16 {
                let initial_norm = crate::vecops::nrm2(x.view());
                if initial_norm <= 1e-16 {
                    return x.clone();
                }
                let mut initial = x.clone();
                crate::vecops::scale(1.0 / initial_norm, &mut initial);
                return initial;
            }
            crate::vecops::scale(1.0 / norm, &mut y);
            return y;
        }
        let y = x + v;
        let n = nrm(&y);
        if n <= 1e-16 {
            let n0 = nrm(x);
            if n0 <= 1e-16 {
                return x.clone();
            }
            return x / n0;
        }
        y / n
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }

    fn ehess2rhess(
        &self,
        x: &Array1<f64>,
        egrad: &Array1<f64>,
        ehess: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        (same_len(x, egrad) && same_len(x, ehess) && same_len(x, v))
            .then(|| hess_unit(x.view(), egrad.view(), ehess.view(), v.view()))
    }

    fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
        same_len(x, v).then(|| exp_unit(x.view(), v.view()))
    }

    fn log(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
        same_len(x, y).then(|| log_unit(x.view(), y.view()))
    }

    fn dist(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
        same_len(x, y).then(|| dist_unit(x.view(), y.view()))
    }

    fn parallel_transport_along(
        &self,
        x: &Array1<f64>,
        d: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        (same_len(x, d) && same_len(x, v))
            .then(|| transport_along_unit(x.view(), d.view(), v.view()))
    }

    fn typical_dist(&self, _n: usize) -> f64 {
        std::f64::consts::PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn project_is_orthogonal_to_x() {
        let x = array![1.0, 0.0, 0.0];
        let v = array![2.0, 3.0, 4.0];
        let t = Sphere.project(&x, &v);
        assert!(dot(&x, &t).abs() < 1e-15);
        assert!((t[1] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn retract_stays_on_the_sphere() {
        let x = array![0.0, 1.0, 0.0];
        let v = array![0.1, 0.0, -0.2];
        let y = Sphere.retract(&x, &v);
        assert!((nrm(&y) - 1.0).abs() < 1e-14);
    }
    #[cfg(feature = "par")]
    #[test]
    fn par_retract_stays_on_the_sphere() {
        retract_stays_on_the_sphere();
    }

    #[test]
    fn exp_walks_the_great_circle_and_log_undoes_it() {
        let x = array![1.0, 0.0, 0.0];
        let v = array![0.0, 2.0, 0.0];
        let y = Sphere.exp(&x, &v).unwrap();
        assert!((y[0] - 2.0_f64.cos()).abs() < 1e-15);
        assert!((y[1] - 2.0_f64.sin()).abs() < 1e-15);
        let back = Sphere.log(&x, &y).unwrap();
        assert!((&back - &v).iter().all(|e| e.abs() < 1e-14), "{back:?}");
        assert!((Sphere.dist(&x, &y).unwrap() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn parallel_transport_keeps_lengths_and_carries_the_velocity() {
        let x = array![0.0, 0.6, 0.8];
        let d = Sphere.project(&x, &array![0.3, 0.2, -0.4]);
        let v = Sphere.project(&x, &array![-0.5, 0.7, 0.1]);
        let y = Sphere.exp(&x, &d).unwrap();
        let w = Sphere.parallel_transport_along(&x, &d, &v).unwrap();
        assert!(dot(&y, &w).abs() < 1e-14, "tangent at the end");
        assert!((dot(&w, &w) - dot(&v, &v)).abs() < 1e-14, "isometric");
        let velocity = Sphere.parallel_transport_along(&x, &d, &d).unwrap();
        let t = nrm(&d);
        let h = 1e-6;
        let ahead = Sphere.exp(&x, &(&d * (1.0 + h / t))).unwrap();
        let behind = Sphere.exp(&x, &(&d * (1.0 - h / t))).unwrap();
        let tangent = (&ahead - &behind) * (t / (2.0 * h));
        assert!((&velocity - &tangent).iter().all(|e| e.abs() < 1e-8));
    }

    #[test]
    fn hessian_adds_the_weingarten_term() {
        let x = array![0.0, 0.0, 1.0];
        let egrad = array![0.0, 0.0, 3.0];
        let ehess = array![1.0, 2.0, 5.0];
        let v = array![1.0, 0.0, 0.0];
        let h = Sphere.ehess2rhess(&x, &egrad, &ehess, &v).unwrap();
        assert!((h[0] - (1.0 - 3.0)).abs() < 1e-15);
        assert!((h[1] - 2.0).abs() < 1e-15);
        assert!(h[2].abs() < 1e-15);
    }

    #[test]
    fn normalization_retraction_agrees_with_the_geodesic_through_second_order() {
        // x = (1, 0), v = (0, a). The degree-2 Taylor of both maps is
        // (1 - a^2/2, a). The gap is cubic: sin a - a (1+a^2)^{-1/2} ~ -a^3/3.
        let x = array![1.0, 0.0];
        let direction = array![0.0, 1.0];
        let quadratic = |a: f64| array![1.0 - 0.5 * a * a, a];
        let gap = |a: f64| {
            let step = &direction * a;
            let retracted = Sphere.retract(&x, &step);
            let geodesic = Sphere.exp(&x, &step).expect("sphere exponential");
            let q = quadratic(a);
            let retract_gap = nrm(&(&retracted - &q));
            let exp_gap = nrm(&(&geodesic - &q));
            let between = nrm(&(&retracted - &geodesic));
            (retract_gap, exp_gap, between)
        };
        let a = 1.0 / 16.0;
        let (retract_gap, exp_gap, between) = gap(a);
        let cubic = a * a * a;
        assert!(retract_gap <= cubic);
        assert!(exp_gap <= cubic);
        assert!(between <= cubic);
        let a_half = 1.0 / 32.0;
        let between_half = gap(a_half).2;
        let ratio = between_half / between;
        assert!((ratio - (1.0 / 8.0)).abs() < a);
    }
}
