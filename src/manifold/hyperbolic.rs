//! Hyperboloid model of \(H^{n-1}\). manopt `hyperbolicfactory`.
//!
//! A point is a length-`n` vector (`n >= 2`) on the upper sheet
//! \(\{ x : \langle x,x\rangle_L = -1,\ x_0 > 0 \}\) of the Minkowski
//! form \(\langle u,v\rangle_L = -u_0 v_0 + \sum_{i\ge 1} u_i v_i\).
//! Packed as time-like \(x_0\) then the space-like tail. This is not
//! the unit sphere and not a 3N cluster packing: a molecule lives on
//! [`super::RigidQuotient`].
//!
//! Projection \(v + \langle x,v\rangle_L\, x\). Retraction is the
//! exponential map \(\cosh\|v\|_L\, x + \sinh(\|v\|_L)/\|v\|_L\, v\).
//! Transport is projection at the arrival point. Ambient reductions
//! go through [`crate::vecops`] (dlpk / rayon under `par`).

use ndarray::{s, Array1, ArrayView1};

use crate::vecops;

use super::Manifold;

/// Hyperboloid \(H^{n-1}\) in the ambient Lorentz metric.
#[derive(Clone, Copy, Debug, Default)]
pub struct Hyperbolic;

/// Minkowski / Lorentz inner product on the ambient packing.
pub fn minkowski(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    vecops::dot(x, y) - 2.0 * x[0] * y[0]
}

/// Time-like component and space-like tail of a packed point.
pub fn unpack(x: &Array1<f64>) -> (f64, Array1<f64>) {
    if x.is_empty() {
        return (0.0, Array1::zeros(0));
    }
    (x[0], x.slice(s![1..]).to_owned())
}

/// Pack a time-like scalar in front of a space-like vector.
pub fn pack(time: f64, space: ArrayView1<f64>) -> Array1<f64> {
    let mut y = Array1::zeros(space.len() + 1);
    y[0] = time;
    y.slice_mut(s![1..]).assign(&space);
    y
}

fn onto_sheet(y: Array1<f64>) -> Array1<f64> {
    if y.len() < 2 {
        return y;
    }
    let q = -minkowski(y.view(), y.view());
    if q <= 1e-16 {
        let mut origin = Array1::zeros(y.len());
        origin[0] = 1.0;
        return origin;
    }
    let mut z = y;
    let scale = 1.0 / q.sqrt();
    z.mapv_inplace(|c| c * scale);
    if z[0] < 0.0 {
        z.mapv_inplace(|c| -c);
    }
    z
}

impl Manifold for Hyperbolic {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n >= 2 {
            Ok(())
        } else {
            Err(2)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != v.len() || x.is_empty() {
            return v.clone();
        }
        let s = minkowski(x.view(), v.view());
        let mut out = v.clone();
        vecops::axpy(s, x.view(), &mut out);
        out
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != v.len() || x.len() < 2 {
            return x.clone();
        }
        let nrm = minkowski(v.view(), v.view()).max(0.0).sqrt();
        if nrm <= 1e-16 {
            return onto_sheet(x.clone());
        }
        let mut y = x * nrm.cosh();
        vecops::axpy(nrm.sinh() / nrm, v.view(), &mut y);
        onto_sheet(y)
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        if egrad.is_empty() {
            return egrad.clone();
        }
        let mut identified = egrad.clone();
        identified[0] = -identified[0];
        self.project(x, &identified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn on_sheet(x: &Array1<f64>) -> bool {
        x.len() >= 2 && x[0] > 0.0 && (minkowski(x.view(), x.view()) + 1.0).abs() < 1e-12
    }

    #[test]
    fn retract_stays_on_the_hyperboloid() {
        let x = pack(2.0_f64.sqrt(), array![1.0, 0.0].view());
        let v = Hyperbolic.project(&x, &array![0.1, 0.3, -0.2]);
        let y = Hyperbolic.retract(&x, &v);
        assert!(on_sheet(&y), "left the hyperboloid {y:?}");
        assert_eq!(y.len(), 3);
    }

    #[test]
    fn project_is_minkowski_orthogonal() {
        let x = pack((1.0_f64 + 0.25 + 0.25).sqrt(), array![0.5, 0.5].view());
        let x = onto_sheet(x);
        let v = array![0.4, -0.2, 0.7];
        let t = Hyperbolic.project(&x, &v);
        assert!(minkowski(x.view(), t.view()).abs() < 1e-14);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let space = array![0.3, -0.4, 0.5];
        let x = pack(2.0, space.view());
        let (t, tail) = unpack(&x);
        assert!((t - 2.0).abs() < 1e-15);
        assert_eq!(tail, space);
        assert_eq!(x.len(), 4);
    }

    #[test]
    fn transport_is_tangent_at_arrival() {
        let x = onto_sheet(array![2.0, 1.0, 1.0]);
        let y = Hyperbolic.retract(&x, &Hyperbolic.project(&x, &array![0.2, 0.0, -0.1]));
        let w = Hyperbolic.transport(&x, &y, &array![0.3, 0.1, -0.4]);
        assert!(minkowski(y.view(), w.view()).abs() < 1e-12);
        assert!(on_sheet(&y));
    }

    #[test]
    fn egrad2rgrad_is_minkowski_not_euclidean() {
        let x = onto_sheet(array![2.0, 1.0, 1.0]);
        let egrad = array![0.5, 0.1, -0.2];
        let r = Hyperbolic.egrad2rgrad(&x, &egrad);
        let mut identified = egrad.clone();
        identified[0] = -identified[0];
        let want = Hyperbolic.project(&x, &identified);
        assert!((&r - &want).mapv(f64::abs).sum() < 1e-14);
        assert!(minkowski(x.view(), r.view()).abs() < 1e-14);
        let naive = Hyperbolic.project(&x, &egrad);
        assert!(vecops::nrm2((&r - &naive).view()) > 1e-8);
    }

    #[test]
    fn geometry_is_not_the_sphere() {
        use crate::manifold::Sphere;
        let x = onto_sheet(array![1.5, 0.8, 0.6]);
        let v = Hyperbolic.project(&x, &array![0.2, -0.1, 0.3]);
        let yh = Hyperbolic.retract(&x, &v);
        let ys = Sphere.retract(&x, &v);
        let gap = vecops::nrm2((&yh - &ys).view());
        assert!(gap > 1e-6, "hyperboloid collapsed onto the sphere");
        assert!(on_sheet(&yh));
        let sphere_n = vecops::nrm2(ys.view());
        assert!((sphere_n - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_3n_cluster_is_not_reinterpreted() {
        // Six numbers can sit on H^5. They are not a two-atom cluster.
        let mut x = Array1::from_elem(6, 0.2);
        x[0] = (1.0 + 5.0 * 0.04).sqrt();
        let v = Hyperbolic.project(&x, &Array1::from_elem(6, 0.05));
        let y = Hyperbolic.retract(&x, &v);
        assert_eq!(y.len(), 6);
        assert!(on_sheet(&y));
        assert!(Hyperbolic.required_dim(6).is_ok());
        assert!(Hyperbolic.required_dim(1).is_err());
        assert!(Hyperbolic.required_dim(2).is_ok());
    }
}
