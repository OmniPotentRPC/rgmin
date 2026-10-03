//! Product of unit-modulus complex numbers \((S^1)^n\).
//!
//! manopt `complexcirclefactory(n)` (default \(m = 1\)): each entry
//! is a point of \(S^1 \subset \mathbb{C}\), identified with the unit
//! circle in \(\mathbb{R}^2\). Packed as interleaved `(re, im)` pairs,
//! length `2 n`. This is not the sphere \(S^{2n-1}\) (one constraint
//! per pair, not one on the whole vector) and not a 3N cluster.
//! Isolated molecules use [`super::RigidQuotient`].
//!
//! Projection is manopt `u - real(conj(u).*z).*z`. Retraction is
//! `sign(z+v)` (normalize each pair). Transport is projection at the
//! arrival point. Pair reductions go through [`crate::vecops`].

use ndarray::{Array1, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// \((S^1)^n\) as `n` interleaved real-imaginary pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComplexCircle {
    /// Number of unit-modulus complex entries. manopt `n` at `m = 1`.
    pub n: usize,
}

impl Default for ComplexCircle {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl ComplexCircle {
    /// `n` unit circles. Illegal `n == 0` fails [`Manifold::required_dim`].
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Packed length `2 n`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.n.checked_mul(2)
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 1 && self.packed_len() == Some(len)
    }

    /// Interleaved pack of real and imaginary parts (equal length).
    pub fn pack(re: ArrayView1<f64>, im: ArrayView1<f64>) -> Array1<f64> {
        let n = re.len().min(im.len());
        let mut out = Array1::zeros(2 * n);
        for k in 0..n {
            out[2 * k] = re[k];
            out[2 * k + 1] = im[k];
        }
        out
    }

    /// Split a packed vector into `(re, im)`. `None` if the length is odd.
    pub fn unpack(x: &Array1<f64>) -> Option<(Array1<f64>, Array1<f64>)> {
        if x.len() % 2 != 0 {
            return None;
        }
        let n = x.len() / 2;
        let mut re = Array1::zeros(n);
        let mut im = Array1::zeros(n);
        for k in 0..n {
            re[k] = x[2 * k];
            im[k] = x[2 * k + 1];
        }
        Some((re, im))
    }

    fn pair<'a>(&self, a: &'a [f64], k: usize) -> ArrayView1<'a, f64> {
        ArrayView1::from(&a[2 * k..2 * k + 2])
    }
}

impl Manifold for ComplexCircle {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.n >= 1 && n == want => Ok(()),
            Some(want) => Err(want),
            None => Err(n),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xs), Some(vs)) = (x.as_slice(), v.as_slice()) else {
            return v.clone();
        };
        if !self.fits(xs.len()) || xs.len() != vs.len() {
            return v.clone();
        }
        let mut out = Vector::from_host(v.clone());
        {
            let ts = out.host_mut();
            let Some(ts) = ts.as_slice_mut() else {
                return v.clone();
            };
            for k in 0..self.n {
                let z = self.pair(xs, k);
                let u = self.pair(vs, k);
                let s = vecops::dot(z, u);
                let mut col = u.to_owned();
                vecops::axpy(-s, z, &mut col);
                ts[2 * k] = col[0];
                ts[2 * k + 1] = col[1];
            }
        }
        out.into_host()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xs), Some(vs)) = (x.as_slice(), v.as_slice()) else {
            return x.clone();
        };
        if !self.fits(xs.len()) || xs.len() != vs.len() {
            return x.clone();
        }
        let mut y = Vector::from_host(x.clone());
        vecops::vaxpy(1.0, &Vector::from_host(v.clone()), &mut y);
        {
            let ys = y.host_mut();
            let Some(ys) = ys.as_slice_mut() else {
                return x.clone();
            };
            for k in 0..self.n {
                let pair = [ys[2 * k], ys[2 * k + 1]];
                let nrm = vecops::nrm2(ArrayView1::from(pair.as_slice()));
                if nrm > 1e-16 {
                    ys[2 * k] = pair[0] / nrm;
                    ys[2 * k + 1] = pair[1] / nrm;
                } else {
                    ys[2 * k] = xs[2 * k];
                    ys[2 * k + 1] = xs[2 * k + 1];
                }
            }
        }
        y.into_host()
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn two_circles() -> Array1<f64> {
        // (1, 0) and (0, 1): 1 and i.
        array![1.0, 0.0, 0.0, 1.0]
    }

    fn pair_moduli(y: &Array1<f64>) -> Vec<f64> {
        let n = y.len() / 2;
        (0..n)
            .map(|k| {
                let p = [y[2 * k], y[2 * k + 1]];
                vecops::nrm2(ArrayView1::from(p.as_slice()))
            })
            .collect()
    }

    #[test]
    fn retract_stays_on_the_set() {
        let m = ComplexCircle { n: 2 };
        let x = two_circles();
        let v = m.project(&x, &array![0.3, -0.1, 0.2, 0.4]);
        let y = m.retract(&x, &v);
        for nrm in pair_moduli(&y) {
            assert!((nrm - 1.0).abs() < 1e-14, "left (S^1)^2 {y:?}");
        }
        assert_eq!(y.len(), 4);
    }

    #[test]
    fn project_is_tangent_to_each_circle() {
        let m = ComplexCircle { n: 2 };
        let x = two_circles();
        let v = array![0.5, 0.25, -0.1, 0.8];
        let t = m.project(&x, &v);
        let xs = x.as_slice().unwrap();
        let ts = t.as_slice().unwrap();
        for k in 0..2 {
            let s = vecops::dot(m.pair(xs, k), m.pair(ts, k));
            assert!(s.abs() < 1e-15, "pair {k} not tangent {s}");
        }
        // z = 1: proj = (0.5, 0.25) - 0.5*(1, 0) = (0, 0.25)
        assert!(t[0].abs() < 1e-15);
        assert!((t[1] - 0.25).abs() < 1e-15);
        // z = i: proj = (-0.1, 0.8) - 0.8*(0, 1) = (-0.1, 0)
        assert!((t[2] + 0.1).abs() < 1e-15);
        assert!(t[3].abs() < 1e-15);
    }

    #[test]
    fn pack_unpack_round_trips() {
        let re = array![1.0, 0.0, -1.0];
        let im = array![0.0, 1.0, 0.0];
        let x = ComplexCircle::pack(re.view(), im.view());
        assert_eq!(x, array![1.0, 0.0, 0.0, 1.0, -1.0, 0.0]);
        let (r2, i2) = ComplexCircle::unpack(&x).unwrap();
        assert!((r2 - re).mapv(f64::abs).sum() < 1e-15);
        assert!((i2 - im).mapv(f64::abs).sum() < 1e-15);
        assert!(ComplexCircle::unpack(&array![1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn product_of_circles_is_not_the_sphere() {
        let m = ComplexCircle { n: 2 };
        let x = two_circles();
        let y = m.retract(&x, &Array1::zeros(4));
        let fro = vecops::nrm2(y.view());
        assert!(
            (fro - 2.0_f64.sqrt()).abs() < 1e-14,
            "must stay ||.|| = sqrt(2) {y:?}"
        );
        assert!((fro - 1.0).abs() > 0.4, "must not be S^3 {y:?}");
    }

    #[test]
    fn wrong_dim_rejects_a_3n_cluster() {
        let m = ComplexCircle { n: 2 };
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(m.project(&x, &v).len(), 114);
        assert_eq!(m.required_dim(114), Err(4));
        assert!(m.required_dim(4).is_ok());
        assert!(ComplexCircle::new(1).required_dim(2).is_ok());
        assert!(ComplexCircle::new(0).required_dim(0).is_err());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = ComplexCircle { n: 2 };
        let x = two_circles();
        let y = m.retract(&x, &Array1::zeros(4));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn kind_is_not_sphere_or_stiefel() {
        use crate::manifold::ManifoldKind;
        assert_ne!(ManifoldKind::ComplexCircle { n: 2 }, ManifoldKind::Sphere);
        assert_ne!(ManifoldKind::ComplexCircle { n: 1 }, ManifoldKind::Stiefel);
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let m = ComplexCircle { n: 1 };
        let x = array![1.0, 0.0];
        let y = array![0.0, 1.0];
        let v = array![0.2, 0.3];
        let t = m.transport(&x, &y, &v);
        let p = m.project(&y, &v);
        assert!((&t - &p).mapv(f64::abs).sum() < 1e-15);
    }
}
