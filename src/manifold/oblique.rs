//! Oblique manifold: product of unit spheres. manopt `obliquefactory`.
//!
//! A point is an n-by-m matrix with unit-norm columns, packed
//! column-major (length `n*m`). Projection is column-wise
//! \(v_j - (x_j\cdot v_j) x_j\). Retraction is column-wise
//! metric projection \((x_j+v_j)/\|x_j+v_j\|\). Transport is
//! projection at the arrival point.
//!
//! This is not a 3N cluster. Isolated molecules use
//! [`super::RigidQuotient`]. A single sphere uses [`super::Sphere`].

#[cfg(test)]
use ndarray::ArrayView1;
use ndarray::{s, Array1, Array2};

use crate::vecops::{dot, nrm2};

use super::Manifold;

/// Product of `m` unit spheres in \(\mathbb{R}^n\).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Oblique {
    /// Ambient dimension of each sphere (column length).
    pub n: usize,
    /// Number of unit-norm columns.
    pub m: usize,
}

impl Oblique {
    /// Product of `m` spheres in \(\mathbb{R}^n\).
    pub fn new(n: usize, m: usize) -> Self {
        Self { n, m }
    }

    /// Packed length `n*m`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.n.checked_mul(self.m)
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 2 && self.m >= 1 && self.packed_len() == Some(len)
    }

    /// Column-major flatten of an n-by-m matrix (manopt `X(:)`).
    pub fn pack(mat: &Array2<f64>) -> Array1<f64> {
        let (n, m) = mat.dim();
        let mut out = Array1::zeros(n * m);
        for j in 0..m {
            for i in 0..n {
                out[i + j * n] = mat[[i, j]];
            }
        }
        out
    }

    /// Inverse of [`Self::pack`] for this `(n, m)`.
    pub fn unpack(&self, x: &Array1<f64>) -> Array2<f64> {
        let mut mat = Array2::zeros((self.n, self.m));
        if !self.fits(x.len()) {
            return mat;
        }
        for j in 0..self.m {
            for i in 0..self.n {
                mat[[i, j]] = x[i + j * self.n];
            }
        }
        mat
    }

    #[cfg(test)]
    fn col<'a>(&self, x: &'a Array1<f64>, j: usize) -> ArrayView1<'a, f64> {
        x.slice(s![j * self.n..(j + 1) * self.n])
    }
}

impl Default for Oblique {
    fn default() -> Self {
        Self { n: 2, m: 2 }
    }
}

fn project_columns(n: usize, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::zeros(x.len());
    for j in 0..(x.len() / n) {
        let xs = x.slice(s![j * n..(j + 1) * n]);
        let vs = v.slice(s![j * n..(j + 1) * n]);
        let s = dot(xs, vs);
        let mut os = out.slice_mut(s![j * n..(j + 1) * n]);
        for (oi, (xi, vi)) in os.iter_mut().zip(xs.iter().zip(vs.iter())) {
            *oi = *vi - s * *xi;
        }
    }
    out
}

fn retract_columns(n: usize, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    let mut y = x + v;
    let m = y.len() / n;
    for j in 0..m {
        let mut col = y.slice_mut(s![j * n..(j + 1) * n]);
        let nrm = nrm2(col.view());
        if nrm <= 1e-16 {
            let xcol = x.slice(s![j * n..(j + 1) * n]);
            let n0 = nrm2(xcol);
            if n0 <= 1e-16 {
                for (yi, xi) in col.iter_mut().zip(xcol.iter()) {
                    *yi = *xi;
                }
            } else {
                for (yi, xi) in col.iter_mut().zip(xcol.iter()) {
                    *yi = *xi / n0;
                }
            }
        } else {
            for yi in col.iter_mut() {
                *yi /= nrm;
            }
        }
    }
    y
}

impl Manifold for Oblique {
    fn required_dim(&self, dim: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.n >= 2 && self.m >= 1 && dim == want => Ok(()),
            Some(want) => Err(want),
            None => Err(dim),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || v.len() != x.len() {
            return v.clone();
        }
        project_columns(self.n, x, v)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || v.len() != x.len() {
            return x + v;
        }
        retract_columns(self.n, x, v)
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn two_by_three() -> Oblique {
        Oblique::new(3, 2)
    }

    fn unit_pair() -> Array1<f64> {
        // Columns (1,0,0) and (0,1,0), column-major.
        array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    }

    #[test]
    fn project_is_columnwise_tangent() {
        let m = two_by_three();
        let x = unit_pair();
        let v = array![2.0, 3.0, 4.0, -1.0, 5.0, 0.5];
        let t = m.project(&x, &v);
        let d0 = dot(m.col(&x, 0), m.col(&t, 0));
        let d1 = dot(m.col(&x, 1), m.col(&t, 1));
        assert!(d0.abs() < 1e-15, "col0 {d0}");
        assert!(d1.abs() < 1e-15, "col1 {d1}");
        assert!((t[1] - 3.0).abs() < 1e-15);
        assert!((t[2] - 4.0).abs() < 1e-15);
        assert!((t[3] + 1.0).abs() < 1e-15);
        assert!((t[5] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn retract_stays_on_product_of_spheres() {
        let m = two_by_three();
        let x = unit_pair();
        let v = array![0.1, 0.2, -0.3, 0.4, 0.0, -0.2];
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 6);
        let n0 = nrm2(m.col(&y, 0));
        let n1 = nrm2(m.col(&y, 1));
        assert!((n0 - 1.0).abs() < 1e-14, "col0 {n0}");
        assert!((n1 - 1.0).abs() < 1e-14, "col1 {n1}");
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let m = two_by_three();
        let x = unit_pair();
        let y = m.retract(&x, &array![0.2, 0.0, 0.1, 0.0, 0.3, -0.1]);
        let v = array![0.5, -0.2, 0.1, 0.3, 0.4, -0.6];
        let t = m.transport(&x, &y, &v);
        let p = m.project(&y, &v);
        for (a, b) in t.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
        let d0 = dot(m.col(&y, 0), m.col(&t, 0));
        let d1 = dot(m.col(&y, 1), m.col(&t, 1));
        assert!(d0.abs() < 1e-14, "col0 {d0}");
        assert!(d1.abs() < 1e-14, "col1 {d1}");
    }

    #[test]
    fn pack_unpack_is_column_major() {
        let m = two_by_three();
        let mat = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let x = Oblique::pack(&mat);
        assert_eq!(x, unit_pair());
        let back = m.unpack(&x);
        assert_eq!(back, mat);
    }

    #[test]
    fn required_dim_rejects_a_3n_cluster() {
        let m = Oblique::new(3, 2);
        assert!(m.required_dim(6).is_ok());
        assert_eq!(m.required_dim(114), Err(6));
        assert_eq!(m.required_dim(9), Err(6));
        assert!(Oblique::new(3, 1).required_dim(3).is_ok());
        assert!(Oblique::new(1, 4).required_dim(4).is_err());
    }

    #[test]
    fn wrong_dim_keeps_length() {
        let m = two_by_three();
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        for i in 0..114 {
            assert!((y[i] - (x[i] + v[i])).abs() < 1e-15);
        }
        assert_eq!(m.project(&x, &v).len(), 114);
    }
}
