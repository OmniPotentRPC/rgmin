//! Euclidean space of centered real matrices. manopt `centeredmatrixfactory`.
//!
//! A point is an `m x n` real matrix packed row-major as length
//! `m*n`. The default geometry (`'cols'`) subtracts the mean column
//! so `X 1_n = 0`. Passing `'rows'` subtracts the mean row so
//! `1_m^T X = 0`. The metric is the Frobenius inner product.
//! Projection is that centering. Retraction is `X + U` then center.
//! Transport is the identity. This is a linear subspace of
//! Euclidean space, not the sphere and not a 3N cluster packing.
//!
//! Reductions go through [`crate::vecops`].

use ndarray::{Array1, Array2, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// Which mean manopt subtracts. Default is [`CenterMode::Cols`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CenterMode {
    /// Mean column is zero: `X * 1_n = 0`. manopt `'cols'`.
    #[default]
    Cols,
    /// Mean row is zero: `1_m^T * X = 0`. manopt `'rows'`.
    Rows,
}

/// Centered `m x n` matrices. Packed row-major, length `m*n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CenteredMatrix {
    /// Number of rows. manopt `m`.
    pub m: usize,
    /// Number of columns. manopt `n`.
    pub n: usize,
    /// Which mean is removed.
    pub mode: CenterMode,
}

impl Default for CenteredMatrix {
    fn default() -> Self {
        Self {
            m: 2,
            n: 2,
            mode: CenterMode::Cols,
        }
    }
}

impl CenteredMatrix {
    /// Centered `m x n` matrices under `mode`.
    pub fn new(m: usize, n: usize, mode: CenterMode) -> Self {
        Self { m, n, mode }
    }

    /// manopt `centeredmatrixfactory(m, n)` / `'cols'`.
    pub fn cols(m: usize, n: usize) -> Self {
        Self::new(m, n, CenterMode::Cols)
    }

    /// manopt `centeredmatrixfactory(m, n, 'rows')`.
    pub fn rows(m: usize, n: usize) -> Self {
        Self::new(m, n, CenterMode::Rows)
    }

    /// Packed length `m*n`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.m.checked_mul(self.n)
    }

    fn fits(self, len: usize) -> bool {
        self.m >= 1 && self.n >= 1 && self.packed_len() == Some(len)
    }

    /// Row-major flatten of an m-by-n matrix.
    pub fn pack(mat: &Array2<f64>) -> Array1<f64> {
        pack(mat.nrows(), mat.ncols(), mat.iter().copied().collect())
    }

    /// Inverse of [`Self::pack`] for this `(m, n)`.
    pub fn unpack(self, x: &Array1<f64>) -> Option<Array2<f64>> {
        let a = unpack(self.m, self.n, x)?;
        Array2::from_shape_vec((self.m, self.n), a).ok()
    }
}

/// Flatten a row-major m-by-n matrix into the ambient vector.
pub fn pack(m: usize, n: usize, a: Vec<f64>) -> Array1<f64> {
    debug_assert_eq!(a.len(), m.saturating_mul(n));
    Array1::from(a)
}

/// Split a length-`m*n` ambient vector into row-major entries.
pub fn unpack(m: usize, n: usize, x: &Array1<f64>) -> Option<Vec<f64>> {
    if m < 1 || n < 1 || m.checked_mul(n) != Some(x.len()) {
        return None;
    }
    Some(x.iter().copied().collect())
}

/// Frobenius inner product. manopt `M.inner = d1(:).'*d2(:)`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = sqrt(dim)` with `dim = mn - m` (cols)
/// or `mn - n` (rows).
pub fn typical_dist(m: usize, n: usize, mode: CenterMode) -> f64 {
    let dim = match mode {
        CenterMode::Cols => m.saturating_mul(n).saturating_sub(m),
        CenterMode::Rows => m.saturating_mul(n).saturating_sub(n),
    };
    (dim as f64).sqrt()
}

/// `true` when the packed matrix has the requested mean zero.
pub fn is_centered(x: &Array1<f64>, m: usize, n: usize, mode: CenterMode) -> bool {
    if unpack(m, n, x).is_none() {
        return false;
    }
    max_mean_abs(m, n, x.as_slice().unwrap_or(&[]), mode) < 1e-10
}

/// Largest absolute constrained mean of a packed matrix.
pub fn max_mean_abs(m: usize, n: usize, a: &[f64], mode: CenterMode) -> f64 {
    if m == 0 || n == 0 || a.len() != m * n {
        return 0.0;
    }
    match mode {
        CenterMode::Cols => {
            let mut worst = 0.0_f64;
            for i in 0..m {
                let row = ArrayView1::from(&a[i * n..i * n + n]);
                worst = worst.max((vecops::sum(row) / n as f64).abs());
            }
            worst
        }
        CenterMode::Rows => {
            let mut worst = 0.0_f64;
            let mut col = vec![0.0; m];
            for j in 0..n {
                for i in 0..m {
                    col[i] = a[i * n + j];
                }
                let view = ArrayView1::from(col.as_slice());
                worst = worst.max((vecops::sum(view) / m as f64).abs());
            }
            worst
        }
    }
}

fn center(m: usize, n: usize, a: &[f64], mode: CenterMode) -> Vec<f64> {
    let mut y = a.to_vec();
    match mode {
        CenterMode::Cols => {
            for i in 0..m {
                let row = ArrayView1::from(&a[i * n..i * n + n]);
                let mean = vecops::sum(row) / n as f64;
                for j in 0..n {
                    y[i * n + j] -= mean;
                }
            }
        }
        CenterMode::Rows => {
            let mut col = vec![0.0; m];
            for j in 0..n {
                for i in 0..m {
                    col[i] = a[i * n + j];
                }
                let mean = vecops::sum(ArrayView1::from(col.as_slice())) / m as f64;
                for i in 0..m {
                    y[i * n + j] -= mean;
                }
            }
        }
    }
    y
}

impl Manifold for CenteredMatrix {
    fn required_dim(&self, dim: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.m >= 1 && self.n >= 1 && dim == want => Ok(()),
            Some(want) => Err(want),
            None => Err(dim),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || v.len() != x.len() {
            return v.clone();
        }
        let flat: Vec<f64> = v.iter().copied().collect();
        Array1::from(center(self.m, self.n, &flat, self.mode))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || v.len() != x.len() {
            return x + v;
        }
        let mut y = Vector::from_host(x.clone());
        vecops::vaxpy(1.0, &Vector::from_host(v.clone()), &mut y);
        let flat = y.into_host();
        let entries: Vec<f64> = flat.iter().copied().collect();
        Array1::from(center(self.m, self.n, &entries, self.mode))
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        v.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_set() {
        let m = CenteredMatrix::cols(2, 3);
        let x = array![1.0, -0.5, -0.5, 2.0, -1.0, -1.0];
        let v = array![0.3, 0.0, -0.1, -0.2, 0.4, 0.1];
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 6);
        assert!(
            is_centered(&y, 2, 3, CenterMode::Cols),
            "left the centered-cols set {y:?}"
        );
        let worst = max_mean_abs(2, 3, y.as_slice().unwrap(), CenterMode::Cols);
        assert!(worst < 1e-14, "row means {worst} y={y:?}");
        let fro = vecops::nrm2(y.view());
        assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
    }

    #[test]
    fn retract_rows_stays_on_the_set() {
        let m = CenteredMatrix::rows(2, 3);
        let x = array![1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        let v = array![0.1, -0.2, 0.3, 0.4, 0.0, -0.1];
        let y = m.retract(&x, &v);
        assert!(
            is_centered(&y, 2, 3, CenterMode::Rows),
            "left the centered-rows set {y:?}"
        );
        let worst = max_mean_abs(2, 3, y.as_slice().unwrap(), CenterMode::Rows);
        assert!(worst < 1e-14, "col means {worst} y={y:?}");
    }

    #[test]
    fn project_cols_subtracts_the_mean_column() {
        let m = CenteredMatrix::cols(2, 2);
        let x = array![1.0, -1.0, 2.0, -2.0];
        let v = array![1.0, 3.0, 2.0, 6.0];
        let t = m.project(&x, &v);
        // rows [1, 3] mean 2 -> [-1, 1]; [2, 6] mean 4 -> [-2, 2]
        assert!((t[0] + 1.0).abs() < 1e-15, "{t:?}");
        assert!((t[1] - 1.0).abs() < 1e-15, "{t:?}");
        assert!((t[2] + 2.0).abs() < 1e-15, "{t:?}");
        assert!((t[3] - 2.0).abs() < 1e-15, "{t:?}");
        assert!((t[0] + t[1]).abs() < 1e-15);
        assert!((t[2] + t[3]).abs() < 1e-15);
    }

    #[test]
    fn project_rows_subtracts_the_mean_row() {
        let m = CenteredMatrix::rows(2, 2);
        let x = array![1.0, 2.0, -1.0, -2.0];
        let v = array![1.0, 3.0, 5.0, 7.0];
        let t = m.project(&x, &v);
        // cols [1, 5] mean 3 -> [-2, 2]; [3, 7] mean 5 -> [-2, 2]
        assert!((t[0] + 2.0).abs() < 1e-15, "{t:?}");
        assert!((t[1] + 2.0).abs() < 1e-15, "{t:?}");
        assert!((t[2] - 2.0).abs() < 1e-15, "{t:?}");
        assert!((t[3] - 2.0).abs() < 1e-15, "{t:?}");
        assert!((t[0] + t[2]).abs() < 1e-15);
        assert!((t[1] + t[3]).abs() < 1e-15);
    }

    #[test]
    fn transport_of_a_tangent_is_itself() {
        let m = CenteredMatrix::cols(2, 2);
        let x = array![1.0, -1.0, 2.0, -2.0];
        let y = array![0.5, -0.5, -0.5, 0.5];
        let v = array![0.2, -0.2, -0.1, 0.1];
        let t = m.transport(&x, &y, &v);
        for i in 0..4 {
            assert!((t[i] - v[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn frobenius_inner_and_typical_dist() {
        let u = array![1.0, -1.0, 0.5, -0.5];
        let v = array![0.0, 0.0, 2.0, -2.0];
        assert!((inner(&u, &v) - 0.0).abs() < 1e-15);
        let u2 = array![1.0, -1.0, 0.0, 0.0];
        let v2 = array![1.0, 0.0, 0.0, 0.0];
        assert!((inner(&u2, &v2) - 1.0).abs() < 1e-15);
        assert!((typical_dist(2, 3, CenterMode::Cols) - 4.0_f64.sqrt()).abs() < 1e-15);
        assert!((typical_dist(2, 3, CenterMode::Rows) - 3.0_f64.sqrt()).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn pack_unpack_round_trips() {
        let mat = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = CenteredMatrix::pack(&mat);
        assert_eq!(x, array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let back = CenteredMatrix::cols(2, 3).unpack(&x).unwrap();
        assert_eq!(back, mat);
        let (m, n) = (2, 3);
        let y = pack(m, n, unpack(m, n, &x).unwrap());
        for i in 0..6 {
            assert!((x[i] - y[i]).abs() < 1e-15);
        }
        assert!(unpack(2, 3, &array![1.0]).is_none());
        assert!(CenteredMatrix::cols(2, 2).unpack(&x).is_none());
    }

    #[test]
    fn wrong_dim_rejects_a_3n_cluster() {
        let m = CenteredMatrix::cols(2, 3);
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(m.project(&x, &v).len(), 114);
        assert_eq!(m.required_dim(114), Err(6));
        assert!(m.required_dim(6).is_ok());
        assert!(CenteredMatrix::new(0, 4, CenterMode::Cols)
            .required_dim(0)
            .is_err());
        assert!(CenteredMatrix::cols(3, 2).required_dim(6).is_ok());
        assert!(CenteredMatrix::cols(3, 2).required_dim(9).is_err());
    }

    #[test]
    fn kind_is_not_sphere_or_stiefel() {
        use crate::manifold::ManifoldKind;
        assert_ne!(
            ManifoldKind::centered_matrix(2, 3, false),
            ManifoldKind::Sphere
        );
        assert_ne!(
            ManifoldKind::centered_matrix(2, 3, false),
            ManifoldKind::Stiefel
        );
        assert_ne!(
            ManifoldKind::centered_matrix(2, 3, false),
            ManifoldKind::Oblique { n: 2, m: 3 }
        );
        assert_ne!(
            ManifoldKind::centered_matrix(2, 3, false),
            ManifoldKind::centered_matrix(2, 3, true)
        );
        assert_eq!(
            ManifoldKind::centered_matrix(2, 3, false).as_str(),
            "centered_matrix"
        );
    }

    #[test]
    fn cols_and_rows_are_different_sets() {
        let x = array![1.0, -1.0, 2.0, -2.0];
        assert!(is_centered(&x, 2, 2, CenterMode::Cols));
        assert!(!is_centered(&x, 2, 2, CenterMode::Rows));
        let y = array![1.0, 2.0, -1.0, -2.0];
        assert!(is_centered(&y, 2, 2, CenterMode::Rows));
        assert!(!is_centered(&y, 2, 2, CenterMode::Cols));
    }
}
