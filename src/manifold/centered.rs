//! Euclidean space of matrices with centered columns or rows.
//!
//! manopt `centeredmatrixfactory(m, n, rows_or_cols)`: the linear
//! subspace of `m x n` matrices whose columns sum to zero
//! (`X * 1 = 0`, the default) or whose rows sum to zero
//! (`1^T X = 0`). Packed row-major, length `m n`. Projection is
//! the centering operator (subtract the mean column or mean
//! row). Retraction is `X + U` then center. Transport is the
//! identity. This token is not [`super::Sphere`], not a 3N
//! cluster packing, and not the reserved SPD / grassmann /
//! hyperbolic tokens 7-10.
//!
//! Reductions go through [`crate::vecops`].

use ndarray::{Array1, Array2, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// Which mean is constrained to vanish.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CenterMode {
    /// `X 1_n = 0`, the mean column is zero.
    #[default]
    Cols,
    /// `1_m^T X = 0`, the mean row is zero.
    Rows,
}

/// Centered `m x n` matrices. Packed row-major, length `m n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CenteredMatrix {
    /// Rows of the matrix. MATLAB `m`.
    pub m: usize,
    /// Columns of the matrix. MATLAB `n`.
    pub n: usize,
    /// `true` = centered rows (`1^T X = 0`). `false` = centered
    /// columns (`X 1 = 0`), the manopt default.
    pub rows: bool,
}

impl Default for CenteredMatrix {
    fn default() -> Self {
        Self {
            m: 2,
            n: 2,
            rows: false,
        }
    }
}

impl CenteredMatrix {
    /// Construct the linear subspace from an explicit centering mode.
    pub fn with_mode(m: usize, n: usize, mode: CenterMode) -> Self {
        Self::new(m, n, mode == CenterMode::Rows)
    }
    /// Matrices whose mean column is zero.
    pub fn cols(m: usize, n: usize) -> Self {
        Self::new(m, n, false)
    }
    /// Matrices whose mean row is zero.
    pub fn rows(m: usize, n: usize) -> Self {
        Self::new(m, n, true)
    }
    /// Centering mode represented by this subspace.
    pub fn mode(self) -> CenterMode {
        if self.rows {
            CenterMode::Rows
        } else {
            CenterMode::Cols
        }
    }
    /// Flatten a matrix in row-major order.
    pub fn pack_matrix(matrix: &Array2<f64>) -> Array1<f64> {
        Array1::from_iter(matrix.iter().copied())
    }
    /// Recover a matrix with this subspace's shape.
    pub fn unpack_matrix(self, x: &Array1<f64>) -> Option<Array2<f64>> {
        let values = unpack(x, self.m, self.n)?;
        Array2::from_shape_vec((self.m, self.n), values).ok()
    }

    /// Centered `m x n` matrices. Illegal `m == 0` or `n == 0`
    /// fails [`Manifold::required_dim`].
    pub fn new(m: usize, n: usize, rows: bool) -> Self {
        Self { m, n, rows }
    }

    /// Packed length `m n`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.m.checked_mul(self.n)
    }

    fn fits(self, len: usize) -> bool {
        self.m >= 1 && self.n >= 1 && self.packed_len() == Some(len)
    }

    /// Flatten a row-major `m x n` matrix into the ambient vector.
    pub fn pack(self, a: ArrayView1<f64>) -> Array1<f64> {
        a.to_owned()
    }

    /// Unpack a packed point. `None` if the length is not `m n`.
    pub fn unpack(self, x: &Array1<f64>) -> Option<Array1<f64>> {
        if self.fits(x.len()) {
            Some(x.clone())
        } else {
            None
        }
    }
}

/// Flatten a row-major `m x n` matrix into the ambient vector.
pub fn pack(m: usize, n: usize, a: Vec<f64>) -> Array1<f64> {
    Array1::from_shape_vec(m.saturating_mul(n), a).unwrap_or_else(|_| Array1::zeros(0))
}

/// Split a packed vector when its length is `m n`.
pub fn unpack(x: &Array1<f64>, m: usize, n: usize) -> Option<Vec<f64>> {
    match m.checked_mul(n) {
        Some(len) if len == x.len() && m >= 1 && n >= 1 => Some(x.iter().copied().collect()),
        _ => None,
    }
}

/// Frobenius inner product. manopt `d1(:).'*d2(:)`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = sqrt(M.dim())`.
pub fn typical_dist(m: usize, n: usize, rows: bool) -> f64 {
    let mn = m.saturating_mul(n);
    let dim = if rows {
        mn.saturating_sub(n)
    } else {
        mn.saturating_sub(m)
    };
    (dim as f64).sqrt()
}

/// `true` when the packed matrix has the required zero row or
/// column means.
pub fn is_centered(x: &Array1<f64>, m: usize, n: usize, rows: bool) -> bool {
    if unpack(x, m, n).is_none() {
        return false;
    }
    means_vanish(m, n, rows, x.as_slice().unwrap_or(&[]))
}

/// Typical distance with an explicit centering mode.
pub fn typical_dist_mode(m: usize, n: usize, mode: CenterMode) -> f64 {
    typical_dist(m, n, mode == CenterMode::Rows)
}

/// Whether the matrix satisfies the requested centering constraints.
pub fn is_centered_mode(x: &Array1<f64>, m: usize, n: usize, mode: CenterMode) -> bool {
    max_mean_abs(m, n, x.as_slice().unwrap_or(&[]), mode) < 1e-10
}

/// Largest absolute constrained mean; infinity for invalid shape or data.
pub fn max_mean_abs(m: usize, n: usize, a: &[f64], mode: CenterMode) -> f64 {
    if m == 0 || n == 0 || m.checked_mul(n) != Some(a.len()) || a.iter().any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    match mode {
        CenterMode::Cols => (0..m).map(|i| row_mean(a, i, n).abs()).fold(0.0, f64::max),
        CenterMode::Rows => (0..n)
            .map(|j| col_mean(a, j, m, n).abs())
            .fold(0.0, f64::max),
    }
}

fn means_vanish(m: usize, n: usize, rows: bool, a: &[f64]) -> bool {
    if a.len() != m.saturating_mul(n) {
        return false;
    }
    if rows {
        for j in 0..n {
            if col_mean(a, j, m, n).abs() > 1e-10 {
                return false;
            }
        }
    } else {
        for i in 0..m {
            if row_mean(a, i, n).abs() > 1e-10 {
                return false;
            }
        }
    }
    true
}

fn row_mean(a: &[f64], i: usize, n: usize) -> f64 {
    let row = ArrayView1::from(&a[i * n..i * n + n]);
    vecops::sum(row) / (n as f64)
}

fn col_mean(a: &[f64], j: usize, m: usize, n: usize) -> f64 {
    let mut col = Vector::zeros_cpu(m);
    {
        let cs = col.host_mut();
        for i in 0..m {
            cs[i] = a[i * n + j];
        }
    }
    vecops::sum(col.host_view()) / (m as f64)
}

fn center_inplace(m: usize, n: usize, rows: bool, a: &mut [f64]) {
    if rows {
        for j in 0..n {
            let mean = col_mean(a, j, m, n);
            for i in 0..m {
                a[i * n + j] -= mean;
            }
        }
    } else {
        for i in 0..m {
            let mean = row_mean(a, i, n);
            for j in 0..n {
                a[i * n + j] -= mean;
            }
        }
    }
}

fn center_array(m: usize, n: usize, rows: bool, x: &Array1<f64>) -> Array1<f64> {
    let mut y = Vector::from_host(x.clone());
    center_inplace(m, n, rows, y.host_mut().as_slice_mut().unwrap_or(&mut []));
    y.into_host()
}

impl Manifold for CenteredMatrix {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.m >= 1 && self.n >= 1 && n == want => Ok(()),
            Some(want) => Err(want),
            None => Err(n),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return v.clone();
        }
        center_array(self.m, self.n, self.rows, v)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return x.clone();
        }
        let mut y = Vector::from_host(x.clone());
        vecops::vaxpy(1.0, &Vector::from_host(v.clone()), &mut y);
        center_inplace(
            self.m,
            self.n,
            self.rows,
            y.host_mut().as_slice_mut().unwrap_or(&mut []),
        );
        y.into_host()
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        Vector::from_host(v.clone()).into_host()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_set() {
        let m = CenteredMatrix::new(2, 3, false);
        let x = array![1.0, -0.5, -0.5, 2.0, 0.0, -2.0];
        let v = array![0.3, -0.1, -0.2, 0.0, 0.4, -0.4];
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 6);
        assert!(is_centered(&y, 2, 3, false), "left the centered set {y:?}");
        assert!((y[0] + y[1] + y[2]).abs() < 1e-14, "row 0 mean {y:?}");
        assert!((y[3] + y[4] + y[5]).abs() < 1e-14, "row 1 mean {y:?}");
        let fro = vecops::nrm2(y.view());
        assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
    }

    #[test]
    fn retract_stays_on_centered_rows() {
        let m = CenteredMatrix::new(2, 3, true);
        let x = array![1.0, 2.0, -3.0, -1.0, -2.0, 3.0];
        let v = array![0.2, -0.1, 0.4, 0.1, 0.3, -0.2];
        let y = m.retract(&x, &v);
        assert!(
            is_centered(&y, 2, 3, true),
            "left the row-centered set {y:?}"
        );
        assert!((y[0] + y[3]).abs() < 1e-14);
        assert!((y[1] + y[4]).abs() < 1e-14);
        assert!((y[2] + y[5]).abs() < 1e-14);
    }

    #[test]
    fn retract_is_translation_then_center() {
        let m = CenteredMatrix::new(2, 2, false);
        let x = array![1.0, -1.0, 2.0, -2.0];
        let v = array![0.2, -0.2, -0.1, 0.1];
        let y = m.retract(&x, &v);
        assert!((y[0] - 1.2).abs() < 1e-15, "{y:?}");
        assert!((y[1] + 1.2).abs() < 1e-15, "{y:?}");
        assert!((y[2] - 1.9).abs() < 1e-15, "{y:?}");
        assert!((y[3] + 1.9).abs() < 1e-15, "{y:?}");
    }

    #[test]
    fn project_is_center() {
        let m = CenteredMatrix::new(2, 2, false);
        let x = array![1.0, -1.0, 0.0, 0.0];
        let v = array![1.0, 3.0, 2.0, 4.0];
        let t = m.project(&x, &v);
        assert!((t[0] + t[1]).abs() < 1e-15);
        assert!((t[2] + t[3]).abs() < 1e-15);
        assert!((t[0] + 1.0).abs() < 1e-15);
        assert!((t[1] - 1.0).abs() < 1e-15);
        assert!((t[2] + 1.0).abs() < 1e-15);
        assert!((t[3] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn transport_of_a_tangent_is_itself() {
        let m = CenteredMatrix::new(2, 2, false);
        let x = array![1.0, -1.0, 0.0, 0.0];
        let y = array![0.0, 0.0, 2.0, -2.0];
        let v = array![0.25, -0.25, -0.1, 0.1];
        let t = m.transport(&x, &y, &v);
        assert!((&t - &v).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn frobenius_inner_and_typical_dist() {
        let u = array![1.0, -1.0, 0.5, -0.5];
        let v = array![0.0, 0.0, 2.0, -2.0];
        assert!((inner(&u, &v) - 2.0).abs() < 1e-15);
        assert!((typical_dist(2, 3, false) - 4.0_f64.sqrt()).abs() < 1e-15);
        assert!((typical_dist(2, 3, true) - 3.0_f64.sqrt()).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn pack_unpack_round_trips() {
        let a = array![1.0, -1.0, 2.0, 0.5, -0.5, -2.0];
        let m = CenteredMatrix::new(2, 3, false);
        let x = m.pack(a.view());
        assert_eq!(x, a);
        let back = m.unpack(&x).unwrap();
        assert!((back - a.clone()).mapv(f64::abs).sum() < 1e-15);
        let (n, flat) = (2usize, unpack(&x, 2, 3).unwrap());
        assert_eq!(n, 2);
        let y = pack(2, 3, flat);
        assert!((&x - &y).mapv(f64::abs).sum() < 1e-15);
        assert!(m.unpack(&array![1.0, 0.0]).is_none());
        assert!(unpack(&x, 3, 3).is_none());
        assert!(
            CenteredMatrix::new(0, 2, false)
                .unpack(&array![1.0, 2.0])
                .is_none()
        );
    }

    #[test]
    fn wrong_dim_rejects_a_3n_cluster() {
        let m = CenteredMatrix::new(2, 2, false);
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert!(
            (&y - &x).mapv(f64::abs).sum() < 1e-15,
            "must not center a cluster"
        );
        assert_eq!(m.project(&x, &v).len(), 114);
        assert_eq!(m.required_dim(114), Err(4));
        assert!(m.required_dim(4).is_ok());
        assert!(CenteredMatrix::new(1, 1, false).required_dim(1).is_ok());
        assert!(CenteredMatrix::new(0, 2, false).required_dim(0).is_err());
        assert!(CenteredMatrix::new(2, 3, true).required_dim(6).is_ok());
        assert!(CenteredMatrix::new(2, 3, true).required_dim(5).is_err());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = CenteredMatrix::new(2, 2, false);
        let x = array![1.0, -1.0, 0.5, -0.5];
        let y = m.retract(&x, &Array1::zeros(4));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
        assert!(is_centered(&y, 2, 2, false));
    }

    #[test]
    fn kind_is_not_sphere_or_stiefel() {
        use crate::manifold::ManifoldKind;
        assert_ne!(
            ManifoldKind::CenteredMatrix {
                m: 2,
                n: 2,
                rows: false
            },
            ManifoldKind::Sphere
        );
        assert_ne!(
            ManifoldKind::CenteredMatrix {
                m: 3,
                n: 1,
                rows: false
            },
            ManifoldKind::Stiefel
        );
        assert_ne!(
            ManifoldKind::CenteredMatrix {
                m: 2,
                n: 2,
                rows: false
            },
            ManifoldKind::Euclidean
        );
        assert_ne!(
            ManifoldKind::CenteredMatrix {
                m: 2,
                n: 2,
                rows: false
            },
            ManifoldKind::Positive { n: 4 }
        );
        assert_ne!(
            ManifoldKind::CenteredMatrix {
                m: 2,
                n: 2,
                rows: false
            },
            ManifoldKind::CenteredMatrix {
                m: 2,
                n: 2,
                rows: true
            }
        );
    }
}

#[cfg(test)]
mod retained_mode_tests {
    use super::*;
    use ndarray::array;
    fn unpack(m: usize, n: usize, x: &Array1<f64>) -> Option<Vec<f64>> {
        super::unpack(x, m, n)
    }

    #[test]
    fn retract_stays_on_the_set() {
        let m = CenteredMatrix::cols(2, 3);
        let x = array![1.0, -0.5, -0.5, 2.0, -1.0, -1.0];
        let v = array![0.3, 0.0, -0.1, -0.2, 0.4, 0.1];
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 6);
        assert!(
            is_centered_mode(&y, 2, 3, CenterMode::Cols),
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
            is_centered_mode(&y, 2, 3, CenterMode::Rows),
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
    fn frobenius_inner_and_typical_dist_mode() {
        let u = array![1.0, -1.0, 0.5, -0.5];
        let v = array![1.0, 1.0, 2.0, 2.0];
        assert!((inner(&u, &v) - 0.0).abs() < 1e-15);
        let u2 = array![1.0, -1.0, 0.0, 0.0];
        let v2 = array![1.0, 0.0, 0.0, 0.0];
        assert!((inner(&u2, &v2) - 1.0).abs() < 1e-15);
        assert!((typical_dist_mode(2, 3, CenterMode::Cols) - 4.0_f64.sqrt()).abs() < 1e-15);
        assert!((typical_dist_mode(2, 3, CenterMode::Rows) - 3.0_f64.sqrt()).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn pack_unpack_round_trips() {
        let mat = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = CenteredMatrix::pack_matrix(&mat);
        assert_eq!(x, array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let back = CenteredMatrix::cols(2, 3).unpack_matrix(&x).unwrap();
        assert_eq!(back, mat);
        let (m, n) = (2, 3);
        let y = pack(m, n, unpack(m, n, &x).unwrap());
        for i in 0..6 {
            assert!((x[i] - y[i]).abs() < 1e-15);
        }
        assert!(unpack(2, 3, &array![1.0]).is_none());
        assert!(CenteredMatrix::cols(2, 2).unpack_matrix(&x).is_none());
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
        assert!(
            CenteredMatrix::with_mode(0, 4, CenterMode::Cols)
                .required_dim(0)
                .is_err()
        );
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
            "centeredmatrix"
        );
    }

    #[test]
    fn cols_and_rows_are_different_sets() {
        let x = array![1.0, -1.0, 2.0, -2.0];
        assert!(is_centered_mode(&x, 2, 2, CenterMode::Cols));
        assert!(!is_centered_mode(&x, 2, 2, CenterMode::Rows));
        let y = array![1.0, 2.0, -1.0, -2.0];
        assert!(is_centered_mode(&y, 2, 2, CenterMode::Rows));
        assert!(!is_centered_mode(&y, 2, 2, CenterMode::Cols));
    }
}
