//! Doubly-stochastic matrices (Birkhoff polytope) with the Fisher metric.
//!
//! manopt `multinomialdoublystochasticfactory`. The point is an n-by-n
//! matrix, packed row-major, with positive entries and unit row and
//! column sums. The tangent space is `{V : V 1 = 0, V^T 1 = 0}`.
//! Fisher-orthogonal projection solves
//! `[I X; X' I] [alpha; beta] = [V 1; V^T 1]` and returns
//! `V - (alpha 1^T + 1 beta^T) ⊙ X` (Douik and Hassibi, arXiv:1802.02628).
//! Retraction is `X ⊙ exp(V ⊘ X)` followed by Sinkhorn. Transport is
//! projection at the arrival point. `egrad2rgrad` is
//! `proj(X, X ⊙ egrad)`.
//!
//! Distinct from [`super::Multinomial`] (one simplex) and from the
//! reserved grassmann / hyperbolic / SPD tokens 7-10.

use ndarray::{Array1, Array2, ArrayView1};

use crate::vecops;

use super::Manifold;

/// Birkhoff polytope, Fisher information metric.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialDoublyStochastic {
    /// Side length. Packed length is `n * n`.
    pub n: usize,
}

/// Side length if `len` is a perfect square `n^2` with `n >= 2`.
pub fn side(len: usize) -> Option<usize> {
    if len < 4 {
        return None;
    }
    let n = (len as f64).sqrt().round() as usize;
    if n >= 2 && n.checked_mul(n) == Some(len) {
        Some(n)
    } else {
        None
    }
}

/// Flatten a row-major n-by-n matrix into the ambient vector.
pub fn pack(n: usize, a: Vec<f64>) -> Array1<f64> {
    Array1::from_shape_vec(n * n, a).unwrap()
}

/// Split a length-n² ambient vector into (n, row-major entries).
pub fn unpack(x: &Array1<f64>) -> Option<(usize, Vec<f64>)> {
    let n = side(x.len())?;
    Some((n, x.iter().copied().collect()))
}

/// Fisher inner product `sum_ij U_ij V_ij / X_ij`.
pub fn inner(x: &Array1<f64>, u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    if x.len() != u.len() || x.len() != v.len() {
        return 0.0;
    }
    let mut w = Array1::zeros(x.len());
    for i in 0..x.len() {
        let den = x[i].max(f64::EPSILON);
        w[i] = u[i] * v[i] / den;
    }
    vecops::sum(w.view())
}

/// manopt `M.typicaldist = n`.
pub fn typical_dist(n: usize) -> f64 {
    n as f64
}

/// `true` when the packed matrix is square, positive, and doubly stochastic.
pub fn is_doubly_stochastic(x: &Array1<f64>) -> bool {
    let Some((n, a)) = unpack(x) else {
        return false;
    };
    if a.iter().any(|&ai| !(ai > 0.0 && ai.is_finite())) {
        return false;
    }
    for i in 0..n {
        let rs = vecops::sum(ArrayView1::from(&a[i * n..(i + 1) * n]));
        if (rs - 1.0).abs() > 1e-8 {
            return false;
        }
    }
    for j in 0..n {
        let mut col = Array1::zeros(n);
        for i in 0..n {
            col[i] = a[i * n + j];
        }
        let cs = vecops::sum(col.view());
        if (cs - 1.0).abs() > 1e-8 {
            return false;
        }
    }
    true
}

impl MultinomialDoublyStochastic {
    /// Factory of side `n`. Illegal `n < 2` fails [`Manifold::required_dim`].
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Flat f64 pack of an n-by-n matrix (manopt `M.vec`).
    pub fn pack(self, a: &Array2<f64>) -> Array1<f64> {
        pack(self.n, a.iter().copied().collect())
    }

    /// Inverse of [`Self::pack`] (manopt `M.mat`).
    pub fn unpack(self, packed: &Array1<f64>) -> Option<Array2<f64>> {
        if packed.len() != self.n * self.n {
            return None;
        }
        Array2::from_shape_vec((self.n, self.n), packed.iter().copied().collect()).ok()
    }

    fn mat(&self, packed: &Array1<f64>) -> Array2<f64> {
        self.unpack(packed)
            .unwrap_or_else(|| Array2::zeros((self.n, self.n)))
    }

    fn pack_mat(&self, a: &Array2<f64>) -> Array1<f64> {
        self.pack(a)
    }

    fn apply_saddle(&self, x: &Array2<f64>, z: &Array1<f64>) -> Array1<f64> {
        let n = self.n;
        let mut out = Array1::zeros(2 * n);
        let alpha = z.slice(ndarray::s![0..n]);
        let beta = z.slice(ndarray::s![n..]);
        for i in 0..n {
            out[i] = z[i] + vecops::dot(x.row(i), beta);
        }
        for j in 0..n {
            let mut col = Array1::zeros(n);
            for i in 0..n {
                col[i] = x[(i, j)];
            }
            out[n + j] = z[n + j] + vecops::dot(col.view(), alpha);
        }
        out
    }

    fn linear_solve(&self, x: &Array2<f64>, b: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let n = self.n;
        let mut z = Array1::zeros(2 * n);
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rsold = vecops::dot(r.view(), r.view());
        if rsold < 1e-30 {
            return (Array1::zeros(n), Array1::zeros(n));
        }
        let maxit = (2 * n + 20).max(100);
        for _ in 0..maxit {
            let ap = self.apply_saddle(x, &p);
            let denom = vecops::dot(p.view(), ap.view());
            if denom.abs() < 1e-30 {
                break;
            }
            let step = rsold / denom;
            vecops::axpy(step, p.view(), &mut z);
            vecops::axpy(-step, ap.view(), &mut r);
            let rsnew = vecops::dot(r.view(), r.view());
            if rsnew.sqrt() < 1e-12 {
                break;
            }
            let beta_cg = rsnew / rsold;
            let mut new_p = r.clone();
            vecops::axpy(beta_cg, p.view(), &mut new_p);
            p = new_p;
            rsold = rsnew;
        }
        (
            z.slice(ndarray::s![0..n]).to_owned(),
            z.slice(ndarray::s![n..]).to_owned(),
        )
    }

    fn fisher_project_mat(&self, x: &Array2<f64>, eta: &Array2<f64>) -> Array2<f64> {
        let n = self.n;
        let mut b = Array1::zeros(2 * n);
        for i in 0..n {
            b[i] = vecops::sum(eta.row(i));
        }
        for j in 0..n {
            let mut col = Array1::zeros(n);
            for i in 0..n {
                col[i] = eta[(i, j)];
            }
            b[n + j] = vecops::sum(col.view());
        }
        let (alpha, beta) = self.linear_solve(x, &b);
        let mut out = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                out[(i, j)] = eta[(i, j)] - (alpha[i] + beta[j]) * x[(i, j)];
            }
        }
        out
    }

    fn sinkhorn(&self, mut a: Array2<f64>) -> Array2<f64> {
        let n = self.n;
        let maxit = 100 + 2 * n;
        for _ in 0..maxit {
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] = a[(i, j)].max(f64::EPSILON);
                }
                let rs = vecops::sum(a.row(i));
                if rs > 0.0 && rs.is_finite() {
                    for j in 0..n {
                        a[(i, j)] /= rs;
                    }
                }
            }
            for j in 0..n {
                let mut col = Array1::zeros(n);
                for i in 0..n {
                    col[i] = a[(i, j)];
                }
                let cs = vecops::sum(col.view());
                if cs > 0.0 && cs.is_finite() {
                    for i in 0..n {
                        a[(i, j)] /= cs;
                    }
                }
            }
        }
        a
    }
}

impl Manifold for MultinomialDoublyStochastic {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if self.n >= 2 && n == self.n * self.n {
            Ok(())
        } else {
            Err(self.n.max(2) * self.n.max(2))
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let n = self.n;
        if x.len() != n * n || v.len() != n * n {
            return v.clone();
        }
        let xm = self.mat(x);
        let vm = self.mat(v);
        self.pack_mat(&self.fisher_project_mat(&xm, &vm))
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        let n = self.n;
        if x.len() != n * n || egrad.len() != n * n {
            return egrad.clone();
        }
        let mut mu = Array1::zeros(n * n);
        for i in 0..n * n {
            mu[i] = x[i] * egrad[i];
        }
        self.project(x, &mu)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let n = self.n;
        if x.len() != n * n || v.len() != n * n {
            return x + v;
        }
        let xm = self.mat(x);
        let vm = self.mat(v);
        let mut y = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let xij = xm[(i, j)].max(f64::EPSILON);
                y[(i, j)] = xij * (vm[(i, j)] / xij).exp();
            }
        }
        let mut y = self.sinkhorn(y);
        for i in 0..n {
            for j in 0..n {
                y[(i, j)] = y[(i, j)].max(f64::EPSILON);
            }
        }
        self.pack_mat(&y)
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn barycenter(n: usize) -> Array1<f64> {
        Array1::from_elem(n * n, 1.0 / n as f64)
    }

    fn row_col_sums(n: usize, y: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let mut rows = Array1::zeros(n);
        let mut cols = Array1::zeros(n);
        for i in 0..n {
            rows[i] = vecops::sum(ArrayView1::from(&y.as_slice().unwrap()[i * n..(i + 1) * n]));
            for j in 0..n {
                cols[j] += y[i * n + j];
            }
        }
        (rows, cols)
    }

    #[test]
    fn retract_stays_doubly_stochastic() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let x = barycenter(2);
        let v = array![0.1, -0.1, -0.1, 0.1];
        let y = m.retract(&x, &v);
        assert!(y.iter().all(|&yi| yi > 0.0), "left the interior {y:?}");
        let (rows, cols) = row_col_sums(2, &y);
        assert!((rows[0] - 1.0).abs() < 1e-12, "row0 {}", rows[0]);
        assert!((rows[1] - 1.0).abs() < 1e-12, "row1 {}", rows[1]);
        assert!((cols[0] - 1.0).abs() < 1e-12, "col0 {}", cols[0]);
        assert!((cols[1] - 1.0).abs() < 1e-12, "col1 {}", cols[1]);
        assert!(is_doubly_stochastic(&y));
        assert_ne!(
            crate::manifold::ManifoldKind::MultinomialDoublyStochastic { n: 2 },
            crate::manifold::ManifoldKind::Sphere
        );
    }

    #[test]
    fn project_is_tangent() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let x = barycenter(2);
        let v = array![1.0, 2.0, 3.0, 4.0];
        let t = m.project(&x, &v);
        let (rows, cols) = row_col_sums(2, &t);
        assert!(rows[0].abs() < 1e-12, "row0 {}", rows[0]);
        assert!(rows[1].abs() < 1e-12, "row1 {}", rows[1]);
        assert!(cols[0].abs() < 1e-12, "col0 {}", cols[0]);
        assert!(cols[1].abs() < 1e-12, "col1 {}", cols[1]);
    }

    #[test]
    fn project_is_tangent_off_barycenter() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let seed = array![0.7, 0.3, 0.3, 0.7];
        let x = m.retract(&seed, &Array1::zeros(4));
        assert!(is_doubly_stochastic(&x), "seed retract {x:?}");
        let v = array![1.0, -0.5, 0.25, 2.0];
        let t = m.project(&x, &v);
        let (rows, cols) = row_col_sums(2, &t);
        assert!(rows[0].abs() < 1e-10, "row0 {}", rows[0]);
        assert!(rows[1].abs() < 1e-10, "row1 {}", rows[1]);
        assert!(cols[0].abs() < 1e-10, "col0 {}", cols[0]);
        assert!(cols[1].abs() < 1e-10, "col1 {}", cols[1]);
    }

    #[test]
    fn required_dim_is_n_squared() {
        let m = MultinomialDoublyStochastic { n: 3 };
        assert_eq!(m.required_dim(9), Ok(()));
        assert_eq!(m.required_dim(3), Err(9));
        assert_eq!(m.required_dim(4), Err(9));
        assert_eq!(side(9), Some(3));
        assert_eq!(side(3), None);
        assert_eq!(side(1), None);
    }

    #[test]
    fn pack_unpack_round_trip() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let a = array![[0.2, 0.8], [0.8, 0.2]];
        let p = m.pack(&a);
        let u = m.unpack(&p).expect("unpack");
        assert_eq!(p, array![0.2, 0.8, 0.8, 0.2]);
        assert_eq!(u, a);
        let (n, flat) = unpack(&p).expect("module unpack");
        assert_eq!(n, 2);
        assert_eq!(pack(n, flat), p);
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let x = barycenter(2);
        let y = m.retract(&x, &array![0.05, -0.05, -0.05, 0.05]);
        let v = array![0.1, -0.2, 0.3, -0.2];
        let t = m.transport(&x, &y, &v);
        let p = m.project(&y, &v);
        assert!((&t - &p).mapv(f64::abs).sum() < 1e-14);
    }

    #[test]
    fn egrad2rgrad_is_fisher() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let x = barycenter(2);
        let egrad = array![1.0, 2.0, 3.0, 4.0];
        let r = m.egrad2rgrad(&x, &egrad);
        let mut mu = Array1::zeros(4);
        for i in 0..4 {
            mu[i] = x[i] * egrad[i];
        }
        let p = m.project(&x, &mu);
        assert!((&r - &p).mapv(f64::abs).sum() < 1e-14);
        let (rows, cols) = row_col_sums(2, &r);
        assert!(rows[0].abs() < 1e-12);
        assert!(cols[0].abs() < 1e-12);
    }

    #[test]
    fn fisher_inner_uses_vecops() {
        let x = barycenter(2);
        let u = array![0.1, -0.1, -0.1, 0.1];
        let ip = inner(&x, &u, &u);
        assert!((ip - 0.08).abs() < 1e-14, "inner {ip}");
        assert!((typical_dist(3) - 3.0).abs() < 1e-15);
    }
}
