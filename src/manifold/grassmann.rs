//! Real Grassmannian \(\mathrm{Gr}(n,p)\). manopt `grassmannfactory`.
//!
//! A point is an n-by-p orthonormal frame packed column-major
//! (length `n*p`, manopt `X(:)`). The geometry is the Riemannian
//! quotient of Stiefel: only the column space matters. Projection
//! is the horizontal lift `U - X (X^T U)`. Retraction is the polar
//! factor of `X+U` (manopt default; `Y (Y^T Y)^{-1/2}`). Transport
//! is projection at the arrival point.
//!
//! This is not the sphere and not a 3N cluster. Isolated molecules
//! use [`super::RigidQuotient`]. `p` is named on the type: length
//! `n*p` does not name `p`.

use ndarray::{Array1, Array2, ArrayView1};

use crate::vecops::{axpy, dot, nrm2};

use super::Manifold;

/// Real Grassmann \(\mathrm{Gr}(n,p)\). `n >= p >= 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grassmann {
    /// Ambient dimension.
    pub n: usize,
    /// Subspace dimension.
    pub p: usize,
}

impl Grassmann {
    /// Frame \(\mathrm{Gr}(n,p)\). Illegal pairs fail [`Manifold::required_dim`].
    pub fn new(n: usize, p: usize) -> Self {
        Self { n, p }
    }

    /// Packed length of one frame, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.n.checked_mul(self.p)
    }

    fn fits(self, len: usize) -> bool {
        self.n >= self.p && self.p >= 1 && self.packed_len() == Some(len)
    }

    /// Column-major flatten of an n-by-p frame. manopt `X(:)`.
    pub fn pack(mat: &Array2<f64>) -> Array1<f64> {
        let (n, p) = mat.dim();
        let mut out = Array1::zeros(n * p);
        for j in 0..p {
            for i in 0..n {
                out[i + j * n] = mat[[i, j]];
            }
        }
        out
    }

    /// Inverse of [`Self::pack`] for this `(n, p)`.
    pub fn unpack(&self, x: &Array1<f64>) -> Array2<f64> {
        let mut mat = Array2::zeros((self.n, self.p));
        if !self.fits(x.len()) {
            return mat;
        }
        for j in 0..self.p {
            for i in 0..self.n {
                mat[[i, j]] = x[i + j * self.n];
            }
        }
        mat
    }

    fn col<'a>(&self, a: &'a [f64], j: usize) -> ArrayView1<'a, f64> {
        ArrayView1::from(&a[j * self.n..(j + 1) * self.n])
    }

    fn write_col(&self, a: &mut [f64], j: usize, col: &Array1<f64>) {
        let sl = &mut a[j * self.n..(j + 1) * self.n];
        if let Some(src) = col.as_slice() {
            sl.copy_from_slice(src);
        } else {
            for (dst, src) in sl.iter_mut().zip(col.iter()) {
                *dst = *src;
            }
        }
    }

    fn xtu(&self, x: &[f64], u: &[f64]) -> Vec<f64> {
        let mut s = vec![0.0; self.p * self.p];
        for a in 0..self.p {
            for b in 0..self.p {
                s[a + self.p * b] = dot(self.col(x, a), self.col(u, b));
            }
        }
        s
    }

    /// Polar factor `Y (Y^T Y)^{-1/2}`. QR fallback if the Gram matrix
    /// is not SPD (rank drop).
    fn polar(&self, y: &[f64]) -> Vec<f64> {
        let gram = self.xtu(y, y);
        match inv_sqrt_spd(&gram, self.p) {
            Some(w) => self.mul_np_pp(y, &w),
            None => self.qr_thin(y),
        }
    }

    fn mul_np_pp(&self, y: &[f64], w: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n * self.p];
        for j in 0..self.p {
            let mut col = Array1::zeros(self.n);
            for a in 0..self.p {
                let ya = self.col(y, a).to_owned();
                axpy(w[a + self.p * j], ya.view(), &mut col);
            }
            self.write_col(&mut out, j, &col);
        }
        out
    }

    fn qr_thin(&self, y: &[f64]) -> Vec<f64> {
        let mut q = y.to_vec();
        for j in 0..self.p {
            let mut v = self.col(&q, j).to_owned();
            for a in 0..j {
                let qa = self.col(&q, a);
                let s = dot(qa, v.view());
                axpy(-s, qa, &mut v);
            }
            let nrm = nrm2(v.view());
            if nrm > 1e-16 {
                v.mapv_inplace(|t| t / nrm);
            }
            self.write_col(&mut q, j, &v);
        }
        q
    }
}

impl Default for Grassmann {
    fn default() -> Self {
        Self { n: 2, p: 1 }
    }
}

impl Manifold for Grassmann {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.n >= self.p && self.p >= 1 && n == want => Ok(()),
            Some(want) => Err(want),
            None => Err(n),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(uv)) = (x.as_slice(), v.as_slice()) else {
            return v.clone();
        };
        if !self.fits(xv.len()) || xv.len() != uv.len() {
            return v.clone();
        }
        let xtu = self.xtu(xv, uv);
        let mut out = uv.to_vec();
        for j in 0..self.p {
            let mut col = self.col(&out, j).to_owned();
            for a in 0..self.p {
                axpy(-xtu[a + self.p * j], self.col(xv, a), &mut col);
            }
            self.write_col(&mut out, j, &col);
        }
        Array1::from(out)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(uv)) = (x.as_slice(), v.as_slice()) else {
            return x.clone();
        };
        if !self.fits(xv.len()) || xv.len() != uv.len() {
            return x.clone();
        }
        let mut y = xv.to_vec();
        for (yi, ui) in y.iter_mut().zip(uv.iter()) {
            *yi += *ui;
        }
        Array1::from(self.polar(&y))
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

/// Inverse square root of a p-by-p SPD Gram matrix, column-major.
/// `None` if a pivot is not positive.
fn inv_sqrt_spd(g: &[f64], p: usize) -> Option<Vec<f64>> {
    if p == 0 {
        return None;
    }
    if p == 1 {
        if g[0] <= 1e-16 {
            return None;
        }
        return Some(vec![1.0 / g[0].sqrt()]);
    }
    let mut a = g.to_vec();
    let mut v = vec![0.0; p * p];
    for i in 0..p {
        v[i + p * i] = 1.0;
    }
    for _ in 0..(16 * p) {
        let mut m = 0.0;
        let mut pi = 0;
        let mut pj = 1;
        for j in 1..p {
            for i in 0..j {
                let x = a[i + p * j].abs();
                if x > m {
                    m = x;
                    pi = i;
                    pj = j;
                }
            }
        }
        if m < 1e-15 {
            break;
        }
        let app = a[pi + p * pi];
        let aqq = a[pj + p * pj];
        let apq = a[pi + p * pj];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..p {
            let aik = a[pi + p * k];
            let ajk = a[pj + p * k];
            a[pi + p * k] = c * aik - s * ajk;
            a[pj + p * k] = s * aik + c * ajk;
        }
        for k in 0..p {
            let aki = a[k + p * pi];
            let akj = a[k + p * pj];
            a[k + p * pi] = c * aki - s * akj;
            a[k + p * pj] = s * aki + c * akj;
        }
        for k in 0..p {
            let vki = v[k + p * pi];
            let vkj = v[k + p * pj];
            v[k + p * pi] = c * vki - s * vkj;
            v[k + p * pj] = s * vki + c * vkj;
        }
    }
    let mut w = vec![0.0; p * p];
    for k in 0..p {
        let lam = a[k + p * k];
        if lam <= 1e-16 {
            return None;
        }
        let scale = 1.0 / lam.sqrt();
        for i in 0..p {
            for j in 0..p {
                w[i + p * j] += v[i + p * k] * scale * v[j + p * k];
            }
        }
    }
    Some(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn frame_4x2() -> Array1<f64> {
        // Columns (1,0,0,0) and (0,1,0,0), column-major.
        array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

    fn yty_err(g: Grassmann, y: &Array1<f64>) -> f64 {
        let yty = g.xtu(y.as_slice().unwrap(), y.as_slice().unwrap());
        let mut e = 0.0;
        for a in 0..g.p {
            for b in 0..g.p {
                let want = if a == b { 1.0 } else { 0.0 };
                e = e.max((yty[a + g.p * b] - want).abs());
            }
        }
        e
    }

    #[test]
    fn project_is_horizontal() {
        let g = Grassmann { n: 4, p: 2 };
        let x = frame_4x2();
        let v = array![0.2, 0.1, 0.3, -0.4, 0.5, -0.2, 0.1, 0.7];
        let t = g.project(&x, &v);
        let xtt = g.xtu(x.as_slice().unwrap(), t.as_slice().unwrap());
        for s in xtt {
            assert!(s.abs() < 1e-12, "{s}");
        }
    }

    #[test]
    fn retract_stays_on_the_set() {
        let g = Grassmann { n: 4, p: 2 };
        let x = frame_4x2();
        let v = g.project(&x, &array![0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, -0.2]);
        let y = g.retract(&x, &v);
        assert!(yty_err(g, &y) < 1e-12, "left Gr(4,2) {y:?}");
    }

    #[test]
    fn pack_round_trips_columns() {
        let g = Grassmann { n: 4, p: 2 };
        let mut mat = Array2::zeros((4, 2));
        mat[[0, 0]] = 1.0;
        mat[[1, 1]] = 1.0;
        let x = Grassmann::pack(&mat);
        assert_eq!(x, frame_4x2());
        let back = g.unpack(&x);
        assert!((back - mat).mapv(f64::abs).sum() < 1e-15);
        assert_eq!(g.unpack(&Array1::zeros(114)).dim(), (4, 2));
    }

    #[test]
    fn wrong_dim_does_not_shrink_and_rejects_3n() {
        let g = Grassmann { n: 4, p: 2 };
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = g.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(g.project(&x, &v).len(), 114);
        assert_eq!(g.required_dim(114), Err(8));
        assert!(g.required_dim(8).is_ok());
        assert!(Grassmann::new(114, 1).required_dim(114).is_ok());
        assert!(Grassmann::new(2, 3).required_dim(6).is_err());
    }

    #[test]
    fn zero_step_is_the_point() {
        let g = Grassmann { n: 4, p: 2 };
        let x = frame_4x2();
        let y = g.retract(&x, &Array1::zeros(8));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-12);
    }

    #[test]
    fn kind_is_not_sphere() {
        use crate::manifold::ManifoldKind;
        assert_ne!(
            ManifoldKind::Grassmann { n: 3, p: 1 },
            ManifoldKind::Sphere
        );
        assert_ne!(
            ManifoldKind::Grassmann { n: 4, p: 2 },
            ManifoldKind::Stiefel
        );
    }
}
