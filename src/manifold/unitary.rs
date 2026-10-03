//! Unitary group \(\mathrm{U}(n)\) for \(n \ge 1\).
//!
//! manopt `unitaryfactory(n)` at `k = 1`. A point is an `n x n`
//! complex unitary matrix packed interleaved `(re, im)` row-major,
//! length `2 n^2`. Tangent vectors are the ambient embedding
//! \(U\Omega\) with \(\Omega^* = -\Omega\), not the Lie-algebra
//! factor alone.
//!
//! Projection is \(U\,\mathrm{skewh}(U^* Z)\). Retraction is
//! `qr_unique` of \(U + V\) (manopt `retr_qr`): Gram-Schmidt
//! columns with a column phase so the diagonal of `R` is real
//! and non-negative. Transport is projection at the arrival
//! point. Ambient add / column norms / Hermitian reductions go
//! through [`crate::vecops`] so the `par` feature applies.
//! Isolated molecules use [`super::RigidQuotient`].

use ndarray::{Array1, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// A packed complex entry: real, then imaginary.
#[derive(Clone, Copy, Debug, Default)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }

    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }

    fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// \(\mathrm{U}(n)\), \(n \ge 1\). Packed interleaved row-major, length `2 n^2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unitary {
    /// Matrix side. Must be `>= 1`.
    pub n: usize,
}

impl Default for Unitary {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl Unitary {
    /// \(\mathrm{U}(n)\) with \(n \ge 1\).
    pub fn new(n: usize) -> Result<Self, usize> {
        if n >= 1 { Ok(Self { n }) } else { Err(n) }
    }

    /// Packed length `2 n^2`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.n.checked_mul(self.n).and_then(|nn| nn.checked_mul(2))
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 1 && self.packed_len() == Some(len)
    }

    /// Interleaved row-major n-by-n complex matrix as length `2 n^2`.
    pub fn pack(&self, mat: &[f64]) -> Option<Array1<f64>> {
        if !self.fits(mat.len()) {
            return None;
        }
        Some(Array1::from(mat.to_vec()))
    }

    /// Interleaved row-major storage of a packed point or tangent.
    pub fn unpack(&self, x: &Array1<f64>) -> Option<Vec<f64>> {
        if !self.fits(x.len()) {
            return None;
        }
        Some(x.to_vec())
    }

    fn at(&self, a: &[f64], i: usize, j: usize) -> C64 {
        let k = 2 * (i * self.n + j);
        C64 {
            re: a[k],
            im: a[k + 1],
        }
    }

    fn put(&self, a: &mut [f64], i: usize, j: usize, z: C64) {
        let k = 2 * (i * self.n + j);
        a[k] = z.re;
        a[k + 1] = z.im;
    }

    /// Column `j` as interleaved length-`2 n` (re, im) pairs.
    fn col(&self, a: &[f64], j: usize) -> Vec<f64> {
        let mut c = vec![0.0; 2 * self.n];
        for i in 0..self.n {
            let z = self.at(a, i, j);
            c[2 * i] = z.re;
            c[2 * i + 1] = z.im;
        }
        c
    }

    fn write_col(&self, a: &mut [f64], j: usize, col: &[f64]) {
        for i in 0..self.n {
            self.put(
                a,
                i,
                j,
                C64 {
                    re: col[2 * i],
                    im: col[2 * i + 1],
                },
            );
        }
    }

    fn mul(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut c = vec![0.0; 2 * n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = C64::ZERO;
                for k in 0..n {
                    acc = acc.add(self.at(a, i, k).mul(self.at(b, k, j)));
                }
                self.put(&mut c, i, j, acc);
            }
        }
        c
    }

    fn hconj(&self, a: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut t = vec![0.0; 2 * n * n];
        for i in 0..n {
            for j in 0..n {
                self.put(&mut t, i, j, self.at(a, j, i).conj());
            }
        }
        t
    }

    /// \(\frac{1}{2}(H - H^*)\): the skew-Hermitian part.
    fn skewh(&self, a: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut s = vec![0.0; 2 * n * n];
        for i in 0..n {
            for j in 0..n {
                let hij = self.at(a, i, j);
                let hji = self.at(a, j, i);
                self.put(&mut s, i, j, hij.sub(hji.conj()).scale(0.5));
            }
        }
        s
    }

    /// Thin QR with real-positive diagonal (Gram-Schmidt + column phase).
    fn qr_unique(&self, y: &mut [f64]) {
        let n = self.n;
        let orig = y.to_vec();
        for j in 0..n {
            let mut v = self.col(y, j);
            for k in 0..j {
                let qk = self.col(y, k);
                let r = herm_dot(&qk, &v);
                caxpy(r, &qk, &mut v);
            }
            let nrm = vecops::nrm2(ArrayView1::from(v.as_slice()));
            if nrm > 1e-16 {
                for t in v.iter_mut() {
                    *t /= nrm;
                }
            }
            // Column phase so R_jj = q_j^* a_j is real and non-negative.
            let aj = self.col(&orig, j);
            let rjj = herm_dot(&v, &aj);
            let abs = rjj.abs();
            if abs > 1e-16 {
                cmul_inplace(rjj.scale(1.0 / abs), &mut v);
            }
            self.write_col(y, j, &v);
        }
    }
}

/// Hermitian inner product of two interleaved columns: \(\sum_i \bar u_i v_i\).
/// Real part is the ambient Euclidean inner product (vecops `dot`).
fn herm_dot(u: &[f64], v: &[f64]) -> C64 {
    let re = vecops::dot(ArrayView1::from(u), ArrayView1::from(v));
    let mut w = vec![0.0; v.len()];
    for k in 0..v.len() / 2 {
        w[2 * k] = v[2 * k + 1];
        w[2 * k + 1] = -v[2 * k];
    }
    let im = vecops::dot(ArrayView1::from(u), ArrayView1::from(w.as_slice()));
    C64 { re, im }
}

/// `v -= r * q` with complex `r` on interleaved pairs.
fn caxpy(r: C64, q: &[f64], v: &mut [f64]) {
    let n = q.len() / 2;
    for k in 0..n {
        let qr = q[2 * k];
        let qi = q[2 * k + 1];
        v[2 * k] -= r.re * qr - r.im * qi;
        v[2 * k + 1] -= r.re * qi + r.im * qr;
    }
}

/// In-place complex scale of an interleaved column.
fn cmul_inplace(r: C64, v: &mut [f64]) {
    let n = v.len() / 2;
    for k in 0..n {
        let vr = v[2 * k];
        let vi = v[2 * k + 1];
        v[2 * k] = r.re * vr - r.im * vi;
        v[2 * k + 1] = r.re * vi + r.im * vr;
    }
}

/// Side length if `len` is `2 n^2` for some \(n \ge 1\).
pub fn side(len: usize) -> Option<usize> {
    if len < 2 || len % 2 != 0 {
        return None;
    }
    let nn = len / 2;
    let n = (nn as f64).sqrt().round() as usize;
    if n >= 1 && n.checked_mul(n) == Some(nn) {
        Some(n)
    } else {
        None
    }
}

/// Split a length-`2 n^2` ambient vector into (n, interleaved entries).
pub fn unpack(x: &Array1<f64>) -> Option<(usize, Vec<f64>)> {
    let n = side(x.len())?;
    Some((n, x.iter().copied().collect()))
}

/// Flatten an interleaved n-by-n complex matrix into the ambient vector.
pub fn pack(n: usize, a: Vec<f64>) -> Array1<f64> {
    Array1::from_shape_vec(2 * n * n, a).unwrap()
}

/// `true` when the packed matrix satisfies \(U^* U = I\) to `1e-8`.
pub fn is_unitary(x: &Array1<f64>) -> bool {
    let Some((n, a)) = unpack(x) else {
        return false;
    };
    let Ok(m) = Unitary::new(n) else {
        return false;
    };
    gram_is_identity(&m, &a, 1e-8)
}

fn gram_is_identity(m: &Unitary, a: &[f64], tol: f64) -> bool {
    let uh = m.hconj(a);
    let g = m.mul(&uh, a);
    let n = m.n;
    for i in 0..n {
        for j in 0..n {
            let z = m.at(&g, i, j);
            let want = if i == j { 1.0 } else { 0.0 };
            if (z.re - want).abs() > tol || z.im.abs() > tol {
                return false;
            }
        }
    }
    true
}

impl Manifold for Unitary {
    fn required_dim(&self, dim: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.n >= 1 && dim == want => Ok(()),
            Some(want) => Err(want),
            None => Err(dim),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(zv)) = (x.as_slice(), v.as_slice()) else {
            return v.clone();
        };
        if !self.fits(xv.len()) || xv.len() != zv.len() {
            return v.clone();
        }
        let uh = self.hconj(xv);
        let h = self.mul(&uh, zv);
        let omega = self.skewh(&h);
        pack(self.n, self.mul(xv, &omega))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return x + v;
        }
        let mut y = Vector::from_host(x.clone());
        vecops::vaxpy(1.0, &Vector::from_host(v.clone()), &mut y);
        {
            let ys = y.host_mut();
            let Some(slice) = ys.as_slice_mut() else {
                return x + v;
            };
            self.qr_unique(slice);
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

    fn identity(n: usize) -> Array1<f64> {
        let mut a = vec![0.0; 2 * n * n];
        for i in 0..n {
            a[2 * (i * n + i)] = 1.0;
        }
        pack(n, a)
    }

    /// Skew-Hermitian ambient step at the identity.
    fn skewh_step(n: usize, scale: f64) -> Array1<f64> {
        let m = Unitary::new(n).unwrap();
        let mut a = vec![0.0; 2 * n * n];
        if n >= 1 {
            // i * scale on the (0,0) diagonal: purely imaginary.
            m.put(&mut a, 0, 0, C64 { re: 0.0, im: scale });
        }
        if n >= 2 {
            m.put(
                &mut a,
                0,
                1,
                C64 {
                    re: -scale,
                    im: 0.5 * scale,
                },
            );
            m.put(
                &mut a,
                1,
                0,
                C64 {
                    re: scale,
                    im: 0.5 * scale,
                },
            );
        }
        pack(n, a)
    }

    fn max_abs(a: &Array1<f64>) -> f64 {
        a.iter().fold(0.0, |m, &t| m.max(t.abs()))
    }

    #[test]
    fn retract_stays_unitary() {
        let m = Unitary::new(2).unwrap();
        let x = identity(2);
        let v = m.project(&x, &skewh_step(2, 0.3));
        let y = m.retract(&x, &v);
        assert!(is_unitary(&y), "left U(2) {y:?}");
        let uh = m.hconj(y.as_slice().unwrap());
        let g = m.mul(&uh, y.as_slice().unwrap());
        for i in 0..2 {
            for j in 0..2 {
                let z = m.at(&g, i, j);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (z.re - want).abs() < 1e-10 && z.im.abs() < 1e-10,
                    "U^* U[{i},{j}] = {z:?}"
                );
            }
        }
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn project_of_a_hermitian_pullback_is_zero() {
        let m = Unitary::new(2).unwrap();
        let x = identity(2);
        // Hermitian Z at U = I: pullback is Z itself.
        let mut z = vec![0.0; 8];
        m.put(&mut z, 0, 0, C64 { re: 2.0, im: 0.0 });
        m.put(&mut z, 0, 1, C64 { re: 0.4, im: 0.3 });
        m.put(&mut z, 1, 0, C64 { re: 0.4, im: -0.3 });
        m.put(&mut z, 1, 1, C64 { re: 1.5, im: 0.0 });
        let p = m.project(&x, &pack(2, z));
        assert!(max_abs(&p) < 1e-12, "Hermitian pullback must vanish {p:?}");
    }

    #[test]
    fn project_of_hermitian_pullback_at_a_non_identity() {
        let m = Unitary::new(2).unwrap();
        // Hadamard-like unitary: 1/sqrt(2) * [1, 1; i, -i].
        let s = 0.5_f64.sqrt();
        let mut u = vec![0.0; 8];
        m.put(&mut u, 0, 0, C64 { re: s, im: 0.0 });
        m.put(&mut u, 0, 1, C64 { re: s, im: 0.0 });
        m.put(&mut u, 1, 0, C64 { re: 0.0, im: s });
        m.put(&mut u, 1, 1, C64 { re: 0.0, im: -s });
        let x = pack(2, u);
        assert!(is_unitary(&x));
        let mut h = vec![0.0; 8];
        m.put(&mut h, 0, 0, C64 { re: 1.0, im: 0.0 });
        m.put(&mut h, 0, 1, C64 { re: 0.2, im: 0.3 });
        m.put(&mut h, 1, 0, C64 { re: 0.2, im: -0.3 });
        m.put(&mut h, 1, 1, C64 { re: 2.0, im: 0.0 });
        let z = pack(2, m.mul(x.as_slice().unwrap(), &h));
        let p = m.project(&x, &z);
        assert!(
            max_abs(&p) < 1e-12,
            "U H with H Hermitian must vanish {p:?}"
        );
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let m = Unitary::new(2).unwrap();
        let x = identity(2);
        let v = m.project(&x, &skewh_step(2, 0.25));
        let y = m.retract(&x, &v);
        let t = m.transport(&x, &y, &v);
        let p = m.project(&y, &v);
        for (a, b) in t.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!(is_unitary(&y));
    }

    #[test]
    fn pack_unpack_is_interleaved_row_major() {
        let m = Unitary::new(1).unwrap();
        let mat = [0.0, 1.0];
        let x = m.pack(&mat).unwrap();
        assert_eq!(x.len(), 2);
        assert_eq!(m.unpack(&x).unwrap(), mat);
        assert!(m.pack(&[1.0]).is_none());
        let (n, flat) = unpack(&x).unwrap();
        assert_eq!(n, 1);
        let y = pack(n, flat);
        assert!((x[0] - y[0]).abs() < 1e-15);
        assert!((x[1] - y[1]).abs() < 1e-15);
    }

    #[test]
    fn u1_matches_complex_circle() {
        use crate::manifold::ComplexCircle;
        let u = Unitary::new(1).unwrap();
        let c = ComplexCircle { n: 1 };
        let x = array![1.0, 0.0];
        let v = array![0.2, 0.3];
        let pu = u.project(&x, &v);
        let pc = c.project(&x, &v);
        assert!((&pu - &pc).mapv(f64::abs).sum() < 1e-14);
        let yu = u.retract(&x, &v);
        let yc = c.retract(&x, &v);
        assert!((&yu - &yc).mapv(f64::abs).sum() < 1e-14);
    }

    #[test]
    fn kind_is_not_sphere_or_complex_circle() {
        use crate::manifold::ManifoldKind;
        assert_ne!(ManifoldKind::unitary(2), ManifoldKind::Sphere);
        assert_ne!(
            ManifoldKind::unitary(1),
            ManifoldKind::ComplexCircle { n: 1 }
        );
        assert_ne!(ManifoldKind::unitary(2), ManifoldKind::Stiefel);
        assert_eq!(ManifoldKind::unitary(2).as_str(), "unitary");
    }

    #[test]
    fn wrong_dim_keeps_length() {
        let m = Unitary::new(2).unwrap();
        let x = Array1::from_elem(12, 0.1);
        let v = Array1::from_elem(12, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 12);
        assert_eq!(m.project(&x, &v).len(), 12);
        assert!(m.required_dim(8).is_ok());
        assert!(m.required_dim(12).is_err());
        assert!(m.required_dim(2).is_err());
        assert!(m.required_dim(114).is_err());
        assert!(Unitary::new(1).unwrap().required_dim(2).is_ok());
        assert!(Unitary::new(0).is_err());
        assert!(side(8).is_some());
        assert!(side(2).is_some());
        assert!(side(12).is_none());
        assert!(side(1).is_none());
        assert!(side(0).is_none());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = Unitary::new(2).unwrap();
        let x = identity(2);
        let y = m.retract(&x, &Array1::zeros(8));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
        assert!(is_unitary(&y));
    }

    #[test]
    fn retract_from_non_identity_stays_unitary() {
        let m = Unitary::new(2).unwrap();
        let s = 0.5_f64.sqrt();
        let mut u = vec![0.0; 8];
        m.put(&mut u, 0, 0, C64 { re: s, im: 0.0 });
        m.put(&mut u, 0, 1, C64 { re: s, im: 0.0 });
        m.put(&mut u, 1, 0, C64 { re: 0.0, im: s });
        m.put(&mut u, 1, 1, C64 { re: 0.0, im: -s });
        let x = pack(2, u);
        assert!(is_unitary(&x));
        let v = m.project(&x, &skewh_step(2, 0.4));
        let y = m.retract(&x, &v);
        assert!(is_unitary(&y), "left U(2) {y:?}");
        let uh = m.hconj(y.as_slice().unwrap());
        let g = m.mul(&uh, y.as_slice().unwrap());
        for i in 0..2 {
            for j in 0..2 {
                let z = m.at(&g, i, j);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (z.re - want).abs() < 1e-10 && z.im.abs() < 1e-10,
                    "U^* U[{i},{j}] = {z:?}"
                );
            }
        }
    }

    #[test]
    fn retract_u3_stays_unitary() {
        let m = Unitary::new(3).unwrap();
        let x = identity(3);
        let v = m.project(&x, &skewh_step(3, 0.2));
        let y = m.retract(&x, &v);
        assert!(is_unitary(&y), "left U(3) {y:?}");
        assert_eq!(y.len(), 18);
        let uh = m.hconj(y.as_slice().unwrap());
        let g = m.mul(&uh, y.as_slice().unwrap());
        for i in 0..3 {
            for j in 0..3 {
                let z = m.at(&g, i, j);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (z.re - want).abs() < 1e-10 && z.im.abs() < 1e-10,
                    "U^* U[{i},{j}] = {z:?}"
                );
            }
        }
    }

    #[test]
    fn projected_step_is_real_orthogonal_to_the_point() {
        let m = Unitary::new(2).unwrap();
        let x = identity(2);
        let v = m.project(&x, &skewh_step(2, 0.3));
        let s = vecops::dot(x.view(), v.view());
        assert!(s.abs() < 1e-14, "Re <U, U Omega> must vanish {s}");
    }
}
