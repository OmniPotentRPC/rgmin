//! Real Grassmannian \(\mathrm{Gr}(n,p)\). manopt `grassmannfactory`.
//!
//! A point is an n-by-p orthonormal frame packed column-major
//! (length `n*p`, manopt `X(:)`). The geometry is the Riemannian
//! quotient of Stiefel: only the column space matters. Projection
//! is the horizontal lift `U - X (X^T U)`. Retraction is the polar
//! factor of `X+U` (manopt default; `Y (Y^T Y)^{-1/2}`). Transport
//! is projection at the arrival point.
//!
//! `p == 1` is \(\mathrm{RP}^{n-1}\). A 3N cluster is
//! [`super::RigidQuotient`], not \(\mathrm{Gr}(3N, 1)\). Length
//! `n*p` does not name `p`; the pair lives on this type.

use ndarray::{Array1, ArrayView1};

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
    /// `Ok` when `n >= p` and `p >= 1` (manopt `grassmannfactory`).
    pub fn new(n: usize, p: usize) -> Result<Self, (usize, usize)> {
        if n >= p && p >= 1 {
            Ok(Self { n, p })
        } else {
            Err((n, p))
        }
    }

    /// Packed length of one frame.
    pub fn packed_len(self) -> usize {
        self.n.saturating_mul(self.p)
    }

    /// Column-major pack of an n-by-p frame. manopt `X(:)`.
    pub fn pack(&self, columns: &[ArrayView1<f64>]) -> Option<Array1<f64>> {
        if columns.len() != self.p {
            return None;
        }
        let mut out = Vec::with_capacity(self.packed_len());
        for c in columns {
            if c.len() != self.n {
                return None;
            }
            out.extend(c.iter().copied());
        }
        Some(Array1::from(out))
    }

    /// Split a packed frame into `p` columns of length `n`.
    pub fn unpack(&self, x: &Array1<f64>) -> Option<Vec<Array1<f64>>> {
        if x.len() != self.packed_len() {
            return None;
        }
        Some(
            (0..self.p)
                .map(|j| x.slice(ndarray::s![j * self.n..(j + 1) * self.n]).to_owned())
                .collect(),
        )
    }

    fn col<'a>(&self, a: &'a [f64], j: usize) -> ArrayView1<'a, f64> {
        ArrayView1::from(&a[j * self.n..(j + 1) * self.n])
    }

    fn write_col(&self, a: &mut [f64], j: usize, col: &Array1<f64>) {
        a[j * self.n..(j + 1) * self.n].copy_from_slice(col.as_slice().unwrap());
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
        let mut out = vec![0.0; self.packed_len()];
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
            self.orth_against(&q, j, &mut v);
            let mut nrm = nrm2(v.view());
            if nrm <= 1e-16 {
                // Rank drop: complete with a leftover standard basis
                // vector so the polar fallback still lands on St(n,p).
                for i in 0..self.n {
                    v = Array1::zeros(self.n);
                    v[i] = 1.0;
                    self.orth_against(&q, j, &mut v);
                    nrm = nrm2(v.view());
                    if nrm > 1e-16 {
                        break;
                    }
                }
            }
            if nrm > 1e-16 {
                v.mapv_inplace(|t| t / nrm);
            }
            self.write_col(&mut q, j, &v);
        }
        q
    }

    fn orth_against(&self, q: &[f64], j: usize, v: &mut Array1<f64>) {
        for a in 0..j {
            let qa = self.col(q, a);
            let s = dot(qa, v.view());
            axpy(-s, qa, v);
        }
    }

    /// Intrinsic dimension \(p(n-p)\). manopt `M.dim`.
    pub fn dim(self) -> usize {
        self.p.saturating_mul(self.n.saturating_sub(self.p))
    }
}

impl Default for Grassmann {
    fn default() -> Self {
        Self { n: 2, p: 1 }
    }
}

impl Manifold for Grassmann {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n == self.packed_len() {
            Ok(())
        } else {
            Err(self.packed_len())
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(uv)) = (x.as_slice(), v.as_slice()) else {
            return v.clone();
        };
        if xv.len() != self.packed_len() || uv.len() != self.packed_len() {
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
        if xv.len() != self.packed_len() || uv.len() != self.packed_len() {
            return x.clone();
        }
        let mut y = Array1::from(xv.to_vec());
        axpy(1.0, ArrayView1::from(uv), &mut y);
        Array1::from(self.polar(y.as_slice().unwrap()))
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
        let mut off = 0.0;
        for j in 1..p {
            for i in 0..j {
                off += a[i + p * j].abs();
            }
        }
        if off < 1e-15 {
            break;
        }
        for i in 0..p {
            for j in (i + 1)..p {
                let apq = a[i + p * j];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = a[i + p * i];
                let aqq = a[j + p * j];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..p {
                    if k == i || k == j {
                        continue;
                    }
                    let aki = a[k + p * i];
                    let akj = a[k + p * j];
                    a[k + p * i] = c * aki - s * akj;
                    a[k + p * j] = s * aki + c * akj;
                    a[i + p * k] = a[k + p * i];
                    a[j + p * k] = a[k + p * j];
                }
                a[i + p * i] = c * c * app + s * s * aqq - 2.0 * s * c * apq;
                a[j + p * j] = s * s * app + c * c * aqq + 2.0 * s * c * apq;
                a[i + p * j] = 0.0;
                a[j + p * i] = 0.0;
                for k in 0..p {
                    let vki = v[k + p * i];
                    let vkj = v[k + p * j];
                    v[k + p * i] = c * vki - s * vkj;
                    v[k + p * j] = s * vki + c * vkj;
                }
            }
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
    use crate::manifold::Sphere;
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
        // Off-diagonal Gram: a horizontal step with both leftover
        // coordinates mixed so Y^T Y is not diagonal.
        let v2 = array![0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.2, 0.4];
        let y2 = g.retract(&x, &v2);
        assert!(yty_err(g, &y2) < 1e-12, "off-diag polar left Gr(4,2) {y2:?}");
    }

    #[test]
    fn rank_drop_retract_stays_on_the_set() {
        let g = Grassmann { n: 4, p: 2 };
        let x = frame_4x2();
        // Y = X + U has a zero first column: polar Gram is singular.
        let u = array![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = g.retract(&x, &u);
        assert!(yty_err(g, &y) < 1e-12, "rank-drop left Gr(4,2) {y:?}");
        let rp = Grassmann::new(3, 1).unwrap();
        let x1 = array![0.0, 1.0, 0.0];
        let y1 = rp.retract(&x1, &(-&x1));
        assert!(yty_err(rp, &y1) < 1e-12, "through-origin left RP^2 {y1:?}");
        assert_eq!(g.dim(), 4);
        assert_eq!(rp.dim(), 2);
        assert_eq!(Grassmann::new(2, 3), Err((2, 3)));
        assert!(Grassmann::new(4, 2).is_ok());
    }

    #[test]
    fn polar_transport_is_representation_equivariant() {
        // manopt grassmannfactory: transp(XQ, retr(XQ, VQ), UQ)
        // equals transp(X, retr(X, V), U) Q. Polar, not QR.
        let g = Grassmann { n: 4, p: 2 };
        let x = frame_4x2();
        let u = array![0.0, 0.0, 0.3, -0.1, 0.0, 0.0, 0.2, 0.4];
        let v = array![0.0, 0.0, 0.1, 0.2, 0.0, 0.0, -0.3, 0.1];
        let th = std::f64::consts::FRAC_PI_6;
        let (c, s) = (th.cos(), th.sin());
        let q = [c, s, -s, c];
        let xv = x.as_slice().unwrap();
        let xq = Array1::from(g.mul_np_pp(xv, &q));
        let uq = Array1::from(g.mul_np_pp(u.as_slice().unwrap(), &q));
        let vq = Array1::from(g.mul_np_pp(v.as_slice().unwrap(), &q));
        let y = g.retract(&x, &v);
        let yq = g.retract(&xq, &vq);
        let t = g.transport(&x, &y, &u);
        let tq = g.transport(&xq, &yq, &uq);
        let t_q = Array1::from(g.mul_np_pp(t.as_slice().unwrap(), &q));
        let err = (&tq - &t_q).mapv(f64::abs).sum();
        assert!(err < 1e-10, "polar transport not Q-equivariant {err} {tq:?} {t_q:?}");
        assert!(yty_err(g, &y) < 1e-12);
        assert!(yty_err(g, &yq) < 1e-12);
    }

    #[test]
    fn p1_matches_the_sphere_and_is_not_a_3n_cluster() {
        let g = Grassmann::new(3, 1).unwrap();
        let x = array![0.0, 1.0, 0.0];
        let v = array![0.1, 0.0, -0.2];
        let ys = Sphere.retract(&x, &v);
        let yg = g.retract(&x, &v);
        assert!((&ys - &yg).mapv(f64::abs).sum() < 1e-14);
        let ps = Sphere.project(&x, &v);
        let pg = g.project(&x, &v);
        assert!((&ps - &pg).mapv(f64::abs).sum() < 1e-14);
        assert!(g.required_dim(3).is_ok());
        assert_eq!(g.required_dim(114), Err(3));
        assert!(Grassmann::new(114, 1).unwrap().required_dim(114).is_ok());
        assert_eq!(
            Grassmann { n: 4, p: 2 }.required_dim(114),
            Err(8)
        );
    }

    #[test]
    fn pack_round_trips_columns() {
        let g = Grassmann { n: 4, p: 2 };
        let c0 = array![1.0, 0.0, 0.0, 0.0];
        let c1 = array![0.0, 1.0, 0.0, 0.0];
        let x = g.pack(&[c0.view(), c1.view()]).unwrap();
        assert_eq!(x, frame_4x2());
        let cols = g.unpack(&x).unwrap();
        assert_eq!(cols[0], c0);
        assert_eq!(cols[1], c1);
        assert!(g.pack(&[c0.view()]).is_none());
        assert!(g.unpack(&Array1::zeros(114)).is_none());
    }

    #[test]
    fn zero_step_is_the_point() {
        let g = Grassmann { n: 4, p: 2 };
        let x = frame_4x2();
        let y = g.retract(&x, &Array1::zeros(8));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-12);
    }

    #[test]
    fn new_rejects_p_gt_n_and_p_zero() {
        assert_eq!(Grassmann::new(2, 3).err(), Some((2, 3)));
        assert_eq!(Grassmann::new(4, 0).err(), Some((4, 0)));
        assert!(Grassmann::new(5, 3).is_ok());
    }

    #[test]
    fn inv_sqrt_of_a_diagonal_gram() {
        let w = inv_sqrt_spd(&[4.0, 0.0, 0.0, 9.0], 2).unwrap();
        assert!((w[0] - 0.5).abs() < 1e-12);
        assert!((w[3] - 1.0 / 3.0).abs() < 1e-12);
        assert!(w[1].abs() < 1e-12 && w[2].abs() < 1e-12);
    }

    #[test]
    fn retract_stays_on_gr53() {
        let g = Grassmann::new(5, 3).unwrap();
        let mut x = Array1::zeros(15);
        x[0] = 1.0;
        x[6] = 1.0;
        x[12] = 1.0;
        let ambient = array![
            0.1, 0.2, -0.1, 0.3, 0.0, //
            -0.2, 0.1, 0.2, 0.0, 0.4, //
            0.0, -0.3, 0.1, 0.2, -0.1
        ];
        let v = g.project(&x, &ambient);
        let xtv = g.xtu(x.as_slice().unwrap(), v.as_slice().unwrap());
        for s in xtv {
            assert!(s.abs() < 1e-12, "step not horizontal {s}");
        }
        let y = g.retract(&x, &v);
        assert!(yty_err(g, &y) < 1e-12, "left Gr(5,3) {y:?}");
        let t = g.transport(&x, &y, &v);
        let ytt = g.xtu(y.as_slice().unwrap(), t.as_slice().unwrap());
        for s in ytt {
            assert!(s.abs() < 1e-12, "transport not horizontal {s}");
        }
    }

    #[test]
    fn inv_sqrt_of_an_off_diagonal_gram() {
        // G = [[2, 1], [1, 2]]; G^{-1/2} = (1/2) [[1/sqrt(3)+1, 1/sqrt(3)-1],
        // [1/sqrt(3)-1, 1/sqrt(3)+1]].
        let w = inv_sqrt_spd(&[2.0, 1.0, 1.0, 2.0], 2).unwrap();
        let a = 0.5 * (1.0 / 3.0_f64.sqrt() + 1.0);
        let b = 0.5 * (1.0 / 3.0_f64.sqrt() - 1.0);
        assert!((w[0] - a).abs() < 1e-12, "{}", w[0]);
        assert!((w[3] - a).abs() < 1e-12, "{}", w[3]);
        assert!((w[1] - b).abs() < 1e-12, "{}", w[1]);
        assert!((w[2] - b).abs() < 1e-12, "{}", w[2]);
    }

    #[test]
    fn shaped_retract_stays_on_gr42_and_is_not_rp() {
        use crate::manifold::ManifoldKind;
        let x = frame_4x2();
        let v = array![0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1];
        let y = ManifoldKind::Grassmann.retract_shaped(Some((4, 2)), &x, &v);
        let y_rp = ManifoldKind::Grassmann.retract(&x, &v);
        assert!(yty_err(Grassmann { n: 4, p: 2 }, &y) < 1e-12, "left Gr(4,2) {y:?}");
        assert!(
            (&y - &y_rp).mapv(f64::abs).sum() > 1e-6,
            "shaped Gr(4,2) must not be RP^7 of the packed vector"
        );
        let t = ManifoldKind::Grassmann.transport_shaped(Some((4, 2)), &x, &y, &v);
        let xtt = Grassmann { n: 4, p: 2 }.xtu(y.as_slice().unwrap(), t.as_slice().unwrap());
        for s in xtt {
            assert!(s.abs() < 1e-12, "transport not horizontal {s}");
        }
    }
}
