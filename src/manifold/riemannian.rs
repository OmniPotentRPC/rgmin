//! Stiefel, SPD and their products as one runtime manifold.
//!
//! [`RiemannianManifold`] works on caller-owned column-major `f64`
//! slices. A Stiefel factor \(\mathrm{St}(n,p)\) is an `n x p` frame with
//! \(X^\top X = I_p\) under the embedded (Euclidean) metric. An SPD factor
//! \(\mathrm{SPD}(p)\) is a `p x p` symmetric positive definite matrix
//! under the affine-invariant or the log-Euclidean metric. A product packs
//! its factors one after another, each column-major, in the order given.
//! A manifold with one factor is that factor alone.
//!
//! Every operation checks lengths and finiteness and returns
//! [`Error::ManifoldArgument`] rather than guessing. An SPD point must be
//! symmetric and positive definite.
//!
//! - Edelman, Arias, Smith, "The geometry of algorithms with orthogonality
//!   constraints", SIAM J. Matrix Anal. Appl. 20, 303 (1998),
//!   <https://doi.org/10.1137/S0895479895290954>. Stiefel exponential.
//! - Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix
//!   Manifolds*, <https://doi.org/10.1515/9781400830244>. QR and polar
//!   retractions, transport by projection.
//! - Pennec, Fillard, Ayache, "A Riemannian framework for tensor
//!   computing", Int. J. Comput. Vis. 66, 41 (2006),
//!   <https://doi.org/10.1007/s11263-005-3222-z>. Affine-invariant metric.
//! - Arsigny, Fillard, Pennec, Ayache, "Geometric means in a novel vector
//!   space structure on symmetric positive-definite matrices", SIAM J.
//!   Matrix Anal. Appl. 29, 328 (2007), <https://doi.org/10.1137/050637996>.
//!   Log-Euclidean metric.
//!
//! One gradient step on \(\mathrm{St}(3,1) \times \mathrm{SPD}(1)\):
//! ```
//! use rgmin::{ManifoldFactor, RiemannianManifold, SpdMetric, StiefelRetraction};
//!
//! let m = RiemannianManifold::product([
//!     RiemannianManifold::stiefel(3, 1, StiefelRetraction::Qr).unwrap(),
//!     RiemannianManifold::spd(1, SpdMetric::AffineInvariant).unwrap(),
//! ])
//! .unwrap();
//! assert_eq!(m.packed_len(), 4);
//! let x = [1.0, 0.0, 0.0, 2.0];
//! let egrad = [0.0, 1.0, 0.0, 1.0];
//! let (mut rgrad, mut y) = ([0.0; 4], [0.0; 4]);
//! m.gradient_step(&x, &egrad, 0.1, &mut rgrad, &mut y).unwrap();
//! assert!((y[0] * y[0] + y[1] * y[1] + y[2] * y[2] - 1.0).abs() < 1e-15);
//! assert!(y[3] > 0.0 && y[3] < 2.0);
//! assert!(matches!(m.factors()[1], ManifoldFactor::Spd { p: 1, .. }));
//! ```

use std::ops::Range;

use ndarray::Array1;

use super::Manifold;
use crate::{Error, Result};

/// Retraction of a Stiefel factor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StiefelRetraction {
    /// Q factor of \(X + V\) with a positive diagonal R. First order.
    #[default]
    Qr,
    /// Polar factor \((X + V)\,((X + V)^\top (X + V))^{-1/2}\). Second order.
    Polar,
}

/// Metric of an SPD factor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpdMetric {
    /// \(\langle U, V\rangle_X = \mathrm{tr}(X^{-1} U X^{-1} V)\).
    #[default]
    AffineInvariant,
    /// \(\langle U, V\rangle_X = \langle D\log(X)[U], D\log(X)[V]\rangle_F\).
    LogEuclidean,
}

/// One factor of a [`RiemannianManifold`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifoldFactor {
    /// \(\mathrm{St}(n,p)\), `1 <= p <= n`. Column-major `n x p`, length `n p`.
    Stiefel {
        /// Rows of the frame.
        n: usize,
        /// Orthonormal columns.
        p: usize,
        /// Retraction used by [`RiemannianManifold::retract`].
        retraction: StiefelRetraction,
    },
    /// \(\mathrm{SPD}(p)\), `p >= 1`. Column-major `p x p`, length `p^2`.
    /// The retraction is the exponential map of `metric`.
    Spd {
        /// Side length.
        p: usize,
        /// Riemannian metric.
        metric: SpdMetric,
    },
}

impl ManifoldFactor {
    /// Packed length of one point.
    pub fn packed_len(&self) -> usize {
        match *self {
            Self::Stiefel { n, p, .. } => n * p,
            Self::Spd { p, .. } => p * p,
        }
    }

    /// `(rows, columns)` of one point.
    pub fn shape(&self) -> (usize, usize) {
        match *self {
            Self::Stiefel { n, p, .. } => (n, p),
            Self::Spd { p, .. } => (p, p),
        }
    }

    fn project(&self, x: &[f64], v: &[f64], out: &mut [f64]) {
        match *self {
            Self::Stiefel { n, p, .. } => stiefel_project(n, p, x, v, out),
            Self::Spd { p, .. } => out.copy_from_slice(&sym(p, v)),
        }
    }

    fn egrad_to_rgrad(&self, x: &[f64], g: &[f64], out: &mut [f64]) -> Result<()> {
        match *self {
            Self::Stiefel { n, p, .. } => stiefel_project(n, p, x, g, out),
            Self::Spd { p, metric } => {
                let pt = SpdPoint::new(p, x, "x")?;
                let gs = sym(p, g);
                let r = match metric {
                    SpdMetric::AffineInvariant => sym(p, &mul3(p, &pt.x, &gs, &pt.x)),
                    SpdMetric::LogEuclidean => {
                        let gamma = log_divided(&pt.lam);
                        let gg: Vec<f64> = gamma.iter().map(|v| v * v).collect();
                        pt.dlog_inv_by(&gs, &gg)
                    }
                };
                out.copy_from_slice(&r);
            }
        }
        Ok(())
    }

    fn retract(&self, x: &[f64], v: &[f64], out: &mut [f64]) -> Result<()> {
        match *self {
            Self::Stiefel {
                n,
                p,
                retraction: StiefelRetraction::Qr,
            } => {
                let mut y: Vec<f64> = x.iter().zip(v).map(|(a, b)| a + b).collect();
                qr_q(n, p, &mut y)?;
                out.copy_from_slice(&y);
                Ok(())
            }
            Self::Stiefel {
                n,
                p,
                retraction: StiefelRetraction::Polar,
            } => {
                let y: Vec<f64> = x.iter().zip(v).map(|(a, b)| a + b).collect();
                out.copy_from_slice(&polar(n, p, &y)?);
                Ok(())
            }
            Self::Spd { .. } => self.exp(x, v, out),
        }
    }

    fn exp(&self, x: &[f64], v: &[f64], out: &mut [f64]) -> Result<()> {
        match *self {
            Self::Stiefel { n, p, .. } => {
                let mut t = vec![0.0; n * p];
                stiefel_project(n, p, x, v, &mut t);
                out.copy_from_slice(&stiefel_exp(n, p, x, &t));
            }
            Self::Spd { p, metric } => {
                let pt = SpdPoint::new(p, x, "x")?;
                let vs = sym(p, v);
                let y = match metric {
                    SpdMetric::AffineInvariant => {
                        let half = pt.func(f64::sqrt);
                        let ihalf = pt.func(|l| 1.0 / l.sqrt());
                        let m = sym(p, &mul3(p, &ihalf, &vs, &ihalf));
                        let em = sym_func(p, &m, f64::exp);
                        mul3(p, &half, &em, &half)
                    }
                    SpdMetric::LogEuclidean => {
                        let gamma = log_divided(&pt.lam);
                        let d = pt.conj_by(&vs, &gamma, |a, g| a * g);
                        let l: Vec<f64> = pt
                            .func(f64::ln)
                            .iter()
                            .zip(&d)
                            .map(|(a, b)| a + b)
                            .collect();
                        sym_func(p, &sym(p, &l), f64::exp)
                    }
                };
                out.copy_from_slice(&sym(p, &y));
            }
        }
        Ok(())
    }

    fn log(&self, x: &[f64], y: &[f64], out: &mut [f64]) -> Result<()> {
        match *self {
            Self::Stiefel { .. } => Err(bad(
                "log has no closed form on the Stiefel manifold".to_string()
            )),
            Self::Spd { p, metric } => {
                let px = SpdPoint::new(p, x, "x")?;
                let r = match metric {
                    SpdMetric::AffineInvariant => {
                        SpdPoint::new(p, y, "y")?;
                        let half = px.func(f64::sqrt);
                        let ihalf = px.func(|l| 1.0 / l.sqrt());
                        let m = SpdPoint::new(p, &sym(p, &mul3(p, &ihalf, y, &ihalf)), "y")?;
                        mul3(p, &half, &m.func(f64::ln), &half)
                    }
                    SpdMetric::LogEuclidean => {
                        let py = SpdPoint::new(p, y, "y")?;
                        let w: Vec<f64> = py
                            .func(f64::ln)
                            .iter()
                            .zip(px.func(f64::ln))
                            .map(|(a, b)| a - b)
                            .collect();
                        px.dlog_inv_by(&sym(p, &w), &log_divided(&px.lam))
                    }
                };
                out.copy_from_slice(&sym(p, &r));
                Ok(())
            }
        }
    }

    fn transport(&self, x_from: &[f64], x_to: &[f64], v: &[f64], out: &mut [f64]) -> Result<()> {
        match *self {
            Self::Stiefel { n, p, .. } => stiefel_project(n, p, x_to, v, out),
            Self::Spd { p, metric } => {
                let px = SpdPoint::new(p, x_from, "x_from")?;
                let py = SpdPoint::new(p, x_to, "x_to")?;
                let vs = sym(p, v);
                let r = match metric {
                    SpdMetric::AffineInvariant => {
                        let half = px.func(f64::sqrt);
                        let ihalf = px.func(|l| 1.0 / l.sqrt());
                        let m = SpdPoint::new(p, &sym(p, &mul3(p, &ihalf, x_to, &ihalf)), "x_to")?;
                        let e = mul3(p, &half, &m.func(f64::sqrt), &ihalf);
                        mul3(p, &e, &vs, &transpose(p, p, &e))
                    }
                    SpdMetric::LogEuclidean => {
                        let w = px.conj_by(&vs, &log_divided(&px.lam), |a, g| a * g);
                        py.dlog_inv_by(&w, &log_divided(&py.lam))
                    }
                };
                out.copy_from_slice(&sym(p, &r));
            }
        }
        Ok(())
    }

    fn inner(&self, x: &[f64], u: &[f64], v: &[f64]) -> Result<f64> {
        match *self {
            Self::Stiefel { .. } => Ok(frob(u, v)),
            Self::Spd { p, metric } => {
                let pt = SpdPoint::new(p, x, "x")?;
                let (us, vs) = (sym(p, u), sym(p, v));
                Ok(match metric {
                    SpdMetric::AffineInvariant => {
                        let ihalf = pt.func(|l| 1.0 / l.sqrt());
                        frob(&mul3(p, &ihalf, &us, &ihalf), &mul3(p, &ihalf, &vs, &ihalf))
                    }
                    SpdMetric::LogEuclidean => {
                        let gamma = log_divided(&pt.lam);
                        frob(
                            &pt.conj_by(&us, &gamma, |a, g| a * g),
                            &pt.conj_by(&vs, &gamma, |a, g| a * g),
                        )
                    }
                })
            }
        }
    }
}

/// Product of Stiefel and SPD factors on column-major `f64` buffers.
///
/// The packed point is the concatenation of each factor's column-major
/// block in factor order. Tangent vectors and Euclidean gradients use the
/// same packing. Outputs are written to caller slices of the same length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RiemannianManifold {
    factors: Vec<ManifoldFactor>,
    len: usize,
}

fn bad(what: String) -> Error {
    Error::ManifoldArgument(what)
}

impl RiemannianManifold {
    /// \(\mathrm{St}(n,p)\) with `1 <= p <= n`.
    pub fn stiefel(n: usize, p: usize, retraction: StiefelRetraction) -> Result<Self> {
        if p == 0 || n < p {
            return Err(bad(format!(
                "stiefel needs 1 <= p <= n, got n = {n}, p = {p}"
            )));
        }
        let len = n
            .checked_mul(p)
            .ok_or_else(|| bad("stiefel: n * p overflows".to_string()))?;
        Ok(Self {
            factors: vec![ManifoldFactor::Stiefel { n, p, retraction }],
            len,
        })
    }

    /// \(\mathrm{SPD}(p)\) with `p >= 1` under `metric`.
    pub fn spd(p: usize, metric: SpdMetric) -> Result<Self> {
        if p == 0 {
            return Err(bad("spd needs p >= 1".to_string()));
        }
        let len = p
            .checked_mul(p)
            .ok_or_else(|| bad("spd: p * p overflows".to_string()))?;
        Ok(Self {
            factors: vec![ManifoldFactor::Spd { p, metric }],
            len,
        })
    }

    /// Product of `parts` in order. Nested products are flattened.
    /// An empty list is refused.
    pub fn product(parts: impl IntoIterator<Item = Self>) -> Result<Self> {
        let factors: Vec<ManifoldFactor> = parts.into_iter().flat_map(|m| m.factors).collect();
        if factors.is_empty() {
            return Err(bad("product needs at least one factor".to_string()));
        }
        let mut len = 0usize;
        for f in &factors {
            len = len
                .checked_add(f.packed_len())
                .ok_or_else(|| bad("product: packed length overflows".to_string()))?;
        }
        Ok(Self { factors, len })
    }

    /// Factors in packing order.
    pub fn factors(&self) -> &[ManifoldFactor] {
        &self.factors
    }

    /// Packed length of one point, tangent vector or gradient.
    pub fn packed_len(&self) -> usize {
        self.len
    }

    /// Index range of each factor in the packed buffer.
    pub fn ranges(&self) -> Vec<Range<usize>> {
        let mut start = 0;
        self.factors
            .iter()
            .map(|f| {
                let r = start..start + f.packed_len();
                start = r.end;
                r
            })
            .collect()
    }

    fn check(&self, name: &str, a: &[f64]) -> Result<()> {
        if a.len() != self.len {
            return Err(bad(format!(
                "{name}: length {} is not {}",
                a.len(),
                self.len
            )));
        }
        if a.iter().any(|v| !v.is_finite()) {
            return Err(bad(format!("{name}: non-finite entry")));
        }
        Ok(())
    }

    fn check_out(&self, name: &str, a: &[f64]) -> Result<()> {
        if a.len() != self.len {
            return Err(bad(format!(
                "{name}: length {} is not {}",
                a.len(),
                self.len
            )));
        }
        Ok(())
    }

    fn each<F>(&self, out: &mut [f64], mut op: F) -> Result<()>
    where
        F: FnMut(&ManifoldFactor, Range<usize>, &mut [f64]) -> Result<()>,
    {
        let mut scratch = vec![0.0; self.len];
        for (f, r) in self.factors.iter().zip(self.ranges()) {
            op(f, r.clone(), &mut scratch[r])?;
        }
        out.copy_from_slice(&scratch);
        Ok(())
    }

    /// Orthogonal projection of the ambient `v` onto the tangent space at `x`.
    /// Stiefel: \(V - X\,\mathrm{sym}(X^\top V)\). SPD: \(\mathrm{sym}(V)\).
    pub fn project(&self, x: &[f64], v: &[f64], out: &mut [f64]) -> Result<()> {
        self.check("x", x)?;
        self.check("v", v)?;
        self.check_out("out", out)?;
        self.each(out, |f, r, o| {
            f.project(&x[r.clone()], &v[r], o);
            Ok(())
        })
    }

    /// Riemannian gradient at `x` from the Euclidean gradient `egrad`.
    /// Stiefel: the projection. SPD affine-invariant: \(X\,\mathrm{sym}(G)\,X\).
    /// SPD log-Euclidean: \(D\log(X)^{-2}[\mathrm{sym}(G)]\).
    pub fn egrad_to_rgrad(&self, x: &[f64], egrad: &[f64], out: &mut [f64]) -> Result<()> {
        self.check("x", x)?;
        self.check("egrad", egrad)?;
        self.check_out("out", out)?;
        self.each(out, |f, r, o| f.egrad_to_rgrad(&x[r.clone()], &egrad[r], o))
    }

    /// Retraction of the step `v` at `x`. Stiefel uses its
    /// [`StiefelRetraction`]; SPD uses the exponential of its metric.
    pub fn retract(&self, x: &[f64], v: &[f64], out: &mut [f64]) -> Result<()> {
        self.check("x", x)?;
        self.check("v", v)?;
        self.check_out("out", out)?;
        self.each(out, |f, r, o| f.retract(&x[r.clone()], &v[r], o))
    }

    /// Riemannian exponential at `x`. `v` is projected to the tangent
    /// space first. Stiefel: the embedded-metric geodesic of Edelman, Arias
    /// and Smith. SPD: the geodesic of the factor's metric.
    pub fn exp(&self, x: &[f64], v: &[f64], out: &mut [f64]) -> Result<()> {
        self.check("x", x)?;
        self.check("v", v)?;
        self.check_out("out", out)?;
        self.each(out, |f, r, o| f.exp(&x[r.clone()], &v[r], o))
    }

    /// Riemannian logarithm: the tangent `v` at `x` with `exp(x, v) = y`.
    /// SPD factors only; a Stiefel factor is refused.
    pub fn log(&self, x: &[f64], y: &[f64], out: &mut [f64]) -> Result<()> {
        self.check("x", x)?;
        self.check("y", y)?;
        self.check_out("out", out)?;
        self.each(out, |f, r, o| f.log(&x[r.clone()], &y[r], o))
    }

    /// Vector transport of the tangent `v` from `x_from` to `x_to`.
    /// Stiefel: projection at `x_to`. SPD: parallel transport of the metric.
    pub fn transport(
        &self,
        x_from: &[f64],
        x_to: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<()> {
        self.check("x_from", x_from)?;
        self.check("x_to", x_to)?;
        self.check("v", v)?;
        self.check_out("out", out)?;
        self.each(out, |f, r, o| {
            f.transport(&x_from[r.clone()], &x_to[r.clone()], &v[r], o)
        })
    }

    /// Riemannian inner product of tangents `u` and `v` at `x`: the sum of
    /// the factor metrics. SPD factors use the symmetric parts.
    pub fn inner(&self, x: &[f64], u: &[f64], v: &[f64]) -> Result<f64> {
        self.check("x", x)?;
        self.check("u", u)?;
        self.check("v", v)?;
        let mut total = 0.0;
        for (f, r) in self.factors.iter().zip(self.ranges()) {
            total += f.inner(&x[r.clone()], &u[r.clone()], &v[r])?;
        }
        Ok(total)
    }

    /// One Riemannian gradient step: `rgrad_out = egrad_to_rgrad(x, egrad)`
    /// and `x_out = retract(x, -t rgrad_out)`. `t` must be finite.
    pub fn gradient_step(
        &self,
        x: &[f64],
        egrad: &[f64],
        t: f64,
        rgrad_out: &mut [f64],
        x_out: &mut [f64],
    ) -> Result<()> {
        if !t.is_finite() {
            return Err(bad("step size is non-finite".to_string()));
        }
        self.check_out("x_out", x_out)?;
        self.egrad_to_rgrad(x, egrad, rgrad_out)?;
        let step: Vec<f64> = rgrad_out.iter().map(|g| -t * g).collect();
        self.retract(x, &step, x_out)
    }
}

/// The [`Manifold`] trait has no error channel. A refused call returns
/// `v` (projection, transport, gradient) or `x + v` (retraction), as the
/// packed kinds do on a wrong length.
impl Manifold for RiemannianManifold {
    fn required_dim(&self, n: usize) -> std::result::Result<(), usize> {
        if n == self.len { Ok(()) } else { Err(self.len) }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut out = vec![0.0; v.len()];
        match RiemannianManifold::project(self, &x.to_vec(), &v.to_vec(), &mut out) {
            Ok(()) => Array1::from(out),
            Err(_) => v.clone(),
        }
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        let mut out = vec![0.0; egrad.len()];
        match self.egrad_to_rgrad(&x.to_vec(), &egrad.to_vec(), &mut out) {
            Ok(()) => Array1::from(out),
            Err(_) => egrad.clone(),
        }
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut out = vec![0.0; x.len()];
        match RiemannianManifold::retract(self, &x.to_vec(), &v.to_vec(), &mut out) {
            Ok(()) => Array1::from(out),
            Err(_) => x + v,
        }
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut out = vec![0.0; v.len()];
        match RiemannianManifold::transport(
            self,
            &x_from.to_vec(),
            &x_to.to_vec(),
            &v.to_vec(),
            &mut out,
        ) {
            Ok(()) => Array1::from(out),
            Err(_) => v.clone(),
        }
    }
}

// Dense column-major helpers. Entry (i, j) of an m x n matrix is a[i + m j].

fn frob(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `a (m x k) b (k x n)`.
fn mul(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for j in 0..n {
        for l in 0..k {
            let blj = b[l + k * j];
            if blj == 0.0 {
                continue;
            }
            for i in 0..m {
                c[i + m * j] += a[i + m * l] * blj;
            }
        }
    }
    c
}

/// `a^T b` for `a` of `m x k` and `b` of `m x n`.
fn mul_tn(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; k * n];
    for j in 0..n {
        for i in 0..k {
            c[i + k * j] = frob(&a[m * i..m * (i + 1)], &b[m * j..m * (j + 1)]);
        }
    }
    c
}

fn mul3(p: usize, a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
    mul(p, p, p, &mul(p, p, p, a, b), c)
}

fn transpose(m: usize, n: usize, a: &[f64]) -> Vec<f64> {
    let mut t = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            t[j + n * i] = a[i + m * j];
        }
    }
    t
}

fn sym(p: usize, a: &[f64]) -> Vec<f64> {
    let mut s = vec![0.0; p * p];
    for j in 0..p {
        for i in 0..=j {
            let v = 0.5 * (a[i + p * j] + a[j + p * i]);
            s[i + p * j] = v;
            s[j + p * i] = v;
        }
    }
    s
}

fn eye(p: usize) -> Vec<f64> {
    let mut a = vec![0.0; p * p];
    for i in 0..p {
        a[i + p * i] = 1.0;
    }
    a
}

fn norm1(m: usize, a: &[f64]) -> f64 {
    (0..m)
        .map(|j| a[m * j..m * (j + 1)].iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Cyclic Jacobi eigendecomposition of a symmetric `p x p` matrix.
/// Returns the eigenvalues and the eigenvectors as columns.
fn sym_eig(p: usize, a: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut q = eye(p);
    for _ in 0..100 {
        let total: f64 = a.iter().map(|v| v * v).sum();
        let mut off = 0.0;
        for j in 0..p {
            for i in 0..p {
                if i != j {
                    off += a[i + p * j] * a[i + p * j];
                }
            }
        }
        if off == 0.0 || off <= f64::EPSILON * f64::EPSILON * total {
            break;
        }
        for c in 1..p {
            for r in 0..c {
                let arc = a[r + p * c];
                if arc == 0.0 {
                    continue;
                }
                let tau = (a[c + p * c] - a[r + p * r]) / (2.0 * arc);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cs = 1.0 / (1.0 + t * t).sqrt();
                let sn = t * cs;
                for k in 0..p {
                    let (akr, akc) = (a[k + p * r], a[k + p * c]);
                    a[k + p * r] = cs * akr - sn * akc;
                    a[k + p * c] = sn * akr + cs * akc;
                }
                for k in 0..p {
                    let (ark, ack) = (a[r + p * k], a[c + p * k]);
                    a[r + p * k] = cs * ark - sn * ack;
                    a[c + p * k] = sn * ark + cs * ack;
                }
                for k in 0..p {
                    let (qkr, qkc) = (q[k + p * r], q[k + p * c]);
                    q[k + p * r] = cs * qkr - sn * qkc;
                    q[k + p * c] = sn * qkr + cs * qkc;
                }
            }
        }
    }
    ((0..p).map(|i| a[i + p * i]).collect(), q)
}

/// `Q diag(d) Q^T`, exactly symmetric.
fn from_eig(p: usize, q: &[f64], d: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; p * p];
    for j in 0..p {
        for i in 0..=j {
            let v: f64 = (0..p).map(|k| q[i + p * k] * d[k] * q[j + p * k]).sum();
            out[i + p * j] = v;
            out[j + p * i] = v;
        }
    }
    out
}

/// `f(A)` for a symmetric `A`.
fn sym_func(p: usize, a: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
    let (lam, q) = sym_eig(p, a);
    let d: Vec<f64> = lam.into_iter().map(f).collect();
    from_eig(p, &q, &d)
}

/// Divided differences of `log` on the eigenvalues: the Daleckii-Krein
/// kernel of \(D\log(X)\). Entry `(i, j)` is
/// \((\log\lambda_i - \log\lambda_j)/(\lambda_i - \lambda_j)\), and
/// \(1/\lambda_i\) on a tie.
fn log_divided(lam: &[f64]) -> Vec<f64> {
    let p = lam.len();
    let mut g = vec![0.0; p * p];
    for j in 0..p {
        for i in 0..p {
            let d = lam[i] - lam[j];
            g[i + p * j] = if d == 0.0 {
                1.0 / lam[j]
            } else {
                (d / lam[j]).ln_1p() / d
            };
        }
    }
    g
}

/// Matrix exponential of a general `m x m` matrix: scaling and squaring
/// with a Taylor series once the 1-norm is at most 1/2.
fn expm(m: usize, a: &[f64]) -> Vec<f64> {
    let mut s = 0i32;
    let mut scaled = norm1(m, a);
    while scaled > 0.5 && s < 1100 {
        scaled *= 0.5;
        s += 1;
    }
    let f = 0.5f64.powi(s);
    let b: Vec<f64> = a.iter().map(|v| v * f).collect();
    let mut e = eye(m);
    let mut term = eye(m);
    for k in 1..=30 {
        term = mul(m, m, m, &term, &b);
        let inv = 1.0 / k as f64;
        term.iter_mut().for_each(|v| *v *= inv);
        e.iter_mut().zip(&term).for_each(|(a, b)| *a += b);
        if norm1(m, &term) <= 1e-3 * f64::EPSILON {
            break;
        }
    }
    for _ in 0..s {
        e = mul(m, m, m, &e, &e);
    }
    e
}

fn stiefel_project(n: usize, p: usize, x: &[f64], v: &[f64], out: &mut [f64]) {
    let s = sym(p, &mul_tn(n, p, p, x, v));
    let xs = mul(n, p, p, x, &s);
    for (o, (a, b)) in out.iter_mut().zip(v.iter().zip(&xs)) {
        *o = a - b;
    }
}

/// Q factor of `y` (`n x p`) with a positive diagonal R, in place. Modified
/// Gram-Schmidt with one reorthogonalisation pass.
fn qr_q(n: usize, p: usize, y: &mut [f64]) -> Result<()> {
    for j in 0..p {
        let (done, rest) = y.split_at_mut(n * j);
        let col = &mut rest[..n];
        let before = frob(col, col).sqrt();
        for _ in 0..2 {
            for a in 0..j {
                let qa = &done[n * a..n * (a + 1)];
                let r = frob(qa, col);
                col.iter_mut().zip(qa).for_each(|(c, q)| *c -= r * q);
            }
        }
        let nrm = frob(col, col).sqrt();
        if nrm.is_nan() || nrm <= 1e-12 * before {
            return Err(bad(format!(
                "QR retraction: column {j} of x + v is linearly dependent"
            )));
        }
        col.iter_mut().for_each(|c| *c /= nrm);
    }
    Ok(())
}

/// Polar factor `y (y^T y)^{-1/2}` of an `n x p` matrix.
fn polar(n: usize, p: usize, y: &[f64]) -> Result<Vec<f64>> {
    let (lam, q) = sym_eig(p, &sym(p, &mul_tn(n, p, p, y, y)));
    let top = lam.iter().copied().fold(0.0, f64::max);
    if top <= 0.0 || lam.iter().any(|&l| l.is_nan() || l <= 1e-24 * top) {
        return Err(bad("polar retraction: x + v is rank deficient".to_string()));
    }
    let d: Vec<f64> = lam.iter().map(|l| 1.0 / l.sqrt()).collect();
    Ok(mul(n, p, p, y, &from_eig(p, &q, &d)))
}

/// Embedded-metric geodesic `[X, V] expm([[A, -S], [I, A]]) [expm(-A); 0]`
/// with `A = X^T V` and `S = V^T V` (Edelman, Arias, Smith 1998, eq. 2.41).
fn stiefel_exp(n: usize, p: usize, x: &[f64], v: &[f64]) -> Vec<f64> {
    let xtv = mul_tn(n, p, p, x, v);
    let mut a = vec![0.0; p * p];
    for j in 0..p {
        for i in 0..p {
            a[i + p * j] = 0.5 * (xtv[i + p * j] - xtv[j + p * i]);
        }
    }
    let s = mul_tn(n, p, p, v, v);
    let m = 2 * p;
    let mut block = vec![0.0; m * m];
    for j in 0..p {
        for i in 0..p {
            block[i + m * j] = a[i + p * j];
            block[i + m * (j + p)] = -s[i + p * j];
            block[(i + p) + m * (j + p)] = a[i + p * j];
        }
        block[(j + p) + m * j] = 1.0;
    }
    let e = expm(m, &block);
    let neg_a: Vec<f64> = a.iter().map(|v| -v).collect();
    let w = mul(m, p, p, &e[..m * p], &expm(p, &neg_a));
    let mut top = vec![0.0; p * p];
    let mut bottom = vec![0.0; p * p];
    for j in 0..p {
        for i in 0..p {
            top[i + p * j] = w[i + m * j];
            bottom[i + p * j] = w[(i + p) + m * j];
        }
    }
    let xa = mul(n, p, p, x, &top);
    let vb = mul(n, p, p, v, &bottom);
    xa.iter().zip(&vb).map(|(a, b)| a + b).collect()
}

/// Eigendecomposition of an SPD point.
struct SpdPoint {
    p: usize,
    x: Vec<f64>,
    lam: Vec<f64>,
    q: Vec<f64>,
}

impl SpdPoint {
    fn new(p: usize, x: &[f64], name: &str) -> Result<Self> {
        let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for j in 0..p {
            for i in 0..j {
                if (x[i + p * j] - x[j + p * i]).abs() > 1e-10 * scale.max(1.0) {
                    return Err(bad(format!("{name}: SPD point is not symmetric")));
                }
            }
        }
        let xs = sym(p, x);
        let (lam, q) = sym_eig(p, &xs);
        if lam
            .iter()
            .any(|&l| l.is_nan() || l <= 0.0 || l.is_infinite())
        {
            return Err(bad(format!("{name}: SPD point is not positive definite")));
        }
        Ok(Self { p, x: xs, lam, q })
    }

    fn func(&self, f: impl Fn(f64) -> f64) -> Vec<f64> {
        let d: Vec<f64> = self.lam.iter().map(|&l| f(l)).collect();
        from_eig(self.p, &self.q, &d)
    }

    /// `Q op(Q^T A Q, K) Q^T` entrywise with kernel `K`.
    fn conj_by(&self, a: &[f64], k: &[f64], op: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        let p = self.p;
        let mut b = mul_tn(p, p, p, &self.q, &mul(p, p, p, a, &self.q));
        b.iter_mut().zip(k).for_each(|(v, g)| *v = op(*v, *g));
        sym(
            p,
            &mul(
                p,
                p,
                p,
                &mul(p, p, p, &self.q, &b),
                &transpose(p, p, &self.q),
            ),
        )
    }

    /// `Q ((Q^T A Q) / K) Q^T`.
    fn dlog_inv_by(&self, a: &[f64], k: &[f64]) -> Vec<f64> {
        self.conj_by(a, k, |v, g| v / g)
    }
}
