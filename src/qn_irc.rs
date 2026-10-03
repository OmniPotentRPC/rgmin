//! Sella `QuasiNewton` / `QuasiNewtonIRC` plus the IRC restricted step.
//!
//! `QuasiNewtonIRC.get_s` (zadorlab/sella `optimize/stepper.py`):
//!
//! \[
//! s = V\frac{-(V^\top g + \alpha V^\top d_1)}{|L|+\alpha}.
//! \]
//!
//! The host searches \(\alpha \ge 0\) so \(\|(s+d_1)\odot\sqrt{m}\| = dx\)
//! (`IRCTrustRegion`). When the unrestricted Newton step already sits
//! inside the ball, Gonzalez--Schlegel equality is restored by the
//! radial [`IrcTrust::project`]. Algebra is [`crate::vecops`].

use ndarray::{Array1, Array2};

use crate::hvp::sym_eig_jacobi;
use crate::irc_trust::IrcTrust;
use crate::vecops::{dot, nrm2};

const CURVATURE: f64 = 1e-12;
const DENOM_FLOOR: f64 = 1e-16;
const ALPHA_MAXITER: usize = 64;

/// Dense BFGS Hessian in the mass-weighted metric (Nocedal--Wright 6.19).
///
/// Starts as \(I\). Pair updates are MW: \(s_{\mathrm{mw}} = s \odot \sqrt{m}\),
/// \(y_{\mathrm{mw}} = y \oslash \sqrt{m}\).
#[derive(Clone, Debug)]
pub struct BfgsModel {
    b: Array2<f64>,
    pairs: usize,
}

impl BfgsModel {
    /// Identity Hessian of order `n`.
    pub fn identity(n: usize) -> Self {
        Self {
            b: Array2::<f64>::eye(n),
            pairs: 0,
        }
    }

    /// Drop stored curvature back to \(I\).
    pub fn forget(&mut self) {
        let n = self.b.nrows();
        self.b = Array2::<f64>::eye(n);
        self.pairs = 0;
    }

    /// Rank-1 seed \(B = I + (\lambda-1)uu^\top\) for a unit MW mode.
    ///
    /// IRC plants \(\lambda < 0\) along the imaginary mode so
    /// [`Self::is_posdef`] stays false until the path leaves the
    /// saddle. `pairs` stays 0; only [`Self::update`] counts.
    pub fn seed_mode(&mut self, u: &Array1<f64>, lam: f64) {
        let n = self.b.nrows().min(u.len());
        let mut nrm = 0.0;
        for i in 0..n {
            nrm += u[i] * u[i];
        }
        nrm = nrm.sqrt();
        if nrm <= 1e-16 {
            return;
        }
        let scale = (lam - 1.0) / (nrm * nrm);
        for i in 0..n {
            for j in 0..n {
                self.b[(i, j)] += scale * u[i] * u[j];
            }
        }
    }

    /// True after at least one accepted pair and every MW eigenvalue
    /// is positive. A fresh identity is not a well.
    pub fn is_posdef(&self) -> bool {
        if self.pairs == 0 {
            return false;
        }
        self.eigh().0.iter().all(|&e| e > 1e-8)
    }

    /// Current MW Hessian.
    pub fn hessian(&self) -> &Array2<f64> {
        &self.b
    }

    /// Nocedal--Wright 6.19 on a MW pair. Skips when \(y^\top s \le 0\).
    pub fn update(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        let ys = crate::vecops::dot(y.view(), s.view());
        if ys <= CURVATURE {
            return;
        }
        bfgs_hessian_update(&mut self.b, s, y);
        self.pairs += 1;
    }

    /// Sella TS-BFGS pair. Counts as a pair even when the mode is
    /// negative, so [`Self::is_posdef`] sees the true spectrum.
    pub fn update_ts(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        crate::sella_step::ts_bfgs_update(&mut self.b, s, y);
        self.pairs += 1;
    }

    /// Jacobi spectrum of the stored MW Hessian. Columns of `evecs` are modes.
    pub fn eigh(&self) -> (Array1<f64>, Array2<f64>) {
        sym_eig_jacobi(self.b.clone())
    }
}

/// BFGS Hessian update, Nocedal--Wright 6.19:
/// \(B_+ = B - (Bs)(Bs)^\top/(s^\top Bs) + yy^\top/(y^\top s)\).
pub fn bfgs_hessian_update(b: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    let ys = dot(y.view(), s.view());
    if ys <= CURVATURE {
        return;
    }
    let bs = b.dot(s);
    let sbs = dot(s.view(), bs.view());
    if sbs.abs() <= CURVATURE {
        return;
    }
    let n = s.len();
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] -= bs[i] * bs[j] / sbs;
            b[(i, j)] += y[i] * y[j] / ys;
        }
    }
}

/// Sella `QuasiNewton.get_s`. `order` negative-curvature modes keep
/// \(L_i \leftarrow -|L_i|\) and flip the \(\alpha\) sign on those axes.
pub fn qn_get_s(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g: &Array1<f64>,
    order: usize,
    alpha: f64,
) -> (Array1<f64>, Array1<f64>) {
    let vg = evecs.t().dot(g);
    let n = evals.len();
    let mut sproj = Array1::zeros(n);
    let mut dsproj = Array1::zeros(n);
    let order = order.min(n);
    for i in 0..n {
        let mut lam = evals[i].abs();
        let mut one = 1.0;
        if i < order {
            lam = -lam;
            one = -1.0;
        }
        let denom = lam + alpha * one;
        let denom = if denom.abs() < DENOM_FLOOR {
            DENOM_FLOOR.copysign(if denom == 0.0 { 1.0 } else { denom })
        } else {
            denom
        };
        sproj[i] = vg[i] / denom;
        dsproj[i] = sproj[i] / denom;
    }
    let s = -evecs.dot(&sproj);
    let dsda = evecs.dot(&dsproj);
    (s, dsda)
}

/// Sella `TrustRegion` + `QuasiNewton.get_s`: \(\|s\| \le \delta\).
pub fn qn_restricted(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g: &Array1<f64>,
    order: usize,
    delta: f64,
) -> Array1<f64> {
    let (s0, _) = qn_get_s(evals, evecs, g, order, 0.0);
    let n0 = nrm2(s0.view());
    if n0 <= delta + 1e-14 {
        return s0;
    }
    let mut alpha: f64 = 0.0;
    let mut lower: f64 = 0.0;
    let mut upper: f64 = f64::INFINITY;
    let mut best = s0;
    for _ in 0..ALPHA_MAXITER {
        let (s, ds) = qn_get_s(evals, evecs, g, order, alpha);
        let val = nrm2(s.view());
        let err = val - delta;
        best = s.clone();
        if err.abs() <= 1e-10 {
            return s;
        }
        if lower.is_finite() && upper.is_finite() && (upper - lower) < 1e-14 * (1.0 + upper) {
            return s;
        }
        if err < 0.0 {
            upper = alpha;
        } else {
            lower = alpha;
        }
        let dval = if val > 1e-16 {
            dot(s.view(), ds.view()) / val
        } else {
            0.0
        };
        let mut a1 = alpha - err / dval;
        if !a1.is_finite() || a1 <= lower || a1 >= upper {
            if upper.is_infinite() {
                a1 = alpha + (1.0 + 0.5 * alpha);
            } else {
                a1 = 0.5 * (lower + upper);
            }
        }
        alpha = a1;
    }
    best
}

/// Sella `QuasiNewtonIRC.get_s` in the same coordinates as `g` and `d1`.
pub fn qn_irc_get_s(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g: &Array1<f64>,
    d1: &Array1<f64>,
    alpha: f64,
) -> (Array1<f64>, Array1<f64>) {
    let vg = evecs.t().dot(g);
    let vd1 = evecs.t().dot(d1);
    let n = evals.len();
    let mut sproj = Array1::zeros(n);
    let mut dsproj = Array1::zeros(n);
    for i in 0..n {
        let denom = (evals[i].abs() + alpha).max(DENOM_FLOOR);
        sproj[i] = -(vg[i] + alpha * vd1[i]) / denom;
        dsproj[i] = -(sproj[i] + vd1[i]) / denom;
    }
    (evecs.dot(&sproj), evecs.dot(&dsproj))
}

/// Cartesian gradient and \(d_1\) into the mass-weighted stepper metric.
pub fn to_mw(v: &Array1<f64>, sqrtm: &Array1<f64>, invert: bool) -> Array1<f64> {
    let n = v.len().min(sqrtm.len());
    let mut out = Array1::zeros(v.len());
    for i in 0..n {
        let sm = sqrtm[i].max(1e-16);
        out[i] = if invert { v[i] / sm } else { v[i] * sm };
    }
    out
}

/// Restricted IRC increment: QN-IRC in the MW metric, then
/// [`IrcTrust`] equality \(\|(s+d_1)\odot\sqrt{m}\|=dx\).
pub fn qn_irc_restricted(
    trust: &IrcTrust,
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g: &Array1<f64>,
    allow_interior: bool,
) -> Array1<f64> {
    let g_mw = to_mw(g, &trust.sqrtm, true);
    let d1_mw = to_mw(&trust.d1, &trust.sqrtm, false);
    let (s0_mw, _) = qn_irc_get_s(evals, evecs, &g_mw, &d1_mw, 0.0);
    let s0 = to_mw(&s0_mw, &trust.sqrtm, true);
    let cons0 = trust.cons(&s0);
    // Interior Newton is a well-settling step. On a saddle the
    // model is not yet positive definite; stay on the GS2 sphere.
    if cons0 <= trust.dx + 1e-14 {
        if allow_interior {
            return s0;
        }
        return trust.project(&s0);
    }
    let s = alpha_search(trust, evals, evecs, &g_mw, &d1_mw);
    if trust.on_bound(&s, 1e-8) {
        s
    } else {
        trust.project(&s)
    }
}

/// Identity-Hessian QN-IRC restricted step (first inner iteration).
pub fn qn_irc_restricted_identity(trust: &IrcTrust, g: &Array1<f64>) -> Array1<f64> {
    let n = g.len();
    let evals = Array1::ones(n);
    let evecs = Array2::<f64>::eye(n);
    qn_irc_restricted(trust, &evals, &evecs, g, true)
}

fn alpha_search(
    trust: &IrcTrust,
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g_mw: &Array1<f64>,
    d1_mw: &Array1<f64>,
) -> Array1<f64> {
    let mut alpha: f64 = 0.0;
    let mut lower: f64 = 0.0;
    let mut upper: f64 = f64::INFINITY;
    let mut best = to_mw(
        &qn_irc_get_s(evals, evecs, g_mw, d1_mw, 0.0).0,
        &trust.sqrtm,
        true,
    );
    for _ in 0..ALPHA_MAXITER {
        let (s_mw, ds_mw) = qn_irc_get_s(evals, evecs, g_mw, d1_mw, alpha);
        let s = to_mw(&s_mw, &trust.sqrtm, true);
        let ds = to_mw(&ds_mw, &trust.sqrtm, true);
        let val = trust.cons(&s);
        let err = val - trust.dx;
        best = s.clone();
        if err.abs() <= 1e-10 {
            return s;
        }
        if lower.is_finite() && upper.is_finite() && (upper - lower) < 1e-14 * (1.0 + upper) {
            return s;
        }
        // QN-IRC slope is -1: cons shrinks as alpha grows.
        if err < 0.0 {
            upper = alpha;
        } else {
            lower = alpha;
        }
        let dval = cons_deriv(trust, &s, &ds);
        let mut a1 = alpha - err / dval;
        if !a1.is_finite() || a1 <= lower || a1 >= upper {
            if upper.is_infinite() {
                a1 = alpha + (1.0 + 0.5 * alpha);
            } else {
                a1 = 0.5 * (lower + upper);
            }
        }
        alpha = a1;
    }
    best
}

fn cons_deriv(trust: &IrcTrust, s: &Array1<f64>, ds: &Array1<f64>) -> f64 {
    let n = s
        .len()
        .min(trust.d1.len())
        .min(trust.sqrtm.len())
        .min(ds.len());
    let mut w = Array1::zeros(n);
    let mut dw = Array1::zeros(n);
    for i in 0..n {
        w[i] = (s[i] + trust.d1[i]) * trust.sqrtm[i];
        dw[i] = ds[i] * trust.sqrtm[i];
    }
    let val = nrm2(w.view()).max(1e-16);
    dot(w.view(), dw.view()) / val
}

/// MW pair from a Cartesian displacement and gradient change.
pub fn mw_pair(
    s: &Array1<f64>,
    y: &Array1<f64>,
    sqrtm: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let mut s_mw = Array1::zeros(s.len());
    let mut y_mw = Array1::zeros(y.len());
    let n = s.len().min(y.len()).min(sqrtm.len());
    for i in 0..n {
        let sm = sqrtm[i].max(1e-16);
        s_mw[i] = s[i] * sm;
        y_mw[i] = y[i] / sm;
    }
    (s_mw, y_mw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identity_qn_irc_alpha_zero_is_minus_g() {
        let evals = array![1.0, 1.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![0.4, -0.2];
        let d1 = array![0.1, 0.0];
        let (s, _) = qn_irc_get_s(&evals, &evecs, &g, &d1, 0.0);
        assert!((s[0] + g[0]).abs() < 1e-14);
        assert!((s[1] + g[1]).abs() < 1e-14);
    }

    #[test]
    fn large_alpha_qn_irc_is_minus_d1() {
        let evals = array![2.0, 0.5];
        let evecs = Array2::<f64>::eye(2);
        let g = array![4.0, -1.0];
        let d1 = array![0.3, -0.1];
        let (s, _) = qn_irc_get_s(&evals, &evecs, &g, &d1, 1.0e12);
        assert!((s[0] + d1[0]).abs() < 1e-8, "{}", s[0]);
        assert!((s[1] + d1[1]).abs() < 1e-8, "{}", s[1]);
    }

    #[test]
    fn qn_restricted_clips_a_long_newton_step() {
        let evals = array![1.0, 1.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![4.0, 0.0];
        let s = qn_restricted(&evals, &evecs, &g, 0, 0.5);
        let n = nrm2(s.view());
        assert!((n - 0.5).abs() < 1e-9, "||s||={n}");
    }

    #[test]
    fn qn_restricted_keeps_a_short_newton_step() {
        let evals = array![4.0, 4.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![0.4, 0.0];
        let s = qn_restricted(&evals, &evecs, &g, 0, 0.5);
        assert!((s[0] + 0.1).abs() < 1e-12);
        assert!(nrm2(s.view()) < 0.5);
    }

    #[test]
    fn qn_alpha_zero_is_signed_newton() {
        let evals = array![4.0, -2.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![8.0, 2.0];
        let (s, _) = qn_get_s(&evals, &evecs, &g, 1, 0.0);
        // Sella flips the first `order` slots, not the negative
        // eigenvalues by value: L = (-|4|, |-2|) = (-4, 2);
        // s = -g/L = (2, -1).
        assert!((s[0] - 2.0).abs() < 1e-12);
        assert!((s[1] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn restricted_identity_sits_on_the_mw_sphere() {
        let masses = [1.0, 16.0];
        let d1 = array![0.05, 0.0, 0.0, 0.0, 0.0, 0.0];
        let trust = IrcTrust::from_atom_masses(d1, &masses, 0.2);
        let g = array![1.2, -0.3, 0.1, 0.4, 0.0, -0.2];
        let s = qn_irc_restricted_identity(&trust, &g);
        assert!(
            trust.on_bound(&s, 1e-9),
            "cons={} dx={}",
            trust.cons(&s),
            trust.dx
        );
    }

    #[test]
    fn bfgs_then_restricted_stays_on_the_sphere() {
        let masses = [12.0, 1.0];
        let d1 = Array1::zeros(6);
        let trust = IrcTrust::from_atom_masses(d1, &masses, 0.15);
        let mut model = BfgsModel::identity(6);
        let s_pair = array![0.02, 0.0, 0.0, -0.01, 0.0, 0.0];
        let y_pair = array![0.08, 0.0, 0.0, -0.04, 0.0, 0.0];
        let (sm, ym) = mw_pair(&s_pair, &y_pair, &trust.sqrtm);
        model.update(&sm, &ym);
        let (evals, evecs) = model.eigh();
        let g = array![0.5, 0.1, 0.0, -0.2, 0.0, 0.0];
        let s = qn_irc_restricted(&trust, &evals, &evecs, &g, true);
        assert!(
            trust.on_bound(&s, 1e-9),
            "cons={} dx={}",
            trust.cons(&s),
            trust.dx
        );
    }

    #[test]
    fn fresh_identity_is_not_posdef() {
        assert!(!BfgsModel::identity(4).is_posdef());
    }

    #[test]
    fn seeded_imaginary_mode_stays_indefinite_after_an_orthogonal_pair() {
        let mut model = BfgsModel::identity(4);
        let u = array![1.0, 0.0, 0.0, 0.0];
        model.seed_mode(&u, -1.0);
        let s = array![0.0, 0.1, 0.0, 0.0];
        let y = array![0.0, 0.4, 0.0, 0.0];
        model.update(&s, &y);
        assert!(!model.is_posdef());
        let (evals, _) = model.eigh();
        assert!(
            evals.iter().any(|&e| e < 0.0),
            "seeded mode must survive an orthogonal pair: {evals:?}"
        );
    }

    #[test]
    fn small_newton_stays_inside_the_ball() {
        let masses = [1.0, 1.0];
        let d1 = Array1::zeros(6);
        let trust = IrcTrust::from_atom_masses(d1, &masses, 0.2);
        let g = array![0.01, 0.0, 0.0, 0.0, 0.0, 0.0];
        let s = qn_irc_restricted_identity(&trust, &g);
        assert!(
            trust.cons(&s) < trust.dx - 1e-8,
            "interior Newton must not be grown to dx: cons={}",
            trust.cons(&s)
        );
        assert!((s[0] + g[0]).abs() < 1e-12);
    }

    #[test]
    fn mw_pair_scales_by_sqrt_mass() {
        let s = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let y = array![2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let sqrtm = array![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let (sm, ym) = mw_pair(&s, &y, &sqrtm);
        assert!((sm[0] - 1.0).abs() < 1e-14);
        assert!((sm[3] - 2.0).abs() < 1e-14);
        assert!((ym[0] - 2.0).abs() < 1e-14);
        assert!((ym[3] - 1.0).abs() < 1e-14);
    }
}
