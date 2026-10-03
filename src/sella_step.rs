//! Sella steppers that are not the IRC sphere: RFO, P-RFO, RAS.
//!
//! `RationalFunctionOptimization.get_s` and
//! `PartitionedRationalFunctionOptimization` from zadorlab/sella
//! `optimize/stepper.py`. `RestrictedAtomicStep` from
//! `optimize/restricted_step.py`. Trust-region \(\|s\|\le\delta\) for
//! QN is [`crate::qn_irc::qn_restricted`]. Algebra is [`crate::vecops`].

use ndarray::{Array1, Array2};

use crate::hvp::sym_eig_jacobi;
use crate::vecops::{axpy, dot, nrm2};

const DENOM_FLOOR: f64 = 1e-12;
const ALPHA_MAXITER: usize = 64;

/// Sella `RationalFunctionOptimization.get_s`.
///
/// Augmented matrix \(\alpha\begin{bmatrix}\alpha H & g\\ g^\top & 0\end{bmatrix}\);
/// the step is the `order`-th eigenvector, scaled by \(\alpha/V_{n}\).
pub fn rfo_get_s(h: &Array2<f64>, g: &Array1<f64>, order: usize, alpha: f64) -> Array1<f64> {
    rfo_step_and_shift(h, g, order, alpha).0
}

pub(crate) fn rfo_step_and_shift(
    h: &Array2<f64>,
    g: &Array1<f64>,
    order: usize,
    alpha: f64,
) -> (Array1<f64>, f64) {
    let n = g.len();
    if n == 0 || h.nrows() != n || h.ncols() != n {
        return (Array1::zeros(n), 0.0);
    }
    let a = alpha.max(0.0);
    let mut aug = Array2::<f64>::zeros((n + 1, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = h[(i, j)] * a * a;
        }
        aug[(i, n)] = g[i] * a;
        aug[(n, i)] = g[i] * a;
    }
    let (evals, evecs) = sym_eig_jacobi(aug);
    let mut idxs: Vec<usize> = (0..=n).collect();
    idxs.sort_by(|&i, &j| {
        evals[i]
            .partial_cmp(&evals[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let col = idxs[order.min(n)];
    let mut denom = evecs[(n, col)];
    if denom.abs() < DENOM_FLOOR {
        denom = if denom == 0.0 {
            DENOM_FLOOR
        } else {
            denom.signum() * DENOM_FLOOR
        };
    }
    let mut s = Array1::zeros(n);
    for i in 0..n {
        s[i] = evecs[(i, col)] * a / denom;
    }
    (s, evals[col])
}

/// Sella TrustRegion + RFO: \(\|s\|\le\delta\). Alpha lives in \([0,1]\);
/// \(\alpha=1\) is the unrestricted Banerjee step.
pub fn rfo_restricted(h: &Array2<f64>, g: &Array1<f64>, order: usize, delta: f64) -> Array1<f64> {
    let s1 = rfo_get_s(h, g, order, 1.0);
    let n1 = nrm2(s1.view());
    if n1 <= delta + 1e-14 {
        return s1;
    }
    let mut lo = 0.0;
    let mut hi = 1.0;
    let mut best = s1;
    for _ in 0..ALPHA_MAXITER {
        let mid = 0.5 * (lo + hi);
        let s = rfo_get_s(h, g, order, mid);
        let val = nrm2(s.view());
        best = s;
        if (val - delta).abs() <= 1e-10 {
            return best;
        }
        if val > delta {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    best
}

/// Sella `PartitionedRationalFunctionOptimization`: RFO in the
/// `order` uphill modes plus RFO in the downhill complement, then
/// a Euclidean trust clip.
pub fn prfo_restricted(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g: &Array1<f64>,
    order: usize,
    delta: f64,
) -> Array1<f64> {
    let n = g.len();
    let order = order.min(n);
    if n == 0 {
        return Array1::zeros(0);
    }
    let mut s = Array1::zeros(n);
    if order > 0 {
        let mut gmax = Array1::zeros(order);
        let mut hmax = Array2::<f64>::zeros((order, order));
        for i in 0..order {
            hmax[(i, i)] = evals[i];
            for k in 0..n {
                gmax[i] += evecs[(k, i)] * g[k];
            }
        }
        let smax = rfo_restricted(&hmax, &gmax, order, delta);
        for i in 0..order {
            let mut col = Array1::zeros(n);
            for k in 0..n {
                col[k] = evecs[(k, i)];
            }
            axpy(smax[i], col.view(), &mut s);
        }
    }
    let nmin = n - order;
    if nmin > 0 {
        let mut gmin = Array1::zeros(nmin);
        let mut hmin = Array2::<f64>::zeros((nmin, nmin));
        for i in 0..nmin {
            hmin[(i, i)] = evals[order + i];
            for k in 0..n {
                gmin[i] += evecs[(k, order + i)] * g[k];
            }
        }
        let smin = rfo_restricted(&hmin, &gmin, 0, delta);
        for i in 0..nmin {
            let mut col = Array1::zeros(n);
            for k in 0..n {
                col[k] = evecs[(k, order + i)];
            }
            axpy(smin[i], col.view(), &mut s);
        }
    }
    let nrm = nrm2(s.view());
    if nrm > delta && nrm > 1e-16 {
        s.mapv_inplace(|v| v * (delta / nrm));
    }
    s
}

/// Sella `RestrictedAtomicStep`: scale so the largest per-atom
/// Euclidean displacement is \(\le\delta\). `s` is 3N Cartesian.
pub fn ras_clip(s: &Array1<f64>, delta: f64) -> Array1<f64> {
    let atoms = s.len() / 3;
    if atoms == 0 {
        let n = nrm2(s.view());
        if n <= delta || n <= 1e-16 {
            return s.clone();
        }
        return s * (delta / n);
    }
    let mut maxn = 0.0;
    for i in 0..atoms {
        let r = (s[3 * i] * s[3 * i] + s[3 * i + 1] * s[3 * i + 1] + s[3 * i + 2] * s[3 * i + 2])
            .sqrt();
        if r > maxn {
            maxn = r;
        }
    }
    if maxn <= delta || maxn <= 1e-16 {
        return s.clone();
    }
    s * (delta / maxn)
}

/// Sella `TS-BFGS` for a single pair (`hessian_update._MS_TS_BFGS`).
///
/// Uses \(|B|\) in the secant weight so a negative mode is not
/// forced positive. `B` is overwritten in place and symmetrized.
pub fn ts_bfgs_update(b: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    let n = s.len();
    if n == 0 || b.nrows() != n || b.ncols() != n {
        return;
    }
    if nrm2(s.view()) < 1e-8 {
        return;
    }
    let (lams, vecs) = sym_eig_jacobi(b.clone());
    let bs = b.dot(s);
    let j = y - &bs;
    let mut vts = Array1::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for k in 0..n {
            acc += vecs[(k, i)] * s[k];
        }
        vts[i] = lams[i].abs() * acc;
    }
    let mut absbs = Array1::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for k in 0..n {
            acc += vecs[(i, k)] * vts[k];
        }
        absbs[i] = acc;
    }
    let sy = dot(s.view(), y.view());
    let sa = dot(s.view(), absbs.view());
    let denom = sy * sy + sa * sa;
    if denom.abs() < 1e-16 {
        return;
    }
    let mut u = Array1::zeros(n);
    for i in 0..n {
        u[i] = (sy * y[i] + sa * absbs[i]) / denom;
    }
    let js = dot(j.view(), s.view());
    for i in 0..n {
        for k in 0..n {
            b[(i, k)] += u[i] * j[k] + j[i] * u[k] - js * u[i] * u[k];
        }
    }
    for i in 0..n {
        for k in i + 1..n {
            let v = 0.5 * (b[(i, k)] + b[(k, i)]);
            b[(i, k)] = v;
            b[(k, i)] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rfo_on_identity_points_downhill() {
        let h = Array2::<f64>::eye(2);
        let g = array![2.0, 0.0];
        let s = rfo_get_s(&h, &g, 0, 1.0);
        assert!(s[0] < 0.0, "s={s:?}");
        assert!(s[0].abs() > 1e-8);
    }

    #[test]
    fn rfo_restricted_sits_on_or_inside_delta() {
        let h = Array2::<f64>::eye(2);
        let g = array![8.0, 0.0];
        let s = rfo_restricted(&h, &g, 0, 0.25);
        let n = nrm2(s.view());
        assert!(n <= 0.25 + 1e-8, "||s||={n}");
        assert!(n > 0.1, "step vanished: {n}");
    }

    #[test]
    fn prfo_order_one_moves_along_the_soft_mode() {
        let evals = array![-1.0, 4.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![0.5, 0.4];
        let s = prfo_restricted(&evals, &evecs, &g, 1, 0.5);
        assert!(s[0].abs() > 1e-8, "no uphill component: {s:?}");
        assert!(nrm2(s.view()) <= 0.5 + 1e-8);
    }

    #[test]
    fn ts_bfgs_keeps_a_negative_mode() {
        let mut b = Array2::<f64>::zeros((2, 2));
        b[(0, 0)] = -1.0;
        b[(1, 1)] = 4.0;
        let s = array![0.0, 0.1];
        let y = array![0.0, 0.4];
        ts_bfgs_update(&mut b, &s, &y);
        let (evals, _) = crate::hvp::sym_eig_jacobi(b);
        assert!(
            evals.iter().any(|&e| e < 0.0),
            "TS-BFGS must keep the saddle mode: {evals:?}"
        );
    }

    #[test]
    fn ras_clip_scales_the_largest_atom() {
        let mut s = Array1::zeros(6);
        s[0] = 0.4;
        s[1] = 0.3;
        s[3] = 0.05;
        let c = ras_clip(&s, 0.1);
        let r0 = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let r1 = (c[3] * c[3] + c[4] * c[4] + c[5] * c[5]).sqrt();
        assert!((r0 - 0.1).abs() < 1e-12, "r0={r0}");
        assert!(r1 < 0.1);
    }
}
