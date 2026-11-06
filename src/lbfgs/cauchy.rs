//! Generalized Cauchy point and subspace minimization for
//! [`super::Lbfgs::minimize_box`].
//!
//! Byrd, Lu, Nocedal and Zhu, *A limited memory algorithm for bound
//! constrained optimization*, <https://doi.org/10.1137/0916069>, sections
//! 3 to 5 (algorithm CP and the direct primal method). The projection of
//! the subspace step is the Morales and Nocedal remark,
//! <https://doi.org/10.1145/2049662.2049669>. The compact form is Byrd,
//! Nocedal and Schnabel, <https://doi.org/10.1007/BF01582063>.

use ndarray::{Array1, Array2, ArrayView1, Axis};

use super::Pair;

/// `B = θ I - W M W^T` with `W = [Y, θ S]` and
/// `M^{-1} = [[-D, L^T], [L, θ S^T S]]`.
pub(super) struct Compact {
    theta: f64,
    w: Array2<f64>,
    k: Array2<f64>,
}

impl Compact {
    /// The compact matrix of `pairs`, oldest first. No pairs is `B = I`.
    pub(super) fn new(n: usize, pairs: &[Pair]) -> Self {
        let m = pairs.len();
        let theta = pairs.last().map_or(1.0, |p| p.y.dot(&p.y) * p.rho);
        let mut w = Array2::zeros((n, 2 * m));
        let mut k = Array2::zeros((2 * m, 2 * m));
        for (i, pi) in pairs.iter().enumerate() {
            w.column_mut(i).assign(&pi.y);
            w.column_mut(m + i).assign(&(theta * &pi.s));
            k[(i, i)] = -1.0 / pi.rho;
            for (j, pj) in pairs.iter().enumerate() {
                if i > j {
                    k[(m + i, j)] = pi.s.dot(&pj.y);
                    k[(j, m + i)] = k[(m + i, j)];
                }
                k[(m + i, m + j)] = theta * pi.s.dot(&pj.s);
            }
        }
        Self { theta, w, k }
    }

    fn mv(&self, v: &Array1<f64>) -> Option<Array1<f64>> {
        solve(self.k.clone(), v.clone())
    }
}

/// Solves `a x = b` by Gaussian elimination with row exchanges. The pivot
/// test is relative to the largest entry, since `s·y` near a minimum is far
/// below any fixed floor.
fn solve(mut a: Array2<f64>, mut b: Array1<f64>) -> Option<Array1<f64>> {
    let n = b.len();
    let floor = f64::EPSILON * a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[(i, c)].abs().total_cmp(&a[(j, c)].abs()))?;
        let pivot = a[(p, c)].abs();
        if pivot <= floor || !pivot.is_finite() {
            return None;
        }
        for j in 0..n {
            a.swap((c, j), (p, j));
        }
        b.swap(c, p);
        for r in (c + 1)..n {
            let f = a[(r, c)] / a[(c, c)];
            for j in c..n {
                a[(r, j)] -= f * a[(c, j)];
            }
            b[r] -= f * b[c];
        }
    }
    for r in (0..n).rev() {
        let tail: f64 = ((r + 1)..n).map(|j| a[(r, j)] * b[j]).sum();
        b[r] = (b[r] - tail) / a[(r, r)];
    }
    Some(b)
}

/// Algorithm CP: the first local minimizer of the quadratic model along the
/// projected steepest-descent path `clip(x - t g, lo, hi)`. Returns the
/// point and `c = W^T (x_cp - x)`, or `None` when `M` is singular.
pub(super) fn cauchy(
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    lo: ArrayView1<f64>,
    hi: ArrayView1<f64>,
    b: &Compact,
) -> Option<(Array1<f64>, Array1<f64>)> {
    let mut d = Array1::zeros(x.len());
    let mut breaks = Vec::new();
    let mut bounded = true;
    for i in 0..x.len() {
        let t = match g[i] {
            gi if gi < 0.0 => (x[i] - hi[i]) / gi,
            gi if gi > 0.0 => (x[i] - lo[i]) / gi,
            _ => 0.0,
        };
        if t > 0.0 {
            d[i] = -g[i];
            if t.is_finite() {
                breaks.push((t, i));
            } else {
                bounded = false;
            }
        }
    }
    breaks.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut xcp = x.to_owned();
    let mut p = b.w.t().dot(&d);
    let mut c = Array1::zeros(p.len());
    let mut f1 = -d.dot(&d);
    if f1 == 0.0 {
        return Some((xcp, c));
    }
    let f2_org = -b.theta * f1 - p.dot(&b.mv(&p)?);
    let mut f2 = f2_org;
    let mut dtm = -f1 / f2;
    let mut t_old = 0.0;
    for (k, &(t, i)) in breaks.iter().enumerate() {
        let dt = t - t_old;
        if dtm < dt {
            break;
        }
        t_old = t;
        c.scaled_add(dt, &p);
        let di = d[i];
        xcp[i] = if di > 0.0 { hi[i] } else { lo[i] };
        let zi = xcp[i] - x[i];
        d[i] = 0.0;
        let wi = b.w.row(i).to_owned();
        let mw = b.mv(&wi)?;
        f1 += dt * f2 + di * di - b.theta * di * zi + di * c.dot(&mw);
        f2 += -b.theta * di * di + 2.0 * di * p.dot(&mw) - di * di * wi.dot(&mw);
        p.scaled_add(-di, &wi);
        f2 = f2.max(f64::EPSILON * f2_org);
        dtm = if k + 1 == breaks.len() && bounded {
            0.0
        } else {
            -f1 / f2
        };
    }
    let dtm = dtm.max(0.0);
    xcp.zip_mut_with(&d, |v, &di| {
        if di != 0.0 {
            *v += (t_old + dtm) * di;
        }
    });
    c.scaled_add(dtm, &p);
    Some((xcp, c))
}

/// Direct primal subspace minimization from the Cauchy point over the
/// variables it leaves strictly inside the box, then the projection test of
/// Morales and Nocedal. Returns the end point `x̄` of the search direction.
pub(super) fn subspace(
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    lo: ArrayView1<f64>,
    hi: ArrayView1<f64>,
    b: &Compact,
    xcp: Array1<f64>,
    c: &Array1<f64>,
) -> Option<Array1<f64>> {
    let free: Vec<usize> = (0..x.len())
        .filter(|&i| lo[i] < xcp[i] && xcp[i] < hi[i])
        .collect();
    if free.is_empty() || b.w.ncols() == 0 {
        return Some(xcp);
    }
    let th = b.theta;
    let mc = b.mv(c)?;
    let wf = b.w.select(Axis(0), &free);
    let r: Array1<f64> = free
        .iter()
        .zip(wf.rows())
        .map(|(&i, w)| w.dot(&mc) - g[i] - th * (xcp[i] - x[i]))
        .collect();
    let u = solve(&b.k - &(wf.t().dot(&wf) / th), wf.t().dot(&r))?;
    let d = &r / th + &(wf.dot(&u) / (th * th));
    let mut xbar = xcp.clone();
    let mut clipped = false;
    for (&i, &di) in free.iter().zip(&d) {
        xbar[i] = (xcp[i] + di).max(lo[i]).min(hi[i]);
        clipped |= xbar[i] != xcp[i] + di;
    }
    if !clipped || (&xbar - &x).dot(&g) <= 0.0 {
        return Some(xbar);
    }
    let alpha = free.iter().zip(&d).fold(1.0_f64, |a, (&i, &di)| match di {
        _ if di < 0.0 => a.min((lo[i] - xcp[i]) / di),
        _ if di > 0.0 => a.min((hi[i] - xcp[i]) / di),
        _ => a,
    });
    for (&i, &di) in free.iter().zip(&d) {
        xbar[i] = (xcp[i] + alpha * di).max(lo[i]).min(hi[i]);
    }
    Some(xbar)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn pair(s: Array1<f64>, y: Array1<f64>) -> Pair {
        let rho = 1.0 / s.dot(&y);
        Pair { s, y, rho }
    }

    /// Coordinate pairs `s = e_i`, `y = a_i e_i` make `B = diag(a)` exactly.
    fn diagonal(a: &[f64]) -> Compact {
        let n = a.len();
        let pairs: Vec<Pair> = (0..n)
            .map(|i| {
                let e = Array1::from_shape_fn(n, |j| f64::from(u8::from(i == j)));
                pair(e.clone(), a[i] * e)
            })
            .collect();
        Compact::new(n, &pairs)
    }

    fn dense(b: &Compact) -> Array2<f64> {
        let n = b.w.nrows();
        let mut out = Array2::from_diag_elem(n, b.theta);
        for i in 0..n {
            let mw = b.mv(&b.w.row(i).to_owned()).unwrap();
            out.row_mut(i).scaled_add(-1.0, &b.w.dot(&mw));
        }
        out
    }

    fn close(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
        a.iter()
            .zip(b)
            .all(|(u, v)| (u - v).abs() <= tol * (1.0 + v.abs()))
    }

    #[test]
    fn compact_form_of_coordinate_pairs_is_the_diagonal() {
        let b = dense(&diagonal(&[1.0, 4.0, 9.0]));
        let want = Array2::from_diag(&array![1.0, 4.0, 9.0]);
        assert!(b.iter().zip(&want).all(|(u, v)| (u - v).abs() < 1e-14));
    }

    #[test]
    fn identity_cauchy_point_bends_at_the_first_bound() {
        // With B = I, x_0 meets its bound 1/2 at t = 1/2 on the path (t, 2t).
        // After that, m' vanishes at t = 1, where x_1 = 2, since
        // m'(t) = -4 + 4t on (1/2, 3/2).
        let b = Compact::new(2, &[]);
        let (xcp, _) = cauchy(
            array![0.0, 0.0].view(),
            array![-1.0, -2.0].view(),
            array![-1.0, -1.0].view(),
            array![0.5, 3.0].view(),
            &b,
        )
        .unwrap();
        assert_eq!(xcp, array![0.5, 2.0]);
    }

    #[test]
    fn cauchy_point_with_pairs_passes_one_breakpoint() {
        // B = diag(1, 4), g = (-2, -4). The free minimum lies at t = 5/17,
        // past the breakpoint of x_0 = 0.4 at t = 0.2. On the next segment
        // m' = -16 + 16 (0.8) + 64 (t - 0.2), zero at t = 0.25.
        let b = diagonal(&[1.0, 4.0]);
        let (xcp, c) = cauchy(
            array![0.0, 0.0].view(),
            array![-2.0, -4.0].view(),
            array![-10.0, -10.0].view(),
            array![0.4, 10.0].view(),
            &b,
        )
        .unwrap();
        assert!(close(&xcp, &array![0.4, 1.0], 1e-14));
        assert!(close(&c, &b.w.t().dot(&xcp), 1e-14));
    }

    #[test]
    fn interior_cauchy_point_is_the_model_minimizer_along_minus_g() {
        let pairs = [
            pair(array![1.0, 0.5, -0.2], array![3.0, 1.1, 0.1]),
            pair(array![-0.3, 1.0, 0.4], array![0.2, 2.6, 1.3]),
        ];
        let b = Compact::new(3, &pairs);
        let g = array![0.7, -1.2, 0.4];
        let t = g.dot(&g) / g.dot(&dense(&b).dot(&g));
        let inf = Array1::from_elem(3, f64::INFINITY);
        let (xcp, _) = cauchy(
            Array1::zeros(3).view(),
            g.view(),
            (-&inf).view(),
            inf.view(),
            &b,
        )
        .unwrap();
        assert!(close(&xcp, &(-t * &g), 1e-13));
    }

    #[test]
    fn subspace_step_reaches_the_box_minimum_of_a_separable_quadratic() {
        // B = diag(1, 4, 9), g = -(2, 4, 9): the Newton point (2, 1, 1)
        // projects to the box minimum (0.4, 1, 1).
        let b = diagonal(&[1.0, 4.0, 9.0]);
        let (x, g) = (Array1::zeros(3), array![-2.0, -4.0, -9.0]);
        let (lo, hi) = (Array1::from_elem(3, -10.0), array![0.4, 10.0, 10.0]);
        let (xcp, c) = cauchy(x.view(), g.view(), lo.view(), hi.view(), &b).unwrap();
        assert!(xcp[0] < 0.4);
        let xbar = subspace(x.view(), g.view(), lo.view(), hi.view(), &b, xcp, &c).unwrap();
        assert!(close(&xbar, &array![0.4, 1.0, 1.0], 1e-13));
    }

    #[test]
    fn unbounded_direction_is_the_two_loop_direction() {
        let mut lbfgs = super::super::Lbfgs::with_capacity(4);
        let a = array![[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]];
        for s in [
            array![1.0, 0.0, 0.3],
            array![0.2, -1.0, 0.1],
            array![0.0, 0.4, 1.0],
        ] {
            let y = a.dot(&s);
            lbfgs.push(s, y);
        }
        let (x, g) = (array![0.3, -0.2, 0.5], array![1.0, -2.0, 0.5]);
        let inf = Array1::from_elem(3, f64::INFINITY);
        let b = Compact::new(3, &lbfgs.memory);
        let (xcp, c) = cauchy(x.view(), g.view(), (-&inf).view(), inf.view(), &b).unwrap();
        let xbar = subspace(x.view(), g.view(), (-&inf).view(), inf.view(), &b, xcp, &c).unwrap();
        assert!(close(&(xbar - &x), &lbfgs.direction(g.view()), 1e-12));
    }
}
