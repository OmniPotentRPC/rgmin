//! Compact limited-memory trust-region step.
//!
//! The Hessian is the Byrd–Nocedal–Schnabel product
//! `B = γ I − W N⁻¹ Wᵀ`, with `W = [γ S, Y]` and
//! `N = [γ SᵀS, L; Lᵀ, −D]`. `γ = (y·y)/(s·y)` from the newest
//! stored pair, `D = diag(s·y)`, and `L` is strictly lower
//! triangular, `Lᵢⱼ = sᵢ·yⱼ` for `i > j`.
//!
//! The trust region is the per-atom ball of radius `cap`, not the
//! Euclidean ball. Inside the ball the step is the two-loop
//! minimizer. On the boundary, `λ > 0` solves `(B + λ I) s = −g`
//! with `max_atom(s) = cap`.
//!
//! Byrd, Nocedal, and Schnabel, *Representations of quasi-Newton
//! matrices and their use in limited memory methods*, Math. Program.
//! 63, 129 (1994), <https://doi.org/10.1007/BF01582063>.
//! Burdakov, Gong, Yuan, and Zikrin, Math. Program. Comput. 9, 501
//! (2017), <https://doi.org/10.1007/s12532-016-0109-7>.
//! Brust, Erway, and Marcia, Comput. Optim. Appl. 66, 245 (2017),
//! <https://doi.org/10.1007/s10589-016-9868-3>.

/// Per-atom trust step for pairs `(s, y)` stored oldest first.
/// `None` when no pair has `s·y > 0` or the shifted solve fails.
pub(crate) fn atom_capped_step(
    pairs: &[(Vec<f64>, Vec<f64>)],
    g: &[f64],
    cap: f64,
) -> Option<Vec<f64>> {
    if !(cap.is_finite() && cap > 0.0) || g.is_empty() {
        return None;
    }
    let model = Model::new(pairs, g.len())?;
    let free = two_loop(&model, g);
    if free.iter().all(|v| v.is_finite()) && dot(&free, g) < 0.0 && max_atom(&free) <= cap {
        return Some(free);
    }
    let mut lo = 0.0;
    let mut hi = model.gamma.max(1.0);
    let mut fitted = false;
    for _ in 0..24 {
        let s = model.shifted(g, hi)?;
        if max_atom(&s) <= cap {
            fitted = true;
            break;
        }
        hi *= 4.0;
        if !hi.is_finite() {
            return None;
        }
    }
    if !fitted {
        return None;
    }
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        let s = model.shifted(g, mid)?;
        if max_atom(&s) > cap {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut s = model.shifted(g, hi)?;
    let motion = max_atom(&s);
    if motion > cap && motion > 0.0 {
        let scale = cap / motion;
        for v in &mut s {
            *v *= scale;
        }
    }
    (dot(&s, g) < 0.0 && s.iter().all(|v| v.is_finite())).then_some(s)
}

struct Model<'a> {
    s: Vec<&'a [f64]>,
    y: Vec<&'a [f64]>,
    gamma: f64,
    n: usize,
    m: usize,
}

impl<'a> Model<'a> {
    fn new(pairs: &'a [(Vec<f64>, Vec<f64>)], n: usize) -> Option<Self> {
        let mut s = Vec::new();
        let mut y = Vec::new();
        for (si, yi) in pairs {
            if si.len() != n || yi.len() != n {
                continue;
            }
            let sy = dot(si, yi);
            if sy > 0.0 && dot(yi, yi) > 0.0 {
                s.push(si.as_slice());
                y.push(yi.as_slice());
            }
        }
        let m = s.len();
        if m == 0 {
            return None;
        }
        let gamma = {
            let sy = dot(s[m - 1], y[m - 1]);
            let yy = dot(y[m - 1], y[m - 1]);
            yy / sy
        };
        (gamma.is_finite() && gamma > 0.0).then_some(Self { s, y, gamma, n, m })
    }

    /// `B v` with `B = γ I − W N⁻¹ Wᵀ`.
    #[cfg(test)]
    fn apply(&self, v: &[f64]) -> Option<Vec<f64>> {
        let wtv = self.wt(v);
        let ninv = solve_linear(2 * self.m, &self.n_matrix(), &wtv)?;
        let mut out = vec![0.0; self.n];
        for i in 0..self.n {
            out[i] = self.gamma * v[i];
        }
        self.w_axpy(-1.0, &ninv, &mut out);
        Some(out)
    }

    /// `(B + λ I) s = −g`.
    fn shifted(&self, g: &[f64], lambda: f64) -> Option<Vec<f64>> {
        let alpha = self.gamma + lambda;
        if !(alpha.is_finite() && alpha > 0.0) {
            return None;
        }
        let wtg = self.wt(g);
        let q = self.q_matrix(alpha);
        let z = solve_linear(2 * self.m, &q, &wtg)?;
        let mut s = vec![0.0; self.n];
        for i in 0..self.n {
            s[i] = -g[i] / alpha;
        }
        self.w_axpy(1.0 / alpha, &z, &mut s);
        s.iter().all(|v| v.is_finite()).then_some(s)
    }

    fn wt(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; 2 * self.m];
        for i in 0..self.m {
            out[i] = self.gamma * dot(self.s[i], v);
            out[self.m + i] = dot(self.y[i], v);
        }
        out
    }

    fn w_axpy(&self, scale: f64, z: &[f64], out: &mut [f64]) {
        for i in 0..self.m {
            let cs = scale * z[i] * self.gamma;
            let cy = scale * z[self.m + i];
            let s = self.s[i];
            let y = self.y[i];
            for k in 0..self.n {
                out[k] += cs * s[k] + cy * y[k];
            }
        }
    }

    /// Column-major? Row-major `N`.
    fn n_matrix(&self) -> Vec<f64> {
        let m = self.m;
        let dim = 2 * m;
        let mut nmat = vec![0.0; dim * dim];
        for i in 0..m {
            for j in 0..m {
                let ss = dot(self.s[i], self.s[j]);
                nmat[idx(dim, i, j)] = self.gamma * ss;
                // L_ij = s_i · y_j if i > j else 0. N[i, m+j] = L_ij.
                if i > j {
                    nmat[idx(dim, i, m + j)] = dot(self.s[i], self.y[j]);
                }
                // N[m+i, j] = L_ji = s_j · y_i if j > i else 0.
                if j > i {
                    nmat[idx(dim, m + i, j)] = dot(self.s[j], self.y[i]);
                }
            }
            nmat[idx(dim, m + i, m + i)] = -dot(self.s[i], self.y[i]);
        }
        nmat
    }

    fn q_matrix(&self, alpha: f64) -> Vec<f64> {
        let m = self.m;
        let dim = 2 * m;
        let mut q = vec![0.0; dim * dim];
        let nmat = self.n_matrix();
        for i in 0..m {
            for j in 0..m {
                let ss = dot(self.s[i], self.s[j]);
                let sy = dot(self.s[i], self.y[j]);
                let ys = dot(self.y[i], self.s[j]);
                let yy = dot(self.y[i], self.y[j]);
                q[idx(dim, i, j)] = self.gamma * self.gamma * ss;
                q[idx(dim, i, m + j)] = self.gamma * sy;
                q[idx(dim, m + i, j)] = self.gamma * ys;
                q[idx(dim, m + i, m + j)] = yy;
            }
        }
        for i in 0..dim {
            for j in 0..dim {
                q[idx(dim, i, j)] -= alpha * nmat[idx(dim, i, j)];
            }
        }
        q
    }
}

fn idx(dim: usize, row: usize, col: usize) -> usize {
    row * dim + col
}

fn two_loop(model: &Model<'_>, g: &[f64]) -> Vec<f64> {
    let m = model.m;
    let mut q = g.to_vec();
    let mut alpha = vec![0.0; m];
    let mut rho = vec![0.0; m];
    for (i, rho_i) in rho.iter_mut().enumerate() {
        let sy = dot(model.s[i], model.y[i]);
        *rho_i = 1.0 / sy;
    }
    for i in (0..m).rev() {
        let a = rho[i] * dot(model.s[i], &q);
        alpha[i] = a;
        axpy(-a, model.y[i], &mut q);
    }
    let h0 = 1.0 / model.gamma;
    for v in &mut q {
        *v *= h0;
    }
    for i in 0..m {
        let b = rho[i] * dot(model.y[i], &q);
        axpy(alpha[i] - b, model.s[i], &mut q);
    }
    for v in &mut q {
        *v = -*v;
    }
    q
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        s += *x * *y;
    }
    s
}

fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * *xi;
    }
}

fn max_atom(v: &[f64]) -> f64 {
    if let Some(cell) = crate::cell::active(v.len()) {
        return cell.max_motion(v.len(), |k| v[k]);
    }
    let mut m = 0.0_f64;
    let mut i = 0;
    while i < v.len() {
        let end = (i + 3).min(v.len());
        let r2: f64 = v[i..end].iter().map(|x| x * x).sum();
        m = m.max(r2.sqrt());
        i = end;
    }
    m
}

fn solve_linear(n: usize, a: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
    if n == 0 || a.len() != n * n || rhs.len() != n {
        return None;
    }
    let mut m = a.to_vec();
    let mut b = rhs.to_vec();
    for k in 0..n {
        let mut piv = k;
        let mut best = m[k * n + k].abs();
        for i in (k + 1)..n {
            let v = m[i * n + k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if !(best.is_finite()
            && best > 1e-14 * (1.0 + rhs.iter().fold(0.0_f64, |s, v| s.max(v.abs()))))
        {
            return None;
        }
        if piv != k {
            for j in 0..n {
                m.swap(k * n + j, piv * n + j);
            }
            b.swap(k, piv);
        }
        let diag = m[k * n + k];
        for i in (k + 1)..n {
            let f = m[i * n + k] / diag;
            for j in (k + 1)..n {
                let sub = f * m[k * n + j];
                m[i * n + j] -= sub;
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= m[i * n + j] * x[j];
        }
        let diag = m[i * n + i];
        if !diag.is_finite() || diag.abs() < 1e-18 {
            return None;
        }
        x[i] = s / diag;
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// `B s` for the tests.
#[cfg(test)]
fn apply_hessian(pairs: &[(Vec<f64>, Vec<f64>)], v: &[f64]) -> Option<Vec<f64>> {
    Model::new(pairs, v.len())?.apply(v)
}

#[cfg(test)]
fn shifted(pairs: &[(Vec<f64>, Vec<f64>)], g: &[f64], lambda: f64) -> Option<Vec<f64>> {
    Model::new(pairs, g.len())?.shifted(g, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(s: Vec<f64>, y: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        (s, y)
    }

    #[test]
    fn one_pair_obeys_the_secant() {
        let pairs = [pair(vec![1.0, 0.0, 0.0, 0.2], vec![2.0, 0.0, 0.5, 0.1])];
        let bs = apply_hessian(&pairs, &pairs[0].0).unwrap();
        for (i, (got, want)) in bs.iter().zip(pairs[0].1.iter()).enumerate() {
            assert!((got - want).abs() < 1e-8, "secant {i}: {got} vs {want}");
        }
    }

    #[test]
    fn a_positive_shift_solves_the_compact_system() {
        let pairs = [pair(
            vec![0.4, -0.1, 0.0, 0.2, 0.0, 0.1],
            vec![1.2, -0.2, 0.1, 0.5, 0.0, 0.2],
        )];
        let g = vec![0.3, -0.4, 0.2, 0.1, -0.2, 0.05];
        let lambda = 0.7;
        let s = shifted(&pairs, &g, lambda).unwrap();
        let bs = apply_hessian(&pairs, &s).unwrap();
        for i in 0..g.len() {
            let r = bs[i] + lambda * s[i] + g[i];
            assert!(r.abs() < 1e-7, "residual {i} = {r}");
        }
        assert!(dot(&s, &g) < 0.0, "not descent");
    }

    #[test]
    fn the_boundary_step_meets_the_per_atom_cap() {
        let pairs = [pair(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        )];
        // A soft curvature makes the free step long.
        let g = vec![1.0, 0.5, 0.0, -0.4, 0.2, 0.0];
        let cap = 0.05;
        let s = atom_capped_step(&pairs, &g, cap).unwrap();
        let motion = max_atom(&s);
        assert!(
            (motion - cap).abs() < 1e-6,
            "max atom {motion} cap {cap} step {s:?}"
        );
        assert!(dot(&s, &g) < 0.0);
    }

    #[test]
    fn a_short_free_step_is_the_two_loop_step() {
        let pairs = [pair(vec![0.1, 0.0, 0.0], vec![4.0, 0.0, 0.0])];
        let g = vec![0.01, -0.02, 0.0];
        let s = atom_capped_step(&pairs, &g, 10.0).unwrap();
        let model = Model::new(&pairs, 3).unwrap();
        let free = two_loop(&model, &g);
        for i in 0..3 {
            assert!(
                (s[i] - free[i]).abs() < 1e-10,
                "{i}: {} vs {}",
                s[i],
                free[i]
            );
        }
    }
}
