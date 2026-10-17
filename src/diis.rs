//! Pulay residual subspace step, with the GDIIS fixed-point image.
//!
//! The coefficients on the stored gradients sum to one and minimize
//! the norm of that combination. The step is the same combination of
//! the error-vector images `x_i - α g_i`, not of the positions alone.
//! Pulay, *Convergence acceleration of iterative sequences. The case
//! of SCF iteration*, <https://doi.org/10.1016/0009-2614(80)80396-4>.
//! Császár and Pulay, *Geometry optimization by direct inversion in
//! the iterative subspace*, J. Mol. Struct. 114, 31 (1984),
//! <https://doi.org/10.1016/0022-2860(84)87198-7>.
//! Farkas and Schlegel, *Methods for optimizing large molecules*,
//! J. Chem. Phys. 111, 10806 (1999),
//! <https://doi.org/10.1063/1.480484>.

use ndarray::Array1;

use crate::step::max_atom_norm;

/// Stored positions and gradients, oldest first.
#[derive(Clone, Debug)]
pub struct DiisState {
    /// Pairs kept. At least 2.
    pub memory: usize,
    xs: Vec<Array1<f64>>,
    gs: Vec<Array1<f64>>,
}

impl DiisState {
    /// Empty history.
    pub fn new(memory: usize) -> Self {
        Self {
            memory: memory.max(2),
            xs: Vec::new(),
            gs: Vec::new(),
        }
    }

    /// Drop every stored pair.
    pub fn clear(&mut self) {
        self.xs.clear();
        self.gs.clear();
    }

    /// Append one pair and drop the oldest past [`Self::memory`].
    pub fn push(&mut self, x: Array1<f64>, g: Array1<f64>) {
        self.xs.push(x);
        self.gs.push(g);
        let extra = self.xs.len().saturating_sub(self.memory);
        if extra > 0 {
            self.xs.drain(0..extra);
            self.gs.drain(0..extra);
        }
    }

    /// Stored gradients, oldest first.
    pub fn gradients(&self) -> &[Array1<f64>] {
        &self.gs
    }

    /// Stored positions, oldest first.
    pub fn positions(&self) -> &[Array1<f64>] {
        &self.xs
    }
}

/// Coefficients `c` with `sum c_i = 1` minimizing `||sum c_i g_i||`.
///
/// `None` when the bordered Gram matrix has no usable pivot.
pub fn pulay_coefficients(gs: &[Array1<f64>]) -> Option<Vec<f64>> {
    let n = gs.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(vec![1.0]);
    }
    let m = n + 1;
    let mut a = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for i in 0..n {
        for j in 0..n {
            a[i * m + j] = gs[i].dot(&gs[j]);
        }
        a[i * m + n] = 1.0;
        a[n * m + i] = 1.0;
    }
    rhs[n] = 1.0;
    if !solve_inplace(&mut a, &mut rhs, m) {
        return None;
    }
    let c = rhs[..n].to_vec();
    if c.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(c)
}

/// `sum c_i x_i`. Lengths must match.
///
/// The bare position mix is the step GDIIS replaced. The unit test
/// compares it with [`gdiis_point`].
#[allow(dead_code)]
pub fn extrapolated(xs: &[Array1<f64>], c: &[f64]) -> Array1<f64> {
    let mut y = Array1::zeros(xs[0].len());
    for (w, x) in c.iter().zip(xs.iter()) {
        y = y + *w * x;
    }
    y
}

/// GDIIS image `sum c_i (x_i - α g_i)`.
///
/// `None` when the Pulay system has no pivot or the image is farther
/// than `8 * cap` from the latest point. The caller then follows `-g`.
pub fn gdiis_point(
    xs: &[Array1<f64>],
    gs: &[Array1<f64>],
    alpha: f64,
    cap: Option<f64>,
) -> Option<Array1<f64>> {
    if xs.len() < 2 || xs.len() != gs.len() || !alpha.is_finite() {
        return None;
    }
    let c = pulay_coefficients(gs)?;
    if c.iter().any(|w| !w.is_finite()) {
        return None;
    }
    let dim = xs[0].len();
    let mut y = Array1::zeros(dim);
    for ((w, x), g) in c.iter().zip(xs.iter()).zip(gs.iter()) {
        if x.len() != dim || g.len() != dim {
            return None;
        }
        for i in 0..dim {
            y[i] += *w * (x[i] - alpha * g[i]);
        }
    }
    if y.iter().any(|v: &f64| !v.is_finite()) {
        return None;
    }
    if let Some(cap) = cap.filter(|c| c.is_finite() && *c > 0.0) {
        let disp = &y - xs.last()?;
        if max_atom_norm(disp.view()) > 8.0 * cap {
            return None;
        }
    }
    Some(y)
}

fn solve_inplace(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for k in 0..n {
        let mut piv = k;
        let mut best = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-14 {
            return false;
        }
        if piv != k {
            for j in k..n {
                a.swap(k * n + j, piv * n + j);
            }
            b.swap(k, piv);
        }
        let diag = a[k * n + k];
        for i in (k + 1)..n {
            let f = a[i * n + k] / diag;
            a[i * n + k] = 0.0;
            for j in (k + 1)..n {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    for k in (0..n).rev() {
        let mut s = b[k];
        for j in (k + 1)..n {
            s -= a[k * n + j] * b[j];
        }
        let diag = a[k * n + k];
        if diag.abs() < 1e-14 {
            return false;
        }
        b[k] = s / diag;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::Control;
    use crate::method::Method;
    use crate::oracle::Oracle;
    use crate::session::Solver;
    use ndarray::{ArrayView1, array};

    #[test]
    fn opposite_residuals_meet_in_the_middle() {
        let gs = [array![1.0], array![-1.0]];
        let c = pulay_coefficients(&gs).expect("two independent residuals");
        assert!((c[0] + c[1] - 1.0).abs() < 1e-12);
        assert!((c[0] - 0.5).abs() < 1e-10);
        let y = extrapolated(&[array![0.0], array![2.0]], &c);
        assert!((y[0] - 1.0).abs() < 1e-10);
    }

    /// Bare `sum c_i x_i` of these two points is `(0.5, 0.5)`. The GDIIS
    /// image of `x - g` on `f = ½||x||²` with `α = 1` is the minimizer.
    #[test]
    fn gdiis_image_of_the_quadratic_is_the_minimizer() {
        let xs = [array![1.0, 0.0], array![0.0, 1.0]];
        let gs = [array![1.0, 0.0], array![0.0, 1.0]];
        let y = gdiis_point(&xs, &gs, 1.0, None).expect("independent residuals");
        assert!(y[0].abs() < 1e-10, "{y:?}");
        assert!(y[1].abs() < 1e-10, "{y:?}");
        let bare = extrapolated(&xs, &pulay_coefficients(&gs).unwrap());
        assert!((bare[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn gdiis_rejects_a_step_far_past_the_cap() {
        let xs = [array![0.0, 0.0, 0.0], array![0.0, 0.0, 0.0]];
        let gs = [array![-100.0, 0.0, 0.0], array![0.0, -100.0, 0.0]];
        assert!(gdiis_point(&xs, &gs, 1.0, Some(0.2)).is_none());
    }

    #[test]
    fn identical_residuals_have_no_pivot() {
        let gs = [array![1.0, 0.0], array![1.0, 0.0]];
        assert!(pulay_coefficients(&gs).is_none());
    }

    #[test]
    fn quadratic_gradient_falls() {
        let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| (0.5 * x[0] * x[0], array![x[0]]));
        let mut solver = Solver::new(
            Method::diis(),
            Control {
                maxiter: 30,
                gtol: 1e-10,
                istep: 0.4,
                maxmove: None,
                ftol_rel: None,
                noise_floor: None,
            },
            1,
        );
        let mut x = array![1.0];
        let mut norm = f64::INFINITY;
        for _ in 0..30 {
            norm = solver.step(&obj, &mut x).expect("step").grad_norm;
            if norm < 1e-6 {
                break;
            }
        }
        assert!(norm < 1e-6, "grad norm {norm} at {x:?}");
    }

    /// A full steepest step overshoots this stiff bowl. DIIS has to shorten
    /// it; freezing at the start leaves the gradient at 100.
    #[test]
    fn diis_shortens_an_overshooting_steepest_step() {
        let obj = Oracle::unbounded(1, |x: ArrayView1<f64>| {
            (50.0 * x[0] * x[0], array![100.0 * x[0]])
        });
        let mut solver = Solver::new(
            Method::diis(),
            Control {
                maxiter: 40,
                gtol: 1e-8,
                istep: 1.0,
                maxmove: None,
                ftol_rel: None,
                noise_floor: None,
            },
            1,
        );
        let mut x = array![1.0];
        let mut norm = f64::INFINITY;
        for _ in 0..40 {
            norm = solver.step(&obj, &mut x).expect("step").grad_norm;
            if norm < 1e-4 {
                break;
            }
        }
        assert!(norm < 1e-4, "grad norm {norm} at {x:?}");
    }
}
