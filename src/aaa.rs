//! AAA rational approximation and barycentric evaluation.
//!
//! Nakatsukasa, Sète and Trefethen, "The AAA algorithm for rational
//! approximation", SIAM J. Sci. Comput. 40(3) (2018) A1494–A1522,
//! <https://doi.org/10.1137/16M1106122>. Their theorem on rational
//! barycentric representations: distinct support points and nonzero
//! weights give a rational of type `(m-1, m-1)` that interpolates the
//! support values. The weights are the smallest right singular vector of
//! the Loewner matrix on the samples outside the support. A numerical
//! Froissart doublet is a pole with a negligible residue; their cleanup
//! step removes the support point nearest each one and solves for the
//! weights again.
//!
//! The fit and the pole search run on samples mapped to `[-1, 1]` and
//! values divided by a power of two at the peak `|f|`. The weights, the
//! support and the stopping test are therefore the same for `f(x)` and
//! `β f(α x + γ)`, and nothing is squared at the caller's scale.

use crate::error::{Error, Result};

/// Stopping rule and cleanup for [`aaa`].
///
/// The default is the paper's: a residual of `1e-13` relative to the
/// largest `|f|` on the samples, at most 100 support points, and cleanup
/// on.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct AaaParams {
    /// Greedy growth stops when the largest residual on the samples
    /// outside the support is at most `tol` times the largest `|f|`.
    /// Must be positive and finite.
    pub tol: f64,
    /// Largest support the greedy step may build. The effective cap is
    /// `min(max_support, samples.len() - 1)`, so at least one sample is
    /// left to measure the residual. Zero is refused. Exhausting the cap
    /// with the residual still above `tol` returns the fit that was
    /// reached; [`Barycentric::max_residual`] reports it.
    pub max_support: usize,
    /// Remove numerical Froissart doublets after the greedy step.
    pub cleanup: bool,
    /// A pole is a doublet when `|residue| < cleanup_tol · max|f| · h`,
    /// with `h` half the width of the sample interval. This is the paper's
    /// absolute residue test made invariant under scaling of `x` and `f`.
    /// Must be finite and non-negative.
    pub cleanup_tol: f64,
}

impl Default for AaaParams {
    fn default() -> Self {
        Self {
            tol: 1.0e-13,
            max_support: 100,
            cleanup: true,
            cleanup_tol: 1.0e-13,
        }
    }
}

impl AaaParams {
    /// Relative tolerance `tol` and support cap `max_support`, with the
    /// default cleanup.
    pub fn new(tol: f64, max_support: usize) -> Self {
        Self {
            tol,
            max_support,
            ..Self::default()
        }
    }
}

with_setters!(AaaParams {
    with_tol => tol: val f64,
    with_max_support => max_support: val usize,
    with_cleanup => cleanup: val bool,
    with_cleanup_tol => cleanup_tol: val f64,
});

/// One pole of a barycentric rational, with its residue.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct Pole {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
    /// Real part of the residue.
    pub residue_re: f64,
    /// Imaginary part of the residue.
    pub residue_im: f64,
    /// The residue is below the cleanup threshold of [`AaaParams`], or the
    /// pole sits on a support point.
    pub doublet: bool,
}

/// Frozen barycentric quotient `Σ w_j f_j/(x-z_j) / Σ w_j/(x-z_j)`.
///
/// Every weight is finite and nonzero. [`aaa`] removes a support point
/// whose weight comes out zero and solves for the weights again, so each
/// support value is interpolated.
#[derive(Clone, Debug)]
pub struct Barycentric {
    z: Vec<f64>,
    f: Vec<f64>,
    w: Vec<f64>,
    frame: Frame,
    nodes: Vec<f64>,
    scaled: Vec<f64>,
    peak: f64,
    cleanup_tol: f64,
    max_residual: f64,
}

impl Barycentric {
    /// Support points.
    pub fn support(&self) -> &[f64] {
        &self.z
    }

    /// Interpolated values at the support.
    pub fn values(&self) -> &[f64] {
        &self.f
    }

    /// Barycentric weights, with unit 2-norm.
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// Support count.
    pub fn len(&self) -> usize {
        self.z.len()
    }

    /// `true` when the support is empty.
    pub fn is_empty(&self) -> bool {
        self.z.is_empty()
    }

    /// Largest absolute residual on the samples outside the support.
    ///
    /// Measured after cleanup, so it can exceed the requested tolerance
    /// when a doublet was removed.
    pub fn max_residual(&self) -> f64 {
        self.max_residual
    }

    /// Evaluate the frozen quotient. A support point returns its value.
    pub fn eval(&self, x: f64) -> Option<f64> {
        if !x.is_finite() {
            return None;
        }
        if let Some(index) = self.z.iter().position(|&node| node == x) {
            return Some(self.f[index]);
        }
        let value = eval_scaled(self.frame.to_unit(x), &self.nodes, &self.scaled, &self.w)?;
        let value = value * self.frame.value_scale;
        value.is_finite().then_some(value)
    }

    /// Every finite pole, nearest the sample interval first.
    ///
    /// The poles are the zeros of the barycentric denominator, found all
    /// at once by Aberth's iteration. Two zeros closer than
    /// `sqrt(EPSILON)` sample half-widths are one pole: rounding splits a
    /// double zero by about that much. A zero within that distance of the
    /// real axis is a real pole. A zero more than `1/EPSILON` half-widths
    /// from the interval centre is a pole at infinity and is not listed.
    pub fn poles(&self) -> Result<Vec<Pole>> {
        let mut found = unit_poles(&self.nodes, &self.scaled, &self.w)?;
        found.sort_by(|left, right| {
            interval_distance(left.at).total_cmp(&interval_distance(right.at))
        });
        Ok(found
            .into_iter()
            .map(|pole| self.caller_pole(pole))
            .collect())
    }

    /// Poles within one sample-interval width of the samples.
    ///
    /// With samples spanning `[lo, hi]`, a pole is kept when its distance
    /// to that segment is at most `hi - lo`. Real poles are kept. This is
    /// the set used to read a singular-line distance.
    pub fn poles_near_interval(&self) -> Result<Vec<Pole>> {
        let mut found = unit_poles(&self.nodes, &self.scaled, &self.w)?;
        found.retain(|pole| interval_distance(pole.at) <= NEAR_INTERVAL);
        found.sort_by(|left, right| {
            interval_distance(left.at).total_cmp(&interval_distance(right.at))
        });
        Ok(found
            .into_iter()
            .map(|pole| self.caller_pole(pole))
            .collect())
    }

    fn caller_pole(&self, pole: UnitPole) -> Pole {
        let doublet = is_doublet(&pole, &self.nodes, self.peak, self.cleanup_tol);
        let residue_scale = self.frame.half * self.frame.value_scale;
        Pole {
            re: self.frame.center + self.frame.half * pole.at.re,
            im: self.frame.half * pole.at.im,
            residue_re: pole.residue.re * residue_scale,
            residue_im: pole.residue.im * residue_scale,
            doublet,
        }
    }
}

/// Distance from `[-1, 1]`, in sample half-widths, within which
/// [`Barycentric::poles_near_interval`] keeps a pole: one interval width.
const NEAR_INTERVAL: f64 = 2.0;

/// Greedy AAA fit of real samples, followed by the doublet cleanup.
///
/// Refuses mismatched or non-finite input, fewer than two samples,
/// samples that coincide at the resolution of their interval, and
/// parameters outside the ranges [`AaaParams`] documents.
pub fn aaa(samples: &[f64], values: &[f64], params: AaaParams) -> Result<Barycentric> {
    if samples.len() != values.len() || samples.len() < 2 {
        return Err(Error::Aaa {
            what: "AAA needs at least two matching samples",
        });
    }
    if !(params.tol > 0.0 && params.tol.is_finite()) || params.max_support == 0 {
        return Err(Error::Aaa {
            what: "AAA tolerance or support cap is not positive",
        });
    }
    if !(params.cleanup_tol >= 0.0 && params.cleanup_tol.is_finite()) {
        return Err(Error::Aaa {
            what: "AAA cleanup tolerance is negative or not finite",
        });
    }
    if samples.iter().chain(values).any(|value| !value.is_finite()) {
        return Err(Error::Aaa {
            what: "AAA samples must be finite",
        });
    }
    let frame = Frame::of(samples, values)?;
    let t: Vec<f64> = samples.iter().map(|&x| frame.to_unit(x)).collect();
    let g: Vec<f64> = values.iter().map(|&f| f / frame.value_scale).collect();
    let mut sorted = t.clone();
    sorted.sort_by(f64::total_cmp);
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Error::Aaa {
            what: "AAA sample points are not distinct",
        });
    }
    let peak = g.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let target = params.tol * peak;
    let cap = params.max_support.min(samples.len() - 1);

    let mut first = 0usize;
    for index in 1..g.len() {
        if g[index].abs() > g[first].abs() {
            first = index;
        }
    }
    let mut remain = vec![true; samples.len()];
    let mut support = vec![first];
    remain[first] = false;
    let mut weights = vec![1.0];
    loop {
        let (worst, worst_at) = worst_residual(&t, &g, &support, &weights, &remain)?;
        let Some(index) = worst_at else {
            break;
        };
        if worst <= target || support.len() >= cap {
            break;
        }
        support.push(index);
        remain[index] = false;
        weights = loewner_weights(&t, &g, &support, &remain)?;
    }
    prune(
        &t,
        &g,
        &mut support,
        &mut weights,
        &mut remain,
        peak,
        params,
    )?;
    let (worst, _) = worst_residual(&t, &g, &support, &weights, &remain)?;
    Ok(Barycentric {
        z: support.iter().map(|&index| samples[index]).collect(),
        f: support.iter().map(|&index| values[index]).collect(),
        w: weights,
        frame,
        nodes: support.iter().map(|&index| t[index]).collect(),
        scaled: support.iter().map(|&index| g[index]).collect(),
        peak,
        cleanup_tol: params.cleanup_tol,
        max_residual: worst * frame.value_scale,
    })
}

/// Affine map of the samples onto `[-1, 1]` and a power-of-two value scale.
#[derive(Clone, Copy, Debug)]
struct Frame {
    center: f64,
    half: f64,
    value_scale: f64,
}

impl Frame {
    fn of(samples: &[f64], values: &[f64]) -> Result<Self> {
        let mut lo = samples[0];
        let mut hi = samples[0];
        for &sample in samples {
            lo = lo.min(sample);
            hi = hi.max(sample);
        }
        // Halving first keeps both finite for any pair of finite ends.
        let center = 0.5 * lo + 0.5 * hi;
        let half = 0.5 * hi - 0.5 * lo;
        if !(half > 0.0 && half.is_finite()) {
            return Err(Error::Aaa {
                what: "AAA sample points are not distinct",
            });
        }
        let peak = values
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        // `2^(e-1) <= peak < 2^e`: the division is exact and the scaled
        // peak is in `[1, 2)`, also at the top of the range.
        let value_scale = if peak > 0.0 {
            let (_, exponent) = libm::frexp(peak);
            libm::ldexp(1.0, exponent - 1)
        } else {
            1.0
        };
        Ok(Self {
            center,
            half,
            value_scale,
        })
    }

    fn to_unit(self, x: f64) -> f64 {
        (x - self.center) / self.half
    }
}

/// Remove zero weights and, with cleanup on, the support point nearest
/// each doublet, then solve for the weights again. Every pass drops at
/// least one support point, so the loop ends.
fn prune(
    t: &[f64],
    g: &[f64],
    support: &mut Vec<usize>,
    weights: &mut Vec<f64>,
    remain: &mut [bool],
    peak: f64,
    params: AaaParams,
) -> Result<()> {
    loop {
        if support.len() < 2 {
            *weights = vec![1.0];
            return Ok(());
        }
        let mut drop: Vec<usize> = weights
            .iter()
            .enumerate()
            .filter(|(_, weight)| **weight == 0.0 || !weight.is_finite())
            .map(|(position, _)| position)
            .collect();
        if drop.is_empty() && params.cleanup {
            let nodes: Vec<f64> = support.iter().map(|&index| t[index]).collect();
            let scaled: Vec<f64> = support.iter().map(|&index| g[index]).collect();
            for pole in unit_poles(&nodes, &scaled, weights)? {
                if is_doublet(&pole, &nodes, peak, params.cleanup_tol) {
                    drop.push(nearest_node(pole.at, &nodes));
                }
            }
        }
        if drop.is_empty() {
            return Ok(());
        }
        drop.sort_unstable();
        drop.dedup();
        if drop.len() == support.len() {
            drop.remove(0);
        }
        for &position in drop.iter().rev() {
            remain[support[position]] = true;
            support.remove(position);
        }
        *weights = if support.len() > 1 {
            loewner_weights(t, g, support, remain)?
        } else {
            vec![1.0]
        };
    }
}

fn nearest_node(at: C, nodes: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_gap = f64::INFINITY;
    for (position, &node) in nodes.iter().enumerate() {
        let gap = (at.re - node).hypot(at.im);
        if gap < best_gap {
            best_gap = gap;
            best = position;
        }
    }
    best
}

/// Largest `|g - r|` over the remaining samples, and where it occurs.
fn worst_residual(
    t: &[f64],
    g: &[f64],
    support: &[usize],
    weights: &[f64],
    remain: &[bool],
) -> Result<(f64, Option<usize>)> {
    let nodes: Vec<f64> = support.iter().map(|&index| t[index]).collect();
    let scaled: Vec<f64> = support.iter().map(|&index| g[index]).collect();
    let mut worst = 0.0_f64;
    let mut worst_at = None;
    for (index, kept) in remain.iter().copied().enumerate() {
        if !kept {
            continue;
        }
        let Some(predicted) = eval_scaled(t[index], &nodes, &scaled, weights) else {
            return Err(Error::Aaa {
                what: "the barycentric quotient was not finite",
            });
        };
        let gap = (g[index] - predicted).abs();
        if worst_at.is_none() || gap > worst {
            worst = gap;
            worst_at = Some(index);
        }
    }
    Ok((worst, worst_at))
}

fn eval_scaled(t: f64, nodes: &[f64], scaled: &[f64], weights: &[f64]) -> Option<f64> {
    if !t.is_finite() || nodes.is_empty() {
        return None;
    }
    if let Some(index) = nodes.iter().position(|&node| node == t) {
        return Some(scaled[index]);
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for ((&node, &value), &weight) in nodes.iter().zip(scaled).zip(weights) {
        let cauchy = weight / (t - node);
        num += cauchy * value;
        den += cauchy;
    }
    let value = num / den;
    value.is_finite().then_some(value)
}

fn loewner_weights(t: &[f64], g: &[f64], support: &[usize], remain: &[bool]) -> Result<Vec<f64>> {
    let cols = support.len();
    let rows: Vec<usize> = remain
        .iter()
        .enumerate()
        .filter_map(|(index, kept)| kept.then_some(index))
        .collect();
    if rows.is_empty() || cols == 0 {
        return Err(Error::Aaa {
            what: "the Loewner matrix is empty",
        });
    }
    let mut matrix = vec![0.0; rows.len() * cols];
    for (row, &sample) in rows.iter().enumerate() {
        for (col, &node) in support.iter().enumerate() {
            let gap = t[sample] - t[node];
            if gap == 0.0 {
                return Err(Error::Aaa {
                    what: "a Loewner entry hit a repeated node",
                });
            }
            matrix[row * cols + col] = (g[sample] - g[node]) / gap;
        }
    }
    let mut weights = smallest_right_singular(&matrix, rows.len(), cols)?;
    let mut pivot = 0usize;
    for index in 1..weights.len() {
        if weights[index].abs() > weights[pivot].abs() {
            pivot = index;
        }
    }
    if weights[pivot] < 0.0 {
        for weight in &mut weights {
            *weight = -*weight;
        }
    }
    Ok(weights)
}

/// Cyclic sweeps allowed before the Jacobi iteration is refused.
///
/// Cyclic Jacobi converges quadratically once the columns are nearly
/// orthogonal; a sweep count past this means the stop is unreachable.
const JACOBI_SWEEPS: usize = 64;

/// Unit right singular vector for the smallest singular value.
///
/// The matrix is divided by its largest entry, reduced to its `R` factor
/// by Householder reflections, and `R` is orthogonalized by one-sided
/// Jacobi. A pair `(p, q)` is resolved when
/// `|r_p · r_q| <= n EPSILON ‖r_p‖ ‖r_q‖`, the rounding bound of an
/// `n`-term inner product, so a sweep with no rotation is reachable.
fn smallest_right_singular(matrix: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>> {
    if rows == 0 || cols == 0 || matrix.len() != rows * cols {
        return Err(Error::Aaa {
            what: "the singular-vector matrix has the wrong shape",
        });
    }
    let mut peak = 0.0_f64;
    for value in matrix {
        if !value.is_finite() {
            return Err(Error::Aaa {
                what: "the Loewner matrix is not finite",
            });
        }
        peak = peak.max(value.abs());
    }
    if peak == 0.0 {
        return Err(Error::Aaa {
            what: "the Loewner matrix has zero scale",
        });
    }
    if cols == 1 {
        return Ok(vec![1.0]);
    }
    let scaled: Vec<f64> = matrix.iter().map(|value| value / peak).collect();
    let mut r = householder_r(&scaled, rows, cols)?;
    let n = cols;
    let mut vectors = vec![0.0; n * n];
    for index in 0..n {
        vectors[index * n + index] = 1.0;
    }
    let mass = frobenius_norm(&r)?;
    // A column at EPSILON of the matrix norm is roundoff, including a
    // numerical null column, and is not rotated against.
    let floor = (f64::EPSILON * mass) * (f64::EPSILON * mass);
    let tol = n as f64 * f64::EPSILON;
    let mut converged = false;
    for _ in 0..JACOBI_SWEEPS {
        let mut rotated = false;
        for left in 0..n {
            for right in (left + 1)..n {
                let mut aa = 0.0;
                let mut bb = 0.0;
                let mut cc = 0.0;
                for row in 0..n {
                    let lv = r[row * n + left];
                    let rv = r[row * n + right];
                    aa += lv * lv;
                    bb += rv * rv;
                    cc += lv * rv;
                }
                if aa <= floor || bb <= floor || cc.abs() <= tol * aa.sqrt() * bb.sqrt() {
                    continue;
                }
                let zeta = (bb - aa) / (2.0 * cc);
                let t = zeta.signum() / (zeta.abs() + zeta.hypot(1.0));
                let cosine = 1.0 / t.hypot(1.0);
                let sine = cosine * t;
                if !(cosine.is_finite() && sine.is_finite()) {
                    return Err(Error::Aaa {
                        what: "the Loewner matrix is not finite",
                    });
                }
                rotate(&mut r, n, left, right, cosine, sine);
                rotate(&mut vectors, n, left, right, cosine, sine);
                rotated = true;
            }
        }
        if !rotated {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(Error::Aaa {
            what: "the Jacobi sweep did not converge",
        });
    }
    let mut best = 0usize;
    let mut best_norm = f64::INFINITY;
    for col in 0..n {
        let column: Vec<f64> = (0..n).map(|row| r[row * n + col]).collect();
        let norm = frobenius_norm(&column)?;
        if norm < best_norm {
            best_norm = norm;
            best = col;
        }
    }
    let mut out: Vec<f64> = (0..n).map(|row| vectors[row * n + best]).collect();
    let norm = frobenius_norm(&out)?;
    if !(norm > 0.0 && norm.is_finite()) {
        return Err(Error::Aaa {
            what: "the Loewner singular vector vanished",
        });
    }
    for value in &mut out {
        *value /= norm;
    }
    Ok(out)
}

fn rotate(columns: &mut [f64], n: usize, left: usize, right: usize, cosine: f64, sine: f64) {
    let rows = columns.len() / n;
    for row in 0..rows {
        let lv = columns[row * n + left];
        let rv = columns[row * n + right];
        columns[row * n + left] = lv * cosine - rv * sine;
        columns[row * n + right] = lv * sine + rv * cosine;
    }
}

/// The `cols × cols` triangular factor of a row-major `rows × cols`
/// matrix, padded with zero rows when `rows < cols`.
fn householder_r(matrix: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>> {
    let mut a = matrix.to_vec();
    let steps = rows.min(cols);
    let mut reflector = vec![0.0; rows];
    for col in 0..steps {
        let length = rows - col;
        for row in col..rows {
            reflector[row - col] = a[row * cols + col];
        }
        let norm = frobenius_norm(&reflector[..length])?;
        if norm == 0.0 {
            continue;
        }
        let alpha = if reflector[0] < 0.0 { norm } else { -norm };
        reflector[0] -= alpha;
        let vnorm = frobenius_norm(&reflector[..length])?;
        if vnorm == 0.0 {
            continue;
        }
        for value in &mut reflector[..length] {
            *value /= vnorm;
        }
        for target in col..cols {
            let mut dot = 0.0;
            for row in col..rows {
                dot += reflector[row - col] * a[row * cols + target];
            }
            for row in col..rows {
                a[row * cols + target] -= 2.0 * reflector[row - col] * dot;
            }
        }
        a[col * cols + col] = alpha;
        for row in (col + 1)..rows {
            a[row * cols + col] = 0.0;
        }
    }
    let mut r = vec![0.0; cols * cols];
    for row in 0..steps {
        for col in row..cols {
            r[row * cols + col] = a[row * cols + col];
        }
    }
    Ok(r)
}

/// Euclidean norm with a running scale, as LAPACK's `dnrm2` and `dlassq`
/// compute it. No square is formed at the entries' own magnitude, so it
/// neither overflows nor underflows for finite input whose norm is finite.
fn frobenius_norm(values: &[f64]) -> Result<f64> {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for &value in values {
        if !value.is_finite() {
            return Err(Error::Aaa {
                what: "the Loewner matrix is not finite",
            });
        }
        if value == 0.0 {
            continue;
        }
        let size = value.abs();
        if scale < size {
            let ratio = scale / size;
            ssq = 1.0 + ssq * ratio * ratio;
            scale = size;
        } else {
            let ratio = size / scale;
            ssq += ratio * ratio;
        }
    }
    let norm = scale * ssq.sqrt();
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(Error::Aaa {
            what: "the Loewner matrix is not finite",
        })
    }
}

/// A pole in the unit frame.
#[derive(Clone, Copy, Debug)]
struct UnitPole {
    at: C,
    residue: C,
}

/// `|residue| < cleanup_tol · peak` in the unit frame, where the sample
/// half-width is one, or the pole is on a support point.
fn is_doublet(pole: &UnitPole, nodes: &[f64], peak: f64, cleanup_tol: f64) -> bool {
    let on_support = nodes
        .iter()
        .any(|&node| (pole.at.re - node).hypot(pole.at.im) <= f64::EPSILON * node.abs().max(1.0));
    on_support || pole.residue.abs() < cleanup_tol * peak
}

/// Distance from `[-1, 1]`.
fn interval_distance(at: C) -> f64 {
    (at.re.abs() - 1.0).max(0.0).hypot(at.im)
}

/// Zeros of `d(t) = Σ w_j/(t - t_j)` with residues `n/d'`, by Aberth's
/// simultaneous iteration on `q = d ∏ (t - t_j)`, a polynomial of degree
/// at most `m - 1`.
fn unit_poles(nodes: &[f64], scaled: &[f64], weights: &[f64]) -> Result<Vec<UnitPole>> {
    let m = nodes.len();
    if m < 2 {
        return Ok(Vec::new());
    }
    if weights.len() != m || weights.iter().any(|w| *w == 0.0 || !w.is_finite()) {
        return Err(Error::Aaa {
            what: "a barycentric weight is zero or not finite",
        });
    }
    let mut spacing = 2.0_f64;
    let mut sorted = nodes.to_vec();
    sorted.sort_by(f64::total_cmp);
    for pair in sorted.windows(2) {
        spacing = spacing.min(pair[1] - pair[0]);
    }
    let degree = m - 1;
    // Start on the unit circle, turned off the real axis so that a real
    // zero and a conjugate pair are not given mirrored starts.
    let mut roots: Vec<C> = (0..degree)
        .map(|k| {
            let angle = std::f64::consts::TAU * (k as f64 + 0.25) / degree as f64;
            C::new(angle.cos(), angle.sin())
        })
        .collect();
    let mut done = vec![false; degree];
    let mut at_infinity = vec![false; degree];
    // Aberth converges cubically from these starts; the budget is a
    // generous multiple of the degree.
    let budget = 100 + 10 * degree;
    for _ in 0..budget {
        if done.iter().all(|&flag| flag) {
            break;
        }
        for k in 0..degree {
            if done[k] {
                continue;
            }
            let z = roots[k];
            if z.abs() > 1.0 / f64::EPSILON {
                at_infinity[k] = true;
                done[k] = true;
                continue;
            }
            let Some(parts) = denominator_parts(z, nodes, weights) else {
                roots[k] = z + C::new(0.0, f64::EPSILON * spacing);
                continue;
            };
            if parts.den.abs() <= f64::EPSILON * parts.magnitude {
                done[k] = true;
                continue;
            }
            let log_derivative = parts.der / parts.den + parts.node_sum;
            if log_derivative.abs() == 0.0 {
                continue;
            }
            let newton = C::real(1.0) / log_derivative;
            let mut repel = C::new(0.0, 0.0);
            for (other, &root) in roots.iter().enumerate() {
                if other != k {
                    repel = repel + C::real(1.0) / (z - root);
                }
            }
            let step = newton / (C::real(1.0) - newton * repel);
            if !(step.re.is_finite() && step.im.is_finite()) {
                return Err(Error::Aaa {
                    what: "the pole iteration was not finite",
                });
            }
            roots[k] = z - step;
            if step.abs() <= f64::EPSILON * roots[k].abs().max(spacing) {
                done[k] = true;
            }
        }
    }
    if !done.iter().all(|&flag| flag) {
        return Err(Error::Aaa {
            what: "the pole iteration did not converge",
        });
    }
    let resolution = f64::EPSILON.sqrt();
    let mut found: Vec<UnitPole> = Vec::with_capacity(degree);
    for (k, mut z) in roots.into_iter().enumerate() {
        if at_infinity[k] {
            continue;
        }
        let local = resolution * z.abs().max(1.0);
        if z.im.abs() <= local {
            z = C::real(z.re);
        }
        if found.iter().any(|kept| (kept.at - z).abs() <= local) {
            continue;
        }
        let Some(parts) = denominator_parts(z, nodes, weights) else {
            continue;
        };
        let mut num = C::new(0.0, 0.0);
        for ((&node, &value), &weight) in nodes.iter().zip(scaled).zip(weights) {
            num = num + C::real(weight * value) / (z - C::real(node));
        }
        let residue = num / parts.der;
        if !(residue.re.is_finite() && residue.im.is_finite()) {
            continue;
        }
        found.push(UnitPole { at: z, residue });
    }
    Ok(found)
}

struct DenominatorParts {
    den: C,
    der: C,
    node_sum: C,
    magnitude: f64,
}

/// `d`, `d'`, `Σ 1/(z - t_j)` and `Σ |w_j/(z - t_j)|` at `z`, or `None`
/// on a node.
fn denominator_parts(z: C, nodes: &[f64], weights: &[f64]) -> Option<DenominatorParts> {
    let mut den = C::new(0.0, 0.0);
    let mut der = C::new(0.0, 0.0);
    let mut node_sum = C::new(0.0, 0.0);
    let mut magnitude = 0.0;
    for (&node, &weight) in nodes.iter().zip(weights) {
        let gap = z - C::real(node);
        if gap.abs() == 0.0 {
            return None;
        }
        let inv = C::real(1.0) / gap;
        den = den + C::real(weight) * inv;
        der = der - C::real(weight) * inv * inv;
        node_sum = node_sum + inv;
        magnitude += weight.abs() * inv.abs();
    }
    Some(DenominatorParts {
        den,
        der,
        node_sum,
        magnitude,
    })
}

/// Poles of a type-`[numerator_degree, 2]` least-squares Chebyshev–Padé fit.
///
/// The denominator is `1 + a u + b u²` on samples in `[-1, 1]`. This is the
/// degree pattern Tee and Trefethen use to place a singular line. It is a
/// proposal, not an enclosure.
pub fn chebyshev_pade_poles(
    samples: &[f64],
    values: &[f64],
    numerator_degree: usize,
) -> Result<Vec<Pole>> {
    if samples.len() != values.len() || samples.is_empty() {
        return Err(Error::Aaa {
            what: "Padé samples do not match",
        });
    }
    let cols = numerator_degree.checked_add(3).ok_or(Error::Aaa {
        what: "Padé degree overflowed",
    })?;
    if samples.len() < cols {
        return Err(Error::Aaa {
            what: "Padé needs more samples than unknowns",
        });
    }
    let mut matrix = vec![0.0; samples.len() * cols];
    for (row, (&sample, &value)) in samples.iter().zip(values).enumerate() {
        if !sample.is_finite() || !value.is_finite() || outside_unit_interval(sample) {
            return Err(Error::Aaa {
                what: "Padé samples must lie in [-1, 1]",
            });
        }
        let mut t_prev = 1.0;
        let mut t_curr = sample;
        matrix[row * cols] = t_prev;
        if numerator_degree >= 1 {
            matrix[row * cols + 1] = t_curr;
        }
        for degree in 2..=numerator_degree {
            let next = 2.0 * sample * t_curr - t_prev;
            matrix[row * cols + degree] = next;
            t_prev = t_curr;
            t_curr = next;
        }
        matrix[row * cols + numerator_degree + 1] = -value * sample;
        matrix[row * cols + numerator_degree + 2] = -value * sample * sample;
    }
    let coef = least_squares(&matrix, samples.len(), cols, values)?;
    let linear = coef[numerator_degree + 1];
    let quadratic = coef[numerator_degree + 2];
    let mut roots = quadratic_roots(quadratic, linear, 1.0);
    roots.retain(|root| root.re.is_finite() && root.im.is_finite());
    let mut poles = Vec::with_capacity(roots.len());
    for root in roots {
        let (residue_re, residue_im) =
            pade_residue(&coef, numerator_degree, linear, quadratic, root);
        let residue = residue_re.hypot(residue_im);
        let scale = sample_scale(values)?;
        poles.push(Pole {
            re: root.re,
            im: root.im,
            residue_re,
            residue_im,
            doublet: residue <= f64::EPSILON * scale,
        });
    }
    Ok(poles)
}

#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl std::ops::Add for C {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl std::ops::Sub for C {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl std::ops::Mul for C {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl std::ops::Div for C {
    type Output = Self;
    /// Smith's division: the larger component of the divisor is factored
    /// out, so no square of it is formed.
    fn div(self, other: Self) -> Self {
        if other.re.abs() >= other.im.abs() {
            let ratio = other.im / other.re;
            let den = other.re + other.im * ratio;
            Self {
                re: (self.re + self.im * ratio) / den,
                im: (self.im - self.re * ratio) / den,
            }
        } else {
            let ratio = other.re / other.im;
            let den = other.re * ratio + other.im;
            Self {
                re: (self.re * ratio + self.im) / den,
                im: (self.im * ratio - self.re) / den,
            }
        }
    }
}

fn positive_peak(values: &[f64]) -> Result<f64> {
    let mut peak = 0.0_f64;
    for value in values {
        if !value.is_finite() {
            return Err(Error::Aaa {
                what: "a value is not finite",
            });
        }
        peak = peak.max(value.abs());
    }
    if peak == 0.0 {
        return Err(Error::Aaa {
            what: "the values have zero scale",
        });
    }
    Ok(peak)
}

fn sample_scale(values: &[f64]) -> Result<f64> {
    positive_peak(values)
}

fn outside_unit_interval(sample: f64) -> bool {
    let past = sample.abs() - 1.0;
    past > 0.0 && past > f64::EPSILON * sample.abs()
}

fn quadratic_roots(a: f64, b: f64, c: f64) -> Vec<C> {
    // `a u² + b u + c`.
    let lower = b.abs().max(c.abs());
    if lower == 0.0 {
        return if a == 0.0 {
            Vec::new()
        } else {
            vec![C::new(0.0, 0.0)]
        };
    }
    if a.abs() <= f64::EPSILON * lower {
        if b.abs() <= f64::EPSILON * c.abs() {
            return Vec::new();
        }
        return vec![C::real(-c / b)];
    }
    let disc = C::real(b * b - 4.0 * a * c);
    let root = sqrt_c(disc);
    let two_a = C::real(2.0 * a);
    vec![(C::real(-b) + root) / two_a, (C::real(-b) - root) / two_a]
}

fn sqrt_c(z: C) -> C {
    if z.re == 0.0 && z.im == 0.0 {
        return C::new(0.0, 0.0);
    }
    let mag = z.abs();
    if z.re >= 0.0 {
        let re = ((mag + z.re) * 0.5).sqrt();
        if re == 0.0 {
            return C::new(0.0, 0.0);
        }
        C::new(re, z.im / (2.0 * re))
    } else {
        let im = ((mag - z.re) * 0.5).sqrt().copysign(z.im);
        if im == 0.0 {
            return C::new(0.0, 0.0);
        }
        C::new(z.im / (2.0 * im), im)
    }
}

fn chebyshev_complex(coef: &[f64], degree: usize, z: C) -> C {
    let mut sum = C::real(coef[0]);
    if degree == 0 {
        return sum;
    }
    sum = sum + z * C::real(coef[1]);
    let mut previous = C::real(1.0);
    let mut current = z;
    for coef_k in coef.iter().take(degree + 1).skip(2) {
        let next = C::real(2.0) * z * current - previous;
        sum = sum + next * C::real(*coef_k);
        previous = current;
        current = next;
    }
    sum
}

fn pade_residue(coef: &[f64], degree: usize, linear: f64, quadratic: f64, root: C) -> (f64, f64) {
    let numerator = chebyshev_complex(coef, degree, root);
    let derivative = C::real(2.0 * quadratic) * root + C::real(linear);
    if derivative.abs() == 0.0 {
        return (0.0, 0.0);
    }
    let residue = numerator / derivative;
    (residue.re, residue.im)
}

fn least_squares(matrix: &[f64], rows: usize, cols: usize, rhs: &[f64]) -> Result<Vec<f64>> {
    if rows < cols || matrix.len() != rows * cols || rhs.len() != rows {
        return Err(Error::Aaa {
            what: "the Padé least squares shape is wrong",
        });
    }
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    for col in 0..cols {
        let mut norm = 0.0;
        for row in col..rows {
            let value = a[row * cols + col];
            norm += value * value;
        }
        let norm = norm.sqrt();
        if !(norm > 0.0 && norm.is_finite()) {
            return Err(Error::Aaa {
                what: "the Padé matrix is rank deficient",
            });
        }
        let pivot = a[col * cols + col];
        let sign = if pivot < 0.0 { -1.0 } else { 1.0 };
        let mut reflector = vec![0.0; rows - col];
        reflector[0] = pivot + sign * norm;
        for row in (col + 1)..rows {
            reflector[row - col] = a[row * cols + col];
        }
        let mut reflector_norm = 0.0;
        for value in &reflector {
            reflector_norm += value * value;
        }
        let reflector_norm = reflector_norm.sqrt();
        if reflector_norm == 0.0 {
            return Err(Error::Aaa {
                what: "a Padé reflector vanished",
            });
        }
        for value in &mut reflector {
            *value /= reflector_norm;
        }
        for target in col..cols {
            let mut dot = 0.0;
            for row in col..rows {
                dot += reflector[row - col] * a[row * cols + target];
            }
            for row in col..rows {
                a[row * cols + target] -= 2.0 * reflector[row - col] * dot;
            }
        }
        let mut dot = 0.0;
        for row in col..rows {
            dot += reflector[row - col] * b[row];
        }
        for row in col..rows {
            b[row] -= 2.0 * reflector[row - col] * dot;
        }
    }
    let mut solution = vec![0.0; cols];
    for col in (0..cols).rev() {
        let mut acc = b[col];
        for other in (col + 1)..cols {
            acc -= a[col * cols + other] * solution[other];
        }
        let pivot = a[col * cols + col];
        let peak = positive_peak(&a)?;
        if pivot.abs() <= f64::EPSILON * peak {
            return Err(Error::Aaa {
                what: "the Padé factor is singular",
            });
        }
        solution[col] = acc / pivot;
        if !solution[col].is_finite() {
            return Err(Error::Aaa {
                what: "the Padé coefficient is not finite",
            });
        }
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chebyshev(count: usize) -> Vec<f64> {
        (0..count)
            .map(|index| ((index as f64 + 0.5) * std::f64::consts::PI / count as f64).cos())
            .collect()
    }

    fn fit(count: usize, f: impl Fn(f64) -> f64, params: AaaParams) -> Barycentric {
        let nodes = chebyshev(count);
        let values: Vec<f64> = nodes.iter().map(|&node| f(node)).collect();
        let fit = aaa(&nodes, &values, params).expect("aaa");
        assert!(
            fit.weights()
                .iter()
                .all(|weight| *weight != 0.0 && weight.is_finite()),
            "weights {:?}",
            fit.weights()
        );
        fit
    }

    fn grid_error(fit: &Barycentric, f: impl Fn(f64) -> f64) -> f64 {
        let mut worst = 0.0_f64;
        for index in 0..=4000 {
            let x = -1.0 + 2.0 * index as f64 / 4000.0;
            let got = fit.eval(x).expect("finite on the grid");
            worst = worst.max((got - f(x)).abs());
        }
        worst
    }

    fn real_poles_inside(poles: &[Pole]) -> usize {
        poles
            .iter()
            .filter(|pole| pole.im == 0.0 && pole.re.abs() <= 1.0)
            .count()
    }

    #[test]
    fn smallest_singular_vector_matches_a_dense_svd() {
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weights = smallest_right_singular(&matrix, 4, 2).expect("svd");
        let expect = [0.767187395_f64, -0.641423027];
        let sign = weights[0].signum();
        let mut gap = 0.0_f64;
        for (got, want) in weights.iter().zip(&expect) {
            gap = gap.max((sign * got - want).abs());
        }
        assert!(gap < 1.0e-8, "singular vector gap {gap}");
    }

    #[test]
    fn zero_scale_singular_vector_is_refused() {
        let matrix = [0.0; 8];
        let error = smallest_right_singular(&matrix, 4, 2).expect_err("zero scale");
        match error {
            Error::Aaa { what } => assert_eq!(what, "the Loewner matrix has zero scale"),
            other => panic!("unexpected error {other}"),
        }
    }

    /// Smallest right singular vector from the closed-form eigenvalues
    /// of the 3×3 Gram matrix. This is not a Jacobi sweep.
    fn oracle_smallest_right(matrix: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        assert_eq!(cols, 3, "the oracle is the 3×3 characteristic polynomial");
        let peak = matrix
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let matrix: Vec<f64> = matrix.iter().map(|value| value / peak).collect();
        let mut gram = [0.0; 9];
        for left in 0..cols {
            for right in 0..cols {
                let mut dot = 0.0;
                for row in 0..rows {
                    dot += matrix[row * cols + left] * matrix[row * cols + right];
                }
                gram[left * cols + right] = dot;
            }
        }
        let (a, b, c) = (gram[0], gram[1], gram[2]);
        let (d, e, f) = (gram[4], gram[5], gram[8]);
        let q = (a + d + f) / 3.0;
        let (b00, b11, b22) = (a - q, d - q, f - q);
        let p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * (b * b + c * c + e * e);
        let p = (p2 / 6.0).sqrt();
        assert!(
            p > 0.0 && p.is_finite(),
            "the Gram matrix has no distinct scale"
        );
        let inv = 1.0 / p;
        let (m00, m01, m02) = (b00 * inv, b * inv, c * inv);
        let (m11, m12, m22) = (b11 * inv, e * inv, b22 * inv);
        let det = m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02)
            + m02 * (m01 * m12 - m11 * m02);
        // A repeated eigenvalue puts `det / 2` at ±1, and rounding can
        // carry it past.
        let half = (det * 0.5).clamp(-1.0, 1.0);
        let phi = half.acos() / 3.0;
        let turn = 2.0 * std::f64::consts::PI / 3.0;
        let mut lambda = q + 2.0 * p * phi.cos();
        for offset in [turn, 2.0 * turn] {
            let eig = q + 2.0 * p * (phi + offset).cos();
            if eig < lambda {
                lambda = eig;
            }
        }
        let rows_m = [[a - lambda, b, c], [b, d - lambda, e], [c, e, f - lambda]];
        let mut best = [0.0; 3];
        let mut best_norm = 0.0;
        for left in 0..3 {
            for right in (left + 1)..3 {
                let cross = [
                    rows_m[left][1] * rows_m[right][2] - rows_m[left][2] * rows_m[right][1],
                    rows_m[left][2] * rows_m[right][0] - rows_m[left][0] * rows_m[right][2],
                    rows_m[left][0] * rows_m[right][1] - rows_m[left][1] * rows_m[right][0],
                ];
                let norm: f64 = cross.iter().map(|value| value * value).sum();
                if norm > best_norm {
                    best_norm = norm;
                    best = cross;
                }
            }
        }
        let norm = best_norm.sqrt();
        assert!(
            norm > 0.0 && norm.is_finite(),
            "the smallest eigenvector vanished"
        );
        best.iter().map(|value| value / norm).collect()
    }

    fn direction_gap(left: &[f64], right: &[f64]) -> f64 {
        let dot: f64 = left.iter().zip(right).map(|(lhs, rhs)| lhs * rhs).sum();
        1.0 - dot.abs()
    }

    #[test]
    fn singular_vector_matches_an_oracle_at_tiny_and_huge_scale() {
        // Three coupled columns: one Jacobi sweep cannot finish them.
        let base = [
            1.0, 0.2, -0.4, 0.3, 1.1, 0.5, -0.7, 0.4, 1.2, 0.8, -0.6, 0.3, 0.2, 0.9, -0.8,
        ];
        let (rows, cols) = (5usize, 3usize);
        for scale in [1.0e-200, 1.0e-18, 1.0, 1.0e18, 1.0e200] {
            let scaled: Vec<f64> = base.iter().map(|value| value * scale).collect();
            let got = smallest_right_singular(&scaled, rows, cols).expect("svd");
            let oracle = oracle_smallest_right(&scaled, rows, cols);
            let gap = direction_gap(&got, &oracle);
            assert!(
                gap <= (cols as f64) * f64::EPSILON.sqrt(),
                "scale {scale} gap {gap} got {got:?} oracle {oracle:?}"
            );
        }
    }

    #[test]
    fn singular_vector_matches_an_oracle_with_a_double_eigenvalue() {
        // `diag(1, 2, 2) Qᵀ` with `Q` a Householder reflector: the Gram
        // eigenvalues are 1, 4, 4 and the smallest vector is `Q e₁`.
        let v = [1.0_f64, 2.0, 2.0];
        let vv: f64 = v.iter().map(|value| value * value).sum();
        let mut q = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                let identity = if row == col { 1.0 } else { 0.0 };
                q[row * 3 + col] = identity - 2.0 * v[row] * v[col] / vv;
            }
        }
        let sigma = [1.0, 2.0, 2.0];
        let mut matrix = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                matrix[row * 3 + col] = sigma[row] * q[col * 3 + row];
            }
        }
        let want = [q[0], q[3], q[6]];
        let got = smallest_right_singular(&matrix, 3, 3).expect("svd");
        let oracle = oracle_smallest_right(&matrix, 3, 3);
        assert!(direction_gap(&got, &want) <= 3.0 * f64::EPSILON.sqrt());
        assert!(direction_gap(&oracle, &want) <= 3.0 * f64::EPSILON.sqrt());
    }

    #[test]
    fn jacobi_reaches_its_stop_on_small_and_wide_shapes() {
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for trial in 0..3000 {
            let cols = 2 + trial % 7;
            let rows = 1 + (trial / 7) % 12;
            let matrix: Vec<f64> = (0..rows * cols).map(|_| next()).collect();
            let got = smallest_right_singular(&matrix, rows, cols).expect("jacobi stop");
            let norm = frobenius_norm(&got).expect("norm");
            assert!((norm - 1.0).abs() <= 4.0 * f64::EPSILON, "norm {norm}");
        }
    }

    #[test]
    fn frobenius_norm_neither_overflows_nor_underflows() {
        for scale in [1.0e-300, 1.0e-200, 1.0, 1.0e200, 1.0e300] {
            let norm = frobenius_norm(&[3.0 * scale, 0.0, -4.0 * scale]).expect("norm");
            assert!(
                (norm / scale - 5.0).abs() <= 4.0 * f64::EPSILON,
                "scale {scale} norm {norm}"
            );
        }
        assert!(frobenius_norm(&[f64::NAN]).is_err());
    }

    #[test]
    fn support_cap_leaves_a_sample_outside_the_support() {
        let fit = aaa(&[0.0, 1.0], &[0.0, 1.0], AaaParams::new(1.0e-13, 2)).expect("aaa");
        assert_eq!(fit.support(), &[1.0]);
        assert_eq!(fit.max_residual(), 1.0);
        let nodes = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let values = nodes.map(|node: f64| (3.0 * node).sin());
        let fit = aaa(&nodes, &values, AaaParams::new(1.0e-13, usize::MAX)).expect("aaa");
        assert!(fit.len() < nodes.len());
    }

    #[test]
    fn parameters_have_defaults_setters_and_refusals() {
        let params = AaaParams::default();
        assert_eq!(params.tol, 1.0e-13);
        assert_eq!(params.max_support, 100);
        assert!(params.cleanup);
        assert_eq!(params.cleanup_tol, 1.0e-13);
        let params = params
            .with_tol(1.0e-8)
            .with_max_support(7)
            .with_cleanup(false)
            .with_cleanup_tol(0.0);
        assert_eq!(
            (
                params.tol,
                params.max_support,
                params.cleanup,
                params.cleanup_tol
            ),
            (1.0e-8, 7, false, 0.0)
        );
        let nodes = [0.0, 1.0, 2.0];
        for bad in [
            AaaParams::default().with_max_support(0),
            AaaParams::default().with_tol(0.0),
            AaaParams::default().with_cleanup_tol(-1.0),
            AaaParams::default().with_cleanup_tol(f64::NAN),
        ] {
            assert!(aaa(&nodes, &nodes, bad).is_err());
        }
        assert!(aaa(&[0.0, 0.0, 1.0], &nodes, AaaParams::default()).is_err());
    }

    #[test]
    fn barycentric_quotient_interpolates_its_support() {
        let z: Vec<f64> = (0..8)
            .map(|index| -1.0 + 2.0 * index as f64 / 7.0)
            .collect();
        let f: Vec<f64> = z.iter().map(|node| (node - 0.2).tanh()).collect();
        let fit = aaa(&z, &f, AaaParams::new(1.0e-12, 8)).expect("aaa");
        for (node, value) in fit.support().iter().zip(fit.values()) {
            assert_eq!(fit.eval(*node), Some(*value));
        }
    }

    #[test]
    fn zero_weight_is_removed_and_the_weights_solved_again() {
        let t = chebyshev(40);
        let g: Vec<f64> = t.iter().map(|node| node.exp()).collect();
        let mut support = vec![0, 13, 27];
        let mut remain = vec![true; t.len()];
        for &index in &support {
            remain[index] = false;
        }
        let mut weights = loewner_weights(&t, &g, &support, &remain).expect("weights");
        weights[1] = 0.0;
        let params = AaaParams::default().with_cleanup(false);
        prune(&t, &g, &mut support, &mut weights, &mut remain, 1.0, params).expect("prune");
        assert_eq!(support, vec![0, 27]);
        assert!(remain[13]);
        assert!(weights.iter().all(|weight| *weight != 0.0));
        assert!(unit_poles(&[0.0, 1.0], &[1.0, 2.0], &[1.0, 0.0]).is_err());
    }

    #[test]
    fn cos_50x_on_2000_chebyshev_nodes_converges() {
        let f = |x: f64| (50.0 * x).cos();
        let fit = fit(2000, f, AaaParams::default().with_max_support(200));
        assert!(fit.len() < 200, "support {}", fit.len());
        // Cleanup may lift the residual past `tol`; it stays within a decade.
        assert!(fit.max_residual() <= 1.0e-12, "{}", fit.max_residual());
        assert!(grid_error(&fit, f) <= 1.0e-12);
        let poles = fit.poles_near_interval().expect("poles");
        assert_eq!(real_poles_inside(&poles), 0);
        assert!(poles.iter().all(|pole| !pole.doublet));
    }

    #[test]
    fn pole_just_outside_the_interval_is_kept() {
        let f = |x: f64| 1.0 / (x - 1.01);
        let fit = fit(500, f, AaaParams::default());
        let poles = fit.poles_near_interval().expect("poles");
        assert_eq!(poles.len(), 1, "{poles:?}");
        let pole = poles[0];
        assert_eq!(pole.im, 0.0);
        assert!((pole.re - 1.01).abs() <= 1.0e-10, "{}", pole.re);
        assert!(
            (pole.residue_re - 1.0).abs() <= 1.0e-10,
            "{}",
            pole.residue_re
        );
        assert_eq!(pole.residue_im, 0.0);
        assert!(!pole.doublet);
    }

    #[test]
    fn meromorphic_poles_and_residues_match_the_analytic_values() {
        // 1/(x-1.5) + 3/(x+2) + 1/(x-(0.2+0.3i)) + 1/(x-(0.2-0.3i)).
        let f = |x: f64| {
            1.0 / (x - 1.5) + 3.0 / (x + 2.0) + 2.0 * (x - 0.2) / ((x - 0.2).powi(2) + 0.09)
        };
        let fit = fit(1000, f, AaaParams::default());
        let poles = fit.poles_near_interval().expect("poles");
        assert_eq!(poles.len(), 4, "{poles:?}");
        let expect = [
            (1.5, 0.0, 1.0, 0.0),
            (-2.0, 0.0, 3.0, 0.0),
            (0.2, 0.3, 1.0, 0.0),
            (0.2, -0.3, 1.0, 0.0),
        ];
        for (re, im, res_re, res_im) in expect {
            let pole = poles
                .iter()
                .min_by(|left, right| {
                    (left.re - re)
                        .hypot(left.im - im)
                        .total_cmp(&(right.re - re).hypot(right.im - im))
                })
                .expect("pole");
            assert!((pole.re - re).hypot(pole.im - im) <= 1.0e-10, "{pole:?}");
            let residue_gap = (pole.residue_re - res_re).hypot(pole.residue_im - res_im);
            assert!(residue_gap <= 1.0e-10 * res_re, "{pole:?}");
            assert!(!pole.doublet);
        }
    }

    #[test]
    fn abs_has_no_pole_on_the_interval_and_poles_cluster_at_the_kink() {
        let fit = fit(2000, f64::abs, AaaParams::default());
        assert!(fit.max_residual() <= AaaParams::default().tol);
        let poles = fit.poles_near_interval().expect("poles");
        assert_eq!(real_poles_inside(&poles), 0);
        let nearest = poles[0];
        assert!(nearest.re.hypot(nearest.im) < 1.0e-2, "{nearest:?}");
    }

    #[test]
    fn exp_is_fit_with_every_pole_away_from_the_interval() {
        let fit = fit(1000, f64::exp, AaaParams::default());
        assert!(grid_error(&fit, f64::exp) <= 1.0e-12);
        for pole in fit.poles().expect("poles") {
            assert!(
                interval_distance(C::new(pole.re, pole.im)) >= 0.5,
                "{pole:?}"
            );
        }
    }

    #[test]
    fn cleanup_removes_froissart_doublets() {
        // A tolerance far below rounding is the paper's way to make doublets.
        let params = AaaParams::default().with_tol(1.0e-300).with_max_support(40);
        let raw = fit(1000, f64::exp, params.with_cleanup(false));
        let raw_poles = raw.poles().expect("poles");
        assert!(raw_poles.iter().any(|pole| pole.doublet));
        let clean = fit(1000, f64::exp, params);
        assert!(clean.len() < raw.len());
        assert!(
            clean
                .poles()
                .expect("poles")
                .iter()
                .all(|pole| !pole.doublet)
        );
        assert!(grid_error(&clean, f64::exp) <= 1.0e-12);
    }

    #[test]
    fn extreme_scaling_keeps_the_pole_and_the_residue() {
        let u = chebyshev(500);
        let reference = aaa(
            &u,
            &u.iter().map(|t| 1.0 / (t - 1.01)).collect::<Vec<_>>(),
            AaaParams::default(),
        )
        .expect("aaa");
        for (sx, sf) in [
            (1.0e200, 1.0),
            (1.0e-200, 1.0),
            (1.0, 1.0e200),
            (1.0, 1.0e-200),
            (1.0e200, 1.0e-200),
            (1.0e-200, 1.0e200),
        ] {
            let x: Vec<f64> = u.iter().map(|t| t * sx).collect();
            let y: Vec<f64> = u.iter().map(|t| sf / (t - 1.01)).collect();
            let fit = aaa(&x, &y, AaaParams::default()).expect("aaa");
            assert_eq!(fit.len(), reference.len(), "sx {sx} sf {sf}");
            let poles = fit.poles_near_interval().expect("poles");
            assert_eq!(poles.len(), 1, "sx {sx} sf {sf}");
            assert!(
                (poles[0].re / sx - 1.01).abs() <= 1.0e-10,
                "sx {sx} sf {sf}"
            );
            let residue = poles[0].residue_re / (sx * sf);
            assert!(
                (residue - 1.0).abs() <= 1.0e-10,
                "sx {sx} sf {sf} {residue}"
            );
            let probe = 0.3 * sx;
            let got = fit.eval(probe).expect("finite") / sf;
            assert!(
                (got - 1.0 / (0.3 - 1.01)).abs() <= 1.0e-10,
                "sx {sx} sf {sf}"
            );
        }
    }

    #[test]
    fn tanh_support_and_pole_match_the_caricature() {
        for gamma in [1.0, 10.0, 80.0] {
            let tanh = |x: f64| (gamma * x).tanh();
            let fit = fit(3000, tanh, AaaParams::new(1.0e-8, 3000));
            assert!(fit.len() < 3000, "gamma {gamma}");
            assert!(
                fit.max_residual() <= 1.0e-8,
                "gamma {gamma} residual {}",
                fit.max_residual()
            );
            let poles = fit.poles_near_interval().expect("poles");
            assert!(poles.len() < fit.len(), "gamma {gamma} {}", poles.len());
            let target = std::f64::consts::PI / (2.0 * gamma);
            let nearest = poles
                .iter()
                .filter(|pole| !pole.doublet)
                .min_by(|left, right| left.im.abs().total_cmp(&right.im.abs()))
                .expect("a pole");
            let relative = (nearest.im.abs() - target).abs() / target;
            assert!(relative < 1.0e-4, "gamma {gamma} pole {}", nearest.im);
            let residue = nearest.residue_re.hypot(nearest.residue_im);
            assert!(
                (residue - 1.0 / gamma).abs() * gamma < 1.0e-3,
                "gamma {gamma}"
            );
            assert!(grid_error(&fit, tanh) <= 1.0e-8, "gamma {gamma}");
        }
    }

    #[test]
    fn tanh_poles_are_distinct_at_the_resolution_of_the_interval() {
        let fit = fit(3000, |x| (80.0 * x).tanh(), AaaParams::new(1.0e-8, 3000));
        let poles = fit.poles().expect("poles");
        assert!(poles.len() < fit.len(), "{} poles", poles.len());
        let resolution = f64::EPSILON.sqrt();
        for (index, left) in poles.iter().enumerate() {
            for right in &poles[index + 1..] {
                let gap = (left.re - right.re).hypot(left.im - right.im);
                assert!(gap > resolution, "{left:?} {right:?}");
            }
        }
        // The poles of tanh(80x) are i π (k + 1/2) / 80.
        let first = std::f64::consts::PI / 160.0;
        for sign in [1.0, -1.0] {
            let pole = poles
                .iter()
                .min_by(|left, right| {
                    left.re
                        .hypot(left.im - sign * first)
                        .total_cmp(&right.re.hypot(right.im - sign * first))
                })
                .expect("pole");
            assert!(pole.re.hypot(pole.im - sign * first) <= 1.0e-8, "{pole:?}");
        }
    }

    #[test]
    fn shifted_tanh_pole_sits_on_the_front() {
        let gamma = 20.0;
        let fit = fit(
            3000,
            |x| (gamma * (x - 0.3)).tanh(),
            AaaParams::new(1.0e-8, 3000),
        );
        let poles = fit.poles_near_interval().expect("poles");
        let target_im = std::f64::consts::PI / (2.0 * gamma);
        let nearest = poles
            .iter()
            .filter(|pole| !pole.doublet)
            .min_by(|left, right| {
                (left.re - 0.3)
                    .hypot(left.im.abs())
                    .total_cmp(&(right.re - 0.3).hypot(right.im.abs()))
            })
            .expect("pole");
        assert!((nearest.re - 0.3).abs() < 1.0e-6, "real {}", nearest.re);
        assert!(
            (nearest.im.abs() - target_im).abs() / target_im < 1.0e-5,
            "im {}",
            nearest.im
        );
    }

    #[test]
    fn chebyshev_pade_of_a_mild_tanh_finds_the_pole() {
        let nodes = chebyshev(256);
        let values: Vec<f64> = nodes.iter().map(|node| node.tanh()).collect();
        let poles = chebyshev_pade_poles(&nodes, &values, 16).expect("pade");
        let target = std::f64::consts::PI / 2.0;
        let best = poles
            .iter()
            .min_by(|left, right| {
                (left.im.abs() - target)
                    .abs()
                    .total_cmp(&(right.im.abs() - target).abs())
            })
            .expect("root");
        assert!(
            (best.im.abs() - target).abs() / target < 1.0e-4,
            "im {}",
            best.im
        );
    }
}
