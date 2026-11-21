//! AAA rational approximation and barycentric evaluation.
//!
//! Nakatsukasa, Sète, and Trefethen, SIAM J. Sci. Comput. 40 (2018)
//! A1494–A1522, Theorem 2.1: fixed support points and nonzero weights
//! give a rational of type `(m-1, m-1)` that interpolates the support
//! values. The weights are the smallest right singular vector of the
//! Loewner matrix. A pole used as a map width is a Newton zero of the
//! barycentric denominator nearest the sample interval. A Froissart
//! doublet is a pole whose residue is on the scale of roundoff.

use crate::error::{Error, Result};

/// Stopping rule for [`aaa`].
///
/// Both fields are caller inputs. The tolerance is the residual the
/// caller will accept, and the support cap is the caller's budget.
/// Exhausting the cap with a residual still above the tolerance returns
/// the fit that was reached; a non-positive cap is refused.
#[derive(Clone, Copy, Debug)]
pub struct AaaParams {
    /// Absolute residual at which greedy support growth stops.
    pub tol: f64,
    /// Largest support the greedy step may build.
    pub max_support: usize,
}

/// One pole of a barycentric rational, with its residue.
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
    /// Residue on the scale of roundoff, or a pole sitting on a support point.
    pub doublet: bool,
}

/// Frozen barycentric quotient `Σ w_j f_j/(z-z_j) / Σ w_j/(z-z_j)`.
#[derive(Clone, Debug)]
pub struct Barycentric {
    z: Vec<f64>,
    f: Vec<f64>,
    w: Vec<f64>,
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

    /// Barycentric weights.
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

    /// Largest absolute residual on the samples that were not selected.
    pub fn max_residual(&self) -> f64 {
        self.max_residual
    }

    /// Evaluate the frozen quotient. A sample at a support point returns that value.
    pub fn eval(&self, x: f64) -> Option<f64> {
        eval_support(x, &self.z, &self.f, &self.w)
    }

    /// Poles near `[-1, 1]` found as Newton zeros of the denominator.
    ///
    /// The search is local to that interval. It is the set used to read a
    /// singular-line distance, not a complete catalogue of the plane.
    pub fn poles_near_interval(&self) -> Result<Vec<Pole>> {
        if self.z.len() < 2 {
            return Ok(Vec::new());
        }
        if self
            .w
            .iter()
            .any(|weight| !weight.is_finite() || *weight == 0.0)
        {
            return Err(Error::Aaa {
                what: "a barycentric weight is zero or not finite",
            });
        }
        let scale = sample_scale(&self.f)?;
        let geometry = real_scale(&self.z).ok_or(Error::Aaa {
            what: "the support has no positive scale",
        })?;
        let mut found = Vec::new();
        for seed in pole_seeds(&self.z, &geometry) {
            if let Some(pole) = newton_pole(seed, &self.z, &self.f, &self.w, scale, &geometry) {
                if found.iter().any(|kept: &Pole| {
                    let dre = kept.re - pole.re;
                    let dim = kept.im - pole.im;
                    dre.hypot(dim) <= f64::EPSILON * geometry.diameter
                }) {
                    continue;
                }
                found.push(pole);
            }
        }
        Ok(found)
    }
}

/// Greedy AAA fit of real samples.
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
    for (sample, value) in samples.iter().zip(values) {
        if !sample.is_finite() || !value.is_finite() {
            return Err(Error::Aaa {
                what: "AAA samples must be finite",
            });
        }
    }
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            if samples[i] == samples[j] {
                return Err(Error::Aaa {
                    what: "AAA sample points are not distinct",
                });
            }
        }
    }
    let mut remain = vec![true; samples.len()];
    let mut first = 0usize;
    for i in 1..values.len() {
        if values[i].abs() > values[first].abs() {
            first = i;
        }
    }
    let mut support = vec![first];
    remain[first] = false;
    let mut weights = vec![1.0];
    let mut max_residual = f64::INFINITY;
    let cap = params.max_support.min(samples.len());
    while support.len() <= cap {
        if support.len() > 1 {
            weights = loewner_weights(samples, values, &support, &remain)?;
        }
        let mut worst_at = None;
        let mut worst = -1.0_f64;
        for (index, selected) in remain.iter().copied().enumerate() {
            if !selected {
                continue;
            }
            let Some(predicted) =
                eval_on_support(samples[index], samples, values, &support, &weights)
            else {
                return Err(Error::Aaa {
                    what: "the barycentric quotient was not finite",
                });
            };
            let gap = (values[index] - predicted).abs();
            if gap > worst {
                worst = gap;
                worst_at = Some(index);
            }
        }
        max_residual = if worst_at.is_some() { worst } else { 0.0 };
        if worst_at.is_none() || worst <= params.tol || support.len() == cap {
            break;
        }
        let Some(index) = worst_at else {
            break;
        };
        support.push(index);
        remain[index] = false;
    }
    let z = support.iter().map(|&index| samples[index]).collect();
    let f = support.iter().map(|&index| values[index]).collect();
    Ok(Barycentric {
        z,
        f,
        w: weights,
        max_residual,
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

#[derive(Clone, Copy)]
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
    fn div(self, other: Self) -> Self {
        let den = other.re * other.re + other.im * other.im;
        Self {
            re: (self.re * other.re + self.im * other.im) / den,
            im: (self.im * other.re - self.re * other.im) / den,
        }
    }
}

fn eval_on_support(
    x: f64,
    samples: &[f64],
    values: &[f64],
    support: &[usize],
    weights: &[f64],
) -> Option<f64> {
    let z: Vec<f64> = support.iter().map(|&index| samples[index]).collect();
    let f: Vec<f64> = support.iter().map(|&index| values[index]).collect();
    eval_support(x, &z, &f, weights)
}

fn eval_support(x: f64, z: &[f64], f: &[f64], w: &[f64]) -> Option<f64> {
    if !x.is_finite() || z.is_empty() || z.len() != f.len() || z.len() != w.len() {
        return None;
    }
    let mut nearest = 0usize;
    let mut nearest_abs = f64::INFINITY;
    for (index, &node) in z.iter().enumerate() {
        let gap = (x - node).abs();
        if gap < nearest_abs {
            nearest_abs = gap;
            nearest = index;
        }
    }
    let mut node_scale = x.abs();
    for &node in z {
        node_scale = node_scale.max(node.abs());
    }
    if node_scale > 0.0 && nearest_abs <= f64::EPSILON * node_scale {
        return Some(f[nearest]);
    }
    if nearest_abs == 0.0 {
        return Some(f[nearest]);
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for index in 0..z.len() {
        let cauchy = w[index] / (x - z[index]);
        num += cauchy * f[index];
        den += cauchy;
    }
    let value = num / den;
    if value.is_finite() { Some(value) } else { None }
}

fn loewner_weights(
    samples: &[f64],
    values: &[f64],
    support: &[usize],
    remain: &[bool],
) -> Result<Vec<f64>> {
    let cols = support.len();
    let rows: Vec<usize> = remain
        .iter()
        .enumerate()
        .filter_map(|(index, kept)| if *kept { Some(index) } else { None })
        .collect();
    if rows.is_empty() || cols == 0 {
        return Err(Error::Aaa {
            what: "the Loewner matrix is empty",
        });
    }
    let mut matrix = vec![0.0; rows.len() * cols];
    for (row, &sample_index) in rows.iter().enumerate() {
        for (col, &support_index) in support.iter().enumerate() {
            let gap = samples[sample_index] - samples[support_index];
            if gap == 0.0 {
                return Err(Error::Aaa {
                    what: "a Loewner entry hit a repeated node",
                });
            }
            matrix[row * cols + col] = (values[sample_index] - values[support_index]) / gap;
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

fn smallest_right_singular(matrix: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>> {
    if rows == 0 || cols == 0 || matrix.len() != rows * cols {
        return Err(Error::Aaa {
            what: "the singular-vector matrix has the wrong shape",
        });
    }
    if cols == 1 {
        return Ok(vec![1.0]);
    }
    let mut columns = matrix.to_vec();
    let mut vectors = vec![0.0; cols * cols];
    for index in 0..cols {
        vectors[index * cols + index] = 1.0;
    }
    // A zero matrix has no distinguished singular vector.
    let opening = squared_frobenius(&columns)?;
    if opening == 0.0 {
        return Err(Error::Aaa {
            what: "the Loewner matrix has zero scale",
        });
    }
    // One sweep per significand bit per column. Past that the rotation
    // is not converging and the caller gets a typed refusal.
    let sweep_budget = cols.saturating_mul(f64::MANTISSA_DIGITS as usize);
    let mut sweeps = 0usize;
    loop {
        let mass = squared_frobenius(&columns)?;
        if mass == 0.0 {
            return Err(Error::Aaa {
                what: "the Loewner matrix has zero scale",
            });
        }
        for left in 0..cols {
            for right in (left + 1)..cols {
                let mut aa = 0.0;
                let mut bb = 0.0;
                let mut cc = 0.0;
                for row in 0..rows {
                    let left_value = columns[row * cols + left];
                    let right_value = columns[row * cols + right];
                    aa += left_value * left_value;
                    bb += right_value * right_value;
                    cc += left_value * right_value;
                }
                // A column at EPSILON² of the matrix energy is roundoff,
                // including a numerical null column. Both tests scale as α².
                if !pair_needs_rotation(aa, bb, cc, mass) {
                    continue;
                }
                let tau = (bb - aa) / (2.0 * cc);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let cosine = 1.0 / (1.0 + t * t).sqrt();
                let sine = cosine * t;
                if !cosine.is_finite() || !sine.is_finite() {
                    return Err(Error::Aaa {
                        what: "the Loewner matrix is not finite",
                    });
                }
                for row in 0..rows {
                    let left_value = columns[row * cols + left];
                    let right_value = columns[row * cols + right];
                    columns[row * cols + left] = left_value * cosine - right_value * sine;
                    columns[row * cols + right] = left_value * sine + right_value * cosine;
                }
                for row in 0..cols {
                    let left_value = vectors[row * cols + left];
                    let right_value = vectors[row * cols + right];
                    vectors[row * cols + left] = left_value * cosine - right_value * sine;
                    vectors[row * cols + right] = left_value * sine + right_value * cosine;
                }
            }
        }
        // |col_i · col_j| and ||col_i|| ||col_j|| both scale as α².
        // The sweep is done when every resolved pair meets that test.
        let relative = gram_relative_off(&columns, rows, cols)?;
        if relative <= f64::EPSILON {
            break;
        }
        sweeps += 1;
        if sweeps >= sweep_budget {
            return Err(Error::Aaa {
                what: "the Jacobi sweep did not converge",
            });
        }
    }
    let mut best = 0usize;
    let mut best_value = f64::INFINITY;
    for col in 0..cols {
        let mut norm = 0.0;
        for row in 0..rows {
            let value = columns[row * cols + col];
            norm += value * value;
        }
        if norm < best_value {
            best_value = norm;
            best = col;
        }
    }
    let mut out = vec![0.0; cols];
    let mut norm = 0.0;
    for row in 0..cols {
        out[row] = vectors[row * cols + best];
        norm += out[row] * out[row];
    }
    let norm = norm.sqrt();
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

/// Sum of squares of a real matrix. A non-finite entry is refused.
fn squared_frobenius(values: &[f64]) -> Result<f64> {
    let mut mass = 0.0;
    for value in values {
        if !value.is_finite() {
            return Err(Error::Aaa {
                what: "the Loewner matrix is not finite",
            });
        }
        mass += value * value;
    }
    if !mass.is_finite() {
        return Err(Error::Aaa {
            what: "the Loewner matrix is not finite",
        });
    }
    Ok(mass)
}

/// `true` when the pair is above roundoff and not yet relatively orthogonal.
///
/// Column energy and `EPSILON² * mass` both scale as α². The inner
/// product and the product of the column norms both scale as α² as well.
fn pair_needs_rotation(aa: f64, bb: f64, cc: f64, mass: f64) -> bool {
    if !(aa.is_finite() && bb.is_finite() && cc.is_finite() && mass.is_finite()) {
        return true;
    }
    let roundoff = f64::EPSILON * f64::EPSILON * mass;
    if !(aa > roundoff && bb > roundoff) {
        return false;
    }
    let scale = aa.sqrt() * bb.sqrt();
    cc.abs() > f64::EPSILON * scale
}

fn gram_relative_off(columns: &[f64], rows: usize, cols: usize) -> Result<f64> {
    let mass = squared_frobenius(columns)?;
    let mut relative = 0.0_f64;
    for left in 0..cols {
        for right in (left + 1)..cols {
            let mut aa = 0.0;
            let mut bb = 0.0;
            let mut cc = 0.0;
            for row in 0..rows {
                let left_value = columns[row * cols + left];
                let right_value = columns[row * cols + right];
                aa += left_value * left_value;
                bb += right_value * right_value;
                cc += left_value * right_value;
            }
            if !aa.is_finite() || !bb.is_finite() || !cc.is_finite() {
                return Err(Error::Aaa {
                    what: "the Loewner matrix is not finite",
                });
            }
            let roundoff = f64::EPSILON * f64::EPSILON * mass;
            if !(aa > roundoff && bb > roundoff) {
                continue;
            }
            let scale = aa.sqrt() * bb.sqrt();
            if !(scale.is_finite() && scale > 0.0) {
                return Err(Error::Aaa {
                    what: "the Loewner matrix is not finite",
                });
            }
            relative = relative.max(cc.abs() / scale);
        }
    }
    if !relative.is_finite() {
        return Err(Error::Aaa {
            what: "the Loewner matrix is not finite",
        });
    }
    Ok(relative)
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

struct RealScale {
    lo: f64,
    hi: f64,
    diameter: f64,
    spacing: f64,
}

fn real_scale(support: &[f64]) -> Option<RealScale> {
    if support.len() < 2 {
        return None;
    }
    let mut lo = support[0];
    let mut hi = support[0];
    for &node in &support[1..] {
        lo = lo.min(node);
        hi = hi.max(node);
    }
    let diameter = hi - lo;
    if !(diameter.is_finite() && diameter > 0.0) {
        return None;
    }
    let mut spacing = diameter;
    for left in 0..support.len() {
        for right in (left + 1)..support.len() {
            let gap = (support[left] - support[right]).abs();
            if gap > 0.0 && gap < spacing {
                spacing = gap;
            }
        }
    }
    if !(spacing.is_finite() && spacing > 0.0) {
        return None;
    }
    Some(RealScale {
        lo,
        hi,
        diameter,
        spacing,
    })
}

/// Imaginary seeds at the scales the support itself resolves.
///
/// The heights run from the smallest node gap to the support diameter,
/// with one height per support node. That is the set of distances the
/// samples can tell apart.
fn pole_seeds(support: &[f64], geometry: &RealScale) -> Vec<C> {
    let mut seeds = Vec::new();
    let levels = support.len();
    let ratio = geometry.diameter / geometry.spacing;
    for index in 0..levels {
        let t = index as f64 / (levels - 1) as f64;
        let height = geometry.spacing * ratio.powf(t);
        if !(height.is_finite() && height > 0.0) {
            continue;
        }
        for &node in support {
            seeds.push(C::new(node, height));
            seeds.push(C::new(node, -height));
        }
    }
    seeds
}

fn newton_pole(
    mut z: C,
    support: &[f64],
    values: &[f64],
    weights: &[f64],
    scale: f64,
    geometry: &RealScale,
) -> Option<Pole> {
    // A root is a denominator at its own rounding, or a correction at
    // an ulp of the local scale. The budget is one correction per
    // significand bit per support node; a seed that misses it is not a pole.
    let step_budget = support.len().saturating_mul(f64::MANTISSA_DIGITS as usize);
    let mut steps = 0usize;
    loop {
        let (den, der, _num) = barycentric_parts(z, support, values, weights)?;
        if der.abs() == 0.0 || !den.re.is_finite() || !den.im.is_finite() {
            return None;
        }
        let mut weight_scale = 0.0_f64;
        for (node, weight) in support.iter().zip(weights) {
            let gap = (z.re - node).hypot(z.im);
            if gap == 0.0 {
                return None;
            }
            weight_scale += weight.abs() / gap;
        }
        if !weight_scale.is_finite() || weight_scale == 0.0 {
            return None;
        }
        let local = z.abs().max(geometry.spacing);
        if den.abs() <= f64::EPSILON * weight_scale {
            break;
        }
        let step = den / der;
        let step_abs = step.abs();
        if !step_abs.is_finite() {
            return None;
        }
        z = z - step;
        if step_abs <= f64::EPSILON * local {
            break;
        }
        steps += 1;
        if steps >= step_budget {
            return None;
        }
    }
    if !z.re.is_finite() || !z.im.is_finite() {
        return None;
    }
    if z.im.abs() <= f64::EPSILON * geometry.spacing {
        return None;
    }
    if z.re < geometry.lo - geometry.spacing || z.re > geometry.hi + geometry.spacing {
        return None;
    }
    let (den, der, num) = barycentric_parts(z, support, values, weights)?;
    if !den.re.is_finite() || !den.im.is_finite() {
        return None;
    }
    let residue = if der.abs() == 0.0 {
        return None;
    } else {
        num / der
    };
    if !residue.re.is_finite() || !residue.im.is_finite() {
        return None;
    }
    let on_support = support.iter().any(|node| {
        let gap = (z.re - node).abs().max(z.im.abs());
        gap <= f64::EPSILON * geometry.spacing.max(node.abs())
    });
    Some(Pole {
        re: z.re,
        im: z.im,
        residue_re: residue.re,
        residue_im: residue.im,
        doublet: on_support || residue.abs() <= f64::EPSILON * scale,
    })
}

fn barycentric_parts(z: C, support: &[f64], values: &[f64], weights: &[f64]) -> Option<(C, C, C)> {
    let mut den = C::new(0.0, 0.0);
    let mut der = C::new(0.0, 0.0);
    let mut num = C::new(0.0, 0.0);
    for index in 0..support.len() {
        let gap = z - C::real(support[index]);
        if gap.abs() == 0.0 {
            return None;
        }
        let inv = C::real(weights[index]) / gap;
        den = den + inv;
        num = num + inv * C::real(values[index]);
        der = der + C::real(-weights[index]) / (gap * gap);
    }
    Some((den, der, num))
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

    #[test]
    fn smallest_singular_vector_matches_a_dense_svd() {
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weights = smallest_right_singular(&matrix, 4, 2).expect("svd");
        let expect = [0.767187395_f64, -0.641423027];
        let mut gap = 0.0_f64;
        for (got, want) in weights.iter().zip(&expect) {
            gap = gap.max((got - want).abs());
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
        let mut half = det * 0.5;
        if half > 1.0 {
            half = 1.0;
        }
        if half < -1.0 {
            half = -1.0;
        }
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
                let mut norm = 0.0;
                for value in cross {
                    norm += value * value;
                }
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
        let direction: Vec<f64> = best.iter().map(|value| value / norm).collect();
        let mut residual = 0.0;
        for row in 0..3 {
            let mut acc = 0.0;
            for col in 0..3 {
                let entry = if row == col {
                    gram[row * 3 + col] - lambda
                } else {
                    gram[row * 3 + col]
                };
                acc += entry * direction[col];
            }
            residual += acc * acc;
        }
        let gram_norm = gram.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!(
            residual.sqrt() <= (cols as f64) * f64::EPSILON * gram_norm,
            "oracle eigenresidual {} against gram {}",
            residual.sqrt(),
            gram_norm
        );
        direction
    }

    fn direction_gap(left: &[f64], right: &[f64]) -> f64 {
        let mut dot = 0.0;
        for (lhs, rhs) in left.iter().zip(right) {
            dot += lhs * rhs;
        }
        1.0 - dot.abs()
    }

    #[test]
    fn singular_vector_matches_an_oracle_at_tiny_and_huge_scale() {
        // Three coupled columns. One Jacobi sweep cannot finish the
        // diagonalization, so a stop that ignores later sweeps drifts
        // from the closed-form Gram eigenpair.
        let base = [
            1.0, 0.2, -0.4, 0.3, 1.1, 0.5, -0.7, 0.4, 1.2, 0.8, -0.6, 0.3, 0.2, 0.9, -0.8,
        ];
        let rows = 5usize;
        let cols = 3usize;
        for scale in [1.0e-18, 1.0e18] {
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
    fn barycentric_quotient_interpolates_its_support() {
        let z: Vec<f64> = (0..8)
            .map(|index| -1.0 + 2.0 * index as f64 / 7.0)
            .collect();
        let f: Vec<f64> = z.iter().map(|node| (node - 0.2).tanh()).collect();
        let fit = aaa(
            &z,
            &f,
            AaaParams {
                tol: 1.0e-12,
                max_support: 8,
            },
        )
        .expect("aaa");
        for (node, value) in fit.support().iter().zip(fit.values()) {
            let got = fit.eval(*node).expect("support value");
            assert!((got - value).abs() <= 1.0e-12 * (1.0 + value.abs()));
        }
    }

    fn chebyshev(count: usize) -> Vec<f64> {
        (0..count)
            .map(|index| ((index as f64 + 0.5) * std::f64::consts::PI / count as f64).cos())
            .collect()
    }

    #[test]
    fn tanh_support_and_pole_match_the_caricature() {
        for gamma in [1.0, 10.0, 80.0] {
            let nodes = chebyshev(3000);
            let values: Vec<f64> = nodes.iter().map(|node| (gamma * node).tanh()).collect();
            let fit = aaa(
                &nodes,
                &values,
                AaaParams {
                    tol: 1.0e-8,
                    max_support: nodes.len(),
                },
            )
            .expect("tanh aaa");
            assert!(
                fit.len() < nodes.len(),
                "gamma {gamma} exhausted the sample instead of the residual"
            );
            assert!(
                fit.max_residual() <= 1.0e-8,
                "gamma {gamma} residual {}",
                fit.max_residual()
            );
            let poles = fit.poles_near_interval().expect("poles");
            let target = std::f64::consts::PI / (2.0 * gamma);
            let nearest = poles
                .iter()
                .filter(|pole| !pole.doublet)
                .min_by(|left, right| left.im.abs().partial_cmp(&right.im.abs()).unwrap())
                .expect("a pole");
            let relative = (nearest.im.abs() - target).abs() / target;
            assert!(
                relative < 1.0e-4,
                "gamma {gamma} pole {} relative {relative}",
                nearest.im
            );
            let residue = nearest.residue_re.hypot(nearest.residue_im);
            let residue_relative = (residue - 1.0 / gamma).abs() * gamma;
            assert!(residue_relative < 1.0e-3, "gamma {gamma} residue {residue}");
            let mut check = 0.0_f64;
            for index in 0..2001 {
                let node = -1.0 + 2.0 * index as f64 / 2000.0;
                let got = fit.eval(node).expect("grid");
                check = check.max((got - (gamma * node).tanh()).abs());
            }
            assert!(check <= 1.0e-8, "gamma {gamma} check {check}");
        }
    }

    #[test]
    fn shifted_tanh_pole_sits_on_the_front() {
        let gamma = 20.0;
        let nodes = chebyshev(3000);
        let values: Vec<f64> = nodes
            .iter()
            .map(|node| (gamma * (node - 0.3)).tanh())
            .collect();
        let fit = aaa(
            &nodes,
            &values,
            AaaParams {
                tol: 1.0e-8,
                max_support: nodes.len(),
            },
        )
        .expect("aaa");
        let poles = fit.poles_near_interval().expect("poles");
        let target_im = std::f64::consts::PI / (2.0 * gamma);
        let nearest = poles
            .iter()
            .filter(|pole| !pole.doublet)
            .min_by(|left, right| {
                (left.re - 0.3)
                    .hypot(left.im.abs())
                    .partial_cmp(&(right.re - 0.3).hypot(right.im.abs()))
                    .unwrap()
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
                    .partial_cmp(&(right.im.abs() - target).abs())
                    .unwrap()
            })
            .expect("root");
        assert!(
            (best.im.abs() - target).abs() / target < 1.0e-4,
            "im {}",
            best.im
        );
    }
}
