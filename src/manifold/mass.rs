//! Diagonal mass metric on an embedded manifold.
//!
//! The kinetic energy is \(\tfrac12 v^\top M v\) with \(M\) diagonal and
//! positive on the free coordinates. RATTLE (Andersen, J. Comput. Phys.
//! 52, 24, 1983, <https://doi.org/10.1016/0021-9991(83)90014-1>) projects a
//! velocity onto the tangent space by a correction along \(M^{-1} N\),
//! where the columns of \(N\) span the Euclidean normal space:
//!
//! \[
//! \Pi v = v - M^{-1} N (N^\top M^{-1} N)^{-1} N^\top v.
//! \]
//!
//! \(\Pi\) is the orthogonal projector for the inner product \(u^\top M v\),
//! and \(N^\top \Pi v = 0\). A zero mass is a frozen coordinate: it is left
//! unchanged and contributes nothing to \(M^{-1}\). The same normal basis
//! gives the Lagrange multiplier of a force, \(\lambda = (N^\top M^{-1} N)^{-1}
//! N^\top M^{-1} F\), so that \(M^{-1}(F - N\lambda)\) is tangent.

use ndarray::{Array1, Array2, ArrayView2};

use super::Manifold;
use crate::error::{Error, Result};

/// Positive diagonal masses, one entry per ambient coordinate.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MassMetric {
    masses: Array1<f64>,
}

impl Default for MassMetric {
    fn default() -> Self {
        Self {
            masses: Array1::from_elem(1, 1.0),
        }
    }
}

impl MassMetric {
    /// `masses` finite and non-negative, with at least one positive entry.
    pub fn new(masses: Array1<f64>) -> Result<Self> {
        if masses.is_empty() {
            return Err(Error::MassMetric {
                what: "a mass metric needs a coordinate",
            });
        }
        if !masses.iter().all(|m| m.is_finite() && *m >= 0.0) {
            return Err(Error::MassMetric {
                what: "masses must be finite and non-negative",
            });
        }
        if !masses.iter().any(|m| *m > 0.0) {
            return Err(Error::MassMetric {
                what: "at least one mass must be positive",
            });
        }
        Ok(Self { masses })
    }

    /// `n` copies of `mass`.
    pub fn uniform(n: usize, mass: f64) -> Result<Self> {
        if n == 0 {
            return Err(Error::MassMetric {
                what: "a mass metric needs a coordinate",
            });
        }
        Self::new(Array1::from_elem(n, mass))
    }

    /// Repeat each atom mass across three Cartesian coordinates.
    pub fn per_atom(atom_masses: &[f64]) -> Result<Self> {
        if atom_masses.is_empty() {
            return Err(Error::MassMetric {
                what: "a mass metric needs an atom",
            });
        }
        let mut masses = Vec::with_capacity(atom_masses.len().saturating_mul(3));
        for mass in atom_masses {
            masses.extend_from_slice(&[*mass, *mass, *mass]);
        }
        Self::new(Array1::from_vec(masses))
    }

    /// Replace the diagonal.
    #[must_use]
    pub fn with_masses(self, masses: Array1<f64>) -> Result<Self> {
        Self::new(masses)
    }

    /// The diagonal, one entry per coordinate.
    pub fn masses(&self) -> &Array1<f64> {
        &self.masses
    }

    /// Ambient dimension.
    pub fn len(&self) -> usize {
        self.masses.len()
    }
}

/// Lagrange multiplier of a force and the reaction \(N\lambda\).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ConstraintReaction {
    /// One multiplier per normal, in the order of the normal basis.
    pub multipliers: Array1<f64>,
    /// Euclidean reaction \(N\lambda\) removed from the force.
    pub reaction: Array1<f64>,
}

impl Default for ConstraintReaction {
    fn default() -> Self {
        Self {
            multipliers: Array1::zeros(0),
            reaction: Array1::zeros(0),
        }
    }
}

impl ConstraintReaction {
    /// Set [`Self::multipliers`].
    #[must_use]
    pub fn with_multipliers(mut self, multipliers: Array1<f64>) -> Self {
        self.multipliers = multipliers;
        self
    }

    /// Set [`Self::reaction`].
    #[must_use]
    pub fn with_reaction(mut self, reaction: Array1<f64>) -> Self {
        self.reaction = reaction;
        self
    }
}

fn refuse(what: &'static str) -> Error {
    Error::MassMetric { what }
}

fn adot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(u, v)| u * v).sum()
}

/// Orthonormal Euclidean normals of `T_x`, built from `I - P_x`.
pub fn normal_frame<M>(manifold: &M, x: &Array1<f64>) -> Result<Array2<f64>>
where
    M: Manifold + ?Sized,
{
    let n = x.len();
    if n == 0 || !x.iter().all(|v| v.is_finite()) {
        return Err(refuse("the point must be finite and non-empty"));
    }
    let mut cols: Vec<Array1<f64>> = Vec::new();
    let mut scale = 1.0;
    for i in 0..n {
        let mut basis = Array1::zeros(n);
        basis[i] = 1.0;
        let projected = manifold.project(x, &basis);
        if projected.len() != n {
            return Err(refuse("the projector changed the dimension"));
        }
        let mut w = &basis - &projected;
        for _ in 0..2 {
            for col in &cols {
                let coeff = adot(col, &w);
                w = &w - col * coeff;
            }
        }
        let nrm = adot(&w, &w).sqrt();
        if nrm > scale {
            scale = nrm;
        }
        let tol = f64::EPSILON.sqrt() * scale;
        if nrm > tol {
            cols.push(&w / nrm);
        }
    }
    let mut out = Array2::zeros((n, cols.len()));
    for (j, col) in cols.iter().enumerate() {
        out.column_mut(j).assign(col);
    }
    Ok(out)
}

fn normals_of<M>(manifold: &M, x: &Array1<f64>) -> Result<Array2<f64>>
where
    M: Manifold + ?Sized,
{
    if let Some(normals) = manifold.euclidean_normals(x) {
        if normals.nrows() != x.len() {
            return Err(refuse("the normal basis has the wrong length"));
        }
        return Ok(normals);
    }
    normal_frame(manifold, x)
}

/// Project `v` onto the tangent space in the mass metric.
pub fn project_mass<M>(
    manifold: &M,
    metric: &MassMetric,
    x: &Array1<f64>,
    v: &Array1<f64>,
) -> Result<Array1<f64>>
where
    M: Manifold + ?Sized,
{
    if metric.len() != x.len() || v.len() != x.len() {
        return Err(refuse("point, velocity, and masses differ in length"));
    }
    let normals = normals_of(manifold, x)?;
    apply_projector(normals.view(), metric, v)
}

/// Restore `x` to a flat constraint along \(M^{-1} N\).
///
/// `anchor` is a point of the constraint. For one normal \(n\) this is
/// \(x - w\, n^\top(x - a)\) with \(w = M^{-1} n / (n^\top M^{-1} n)\).
pub fn restore_mass(
    normals: ArrayView2<f64>,
    metric: &MassMetric,
    anchor: &Array1<f64>,
    x: &Array1<f64>,
) -> Result<Array1<f64>> {
    if metric.len() != x.len() || anchor.len() != x.len() || normals.nrows() != x.len() {
        return Err(refuse("point, anchor, and masses differ in length"));
    }
    let c = normals.ncols();
    if c == 1 {
        // In-place RATTLE step, the same product as the flat hyperplane
        // integrator: x -= w (n·(x-a)) with w = n * (1/(α m)).
        let masses = metric
            .masses()
            .as_slice()
            .ok_or_else(|| refuse("the mass diagonal is not contiguous"))?;
        let normal = normals.column(0);
        let mut alpha = 0.0;
        for i in 0..x.len() {
            if masses[i] > 0.0 {
                alpha += normal[i] * normal[i] / masses[i];
            }
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(refuse("the mass-metric gram matrix is singular"));
        }
        let mut nx = 0.0;
        for i in 0..x.len() {
            nx += normal[i] * (x[i] - anchor[i]);
        }
        let mut out = x.to_vec();
        for i in 0..x.len() {
            let w = if masses[i] > 0.0 {
                normal[i] * (1.0 / (alpha * masses[i]))
            } else {
                0.0
            };
            out[i] -= w * nx;
        }
        return Ok(Array1::from_vec(out));
    }
    let displacement = x - anchor;
    let removed = apply_projector(normals, metric, &displacement)?;
    Ok(anchor + &removed)
}

/// \(\lambda\) and \(N\lambda\) for the force `f`.
pub fn constraint_reaction<M>(
    manifold: &M,
    metric: &MassMetric,
    x: &Array1<f64>,
    force: &Array1<f64>,
) -> Result<ConstraintReaction>
where
    M: Manifold + ?Sized,
{
    if metric.len() != x.len() || force.len() != x.len() {
        return Err(refuse("point, force, and masses differ in length"));
    }
    let normals = normals_of(manifold, x)?;
    let multipliers = multipliers_of(normals.view(), metric, force)?;
    let mut reaction = Array1::zeros(x.len());
    for (j, lambda) in multipliers.iter().enumerate() {
        let column = normals.column(j);
        for (slot, normal) in reaction.iter_mut().zip(column.iter()) {
            *slot += normal * lambda;
        }
    }
    Ok(ConstraintReaction {
        multipliers,
        reaction,
    })
}

/// Dimension of the free tangent space: positive-mass coordinates minus
/// the number of normals.
pub fn tangent_dimension<M>(manifold: &M, metric: &MassMetric, x: &Array1<f64>) -> Result<usize>
where
    M: Manifold + ?Sized,
{
    if metric.len() != x.len() {
        return Err(refuse("point and masses differ in length"));
    }
    let normals = normals_of(manifold, x)?;
    let free = metric.masses().iter().filter(|m| **m > 0.0).count();
    free.checked_sub(normals.ncols())
        .ok_or_else(|| refuse("the normals outnumber the free coordinates"))
}

fn multipliers_of(
    normals: ArrayView2<f64>,
    metric: &MassMetric,
    vector: &Array1<f64>,
) -> Result<Array1<f64>> {
    let n = vector.len();
    let c = normals.ncols();
    if c == 0 {
        return Ok(Array1::zeros(0));
    }
    let masses = metric
        .masses()
        .as_slice()
        .ok_or_else(|| refuse("the mass diagonal is not contiguous"))?;
    if c == 1 {
        let normal = normals.column(0);
        let mut alpha = 0.0;
        let mut rhs = 0.0;
        for i in 0..n {
            if masses[i] > 0.0 {
                let inv = 1.0 / masses[i];
                alpha += normal[i] * normal[i] * inv;
                rhs += normal[i] * vector[i] * inv;
            }
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(refuse("the mass-metric gram matrix is singular"));
        }
        return Ok(Array1::from_elem(1, rhs / alpha));
    }
    let mut gram = vec![0.0; c * c];
    let mut rhs = vec![0.0; c];
    for i in 0..n {
        if masses[i] == 0.0 {
            continue;
        }
        let inv = 1.0 / masses[i];
        for a in 0..c {
            let na = normals[(i, a)];
            rhs[a] += na * vector[i] * inv;
            for b in 0..c {
                gram[a * c + b] += na * normals[(i, b)] * inv;
            }
        }
    }
    solve_linear(&mut gram, &mut rhs, c)?;
    Ok(Array1::from_vec(rhs))
}

/// Mass-metric projection. Codimension one uses the RATTLE product
/// \(w = M^{-1} n / (n^\top M^{-1} n)\), \(v \leftarrow v - w (n^\top v)\),
/// which is the arithmetic of the flat hyperplane step.
pub fn apply_projector(
    normals: ArrayView2<f64>,
    metric: &MassMetric,
    v: &Array1<f64>,
) -> Result<Array1<f64>> {
    let n = v.len();
    if metric.len() != n || normals.nrows() != n {
        return Err(refuse("velocity, normals, and masses differ in length"));
    }
    if !v.iter().all(|value| value.is_finite()) {
        return Err(refuse("the velocity must be finite"));
    }
    let c = normals.ncols();
    if c == 0 {
        return Ok(v.clone());
    }
    let masses = metric
        .masses()
        .as_slice()
        .ok_or_else(|| refuse("the mass diagonal is not contiguous"))?;
    if c == 1 {
        let normal = normals.column(0);
        let mut alpha = 0.0;
        for i in 0..n {
            if masses[i] > 0.0 {
                alpha += normal[i] * normal[i] / masses[i];
            }
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(refuse("the mass-metric gram matrix is singular"));
        }
        let mut nv = 0.0;
        for i in 0..n {
            nv += normal[i] * v[i];
        }
        let mut out = v.to_vec();
        for i in 0..n {
            let w = if masses[i] > 0.0 {
                normal[i] * (1.0 / (alpha * masses[i]))
            } else {
                0.0
            };
            out[i] -= w * nv;
        }
        return Ok(Array1::from_vec(out));
    }
    let lambda = velocity_lambda(normals, metric, v)?;
    let mut out = v.to_vec();
    for i in 0..n {
        if masses[i] == 0.0 {
            continue;
        }
        let mut corr = 0.0;
        for (j, lam) in lambda.iter().enumerate() {
            corr += normals[(i, j)] * lam;
        }
        out[i] -= corr / masses[i];
    }
    Ok(Array1::from_vec(out))
}

/// \(\lambda = (N^\top M^{-1} N)^{-1} N^\top v\). The right-hand side is the
/// Euclidean product \(N^\top v\), not \(N^\top M^{-1} v\).
fn velocity_lambda(
    normals: ArrayView2<f64>,
    metric: &MassMetric,
    v: &Array1<f64>,
) -> Result<Array1<f64>> {
    let n = v.len();
    let c = normals.ncols();
    let masses = metric
        .masses()
        .as_slice()
        .ok_or_else(|| refuse("the mass diagonal is not contiguous"))?;
    let mut gram = vec![0.0; c * c];
    let mut rhs = vec![0.0; c];
    for i in 0..n {
        for a in 0..c {
            rhs[a] += normals[(i, a)] * v[i];
        }
        if masses[i] == 0.0 {
            continue;
        }
        let inv = 1.0 / masses[i];
        for a in 0..c {
            for b in 0..c {
                gram[a * c + b] += normals[(i, a)] * normals[(i, b)] * inv;
            }
        }
    }
    solve_linear(&mut gram, &mut rhs, c)?;
    Ok(Array1::from_vec(rhs))
}

fn solve_linear(a: &mut [f64], rhs: &mut [f64], c: usize) -> Result<()> {
    let mut max_entry = 0.0;
    for value in a.iter() {
        if value.abs() > max_entry {
            max_entry = value.abs();
        }
    }
    let tol = (c as f64) * f64::EPSILON * max_entry.max(1.0);
    for k in 0..c {
        let mut pivot = k;
        let mut best = a[k * c + k].abs();
        for i in (k + 1)..c {
            let value = a[i * c + k].abs();
            if value > best {
                best = value;
                pivot = i;
            }
        }
        if !(best.is_finite() && best > tol) {
            return Err(refuse("the mass-metric gram matrix is singular"));
        }
        if pivot != k {
            for j in k..c {
                a.swap(k * c + j, pivot * c + j);
            }
            rhs.swap(k, pivot);
        }
        let diag = a[k * c + k];
        for i in (k + 1)..c {
            let factor = a[i * c + k] / diag;
            a[i * c + k] = factor;
            for j in (k + 1)..c {
                a[i * c + j] -= factor * a[k * c + j];
            }
            rhs[i] -= factor * rhs[k];
        }
    }
    for k in (0..c).rev() {
        let mut sum = rhs[k];
        for j in (k + 1)..c {
            sum -= a[k * c + j] * rhs[j];
        }
        rhs[k] = sum / a[k * c + k];
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(refuse("the mass-metric gram matrix is singular"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::Euclidean;
    use ndarray::array;

    #[test]
    fn codimension_one_matches_the_rattle_product() {
        let metric = MassMetric::new(array![4.0, 1.0, 4.0]).unwrap();
        let normal = array![0.0, 1.0, 0.0];
        let mut normals = Array2::zeros((3, 1));
        normals.column_mut(0).assign(&normal);
        let v = array![0.5, -2.0, 0.25];
        let got = apply_projector(normals.view(), &metric, &v).unwrap();
        let alpha = 1.0 / 1.0;
        let nv = -2.0;
        let w1 = 1.0 * (1.0 / (alpha * 1.0));
        assert_eq!(got[0].to_bits(), 0.5_f64.to_bits());
        let expected = -2.0_f64 - w1 * nv;
        assert_eq!(got[1].to_bits(), expected.to_bits());
        assert_eq!(got[2].to_bits(), 0.25_f64.to_bits());
        let again = apply_projector(normals.view(), &metric, &got).unwrap();
        let mut residual = 0.0;
        for i in 0..3 {
            residual += normal[i] * again[i];
        }
        assert_eq!(residual.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn two_axes_remove_only_those_components() {
        let metric = MassMetric::new(array![1.0, 2.0, 3.0]).unwrap();
        let mut normals = Array2::zeros((3, 2));
        normals[(0, 0)] = 1.0;
        normals[(1, 1)] = 1.0;
        let v = array![1.0, 1.0, 1.0];
        let got = apply_projector(normals.view(), &metric, &v).unwrap();
        assert_eq!(got[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(got[1].to_bits(), 0.0_f64.to_bits());
        assert_eq!(got[2].to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn equal_mass_multiplier_is_the_normal_force() {
        let metric = MassMetric::uniform(2, 3.0).unwrap();
        let mut normals = Array2::zeros((2, 1));
        normals[(0, 0)] = 0.6;
        normals[(1, 0)] = 0.8;
        let force = array![1.5, -0.5];
        let reaction = multipliers_of(normals.view(), &metric, &force).unwrap();
        let expect = 0.6 * 1.5 + 0.8 * -0.5;
        let gap = (reaction[0] - expect).abs();
        let scale = expect.abs().max(1.0);
        assert!(gap <= 4.0 * f64::EPSILON * scale);
    }

    #[test]
    fn euclidean_space_has_no_normal() {
        let metric = MassMetric::uniform(4, 1.0).unwrap();
        let x = array![0.0, 1.0, 2.0, 3.0];
        let v = array![1.0, -1.0, 0.5, 0.25];
        let got = project_mass(&Euclidean, &metric, &x, &v).unwrap();
        assert_eq!(got, v);
        assert_eq!(tangent_dimension(&Euclidean, &metric, &x).unwrap(), 4);
    }
}
