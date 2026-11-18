//! Weighted Riemannian product \(\mathcal{M} = \mathcal{M}_1 \times \cdots
//! \times \mathcal{M}_k\).
//!
//! A point packs the factor points end to end. The metric is
//! \(g = \sum_i w_i\, g_i\) with every \(w_i > 0\), so the tangent space
//! splits into orthogonal blocks. Projection, retraction, vector and
//! parallel transport, exponential and logarithm act factor by factor.
//! The weights enter the inner product, the Riemannian gradient
//! \((w_i^{-1}\,\mathrm{grad}_i f)_i\), the Hessian
//! \((w_i^{-1}\,\mathrm{Hess}_i f)_i\), the distance
//! \(\sqrt{\sum_i w_i d_i^2}\) and the typical distance. All weights one
//! is the product of manopt `productmanifold`, ManifoldsBase.jl
//! `ProductManifold` (the one Manifolds.jl and Manopt.jl use), pymanopt
//! `Product` and geomstats `ProductRiemannianMetric`; geomstats weights a
//! factor by wrapping its metric in `ScalarProductMetric`.
//!
//! The map \(z_i = \sqrt{w_i}\,x_i\) is an isometry onto the product of
//! the scaled factors \(\sqrt{w_i}\,\mathcal{M}_i\) with unit weights.
//! There the metric of every embedded factor is the Euclidean one of the
//! packed vector, which is the metric the session steppers assume; see
//! [`ProductManifold::chart`]. A block stores \(s_i x_i\) for a factor
//! point \(x_i\) and carries the metric \(c_i\, g_i\) in those
//! coordinates, with \(c_i s_i^2 = w_i\): \(s_i = 1, c_i = w_i\) in natural
//! coordinates and \(s_i = \sqrt{w_i}, c_i = 1\) in the chart.
//!
//! A factor may also carry a [`TangentMetric`] \(G\), an operator on its
//! tangent spaces that the caller supplies, such as a harmonic Newton
//! preconditioner. Its metric is then \(w_i\, g_i(u, G v)\), its gradient
//! \(w_i^{-1} G^{-1} \mathrm{grad}_i f\) and its Hessian model
//! \(w_i^{-1} G^{-1} \mathrm{Hess}_i f\), which is self-adjoint in that
//! metric and exact at critical points. Its geodesics are not known in
//! closed form, so the exponential, logarithm, distance and parallel
//! transport of such a product are `None`.
//!
//! The derivation is the explanation page `product-manifold`.
//! Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix
//! Manifolds*, <https://doi.org/10.1515/9781400830244>.
//! Boumal, *An Introduction to Optimization on Smooth Manifolds*,
//! <https://doi.org/10.1017/9781009166164>.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use ndarray::{Array1, s};

use crate::error::{Error, Result};

use super::{Manifold, ManifoldKind};

/// A metric operator \(G_x\) on the tangent spaces of one product factor.
///
/// The factor's contribution to the product metric becomes
/// \(w\, g(u, G_x v)\), where \(g\) is the factor's own metric. At every
/// factor point `x`, \(G_x\) must map the tangent space to itself and be
/// self-adjoint and positive definite in \(g\), and [`Self::solve`] must
/// invert [`Self::apply`] there. The product projects both results onto
/// the tangent space; that removes rounding, not a wrong operator. Both
/// methods receive the factor point and tangent vectors in the factor's
/// own packing.
pub trait TangentMetric: Send + Sync {
    /// \(G_x v\) for a tangent vector `v` at the factor point `x`.
    fn apply(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;

    /// \(G_x^{-1} r\) for a tangent vector `r` at the factor point `x`.
    fn solve(&self, x: &Array1<f64>, r: &Array1<f64>) -> Array1<f64>;
}

/// A shared [`TangentMetric`], equal only to itself.
#[derive(Clone)]
struct Metric(Arc<dyn TangentMetric>);

impl fmt::Debug for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TangentMetric")
    }
}

impl PartialEq for Metric {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// One factor of a [`ProductManifold`]: a geometry, its packed length,
/// its metric weight and an optional [`TangentMetric`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ProductFactor {
    /// Geometry of the factor.
    pub kind: ManifoldKind,
    /// Packed length of a factor point.
    pub len: usize,
    /// Weight \(w_i > 0\) of the factor metric in the product metric.
    pub weight: f64,
    metric: Option<Metric>,
}

impl Default for ProductFactor {
    fn default() -> Self {
        Self {
            kind: ManifoldKind::Euclidean,
            len: 1,
            weight: 1.0,
            metric: None,
        }
    }
}

impl ProductFactor {
    /// `kind` at packed length `len`, weight one.
    pub fn new(kind: ManifoldKind, len: usize) -> Self {
        Self {
            kind,
            len,
            weight: 1.0,
            metric: None,
        }
    }

    /// Set the geometry.
    pub fn with_kind(mut self, kind: ManifoldKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set the packed length.
    pub fn with_len(mut self, len: usize) -> Self {
        self.len = len;
        self
    }

    /// Set the metric weight.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Set the tangent metric. A factor that is itself a product takes
    /// its tangent metrics on its own factors.
    pub fn with_metric(mut self, metric: Arc<dyn TangentMetric>) -> Self {
        self.metric = Some(Metric(metric));
        self
    }

    /// The tangent metric, if one is set.
    pub fn metric(&self) -> Option<&Arc<dyn TangentMetric>> {
        self.metric.as_ref().map(|m| &m.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Block {
    kind: ManifoldKind,
    start: usize,
    len: usize,
    /// \(c_i\): multiplier of the factor metric in stored coordinates.
    metric: f64,
    /// \(s_i\): the block stores \(s_i x_i\).
    scale: f64,
    tangent: Option<Metric>,
}

impl Block {
    fn range(&self) -> Range<usize> {
        self.start..self.start + self.len
    }

    fn weight(&self) -> f64 {
        self.metric * self.scale * self.scale
    }

    fn part(&self, a: &Array1<f64>) -> Array1<f64> {
        a.slice(s![self.range()]).to_owned()
    }

    /// The block divided by \(s_i\): a factor point from a stored point,
    /// or a factor tangent from a stored tangent.
    fn unscale(&self, a: &Array1<f64>) -> Array1<f64> {
        let v = a.slice(s![self.range()]);
        if self.scale == 1.0 {
            v.to_owned()
        } else {
            v.mapv(|e| e / self.scale)
        }
    }

    fn rescale(&self, mut a: Array1<f64>) -> Array1<f64> {
        if self.scale != 1.0 {
            a.mapv_inplace(|e| e * self.scale);
        }
        a
    }

    fn over_metric(&self, mut a: Array1<f64>) -> Array1<f64> {
        if self.metric != 1.0 {
            a.mapv_inplace(|e| e / self.metric);
        }
        a
    }

    /// \(G_x v\) at the factor point `x`, or `v` without a tangent metric.
    fn apply_tangent(&self, x: &Array1<f64>, v: Array1<f64>) -> Array1<f64> {
        match &self.tangent {
            Some(m) => self.kind.project(x, &m.0.apply(x, &v)),
            None => v,
        }
    }

    /// \(G_x^{-1} r\) at the factor point `x`, or `r` without a tangent
    /// metric.
    fn solve_tangent(&self, x: &Array1<f64>, r: Array1<f64>) -> Array1<f64> {
        match &self.tangent {
            Some(m) => self.kind.project(x, &m.0.solve(x, &r)),
            None => r,
        }
    }
}

/// Weighted Riemannian product of [`ManifoldKind`] factors, packed end to
/// end. Nested products are flattened; their weights multiply.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductManifold {
    blocks: Vec<Block>,
    len: usize,
}

// Weights and scales are validated finite and positive, never NaN.
impl Eq for ProductManifold {}

impl ProductManifold {
    /// Product of `factors` in packing order.
    ///
    /// Fails with [`Error::ProductFactor`] when there is no factor, a
    /// weight is not finite and positive, a length is zero or does not
    /// fit its kind, or a nested product carries a tangent metric of its
    /// own.
    pub fn new<I>(factors: I) -> Result<Self>
    where
        I: IntoIterator<Item = ProductFactor>,
    {
        let mut blocks = Vec::new();
        let mut len = 0usize;
        for (index, factor) in factors.into_iter().enumerate() {
            let refuse = |what| Error::ProductFactor { index, what };
            if !(factor.weight.is_finite() && factor.weight > 0.0) {
                return Err(refuse("weight must be finite and positive"));
            }
            if factor.len == 0 {
                return Err(refuse("packed length must be positive"));
            }
            if factor.kind.required_dim(factor.len).is_err() {
                return Err(refuse("packed length does not fit the kind"));
            }
            match &factor.kind {
                ManifoldKind::Product(_) if factor.metric.is_some() => {
                    return Err(refuse(
                        "a nested product takes tangent metrics on its factors",
                    ));
                }
                ManifoldKind::Product(inner) => {
                    for b in &inner.blocks {
                        blocks.push(Block {
                            kind: b.kind.clone(),
                            start: len + b.start,
                            len: b.len,
                            metric: factor.weight * b.metric,
                            scale: b.scale,
                            tangent: b.tangent.clone(),
                        });
                    }
                }
                kind => blocks.push(Block {
                    kind: kind.clone(),
                    start: len,
                    len: factor.len,
                    metric: factor.weight,
                    scale: 1.0,
                    tangent: factor.metric.clone(),
                }),
            }
            len = len
                .checked_add(factor.len)
                .ok_or_else(|| refuse("packed length overflows"))?;
        }
        if blocks.is_empty() {
            return Err(Error::ProductFactor {
                index: 0,
                what: "a product needs at least one factor",
            });
        }
        Ok(Self { blocks, len })
    }

    /// manopt `powermanifold`: `count` unit-weight copies of `kind`, each
    /// of packed length `block`.
    pub fn power(kind: ManifoldKind, block: usize, count: usize) -> Result<Self> {
        if count == 0 {
            return Err(Error::ProductFactor {
                index: 0,
                what: "a power needs at least one copy",
            });
        }
        Self::new(std::iter::repeat_n(ProductFactor::new(kind, block), count))
    }

    /// Packed length of a point.
    pub fn packed_len(&self) -> usize {
        self.len
    }

    /// Number of factors after flattening.
    pub fn factor_count(&self) -> usize {
        self.blocks.len()
    }

    /// Factors after flattening, with their weights in the metric and
    /// their tangent metrics.
    pub fn factors(&self) -> Vec<ProductFactor> {
        self.blocks
            .iter()
            .map(|b| ProductFactor {
                metric: b.tangent.clone(),
                ..ProductFactor::new(b.kind.clone(), b.len).with_weight(b.weight())
            })
            .collect()
    }

    /// Index of the first factor that carries a [`TangentMetric`].
    pub fn tangent_metric_factor(&self) -> Option<usize> {
        self.blocks.iter().position(|b| b.tangent.is_some())
    }

    /// Slice of the packed vector that holds factor `index`.
    pub fn range(&self, index: usize) -> Option<Range<usize>> {
        self.blocks.get(index).map(Block::range)
    }

    /// `true` when every factor keeps a straight step on its set.
    pub fn is_flat(&self) -> bool {
        self.blocks.iter().all(|b| b.kind.is_flat())
    }

    /// Factor kinds and their slices, in packed order.
    pub(crate) fn kinds(&self) -> impl Iterator<Item = (&ManifoldKind, Range<usize>)> {
        self.blocks.iter().map(|b| (&b.kind, b.range()))
    }

    /// `true` when a factor is a rigid-motion quotient, whose horizontal
    /// projection needs the session's masses and periodic flag.
    pub(crate) fn has_rigid(&self) -> bool {
        self.kinds()
            .any(|(kind, _)| matches!(kind, ManifoldKind::MwRigid | ManifoldKind::RigidQuotient))
    }

    /// `true` when every factor carries its own metric with weight one in
    /// these coordinates, so the chart is the identity.
    pub fn is_chart(&self) -> bool {
        self.blocks.iter().all(|b| b.metric == 1.0)
    }

    /// Per-coordinate factor \(\sqrt{c_i}\) of the isometric chart: the
    /// point `x` of this product is the point `d * x` of [`Self::chart`],
    /// tangent vectors map the same way, and a Euclidean gradient maps
    /// as `g / d`.
    pub fn chart_scales(&self) -> Array1<f64> {
        let mut d = Array1::ones(self.len);
        for b in &self.blocks {
            d.slice_mut(s![b.range()]).fill(b.metric.sqrt());
        }
        d
    }

    /// The same Riemannian manifold in the coordinates \(z_i =
    /// \sqrt{w_i}\,x_i\), where every factor has weight one and the
    /// metric of an embedded factor without a [`TangentMetric`] is the
    /// Euclidean inner product of the packed vector. Tangent metrics carry
    /// over unchanged, since they act on the factor's own tangent vectors.
    pub fn chart(&self) -> Self {
        let blocks = self
            .blocks
            .iter()
            .map(|b| Block {
                metric: 1.0,
                scale: b.scale * b.metric.sqrt(),
                ..b.clone()
            })
            .collect();
        Self {
            blocks,
            len: self.len,
        }
    }

    fn fits(&self, arrays: &[&Array1<f64>]) -> bool {
        arrays.iter().all(|a| a.len() == self.len)
    }

    /// Assemble the blocks `op` returns. `None` as soon as one is `None`
    /// or has the wrong length.
    fn assemble<F>(&self, mut op: F) -> Option<Array1<f64>>
    where
        F: FnMut(&Block) -> Option<Array1<f64>>,
    {
        let mut out = Array1::zeros(self.len);
        for b in &self.blocks {
            let part = op(b)?;
            if part.len() != b.len {
                return None;
            }
            out.slice_mut(s![b.range()]).assign(&part);
        }
        Some(out)
    }
}

impl Manifold for ProductManifold {
    fn required_dim(&self, n: usize) -> std::result::Result<(), usize> {
        if n == self.len { Ok(()) } else { Err(self.len) }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(&[x, v]) {
            return v.clone();
        }
        self.assemble(|b| Some(b.kind.project(&b.unscale(x), &b.part(v))))
            .unwrap_or_else(|| v.clone())
    }

    /// Blockwise \(c_i^{-1} G^{-1} \mathrm{grad}_i f\), with \(G = I\)
    /// on a factor without a [`TangentMetric`].
    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        if !self.fits(&[x, egrad]) {
            return egrad.clone();
        }
        self.assemble(|b| {
            let at = b.unscale(x);
            let grad = b.kind.egrad2rgrad(&at, &b.part(egrad));
            Some(b.over_metric(b.solve_tangent(&at, grad)))
        })
        .unwrap_or_else(|| egrad.clone())
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(&[x, v]) {
            return x.clone();
        }
        self.assemble(|b| {
            let step = b.kind.retract(&b.unscale(x), &b.unscale(v));
            Some(b.rescale(step))
        })
        .unwrap_or_else(|| x.clone())
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(&[x_from, x_to, v]) {
            return v.clone();
        }
        self.assemble(|b| {
            Some(
                b.kind
                    .transport(&b.unscale(x_from), &b.unscale(x_to), &b.part(v)),
            )
        })
        .unwrap_or_else(|| v.clone())
    }

    /// \(\sum_i c_i\, g_i(u_i, G v_i)\), with \(G = I\) on a factor without
    /// a [`TangentMetric`].
    fn inner(&self, x: &Array1<f64>, u: &Array1<f64>, v: &Array1<f64>) -> f64 {
        if !self.fits(&[x, u, v]) {
            return f64::NAN;
        }
        self.blocks
            .iter()
            .map(|b| {
                let at = b.unscale(x);
                let gv = b.apply_tangent(&at, b.part(v));
                if gv.len() == b.len {
                    b.metric * b.kind.inner(&at, &b.part(u), &gv)
                } else {
                    f64::NAN
                }
            })
            .sum()
    }

    fn norm(&self, x: &Array1<f64>, v: &Array1<f64>) -> f64 {
        let q = self.inner(x, v, v);
        if q.is_nan() { q } else { q.max(0.0).sqrt() }
    }

    /// Blockwise \(c_i^{-1} G^{-1} \mathrm{Hess}_i\) at the factor point,
    /// with the factor gradient \(\nabla_i f / s_i\): the Weingarten term
    /// of the scaled factor \(s_i \mathcal{M}_i\) is \(1/s_i\) times the
    /// unit one. With a [`TangentMetric`] this is the Hessian model
    /// \(G^{-1} \mathrm{Hess}_i\), which differs from the Hessian of the
    /// metric \(g(\cdot, G\,\cdot)\) by a term linear in the gradient.
    fn ehess2rhess(
        &self,
        x: &Array1<f64>,
        egrad: &Array1<f64>,
        ehess: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        if !self.fits(&[x, egrad, ehess, v]) {
            return None;
        }
        self.assemble(|b| {
            let at = b.unscale(x);
            let h = b
                .kind
                .ehess2rhess(&at, &b.unscale(egrad), &b.part(ehess), &b.part(v))?;
            Some(b.over_metric(b.solve_tangent(&at, h)))
        })
    }

    fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
        if !self.fits(&[x, v]) || self.tangent_metric_factor().is_some() {
            return None;
        }
        self.assemble(|b| Some(b.rescale(b.kind.exp(&b.unscale(x), &b.unscale(v))?)))
    }

    fn log(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
        if !self.fits(&[x, y]) || self.tangent_metric_factor().is_some() {
            return None;
        }
        self.assemble(|b| Some(b.rescale(b.kind.log(&b.unscale(x), &b.unscale(y))?)))
    }

    fn dist(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
        if !self.fits(&[x, y]) || self.tangent_metric_factor().is_some() {
            return None;
        }
        let mut sum = 0.0;
        for b in &self.blocks {
            let d = b.kind.dist(&b.unscale(x), &b.unscale(y))?;
            sum += b.weight() * d * d;
        }
        Some(sum.sqrt())
    }

    fn parallel_transport_along(
        &self,
        x: &Array1<f64>,
        d: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        if !self.fits(&[x, d, v]) || self.tangent_metric_factor().is_some() {
            return None;
        }
        self.assemble(|b| {
            b.kind
                .parallel_transport_along(&b.unscale(x), &b.unscale(d), &b.part(v))
        })
    }

    fn parallel_transport(
        &self,
        x_from: &Array1<f64>,
        x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        if !self.fits(&[x_from, x_to, v]) || self.tangent_metric_factor().is_some() {
            return None;
        }
        self.assemble(|b| {
            b.kind
                .parallel_transport(&b.unscale(x_from), &b.unscale(x_to), &b.part(v))
        })
    }

    fn transport_is_isometric(&self) -> bool {
        self.blocks
            .iter()
            .all(|b| b.tangent.is_none() && b.kind.transport_is_isometric())
    }

    fn drift_is_straight(&self) -> bool {
        self.is_flat()
    }

    fn euclidean_normals(&self, x: &Array1<f64>) -> Option<ndarray::Array2<f64>> {
        if x.len() != self.len {
            return None;
        }
        let mut columns = Vec::new();
        for block in &self.blocks {
            let local = block.kind.euclidean_normals(&block.unscale(x))?;
            for index in 0..local.ncols() {
                let mut ambient = Array1::zeros(self.len);
                let column = local.column(index);
                for row in 0..block.len {
                    let packed = if block.scale == 1.0 {
                        column[row]
                    } else {
                        column[row] / block.scale
                    };
                    ambient[block.start + row] = packed;
                }
                let norm = ambient
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if !(norm.is_finite() && norm > 0.0) {
                    return None;
                }
                let inv = 1.0 / norm;
                for value in ambient.iter_mut() {
                    *value *= inv;
                }
                columns.push(ambient);
            }
        }
        let mut out = ndarray::Array2::zeros((self.len, columns.len()));
        for (index, column) in columns.iter().enumerate() {
            out.column_mut(index).assign(column);
        }
        Some(out)
    }

    /// \(\sqrt{\sum_i w_i\,\mathrm{typicaldist}_i^2}\); manopt's at unit
    /// weights. Tangent metrics do not enter it.
    fn typical_dist(&self, _n: usize) -> f64 {
        self.blocks
            .iter()
            .map(|b| {
                let t = b.kind.typical_dist(b.len);
                b.weight() * t * t
            })
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vecops::dot;
    use ndarray::array;

    fn hyperplane(wp: f64, wn: f64) -> ProductManifold {
        ProductManifold::new([
            ProductFactor::new(ManifoldKind::Euclidean, 3).with_weight(wp),
            ProductFactor::new(ManifoldKind::Sphere, 3).with_weight(wn),
        ])
        .unwrap()
    }

    #[test]
    fn inner_is_the_weighted_sum() {
        let m = hyperplane(4.0, 0.25);
        let x = array![1.0, 2.0, 3.0, 0.0, 0.6, 0.8];
        let u = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let v = array![2.0, 1.0, 0.0, 3.0, 0.0, 0.0];
        assert!((m.inner(&x, &u, &v) - (4.0 * 2.0 + 0.25 * 3.0)).abs() < 1e-15);
        assert!((m.norm(&x, &u) - (4.25_f64).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn rgrad_represents_the_differential() {
        let m = hyperplane(4.0, 0.25);
        let x = array![1.0, 2.0, 3.0, 0.0, 0.6, 0.8];
        let g = array![0.3, -1.0, 2.0, 0.5, -0.2, 0.7];
        let r = m.egrad2rgrad(&x, &g);
        for v in [
            array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            array![0.0, 1.0, -1.0, 0.0, 0.8, -0.6],
        ] {
            let t = m.project(&x, &v);
            assert!((m.inner(&x, &r, &t) - dot(g.view(), t.view())).abs() < 1e-14);
        }
        assert!((r[0] - 0.3 / 4.0).abs() < 1e-15);
    }

    #[test]
    fn nested_products_flatten_and_weights_multiply() {
        let inner = ManifoldKind::Product(std::sync::Arc::new(hyperplane(2.0, 3.0)));
        let outer = ProductManifold::new([
            ProductFactor::new(inner, 6).with_weight(5.0),
            ProductFactor::new(ManifoldKind::So3, 9),
        ])
        .unwrap();
        assert_eq!(outer.factor_count(), 3);
        assert_eq!(outer.packed_len(), 15);
        let weights: Vec<f64> = outer.factors().iter().map(|f| f.weight).collect();
        assert_eq!(weights, vec![10.0, 15.0, 1.0]);
        assert_eq!(outer.range(2), Some(6..15));
    }

    #[test]
    fn construction_refuses_bad_factors() {
        let bad_weight = ProductFactor::new(ManifoldKind::Sphere, 3).with_weight(0.0);
        assert!(matches!(
            ProductManifold::new([bad_weight]),
            Err(Error::ProductFactor { index: 0, .. })
        ));
        let bad_len = ProductFactor::new(ManifoldKind::So3, 8);
        assert!(matches!(
            ProductManifold::new([ProductFactor::default(), bad_len]),
            Err(Error::ProductFactor { index: 1, .. })
        ));
        assert!(ProductManifold::new([]).is_err());
        assert!(ProductManifold::power(ManifoldKind::Sphere, 3, 0).is_err());
    }

    #[test]
    fn the_chart_is_an_isometry() {
        let m = hyperplane(9.0, 0.04);
        let c = m.chart();
        assert!(c.is_chart() && !m.is_chart());
        let d = m.chart_scales();
        let x = array![1.0, 2.0, 3.0, 0.0, 0.6, 0.8];
        let u = m.project(&x, &array![0.1, 0.2, -0.3, 0.4, 0.5, -0.6]);
        let v = m.project(&x, &array![-0.2, 0.0, 0.5, 1.0, -0.3, 0.1]);
        let z = &x * &d;
        let (uz, vz) = (&u * &d, &v * &d);
        assert!((m.inner(&x, &u, &v) - dot(uz.view(), vz.view())).abs() < 1e-14);
        assert!((c.inner(&z, &uz, &vz) - m.inner(&x, &u, &v)).abs() < 1e-14);
        let y = m.retract(&x, &u);
        let yz = c.retract(&z, &uz);
        assert!((&yz - &(&y * &d)).mapv(f64::abs).sum() < 1e-14);
        let g = array![0.3, -1.0, 2.0, 0.5, -0.2, 0.7];
        let rz = c.egrad2rgrad(&z, &(&g / &d));
        let r = m.egrad2rgrad(&x, &g);
        assert!((&rz - &(&r * &d)).mapv(f64::abs).sum() < 1e-14);
        let dz = c.dist(&z, &yz).unwrap();
        assert!((dz - m.dist(&x, &y).unwrap()).abs() < 1e-14);
    }

    /// \(G = \mathrm{diag}(1, 2, 4)\) on a Euclidean factor.
    struct Diagonal;

    impl TangentMetric for Diagonal {
        fn apply(&self, _x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
            v * &array![1.0, 2.0, 4.0]
        }

        fn solve(&self, _x: &Array1<f64>, r: &Array1<f64>) -> Array1<f64> {
            r / &array![1.0, 2.0, 4.0]
        }
    }

    /// \(G_x = I + b b^\top\) on \(T_x S^2\), \(b = P_x a\), inverted by
    /// Sherman–Morrison.
    struct RankOne;

    impl RankOne {
        fn b(x: &Array1<f64>) -> Array1<f64> {
            ManifoldKind::Sphere.project(x, &array![1.0, -2.0, 0.5])
        }
    }

    impl TangentMetric for RankOne {
        fn apply(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
            let b = Self::b(x);
            v + &(&b * b.dot(v))
        }

        fn solve(&self, x: &Array1<f64>, r: &Array1<f64>) -> Array1<f64> {
            let b = Self::b(x);
            r - &(&b * (b.dot(r) / (1.0 + b.dot(&b))))
        }
    }

    fn preconditioned() -> ProductManifold {
        ProductManifold::new([
            ProductFactor::new(ManifoldKind::Euclidean, 3)
                .with_weight(4.0)
                .with_metric(Arc::new(Diagonal)),
            ProductFactor::new(ManifoldKind::Sphere, 3)
                .with_weight(0.25)
                .with_metric(Arc::new(RankOne)),
        ])
        .unwrap()
    }

    #[test]
    fn a_tangent_metric_keeps_the_gradient_the_riesz_vector() {
        let m = preconditioned();
        let x = array![1.0, 2.0, 3.0, 0.0, 0.6, 0.8];
        let g = array![0.3, -1.0, 2.0, 0.5, -0.2, 0.7];
        let r = m.egrad2rgrad(&x, &g);
        let u = m.project(&x, &array![0.4, -0.1, 0.2, 1.0, 0.3, -0.5]);
        let v = m.project(&x, &array![-0.2, 0.7, 0.1, -0.3, 0.9, 0.2]);
        for t in [&u, &v] {
            assert!((m.inner(&x, &r, t) - dot(g.view(), t.view())).abs() < 1e-14);
        }
        assert!((m.inner(&x, &u, &v) - m.inner(&x, &v, &u)).abs() < 1e-14);
        assert!((r[2] - 2.0 / 16.0).abs() < 1e-15);
        let ehess = |w: &Array1<f64>| w * &array![1.0, 3.0, -2.0, 2.0, -1.0, 5.0];
        let hu = m.ehess2rhess(&x, &g, &ehess(&u), &u).unwrap();
        let hv = m.ehess2rhess(&x, &g, &ehess(&v), &v).unwrap();
        assert!((m.inner(&x, &hu, &v) - m.inner(&x, &u, &hv)).abs() < 1e-14);
    }

    #[test]
    fn a_tangent_metric_drops_the_geodesic_operations() {
        let m = preconditioned();
        let x = array![1.0, 2.0, 3.0, 0.0, 0.6, 0.8];
        let v = m.project(&x, &array![0.1, 0.0, 0.0, 0.2, 0.0, 0.0]);
        let y = m.retract(&x, &v);
        assert_eq!(m.tangent_metric_factor(), Some(0));
        assert!(m.exp(&x, &v).is_none() && m.log(&x, &y).is_none());
        assert!(m.dist(&x, &y).is_none());
        assert!(m.parallel_transport_along(&x, &v, &v).is_none());
        assert!(!m.transport_is_isometric());
        assert_eq!(hyperplane(1.0, 1.0).tangent_metric_factor(), None);
    }

    #[test]
    fn nested_products_keep_their_tangent_metrics() {
        let nested = preconditioned();
        let inner = ManifoldKind::Product(Arc::new(nested.clone()));
        let outer = ProductManifold::new([
            ProductFactor::new(ManifoldKind::So3, 9),
            ProductFactor::new(inner.clone(), 6),
        ])
        .unwrap();
        assert_eq!(outer.tangent_metric_factor(), Some(1));
        let factors = outer.factors();
        let original = nested.factors();
        assert!(factors[0].metric().is_none());
        assert!(Arc::ptr_eq(
            factors[1].metric().unwrap(),
            original[0].metric().unwrap()
        ));
        assert_ne!(
            original[0],
            ProductFactor::new(ManifoldKind::Euclidean, 3).with_weight(4.0)
        );
        let wrapped = ProductFactor::new(inner, 6).with_metric(Arc::new(RankOne));
        assert!(matches!(
            ProductManifold::new([wrapped]),
            Err(Error::ProductFactor { index: 0, .. })
        ));
    }

    #[test]
    fn power_is_a_unit_weight_product() {
        let p = ProductManifold::power(ManifoldKind::Sphere, 3, 4).unwrap();
        assert_eq!(p.packed_len(), 12);
        assert!(p.is_chart());
        assert!(p.factors().iter().all(|f| f.weight == 1.0 && f.len == 3));
        assert!((p.typical_dist(12) - 2.0 * std::f64::consts::PI).abs() < 1e-12);
    }
}
