//! Weighted products step in their metric chart.
//!
//! A session stores a [`ManifoldKind::Product`] with weights other than
//! one as [`ProductManifold::chart`]: the same manifold in the
//! coordinates \(z = d \odot x\), with \(d = \sqrt{w_i}\) on factor
//! \(i\). There the product metric of embedded factors is the Euclidean
//! inner product of the packed vector, so the steps, line searches, trust
//! regions, secant pairs and stopping tests, which are written for that
//! inner product, measure in the product metric unchanged, and an
//! iterate does not depend on the units of a factor once its weight
//! carries them. The caller keeps natural coordinates: points and tangent
//! vectors map as \(z = d \odot x\), Euclidean gradients as
//! \(\nabla_z f = \nabla_x f \oslash d\) and Hessians as \(D^{-1}
//! \nabla^2_x f\, D^{-1}\).

use std::sync::Arc;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1};

use super::Solver;
use crate::error::{Error, Result};
use crate::hvp::HessianVector;
use crate::manifold::{ManifoldKind, ProductManifold};
use crate::newton::HessianObjective;
use crate::report::Report;

/// `obj` read in the chart `z = d * x`.
pub(super) struct Charted<'a, O: ?Sized> {
    obj: &'a O,
    d: &'a Array1<f64>,
    bounds: Bounds<f64>,
}

impl<'a, O: Objective<f64> + ?Sized> Charted<'a, O> {
    fn new(obj: &'a O, d: &'a Array1<f64>) -> Self {
        let b = obj.bounds();
        let bounds = Bounds::new(&b.low * d, &b.high * d, b.slack);
        Self { obj, d, bounds }
    }

    fn natural(&self, z: ArrayView1<f64>) -> Array1<f64> {
        &z / self.d
    }
}

impl<O: Objective<f64> + ?Sized> Objective<f64> for Charted<'_, O> {
    fn dim(&self) -> usize {
        self.obj.dim()
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, z: ArrayView1<f64>) -> f64 {
        self.obj.eval(self.natural(z).view())
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> Gradient<f64> for Charted<'_, O> {
    fn dim(&self) -> usize {
        Objective::dim(self.obj)
    }

    fn grad(&self, z: ArrayView1<f64>) -> Array1<f64> {
        Gradient::grad(self.obj, self.natural(z).view()) / self.d
    }
}

impl<O: DifferentiableObjective<f64> + ?Sized> DifferentiableObjective<f64> for Charted<'_, O> {
    fn value_and_gradient(&self, z: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let (value, grad) = self.obj.value_and_gradient(self.natural(z).view());
        (value, grad / self.d)
    }
}

impl<O: HessianObjective + ?Sized> HessianObjective for Charted<'_, O> {
    fn hessian(&self, z: ArrayView1<f64>) -> Array2<f64> {
        let mut h = self.obj.hessian(self.natural(z).view());
        for ((i, j), e) in h.indexed_iter_mut() {
            *e /= self.d[i] * self.d[j];
        }
        h
    }
}

impl<O: HessianVector + ?Sized> HessianVector for Charted<'_, O> {
    fn hessian_vector(&self, z: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let hv = self
            .obj
            .hessian_vector(self.natural(z).view(), (&v / self.d).view());
        hv / self.d
    }
}

/// Coordinates and the horizontal gradient, a tangent vector, pulled back
/// from the chart. The gradient norm is already its product-metric length.
fn report_to_natural(mut report: Report, d: &Array1<f64>) -> Report {
    if report.coords.len() == d.len() {
        report.coords /= d;
    }
    if let Some(g) = report.grad.as_mut().filter(|g| g.len() == d.len()) {
        *g /= d;
    }
    report
}

/// The kind a session stores for `kind`, and the chart scales when it is
/// a product whose metric is not already the factors' own.
pub(super) fn chart_form(kind: ManifoldKind) -> (ManifoldKind, Option<Arc<Array1<f64>>>) {
    match &kind {
        ManifoldKind::Product(p) if !p.is_chart() => (
            ManifoldKind::Product(Arc::new(p.chart())),
            Some(Arc::new(p.chart_scales())),
        ),
        _ => (kind, None),
    }
}

impl Solver {
    /// Run `step` with `obj` and `x` read in the chart `z = d * x`, then
    /// write the natural point back into `x`. A point handed back is
    /// recognized on the next call, so the chart point and the cached
    /// evaluation survive the round trip bit for bit.
    pub(super) fn in_chart<O, F>(
        &mut self,
        d: &Arc<Array1<f64>>,
        obj: &O,
        x: &mut Array1<f64>,
        step: F,
    ) -> Result<Report>
    where
        O: Objective<f64> + ?Sized,
        F: FnOnce(&mut Self, &Charted<'_, O>, &mut Array1<f64>) -> Result<Report>,
    {
        if x.len() != self.dim || self.dim != Objective::dim(obj) {
            return Err(Error::Dim {
                got: x.len(),
                dim: self.dim,
            });
        }
        self.check_manifold(x.len())?;
        self.refuse_in_chart()?;
        let charted = Charted::new(obj, d);
        let mut z = match &self.chart_seen {
            Some((seen, z)) if seen == x => z.clone(),
            _ => &*x * &**d,
        };
        let out = step(self, &charted, &mut z);
        *x = &z / &**d;
        self.chart_seen = Some((x.clone(), z));
        match out {
            Ok(report) => Ok(report_to_natural(report, d)),
            Err(Error::StopUnmet { reason, report }) => Err(Error::StopUnmet {
                reason,
                report: Box::new(report_to_natural(*report, d)),
            }),
            Err(Error::NotConverged(report)) => {
                Err(Error::NotConverged(Box::new(report_to_natural(*report, d))))
            }
            Err(err) => Err(err),
        }
    }

    /// Settings that read raw coordinates as atoms or boxes, which the
    /// chart rescales.
    fn refuse_in_chart(&self) -> Result<()> {
        let what = if self.exp.is_some() {
            Some("the exponential preconditioner")
        } else if self.lindh_spec.is_some() {
            Some("the Lindh model Hessian")
        } else if self.project_rigid {
            Some("rigid-motion projection of the whole vector")
        } else {
            None
        };
        #[cfg(feature = "highs")]
        let what = what.or_else(|| {
            (self.highs
                && (self.box_lo.is_some() || self.box_hi.is_some() || !self.equalities.is_empty()))
            .then_some("the HiGHS box and equalities")
        });
        what.map_or(Ok(()), |what| Err(Error::ProductChart { what }))
    }

    /// Eckart projection of the rigid factors of `product`, with the
    /// session masses and periodic flag, written over `out`. MwRigid
    /// blocks read [`Self::set_masses`] as one molecule reused on every
    /// block or as the concatenation over the MwRigid blocks.
    pub(super) fn project_rigid_blocks(
        &self,
        product: &ProductManifold,
        at: &Array1<f64>,
        v: &Array1<f64>,
        out: &mut Array1<f64>,
    ) {
        let weighted_atoms: usize = product
            .kinds()
            .filter(|(kind, _)| matches!(kind, ManifoldKind::MwRigid))
            .map(|(_, range)| range.len() / 3)
            .sum();
        let stored = self.masses.as_ref().and_then(|m| m.as_slice());
        let mut seen = 0;
        for (kind, range) in product.kinds() {
            let atoms = range.len() / 3;
            let masses = match kind {
                ManifoldKind::MwRigid => {
                    let start = seen;
                    seen += atoms;
                    match stored {
                        Some(m) if m.len() == atoms => Some(m),
                        Some(m) if m.len() == weighted_atoms => Some(&m[start..start + atoms]),
                        _ => None,
                    }
                }
                ManifoldKind::RigidQuotient => None,
                _ => continue,
            };
            let mut block = v.slice(ndarray::s![range.clone()]).to_owned();
            crate::rigid::project_horizontal(
                &mut block,
                at.slice(ndarray::s![range.clone()]),
                masses,
                !self.periodic,
            );
            out.slice_mut(ndarray::s![range]).assign(&block);
        }
    }
}
