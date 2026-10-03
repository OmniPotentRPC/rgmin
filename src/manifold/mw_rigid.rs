//! Mass-weighted \(R^{3N}/\mathrm{SE}(3)\) (Eckart / Page–McIver).
//!
//! Sella IRC and gpr_optim `IRCDriver` work in mass-weighted Cartesians
//! (`x_mw = sqrt(m) x`; Page and McIver, J. Chem. Phys. 88, 922 (1988),
//! doi:10.1063/1.454172; Ishida, Morokuma, Komornicki 1977,
//! doi:10.1063/1.434152). The tangent is the mass-weighted kernel of
//! translations and infinitesimal rotations.
//!
//! Per-atom masses live on the session (`Solver::set_masses`). This
//! type uses unit mass so it matches [`super::RigidQuotient`] until
//! the session applies the metric.

use ndarray::Array1;

use crate::rigid::project_out_rot_trans;

use super::Manifold;

/// Eckart frame. Unit mass here; the session supplies real masses.
#[derive(Clone, Copy, Debug, Default)]
pub struct MwRigid;

impl Manifold for MwRigid {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n >= 6 && n.is_multiple_of(3) {
            Ok(())
        } else {
            Err(n)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut w = v.clone();
        project_out_rot_trans(&mut w, x.view());
        w
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        x + v
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}
