//! Mass-weighted \(R^{3N}/\mathrm{SE}(3)\) (Eckart / Page–McIver).
//!
//! This manifold holds `q_i = sqrt(m_i) x_i`, not raw Cartesians.
//! Sella IRC and gpr_optim `IRCDriver` use that chart
//! (`x_mw = sqrt(m) x`; Page and McIver, J. Chem. Phys. 88, 922 (1988),
//! doi:10.1063/1.454172; Ishida, Morokuma, Komornicki 1977,
//! doi:10.1063/1.434152). In `q` the Eckart product is Euclidean, so
//! the unit-mass projector below is the right one. Do not also turn on
//! the Cartesian mass metric: that is the other switch, for a velocity
//! stored as `x`.

use ndarray::Array1;

use crate::rigid::project_out_rot_trans;

use super::Manifold;

/// Eckart frame in `q = sqrt(m) x`. Unit mass is the metric of that chart.
#[derive(Clone, Copy, Debug, Default)]
pub struct MwRigid;

impl Manifold for MwRigid {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n >= 3 && n.is_multiple_of(3) {
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
