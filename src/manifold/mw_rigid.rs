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
        #[cfg(feature = "par")]
        if x.len() >= crate::vecops::PAR_MIN_LEN && x.len() == v.len() {
            let mut y = x.clone();
            crate::vecops::axpy(1.0, v.view(), &mut y);
            return y;
        }
        x + v
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vecops::{self, Vector};
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_set() {
        let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let v = array![0.0, 0.1, 0.0, 0.0, -0.05, 0.05, 0.0, -0.05, -0.05];
        let t = MwRigid.project(&x, &v);
        let y = MwRigid.retract(&x, &t);
        assert_eq!(y.len(), 9);
        let mut inc = Vector::from_host(y.clone());
        vecops::vaxpy(-1.0, &Vector::from_host(x.clone()), &mut inc);
        let inc = inc.into_host();
        let re = MwRigid.project(&x, &inc);
        for (a, b) in inc.iter().zip(re.iter()) {
            assert!((a - b).abs() < 1e-12, "{inc:?} vs {re:?}");
        }
        let trans = array![0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0];
        let p = MwRigid.project(&y, &trans);
        assert!(vecops::nrm2(p.view()) < 1e-12, "{p:?}");
    }

    #[cfg(feature = "par")]
    #[test]
    fn par_retract_stays_on_the_set() {
        retract_stays_on_the_set();
    }
}
