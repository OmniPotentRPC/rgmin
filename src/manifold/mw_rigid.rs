//! Mass-weighted \(R^{3N}/\mathrm{SE}(3)\) (Eckart / Page–McIver).
//!
//! Sella IRC and gpr_optim `IRCDriver` work in mass-weighted Cartesians
//! (`x_mw = sqrt(m) x`; Page and McIver, J. Chem. Phys. 88, 922 (1988),
//! doi:10.1063/1.454172; Ishida, Morokuma, Komornicki 1977,
//! doi:10.1063/1.434152). The tangent is the mass-weighted kernel of
//! translations and infinitesimal rotations.
//!
//! Per-atom masses live on this value. [`Self::new`] is unit mass and
//! matches [`super::RigidQuotient`]. [`crate::Solver::set_masses`]
//! writes the same masses onto a session's `MwRigid` kind.

use std::sync::Arc;

use ndarray::Array1;

use crate::rigid::project_horizontal;

use super::Manifold;

/// Eckart frame. One mass per atom; [`Self::new`] is unit mass.
#[derive(Clone, Debug, Default)]
pub struct MwRigid {
    pub(crate) masses: Option<Arc<[f64]>>,
}

impl MwRigid {
    /// Unit mass. The projection matches [`super::RigidQuotient`].
    pub fn new() -> Self {
        Self::default()
    }

    /// One mass per atom. An empty slice is unit mass.
    pub fn with_masses(masses: &[f64]) -> Self {
        Self {
            masses: if masses.is_empty() {
                None
            } else {
                Some(Arc::from(masses))
            },
        }
    }

    pub(crate) fn masses(&self) -> Option<&[f64]> {
        self.masses.as_deref()
    }
}

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
        // A stored count that is not one mass per atom leaves `v` as it is.
        let _ = project_horizontal(&mut w, x.view(), self.masses(), true);
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
        let geom = MwRigid::new();
        let t = geom.project(&x, &v);
        let y = geom.retract(&x, &t);
        assert_eq!(y.len(), 9);
        let mut inc = Vector::from_host(y.clone());
        vecops::vaxpy(-1.0, &Vector::from_host(x.clone()), &mut inc);
        let inc = inc.into_host();
        let re = geom.project(&x, &inc);
        for (a, b) in inc.iter().zip(re.iter()) {
            assert!((a - b).abs() < 1e-12, "{inc:?} vs {re:?}");
        }
        let trans = array![0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0];
        let p = geom.project(&y, &trans);
        assert!(vecops::nrm2(p.view()) < 1e-12, "{p:?}");
    }

    #[test]
    fn wrong_mass_count_does_not_use_unit_mass() {
        let x = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
        let v = array![0.3, 0.1, 0.0, -0.2, 0.4, 0.0, 0.1, -0.2, 0.05];
        let projected = MwRigid::with_masses(&[12.0, 1.0]).project(&x, &v);
        for (a, b) in projected.iter().zip(v.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "a short mass table changed the vector"
            );
        }
        let unit = MwRigid::new().project(&x, &v);
        assert!(
            vecops::nrm2((&unit - &v).view()) > 0.1,
            "unit mass did not move this vector"
        );
    }

    #[cfg(feature = "par")]
    #[test]
    fn par_retract_stays_on_the_set() {
        retract_stays_on_the_set();
    }
}
