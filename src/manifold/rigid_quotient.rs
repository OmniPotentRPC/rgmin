//! Isolated-molecule shape space \(R^{3N}/\mathrm{SE}(3)\).
//!
//! Sella Cartesian PES: `fix_translation` plus `fix_rotation` on the
//! whole cluster (Hermes, Sarsfield, Zádor, JCTC 2022,
//! doi:10.1021/acs.jctc.2c00395). The tangent is the horizontal space
//! of eOn `projectOutRotTrans`. The retraction is the horizontal lift
//! `x + v`; energy is SE(3)-invariant so a rigid increment is a no-op.
//!
//! This is the molecular manifold for OptBench cluster minimization.
//! \(S^{3N-1}\) and a lone SO(3) 9-vector are a different geometry.

use ndarray::Array1;

use crate::rigid::project_out_rot_trans;

use super::Manifold;

/// Horizontal lift of \(R^{3N}/\mathrm{SE}(3)\). Unit-mass Eckart.
#[derive(Clone, Copy, Debug, Default)]
pub struct RigidQuotient;

impl Manifold for RigidQuotient {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn translation_is_vertical() {
        let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let v = array![0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0];
        let t = RigidQuotient.project(&x, &v);
        let n = t.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(n < 1e-12, "{t:?}");
    }

    #[test]
    fn keeps_length() {
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = RigidQuotient.retract(&x, &v);
        assert_eq!(y.len(), 114);
    }
}
