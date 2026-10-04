//! Ambient Euclidean space. Identity projection and retraction.

use ndarray::Array1;

use super::Manifold;

/// Unconstrained Euclidean geometry.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

impl Manifold for Euclidean {
    fn project(&self, _x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        v.clone()
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

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        v.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_is_translation() {
        let x = array![1.0, 2.0];
        let v = array![0.5, -1.0];
        let y = Euclidean.retract(&x, &v);
        assert!((y[0] - 1.5).abs() < 1e-15);
        assert!((y[1] - 1.0).abs() < 1e-15);
    }
    #[test]
    fn retract_stays_on_the_euclidean_set() {
        let x = array![-2.0, 0.5, 3.0];
        let v = array![0.25, -1.0, 0.0];
        let y = Euclidean.retract(&x, &v);
        assert_eq!(y.len(), 3);
        assert!((y[0] + 1.75).abs() < 1e-15);
        assert!((y[1] + 0.5).abs() < 1e-15);
        assert!((y[2] - 3.0).abs() < 1e-15);
        let t = Euclidean.transport(&x, &y, &v);
        assert_eq!(t, v);
    }

    #[cfg(feature = "par")]
    #[test]
    fn par_retract_stays_on_the_euclidean_set() {
        retract_stays_on_the_euclidean_set();
    }

}
