//! Sella `IRCTrustRegion` / Gonzalez--Schlegel MW sphere.
//!
//! Inner IRC step: \(\|(s + d_1)\odot\sqrt{m}\| = dx\).
//! \(d_1\) is the accumulated displacement from the last accepted
//! point. This is not [`crate::manifold::Sphere`] (unit \(S^{n-1}\)
//! about the origin).

use ndarray::Array1;

use crate::vecops::nrm2;

/// Per-atom masses (length N) to a 3N \(\sqrt{m}\) weight.
pub fn sqrt_masses_3n(masses: &[f64]) -> Array1<f64> {
    let mut out = Array1::zeros(masses.len() * 3);
    for (i, &m) in masses.iter().enumerate() {
        let s = m.max(0.0).sqrt();
        out[3 * i] = s;
        out[3 * i + 1] = s;
        out[3 * i + 2] = s;
    }
    out
}

/// Restricted step onto \(\|(s+d_1)\odot\sqrt{m}\| = dx\).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct IrcTrust {
    /// Displacement already taken from the last accepted IRC point.
    pub d1: Array1<f64>,
    /// Repeated \(\sqrt{m}\) weights, length 3N.
    pub sqrtm: Array1<f64>,
    /// Mass-weighted sphere radius.
    pub dx: f64,
}

with_setters!(IrcTrust {
    with_d1 => d1: val Array1<f64>,
    with_sqrtm => sqrtm: val Array1<f64>,
    with_dx => dx: val f64,
});

/// Smallest mass weight treated as present.
///
/// The degenerate axis and the radial divisor both keep a weight at
/// least this large. A lighter weight is not an axis, and the divisor
/// uses this floor.
const MASS_WEIGHT_FLOOR: f64 = 1e-16;

fn weight_is_present(weight: f64) -> bool {
    weight >= MASS_WEIGHT_FLOOR
}

impl IrcTrust {
    /// Build from the displacement `d1`, the length-3N square-root masses
    /// `sqrtm`, and the mass-weighted radius `dx`, each taken as given.
    pub fn new(d1: Array1<f64>, sqrtm: Array1<f64>, dx: f64) -> Self {
        Self { d1, sqrtm, dx }
    }

    /// Build from per-atom masses (length N).
    pub fn from_atom_masses(d1: Array1<f64>, masses: &[f64], dx: f64) -> Self {
        Self {
            d1,
            sqrtm: sqrt_masses_3n(masses),
            dx: dx.max(0.0),
        }
    }

    /// \(\|(s + d_1)\odot\sqrt{m}\|\).
    pub fn cons(&self, s: &Array1<f64>) -> f64 {
        let n = s.len().min(self.d1.len()).min(self.sqrtm.len());
        if n == 0 {
            return 0.0;
        }
        let mut acc = 0.0;
        for i in 0..n {
            let t = (s[i] + self.d1[i]) * self.sqrtm[i];
            acc += t * t;
        }
        acc.sqrt()
    }

    /// Radial projection of `s` onto the MW sphere of radius `dx`.
    pub fn project(&self, s: &Array1<f64>) -> Array1<f64> {
        let n = s.len().min(self.d1.len()).min(self.sqrtm.len());
        let mut out = s.clone();
        if n == 0 {
            return out;
        }
        let mut w = Array1::zeros(n);
        for i in 0..n {
            w[i] = (s[i] + self.d1[i]) * self.sqrtm[i];
        }
        let norm = nrm2(w.view());
        if norm <= MASS_WEIGHT_FLOOR {
            // Degenerate: sit on the first represented weighted axis.
            let axis = (0..n).find(|&i| weight_is_present(self.sqrtm[i]));
            if axis.is_none() || self.dx == 0.0 {
                for i in 0..n {
                    out[i] = -self.d1[i];
                }
                return out;
            }
            for i in 0..n {
                out[i] = -self.d1[i];
            }
            if let Some(axis) = axis {
                out[axis] = self.dx / self.sqrtm[axis] - self.d1[axis];
            }
            return out;
        }
        let scale = self.dx / norm;
        for i in 0..n {
            let wi = w[i] * scale;
            let sm = if weight_is_present(self.sqrtm[i]) {
                self.sqrtm[i]
            } else {
                MASS_WEIGHT_FLOOR
            };
            out[i] = wi / sm - self.d1[i];
        }
        out
    }

    /// True when `cons(s)` sits on `dx` to `tol`.
    pub fn on_bound(&self, s: &Array1<f64>, tol: f64) -> bool {
        (self.cons(s) - self.dx).abs() <= tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn unequal_masses_sit_on_the_mw_sphere() {
        let masses = [1.0, 16.0];
        let d1 = array![0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let tr = IrcTrust::from_atom_masses(d1, &masses, 0.2);
        let s = array![0.5, 0.1, -0.2, 0.3, 0.0, 0.4];
        let p = tr.project(&s);
        assert!(tr.on_bound(&p, 1e-12), "cons={} dx={}", tr.cons(&p), tr.dx);
    }

    #[test]
    fn already_on_sphere_is_a_fixed_point() {
        let masses = [12.0, 1.0];
        let d1 = Array1::zeros(6);
        let tr = IrcTrust::from_atom_masses(d1, &masses, 0.15);
        let mut s = array![0.15, 0.0, 0.0, 0.0, 0.0, 0.0];
        s = tr.project(&s);
        let again = tr.project(&s);
        for (a, b) in s.iter().zip(again.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    /// A weight equal to the floor is present, because that is the
    /// smallest divisor the radial scale will use. The degenerate
    /// projection therefore puts the radius on that axis
    /// (`dx / weight`). A lighter weight is not an axis, and the
    /// radial scale divides by the floor instead. Both sites ask
    /// whether the weight is at least `MASS_WEIGHT_FLOOR`.
    #[test]
    fn mass_floor_treats_its_own_value_as_present() {
        let floor = MASS_WEIGHT_FLOOR;
        let on_floor = IrcTrust::new(array![0.0, 0.0], array![0.0, floor], floor);
        let placed = on_floor.project(&array![0.0, 0.0]);
        assert_eq!(placed, array![0.0, 1.0], "a floor weight is an axis");

        let light = floor * 0.5;
        let absent = IrcTrust::new(array![0.0, 1.0], array![0.0, light], 4.0);
        assert_eq!(absent.project(&array![0.0, -1.0]), array![0.0, -1.0]);

        let scaled = IrcTrust::new(array![0.0, 0.0], array![light, 1.0], 2.0);
        let radial = scaled.project(&array![1.0, 1.0]);
        let norm = (light * light + 1.0).sqrt();
        let scale = 2.0 / norm;
        assert!(
            (radial[0] - light * scale / floor).abs() < 1e-12,
            "{radial:?}"
        );
        assert!((radial[1] - scale).abs() < 1e-12, "{radial:?}");
    }
}
