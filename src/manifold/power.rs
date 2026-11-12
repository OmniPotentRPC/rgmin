//! Cartesian power of one embedded manifold.
//!
//! manopt `powermanifold`: `count` copies of one factor, packed as
//! contiguous blocks of length `block`. Projection, retraction,
//! transport, and the Euclidean-to-Riemannian gradient act on each
//! block with the factor and leave the other blocks alone. A product
//! of spheres is [`super::Oblique`]; this type is the product for the
//! factors that are not already a product, in particular a ring of
//! mass-weighted Eckart frames ([`super::MwRigid`]).
//!
//! Absil, Mahony, and Sepulchre, *Optimization Algorithms on Matrix
//! Manifolds*, Princeton (2008), doi:10.1515/9781400830244.
//! Boumal, *An Introduction to Optimization on Smooth Manifolds*,
//! Cambridge (2023), doi:10.1017/9781009166164.

use ndarray::{Array1, ArrayView1, s};

use super::Manifold;

/// `factor^count`, packed as `count` blocks of length `block`.
#[derive(Clone, Copy, Debug)]
pub struct Power<M> {
    /// Geometry of one block.
    pub factor: M,
    /// Packed length of one block.
    pub block: usize,
    /// Number of blocks.
    pub count: usize,
}

impl<M> Power<M> {
    /// `count` copies of `factor`, each of packed length `block`.
    pub fn new(factor: M, block: usize, count: usize) -> Self {
        Self {
            factor,
            block,
            count,
        }
    }

    /// Packed length, or `None` on overflow or a zero size.
    pub fn packed_len(&self) -> Option<usize> {
        if self.block == 0 || self.count == 0 {
            return None;
        }
        self.block.checked_mul(self.count)
    }

    fn block_view<'a>(&self, x: &'a Array1<f64>, i: usize) -> ArrayView1<'a, f64> {
        let a = i * self.block;
        x.slice(s![a..a + self.block])
    }
}

impl<M: Manifold> Manifold for Power<M> {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if self.packed_len() != Some(n) {
            return Err(n);
        }
        self.factor.required_dim(self.block).map_err(|_| n)
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(x.len().max(v.len()));
        if self.packed_len() != Some(x.len()) || v.len() != x.len() {
            return out;
        }
        for i in 0..self.count {
            let xb = self.block_view(x, i).to_owned();
            let vb = self.block_view(v, i).to_owned();
            let pb = self.factor.project(&xb, &vb);
            let a = i * self.block;
            out.slice_mut(s![a..a + self.block]).assign(&pb);
        }
        out
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(x.len().max(egrad.len()));
        if self.packed_len() != Some(x.len()) || egrad.len() != x.len() {
            return out;
        }
        for i in 0..self.count {
            let xb = self.block_view(x, i).to_owned();
            let gb = self.block_view(egrad, i).to_owned();
            let rb = self.factor.egrad2rgrad(&xb, &gb);
            let a = i * self.block;
            out.slice_mut(s![a..a + self.block]).assign(&rb);
        }
        out
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut out = x.clone();
        if self.packed_len() != Some(x.len()) || v.len() != x.len() {
            return out;
        }
        for i in 0..self.count {
            let xb = self.block_view(x, i).to_owned();
            let vb = self.block_view(v, i).to_owned();
            let yb = self.factor.retract(&xb, &vb);
            let a = i * self.block;
            out.slice_mut(s![a..a + self.block]).assign(&yb);
        }
        out
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(v.len());
        if self.packed_len() != Some(x_from.len())
            || x_to.len() != x_from.len()
            || v.len() != x_from.len()
        {
            return out;
        }
        for i in 0..self.count {
            let a = i * self.block;
            let from = self.block_view(x_from, i).to_owned();
            let to = self.block_view(x_to, i).to_owned();
            let vb = self.block_view(v, i).to_owned();
            let tb = self.factor.transport(&from, &to, &vb);
            out.slice_mut(s![a..a + self.block]).assign(&tb);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{Euclidean, MwRigid, Sphere};
    use crate::vecops::{dot, nrm2};
    use ndarray::array;

    #[test]
    fn sphere_power_is_columnwise_and_retracts_to_the_product() {
        let power = Power::new(Sphere, 3, 2);
        let mut x = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let v = array![0.0, 0.2, 0.0, 0.3, 0.0, 0.0];
        let p = power.project(&x, &v);
        assert!(p[0].abs() < 1e-12, "tangent to the first sphere");
        assert!((p[1] - 0.2).abs() < 1e-12);
        assert!(p[4].abs() < 1e-12, "tangent to the second sphere");
        let y = power.retract(&x, &v);
        let n0 = nrm2(y.slice(s![0..3]));
        let n1 = nrm2(y.slice(s![3..6]));
        assert!((n0 - 1.0).abs() < 1e-12);
        assert!((n1 - 1.0).abs() < 1e-12);
        x[0] = 2.0;
        assert!(power.required_dim(6).is_ok());
        assert!(power.required_dim(5).is_err());
    }

    #[test]
    fn euclidean_power_is_the_ambient_product() {
        let power = Power::new(Euclidean, 2, 3);
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = array![0.5, -0.5, 1.0, 0.0, -1.0, 2.0];
        let y = power.retract(&x, &v);
        assert!((y[0] - 1.5).abs() < 1e-15);
        assert!((y[5] - 8.0).abs() < 1e-15);
        let g = power.egrad2rgrad(&x, &v);
        assert!((dot(g.view(), v.view()) - dot(v.view(), v.view())).abs() < 1e-15);
    }

    #[test]
    fn mw_rigid_power_kills_a_translation_of_one_block_only() {
        // Two diatomics, each block length 6. A pure translation of the
        // first molecule is horizontal-killed; a bond stretch is not.
        let power = Power::new(MwRigid, 6, 2);
        let x = array![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
        ];
        let shift = array![0.2, -0.1, 0.05, 0.2, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let p = power.project(&x, &shift);
        let n_first = nrm2(p.slice(s![0..6]));
        assert!(
            n_first < 1e-10,
            "translation of one molecule is quotiented, {n_first}"
        );
        let stretch = array![0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let s = power.project(&x, &stretch);
        assert!(
            nrm2(s.slice(s![0..6])) > 0.05,
            "a bond stretch stays in the tangent space"
        );
        let y = power.retract(&x, &stretch);
        let transported = power.transport(&x, &y, &s);
        let again = power.project(&y, &transported);
        let gap_v = &transported - &again;
        let gap = nrm2(gap_v.view());
        assert!(gap < 1e-10, "transport lands in the tangent space, {gap}");
    }
}
