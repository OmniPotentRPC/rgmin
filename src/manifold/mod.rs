//! Embedded Riemannian manifolds (manopt_cpp / ROPTLIB waist).
//!
//! Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix
//! Manifolds*, <https://doi.org/10.1515/9781400830244>.
//! Boumal, *An Introduction to Optimization on Smooth Manifolds*,
//! <https://doi.org/10.1017/9781009166164>.
//! manopt_cpp: `proj`, `retr`, `transp` on an embedded Euclidean vector.
//!
//! Isolated molecules and clusters use [`ManifoldKind::RigidQuotient`]
//! (Sella Cartesian `fix_translation` / `fix_rotation`,
//! \(R^{3N}/\mathrm{SE}(3)\)) or [`ManifoldKind::MwRigid`] (Page–McIver
//! mass-weighted Eckart, the IRC metric). Sphere / SO(3)-9 / SE(3)-12
//! / SPD-n² / Grassmann-\(np\) / ComplexCircle-\(2n\) / Oblique-\(nm\)
//! are matrix-manifold embeddings, not a 3N cluster.
//! [`ManifoldKind::Hyperbolic`] is the Lorentz hyperboloid
//! (manopt `hyperbolicfactory`), not the sphere.
//! [`ManifoldKind::ComplexCircle`] is manopt `complexcirclefactory`.
//! [`ManifoldKind::Multinomial`] is the simplex with the Fisher metric.

use ndarray::Array1;

mod complex_circle;
mod euclidean;
mod grassmann;
mod hyperbolic;
mod multinomial;
mod mw_rigid;
mod oblique;
mod rigid_quotient;
mod se3;
mod so3;
mod spd;
mod sphere;
mod stiefel;

pub use complex_circle::ComplexCircle;
pub use euclidean::Euclidean;
pub use grassmann::Grassmann;
pub use hyperbolic::{minkowski, pack as pack_h, unpack as unpack_h, Hyperbolic};
pub use multinomial::Multinomial;
pub use mw_rigid::MwRigid;
pub use oblique::Oblique;
pub use rigid_quotient::RigidQuotient;
pub use se3::Se3;
pub use so3::So3;
pub use spd::{is_spd, pack, side, unpack, Spd};
pub use sphere::Sphere;
pub use stiefel::{Stiefel, StiefelNp};

/// Which embedded geometry a session retracts onto.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ManifoldKind {
    /// Ambient Euclidean. Today's path.
    #[default]
    Euclidean,
    /// Unit sphere \(S^{n-1}\).
    Sphere,
    /// Rotation matrices SO(3), 9-vector row-major.
    So3,
    /// Stiefel \(\mathrm{St}(n,1)\). A single orthonormal column.
    /// `p > 1` is [`Self::StiefelP`]; a 3N cluster is [`Self::RigidQuotient`].
    Stiefel,
    /// Rigid motions SE(3): 3x3 row-major then translation (12).
    Se3,
    /// Isolated-molecule shape space \(R^{3N}/\mathrm{SE}(3)\).
    /// Sella Cartesian + `fix_translation` + `fix_rotation`.
    RigidQuotient,
    /// Mass-weighted Eckart: Sella IRC / Page–McIver metric on
    /// the same quotient. Masses from [`crate::Solver::set_masses`].
    MwRigid,
    /// Symmetric positive definite n-by-n, row-major n².
    /// manopt `sympositivedefinitefactory` (affine-invariant).
    Spd,
    /// Real Grassmann \(\mathrm{Gr}(n, p)\). Packed column-major \(n p\).
    /// A 3N cluster is [`Self::RigidQuotient`], not this embedding.
    Grassmann {
        /// Ambient dimension. Rows of the frame.
        n: usize,
        /// Subspace dimension. Columns of the frame.
        p: usize,
    },
    /// Hyperboloid \(H^{n-1}\) as a length-\(n\) Minkowski vector
    /// (`n >= 2`). manopt `hyperbolicfactory(n-1)` with \(m = 1\).
    Hyperbolic,
    /// Product of unit-modulus complex numbers \((S^1)^n\).
    /// Packed interleaved `(re, im)`, length `2 n`.
    /// manopt `complexcirclefactory(n)`. Not the sphere.
    ComplexCircle {
        /// Number of unit-modulus complex entries.
        n: usize,
    },
}

impl ManifoldKind {
    /// Stiefel column count. `p = 1` is the sphere.
    pub fn stiefel_p(self) -> usize {
        match self {
            Self::Grassmann { p, .. } => p,
            _ => 1,
        }
    }

    /// Product of `n` unit circles. Packed length `2 n`.
    pub fn complex_circle(n: usize) -> Self {
        Self::ComplexCircle { n }
    }

    /// Real Grassmann \(\mathrm{Gr}(n, p)\).
    pub fn grassmann(n: usize, p: usize) -> Self {
        Self::Grassmann { n, p }
    }

    /// C ABI / INI token.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::Sphere => "sphere",
            Self::So3 => "so3",
            Self::Stiefel | Self::StiefelP { .. } => "stiefel",
            Self::Se3 => "se3",
            Self::RigidQuotient => "rigid_quotient",
            Self::MwRigid => "mw_rigid",
            Self::Spd => "spd",
            Self::Grassmann { .. } => "grassmann",
            Self::Hyperbolic => "hyperbolic",
            Self::Oblique { .. } => "oblique",
            Self::Multinomial => "multinomial",
            Self::ComplexCircle { .. } => "complex_circle",
        }
    }
}

/// manopt_cpp `AbstractManifold` on a rank-1 f64 vector.
pub trait Manifold {
    /// `Ok` if `n` is a legal packing for this geometry.
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        let _ = n;
        Ok(())
    }
    /// Tangent projection of an ambient vector at `x`. Same length as `v`.
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
    /// Retraction of the tangent step `v` at `x`. Same length as `x`.
    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
    /// Vector transport of `v` from `x_from` to `x_to`.
    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
    /// Euclidean gradient to Riemannian gradient at `x`.
    ///
    /// Riemannian submanifolds of Euclidean space: this is [`Self::project`].
    /// The hyperboloid uses the Minkowski metric, so it is not that case.
    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        self.project(x, egrad)
    }
}

impl Manifold for ManifoldKind {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        match self {
            Self::Euclidean => Euclidean.required_dim(n),
            Self::Sphere => Sphere.required_dim(n),
            Self::So3 => So3.required_dim(n),
            Self::Stiefel => Stiefel.required_dim(n),
            Self::Se3 => Se3.required_dim(n),
            Self::RigidQuotient => RigidQuotient.required_dim(n),
            Self::MwRigid => MwRigid.required_dim(n),
            Self::Spd => Spd.required_dim(n),
            Self::Grassmann { n: gn, p } => Grassmann { n: *gn, p: *p }.required_dim(n),
            Self::Hyperbolic => Hyperbolic.required_dim(n),
            Self::ComplexCircle { n: cn } => ComplexCircle { n: *cn }.required_dim(n),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => Euclidean.project(x, v),
            Self::Sphere => Sphere.project(x, v),
            Self::So3 => So3.project(x, v),
            Self::Stiefel => Stiefel.project(x, v),
            Self::Se3 => Se3.project(x, v),
            Self::RigidQuotient => RigidQuotient.project(x, v),
            Self::MwRigid => MwRigid.project(x, v),
            Self::Spd => Spd.project(x, v),
            Self::Grassmann { n, p } => Grassmann { n: *n, p: *p }.project(x, v),
            Self::Hyperbolic => Hyperbolic.project(x, v),
            Self::ComplexCircle { n } => ComplexCircle { n: *n }.project(x, v),
        }
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => Euclidean.retract(x, v),
            Self::Sphere => Sphere.retract(x, v),
            Self::So3 => So3.retract(x, v),
            Self::Stiefel => Stiefel.retract(x, v),
            Self::Se3 => Se3.retract(x, v),
            Self::RigidQuotient => RigidQuotient.retract(x, v),
            Self::MwRigid => MwRigid.retract(x, v),
            Self::Spd => Spd.retract(x, v),
            Self::Grassmann { n, p } => Grassmann { n: *n, p: *p }.retract(x, v),
            Self::Hyperbolic => Hyperbolic.retract(x, v),
            Self::ComplexCircle { n } => ComplexCircle { n: *n }.retract(x, v),
        }
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => Euclidean.transport(x_from, x_to, v),
            Self::Sphere => Sphere.transport(x_from, x_to, v),
            Self::So3 => So3.transport(x_from, x_to, v),
            Self::Stiefel => Stiefel.transport(x_from, x_to, v),
            Self::Se3 => Se3.transport(x_from, x_to, v),
            Self::RigidQuotient => RigidQuotient.transport(x_from, x_to, v),
            Self::MwRigid => MwRigid.transport(x_from, x_to, v),
            Self::Spd => Spd.transport(x_from, x_to, v),
            Self::Grassmann { n, p } => Grassmann { n: *n, p: *p }.transport(x_from, x_to, v),
            Self::Hyperbolic => Hyperbolic.transport(x_from, x_to, v),
            Self::ComplexCircle { n } => ComplexCircle { n: *n }.transport(x_from, x_to, v),
        }
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Hyperbolic => Hyperbolic.egrad2rgrad(x, egrad),
            other => other.project(x, egrad),
        }
    }
}
