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
//! are matrix-manifold embeddings, not a 3N cluster.

use ndarray::Array1;

mod euclidean;
mod mw_rigid;
mod rigid_quotient;
mod se3;
mod so3;
mod sphere;
mod stiefel;

pub use euclidean::Euclidean;
pub use mw_rigid::MwRigid;
pub use rigid_quotient::RigidQuotient;
pub use se3::Se3;
pub use so3::So3;
pub use sphere::Sphere;
pub use stiefel::Stiefel;

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
    Stiefel,
    /// Rigid motions SE(3): 3x3 row-major then translation (12).
    Se3,
    /// Isolated-molecule shape space \(R^{3N}/\mathrm{SE}(3)\).
    /// Sella Cartesian + `fix_translation` + `fix_rotation`.
    RigidQuotient,
    /// Mass-weighted Eckart: Sella IRC / Page–McIver metric on
    /// the same quotient. Masses from [`crate::Solver::set_masses`].
    MwRigid,
}

impl ManifoldKind {
    /// True when the retraction is the translation `x + v`, so a point
    /// reached by a Euclidean step is already on the manifold.
    pub(crate) fn retract_is_translation(self) -> bool {
        matches!(self, Self::Euclidean | Self::RigidQuotient | Self::MwRigid)
    }

    /// Stiefel column count. `p = 1` is the sphere.
    pub fn stiefel_p(self) -> usize {
        1
    }

    /// C ABI / INI token.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::Sphere => "sphere",
            Self::So3 => "so3",
            Self::Stiefel => "stiefel",
            Self::Se3 => "se3",
            Self::RigidQuotient => "rigid_quotient",
            Self::MwRigid => "mw_rigid",
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
        }
    }
}
