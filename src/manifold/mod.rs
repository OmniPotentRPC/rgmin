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
//! / Symmetric-n² / SPD-n² / ComplexCircle-2n / EuclideanComplex-2n
//! / Constant-n / Positive-n / CenteredMatrix-mn / Oblique-nm are matrix-manifold embeddings, not a 3N cluster.
//! [`ManifoldKind::Symmetric`] is manopt `symmetricfactory`.
//! [`ManifoldKind::SkewSymmetric`] is manopt `skewsymmetricfactory`.
//! [`ManifoldKind::ComplexCircle`] is manopt `complexcirclefactory`.
//! [`ManifoldKind::EuclideanComplex`] is manopt `euclideancomplexfactory`.
//! [`ManifoldKind::Constant`] is manopt `constantfactory`.
//! [`ManifoldKind::MultinomialDoublyStochastic`] is manopt
//! `multinomialdoublystochasticfactory`.
//! [`ManifoldKind::MultinomialSymmetric`] is manopt
//! `multinomialsymmetricfactory`.
//! [`ManifoldKind::SphereComplex`] is manopt `spherecomplexfactory`.
//! [`ManifoldKind::Positive`] is manopt `positivefactory`.
//! [`ManifoldKind::CenteredMatrix`] is manopt `centeredmatrixfactory`.

use ndarray::Array1;

mod centered;
mod complex_circle;
mod constant;
mod euclidean;
mod grassmann;
mod hyperbolic;
mod poincare;
mod unitary;
mod euclidean_complex;
mod multinomial;
mod multinomial_ds;
mod multinomial_sym;
mod mw_rigid;
mod oblique;
mod positive;
mod rigid_quotient;
mod se3;
mod skewsymmetric;
mod so3;
mod spd;
mod sphere;
mod sphere_complex;
mod stiefel;
mod symmetric;

pub use centered::{
    CenteredMatrix, inner as inner_centered, is_centered, pack as pack_centered,
    typical_dist as typical_dist_centered, unpack as unpack_centered,
};
pub use complex_circle::ComplexCircle;
pub use constant::{
    Constant, inner as inner_const, is_constant, typical_dist as typical_dist_const,
};
pub use euclidean::Euclidean;
pub use grassmann::Grassmann;
pub use hyperbolic::{Hyperbolic, minkowski, pack, unpack};
pub use poincare::PoincareBall;
pub use unitary::{Unitary, is_unitary, pack as pack_unitary, side as side_unitary, unpack as unpack_unitary};
pub use euclidean_complex::{
    EuclideanComplex, inner as inner_cplx, is_euclidean_complex, typical_dist as typical_dist_cplx,
};
pub use multinomial::Multinomial;
pub use multinomial_ds::{
    MultinomialDoublyStochastic, inner as inner_ds, is_doubly_stochastic, pack as pack_ds,
    side as side_ds, typical_dist as typical_dist_ds, unpack as unpack_ds,
};
pub use multinomial_sym::{
    MultinomialSymmetric, inner as inner_msym, is_symmetric_doubly_stochastic, pack as pack_msym,
    side as side_msym, typical_dist as typical_dist_msym, unpack as unpack_msym,
};
pub use mw_rigid::MwRigid;
pub use oblique::Oblique;
pub use positive::{Positive, inner as inner_pos, is_positive, typical_dist as typical_dist_pos};
pub use rigid_quotient::RigidQuotient;
pub use se3::Se3;
pub use skewsymmetric::{
    SkewSymmetric, inner as inner_skew, is_skewsymmetric, pack as pack_skew, side as side_skew,
    typical_dist as typical_dist_skew, unpack as unpack_skew,
};
pub use so3::So3;
pub use spd::{Spd, is_spd, pack as pack_spd, side as side_spd, unpack as unpack_spd};
pub use sphere::Sphere;
pub use sphere_complex::{
    SphereComplex, inner as inner_scplx, is_sphere_complex, typical_dist as typical_dist_scplx,
};
pub use stiefel::{Stiefel, StiefelNp};
pub use symmetric::{
    Symmetric, inner as inner_sym, is_symmetric, pack as pack_sym, side as side_sym,
    typical_dist as typical_dist_sym, unpack as unpack_sym,
};

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
    /// Real Grassmann Gr(n,1), or the session's configured factor shape.
    Grassmann,
    /// Real Grassmann Gr(n,p), packed column-major.
    GrassmannP {
        /// Ambient dimension.
        n: usize,
        /// Number of orthonormal columns.
        p: usize,
    },
    /// Hyperboloid in the Minkowski metric, packed length at least two.
    Hyperbolic,
    /// Open unit ball with the Poincare metric.
    PoincareBall,
    /// Complex unitary U(n), interleaved row-major length 2*n*n.
    Unitary {
        /// Matrix side length.
        n: usize,
    },
    /// Product of `m` unit spheres in \(\mathbb{R}^n\). Packed
    /// column-major, length `n*m`. manopt `obliquefactory`.
    Oblique {
        /// Ambient dimension of each sphere (column length).
        n: usize,
        /// Number of unit-norm columns.
        m: usize,
    },
    /// Simplex \(\{x > 0,\ \sum x = 1\}\) with the Fisher metric.
    /// manopt `multinomialfactory` at \(m = 1\). Length \(n \ge 2\).
    Multinomial,
    /// Stiefel \(\mathrm{St}(n,p)\) for `p > 1`, packed column-major
    /// length `n*p`. Construct with [`ManifoldKind::stiefel`].
    StiefelP {
        /// Ambient dimension. Rows of the frame.
        n: usize,
        /// Orthonormal columns. Must be `> 1`.
        p: usize,
    },
    /// Symmetric positive definite n-by-n, row-major n².
    /// manopt `sympositivedefinitefactory` (affine-invariant).
    Spd,
    /// Real symmetric n-by-n, row-major n².
    /// manopt `symmetricfactory` (Frobenius / Euclidean subspace).
    Symmetric,
    /// Real skew-symmetric n-by-n, row-major n², `n >= 2`.
    /// manopt `skewsymmetricfactory` (Frobenius / Euclidean subspace).
    SkewSymmetric,
    /// Product of unit-modulus complex numbers \((S^1)^n\).
    /// Packed interleaved `(re, im)`, length `2 n`.
    /// manopt `complexcirclefactory(n)`. Not the sphere.
    ComplexCircle {
        /// Number of unit-modulus complex entries.
        n: usize,
    },
    /// Complex Euclidean \(\mathbb{C}^n\). Packed interleaved
    /// `(re, im)`, length `2 n`. manopt `euclideancomplexfactory(n)`.
    /// Not the sphere and not \((S^1)^n\).
    EuclideanComplex {
        /// Number of complex entries.
        n: usize,
    },
    /// Singleton \(\{A\}\) of packed length `n`. manopt
    /// `constantfactory`. Projection and transport are zero;
    /// retraction is the fixed point. Not the sphere.
    Constant {
        /// Packed length of the singleton.
        n: usize,
    },
    /// Doubly-stochastic n-by-n, row-major n², `n >= 2`.
    /// manopt `multinomialdoublystochasticfactory`.
    MultinomialDoublyStochastic {
        /// Side length.
        n: usize,
    },
    /// Symmetric doubly-stochastic n-by-n, row-major n², `n >= 2`.
    /// manopt `multinomialsymmetricfactory`.
    MultinomialSymmetric {
        /// Side length.
        n: usize,
    },
    /// Complex unit sphere in \(\mathbb{C}^n\), packed `2 n`.
    /// manopt `spherecomplexfactory`. Not the real sphere.
    SphereComplex {
        /// Complex dimension.
        n: usize,
    },
    /// Strictly positive orthant of packed length `n`.
    /// manopt `positivefactory`. Not the sphere and not a 3N cluster.
    Positive {
        /// Packed length. MATLAB `m*n` with default second size 1.
        n: usize,
    },
    /// Centered `m x n` matrices, packed row-major `m n`.
    /// manopt `centeredmatrixfactory`.
    CenteredMatrix {
        /// Rows of the matrix. MATLAB `m`.
        m: usize,
        /// Columns of the matrix. MATLAB `n`.
        n: usize,
        /// `true` = centered rows. `false` = centered columns
        /// (manopt default).
        rows: bool,
    },
}

impl ManifoldKind {
    /// True when the retraction is the translation `x + v`.
    pub(crate) fn retract_is_translation(self) -> bool {
        matches!(self, Self::Euclidean | Self::RigidQuotient | Self::MwRigid)
    }

    /// Complex unitary U(n), interleaved row-major length 2*n*n.
    pub fn unitary(n: usize) -> Self {
        Self::Unitary { n }
    }

    /// Bind the Grassmann packing to Gr(n,p); no shape means Gr(len,1).
    pub fn grassmann(shape: Option<(usize, usize)>, len: usize) -> Grassmann {
        match shape {
            Some((n, p)) => Grassmann { n, p },
            None => Grassmann { n: len.max(1), p: 1 },
        }
    }

    /// Retract with an optional Grassmann factor shape.
    pub fn retract_shaped(self, shape: Option<(usize, usize)>, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Grassmann => Self::grassmann(shape, x.len()).retract(x, v),
            other => other.retract(x, v),
        }
    }

    /// Project with an optional Grassmann factor shape.
    pub fn project_shaped(self, shape: Option<(usize, usize)>, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Grassmann => Self::grassmann(shape, x.len()).project(x, v),
            other => other.project(x, v),
        }
    }

    /// Transport with an optional Grassmann factor shape.
    pub fn transport_shaped(self, shape: Option<(usize, usize)>, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Grassmann => Self::grassmann(shape, x_to.len()).transport(x_from, x_to, v),
            other => other.transport(x_from, x_to, v),
        }
    }

    /// Stiefel \(\mathrm{St}(n,p)\). `p = 1` is the sphere packing.
    pub fn stiefel(n: usize, p: usize) -> Self {
        if p <= 1 {
            Self::Stiefel
        } else {
            Self::StiefelP { n, p }
        }
    }

    /// Stiefel column count. `p = 1` is the sphere.
    pub fn stiefel_p(self) -> usize {
        match self {
            Self::StiefelP { p, .. } => p,
            _ => 1,
        }
    }

    /// Oblique \(\mathrm{OB}(n, m)\): product of `m` unit spheres.
    pub fn oblique(n: usize, m: usize) -> Self {
        Self::Oblique { n, m }
    }

    /// Product of `n` unit circles. Packed length `2 n`.
    pub fn complex_circle(n: usize) -> Self {
        Self::ComplexCircle { n }
    }

    /// Complex Euclidean \(\mathbb{C}^n\). Packed length `2 n`.
    pub fn euclidean_complex(n: usize) -> Self {
        Self::EuclideanComplex { n }
    }

    /// Singleton of packed length `n`. manopt `constantfactory`.
    pub fn constant(n: usize) -> Self {
        Self::Constant { n }
    }

    /// Doubly-stochastic n-by-n, packed length `n^2`.
    /// manopt `multinomialdoublystochasticfactory`.
    pub fn multinomial_ds(n: usize) -> Self {
        Self::MultinomialDoublyStochastic { n }
    }

    /// Symmetric doubly-stochastic n-by-n, packed length `n^2`.
    /// manopt `multinomialsymmetricfactory`.
    pub fn multinomial_sym(n: usize) -> Self {
        Self::MultinomialSymmetric { n }
    }

    /// Complex unit sphere in \(\mathbb{C}^n\). Packed length `2 n`.
    /// manopt `spherecomplexfactory`.
    pub fn sphere_complex(n: usize) -> Self {
        Self::SphereComplex { n }
    }

    /// Positive orthant of packed length `n`. manopt `positivefactory`.
    pub fn positive(n: usize) -> Self {
        Self::Positive { n }
    }

    /// Centered `m x n` matrices. manopt `centeredmatrixfactory`.
    /// `rows = false` centers columns (`X 1 = 0`).
    pub fn centered_matrix(m: usize, n: usize, rows: bool) -> Self {
        Self::CenteredMatrix { m, n, rows }
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
            Self::Grassmann | Self::GrassmannP { .. } => "grassmann",
            Self::Hyperbolic => "hyperbolic",
            Self::PoincareBall => "poincare_ball",
            Self::Unitary { .. } => "unitary",
            Self::Oblique { .. } => "oblique",
            Self::Multinomial => "multinomial",
            Self::Spd => "spd",
            Self::Symmetric => "symmetric",
            Self::SkewSymmetric => "skewsymmetric",
            Self::ComplexCircle { .. } => "complex_circle",
            Self::EuclideanComplex { .. } => "euclidean_complex",
            Self::Constant { .. } => "constant",
            Self::MultinomialDoublyStochastic { .. } => "multinomialdoublystochastic",
            Self::MultinomialSymmetric { .. } => "multinomialsymmetric",
            Self::SphereComplex { .. } => "spherecomplex",
            Self::Positive { .. } => "positive",
            Self::CenteredMatrix { .. } => "centeredmatrix",
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
    /// Ambient Euclidean gradient to a Riemannian gradient at `x`.
    /// Embedded manifolds with a Euclidean ambient metric use [`Self::project`].
    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        self.project(x, egrad)
    }
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
            Self::Grassmann => if n >= 2 { Ok(()) } else { Err(n) },
            Self::GrassmannP { n: gn, p } => Grassmann { n: *gn, p: *p }.required_dim(n),
            Self::Hyperbolic => Hyperbolic.required_dim(n),
            Self::PoincareBall => PoincareBall.required_dim(n),
            Self::Unitary { n: un } => Unitary { n: *un }.required_dim(n),
            Self::Oblique { n: an, m } => Oblique { n: *an, m: *m }.required_dim(n),
            Self::Multinomial => Multinomial.required_dim(n),
            Self::StiefelP { n: sn, p } => StiefelNp { n: *sn, p: *p }.required_dim(n),
            Self::Spd => Spd.required_dim(n),
            Self::Symmetric => Symmetric.required_dim(n),
            Self::SkewSymmetric => SkewSymmetric.required_dim(n),
            Self::ComplexCircle { n: cn } => ComplexCircle { n: *cn }.required_dim(n),
            Self::EuclideanComplex { n: en } => EuclideanComplex { n: *en }.required_dim(n),
            Self::Constant { n: kn } => Constant { n: *kn }.required_dim(n),
            Self::MultinomialDoublyStochastic { n: dn } => {
                MultinomialDoublyStochastic { n: *dn }.required_dim(n)
            }
            Self::MultinomialSymmetric { n: sn } => {
                MultinomialSymmetric { n: *sn }.required_dim(n)
            }
            Self::SphereComplex { n: cn } => SphereComplex { n: *cn }.required_dim(n),
            Self::Positive { n: pn } => Positive { n: *pn }.required_dim(n),
            Self::CenteredMatrix { m, n: cn, rows } => CenteredMatrix {
                m: *m,
                n: *cn,
                rows: *rows,
            }
            .required_dim(n),
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
            Self::Grassmann => Self::grassmann(None, x.len()).project(x, v),
            Self::GrassmannP { n, p } => Grassmann { n: *n, p: *p }.project(x, v),
            Self::Hyperbolic => Hyperbolic.project(x, v),
            Self::PoincareBall => PoincareBall.project(x, v),
            Self::Unitary { n } => Unitary { n: *n }.project(x, v),
            Self::Oblique { n, m } => Oblique { n: *n, m: *m }.project(x, v),
            Self::Multinomial => Multinomial.project(x, v),
            Self::StiefelP { n, p } => StiefelNp { n: *n, p: *p }.project(x, v),
            Self::Spd => Spd.project(x, v),
            Self::Symmetric => Symmetric.project(x, v),
            Self::SkewSymmetric => SkewSymmetric.project(x, v),
            Self::ComplexCircle { n } => ComplexCircle { n: *n }.project(x, v),
            Self::EuclideanComplex { n } => EuclideanComplex { n: *n }.project(x, v),
            Self::Constant { n } => Constant { n: *n }.project(x, v),
            Self::MultinomialDoublyStochastic { n } => {
                MultinomialDoublyStochastic { n: *n }.project(x, v)
            }
            Self::MultinomialSymmetric { n } => MultinomialSymmetric { n: *n }.project(x, v),
            Self::SphereComplex { n } => SphereComplex { n: *n }.project(x, v),
            Self::Positive { n } => Positive { n: *n }.project(x, v),
            Self::CenteredMatrix { m, n, rows } => CenteredMatrix {
                m: *m,
                n: *n,
                rows: *rows,
            }
            .project(x, v),
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
            Self::Grassmann => Self::grassmann(None, x.len()).retract(x, v),
            Self::GrassmannP { n, p } => Grassmann { n: *n, p: *p }.retract(x, v),
            Self::Hyperbolic => Hyperbolic.retract(x, v),
            Self::PoincareBall => PoincareBall.retract(x, v),
            Self::Unitary { n } => Unitary { n: *n }.retract(x, v),
            Self::Oblique { n, m } => Oblique { n: *n, m: *m }.retract(x, v),
            Self::Multinomial => Multinomial.retract(x, v),
            Self::StiefelP { n, p } => StiefelNp { n: *n, p: *p }.retract(x, v),
            Self::Spd => Spd.retract(x, v),
            Self::Symmetric => Symmetric.retract(x, v),
            Self::SkewSymmetric => SkewSymmetric.retract(x, v),
            Self::ComplexCircle { n } => ComplexCircle { n: *n }.retract(x, v),
            Self::EuclideanComplex { n } => EuclideanComplex { n: *n }.retract(x, v),
            Self::Constant { n } => Constant { n: *n }.retract(x, v),
            Self::MultinomialDoublyStochastic { n } => {
                MultinomialDoublyStochastic { n: *n }.retract(x, v)
            }
            Self::MultinomialSymmetric { n } => MultinomialSymmetric { n: *n }.retract(x, v),
            Self::SphereComplex { n } => SphereComplex { n: *n }.retract(x, v),
            Self::Positive { n } => Positive { n: *n }.retract(x, v),
            Self::CenteredMatrix { m, n, rows } => CenteredMatrix {
                m: *m,
                n: *n,
                rows: *rows,
            }
            .retract(x, v),
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
            Self::Grassmann => Self::grassmann(None, x_to.len()).transport(x_from, x_to, v),
            Self::GrassmannP { n, p } => Grassmann { n: *n, p: *p }.transport(x_from, x_to, v),
            Self::Hyperbolic => Hyperbolic.transport(x_from, x_to, v),
            Self::PoincareBall => PoincareBall.transport(x_from, x_to, v),
            Self::Unitary { n } => Unitary { n: *n }.transport(x_from, x_to, v),
            Self::Oblique { n, m } => Oblique { n: *n, m: *m }.transport(x_from, x_to, v),
            Self::Multinomial => Multinomial.transport(x_from, x_to, v),
            Self::StiefelP { n, p } => StiefelNp { n: *n, p: *p }.transport(x_from, x_to, v),
            Self::Spd => Spd.transport(x_from, x_to, v),
            Self::Symmetric => Symmetric.transport(x_from, x_to, v),
            Self::SkewSymmetric => SkewSymmetric.transport(x_from, x_to, v),
            Self::ComplexCircle { n } => ComplexCircle { n: *n }.transport(x_from, x_to, v),
            Self::EuclideanComplex { n } => EuclideanComplex { n: *n }.transport(x_from, x_to, v),
            Self::Constant { n } => Constant { n: *n }.transport(x_from, x_to, v),
            Self::MultinomialDoublyStochastic { n } => {
                MultinomialDoublyStochastic { n: *n }.transport(x_from, x_to, v)
            }
            Self::MultinomialSymmetric { n } => {
                MultinomialSymmetric { n: *n }.transport(x_from, x_to, v)
            }
            Self::SphereComplex { n } => SphereComplex { n: *n }.transport(x_from, x_to, v),
            Self::Positive { n } => Positive { n: *n }.transport(x_from, x_to, v),
            Self::CenteredMatrix { m, n, rows } => CenteredMatrix {
                m: *m,
                n: *n,
                rows: *rows,
            }
            .transport(x_from, x_to, v),
        }
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => Euclidean.egrad2rgrad(x, egrad),
            Self::Sphere => Sphere.egrad2rgrad(x, egrad),
            Self::So3 => So3.egrad2rgrad(x, egrad),
            Self::Stiefel => Stiefel.egrad2rgrad(x, egrad),
            Self::Se3 => Se3.egrad2rgrad(x, egrad),
            Self::RigidQuotient => RigidQuotient.egrad2rgrad(x, egrad),
            Self::MwRigid => MwRigid.egrad2rgrad(x, egrad),
            Self::Grassmann => Self::grassmann(None, x.len()).egrad2rgrad(x, egrad),
            Self::GrassmannP { n, p } => Grassmann { n: *n, p: *p }.egrad2rgrad(x, egrad),
            Self::Hyperbolic => Hyperbolic.egrad2rgrad(x, egrad),
            Self::PoincareBall => PoincareBall.egrad2rgrad(x, egrad),
            Self::Unitary { n } => Unitary { n: *n }.egrad2rgrad(x, egrad),
            Self::Oblique { n, m } => Oblique { n: *n, m: *m }.egrad2rgrad(x, egrad),
            Self::Multinomial => Multinomial.egrad2rgrad(x, egrad),
            Self::StiefelP { n, p } => StiefelNp { n: *n, p: *p }.egrad2rgrad(x, egrad),
            Self::Spd => Spd.egrad2rgrad(x, egrad),
            Self::Symmetric => Symmetric.egrad2rgrad(x, egrad),
            Self::SkewSymmetric => SkewSymmetric.egrad2rgrad(x, egrad),
            Self::ComplexCircle { n } => ComplexCircle { n: *n }.egrad2rgrad(x, egrad),
            Self::EuclideanComplex { n } => EuclideanComplex { n: *n }.egrad2rgrad(x, egrad),
            Self::Constant { n } => Constant { n: *n }.egrad2rgrad(x, egrad),
            Self::MultinomialDoublyStochastic { n } => {
                MultinomialDoublyStochastic { n: *n }.egrad2rgrad(x, egrad)
            }
            Self::MultinomialSymmetric { n } => {
                MultinomialSymmetric { n: *n }.egrad2rgrad(x, egrad)
            }
            Self::SphereComplex { n } => SphereComplex { n: *n }.egrad2rgrad(x, egrad),
            Self::Positive { n } => Positive { n: *n }.egrad2rgrad(x, egrad),
            Self::CenteredMatrix { m, n, rows } => CenteredMatrix {
                m: *m,
                n: *n,
                rows: *rows,
            }
            .egrad2rgrad(x, egrad),
        }
    }
}
