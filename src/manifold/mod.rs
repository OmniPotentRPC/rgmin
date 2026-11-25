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
//! [`RiemannianManifold`] is a runtime product of Stiefel and SPD factors
//! on column-major buffers, with the exponential, logarithm, metric and
//! parallel transport of each factor.
//! [`ManifoldKind::Product`] is manopt `productmanifold` with a weight on
//! each factor metric; [`ManifoldKind::power`] is `powermanifold`.
//! [`ManifoldKind::SubspaceSphere`] is the unit sphere of a linear
//! subspace.

use std::sync::Arc;

use ndarray::{Array1, Array2};

/// Bind the geometry a [`ManifoldKind`] names to `$f` and evaluate `$body`.
macro_rules! with_kind {
    ($kind:expr, |$f:ident| $body:expr) => {
        match $kind {
            ManifoldKind::Euclidean => {
                let $f = Euclidean;
                $body
            }
            ManifoldKind::Sphere => {
                let $f = Sphere;
                $body
            }
            ManifoldKind::So3 => {
                let $f = So3;
                $body
            }
            ManifoldKind::Stiefel => {
                let $f = Stiefel;
                $body
            }
            ManifoldKind::Se3 => {
                let $f = Se3;
                $body
            }
            ManifoldKind::RigidQuotient => {
                let $f = RigidQuotient;
                $body
            }
            ManifoldKind::MwRigid => {
                let $f = MwRigid;
                $body
            }
            ManifoldKind::Grassmann => {
                let $f = GrassmannLine;
                $body
            }
            ManifoldKind::GrassmannP { n, p } => {
                let $f = Grassmann { n: *n, p: *p };
                $body
            }
            ManifoldKind::Hyperbolic => {
                let $f = Hyperbolic;
                $body
            }
            ManifoldKind::PoincareBall => {
                let $f = PoincareBall;
                $body
            }
            ManifoldKind::Unitary { n } => {
                let $f = Unitary { n: *n };
                $body
            }
            ManifoldKind::Oblique { n, m } => {
                let $f = Oblique { n: *n, m: *m };
                $body
            }
            ManifoldKind::Multinomial => {
                let $f = Multinomial;
                $body
            }
            ManifoldKind::StiefelP { n, p } => {
                let $f = StiefelNp { n: *n, p: *p };
                $body
            }
            ManifoldKind::Spd => {
                let $f = Spd;
                $body
            }
            ManifoldKind::Symmetric => {
                let $f = Symmetric;
                $body
            }
            ManifoldKind::SkewSymmetric => {
                let $f = SkewSymmetric;
                $body
            }
            ManifoldKind::ComplexCircle { n } => {
                let $f = ComplexCircle { n: *n };
                $body
            }
            ManifoldKind::EuclideanComplex { n } => {
                let $f = EuclideanComplex { n: *n };
                $body
            }
            ManifoldKind::Constant { n } => {
                let $f = Constant { n: *n };
                $body
            }
            ManifoldKind::MultinomialDoublyStochastic { n } => {
                let $f = MultinomialDoublyStochastic { n: *n };
                $body
            }
            ManifoldKind::MultinomialSymmetric { n } => {
                let $f = MultinomialSymmetric { n: *n };
                $body
            }
            ManifoldKind::SphereComplex { n } => {
                let $f = SphereComplex { n: *n };
                $body
            }
            ManifoldKind::Positive { n } => {
                let $f = Positive { n: *n };
                $body
            }
            ManifoldKind::CenteredMatrix { m, n, rows } => {
                let $f = CenteredMatrix {
                    m: *m,
                    n: *n,
                    rows: *rows,
                };
                $body
            }
            ManifoldKind::Product(product) => {
                let $f: &ProductManifold = product;
                $body
            }
            ManifoldKind::SubspaceSphere(sphere) => {
                let $f: &SubspaceSphere = sphere;
                $body
            }
        }
    };
}

/// Geodesic operations of a linear subspace whose tangent projector
/// \(P\) does not depend on the point. The Weingarten term vanishes,
/// geodesics are straight lines and parallel transport is the identity.
macro_rules! flat_geodesics {
    () => {
        fn ehess2rhess(
            &self,
            x: &Array1<f64>,
            _egrad: &Array1<f64>,
            ehess: &Array1<f64>,
            _v: &Array1<f64>,
        ) -> Option<Array1<f64>> {
            (self.required_dim(x.len()).is_ok() && ehess.len() == x.len())
                .then(|| self.project(x, ehess))
        }

        fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
            (self.required_dim(x.len()).is_ok() && v.len() == x.len())
                .then(|| x + &self.project(x, v))
        }

        fn log(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
            (self.required_dim(x.len()).is_ok() && y.len() == x.len())
                .then(|| self.project(x, &(y - x)))
        }

        fn parallel_transport_along(
            &self,
            x: &Array1<f64>,
            d: &Array1<f64>,
            v: &Array1<f64>,
        ) -> Option<Array1<f64>> {
            (self.required_dim(x.len()).is_ok() && d.len() == x.len() && v.len() == x.len())
                .then(|| self.project(x, v))
        }

        fn transport_is_isometric(&self) -> bool {
            true
        }
    };
}

mod centered;
mod complex_circle;
mod constant;
mod euclidean;
mod euclidean_complex;
mod grassmann;
mod hyperbolic;
mod multinomial;
mod multinomial_ds;
mod multinomial_sym;
mod mw_rigid;
mod oblique;
mod poincare;
mod positive;
mod product;
mod riemannian;
mod rigid_quotient;
mod se3;
mod skewsymmetric;
mod so3;
mod spd;
mod sphere;
mod sphere_complex;
mod stiefel;
mod subspace_sphere;
mod symmetric;
mod unitary;

pub use centered::{
    CenterMode, CenteredMatrix, inner as inner_centered, is_centered, is_centered_mode,
    pack as pack_centered, typical_dist as typical_dist_centered,
    typical_dist_mode as typical_dist_centered_mode, unpack as unpack_centered,
};
pub use complex_circle::ComplexCircle;
pub use constant::{
    Constant, inner as inner_const, is_constant, typical_dist as typical_dist_const,
};
pub use euclidean::Euclidean;
pub use euclidean_complex::{
    EuclideanComplex, inner as inner_cplx, is_euclidean_complex, typical_dist as typical_dist_cplx,
};
pub use grassmann::Grassmann;
pub use hyperbolic::{Hyperbolic, minkowski, pack, unpack};
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
pub use poincare::PoincareBall;
pub use positive::{Positive, inner as inner_pos, is_positive, typical_dist as typical_dist_pos};
pub use product::{ProductFactor, ProductManifold, TangentMetric};
pub use riemannian::{ManifoldFactor, RiemannianManifold, SpdMetric, StiefelRetraction};
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
pub use subspace_sphere::SubspaceSphere;
pub use symmetric::{
    Symmetric, inner as inner_sym, is_symmetric, pack as pack_sym, side as side_sym,
    typical_dist as typical_dist_sym, unpack as unpack_sym,
};
pub use unitary::{
    Unitary, inner as inner_unitary, is_unitary, pack as pack_unitary, side as side_unitary,
    typical_dist as typical_dist_unitary, unpack as unpack_unitary,
};

/// Which embedded geometry a session retracts onto.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
    /// Not [`Self::Spd`], [`Self::Grassmann`] or [`Self::Hyperbolic`].
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
    /// manopt `centeredmatrixfactory`. Not [`Self::Spd`],
    /// [`Self::Grassmann`] or [`Self::Hyperbolic`].
    CenteredMatrix {
        /// Rows of the matrix. MATLAB `m`.
        m: usize,
        /// Columns of the matrix. MATLAB `n`.
        n: usize,
        /// `true` = centered rows. `false` = centered columns
        /// (manopt default).
        rows: bool,
    },
    /// Weighted Riemannian product \(\mathcal{M}_1 \times \cdots \times
    /// \mathcal{M}_k\) with metric \(\sum_i w_i g_i\), factors packed end to
    /// end. Construct with [`ManifoldKind::product`] or
    /// [`ManifoldKind::power`].
    Product(Arc<ProductManifold>),
    /// Unit sphere of a linear subspace \(U \subset \mathbb{R}^n\), packed
    /// length `n`. Construct with [`ManifoldKind::sphere_within`] or
    /// [`ManifoldKind::sphere_excluding`].
    SubspaceSphere(Arc<SubspaceSphere>),
}

impl ManifoldKind {
    /// True when the retraction is the translation `x + v`.
    pub(crate) fn retract_is_translation(&self) -> bool {
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
            None => Grassmann {
                n: len.max(1),
                p: 1,
            },
        }
    }

    /// Retract with an optional Grassmann factor shape.
    pub fn retract_shaped(
        &self,
        shape: Option<(usize, usize)>,
        x: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        match self {
            Self::Grassmann => Self::grassmann(shape, x.len()).retract(x, v),
            other => other.retract(x, v),
        }
    }

    /// Project with an optional Grassmann factor shape.
    pub fn project_shaped(
        &self,
        shape: Option<(usize, usize)>,
        x: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        match self {
            Self::Grassmann => Self::grassmann(shape, x.len()).project(x, v),
            other => other.project(x, v),
        }
    }

    /// Transport with an optional Grassmann factor shape.
    pub fn transport_shaped(
        &self,
        shape: Option<(usize, usize)>,
        x_from: &Array1<f64>,
        x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        match self {
            Self::Grassmann => Self::grassmann(shape, x_to.len()).transport(x_from, x_to, v),
            other => other.transport(x_from, x_to, v),
        }
    }

    /// Weighted product of `factors`. See [`ProductManifold::new`].
    pub fn product<I>(factors: I) -> crate::Result<Self>
    where
        I: IntoIterator<Item = ProductFactor>,
    {
        ProductManifold::new(factors).map(|p| Self::Product(Arc::new(p)))
    }

    /// manopt `powermanifold`: `count` unit-weight copies of `factor`,
    /// each of packed length `block`. A power is the product of equal
    /// factors; [`Self::Oblique`] stays the dedicated power of spheres.
    pub fn power(factor: Self, block: usize, count: usize) -> crate::Result<Self> {
        ProductManifold::power(factor, block, count).map(|p| Self::Product(Arc::new(p)))
    }

    /// Unit sphere of the span of the rows of `basis`. See
    /// [`SubspaceSphere::within`].
    pub fn sphere_within(basis: Array2<f64>) -> crate::Result<Self> {
        SubspaceSphere::within(basis).map(|s| Self::SubspaceSphere(Arc::new(s)))
    }

    /// Unit vectors orthogonal to the rows of `basis`. See
    /// [`SubspaceSphere::excluding`].
    pub fn sphere_excluding(basis: Array2<f64>) -> crate::Result<Self> {
        SubspaceSphere::excluding(basis).map(|s| Self::SubspaceSphere(Arc::new(s)))
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
    pub fn stiefel_p(&self) -> usize {
        match self {
            Self::StiefelP { p, .. } => *p,
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

    /// [`Manifold::project`] written back into `v`. The sphere does it
    /// without allocating, with the arithmetic of `Sphere::project`.
    pub(crate) fn project_in_place(&self, x: &Array1<f64>, v: &mut Array1<f64>) {
        if *self == Self::Sphere {
            let s: f64 = x.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            v.zip_mut_with(x, |vi, &xi| *vi -= s * xi);
        } else {
            *v = self.project(x, v);
        }
    }

    /// `true` when every ambient point is a point of the manifold, so the
    /// Euclidean projection onto the manifold cut by a coordinate box is
    /// the box clip itself.
    ///
    /// Euclidean space, complex Euclidean space and the rigid quotients
    /// are open pieces of the packed coordinates, and a product is only
    /// when every factor is. A clip leaves every other set: a sphere loses
    /// its unit norm, a symmetric matrix loses symmetry, a centered matrix
    /// loses its zero mean.
    pub(crate) fn coordinate_clip_stays_on_set(&self) -> bool {
        match self {
            Self::Product(product) => product.coordinate_clip_stays_on_set(),
            Self::Euclidean
            | Self::EuclideanComplex { .. }
            | Self::RigidQuotient
            | Self::MwRigid => true,
            _ => false,
        }
    }

    /// `true` when the retraction keeps a straight step on the set, so a
    /// chord can be capped and clipped after it. A curved kind caps the
    /// tangent step before the retraction instead.
    pub(crate) fn is_flat(&self) -> bool {
        match self {
            Self::Product(product) => product.is_flat(),
            Self::Euclidean
            | Self::RigidQuotient
            | Self::MwRigid
            | Self::Symmetric
            | Self::SkewSymmetric
            | Self::EuclideanComplex { .. }
            | Self::CenteredMatrix { .. }
            | Self::Constant { .. } => true,
            Self::Sphere
            | Self::So3
            | Self::Stiefel
            | Self::Se3
            | Self::Oblique { .. }
            | Self::Multinomial
            | Self::StiefelP { .. }
            | Self::Spd
            | Self::ComplexCircle { .. }
            | Self::MultinomialDoublyStochastic { .. }
            | Self::MultinomialSymmetric { .. }
            | Self::SphereComplex { .. }
            | Self::Positive { .. }
            | Self::Grassmann
            | Self::GrassmannP { .. }
            | Self::Hyperbolic
            | Self::PoincareBall
            | Self::Unitary { .. }
            | Self::SubspaceSphere(_) => false,
        }
    }

    /// C ABI / INI token.
    pub fn as_str(&self) -> &'static str {
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
            Self::Product(_) => "product",
            Self::SubspaceSphere(_) => "subspace_sphere",
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

    /// Riemannian metric \(g_x(u, v)\) between tangent vectors at `x`.
    ///
    /// The default is the ambient Euclidean product, the metric under
    /// which the default [`Self::egrad2rgrad`] (the projection) is the
    /// Riemannian gradient. A kind that overrides one overrides both.
    fn inner(&self, x: &Array1<f64>, u: &Array1<f64>, v: &Array1<f64>) -> f64 {
        let _ = x;
        crate::vecops::dot(u.view(), v.view())
    }

    /// Length of the tangent vector `v` at `x` in [`Self::inner`].
    fn norm(&self, x: &Array1<f64>, v: &Array1<f64>) -> f64 {
        self.inner(x, v, v).max(0.0).sqrt()
    }

    /// Riemannian Hessian action \(\mathrm{Hess} f(x)\,v\), manopt
    /// `ehess2rhess`.
    ///
    /// `egrad` is the Euclidean gradient at `x`, `ehess` the Euclidean
    /// Hessian action \(\nabla^2 f(x)\,v\), and `v` a tangent vector.
    /// The result adds the curvature (Weingarten) term of the embedding
    /// to the projected `ehess`. `None` when the kind does not supply that
    /// term, which is the default.
    ///
    /// `egrad` and `ehess` may come from any smooth extension \(\bar f\)
    /// of the cost off the manifold. On an embedded kind two extensions
    /// differ by a function that vanishes on the manifold, so its
    /// gradient there is normal, and the Weingarten term cancels what that
    /// normal gradient adds to the projected Hessian: the result does not
    /// depend on the choice. The projected `ehess` alone is the Riemannian
    /// Hessian where the normal part of `egrad` adds no curvature; on the
    /// sphere that is \(x^\top\nabla\bar f = 0\), which the degree-zero
    /// extension \(\bar f(x) = f(x/\|x\|)\) satisfies.
    fn ehess2rhess(
        &self,
        x: &Array1<f64>,
        egrad: &Array1<f64>,
        ehess: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        let _ = (x, egrad, ehess, v);
        None
    }

    /// Exponential map \(\exp_x(v)\) where it has a closed form. `None`
    /// otherwise, which is the default.
    fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
        let _ = (x, v);
        None
    }

    /// Logarithm: the tangent `v` at `x` of least length with
    /// \(\exp_x(v) = y\). `None` without a closed form, the default.
    fn log(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
        let _ = (x, y);
        None
    }

    /// Riemannian distance. The default is the length of [`Self::log`].
    fn dist(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
        self.log(x, y).map(|v| self.norm(x, &v))
    }

    /// Parallel transport of the tangent `v` along the geodesic
    /// \(t \mapsto \exp_x(t\,d)\) from `t = 0` to `t = 1`. An isometry of
    /// [`Self::inner`]. `None` without a closed form, the default.
    fn parallel_transport_along(
        &self,
        x: &Array1<f64>,
        d: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        let _ = (x, d, v);
        None
    }

    /// Parallel transport of `v` from `x_from` to `x_to` along the
    /// geodesic of [`Self::log`].
    fn parallel_transport(
        &self,
        x_from: &Array1<f64>,
        x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        let d = self.log(x_from, x_to)?;
        self.parallel_transport_along(x_from, &d, v)
    }

    /// `true` when [`Self::transport`] maps tangent vectors isometrically
    /// in [`Self::inner`], the condition Riemannian BFGS theory puts on
    /// its vector transport. The default is `false`.
    fn transport_is_isometric(&self) -> bool {
        false
    }

    /// manopt `typicaldist` at packed length `n`: the distance scale a
    /// trust region starts from. The default is \(\sqrt{n}\).
    fn typical_dist(&self, n: usize) -> f64 {
        (n as f64).sqrt()
    }
}

/// [`ManifoldKind::Grassmann`]: \(\mathrm{Gr}(n, 1)\) with `n` the packed
/// length of each call, as [`ManifoldKind::grassmann`] binds it.
struct GrassmannLine;

impl Manifold for GrassmannLine {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n >= 2 { Ok(()) } else { Err(n) }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        ManifoldKind::grassmann(None, x.len()).project(x, v)
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        ManifoldKind::grassmann(None, x.len()).egrad2rgrad(x, egrad)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        ManifoldKind::grassmann(None, x.len()).retract(x, v)
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        ManifoldKind::grassmann(None, x_to.len()).transport(x_from, x_to, v)
    }
}

impl Manifold for ManifoldKind {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        with_kind!(self, |f| f.required_dim(n))
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        with_kind!(self, |f| f.project(x, v))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        with_kind!(self, |f| f.retract(x, v))
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        with_kind!(self, |f| f.transport(x_from, x_to, v))
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        with_kind!(self, |f| f.egrad2rgrad(x, egrad))
    }

    fn inner(&self, x: &Array1<f64>, u: &Array1<f64>, v: &Array1<f64>) -> f64 {
        with_kind!(self, |f| f.inner(x, u, v))
    }

    fn norm(&self, x: &Array1<f64>, v: &Array1<f64>) -> f64 {
        with_kind!(self, |f| f.norm(x, v))
    }

    fn ehess2rhess(
        &self,
        x: &Array1<f64>,
        egrad: &Array1<f64>,
        ehess: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        with_kind!(self, |f| f.ehess2rhess(x, egrad, ehess, v))
    }

    fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
        with_kind!(self, |f| f.exp(x, v))
    }

    fn log(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
        with_kind!(self, |f| f.log(x, y))
    }

    fn dist(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
        with_kind!(self, |f| f.dist(x, y))
    }

    fn parallel_transport_along(
        &self,
        x: &Array1<f64>,
        d: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        with_kind!(self, |f| f.parallel_transport_along(x, d, v))
    }

    fn parallel_transport(
        &self,
        x_from: &Array1<f64>,
        x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        with_kind!(self, |f| f.parallel_transport(x_from, x_to, v))
    }

    fn transport_is_isometric(&self) -> bool {
        with_kind!(self, |f| f.transport_is_isometric())
    }

    fn typical_dist(&self, n: usize) -> f64 {
        with_kind!(self, |f| f.typical_dist(n))
    }
}
