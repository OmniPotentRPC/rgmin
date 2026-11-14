/// How much rigid motion to remove.
///
/// Discriminants match `rgmin_solver_set_project_rigid`: `1` is [`Auto`](Self::Auto).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum RigidMode {
    /// Leave the vector unchanged.
    Off = 0,
    /// Isolated: translation and rotation. Any periodic axis: translation
    /// on those axes only, no rotation. Frozen coordinates and broken
    /// axes drop the symmetries they remove.
    #[default]
    Auto = 1,
    /// Translations only. A periodic axis keeps only that translation.
    Translation = 2,
    /// Three translations and three rotations, including on a cell.
    /// A host that wants six modes under periodic boundaries asks for this.
    /// [`Auto`](Self::Auto) will not.
    TranslationRotation = 3,
}

impl RigidMode {
    /// `None` when `code` is not 0, 1, 2, or 3.
    pub fn from_i32(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Off),
            1 => Some(Self::Auto),
            2 => Some(Self::Translation),
            3 => Some(Self::TranslationRotation),
            _ => None,
        }
    }
}

/// Geometry and policy for one projection.
#[derive(Clone, Copy, Debug)]
pub struct RigidRequest<'a> {
    /// Length `3 N`. Cartesian, or `q_i = sqrt(m_i) x_i` when
    /// [`Self::mass_weighted_coords`] is set.
    pub pos: &'a [f64],
    /// Per-atom masses, length `N`. `None` or a wrong length is unit mass.
    pub masses: Option<&'a [f64]>,
    /// `pos` stores `q_i = sqrt(m_i) x_i`. The product is Euclidean.
    /// This wins when [`Self::mass_metric`] is also set: the two are not both on.
    pub mass_weighted_coords: bool,
    /// Cartesian `pos` with inner product `sum m_i u_i · v_i`.
    /// Right for a velocity (`m v`). Wrong for a raw Cartesian force.
    pub mass_metric: bool,
    /// Which modes to build.
    pub mode: RigidMode,
    /// Periodic axes. Under [`RigidMode::Auto`] or [`RigidMode::Translation`],
    /// any true axis drops every rotation and keeps translations only on
    /// the periodic axes.
    pub pbc: [bool; 3],
    /// Length `3 N`. A nonzero byte freezes that coordinate.
    pub frozen: Option<&'a [u8]>,
    /// Translations that are not symmetries. The host sets the mask.
    /// The library does not infer it from a force or a field.
    pub broken_translation: [bool; 3],
    /// Rotations that are not symmetries. Same rule as
    /// [`Self::broken_translation`].
    pub broken_rotation: [bool; 3],
}

impl<'a> RigidRequest<'a> {
    /// Unit-mass Cartesian request. No periodicity, freeze, or broken axis.
    pub fn cartesian(pos: &'a [f64], mode: RigidMode) -> Self {
        Self {
            pos,
            masses: None,
            mass_weighted_coords: false,
            mass_metric: false,
            mode,
            pbc: [false; 3],
            frozen: None,
            broken_translation: [false; 3],
            broken_rotation: [false; 3],
        }
    }
}

/// Caller-owned buffers. One workspace per image. Not thread-local.
///
/// After the first call at a given `N`, [`Self::project`] and
/// [`Self::project_hessian`] do not allocate.
#[derive(Clone, Debug, Default)]
pub struct RigidWorkspace {
    basis: Vec<f64>,
    cand: Vec<f64>,
    reflect: Vec<f64>,
    qwork: Vec<f64>,
    anchor: Vec<f64>,
    hess: Vec<f64>,
    he: Vec<f64>,
    sh: Vec<f64>,
    n: usize,
    anchor_n: usize,
    rank: usize,
}
