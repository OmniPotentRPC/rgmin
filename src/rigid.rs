//! Horizontal space of \(R^{3N}/\mathrm{SE}(3)\), and the finite rigid
//! motion that the same periodic-axis mask removes.
//!
//! The projector is Sella's Cartesian PES (`fix_translation` +
//! `fix_rotation`; Hermes, Sarsfield, Zádor, JCTC 2022,
//! <https://doi.org/10.1021/acs.jctc.2c00395>). Masses turn the inner
//! product into the Page–McIver / Eckart metric used by Sella IRC and
//! gpr_optim `IRCDriver` (Page and McIver 1988,
//! <https://doi.org/10.1063/1.454172>).
//!
//! [`superpose`] is the finite form of that map. The rotation is the
//! unit quaternion of Horn, J. Opt. Soc. Am. A 4, 629 (1987),
//! <https://doi.org/10.1364/josaa.4.000629>, and of Coutsias, Seok, and
//! Dill, J. Comput. Chem. 25, 1849 (2004),
//! <https://doi.org/10.1002/jcc.20110>. Both paths take a
//! [`PeriodicAxes`].

use ndarray::{Array1, ArrayView1};

/// Cartesian axes that are periodic.
///
/// `true` marks a periodic direction. [`project_horizontal`] and
/// [`superpose`] share this value. Rotation is allowed only about an
/// axis that leaves every periodic direction fixed:
///
/// - no periodic axis: all three rotations
/// - one periodic axis: rotation about that axis
/// - two or three periodic axes: no rotation
///
/// Translation is not gated by the mask. The projector removes all
/// three translations for both [`Self::ISOLATED`] and [`Self::CELL`]
/// (Sella's periodic cell is still the quotient by \(T(3)\), with
/// `proj_rot` left off). Superposition subtracts the same weighted
/// centroid, periodic axes included.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PeriodicAxes {
    axes: [bool; 3],
}

impl PeriodicAxes {
    /// No periodic axis. The projector removes translations and all
    /// three infinitesimal rotations.
    pub const ISOLATED: Self = Self { axes: [false; 3] };

    /// Every axis periodic. Sella `proj_rot = false`: translations are
    /// removed and rotations are kept.
    pub const CELL: Self = Self { axes: [true; 3] };

    /// `axes` is `(x, y, z)`.
    pub const fn new(axes: [bool; 3]) -> Self {
        Self { axes }
    }

    /// Session periodic-cell flag. `true` is [`Self::CELL`].
    pub const fn from_cell(periodic: bool) -> Self {
        if periodic { Self::CELL } else { Self::ISOLATED }
    }

    /// `(x, y, z)` periodic flags.
    pub const fn axes(self) -> [bool; 3] {
        self.axes
    }

    /// How many axes are periodic.
    pub const fn count(self) -> usize {
        self.axes[0] as usize + self.axes[1] as usize + self.axes[2] as usize
    }

    /// Axes a rotation may turn about without moving a periodic direction.
    pub const fn rotates(self) -> [bool; 3] {
        match self.count() {
            0 => [true, true, true],
            1 => self.axes,
            _ => [false, false, false],
        }
    }
}

/// Proper rigid motion taking one packed 3N configuration onto another.
///
/// `y = R x + t`. `R` has determinant `+1`. The quaternion is scalar
/// first, `(q0, qx, qy, qz)`, and `q` rather than `-q` so the first
/// nonzero component is positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Superposition {
    /// Unit quaternion, scalar part first.
    pub quaternion: [f64; 4],
    /// Rows of the proper rotation.
    pub rotation: [[f64; 3]; 3],
    /// Translation applied after the rotation.
    pub translation: [f64; 3],
    /// Weighted root-mean-square distance per atom,
    /// \(\sqrt{\sum_i w_i |y_i - x_i|^2 / \sum_i w_i}\).
    pub rmsd: f64,
}

impl Superposition {
    /// Identity motion and a zero residual.
    pub const IDENTITY: Self = Self {
        quaternion: [1.0, 0.0, 0.0, 0.0],
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0, 0.0, 0.0],
        rmsd: 0.0,
    };

    /// `R x + t` for each packed atom of `mobile`.
    #[must_use]
    pub fn apply(&self, mobile: ArrayView1<f64>) -> Array1<f64> {
        let mut out = mobile.to_owned();
        if !mobile.len().is_multiple_of(3) {
            return out;
        }
        let nat = mobile.len() / 3;
        for i in 0..nat {
            let p = [mobile[3 * i], mobile[3 * i + 1], mobile[3 * i + 2]];
            let y = mat_vec(self.rotation, p);
            out[3 * i] = y[0] + self.translation[0];
            out[3 * i + 1] = y[1] + self.translation[1];
            out[3 * i + 2] = y[2] + self.translation[2];
        }
        out
    }
}

impl Default for Superposition {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Optimal proper superposition of `mobile` onto `target`.
///
/// Both arguments are packed Cartesian coordinates, length `3N`.
/// `weights`, when set, has one entry per atom. A missing slice is unit
/// weight. A negative weight is clipped to zero, matching the mass
/// metric of [`project_horizontal`]. A non-finite weight is zero.
///
/// The weighted centroids are removed on every axis. The correlation
/// of the centered coordinates is
/// \(C_{ij} = \sum_a w_a m_{a,i} t_{a,j}\). With no periodic axis the
/// rotation quaternion is the eigenvector of the largest eigenvalue of
/// Horn's symmetric 4×4 key matrix built from `C` (Horn 1987; Coutsias,
/// Seok, and Dill 2004, eq. 10). One periodic axis reduces to the
/// planar angle about that axis (a 2D Kabsch problem in the
/// perpendicular plane). Two or three periodic axes leave the rotation
/// at the identity. The fit does not solve for a scale.
///
/// A repeated leading eigenvalue — a collinear set, a single atom, or
/// any other near-degeneracy within `1e-8` of the spectral radius — is
/// one invariant subspace. The returned quaternion is the unit vector
/// in that subspace closest to the identity, which is an optimal proper
/// rotation (Coutsias, Seok, and Dill 2004, §3). A vanishing total
/// weight, a length that is not `3N`, a weight vector of the wrong
/// length, or a non-finite coordinate returns [`Superposition::IDENTITY`].
/// The result is finite.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgmin::{PeriodicAxes, superpose};
///
/// let mobile = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
/// let target = array![1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 1.0, 3.0, 3.0];
/// let fit = superpose(mobile.view(), target.view(), None, PeriodicAxes::ISOLATED);
/// assert!(fit.rmsd < 1e-12);
/// ```
#[must_use]
pub fn superpose(
    mobile: ArrayView1<f64>,
    target: ArrayView1<f64>,
    weights: Option<&[f64]>,
    periodic: PeriodicAxes,
) -> Superposition {
    let Some(cloud) = Cloud::try_new(mobile, target, weights) else {
        return Superposition::IDENTITY;
    };
    let spin = periodic.rotates();
    let quaternion = if spin == [true, true, true] {
        quaternion_full(&cloud)
    } else if spin[0] {
        quaternion_about(&cloud, 0)
    } else if spin[1] {
        quaternion_about(&cloud, 1)
    } else if spin[2] {
        quaternion_about(&cloud, 2)
    } else {
        [1.0, 0.0, 0.0, 0.0]
    };
    let quaternion = canonicalize(quaternion);
    let rotation = rotation_from_quaternion(quaternion);
    if rotation
        .iter()
        .any(|row| row.iter().any(|c| !c.is_finite()))
    {
        return Superposition::IDENTITY;
    }
    let shifted = mat_vec(rotation, cloud.com_m);
    let translation = [
        cloud.com_t[0] - shifted[0],
        cloud.com_t[1] - shifted[1],
        cloud.com_t[2] - shifted[2],
    ];
    if translation.iter().any(|c| !c.is_finite()) {
        return Superposition::IDENTITY;
    }
    let rmsd = cloud.rmsd(rotation, translation);
    if !rmsd.is_finite() {
        return Superposition::IDENTITY;
    }
    Superposition {
        quaternion,
        rotation,
        translation,
        rmsd,
    }
}

/// eOn `projectOutRotTrans`. Unit-mass Eckart. Isolated molecule (T+R).
pub(crate) fn project_out_rot_trans(vec: &mut Array1<f64>, pos: ArrayView1<f64>) {
    project_horizontal(vec, pos, None, PeriodicAxes::ISOLATED);
}

/// Horizontal space of \(R^{3N}/\mathrm{SE}(3)\) or a masked subgroup.
///
/// `periodic` is the mask [`superpose`] uses. [`PeriodicAxes::CELL`] is
/// Sella `proj_rot = false`. Masses are per atom (length N).
pub(crate) fn project_horizontal(
    vec: &mut Array1<f64>,
    pos: ArrayView1<f64>,
    masses: Option<&[f64]>,
    periodic: PeriodicAxes,
) {
    let n = pos.len();
    if n < 3 || !n.is_multiple_of(3) || vec.len() != n {
        return;
    }
    let spin = periodic.rotates();
    // A full SE(3) projection with fewer than two atoms is a no-op,
    // translations included. Every other mask still removes all three.
    if spin == [true, true, true] && n < 6 {
        return;
    }
    let nat = n / 3;
    let mass_ok = masses.map(|m| m.len() == nat).unwrap_or(false);
    let m_at = |i: usize| -> f64 {
        if mass_ok {
            masses.unwrap()[i].max(0.0)
        } else {
            1.0
        }
    };

    let mut basis = Vec::with_capacity(6);
    for d in 0..3 {
        let mut t = Array1::zeros(n);
        for j in 0..nat {
            t[3 * j + d] = 1.0;
        }
        basis.push(t);
    }
    if spin[0] || spin[1] || spin[2] {
        let mut com = [0.0; 3];
        let mut mtot = 0.0;
        for i in 0..nat {
            let mi = m_at(i);
            mtot += mi;
            com[0] += mi * pos[3 * i];
            com[1] += mi * pos[3 * i + 1];
            com[2] += mi * pos[3 * i + 2];
        }
        if mtot > 0.0 {
            com[0] /= mtot;
            com[1] /= mtot;
            com[2] /= mtot;
        }
        let mut rx = Array1::zeros(n);
        let mut ry = Array1::zeros(n);
        let mut rz = Array1::zeros(n);
        for i in 0..nat {
            let x = pos[3 * i] - com[0];
            let y = pos[3 * i + 1] - com[1];
            let z = pos[3 * i + 2] - com[2];
            rx[3 * i + 1] = -z;
            rx[3 * i + 2] = y;
            ry[3 * i] = z;
            ry[3 * i + 2] = -x;
            rz[3 * i] = -y;
            rz[3 * i + 1] = x;
        }
        if spin[0] {
            basis.push(rx);
        }
        if spin[1] {
            basis.push(ry);
        }
        if spin[2] {
            basis.push(rz);
        }
    }

    let mut ortho: Vec<Array1<f64>> = Vec::with_capacity(6);
    for v in basis {
        let mut u = v;
        for e in &ortho {
            let d = mass_dot(&u, e, nat, &m_at);
            // Subtract in the Euclidean chart; the coefficient used the
            // mass inner product so the result is M-orthogonal to e.
            let ee = mass_dot(e, e, nat, &m_at);
            if ee > 1.0e-18 {
                u.scaled_add(-d / ee, e);
            }
        }
        let nrm2 = mass_dot(&u, &u, nat, &m_at);
        if nrm2 > 1.0e-18 {
            u /= nrm2.sqrt();
            ortho.push(u);
        }
    }
    for e in &ortho {
        let d = mass_dot(vec, e, nat, &m_at);
        let ee = mass_dot(e, e, nat, &m_at);
        if ee > 1.0e-18 {
            vec.scaled_add(-d / ee, e);
        }
    }
}

fn mass_dot<F>(a: &Array1<f64>, b: &Array1<f64>, nat: usize, m_at: &F) -> f64
where
    F: Fn(usize) -> f64,
{
    let mut s = 0.0;
    for i in 0..nat {
        let mi = m_at(i);
        s += mi * (a[3 * i] * b[3 * i] + a[3 * i + 1] * b[3 * i + 1] + a[3 * i + 2] * b[3 * i + 2]);
    }
    s
}

struct Cloud<'a> {
    mobile: ArrayView1<'a, f64>,
    target: ArrayView1<'a, f64>,
    weights: Option<&'a [f64]>,
    nat: usize,
    com_m: [f64; 3],
    com_t: [f64; 3],
}

impl<'a> Cloud<'a> {
    fn try_new(
        mobile: ArrayView1<'a, f64>,
        target: ArrayView1<'a, f64>,
        weights: Option<&'a [f64]>,
    ) -> Option<Self> {
        let n = mobile.len();
        if n != target.len() || n == 0 || !n.is_multiple_of(3) {
            return None;
        }
        let nat = n / 3;
        if let Some(w) = weights
            && w.len() != nat
        {
            return None;
        }
        let mut com_m = [0.0; 3];
        let mut com_t = [0.0; 3];
        let mut wtot = 0.0;
        for i in 0..nat {
            let w = atom_weight(weights, i);
            wtot += w;
            for a in 0..3 {
                let m = mobile[3 * i + a];
                let t = target[3 * i + a];
                if !m.is_finite() || !t.is_finite() {
                    return None;
                }
                com_m[a] += w * m;
                com_t[a] += w * t;
            }
        }
        if wtot <= 0.0 || !wtot.is_finite() {
            return None;
        }
        for a in 0..3 {
            com_m[a] /= wtot;
            com_t[a] /= wtot;
            if !com_m[a].is_finite() || !com_t[a].is_finite() {
                return None;
            }
        }
        Some(Self {
            mobile,
            target,
            weights,
            nat,
            com_m,
            com_t,
        })
    }

    fn weight(&self, i: usize) -> f64 {
        atom_weight(self.weights, i)
    }

    fn centered(&self, i: usize) -> ([f64; 3], [f64; 3], f64) {
        let mut m = [0.0; 3];
        let mut t = [0.0; 3];
        for a in 0..3 {
            m[a] = self.mobile[3 * i + a] - self.com_m[a];
            t[a] = self.target[3 * i + a] - self.com_t[a];
        }
        (m, t, self.weight(i))
    }

    #[allow(clippy::needless_range_loop)]
    fn correlation(&self) -> [[f64; 3]; 3] {
        let mut c = [[0.0; 3]; 3];
        for i in 0..self.nat {
            let (m, t, w) = self.centered(i);
            for row in 0..3 {
                for col in 0..3 {
                    c[row][col] += w * m[row] * t[col];
                }
            }
        }
        c
    }

    fn rmsd(&self, rotation: [[f64; 3]; 3], translation: [f64; 3]) -> f64 {
        let mut acc = 0.0;
        let mut wtot = 0.0;
        for i in 0..self.nat {
            let w = self.weight(i);
            wtot += w;
            let p = [
                self.mobile[3 * i],
                self.mobile[3 * i + 1],
                self.mobile[3 * i + 2],
            ];
            let y = mat_vec(rotation, p);
            let dx = y[0] + translation[0] - self.target[3 * i];
            let dy = y[1] + translation[1] - self.target[3 * i + 1];
            let dz = y[2] + translation[2] - self.target[3 * i + 2];
            acc += w * (dx * dx + dy * dy + dz * dz);
        }
        if wtot <= 0.0 {
            return 0.0;
        }
        let mean = (acc / wtot).max(0.0);
        let r = mean.sqrt();
        if r.is_nan() { 0.0 } else { r }
    }
}

fn atom_weight(weights: Option<&[f64]>, i: usize) -> f64 {
    match weights {
        None => 1.0,
        Some(w) => {
            let v = w[i];
            if v.is_finite() { v.max(0.0) } else { 0.0 }
        }
    }
}

/// Leading gap, relative to the spectral radius, treated as one subspace.
const LEADING_GAP: f64 = 1e-8;

fn quaternion_full(cloud: &Cloud<'_>) -> [f64; 4] {
    let key = horn_key(&cloud.correlation());
    if key.iter().any(|row| row.iter().any(|c| !c.is_finite())) {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let (evals, evecs) = sym_eig_4(key);
    leading_quaternion(evals, evecs)
}

fn quaternion_about(cloud: &Cloud<'_>, axis: usize) -> [f64; 4] {
    // Right-handed pairs in the plane perpendicular to `axis`.
    let (i, j) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let mut along = 0.0;
    let mut across = 0.0;
    for n in 0..cloud.nat {
        let (m, t, w) = cloud.centered(n);
        along += w * (m[i] * t[i] + m[j] * t[j]);
        across += w * (m[i] * t[j] - m[j] * t[i]);
    }
    if !along.is_finite() || !across.is_finite() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let radius = along.hypot(across);
    let (cos_t, sin_t) = if radius > 0.0 && radius.is_finite() {
        (along / radius, across / radius)
    } else {
        (1.0, 0.0)
    };
    let c = cos_t.clamp(-1.0, 1.0);
    let half_c = ((1.0 + c) * 0.5).max(0.0).sqrt();
    let half_s = ((1.0 - c) * 0.5).max(0.0).sqrt().copysign(sin_t);
    let mut q = [half_c, 0.0, 0.0, 0.0];
    q[axis + 1] = half_s;
    q
}

/// Horn's key matrix. `c[i][j]` is \(\sum w\, m_i t_j\).
fn horn_key(c: &[[f64; 3]; 3]) -> [[f64; 4]; 4] {
    let (sxx, sxy, sxz) = (c[0][0], c[0][1], c[0][2]);
    let (syx, syy, syz) = (c[1][0], c[1][1], c[1][2]);
    let (szx, szy, szz) = (c[2][0], c[2][1], c[2][2]);
    [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
}

#[allow(clippy::needless_range_loop)]
fn leading_quaternion(evals: [f64; 4], vecs: [[f64; 4]; 4]) -> [f64; 4] {
    let mut lam = f64::NEG_INFINITY;
    let mut scale = 0.0_f64;
    for e in evals {
        if e.is_finite() {
            lam = lam.max(e);
            scale = scale.max(e.abs());
        }
    }
    if !lam.is_finite() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let tol = LEADING_GAP * scale;
    let mut basis = [[0.0; 4]; 4];
    let mut nbasis = 0;
    for k in 0..4 {
        if !evals[k].is_finite() || evals[k] < lam - tol {
            continue;
        }
        let mut v = vecs[k];
        for b in 0..nbasis {
            let d = dot4(v, basis[b]);
            for i in 0..4 {
                v[i] -= d * basis[b][i];
            }
        }
        let n2 = dot4(v, v);
        if n2 > 1e-24 && n2.is_finite() {
            let inv = 1.0 / n2.sqrt();
            for c in &mut v {
                *c *= inv;
            }
            basis[nbasis] = v;
            nbasis += 1;
        }
    }
    if nbasis == 0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let mut proj = [0.0; 4];
    for b in 0..nbasis {
        let coeff = basis[b][0];
        for i in 0..4 {
            proj[i] += coeff * basis[b][i];
        }
    }
    let n2 = dot4(proj, proj);
    if n2 > 1e-24 && n2.is_finite() {
        canonicalize(scale4(proj, 1.0 / n2.sqrt()))
    } else {
        canonicalize(basis[0])
    }
}

fn rotation_from_quaternion(q: [f64; 4]) -> [[f64; 3]; 3] {
    let q = canonicalize(q);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]
}

fn canonicalize(mut q: [f64; 4]) -> [f64; 4] {
    if q.iter().any(|c| !c.is_finite()) {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let n2 = dot4(q, q);
    if !n2.is_finite() || n2 <= 1e-24 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    q = scale4(q, 1.0 / n2.sqrt());
    let mut pivot = q[0];
    if pivot.abs() <= 1e-12 {
        pivot = 0.0;
        for c in q.iter().skip(1) {
            if c.abs() > 1e-12 {
                pivot = *c;
                break;
            }
        }
    }
    if pivot < 0.0 {
        q = scale4(q, -1.0);
    }
    q
}

fn dot4(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn scale4(mut q: [f64; 4], s: f64) -> [f64; 4] {
    for c in &mut q {
        *c *= s;
    }
    q
}

fn mat_vec(r: [[f64; 3]; 3], x: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * x[0] + r[0][1] * x[1] + r[0][2] * x[2],
        r[1][0] * x[0] + r[1][1] * x[1] + r[1][2] * x[2],
        r[2][0] * x[0] + r[2][1] * x[1] + r[2][2] * x[2],
    ]
}

const JACOBI_BIG_THETA: f64 = 1.0 / f64::EPSILON;

fn jacobi_tangent(theta: f64) -> f64 {
    if theta.abs() > JACOBI_BIG_THETA {
        return 0.5 / theta;
    }
    let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
    sign / (theta.abs() + (theta * theta + 1.0).sqrt())
}

/// Cyclic Jacobi. Columns of the returned basis are the eigenvectors.
#[allow(clippy::needless_range_loop)]
fn sym_eig_4(mut a: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for _ in 0..64 {
        let mut off = 0.0;
        for p in 0..4 {
            for q in (p + 1)..4 {
                off += a[p][q] * a[p][q];
            }
        }
        let mut scale = 1.0_f64;
        for d in 0..4 {
            scale = scale.max(a[d][d].abs());
        }
        if off.sqrt() <= 1e-15 * scale {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                let apq = a[p][q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = jacobi_tangent(theta);
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for i in 0..4 {
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip - s * aiq;
                    a[i][q] = s * aip + c * aiq;
                }
                for i in 0..4 {
                    let api = a[p][i];
                    let aqi = a[q][i];
                    a[p][i] = c * api - s * aqi;
                    a[q][i] = s * api + c * aqi;
                }
                for i in 0..4 {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }
    let evals = [a[0][0], a[1][1], a[2][2], a[3][3]];
    let mut cols = [[0.0; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            cols[col][row] = v[row][col];
        }
    }
    (evals, cols)
}

#[cfg(test)]
fn l2(v: &Array1<f64>) -> f64 {
    crate::vecops::nrm2(v.view())
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn drops_a_pure_translation() {
        let pos = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut v = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        project_out_rot_trans(&mut v, pos.view());
        assert!(l2(&v) < 1e-12);
    }

    #[test]
    fn unequal_masses_still_kill_translation() {
        let pos = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut v = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let masses = [16.0, 1.0, 1.0];
        project_horizontal(&mut v, pos.view(), Some(&masses), PeriodicAxes::ISOLATED);
        assert!(l2(&v) < 1e-12);
    }

    #[test]
    fn unequal_masses_kill_mass_weighted_rotation() {
        let pos = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
        let mut v = array![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let masses = [12.0, 1.0, 1.0];
        project_horizontal(&mut v, pos.view(), Some(&masses), PeriodicAxes::ISOLATED);
        assert!(l2(&v) < 1e-10, "{v:?}");
    }

    #[test]
    fn periodic_keeps_rotation_drops_translation() {
        let pos = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
        let rot = array![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let mut v = rot.clone();
        project_horizontal(&mut v, pos.view(), None, PeriodicAxes::CELL);
        assert!(l2(&v) > 0.5, "rotation was removed under PBC {v:?}");
        let mut t = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        project_horizontal(&mut t, pos.view(), None, PeriodicAxes::CELL);
        assert!(l2(&t) < 1e-12, "translation survived under PBC {t:?}");
    }

    #[test]
    fn one_periodic_axis_removes_only_that_rotation() {
        let pos = array![
            1.0, 0.2, 0.1, -0.4, 1.1, 0.3, -0.7, -0.6, 0.8, 0.2, -0.5, -1.2,
        ];
        let mut about_z = generator(&pos, 2, 1.0, None);
        let before = l2(&about_z);
        project_horizontal(
            &mut about_z,
            pos.view(),
            None,
            PeriodicAxes::new([false, false, true]),
        );
        assert!(l2(&about_z) < 1e-10, "z rotation survived {}", l2(&about_z));

        let mut about_x = generator(&pos, 0, 1.0, None);
        project_horizontal(
            &mut about_x,
            pos.view(),
            None,
            PeriodicAxes::new([false, false, true]),
        );
        assert!(
            l2(&about_x) > 0.5 * before,
            "x rotation was removed {}",
            l2(&about_x)
        );

        let mut shift_z = Array1::zeros(pos.len());
        for i in 0..pos.len() / 3 {
            shift_z[3 * i + 2] = 1.0;
        }
        project_horizontal(
            &mut shift_z,
            pos.view(),
            None,
            PeriodicAxes::new([false, false, true]),
        );
        assert!(l2(&shift_z) < 1e-12, "periodic translation survived");
    }

    #[test]
    fn known_rotation_and_translation_recover_to_1e_10() {
        let points = tetrahedron();
        let cases = [
            (
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0.3, -0.7, 1.1],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 0.0],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [-2.0, 0.4, 0.0],
            ),
        ];
        for (rot, shift) in cases {
            let mobile = transform(&points, rot, shift);
            let target = pack(&points);
            let fit = superpose(mobile.view(), target.view(), None, PeriodicAxes::ISOLATED);
            assert!(fit.rmsd < 1e-10, "rmsd {}", fit.rmsd);
            assert!(
                (det3(fit.rotation) - 1.0).abs() < 1e-8,
                "det {}",
                det3(fit.rotation)
            );
            assert!(finite_fit(&fit));
        }
    }

    #[test]
    fn random_rigid_motions_match_an_independent_kabsch() {
        let mut rng = StdRng::seed_from_u64(0x02_04);
        for _ in 0..12 {
            let points = random_cloud(&mut rng, 6);
            let axis = random_axis(&mut rng);
            let angle = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
            let rot = rodrigues(axis, angle);
            let shift = [
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
            ];
            let weights: Vec<f64> = (0..points.len())
                .map(|_| rng.random_range(0.2..3.0))
                .collect();
            let mobile_pts = apply_points(&points, rot, shift);
            let mobile = pack(&mobile_pts);
            let target = pack(&points);
            let fit = superpose(
                mobile.view(),
                target.view(),
                Some(&weights),
                PeriodicAxes::ISOLATED,
            );
            assert!(fit.rmsd < 1e-10, "quaternion rmsd {}", fit.rmsd);
            assert!((det3(fit.rotation) - 1.0).abs() < 1e-8);
            let kabsch = kabsch_rotation(&mobile_pts, &points, &weights);
            let kabsch_rmsd = weighted_rmsd(&mobile_pts, &points, &weights, kabsch);
            assert!(
                (fit.rmsd - kabsch_rmsd).abs() < 1e-8,
                "quaternion {} kabsch {}",
                fit.rmsd,
                kabsch_rmsd
            );
            assert!(
                matrix_gap(fit.rotation, kabsch) < 1e-8,
                "rotation gap {}",
                matrix_gap(fit.rotation, kabsch)
            );
        }
    }

    #[test]
    fn kabsch_and_quaternion_agree_on_a_noisy_cloud() {
        let mut rng = StdRng::seed_from_u64(0x0d6c);
        let points = random_cloud(&mut rng, 8);
        let rot = rodrigues(random_axis(&mut rng), 0.7);
        let mut moved = apply_points(&points, rot, [0.2, -0.1, 0.4]);
        moved[3][1] += 0.05;
        let weights = vec![1.0; points.len()];
        let fit = superpose(
            pack(&moved).view(),
            pack(&points).view(),
            Some(&weights),
            PeriodicAxes::ISOLATED,
        );
        let kabsch = kabsch_rotation(&moved, &points, &weights);
        let kabsch_rmsd = weighted_rmsd(&moved, &points, &weights, kabsch);
        assert!(
            (fit.rmsd - kabsch_rmsd).abs() < 1e-8,
            "{} vs {}",
            fit.rmsd,
            kabsch_rmsd
        );
        assert!((det3(fit.rotation) - 1.0).abs() < 1e-8);
        assert!(fit.rmsd > 1e-4);
    }

    #[test]
    fn a_reflection_stays_a_proper_rotation() {
        let points = tetrahedron();
        let mut image = points.clone();
        for p in &mut image {
            p[2] = -p[2];
        }
        let weights = vec![1.0; points.len()];
        let fit = superpose(
            pack(&image).view(),
            pack(&points).view(),
            None,
            PeriodicAxes::ISOLATED,
        );
        assert!(det3(fit.rotation) > 0.5, "det {}", det3(fit.rotation));
        assert!(fit.rmsd > 1e-3, "reflection fitted too well {}", fit.rmsd);
        let kabsch = kabsch_rotation(&image, &points, &weights);
        let kabsch_rmsd = weighted_rmsd(&image, &points, &weights, kabsch);
        assert!((fit.rmsd - kabsch_rmsd).abs() < 1e-8);
        assert!(finite_fit(&fit));
    }

    #[test]
    fn one_periodic_axis_is_a_planar_angle() {
        let points = tetrahedron();
        for axis in 0..3 {
            let mut mask = [false; 3];
            mask[axis] = true;
            let periodic = PeriodicAxes::new(mask);
            let rot = rodrigues(unit_axis(axis), 0.8);
            let mobile = transform(&points, rot, [0.4, -0.2, 0.7]);
            let fit = superpose(mobile.view(), pack(&points).view(), None, periodic);
            assert!(fit.rmsd < 1e-10, "axis {axis} rmsd {}", fit.rmsd);
            assert!(
                rotation_is_about(fit.rotation, axis),
                "axis {axis} {:?}",
                fit.rotation
            );

            let other = (axis + 1) % 3;
            let bad = rodrigues(unit_axis(other), 0.9);
            let moved = transform(&points, bad, [0.0, 0.0, 0.0]);
            let refused = superpose(moved.view(), pack(&points).view(), None, periodic);
            assert!(
                refused.rmsd > 1e-2,
                "axis {axis} removed a foreign rotation"
            );
            assert!(rotation_is_about(refused.rotation, axis));
            assert!(finite_fit(&refused));
        }
    }

    #[test]
    fn two_periodic_axes_forbid_rotation_and_still_translate() {
        let points = tetrahedron();
        let periodic = PeriodicAxes::new([true, true, false]);
        let shift = [0.2, -0.5, 0.8];
        let mobile = transform(&points, identity(), shift);
        let fit = superpose(mobile.view(), pack(&points).view(), None, periodic);
        assert!(fit.rmsd < 1e-10, "translation rmsd {}", fit.rmsd);
        assert_eq!(fit.quaternion, [1.0, 0.0, 0.0, 0.0]);

        let rot = rodrigues(unit_axis(2), 0.6);
        let moved = transform(&points, rot, shift);
        let refused = superpose(moved.view(), pack(&points).view(), None, periodic);
        assert!(refused.rmsd > 1e-2);
        assert_eq!(refused.quaternion, [1.0, 0.0, 0.0, 0.0]);
        assert!(finite_fit(&refused));
    }

    #[test]
    fn a_full_cell_matches_the_projector() {
        let pos = array![
            1.0, 0.2, 0.1, -0.4, 1.1, 0.3, -0.7, -0.6, 0.8, 0.2, -0.5, -1.2,
        ];
        let mut rot = generator(&pos, 2, 1.0e-4, None);
        let moved = &pos + &rot;
        project_horizontal(&mut rot, pos.view(), None, PeriodicAxes::CELL);
        assert!(l2(&rot) > 1e-5, "cell projector removed a rotation");
        let fit = superpose(moved.view(), pos.view(), None, PeriodicAxes::CELL);
        assert!(
            fit.rmsd > 1e-6,
            "cell alignment removed a rotation {}",
            fit.rmsd
        );
        assert_eq!(fit.quaternion, [1.0, 0.0, 0.0, 0.0]);

        let shift = Array1::from([0.25, -0.1, 0.4].repeat(pos.len() / 3));
        let translated = &pos + &shift;
        let fitted = superpose(translated.view(), pos.view(), None, PeriodicAxes::CELL);
        assert!(fitted.rmsd < 1e-10, "cell translation rmsd {}", fitted.rmsd);
    }

    #[test]
    fn weights_change_the_fit_and_recover_a_weighted_motion() {
        let points = tetrahedron();
        let weights = [4.0, 0.5, 0.25, 2.0];
        let rot = rodrigues([0.2, -0.5, 0.8], -1.2);
        let mobile = transform(&points, rot, [0.3, 0.1, -0.4]);
        let fit = superpose(
            mobile.view(),
            pack(&points).view(),
            Some(&weights),
            PeriodicAxes::ISOLATED,
        );
        assert!(fit.rmsd < 1e-10, "{}", fit.rmsd);

        let mut heavy = points.clone();
        heavy[1][0] += 0.4;
        let unit = superpose(
            pack(&heavy).view(),
            pack(&points).view(),
            None,
            PeriodicAxes::ISOLATED,
        );
        let weighted = superpose(
            pack(&heavy).view(),
            pack(&points).view(),
            Some(&[1e-6, 10.0, 1e-6, 1e-6]),
            PeriodicAxes::ISOLATED,
        );
        assert!(weighted.rmsd < unit.rmsd);
        assert!(
            weighted.rmsd < 0.05,
            "heavy atom was ignored {}",
            weighted.rmsd
        );
    }

    #[test]
    fn degenerate_inputs_are_finite_and_defined() {
        let one = array![0.4, -0.2, 0.7];
        let target = array![1.0, 1.0, -0.5];
        let fit = superpose(one.view(), target.view(), None, PeriodicAxes::ISOLATED);
        assert!(fit.rmsd < 1e-12);
        assert_eq!(fit.quaternion, [1.0, 0.0, 0.0, 0.0]);
        assert!(finite_fit(&fit));

        let line = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.5, 0.0, 0.0];
        let turned = transform(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            rodrigues(unit_axis(1), std::f64::consts::FRAC_PI_2),
            [0.2, -0.3, 0.4],
        );
        let collinear = superpose(turned.view(), line.view(), None, PeriodicAxes::ISOLATED);
        assert!(collinear.rmsd < 1e-10, "{}", collinear.rmsd);
        assert!((det3(collinear.rotation) - 1.0).abs() < 1e-8);
        assert!(finite_fit(&collinear));

        let mut near = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        near[2][1] = 1e-10;
        let near_fit = superpose(
            pack(&near).view(),
            pack(&near).view(),
            None,
            PeriodicAxes::ISOLATED,
        );
        assert!(near_fit.rmsd < 1e-12);
        assert!(finite_fit(&near_fit));
        assert!((det3(near_fit.rotation) - 1.0).abs() < 1e-6);

        let same = array![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let stacked = superpose(same.view(), same.view(), None, PeriodicAxes::ISOLATED);
        assert!(stacked.rmsd < 1e-12);
        assert_eq!(stacked.quaternion, [1.0, 0.0, 0.0, 0.0]);

        let zeros = [0.0, 0.0, 0.0];
        let unweighted = superpose(
            line.view(),
            line.view(),
            Some(&zeros),
            PeriodicAxes::ISOLATED,
        );
        assert_eq!(unweighted.quaternion, [1.0, 0.0, 0.0, 0.0]);
        assert!(finite_fit(&unweighted));

        let negative = superpose(
            line.view(),
            turned.view(),
            Some(&[-1.0, -2.0, 0.0]),
            PeriodicAxes::ISOLATED,
        );
        assert_eq!(negative.quaternion, [1.0, 0.0, 0.0, 0.0]);

        let mut dirty = line.clone();
        dirty[4] = f64::NAN;
        let nan_fit = superpose(dirty.view(), line.view(), None, PeriodicAxes::ISOLATED);
        assert!(finite_fit(&nan_fit));
        assert_eq!(nan_fit.quaternion, [1.0, 0.0, 0.0, 0.0]);

        let mismatch = superpose(line.view(), one.view(), None, PeriodicAxes::ISOLATED);
        assert_eq!(mismatch.quaternion, [1.0, 0.0, 0.0, 0.0]);
        let bad_w = superpose(
            line.view(),
            line.view(),
            Some(&[1.0]),
            PeriodicAxes::ISOLATED,
        );
        assert_eq!(bad_w.quaternion, [1.0, 0.0, 0.0, 0.0]);
        assert!(finite_fit(&bad_w));
    }

    #[test]
    fn tiny_rotations_match_the_projector() {
        let pos = array![
            1.0, 0.2, 0.1, -0.4, 1.1, 0.3, -0.7, -0.6, 0.8, 0.2, -0.5, -1.2,
        ];
        let masses = [3.0, 1.0, 0.5, 2.0];
        let angle = 1.0e-6;
        for axis in 0..3 {
            let mut infinitesimal = generator(&pos, axis, angle, Some(&masses));
            let moved = &pos + &infinitesimal;
            project_horizontal(
                &mut infinitesimal,
                pos.view(),
                Some(&masses),
                PeriodicAxes::ISOLATED,
            );
            assert!(
                l2(&infinitesimal) < 1e-9,
                "axis {axis} residual {}",
                l2(&infinitesimal)
            );
            let fit = superpose(
                moved.view(),
                pos.view(),
                Some(&masses),
                PeriodicAxes::ISOLATED,
            );
            assert!(fit.rmsd < 1e-10, "axis {axis} rmsd {}", fit.rmsd);
        }

        let mask = PeriodicAxes::new([false, false, true]);
        let mut allowed = generator(&pos, 2, angle, Some(&masses));
        let moved = &pos + &allowed;
        project_horizontal(&mut allowed, pos.view(), Some(&masses), mask);
        assert!(l2(&allowed) < 1e-9, "{}", l2(&allowed));
        let fit = superpose(moved.view(), pos.view(), Some(&masses), mask);
        assert!(fit.rmsd < 1e-10, "{}", fit.rmsd);

        let mut foreign = generator(&pos, 0, 1.0, Some(&masses));
        let foreign_norm = l2(&foreign);
        project_horizontal(&mut foreign, pos.view(), Some(&masses), mask);
        assert!(l2(&foreign) > 0.5 * foreign_norm);
        let finite_foreign = generator(&pos, 0, 0.2, Some(&masses));
        let displaced = &pos + &finite_foreign;
        let refused = superpose(displaced.view(), pos.view(), Some(&masses), mask);
        assert!(refused.rmsd > 1e-3, "{}", refused.rmsd);
        assert!(rotation_is_about(refused.rotation, 2));
    }

    #[test]
    fn horn_matrix_trace_is_zero_and_jacobi_reads_a_diagonal() {
        let c = [[1.0, 0.2, -0.3], [0.4, -0.5, 0.6], [0.1, 0.7, 0.8]];
        let key = horn_key(&c);
        let trace = key[0][0] + key[1][1] + key[2][2] + key[3][3];
        assert!(trace.abs() < 1e-12, "{trace}");
        let diag = [
            [1.5, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, -3.0],
        ];
        let (evals, _) = sym_eig_4(diag);
        let mut got = evals;
        got.sort_by(f64::total_cmp);
        let mut want = [1.5, -0.5, 2.0, -3.0];
        want.sort_by(f64::total_cmp);
        for i in 0..4 {
            assert!((got[i] - want[i]).abs() < 1e-12, "{got:?}");
        }
    }

    fn tetrahedron() -> Vec<[f64; 3]> {
        vec![
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    }

    fn identity() -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    fn unit_axis(axis: usize) -> [f64; 3] {
        let mut k = [0.0; 3];
        k[axis] = 1.0;
        k
    }

    fn pack(points: &[[f64; 3]]) -> Array1<f64> {
        let mut out = Array1::zeros(points.len() * 3);
        for (i, p) in points.iter().enumerate() {
            out[3 * i] = p[0];
            out[3 * i + 1] = p[1];
            out[3 * i + 2] = p[2];
        }
        out
    }

    fn transform(points: &[[f64; 3]], rot: [[f64; 3]; 3], shift: [f64; 3]) -> Array1<f64> {
        pack(&apply_points(points, rot, shift))
    }

    fn apply_points(points: &[[f64; 3]], rot: [[f64; 3]; 3], shift: [f64; 3]) -> Vec<[f64; 3]> {
        points
            .iter()
            .map(|p| {
                let y = mat_vec(rot, *p);
                [y[0] + shift[0], y[1] + shift[1], y[2] + shift[2]]
            })
            .collect()
    }

    fn rodrigues(axis: [f64; 3], theta: f64) -> [[f64; 3]; 3] {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let k = [axis[0] / n, axis[1] / n, axis[2] / n];
        let c = theta.cos();
        let s = theta.sin();
        let v = 1.0 - c;
        [
            [
                c + k[0] * k[0] * v,
                k[0] * k[1] * v - k[2] * s,
                k[0] * k[2] * v + k[1] * s,
            ],
            [
                k[1] * k[0] * v + k[2] * s,
                c + k[1] * k[1] * v,
                k[1] * k[2] * v - k[0] * s,
            ],
            [
                k[2] * k[0] * v - k[1] * s,
                k[2] * k[1] * v + k[0] * s,
                c + k[2] * k[2] * v,
            ],
        ]
    }

    fn random_axis(rng: &mut StdRng) -> [f64; 3] {
        loop {
            let k = [
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ];
            let n2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
            if n2 > 1e-6 {
                return k;
            }
        }
    }

    fn random_cloud(rng: &mut StdRng, n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|_| {
                [
                    rng.random_range(-1.5..1.5),
                    rng.random_range(-1.5..1.5),
                    rng.random_range(-1.5..1.5),
                ]
            })
            .collect()
    }

    fn generator(
        pos: &Array1<f64>,
        axis: usize,
        angle: f64,
        masses: Option<&[f64]>,
    ) -> Array1<f64> {
        let nat = pos.len() / 3;
        let mut com = [0.0; 3];
        let mut wtot = 0.0;
        for i in 0..nat {
            let w = masses.map(|m| m[i].max(0.0)).unwrap_or(1.0);
            wtot += w;
            for a in 0..3 {
                com[a] += w * pos[3 * i + a];
            }
        }
        if wtot > 0.0 {
            for a in 0..3 {
                com[a] /= wtot;
            }
        }
        let mut v = Array1::zeros(pos.len());
        for i in 0..nat {
            let x = pos[3 * i] - com[0];
            let y = pos[3 * i + 1] - com[1];
            let z = pos[3 * i + 2] - com[2];
            let (vx, vy, vz) = match axis {
                0 => (0.0, -z, y),
                1 => (z, 0.0, -x),
                _ => (-y, x, 0.0),
            };
            v[3 * i] = angle * vx;
            v[3 * i + 1] = angle * vy;
            v[3 * i + 2] = angle * vz;
        }
        v
    }

    fn det3(r: [[f64; 3]; 3]) -> f64 {
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    }

    fn finite_fit(fit: &Superposition) -> bool {
        fit.quaternion.iter().all(|c| c.is_finite())
            && fit.translation.iter().all(|c| c.is_finite())
            && fit
                .rotation
                .iter()
                .all(|row| row.iter().all(|c| c.is_finite()))
            && fit.rmsd.is_finite()
    }

    fn rotation_is_about(r: [[f64; 3]; 3], axis: usize) -> bool {
        for a in 0..3 {
            for b in 0..3 {
                if a == axis || b == axis {
                    let want = if a == b { 1.0 } else { 0.0 };
                    if (r[a][b] - want).abs() > 1e-8 {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn matrix_gap(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> f64 {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let d = a[i][j] - b[i][j];
                s += d * d;
            }
        }
        s.sqrt()
    }

    fn weighted_rmsd(
        mobile: &[[f64; 3]],
        target: &[[f64; 3]],
        weights: &[f64],
        rotation: [[f64; 3]; 3],
    ) -> f64 {
        // Independent of `superpose`: the same centering, then `R`.
        let mut cm = [0.0; 3];
        let mut ct = [0.0; 3];
        let mut wtot = 0.0;
        for i in 0..mobile.len() {
            let w = weights[i].max(0.0);
            wtot += w;
            for a in 0..3 {
                cm[a] += w * mobile[i][a];
                ct[a] += w * target[i][a];
            }
        }
        for a in 0..3 {
            cm[a] /= wtot;
            ct[a] /= wtot;
        }
        let shifted = mat_vec(rotation, cm);
        let translation = [ct[0] - shifted[0], ct[1] - shifted[1], ct[2] - shifted[2]];
        let mut acc = 0.0;
        for i in 0..mobile.len() {
            let y = mat_vec(rotation, mobile[i]);
            let d0 = y[0] + translation[0] - target[i][0];
            let d1 = y[1] + translation[1] - target[i][1];
            let d2 = y[2] + translation[2] - target[i][2];
            acc += weights[i].max(0.0) * (d0 * d0 + d1 * d1 + d2 * d2);
        }
        (acc / wtot).max(0.0).sqrt()
    }

    /// Independent Kabsch SVD. `M_ij = Σ w t_i m_j`, `R = U diag(1, 1, ±1) V^T`.
    fn kabsch_rotation(mobile: &[[f64; 3]], target: &[[f64; 3]], weights: &[f64]) -> [[f64; 3]; 3] {
        let nat = mobile.len();
        let mut cm = [0.0; 3];
        let mut ct = [0.0; 3];
        let mut wtot = 0.0;
        for i in 0..nat {
            let w = weights[i].max(0.0);
            wtot += w;
            for a in 0..3 {
                cm[a] += w * mobile[i][a];
                ct[a] += w * target[i][a];
            }
        }
        for a in 0..3 {
            cm[a] /= wtot;
            ct[a] /= wtot;
        }
        let mut cov = [[0.0; 3]; 3];
        for n in 0..nat {
            let w = weights[n].max(0.0);
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += w * (target[n][i] - ct[i]) * (mobile[n][j] - cm[j]);
                }
            }
        }
        let mut ata = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    ata[i][j] += cov[k][i] * cov[k][j];
                }
            }
        }
        let (evals, evecs) = jacobi3(ata);
        let mut order = [0usize, 1, 2];
        order.sort_by(|&i, &j| evals[j].total_cmp(&evals[i]));
        let mut v = [[0.0; 3]; 3];
        let mut sigma = [0.0; 3];
        for col in 0..3 {
            let src = order[col];
            sigma[col] = evals[src].max(0.0).sqrt();
            for row in 0..3 {
                v[row][col] = evecs[src][row];
            }
        }
        let mut u = [[0.0; 3]; 3];
        let mut rank = 0;
        let floor = 1e-10 * sigma[0].max(1.0);
        for col in 0..3 {
            if sigma[col] > floor {
                rank += 1;
                for row in 0..3 {
                    let mut acc = 0.0;
                    for k in 0..3 {
                        acc += cov[row][k] * v[k][col];
                    }
                    u[row][col] = acc / sigma[col];
                }
            }
        }
        complete_frame(&mut u, rank);
        if det3(u) * det3(v) < 0.0 {
            for row in 0..3 {
                u[row][2] = -u[row][2];
            }
        }
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = u[i][0] * v[j][0] + u[i][1] * v[j][1] + u[i][2] * v[j][2];
            }
        }
        r
    }

    fn complete_frame(u: &mut [[f64; 3]; 3], rank: usize) {
        let col = |m: &[[f64; 3]; 3], c: usize| [m[0][c], m[1][c], m[2][c]];
        let put = |m: &mut [[f64; 3]; 3], c: usize, v: [f64; 3]| {
            m[0][c] = v[0];
            m[1][c] = v[1];
            m[2][c] = v[2];
        };
        if rank == 0 {
            *u = identity();
            return;
        }
        if rank == 1 {
            let u0 = col(u, 0);
            let reference = if u0[0].abs() <= u0[1].abs() && u0[0].abs() <= u0[2].abs() {
                [1.0, 0.0, 0.0]
            } else if u0[1].abs() <= u0[2].abs() {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            let u1 = normalize3(cross(u0, reference));
            put(u, 1, u1);
        }
        if rank <= 2 {
            let u2 = normalize3(cross(col(u, 0), col(u, 1)));
            put(u, 2, u2);
        }
    }

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn normalize3(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n > 0.0 {
            [v[0] / n, v[1] / n, v[2] / n]
        } else {
            [1.0, 0.0, 0.0]
        }
    }

    fn jacobi3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
        let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for _ in 0..32 {
            let mut off = 0.0;
            for p in 0..3 {
                for q in (p + 1)..3 {
                    off += a[p][q] * a[p][q];
                }
            }
            if off.sqrt() <= 1e-15 {
                break;
            }
            for p in 0..3 {
                for q in (p + 1)..3 {
                    let apq = a[p][q];
                    if apq.abs() <= 1e-18 {
                        continue;
                    }
                    let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                    let t = if theta >= 0.0 { 1.0 } else { -1.0 }
                        / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for i in 0..3 {
                        let aip = a[i][p];
                        let aiq = a[i][q];
                        a[i][p] = c * aip - s * aiq;
                        a[i][q] = s * aip + c * aiq;
                    }
                    for i in 0..3 {
                        let api = a[p][i];
                        let aqi = a[q][i];
                        a[p][i] = c * api - s * aqi;
                        a[q][i] = s * api + c * aqi;
                    }
                    for i in 0..3 {
                        let vip = v[i][p];
                        let viq = v[i][q];
                        v[i][p] = c * vip - s * viq;
                        v[i][q] = s * vip + c * viq;
                    }
                }
            }
        }
        let mut cols = [[0.0; 3]; 3];
        for col in 0..3 {
            for row in 0..3 {
                cols[col][row] = v[row][col];
            }
        }
        ([a[0][0], a[1][1], a[2][2]], cols)
    }
}
