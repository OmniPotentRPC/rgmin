//! Variable-cell filter: atomic positions and a log-strain cell.
//!
//! The coordinates are the undeformed positions plus the matrix
//! logarithm of the deformation gradient, the parameterisation of
//! ASE's `FrechetCellFilter` (the corrected `ExpCellFilter`). The
//! cell gradient is the Fréchet derivative of that exponential, so a
//! finite difference of the enthalpy matches the gradient. A
//! hydrostatic mask keeps only the trace of the virial, and a
//! fixed-angle mask drops the shear components. An external pressure
//! is added to the Cauchy stress before the virial is formed, and the
//! objective value is the enthalpy `E + pV`.
//!
//! The potential is a [`Mechanical`] value. This crate does not link
//! a potential library: a host that can call
//! `rgpot_potential_calculate_mechanical` implements the trait with
//! the stress that call returns.
//!
//! Tadmor, Smith, Bernstein and Kaxiras, Phys. Rev. B 59, 235 (1999),
//! <https://doi.org/10.1103/PhysRevB.59.235>. The Fréchet form follows
//! ASE `ase/filters.py` and Ortner's JuLIP exponential cell.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};

/// One evaluation: energy in eV, Cartesian forces in eV/Å, Cauchy
/// stress in eV/Å³ with ASE's sign (`sigma = -W/V`).
#[derive(Clone, Debug)]
pub struct Mechanics {
    /// Potential energy, without the pressure term.
    pub energy: f64,
    /// Force on each atom, length `3 * n`, matching the position order.
    pub forces: Vec<f64>,
    /// Row-major Cauchy stress `xx, xy, xz, yx, yy, yz, zx, zy, zz`.
    pub stress: [f64; 9],
}

/// Energy, Cartesian forces and Cauchy stress of one cell.
pub trait Mechanical: Send + Sync {
    /// Evaluate `positions` (length `3n`, Å) in `cell` (row-major
    /// lattice vectors, Å).
    fn evaluate(&self, positions: &[f64], cell: &[f64; 9]) -> Result<Mechanics>;
}

impl<M: Mechanical + ?Sized> Mechanical for &M {
    fn evaluate(&self, positions: &[f64], cell: &[f64; 9]) -> Result<Mechanics> {
        (*self).evaluate(positions, cell)
    }
}

/// Which strain components the virial is allowed to move.
///
/// Voigt order is `xx, yy, zz, yz, xz, xy`, the ASE mask order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellMask {
    /// All six independent components.
    Full,
    /// Isotropic strain. The virial is replaced by its trace.
    Hydrostatic,
    /// Diagonal strain only. Shear (`yz`, `xz`, `xy`) is held.
    FixedAngles,
}

type Mat = [[f64; 3]; 3];

/// Combined atomic and cell objective.
///
/// The vector is `3n + 9` long. The first `3n` entries are positions
/// in the reference cell (the Cartesian positions with the deformation
/// undone). The last nine are `factor * log(U)`, row-major, where `U`
/// is the deformation gradient from the reference cell.
pub struct CellFilter<M> {
    model: M,
    n: usize,
    orig: Mat,
    /// 1 where a Voigt component may move, 0 where it is held.
    mask: [f64; 6],
    hydrostatic: bool,
    pressure: f64,
    factor: f64,
    bounds: Bounds<f64>,
}

impl<M> CellFilter<M> {
    /// Full cell relaxation at zero pressure. `cell` is row-major.
    /// `factor` scales the cell block the way ASE's `exp_cell_factor`
    /// does; `None` uses the atom count.
    pub fn new(model: M, n_atoms: usize, cell: [f64; 9], factor: Option<f64>) -> Result<Self> {
        if n_atoms == 0 {
            return Err(Error::Oracle { what: "no atoms" });
        }
        let orig = mat_from(cell);
        if !det(orig).is_finite() || det(orig).abs() < 1e-18 {
            return Err(Error::Oracle {
                what: "reference cell is singular",
            });
        }
        let factor = factor.unwrap_or(n_atoms as f64);
        if factor <= 0.0 || !factor.is_finite() {
            return Err(Error::Oracle {
                what: "cell factor must be positive",
            });
        }
        let dim = 3 * n_atoms + 9;
        const SPAN: f64 = 1.0e6;
        Ok(Self {
            model,
            n: n_atoms,
            orig,
            mask: [1.0; 6],
            hydrostatic: false,
            pressure: 0.0,
            factor,
            bounds: Bounds::new(
                Array1::from_elem(dim, -SPAN),
                Array1::from_elem(dim, SPAN),
                0.0,
            ),
        })
    }

    /// Replace the strain mask.
    pub fn mask(mut self, mask: CellMask) -> Self {
        match mask {
            CellMask::Full => {
                self.mask = [1.0; 6];
                self.hydrostatic = false;
            }
            CellMask::Hydrostatic => {
                self.mask = [1.0; 6];
                self.hydrostatic = true;
            }
            CellMask::FixedAngles => {
                self.mask = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
                self.hydrostatic = false;
            }
        }
        self
    }

    /// External pressure in eV/Å³, positive in compression. The
    /// objective becomes the enthalpy `E + pV`.
    pub fn pressure(mut self, pressure: f64) -> Self {
        self.pressure = pressure;
        self
    }

    /// Reference cell, row-major.
    pub fn reference_cell(&self) -> [f64; 9] {
        mat_to(self.orig)
    }

    /// Pack Cartesian `positions` in `cell` as filter coordinates.
    pub fn coordinates(&self, positions: &[f64], cell: &[f64; 9]) -> Result<Array1<f64>> {
        if positions.len() != 3 * self.n {
            return Err(Error::Dim {
                got: positions.len(),
                dim: 3 * self.n,
            });
        }
        let u = deform(self.orig, mat_from(*cell))?;
        let log_u = logm(u)?;
        let u_inv_t = transpose(invert(u)?);
        let mut x = Array1::zeros(3 * self.n + 9);
        for a in 0..self.n {
            let p = [positions[3 * a], positions[3 * a + 1], positions[3 * a + 2]];
            let r = mul_vec(u_inv_t, p);
            x[3 * a] = r[0];
            x[3 * a + 1] = r[1];
            x[3 * a + 2] = r[2];
        }
        for i in 0..3 {
            for j in 0..3 {
                x[3 * self.n + 3 * i + j] = self.factor * log_u[i][j];
            }
        }
        Ok(x)
    }

    /// Cartesian positions and the current cell of a coordinate vector.
    pub fn geometry(&self, x: ArrayView1<f64>) -> Result<(Vec<f64>, [f64; 9])> {
        self.realize(x).map(|(p, c, _)| (p, mat_to(c)))
    }

    /// Enthalpy and its gradient at filter coordinates `x`.
    pub fn enthalpy_gradient(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>)>
    where
        M: Mechanical,
    {
        self.sample(x)
    }

    /// Cauchy stress of a coordinate vector, from the model, not from
    /// the masked optimizer gradient.
    pub fn stress(&self, x: ArrayView1<f64>) -> Result<[f64; 9]>
    where
        M: Mechanical,
    {
        let (pos, cell, _) = self.realize(x)?;
        Ok(self.model.evaluate(&pos, &mat_to(cell))?.stress)
    }

    fn realize(&self, x: ArrayView1<f64>) -> Result<(Vec<f64>, Mat, Mat)> {
        let dim = 3 * self.n + 9;
        if x.len() != dim {
            return Err(Error::Dim { got: x.len(), dim });
        }
        let mut log_u = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                log_u[i][j] = x[3 * self.n + 3 * i + j] / self.factor;
            }
        }
        let u = expm(log_u);
        let ut = transpose(u);
        let cell = mul(self.orig, ut);
        let mut pos = vec![0.0; 3 * self.n];
        for a in 0..self.n {
            let r = [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
            let p = mul_vec(ut, r);
            pos[3 * a] = p[0];
            pos[3 * a + 1] = p[1];
            pos[3 * a + 2] = p[2];
        }
        Ok((pos, cell, u))
    }
}

impl<M: Mechanical> CellFilter<M> {
    fn sample(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>)> {
        let (pos, cell, u) = self.realize(x)?;
        let got = self.model.evaluate(&pos, &mat_to(cell))?;
        if got.forces.len() != 3 * self.n {
            return Err(Error::Oracle {
                what: "force length",
            });
        }
        if !got.energy.is_finite() || got.forces.iter().any(|f| !f.is_finite()) {
            return Err(Error::Oracle {
                what: "non-finite energy or force",
            });
        }
        if got.stress.iter().any(|s| !s.is_finite()) {
            return Err(Error::Oracle {
                what: "non-finite stress",
            });
        }
        let volume = det(cell).abs();
        let mut stress = mat_from(got.stress);
        for (i, row) in stress.iter_mut().enumerate() {
            row[i] += self.pressure;
        }
        let mut virial = scale(stress, -volume);
        if self.hydrostatic {
            let trace = (virial[0][0] + virial[1][1] + virial[2][2]) / 3.0;
            virial = [[trace, 0.0, 0.0], [0.0, trace, 0.0], [0.0, 0.0, trace]];
        }
        apply_voigt_mask(&mut virial, self.mask);
        let ut_inv = invert(transpose(u))?;
        let ucf = mul(virial, ut_inv);
        let log_u = logm(u)?;
        let mut cell_force = [[0.0; 3]; 3];
        for mu in 0..3 {
            for nu in 0..3 {
                let mut dir = [[0.0; 3]; 3];
                dir[mu][nu] = 1.0;
                let deriv = expm_frechet(log_u, dir)?;
                cell_force[mu][nu] = frobenius(deriv, ucf);
            }
        }
        let mut grad = Array1::zeros(3 * self.n + 9);
        for a in 0..self.n {
            let f = [
                got.forces[3 * a],
                got.forces[3 * a + 1],
                got.forces[3 * a + 2],
            ];
            let reduced = mul_vec(u, f);
            grad[3 * a] = -reduced[0];
            grad[3 * a + 1] = -reduced[1];
            grad[3 * a + 2] = -reduced[2];
        }
        for i in 0..3 {
            for j in 0..3 {
                grad[3 * self.n + 3 * i + j] = -cell_force[i][j] / self.factor;
            }
        }
        Ok((got.energy + self.pressure * volume, grad))
    }
}

impl<M: Mechanical> Objective<f64> for CellFilter<M> {
    fn dim(&self) -> usize {
        3 * self.n + 9
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.sample(x).map(|(v, _)| v).unwrap_or(f64::NAN)
    }
}

impl<M: Mechanical> Gradient<f64> for CellFilter<M> {
    fn dim(&self) -> usize {
        3 * self.n + 9
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.sample(x)
            .map(|(_, g)| g)
            .unwrap_or_else(|_| Array1::from_elem(3 * self.n + 9, f64::NAN))
    }
}

impl<M: Mechanical> DifferentiableObjective<f64> for CellFilter<M> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self.sample(x) {
            Ok(pair) => pair,
            Err(_) => (f64::NAN, Array1::from_elem(3 * self.n + 9, f64::NAN)),
        }
    }
}

fn apply_voigt_mask(v: &mut Mat, mask: [f64; 6]) {
    let full = [
        [mask[0], mask[5], mask[4]],
        [mask[5], mask[1], mask[3]],
        [mask[4], mask[3], mask[2]],
    ];
    for i in 0..3 {
        for j in 0..3 {
            v[i][j] *= full[i][j];
        }
    }
}

fn mat_from(v: [f64; 9]) -> Mat {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

fn mat_to(m: Mat) -> [f64; 9] {
    [
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    ]
}

fn deform(orig: Mat, current: Mat) -> Result<Mat> {
    let inv = invert(orig)?;
    Ok(transpose(mul(inv, current)))
}

fn det(m: Mat) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn invert(m: Mat) -> Result<Mat> {
    let d = det(m);
    if !d.is_finite() || d.abs() < 1e-18 {
        return Err(Error::Oracle {
            what: "singular matrix",
        });
    }
    let s = 1.0 / d;
    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * s,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * s,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * s,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * s,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * s,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * s,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * s,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * s,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * s,
        ],
    ])
}

fn transpose(m: Mat) -> Mat {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mul(a: Mat, b: Mat) -> Mat {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn mul_vec(m: Mat, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn scale(m: Mat, s: f64) -> Mat {
    let mut c = m;
    for row in &mut c {
        for v in row {
            *v *= s;
        }
    }
    c
}

fn add(a: Mat, b: Mat) -> Mat {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

fn eye() -> Mat {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn frobenius(a: Mat, b: Mat) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += a[i][j] * b[i][j];
        }
    }
    s
}

fn inf_norm(m: &Mat) -> f64 {
    let mut n = 0.0_f64;
    for row in m {
        let s: f64 = row.iter().map(|v| v.abs()).sum();
        n = n.max(s);
    }
    n
}

fn expm(a: Mat) -> Mat {
    let mut m = a;
    let mut s = 0u32;
    let mut norm = inf_norm(&m);
    while norm > 0.25 && s < 16 {
        m = scale(m, 0.5);
        norm *= 0.5;
        s += 1;
    }
    let mut term = eye();
    let mut sum = eye();
    for k in 1..=20 {
        term = scale(mul(term, m), 1.0 / k as f64);
        sum = add(sum, term);
    }
    for _ in 0..s {
        sum = mul(sum, sum);
    }
    sum
}

fn logm(a: Mat) -> Result<Mat> {
    let mut m = a;
    let mut k = 0u32;
    while inf_norm(&add(m, scale(eye(), -1.0))) > 0.25 && k < 16 {
        m = sqrtm(m)?;
        k += 1;
    }
    let x = add(m, scale(eye(), -1.0));
    let mut term = x;
    let mut sum = x;
    for n in 2..=24 {
        term = mul(term, x);
        let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
        sum = add(sum, scale(term, sign / n as f64));
    }
    Ok(scale(sum, (1u32 << k) as f64))
}

fn sqrtm(a: Mat) -> Result<Mat> {
    let mut y = a;
    let mut z = eye();
    for _ in 0..24 {
        let y_inv = invert(y)?;
        let y_next = scale(add(y, invert(z)?), 0.5);
        let z_next = scale(add(z, y_inv), 0.5);
        let shift = inf_norm(&add(y_next, scale(y, -1.0)));
        y = y_next;
        z = z_next;
        if shift < 1e-14 {
            break;
        }
    }
    Ok(y)
}

fn expm_frechet(a: Mat, e: Mat) -> Result<Mat> {
    // exp([[A, E], [0, A]]) = [[exp(A), L], [0, exp(A)]], L = d exp_A (E).
    let mut block = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            block[i][j] = a[i][j];
            block[i + 3][j + 3] = a[i][j];
            block[i][j + 3] = e[i][j];
        }
    }
    let exp = expm6(&block);
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            l[i][j] = exp[i][j + 3];
        }
    }
    Ok(l)
}

fn expm6(a: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut m = *a;
    let mut s = 0u32;
    let mut norm = inf6(&m);
    while norm > 0.25 && s < 16 {
        m = scale6(&m, 0.5);
        norm *= 0.5;
        s += 1;
    }
    let mut term = eye6();
    let mut sum = eye6();
    for k in 1..=20 {
        term = scale6(&mul6(&term, &m), 1.0 / k as f64);
        sum = add6(&sum, &term);
    }
    for _ in 0..s {
        sum = mul6(&sum, &sum);
    }
    sum
}

fn eye6() -> [[f64; 6]; 6] {
    let mut m = [[0.0; 6]; 6];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

fn inf6(m: &[[f64; 6]; 6]) -> f64 {
    let mut n = 0.0_f64;
    for row in m {
        let s: f64 = row.iter().map(|v| v.abs()).sum();
        n = n.max(s);
    }
    n
}

fn scale6(m: &[[f64; 6]; 6], s: f64) -> [[f64; 6]; 6] {
    let mut c = *m;
    for row in &mut c {
        for v in row {
            *v *= s;
        }
    }
    c
}

fn add6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut c = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

fn mul6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut c = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_log_round_trip() {
        let a = [[1.2, 0.1, 0.0], [0.05, 0.9, -0.02], [0.0, 0.03, 1.1]];
        let back = expm(logm(a).unwrap());
        for i in 0..3 {
            for j in 0..3 {
                assert!((back[i][j] - a[i][j]).abs() < 1e-8, "{back:?}");
            }
        }
    }

    #[test]
    fn frechet_matches_a_difference() {
        let a = [[0.1, -0.02, 0.0], [0.01, -0.05, 0.03], [0.0, 0.02, 0.08]];
        let mut e = [[0.0; 3]; 3];
        e[0][1] = 1.0;
        let h = 1e-6;
        let num = scale(
            add(
                expm(add(a, scale(e, h))),
                scale(expm(add(a, scale(e, -h))), -1.0),
            ),
            0.5 / h,
        );
        let got = expm_frechet(a, e).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((got[i][j] - num[i][j]).abs() < 1e-6, "{got:?} {num:?}");
            }
        }
    }
}
