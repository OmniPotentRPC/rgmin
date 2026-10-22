//! Horizontal space of \(R^{3N}/\mathrm{SE}(3)\).
//!
//! Sella Cartesian PES (`fix_translation` + `fix_rotation`; Hermes,
//! Sarsfield, Zádor, JCTC 2022, doi:10.1021/acs.jctc.2c00395). Masses
//! turn the inner product into the Page–McIver / Eckart metric used
//! by Sella IRC and gpr_optim `IRCDriver` (Page and McIver 1988,
//! doi:10.1063/1.454172).

use ndarray::{Array1, ArrayView1};

/// eOn `projectOutRotTrans`. Unit-mass Eckart. Isolated molecule (T+R).
pub(crate) fn project_out_rot_trans(vec: &mut Array1<f64>, pos: ArrayView1<f64>) {
    project_horizontal(vec, pos, None, true);
}

/// Horizontal space of \(R^{3N}/\mathrm{SE}(3)\) or \(R^{3N}/T(3)\).
///
/// `rotate` is Sella `proj_rot`: false under PBC (the cell kills
/// rotational invariance). Masses are per atom (length N).
pub(crate) fn project_horizontal(
    vec: &mut Array1<f64>,
    pos: ArrayView1<f64>,
    masses: Option<&[f64]>,
    rotate: bool,
) {
    let n = pos.len();
    if n < 3 || n % 3 != 0 || vec.len() != n {
        return;
    }
    if rotate && n < 6 {
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

    let mut basis = Vec::with_capacity(if rotate { 6 } else { 3 });
    for d in 0..3 {
        let mut t = Array1::zeros(n);
        for j in 0..nat {
            t[3 * j + d] = 1.0;
        }
        basis.push(t);
    }
    if rotate {
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
        basis.push(rx);
        basis.push(ry);
        basis.push(rz);
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

#[cfg(test)]
fn l2(v: &Array1<f64>) -> f64 {
    crate::vecops::nrm2(v.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
        project_horizontal(&mut v, pos.view(), Some(&masses), true);
        assert!(l2(&v) < 1e-12);
    }

    #[test]
    fn unequal_masses_kill_mass_weighted_rotation() {
        // Infinitesimal rotation about z of a planar triangle.
        let pos = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
        let mut v = array![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let masses = [12.0, 1.0, 1.0];
        project_horizontal(&mut v, pos.view(), Some(&masses), true);
        assert!(l2(&v) < 1e-10, "{v:?}");
    }

    #[test]
    fn periodic_keeps_rotation_drops_translation() {
        let pos = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
        let rot = array![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let mut v = rot.clone();
        project_horizontal(&mut v, pos.view(), None, false);
        assert!(l2(&v) > 0.5, "rotation was removed under PBC {v:?}");
        let mut t = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        project_horizontal(&mut t, pos.view(), None, false);
        assert!(l2(&t) < 1e-12, "translation survived under PBC {t:?}");
    }
}
