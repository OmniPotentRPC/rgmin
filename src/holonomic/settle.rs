//! Analytic SETTLE for a rigid triangle with any masses.
//!
//! Corrections along the anchor bonds lie in the anchor plane, so every
//! atom keeps its coordinate along the anchor normal and the centre of
//! mass stays where the trial put it. They are central forces between
//! the anchor positions, so their torque about the normal, taken with the
//! anchor positions, is zero. The rigid triangle that satisfies these
//! three conditions is the converged SHAKE solution, found here in
//! closed form as Miyamoto and Kollman (1992) do for water:
//! the normal coordinates fix the tilt of the triangle, and the torque
//! fixes its in-plane angle through `C cos(theta) + D sin(theta) = E`.
//! Of the two roots, the one nearer the trial in the mass metric is
//! taken. GROMACS `mdlib/settle.cpp` handles water with two equal
//! hydrogen masses and equal O-H lengths; nothing here needs either.

use ndarray::Array1;

use super::{ConstraintError, Tangent, Triangle};
use crate::error::Result;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scaled(a: V3, s: f64) -> V3 {
    a.map(|c| c * s)
}

fn atom(x: &Array1<f64>, i: usize) -> V3 {
    [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
}

fn centre(p: &[V3; 3], m: &[f64; 3]) -> V3 {
    let total = m[0] + m[1] + m[2];
    let mut c = [0.0; 3];
    for (p, m) in p.iter().zip(m) {
        (0..3).for_each(|k| c[k] += m * p[k] / total);
    }
    c
}

impl Tangent<'_> {
    /// Move the triangle `t` of `x` onto its lengths from this anchor.
    /// The triangle is written only once all three atoms are finite.
    pub(super) fn settle(&self, t: &Triangle, x: &mut Array1<f64>) -> Result<()> {
        let set = self.set;
        let degenerate = || ConstraintError::SettleDegenerate { atoms: t.atoms };
        let m = t.atoms.map(|i| set.masses.as_ref().map_or(1.0, |m| m[i]));
        let old = t.atoms.map(|i| atom(&self.at, i));
        let new = t.atoms.map(|i| atom(x, i));

        let normal = cross(sub(old[1], old[0]), sub(old[2], old[0]));
        let ab = sub(old[1], old[0]);
        let ez = scaled(normal, 1.0 / dot(normal, normal).sqrt());
        let ex = scaled(ab, 1.0 / dot(ab, ab).sqrt());
        let ey = cross(ez, ex);
        let (c0, c1) = (centre(&old, &m), centre(&new, &m));
        let plane = |p: V3, c: V3| {
            let r = sub(p, c);
            [dot(r, ex), dot(r, ey)]
        };

        // The triangle at its lengths, a at the origin, b on the first
        // coordinate line and c above it, then centred.
        let [dab, dac, dbc] = t.edges.map(|k| set.lengths[k]);
        let cx = (dab * dab + dac * dac - dbc * dbc) / (2.0 * dab);
        let cy = (dac * dac - cx * cx).sqrt();
        let shape = [[0.0, 0.0, 0.0], [dab, 0.0, 0.0], [cx, cy, 0.0]];
        let g = centre(&shape, &m);
        let local = shape.map(|p| [p[0] - g[0], p[1] - g[1]]);

        // Normal coordinates z_i = w . local_i; the system is consistent
        // because both sides have a zero mass-weighted sum, so solve it
        // on the two rows with the larger determinant.
        let z = new.map(|p| dot(sub(p, c1), ez));
        let det = |i: usize, j: usize| local[i][0] * local[j][1] - local[i][1] * local[j][0];
        let (i, j) = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .max_by(|a, b| det(a.0, a.1).abs().total_cmp(&det(b.0, b.1).abs()))
            .unwrap_or((0, 1));
        let dij = det(i, j);
        let wx = (z[i] * local[j][1] - z[j] * local[i][1]) / dij;
        let wy = (local[i][0] * z[j] - local[j][0] * z[i]) / dij;
        let tilt = 1.0 - wx * wx - wy * wy;
        if tilt.is_nan() || tilt <= 0.0 {
            return Err(degenerate().into());
        }
        // Images of the shape frame: a = (a_p, wx), b = (b_p, wy), unit and
        // orthogonal, with a_p x b_p > 0 so the triangle keeps its turn.
        let alpha = (1.0 - wx * wx).sqrt();
        let a_p = [alpha, 0.0];
        let b_p = [-wx * wy / alpha, tilt.sqrt() / alpha];
        let q = local.map(|l| [a_p[0] * l[0] + b_p[0] * l[1], a_p[1] * l[0] + b_p[1] * l[1]]);

        let p0 = old.map(|p| plane(p, c0));
        let p1 = new.map(|p| plane(p, c1));
        let (mut cc, mut dd, mut ee) = (0.0, 0.0, 0.0);
        for k in 0..3 {
            cc += m[k] * (p0[k][0] * q[k][1] - p0[k][1] * q[k][0]);
            dd += m[k] * (p0[k][0] * q[k][0] + p0[k][1] * q[k][1]);
            ee += m[k] * (p0[k][0] * p1[k][1] - p0[k][1] * p1[k][0]);
        }
        let r0 = cc.hypot(dd);
        let slack = r0 * r0 - ee * ee;
        if slack.is_nan() || slack < 0.0 {
            return Err(degenerate().into());
        }
        let phase = dd.atan2(cc);
        let half = slack.sqrt().atan2(ee);
        let rotate = |th: f64, v: [f64; 2]| {
            let (s, c) = th.sin_cos();
            [c * v[0] - s * v[1], s * v[0] + c * v[1]]
        };
        let overlap = |th: f64| {
            (0..3)
                .map(|k| {
                    let p = rotate(th, q[k]);
                    m[k] * (p[0] * p1[k][0] + p[1] * p1[k][1])
                })
                .sum::<f64>()
        };
        let (plus, minus) = (phase + half, phase - half);
        let theta = if overlap(plus) >= overlap(minus) {
            plus
        } else {
            minus
        };
        let placed: [V3; 3] = std::array::from_fn(|k| {
            let p = rotate(theta, q[k]);
            std::array::from_fn(|c| c1[c] + ex[c] * p[0] + ey[c] * p[1] + ez[c] * z[k])
        });
        if !placed.iter().flatten().all(|v| v.is_finite()) {
            return Err(degenerate().into());
        }
        for (p, &a) in placed.iter().zip(&t.atoms) {
            (0..3).for_each(|c| x[3 * a + c] = p[c]);
        }
        Ok(())
    }
}
