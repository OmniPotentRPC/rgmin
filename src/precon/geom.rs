//! Shared geometry for the bonded metrics.
//!
//! Distances use an orthorhombic minimum image when cell lengths are
//! supplied. Angle and torsion gradients are the Wilson derivatives
//! that make the reduced Hessian `k (∂q/∂x)(∂q/∂x)^T`.

use crate::error::{Error, Result};

pub(super) fn require_atoms(positions_len: usize) -> Result<usize> {
    if positions_len % 3 != 0 || positions_len == 0 {
        return Err(Error::Precon {
            what: "positions length is not a positive multiple of 3",
        });
    }
    Ok(positions_len / 3)
}

pub(super) fn pos_of(positions: &[f64], atom: usize) -> [f64; 3] {
    [
        positions[3 * atom],
        positions[3 * atom + 1],
        positions[3 * atom + 2],
    ]
}

/// `a - b`, folded into the orthorhombic cell when a length is positive.
pub(super) fn mic(a: [f64; 3], b: [f64; 3], lengths: Option<[f64; 3]>) -> [f64; 3] {
    let mut d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    if let Some(len) = lengths {
        for k in 0..3 {
            if len[k].is_finite() && len[k] > 0.0 {
                d[k] -= len[k] * (d[k] / len[k]).round();
            }
        }
    }
    d
}

pub(super) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(super) fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

pub(super) fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub(super) fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `∂θ/∂x` at `(end, vertex, end)`. `None` when a leg vanishes or the
/// angle is within `1e-3` of 0 or π, where the reduced angle term is
/// dropped.
pub(super) fn angle_grad(
    end_i: [f64; 3],
    vertex: [f64; 3],
    end_k: [f64; 3],
    lengths: Option<[f64; 3]>,
) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    let u = mic(end_i, vertex, lengths);
    let v = mic(end_k, vertex, lengths);
    let du = norm3(u);
    let dv = norm3(v);
    if du < 1e-12 || dv < 1e-12 {
        return None;
    }
    let eu = scale3(u, 1.0 / du);
    let ev = scale3(v, 1.0 / dv);
    let c = dot3(eu, ev).clamp(-1.0, 1.0);
    let s = (1.0 - c * c).sqrt();
    if s <= 1e-3 {
        return None;
    }
    let qu_ev = [ev[0] - c * eu[0], ev[1] - c * eu[1], ev[2] - c * eu[2]];
    let qv_eu = [eu[0] - c * ev[0], eu[1] - c * ev[1], eu[2] - c * ev[2]];
    let gi = scale3(qu_ev, -1.0 / (du * s));
    let gk = scale3(qv_eu, -1.0 / (dv * s));
    let gj = scale3(add3(gi, gk), -1.0);
    Some((gi, gj, gk))
}

/// `∂φ/∂x` for the torsion `(i, j, k, l)`, ASE's branch of the
/// reduced dihedral. `None` when a cross product vanishes.
pub(super) fn dihedral_grad(
    pi: [f64; 3],
    pj: [f64; 3],
    pk: [f64; 3],
    pl: [f64; 3],
    lengths: Option<[f64; 3]>,
) -> Option<[[f64; 3]; 4]> {
    let rij = mic(pi, pj, lengths);
    let rkj = mic(pk, pj, lengths);
    let rkl = mic(pk, pl, lengths);
    let dkj = norm3(rkj);
    if dkj < 1e-12 {
        return None;
    }
    let dkj2 = dkj * dkj;
    let rmj = cross(rij, rkj);
    let rnk = cross(rkj, rkl);
    let dmj2 = dot3(rmj, rmj);
    let dnk2 = dot3(rnk, rnk);
    if dmj2 < 1e-16 || dnk2 < 1e-16 {
        return None;
    }
    let dddri = scale3(rmj, dkj / dmj2);
    let dddrl = scale3(rnk, -dkj / dnk2);
    let rijrkj = dot3(rij, rkj);
    let rkjrkl = dot3(rkj, rkl);
    let gj = add3(
        scale3(dddri, rijrkj / dkj2 - 1.0),
        scale3(dddrl, -rkjrkl / dkj2),
    );
    let gk = add3(
        scale3(dddrl, rkjrkl / dkj2 - 1.0),
        scale3(dddri, -rijrkj / dkj2),
    );
    Some([dddri, gj, gk, dddrl])
}

/// Add `scale * g g^T` on the free atoms. A `None` index is a ghost
/// and contributes no row.
pub(super) fn add_rank1(
    trips: &mut Vec<(usize, usize, f64)>,
    atoms: &[(Option<usize>, [f64; 3])],
    scale: f64,
) {
    if !(scale.is_finite() && scale != 0.0) {
        return;
    }
    for (ia, ga) in atoms {
        let Some(ia) = *ia else { continue };
        for (ib, gb) in atoms {
            let Some(ib) = *ib else { continue };
            for d in 0..3 {
                for e in 0..3 {
                    let v = scale * ga[d] * gb[e];
                    if v != 0.0 {
                        trips.push((3 * ia + d, 3 * ib + e, v));
                    }
                }
            }
        }
    }
}
