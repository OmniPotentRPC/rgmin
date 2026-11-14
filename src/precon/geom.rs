//! Shared geometry for the bonded metrics.
//!
//! Distances use an orthorhombic minimum image when cell lengths are
//! supplied. Angle and torsion gradients are the Wilson derivatives
//! that make the reduced Hessian `k (∂q/∂x)(∂q/∂x)^T`.

use crate::error::{Error, Result};

pub(super) fn require_atoms(positions_len: usize) -> Result<usize> {
    if !positions_len.is_multiple_of(3) || positions_len == 0 {
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

/// Pairs `i < j` whose distance is strictly below `cutoff`.
///
/// Atoms are dropped into bins of width `cutoff` and the occupied bins
/// are visited in lexicographic order. The result is sorted by `(i, j)`,
/// so a parallel bin walk and the serial walk build the same list.
pub(super) fn pairs_in_range(
    pos: &[[f64; 3]],
    cutoff: f64,
    lengths: Option<[f64; 3]>,
) -> Vec<(usize, usize, f64)> {
    let n = pos.len();
    if n < 2 || !(cutoff.is_finite() && cutoff > 0.0) {
        return Vec::new();
    }
    let periodic = match lengths {
        Some(len) => [
            len[0].is_finite() && len[0] > 0.0,
            len[1].is_finite() && len[1] > 0.0,
            len[2].is_finite() && len[2] > 0.0,
        ],
        None => [false, false, false],
    };
    let len = lengths.unwrap_or([0.0, 0.0, 0.0]);
    let mut origin = [f64::INFINITY; 3];
    let mut span = [0.0_f64; 3];
    for p in pos {
        for k in 0..3 {
            if !periodic[k] {
                origin[k] = origin[k].min(p[k]);
            }
        }
    }
    for k in 0..3 {
        if periodic[k] {
            origin[k] = 0.0;
            span[k] = len[k];
        } else {
            let mut hi = f64::NEG_INFINITY;
            for p in pos {
                hi = hi.max(p[k]);
            }
            span[k] = (hi - origin[k]).max(cutoff);
        }
    }
    let mut nbin = [1usize; 3];
    for k in 0..3 {
        let bins = (span[k] / cutoff).floor().max(1.0);
        nbin[k] = bins as usize;
    }
    let mut cells: std::collections::BTreeMap<(i32, i32, i32), Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        let key = bin_of(pos[i], origin, cutoff, nbin, periodic, len);
        cells.entry(key).or_default().push(i);
    }
    let keys: Vec<(i32, i32, i32)> = cells.keys().copied().collect();
    let parts = super::ordered_map(keys.len().max(1), n, super::N_PAR_ASSEMBLE, |c| {
        if c >= keys.len() {
            return Vec::new();
        }
        let key = keys[c];
        let Some(atoms) = cells.get(&key) else {
            return Vec::new();
        };
        let mut local = Vec::new();
        let mut seen = Vec::with_capacity(27);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let Some(nb) = neighbor_bin(key, [dx, dy, dz], nbin, periodic) else {
                        continue;
                    };
                    if seen.contains(&nb) {
                        continue;
                    }
                    seen.push(nb);
                    let Some(others) = cells.get(&nb) else {
                        continue;
                    };
                    for &i in atoms {
                        for &j in others {
                            if j <= i {
                                continue;
                            }
                            let d = mic(pos[i], pos[j], lengths);
                            let r = norm3(d);
                            if r.is_finite() && r > 1e-8 && r < cutoff {
                                local.push((i, j, r));
                            }
                        }
                    }
                }
            }
        }
        local
    });
    let mut pairs = Vec::new();
    for part in parts {
        pairs.extend(part);
    }
    pairs.sort_by_key(|a| (a.0, a.1));
    pairs
}

fn bin_of(
    p: [f64; 3],
    origin: [f64; 3],
    cutoff: f64,
    nbin: [usize; 3],
    periodic: [bool; 3],
    len: [f64; 3],
) -> (i32, i32, i32) {
    let mut key = [0i32; 3];
    for k in 0..3 {
        let n = nbin[k] as i32;
        let idx = if periodic[k] {
            let mut f = p[k] / len[k];
            f -= f.floor();
            if f >= 1.0 {
                f = 0.0;
            }
            (f * n as f64).floor() as i32
        } else {
            ((p[k] - origin[k]) / cutoff).floor() as i32
        };
        key[k] = idx.clamp(0, n - 1);
    }
    (key[0], key[1], key[2])
}

fn neighbor_bin(
    key: (i32, i32, i32),
    delta: [i32; 3],
    nbin: [usize; 3],
    periodic: [bool; 3],
) -> Option<(i32, i32, i32)> {
    let cur = [key.0, key.1, key.2];
    let mut out = [0i32; 3];
    for k in 0..3 {
        let n = nbin[k] as i32;
        let v = cur[k] + delta[k];
        if periodic[k] {
            let mut w = v % n;
            if w < 0 {
                w += n;
            }
            out[k] = w;
        } else if v < 0 || v >= n {
            return None;
        } else {
            out[k] = v;
        }
    }
    Some((out[0], out[1], out[2]))
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

#[cfg(test)]
mod tests {
    use super::pairs_in_range;

    fn brute(pos: &[[f64; 3]], cutoff: f64, lengths: Option<[f64; 3]>) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::new();
        for i in 0..pos.len() {
            for j in (i + 1)..pos.len() {
                let d = super::mic(pos[i], pos[j], lengths);
                let r = super::norm3(d);
                if r.is_finite() && r > 1e-8 && r < cutoff {
                    out.push((i, j, r));
                }
            }
        }
        out
    }

    #[test]
    fn the_cell_list_matches_all_pairs() {
        let mut pos = Vec::new();
        for i in 0..40 {
            pos.push([
                (i % 5) as f64 * 1.2,
                (i / 5) as f64 * 1.35,
                ((i * 3) % 7) as f64 * 0.4,
            ]);
        }
        let cutoff = 2.5;
        let got = pairs_in_range(&pos, cutoff, None);
        let expect = brute(&pos, cutoff, None);
        assert_eq!(got.len(), expect.len());
        for (a, b) in got.iter().zip(&expect) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!((a.2 - b.2).abs() < 1e-12);
        }
        let lengths = Some([6.0, 5.5, 4.0]);
        let got = pairs_in_range(&pos, cutoff, lengths);
        let expect = brute(&pos, cutoff, lengths);
        assert_eq!(got.len(), expect.len());
        for (a, b) in got.iter().zip(&expect) {
            assert_eq!((a.0, a.1), (b.0, b.1));
            assert!((a.2 - b.2).abs() < 1e-12);
        }
    }
}
