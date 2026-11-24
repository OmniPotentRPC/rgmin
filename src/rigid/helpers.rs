fn axes_of(req: &RigidRequest<'_>) -> ([bool; 3], [bool; 3]) {
    if req.mode == RigidMode::Off {
        return ([false; 3], [false; 3]);
    }
    let pbc_any = req.pbc[0] || req.pbc[1] || req.pbc[2];
    let want_rot = match req.mode {
        RigidMode::TranslationRotation => true,
        RigidMode::Auto => !pbc_any,
        RigidMode::Translation | RigidMode::Off => false,
    };
    let mut trans = [false; 3];
    let mut rot = [false; 3];
    for d in 0..3 {
        let on_cell = if req.mode == RigidMode::TranslationRotation {
            true
        } else if pbc_any {
            req.pbc[d]
        } else {
            true
        };
        trans[d] = on_cell && !req.broken_translation[d];
        rot[d] = want_rot && !req.broken_rotation[d];
    }
    (trans, rot)
}

fn atom_mass(req: &RigidRequest<'_>, i: usize) -> f64 {
    req.masses
        .and_then(|m| m.get(i).copied())
        .unwrap_or(1.0)
        .max(0.0)
}

fn com_mass(req: &RigidRequest<'_>, i: usize) -> f64 {
    if req.mass_metric || req.mass_weighted_coords {
        atom_mass(req, i)
    } else {
        1.0
    }
}

fn cart(req: &RigidRequest<'_>, i: usize, d: usize) -> f64 {
    let q = req.pos[3 * i + d];
    if req.mass_weighted_coords {
        let sm = atom_mass(req, i).sqrt();
        if sm > 0.0 { q / sm } else { 0.0 }
    } else {
        q
    }
}

fn trans_amp(req: &RigidRequest<'_>, i: usize) -> f64 {
    if req.mass_weighted_coords {
        atom_mass(req, i).sqrt()
    } else {
        1.0
    }
}

fn weight(req: &RigidRequest<'_>, coord: usize) -> f64 {
    if req.mass_weighted_coords || !req.mass_metric {
        1.0
    } else {
        atom_mass(req, coord / 3)
    }
}

fn metric_dot(a: &[f64], b: &[f64], req: &RigidRequest<'_>) -> f64 {
    if req.mass_weighted_coords || !req.mass_metric {
        return a.iter().zip(b).map(|(u, v)| u * v).sum();
    }
    let mut s = 0.0;
    let nat = a.len() / 3;
    for i in 0..nat {
        let mi = atom_mass(req, i);
        let bse = 3 * i;
        s += mi * (a[bse] * b[bse] + a[bse + 1] * b[bse + 1] + a[bse + 2] * b[bse + 2]);
    }
    s
}

fn scale_cols(a: &mut [f64], n: usize, m: usize, req: &RigidRequest<'_>, up: bool) {
    let nat = n / 3;
    for j in 0..m {
        for i in 0..nat {
            let s = atom_mass(req, i).sqrt();
            let f = if s == 0.0 {
                0.0
            } else if up {
                s
            } else {
                1.0 / s
            };
            for d in 0..3 {
                a[j * n + 3 * i + d] *= f;
            }
        }
    }
}

/// Fold `x` into a scaled sum of squares, `scale * sqrt(ssq)` being the
/// running 2-norm. Squares never overflow or underflow, so the norm of a
/// cluster stays finite and exact to rounding at any coordinate scale.
fn accumulate_norm(ssq: &mut f64, scale: &mut f64, x: f64) {
    let a = x.abs();
    if a == 0.0 {
        return;
    }
    if a > *scale {
        let r = *scale / a;
        *ssq = 1.0 + *ssq * r * r;
        *scale = a;
    } else {
        let r = a / *scale;
        *ssq += r * r;
    }
}

fn scaled_norm(v: &[f64]) -> f64 {
    let mut ssq = 0.0;
    let mut scale = 0.0;
    for x in v {
        accumulate_norm(&mut ssq, &mut scale, *x);
    }
    scale * ssq.sqrt()
}

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let adjusted = value - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

/// Weighted Cartesian centre with both coordinates and weights scaled before
/// accumulation. Finite coordinates near `f64::MAX` therefore never overflow
/// merely because more than one atom contributes to the sum.
fn scaled_center(req: &RigidRequest<'_>, nat: usize) -> [f64; 3] {
    let weight_scale = (0..nat)
        .map(|i| com_mass(req, i))
        .fold(0.0_f64, f64::max);
    if !(weight_scale > 0.0 && weight_scale.is_finite()) {
        return [0.0; 3];
    }

    let mut coordinate_scale = [0.0_f64; 3];
    for i in 0..nat {
        for (d, scale) in coordinate_scale.iter_mut().enumerate() {
            *scale = scale.max(cart(req, i, d).abs());
        }
    }

    let mut total = 0.0;
    let mut total_correction = 0.0;
    let mut scaled_sum = [0.0; 3];
    let mut sum_correction = [0.0; 3];
    for i in 0..nat {
        let weight = com_mass(req, i) / weight_scale;
        compensated_add(&mut total, &mut total_correction, weight);
        for d in 0..3 {
            if coordinate_scale[d] > 0.0 {
                compensated_add(
                    &mut scaled_sum[d],
                    &mut sum_correction[d],
                    weight * (cart(req, i, d) / coordinate_scale[d]),
                );
            }
        }
    }

    let mut center = [0.0; 3];
    if total > 0.0 {
        for d in 0..3 {
            let average = (scaled_sum[d] / total).clamp(-1.0, 1.0);
            center[d] = coordinate_scale[d] * average;
        }
    }
    center
}

/// Divide a rigid-motion candidate by `reference`, the largest norm the
/// motion could have for this geometry: the norm of a translation, or the
/// norm of the lever arms for a rotation. The column then measures how much
/// of the motion is realized, a number in `[0, 1]` that does not change
/// when the coordinates are scaled. A column that is zero or whose
/// reference is not a positive finite number is left alone.
fn normalize_candidate(col: &mut [f64], reference: f64) {
    if !(reference > 0.0 && reference.is_finite()) {
        return;
    }
    for x in col.iter_mut() {
        *x /= reference;
    }
}

/// Eigenvalue of the frozen-row Gram matrix, relative to the largest, below
/// which a combination of candidates counts as vanishing on the frozen
/// coordinates. The eigenvalue is a squared norm and the Jacobi sweep leaves
/// rounding noise near `1e-17` of the peak, so the bound sits above that
/// noise: `1e-14` is a relative norm of `1e-7`.
const NULLSPACE_TOL: f64 = 1.0e-14;
