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
