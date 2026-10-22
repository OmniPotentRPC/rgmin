#[cfg(test)]
fn l2(v: &Array1<f64>) -> f64 {
    crate::vecops::nrm2(v.view())
}

#[cfg(test)]
fn nrm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

#[cfg(test)]
fn lj_grad(pos: &[f64]) -> Vec<f64> {
    let nat = pos.len() / 3;
    let mut g = vec![0.0; pos.len()];
    let cut2 = 40.0_f64 * 40.0;
    for i in 0..nat {
        for j in (i + 1)..nat {
            let mut r2 = 0.0;
            let mut d = [0.0; 3];
            for a in 0..3 {
                d[a] = pos[3 * j + a] - pos[3 * i + a];
                r2 += d[a] * d[a];
            }
            if r2 >= cut2 || r2 == 0.0 {
                continue;
            }
            let inv = 1.0 / r2;
            let sr6 = inv * inv * inv;
            let sr12 = sr6 * sr6;
            let slope = 24.0 * (sr6 - 2.0 * sr12) * inv;
            for a in 0..3 {
                let f = slope * d[a];
                g[3 * i + a] -= f;
                g[3 * j + a] += f;
            }
        }
    }
    g
}

