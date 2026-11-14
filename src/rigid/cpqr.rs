/// Column-pivoted Householder QR. Writes an orthonormal basis of the
/// column space into `q` and returns the numerical rank.
///
/// Rank is the number of R diagonals above `1e-10` times the leading
/// pivot, the test Sella uses. An absolute cutoff is not used.
fn cpqr(a: &mut [f64], vstore: &mut [f64], q: &mut [f64], n: usize, m: usize) -> usize {
    const TOL: f64 = 1e-10;
    if n == 0 || m == 0 {
        return 0;
    }
    let m = m.min(6);
    for x in vstore.iter_mut().take(n * m) {
        *x = 0.0;
    }
    let mut r00 = 0.0;
    let mut rank = 0usize;
    for k in 0..m {
        if k >= n {
            break;
        }
        let mut piv = k;
        let mut best = -1.0;
        for j in k..m {
            let mut s = 0.0;
            for i in k..n {
                let t = a[j * n + i];
                s += t * t;
            }
            if s > best {
                best = s;
                piv = j;
            }
        }
        if piv != k {
            for i in 0..n {
                a.swap(k * n + i, piv * n + i);
            }
        }
        let mut nrm2 = 0.0;
        for i in k..n {
            let t = a[k * n + i];
            nrm2 += t * t;
        }
        let nrm = nrm2.sqrt();
        if k == 0 {
            r00 = nrm;
        }
        if r00 == 0.0 || nrm <= TOL * r00 {
            break;
        }
        let x0 = a[k * n + k];
        let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
        let v0 = x0 + sign * nrm;
        vstore[k * n + k] = v0;
        let mut vnorm2 = v0 * v0;
        for i in (k + 1)..n {
            let xi = a[k * n + i];
            vstore[k * n + i] = xi;
            vnorm2 += xi * xi;
        }
        let inv = 1.0 / vnorm2.sqrt();
        for i in k..n {
            vstore[k * n + i] *= inv;
        }
        for j in k..m {
            let mut dot = 0.0;
            for i in k..n {
                dot += vstore[k * n + i] * a[j * n + i];
            }
            let two = 2.0 * dot;
            for i in k..n {
                a[j * n + i] -= two * vstore[k * n + i];
            }
        }
        rank += 1;
    }
    for j in 0..rank {
        for i in 0..n {
            q[j * n + i] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for k in (0..rank).rev() {
        for j in 0..rank {
            let mut dot = 0.0;
            for i in k..n {
                dot += vstore[k * n + i] * q[j * n + i];
            }
            let two = 2.0 * dot;
            for i in k..n {
                q[j * n + i] -= two * vstore[k * n + i];
            }
        }
    }
    rank
}
