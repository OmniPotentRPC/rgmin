/// Column square-sums from row `k`, four columns at a time.
///
/// Each column is the left fold `s += t * t` in row order. The other
/// columns in the group do not feed that fold.
fn pair_sumsq(a: &[f64], n: usize, k: usize, m: usize, norm: &mut [f64; 6]) {
    let mut j = k;
    while j + 3 < m {
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let mut s3 = 0.0;
        let c0 = j * n;
        let c1 = (j + 1) * n;
        let c2 = (j + 2) * n;
        let c3 = (j + 3) * n;
        for i in k..n {
            let t0 = a[c0 + i];
            let t1 = a[c1 + i];
            let t2 = a[c2 + i];
            let t3 = a[c3 + i];
            s0 += t0 * t0;
            s1 += t1 * t1;
            s2 += t2 * t2;
            s3 += t3 * t3;
        }
        norm[j] = s0;
        norm[j + 1] = s1;
        norm[j + 2] = s2;
        norm[j + 3] = s3;
        j += 4;
    }
    while j + 1 < m {
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        let c0 = j * n;
        let c1 = (j + 1) * n;
        for i in k..n {
            let t0 = a[c0 + i];
            let t1 = a[c1 + i];
            s0 += t0 * t0;
            s1 += t1 * t1;
        }
        norm[j] = s0;
        norm[j + 1] = s1;
        j += 2;
    }
    if j < m {
        let mut s = 0.0;
        let c = j * n;
        for i in k..n {
            let t = a[c + i];
            s += t * t;
        }
        norm[j] = s;
    }
}

/// `dot[j] = sum_i v[i] * col_j[i]` from row `row0`, four columns at a time.
fn pair_dots(
    vcol: &[f64],
    a: &[f64],
    n: usize,
    row0: usize,
    j0: usize,
    j1: usize,
    dots: &mut [f64; 6],
) {
    let mut j = j0;
    while j + 3 < j1 {
        let mut d0 = 0.0;
        let mut d1 = 0.0;
        let mut d2 = 0.0;
        let mut d3 = 0.0;
        let c0 = j * n;
        let c1 = (j + 1) * n;
        let c2 = (j + 2) * n;
        let c3 = (j + 3) * n;
        for i in row0..n {
            let vi = vcol[i];
            d0 += vi * a[c0 + i];
            d1 += vi * a[c1 + i];
            d2 += vi * a[c2 + i];
            d3 += vi * a[c3 + i];
        }
        dots[j] = d0;
        dots[j + 1] = d1;
        dots[j + 2] = d2;
        dots[j + 3] = d3;
        j += 4;
    }
    while j + 1 < j1 {
        let mut d0 = 0.0;
        let mut d1 = 0.0;
        let c0 = j * n;
        let c1 = (j + 1) * n;
        for i in row0..n {
            let vi = vcol[i];
            d0 += vi * a[c0 + i];
            d1 += vi * a[c1 + i];
        }
        dots[j] = d0;
        dots[j + 1] = d1;
        j += 2;
    }
    if j < j1 {
        let mut d = 0.0;
        let c = j * n;
        for i in row0..n {
            d += vcol[i] * a[c + i];
        }
        dots[j] = d;
    }
}

/// `col_j[i] -= two[j] * v[i]` from row `row0`, four columns at a time.
fn pair_update(
    vcol: &[f64],
    a: &mut [f64],
    n: usize,
    row0: usize,
    j0: usize,
    j1: usize,
    two: &[f64; 6],
) {
    let mut j = j0;
    while j + 3 < j1 {
        let t0 = two[j];
        let t1 = two[j + 1];
        let t2 = two[j + 2];
        let t3 = two[j + 3];
        let c0 = j * n;
        let c1 = (j + 1) * n;
        let c2 = (j + 2) * n;
        let c3 = (j + 3) * n;
        for i in row0..n {
            let vi = vcol[i];
            a[c0 + i] -= t0 * vi;
            a[c1 + i] -= t1 * vi;
            a[c2 + i] -= t2 * vi;
            a[c3 + i] -= t3 * vi;
        }
        j += 4;
    }
    while j + 1 < j1 {
        let t0 = two[j];
        let t1 = two[j + 1];
        let c0 = j * n;
        let c1 = (j + 1) * n;
        for i in row0..n {
            let vi = vcol[i];
            a[c0 + i] -= t0 * vi;
            a[c1 + i] -= t1 * vi;
        }
        j += 2;
    }
    if j < j1 {
        let t = two[j];
        let c = j * n;
        for i in row0..n {
            a[c + i] -= t * vcol[i];
        }
    }
}

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
        let mut norm = [0.0; 6];
        pair_sumsq(a, n, k, m, &mut norm);
        let mut piv = k;
        let mut best = -1.0;
        for j in k..m {
            if norm[j] > best {
                best = norm[j];
                piv = j;
            }
        }
        if piv != k {
            for i in 0..n {
                a.swap(k * n + i, piv * n + i);
            }
        }
        // The pivot column moved as a block. Its trailing square-sum is
        // the fold already stored for that column.
        let nrm = norm[piv].sqrt();
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
        let vcol = &vstore[k * n..k * n + n];
        let mut dots = [0.0; 6];
        pair_dots(vcol, a, n, k, k, m, &mut dots);
        let mut two = [0.0; 6];
        for j in k..m {
            two[j] = 2.0 * dots[j];
        }
        pair_update(vcol, a, n, k, k, m, &two);
        rank += 1;
    }
    for j in 0..rank {
        for i in 0..n {
            q[j * n + i] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for k in (0..rank).rev() {
        let vcol = &vstore[k * n..k * n + n];
        let mut dots = [0.0; 6];
        pair_dots(vcol, q, n, k, 0, rank, &mut dots);
        let mut two = [0.0; 6];
        for j in 0..rank {
            two[j] = 2.0 * dots[j];
        }
        pair_update(vcol, q, n, k, 0, rank, &two);
    }
    rank
}

#[cfg(test)]
fn cpqr_serial(a: &mut [f64], vstore: &mut [f64], q: &mut [f64], n: usize, m: usize) -> usize {
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

#[cfg(test)]
mod cpqr_bits {
    use super::{cpqr, cpqr_serial};

    #[test]
    fn swept_columns_match_one_column_at_a_time() {
        for &(n, m, seed) in &[
            (9usize, 6usize, 3u64),
            (24, 6, 11),
            (72, 5, 9),
            (8, 3, 1),
            (6, 6, 4),
            (15, 6, 8),
        ] {
            let mut a = vec![0.0; n * m];
            let mut s = seed | 1;
            for x in &mut a {
                s = s.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
                *x = ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.4;
            }
            let mut a2 = a.clone();
            let mut v1 = vec![0.0; n * m];
            let mut v2 = vec![0.0; n * m];
            let mut q1 = vec![0.0; n * m];
            let mut q2 = vec![0.0; n * m];
            let r1 = cpqr(&mut a, &mut v1, &mut q1, n, m);
            let r2 = cpqr_serial(&mut a2, &mut v2, &mut q2, n, m);
            assert_eq!(r1, r2, "rank n={n} m={m}");
            for i in 0..r1 * n {
                assert_eq!(q1[i].to_bits(), q2[i].to_bits(), "q {i} n={n}");
            }
        }
    }
}
