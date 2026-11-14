fn jacobi6(mut a: [[f64; 6]; 6], n: usize) -> ([f64; 6], [[f64; 6]; 6]) {
    let mut v = [[0.0; 6]; 6];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _ in 0..48 {
        let mut p = 0;
        let mut q = 1;
        let mut big = 0.0;
        if n < 2 {
            break;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let aij = a[i][j].abs();
                if aij > big {
                    big = aij;
                    p = i;
                    q = j;
                }
            }
        }
        if big < 1.0e-15 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            let aip = a[i][p];
            let aiq = a[i][q];
            a[i][p] = c * aip - s * aiq;
            a[p][i] = a[i][p];
            a[i][q] = s * aip + c * aiq;
            a[q][i] = a[i][q];
        }
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }
    let mut eval = [0.0; 6];
    for i in 0..n {
        eval[i] = a[i][i];
    }
    (eval, v)
}
