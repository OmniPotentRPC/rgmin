impl RigidWorkspace {
    fn ensure(&mut self, n: usize) {
        let need = n.saturating_mul(6);
        if self.basis.len() < need {
            self.basis.resize(need, 0.0);
        }
        if self.cand.len() < need {
            self.cand.resize(need, 0.0);
        }
        if self.reflect.len() < need {
            self.reflect.resize(need, 0.0);
        }
        if self.qwork.len() < need {
            self.qwork.resize(need, 0.0);
        }
        self.n = n;
    }

    fn rebuild(&mut self, req: &RigidRequest<'_>) {
        let n = req.pos.len();
        self.rank = 0;
        if req.mode == RigidMode::Off || n < 3 || !n.is_multiple_of(3) {
            self.n = n;
            return;
        }
        self.ensure(n);
        let nat = n / 3;
        let masses_ok = req.masses.is_some_and(|m| m.len() == nat);
        let req = if masses_ok {
            *req
        } else {
            RigidRequest {
                masses: None,
                ..*req
            }
        };
        let weighted = req.mass_weighted_coords;
        let metric = req.mass_metric && !weighted;
        let req = RigidRequest {
            mass_metric: metric,
            ..req
        };
        let (trans, rot) = axes_of(&req);
        let mut com = [0.0; 3];
        let mut mtot = 0.0;
        for i in 0..nat {
            let mi = com_mass(&req, i);
            mtot += mi;
            com[0] += mi * cart(&req, i, 0);
            com[1] += mi * cart(&req, i, 1);
            com[2] += mi * cart(&req, i, 2);
        }
        if mtot > 0.0 {
            com[0] /= mtot;
            com[1] /= mtot;
            com[2] /= mtot;
        }
        let mut m = 0usize;
        for d in 0..3 {
            if !trans[d] {
                continue;
            }
            for i in 0..nat {
                let s = trans_amp(&req, i);
                self.cand[m * n + 3 * i] = if d == 0 { s } else { 0.0 };
                self.cand[m * n + 3 * i + 1] = if d == 1 { s } else { 0.0 };
                self.cand[m * n + 3 * i + 2] = if d == 2 { s } else { 0.0 };
            }
            m += 1;
        }
        for d in 0..3 {
            if !rot[d] {
                continue;
            }
            let w = [d == 0, d == 1, d == 2];
            let omega = [
                if w[0] { 1.0 } else { 0.0 },
                if w[1] { 1.0 } else { 0.0 },
                if w[2] { 1.0 } else { 0.0 },
            ];
            for i in 0..nat {
                let rx = cart(&req, i, 0) - com[0];
                let ry = cart(&req, i, 1) - com[1];
                let rz = cart(&req, i, 2) - com[2];
                let cx = ry * omega[2] - rz * omega[1];
                let cy = rz * omega[0] - rx * omega[2];
                let cz = rx * omega[1] - ry * omega[0];
                let s = if weighted {
                    atom_mass(&req, i).sqrt()
                } else {
                    1.0
                };
                self.cand[m * n + 3 * i] = s * cx;
                self.cand[m * n + 3 * i + 1] = s * cy;
                self.cand[m * n + 3 * i + 2] = s * cz;
            }
            m += 1;
        }
        if m == 0 {
            return;
        }
        m = self.freeze_nullspace(&req, m);
        if m == 0 {
            return;
        }
        if metric {
            scale_cols(&mut self.cand, n, m, &req, true);
        }
        let rank = cpqr(&mut self.cand, &mut self.reflect, &mut self.qwork, n, m);
        self.basis[..rank * n].copy_from_slice(&self.qwork[..rank * n]);
        if metric {
            scale_cols(&mut self.basis, n, rank, &req, false);
        }
        if let Some(frozen) = req.frozen
            && frozen.len() == n
        {
            for k in 0..rank {
                for i in 0..n {
                    if frozen[i] != 0 {
                        self.basis[k * n + i] = 0.0;
                    }
                }
            }
        }
        self.rank = rank;
    }

    /// Combinations of the `m` columns that already vanish on frozen
    /// coordinates. Zeroing the columns and then taking their rank would
    /// also remove motions that are not symmetries. One pinned atom then
    /// keeps the three rotations about that atom.
    fn freeze_nullspace(&mut self, req: &RigidRequest<'_>, m: usize) -> usize {
        let n = self.n;
        let Some(frozen) = req.frozen else {
            return m;
        };
        if frozen.len() != n || !frozen.iter().any(|b| *b != 0) {
            return m;
        }
        let mut g = [[0.0; 6]; 6];
        for i in 0..n {
            if frozen[i] == 0 {
                continue;
            }
            for a in 0..m {
                let va = self.cand[a * n + i];
                for b in 0..m {
                    g[a][b] += va * self.cand[b * n + i];
                }
            }
        }
        let (eval, evec) = jacobi6(g, m);
        let mut peak = 0.0_f64;
        for k in 0..m {
            peak = peak.max(eval[k].abs());
        }
        let mut keep = [0usize; 6];
        let mut nk = 0usize;
        for k in 0..m {
            let small = peak == 0.0 || eval[k] <= 1.0e-20 * peak;
            if small {
                keep[nk] = k;
                nk += 1;
            }
        }
        for s in 0..nk {
            let col = keep[s];
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..m {
                    acc += self.cand[j * n + i] * evec[j][col];
                }
                if frozen[i] != 0 {
                    acc = 0.0;
                }
                self.qwork[s * n + i] = acc;
            }
        }
        self.cand[..nk * n].copy_from_slice(&self.qwork[..nk * n]);
        nk
    }
}
