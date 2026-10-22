impl RigidWorkspace {
    /// Empty buffers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Modes kept on the last [`Self::project`] or [`Self::project_hessian`].
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Orthonormal basis, `rank` vectors of length `n`, in the active metric.
    pub fn basis(&self) -> &[f64] {
        let n = self.rank.saturating_mul(self.n);
        &self.basis[..n.min(self.basis.len())]
    }

    /// Copy `pos` so a later call can project without the geometry argument.
    pub fn set_anchor(&mut self, pos: &[f64]) {
        if self.anchor.len() < pos.len() {
            self.anchor.resize(pos.len(), 0.0);
        }
        self.anchor[..pos.len()].copy_from_slice(pos);
        self.anchor_n = pos.len();
    }

    /// `P v` at `req.pos`. Rebuilds the basis.
    pub fn project(&mut self, v: &mut [f64], req: &RigidRequest<'_>) {
        if v.len() != req.pos.len() || req.pos.len() % 3 != 0 {
            self.rank = 0;
            return;
        }
        self.rebuild(req);
        let n = self.n;
        for k in 0..self.rank {
            let e = &self.basis[k * n..(k + 1) * n];
            let d = metric_dot(v, e, req);
            for i in 0..n {
                v[i] -= d * e[i];
            }
        }
    }

    /// Project `v` at the stored anchor.
    #[allow(clippy::too_many_arguments)]
    pub fn project_at_anchor(
        &mut self,
        v: &mut [f64],
        masses: Option<&[f64]>,
        frozen: Option<&[u8]>,
        mass_weighted_coords: bool,
        mass_metric: bool,
        mode: RigidMode,
        pbc: [bool; 3],
        broken_translation: [bool; 3],
        broken_rotation: [bool; 3],
    ) {
        let n = self.anchor_n;
        let anchor = std::mem::take(&mut self.anchor);
        if n > 0 && anchor.len() >= n && v.len() == n {
            let req = RigidRequest {
                pos: &anchor[..n],
                masses,
                mass_weighted_coords,
                mass_metric,
                mode,
                pbc,
                frozen,
                broken_translation,
                broken_rotation,
            };
            self.project(v, &req);
        }
        self.anchor = anchor;
    }

    /// `P H P` at the stored anchor. Row-major `n x n`.
    #[allow(clippy::too_many_arguments)]
    pub fn project_hessian_at_anchor(
        &mut self,
        h: &mut [f64],
        masses: Option<&[f64]>,
        frozen: Option<&[u8]>,
        mass_weighted_coords: bool,
        mass_metric: bool,
        mode: RigidMode,
        pbc: [bool; 3],
        broken_translation: [bool; 3],
        broken_rotation: [bool; 3],
    ) {
        let n = self.anchor_n;
        let anchor = std::mem::take(&mut self.anchor);
        if n > 0 && anchor.len() >= n && h.len() == n * n {
            let req = RigidRequest {
                pos: &anchor[..n],
                masses,
                mass_weighted_coords,
                mass_metric,
                mode,
                pbc,
                frozen,
                broken_translation,
                broken_rotation,
            };
            self.project_hessian(h, &req);
        }
        self.anchor = anchor;
    }

    /// Replace row-major `H` with `P H P` in the active metric.
    pub fn project_hessian(&mut self, h: &mut [f64], req: &RigidRequest<'_>) {
        let n = req.pos.len();
        if n == 0 || h.len() != n * n {
            return;
        }
        self.rebuild(req);
        let rank = self.rank;
        if rank == 0 {
            return;
        }
        if self.hess.len() < n * n {
            self.hess.resize(n * n, 0.0);
        }
        if self.he.len() < n * 6 {
            self.he.resize(n * 6, 0.0);
        }
        if self.sh.len() < 6 * n {
            self.sh.resize(6 * n, 0.0);
        }
        self.hess[..n * n].copy_from_slice(h);
        for k in 0..rank {
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += self.hess[i * n + j] * self.basis[k * n + j];
                }
                self.he[i * 6 + k] = acc;
            }
            for j in 0..n {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += weight(req, i) * self.basis[k * n + i] * self.hess[i * n + j];
                }
                self.sh[k * n + j] = acc;
            }
        }
        let mut she = [0.0; 36];
        for k in 0..rank {
            for q in 0..rank {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += self.sh[k * n + j] * self.basis[q * n + j];
                }
                she[k * 6 + q] = acc;
            }
        }
        for i in 0..n {
            for j in 0..n {
                let mut acc = self.hess[i * n + j];
                let wj = weight(req, j);
                for k in 0..rank {
                    acc -= self.he[i * 6 + k] * (wj * self.basis[k * n + j]);
                    acc -= self.basis[k * n + i] * self.sh[k * n + j];
                }
                for k in 0..rank {
                    for q in 0..rank {
                        acc +=
                            self.basis[k * n + i] * she[k * 6 + q] * (wj * self.basis[q * n + j]);
                    }
                }
                h[i * n + j] = acc;
            }
        }
    }
}
