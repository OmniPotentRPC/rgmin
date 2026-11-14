//! Drivers that keep the gradient in a caller-owned buffer.
//!
//! [`Solver::run_fill`] is the minimization used when the potential can
//! write `g` into storage that outlives one evaluation. The eOn cap and
//! the default Euclidean FIRE step do not allocate inside the loop.

use ndarray::{Array1, ArrayView1};

use super::{Inner, Solver};
use crate::error::{Error, Result};
use crate::fire::{
    FireKind, FirePreparation, fire_after_v1, fire_displacement_into, fire_prepare_v2,
};
use crate::lbfgs::{EON_FINITE_DIFFERENCE, EON_INVERSE_CURVATURE, EonSeed};
use crate::report::Report;

/// One eOn fill step, for the reset and cap study.
#[derive(Clone, Copy, Debug)]
pub struct EonFeat {
    /// `(g+ - g) · s / s·s`, the directional curvature.
    pub curvature: f64,
    /// `|F|` before the step.
    pub fnorm: f64,
    /// `|F|` after the step.
    pub fnorm_next: f64,
    /// Cosine of the angle between the force and the step.
    pub cos_angle: f64,
    /// Largest per-atom displacement divided by the cap.
    pub cap_ratio: f64,
    /// Mean per-atom displacement.
    pub atom_mean: f64,
    /// Largest per-atom displacement.
    pub atom_max: f64,
    /// Euclidean length of the step.
    pub step_len: f64,
    /// The step cleared the curvature pairs.
    pub reset: u8,
}
use crate::step::{finite, non_finite, scale_step, scale_step_atom};

fn feat_of(
    step: &Array1<f64>,
    gold: &Array1<f64>,
    gnew: &Array1<f64>,
    cap: f64,
    reset: bool,
) -> EonFeat {
    let n = step.len();
    let mut ss = 0.0;
    let mut sy = 0.0;
    let mut gs = 0.0;
    let mut gg = 0.0;
    let mut gn = 0.0;
    let mut atom_sum = 0.0;
    let mut atom_max = 0.0_f64;
    let mut atoms = 0.0;
    let mut i = 0;
    while i < n {
        let end = (i + 3).min(n);
        let mut a2 = 0.0;
        for k in i..end {
            let s = step[k];
            let y = gnew[k] - gold[k];
            let f = -gold[k];
            ss += s * s;
            sy += s * y;
            gs += f * s;
            gg += f * f;
            gn += gnew[k] * gnew[k];
            a2 += s * s;
        }
        let a = a2.sqrt();
        atom_sum += a;
        atom_max = atom_max.max(a);
        atoms += 1.0;
        i = end;
    }
    let fnorm = gg.sqrt();
    let snorm = ss.sqrt();
    EonFeat {
        curvature: if ss > 0.0 { sy / ss } else { 0.0 },
        fnorm,
        fnorm_next: gn.sqrt(),
        cos_angle: if fnorm > 0.0 && snorm > 0.0 {
            gs / (fnorm * snorm)
        } else {
            0.0
        },
        cap_ratio: if cap > 0.0 { atom_max / cap } else { 0.0 },
        atom_mean: if atoms > 0.0 { atom_sum / atoms } else { 0.0 },
        atom_max,
        step_len: snorm,
        reset: u8::from(reset),
    }
}

struct FillEnd {
    value: f64,
    rejected: bool,
}

impl Solver {
    /// Start recording eOn fill steps. The next [`Self::run_fill`] appends.
    pub fn start_eon_log(&mut self) {
        self.eon_log = Some(Vec::new());
    }

    /// Take the recorded steps, if logging was started.
    pub fn take_eon_log(&mut self) -> Vec<EonFeat> {
        self.eon_log.take().unwrap_or_default()
    }

    /// Minimize, with `fill` writing the gradient into the buffer it is given.
    ///
    /// The eOn cap and default FIRE reuse that buffer. Other methods copy
    /// it out and call [`Solver::run`].
    pub fn run_fill<F>(&mut self, x: &mut Array1<f64>, mut fill: F) -> Result<Report>
    where
        F: FnMut(ArrayView1<f64>, &mut Array1<f64>) -> f64,
    {
        if self.fire_is_flat() {
            return self.run_fire_fill(x, &mut fill);
        }
        if self.eon_fill_ok() {
            return self.run_eon_fill(x, &mut fill);
        }
        let _ = fill;
        Err(Error::LbfgsCap {
            what: "run_fill is the unbounded eOn cap or Euclidean FIRE",
        })
    }

    fn eon_fill_ok(&self) -> bool {
        matches!(self.inner, Inner::Lbfgs(_))
            && matches!(self.lbfgs_step(), crate::lbfgs::LbfgsStep::Capped { .. })
            && matches!(self.manifold, crate::manifold::ManifoldKind::Euclidean)
            && !self.project_rigid
            && self.exp.is_none()
    }

    fn run_eon_fill<F>(&mut self, x: &mut Array1<f64>, fill: &mut F) -> Result<Report>
    where
        F: FnMut(ArrayView1<f64>, &mut Array1<f64>) -> f64,
    {
        let n = x.len();
        if self.last_grad.len() != n {
            self.last_grad = Array1::zeros(n);
        }
        if self.scratch.len() != n {
            self.scratch = Array1::zeros(n);
        }
        let cap = self.lbfgs_step().capped_limit().ok_or(Error::LbfgsCap {
            what: "the cap is not positive",
        })?;
        let mut value = fill(x.view(), &mut self.last_grad);
        let tol = self.control.gtol;
        for _ in 0..self.control.maxiter {
            if !finite(value, self.last_grad.view()) {
                return Err(non_finite());
            }
            if self.stop_norm.of(self.last_grad.view()) < tol {
                break;
            }
            let use_cap = if self.trust {
                self.trust_delta.min(cap).max(1.0e-8)
            } else {
                cap
            };
            let end = self.eon_fill_step(x, use_cap, value, fill)?;
            if end.rejected {
                continue;
            }
            value = end.value;
        }
        self.last_value = value;
        self.store_x(x);
        let g = self.last_grad.clone();
        Ok(self.report(value, x, &g))
    }

    fn store_x(&mut self, x: &Array1<f64>) {
        match &mut self.last_pos {
            Some(pos) if pos.len() == x.len() => pos.assign(x),
            _ => self.last_pos = Some(x.clone()),
        }
    }

    /// One default L-BFGS session step: the eOn cap. The curvature probe,
    /// when it runs, is one evaluation and does not move `x`.
    pub(super) fn eon_session_step<O>(
        &mut self,
        obj: &O,
        x: &mut Array1<f64>,
        value: f64,
        grad: Array1<f64>,
    ) -> Result<Report>
    where
        O: eindir_core::DifferentiableObjective<f64> + ?Sized,
    {
        let cap = self.lbfgs_step().capped_limit().ok_or(Error::LbfgsCap {
            what: "the cap is not positive",
        })?;
        self.last_grad.assign(&grad);
        let end = self.eon_fill_step(x, cap, value, &mut |point, buf| {
            let (f, g) = obj.value_and_gradient(point);
            if buf.len() == g.len() {
                buf.assign(&g);
            } else {
                *buf = g;
            }
            f
        })?;
        if end.rejected {
            let g = self.last_grad.clone();
            return Ok(self.report(value, x, &g));
        }
        let g = self.last_grad.clone();
        self.remember(x, end.value, &g);
        Ok(self.report(end.value, x, &g))
    }

    fn eon_fill_step<F>(
        &mut self,
        x: &mut Array1<f64>,
        cap: f64,
        fold: f64,
        fill: &mut F,
    ) -> Result<FillEnd>
    where
        F: FnMut(ArrayView1<f64>, &mut Array1<f64>) -> f64,
    {
        if self.lindh_spec.is_some() {
            let xs: Vec<f64> = x.iter().copied().collect();
            let gs: Vec<f64> = self.last_grad.iter().copied().collect();
            self.ensure_lindh(&xs, &gs, |p| {
                let mut buf = Array1::zeros(p.len());
                let _ = fill(ArrayView1::from(p), &mut buf);
                buf
            });
        }
        let mut txn = match &mut self.inner {
            Inner::Lbfgs(solver) => solver.begin_txn(),
            _ => {
                return Err(Error::LbfgsCap {
                    what: "requires an L-BFGS session",
                });
            }
        };
        let wants_fd = match &self.inner {
            Inner::Lbfgs(solver) => solver.eon_wants_fd(),
            _ => false,
        };
        let seed = if wants_fd {
            self.eon_fd_fill(x, fill)
        } else {
            match &self.inner {
                Inner::Lbfgs(solver) => solver.eon_seed(),
                _ => EonSeed::Scale(EON_INVERSE_CURVATURE),
            }
        };
        let g = std::mem::replace(&mut self.last_grad, Array1::zeros(0));
        let take = match &mut self.inner {
            Inner::Lbfgs(solver) => solver.eon_step_from_seed(g.view(), cap, seed, &mut txn),
            _ => None,
        };
        self.last_grad = g;
        let Some(take) = take else {
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.rollback_txn(txn);
            }
            return Err(Error::LbfgsCap {
                what: "the capped step is zero",
            });
        };
        self.saved_x.assign(x);
        x.scaled_add(1.0, &take.step);
        let step_owned = take.step;
        let reset = match &self.inner {
            Inner::Lbfgs(solver) => solver.forgot,
            _ => false,
        };
        let new_value = fill(x.view(), &mut self.scratch);
        if !finite(new_value, self.scratch.view()) {
            x.assign(&self.saved_x);
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.recycle(step_owned);
                solver.rollback_txn(txn);
            }
            return Err(non_finite());
        }
        if self.trust && self.trust_reject(&step_owned, fold, new_value, cap) {
            x.assign(&self.saved_x);
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.recycle(step_owned);
                solver.rollback_txn(txn);
            }
            return Ok(FillEnd {
                value: fold,
                rejected: true,
            });
        }
        if self.eon_log.is_some() {
            let feat = feat_of(&step_owned, &self.last_grad, &self.scratch, cap, reset);
            if let Some(log) = self.eon_log.as_mut() {
                log.push(feat);
            }
        }
        if let Inner::Lbfgs(solver) = &mut self.inner {
            solver.recycle(step_owned);
        }
        {
            let prev = self.saved_x.as_slice().unwrap();
            let xs = x.as_slice().unwrap();
            let gold = self.last_grad.as_slice().unwrap();
            let gnew = self.scratch.as_slice().unwrap();
            if let Inner::Lbfgs(solver) = &mut self.inner {
                solver.store_pos_diff(prev, xs, gold, gnew, &mut txn);
                solver.commit_txn(txn);
            }
        }
        std::mem::swap(&mut self.last_grad, &mut self.scratch);
        self.steps += 1;
        Ok(FillEnd {
            value: new_value,
            rejected: false,
        })
    }

    /// `true` when the Newton-model ratio rejects this trial.
    /// A rejection shrinks the radius. An accepted step may shrink or
    /// grow it. The radius never leaves `(1e-8, trust_max)`.
    fn trust_reject(&mut self, step: &Array1<f64>, fold: f64, fnew: f64, cap: f64) -> bool {
        let mut gdot = 0.0;
        let mut atom_max = 0.0_f64;
        let mut i = 0;
        let n = step.len();
        while i < n {
            let end = (i + 3).min(n);
            let mut a2 = 0.0;
            for k in i..end {
                gdot += self.last_grad[k] * step[k];
                a2 += step[k] * step[k];
            }
            atom_max = atom_max.max(a2.sqrt());
            i = end;
        }
        let pred = if gdot < 0.0 { -0.5 * gdot } else { 0.0 };
        let ared = fold - fnew;
        let rho = if pred > 1.0e-16 {
            ared / pred
        } else if ared > 0.0 {
            1.0
        } else {
            -1.0
        };
        let on_bound = cap > 0.0 && atom_max >= 0.99 * cap;
        if rho < 0.1 && self.trust_delta > 1.0e-6 {
            self.trust_delta = (self.trust_delta * 0.25).max(1.0e-8);
            return true;
        }
        if rho < 0.25 {
            self.trust_delta = (self.trust_delta * 0.25).max(1.0e-8);
        } else if rho > 0.75 && on_bound {
            self.trust_delta = (self.trust_delta * 2.0).min(self.trust_max);
        }
        false
    }

    /// First-step curvature probe. The box is not applied: [`Solver::run_fill`]
    /// is the unbounded Euclidean step, where the clip is the identity.
    fn eon_fd_fill<F>(&mut self, x: &Array1<f64>, fill: &mut F) -> EonSeed
    where
        F: FnMut(ArrayView1<f64>, &mut Array1<f64>) -> f64,
    {
        let n = x.len();
        let mut n2 = 0.0;
        for v in self.last_grad.iter() {
            n2 += *v * *v;
        }
        let norm = n2.sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return EonSeed::Scale(EON_INVERSE_CURVATURE);
        }
        let mut probe = Array1::zeros(n);
        for i in 0..n {
            let u = -self.last_grad[i] / norm;
            probe[i] = x[i] + EON_FINITE_DIFFERENCE * u;
        }
        if self.scratch.len() != n {
            self.scratch = Array1::zeros(n);
        }
        let probe_v = fill(probe.view(), &mut self.scratch);
        if !finite(probe_v, self.scratch.view()) {
            return EonSeed::Steepest;
        }
        let mut dg_u = 0.0;
        for i in 0..n {
            let u = -self.last_grad[i] / norm;
            dg_u += (self.scratch[i] - self.last_grad[i]) * u;
        }
        let curvature = dg_u / EON_FINITE_DIFFERENCE;
        if curvature.is_finite() && curvature > 0.0 {
            EonSeed::Scale(1.0 / curvature)
        } else {
            EonSeed::Steepest
        }
    }

    fn run_fire_fill<F>(&mut self, x: &mut Array1<f64>, fill: &mut F) -> Result<Report>
    where
        F: FnMut(ArrayView1<f64>, &mut Array1<f64>) -> f64,
    {
        let n = x.len();
        if self.last_grad.len() != n {
            self.last_grad = Array1::zeros(n);
        }
        if self.scratch.len() != n {
            self.scratch = Array1::zeros(n);
        }
        if self.scratch_b.len() != n {
            self.scratch_b = Array1::zeros(n);
        }
        let mut value = fill(x.view(), &mut self.last_grad);
        let tol = self.control.gtol;
        for _ in 0..self.control.maxiter {
            if !finite(value, self.last_grad.view()) {
                return Err(non_finite());
            }
            if self.stop_norm.of(self.last_grad.view()) < tol {
                break;
            }
            value = self.fire_fill_step(x, fill)?;
        }
        self.last_value = value;
        self.store_x(x);
        let g = self.last_grad.clone();
        Ok(self.report(value, x, &g))
    }

    fn fire_fill_step<F>(&mut self, x: &mut Array1<f64>, fill: &mut F) -> Result<f64>
    where
        F: FnMut(ArrayView1<f64>, &mut Array1<f64>) -> f64,
    {
        let v1 = matches!(
            self.inner,
            Inner::Fire(ref state) if matches!(state.kind, FireKind::V1)
        );
        if v1 {
            return self.fire_v1_fill_step(x, fill);
        }
        let n = x.len();
        self.saved_x.assign(x);
        self.scratch.assign(&self.last_grad);
        self.scratch.mapv_inplace(|v| -v);
        let saved = {
            let Inner::Fire(state) = &mut self.inner else {
                return Err(Error::FireStalled { steps: self.steps });
            };
            let saved = state.checkpoint();
            let prep = fire_prepare_v2(state, &self.scratch);
            (saved, prep)
        };
        let (saved, prep) = saved;
        if let FirePreparation::Rollback(correction) = prep {
            for i in 0..n {
                x[i] += correction[i];
            }
            let v = fill(x.view(), &mut self.last_grad);
            if !finite(v, self.last_grad.view()) {
                x.assign(&self.saved_x);
                if let Inner::Fire(state) = &mut self.inner {
                    state.restore(saved);
                }
                return Err(non_finite());
            }
            self.scratch.assign(&self.last_grad);
            self.scratch.mapv_inplace(|v| -v);
        } else if let FirePreparation::Stalled = prep {
            return Err(Error::FireStalled { steps: self.steps });
        }
        if let Inner::Fire(state) = &mut self.inner {
            fire_displacement_into(state, &self.scratch, &mut self.scratch_b);
        }
        // Forward origin, overwriting the force.
        self.scratch.assign(x);
        let (dt, v2) = if let Inner::Fire(state) = &self.inner {
            (state.dt, matches!(state.kind, FireKind::V2))
        } else {
            (0.0, false)
        };
        for i in 0..n {
            x[i] += self.scratch_b[i];
        }
        if let Some(cap) = self.atom_maxmove {
            scale_step_atom(&self.scratch, x, cap);
        } else if let Some(cap) = self.control.maxmove {
            scale_step(&self.scratch, x, cap);
        }
        if v2
            && dt > 0.0
            && let Inner::Fire(state) = &mut self.inner
        {
            for i in 0..n {
                state.vel[i] = (x[i] - self.scratch[i]) / dt;
            }
        }
        let new_value = fill(x.view(), &mut self.last_grad);
        if !finite(new_value, self.last_grad.view()) {
            x.assign(&self.saved_x);
            if let Inner::Fire(state) = &mut self.inner {
                state.restore(saved);
            }
            return Err(non_finite());
        }
        self.scratch.assign(&self.last_grad);
        self.scratch.mapv_inplace(|v| -v);
        if let Inner::Fire(state) = &mut self.inner {
            fire_after_v1(state, &self.scratch);
        }
        self.steps += 1;
        Ok(new_value)
    }

    /// FIRE 1.0 on an unbounded box. The velocity is restored only when
    /// the trial is non-finite, from the checkpoint taken once per step.
    fn fire_v1_fill_step<F>(&mut self, x: &mut Array1<f64>, fill: &mut F) -> Result<f64>
    where
        F: FnMut(ArrayView1<f64>, &mut Array1<f64>) -> f64,
    {
        let n = x.len();
        let xs = x.as_slice().unwrap();
        let gs = self.last_grad.as_slice().unwrap();
        let mut sx = self.saved_x.as_slice_mut().unwrap();
        if sx.len() != n {
            self.saved_x = Array1::from_vec(xs.to_vec());
            sx = self.saved_x.as_slice_mut().unwrap();
        }
        sx.copy_from_slice(xs);
        let mut force = self.scratch.as_slice_mut().unwrap();
        if force.len() != n {
            self.scratch = Array1::zeros(n);
            force = self.scratch.as_slice_mut().unwrap();
        }
        for i in 0..n {
            force[i] = -gs[i];
        }
        {
            let Inner::Fire(state) = &mut self.inner else {
                return Err(Error::FireStalled { steps: self.steps });
            };
            let dt = state.dt;
            let vel = state.vel.as_slice_mut().unwrap();
            let dx = self.scratch_b.as_slice_mut().unwrap();
            for i in 0..n {
                vel[i] += force[i] * dt;
                dx[i] = vel[i] * dt;
            }
        }
        // The checkpoint above is dropped. Retake it before the kick so a
        // non-finite trial can undo the velocity. Done inside the block
        // next to the kick; `saved` is re-read below via the state.
        let cap = self.atom_maxmove.or(self.control.maxmove);
        {
            let dx = self.scratch_b.as_slice().unwrap();
            let origin = self.saved_x.as_slice().unwrap();
            let xv = x.as_slice_mut().unwrap();
            for i in 0..n {
                xv[i] = origin[i] + dx[i];
            }
            if let Some(cap) = cap {
                scale_cap(xv, origin, cap, self.atom_maxmove.is_some());
            }
        }
        let new_value = fill(x.view(), &mut self.last_grad);
        if !finite(new_value, self.last_grad.view()) {
            x.assign(&self.saved_x);
            return Err(non_finite());
        }
        if let Inner::Fire(state) = &mut self.inner {
            let g = self.last_grad.as_slice().unwrap();
            let force = self.scratch.as_slice_mut().unwrap();
            for i in 0..n {
                force[i] = -g[i];
            }
            fire_after_v1(state, &self.scratch);
        }
        self.steps += 1;
        Ok(new_value)
    }
}

fn scale_cap(trial: &mut [f64], origin: &[f64], cap: f64, per_atom: bool) {
    if per_atom {
        let mut max_a = 0.0_f64;
        let mut i = 0;
        while i + 3 <= trial.len() {
            let dx = trial[i] - origin[i];
            let dy = trial[i + 1] - origin[i + 1];
            let dz = trial[i + 2] - origin[i + 2];
            max_a = max_a.max((dx * dx + dy * dy + dz * dz).sqrt());
            i += 3;
        }
        if i < trial.len() {
            let mut r2 = 0.0;
            let mut j = i;
            while j < trial.len() {
                let d = trial[j] - origin[j];
                r2 += d * d;
                j += 1;
            }
            max_a = max_a.max(r2.sqrt());
        }
        if max_a > cap && max_a > 0.0 {
            let s = cap / max_a;
            for i in 0..trial.len() {
                trial[i] = origin[i] + s * (trial[i] - origin[i]);
            }
        }
    } else {
        let mut n2 = 0.0;
        for i in 0..trial.len() {
            let d = trial[i] - origin[i];
            n2 += d * d;
        }
        let nrm = n2.sqrt();
        if nrm > cap && nrm > 0.0 {
            let s = cap / nrm;
            for i in 0..trial.len() {
                trial[i] = origin[i] + s * (trial[i] - origin[i]);
            }
        }
    }
}
