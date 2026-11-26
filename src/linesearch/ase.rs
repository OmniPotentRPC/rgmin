//! The line search of ASE 3.29, `ase.utils.linesearch.LineSearch`.
//!
//! ASE's `BFGSLineSearch` and `LBFGSLineSearch` build a fresh search
//! every step and run its More-Thuente variant from the unit step. This
//! module follows that code statement for statement so that a session
//! in this mode evaluates the trials ASE evaluates.
//!
//! The differences from [`super::LineSearch::Wolfe`] are the step
//! limits and the end of the search. `maxstep` limits each increment
//! from the previous trial, so one search can move an atom further
//! than `maxstep` in total. A trial is clamped to `[1e-8, stpmax]`, and
//! an opening step of 1 above `stpmax` is an error before any
//! evaluation. The search ends when ASE's strong-Wolfe test holds, when
//! one of its warnings fires (rounding, the `xtol` width, a trial at
//! `stpmax` or `1e-8`), or when it flags `no_update`. ASE 3.29 returns
//! the last trial in each of those cases: its failure test compares
//! `task[1:4]` with `"WARN"`, which never holds, so only an input error
//! fails there. Here a search also fails when its evaluation budget is
//! spent or a trial is not finite. A failed search returns `None` and
//! never falls back to the best trial.
//!
//! <https://gitlab.com/ase/ase/-/blob/3.29.0/ase/utils/linesearch.py>

use ndarray::{Array1, ArrayView1};

use super::LineOutcome;

const STPMIN: f64 = 1e-8;
const XTOL: f64 = 1e-14;
const XTRAPL: f64 = 1.1;
const XTRAPU: f64 = 4.0;

/// The ASE search a BFGS or L-BFGS session runs in place of Wolfe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AseSearch {
    /// ASE `stpmax`, the largest step multiplier.
    pub(crate) stpmax: f64,
}

/// How a call to ASE's `step` left the search.
#[derive(Clone, Copy, PartialEq)]
enum Task {
    /// Evaluate the returned trial.
    Fg,
    /// The strong-Wolfe test or a warning ended the search there.
    Done,
    /// An input check failed before the first evaluation.
    Error,
}

/// Python's `max(a, b)`: `a` unless `b > a`.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's `min(a, b)`: `a` unless `b < a`.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// ASE's `standard_gradient_norm(pk * dr)`: the largest per-atom norm.
///
/// A trailing partial triple counts as one more atom.
fn max_atom_norm_scaled(dir: ArrayView1<'_, f64>, dr: f64) -> f64 {
    let n = dir.len();
    let mut longest = f64::NEG_INFINITY;
    let mut i = 0;
    while i < n {
        let end = (i + 3).min(n);
        let mut sum = 0.0;
        for &v in dir.slice(ndarray::s![i..end]) {
            let scaled = v * dr;
            sum += scaled * scaled;
        }
        longest = py_max(longest, sum.sqrt());
        i = end;
    }
    longest
}

/// The persistent state of ASE's `LineSearch` between `step` calls.
struct State<'a> {
    dir: ArrayView1<'a, f64>,
    maxstep: f64,
    stpmax: f64,
    c1: f64,
    c2: f64,
    old_stp: f64,
    no_update: bool,
    bracket: bool,
    stage: u8,
    ginit: f64,
    gtest: f64,
    gx: f64,
    gy: f64,
    finit: f64,
    fx: f64,
    fy: f64,
    stx: f64,
    sty: f64,
    stmin: f64,
    stmax: f64,
    width: f64,
    width1: f64,
}

impl State<'_> {
    fn determine_step(&self, stp: f64) -> f64 {
        let mut dr = stp - self.old_stp;
        let longest = max_atom_norm_scaled(self.dir, dr);
        if longest >= self.maxstep {
            dr *= self.maxstep / longest;
        }
        self.old_stp + dr
    }

    /// ASE's `step` on the `START` task: check the inputs, set up the
    /// interval, and return the first trial.
    fn start(&mut self, stp: f64, f: f64, g: f64) -> (f64, Task) {
        let error = stp < STPMIN
            || stp > self.stpmax
            || g >= 0.0
            || self.c1 < 0.0
            || self.c2 < 0.0
            || self.stpmax < STPMIN;
        if error {
            return (stp, Task::Error);
        }
        self.bracket = false;
        self.stage = 1;
        self.finit = f;
        self.ginit = g;
        self.gtest = self.c1 * self.ginit;
        self.width = self.stpmax - STPMIN;
        self.width1 = self.width / 0.5;
        self.stx = 0.0;
        self.fx = self.finit;
        self.gx = self.ginit;
        self.sty = 0.0;
        self.fy = self.finit;
        self.gy = self.ginit;
        self.stmin = 0.0;
        self.stmax = stp + XTRAPU * stp;
        (self.determine_step(stp), Task::Fg)
    }

    /// ASE's `step` after an evaluation at `stp` with value `f` and
    /// slope `g`.
    fn next(&mut self, stp: f64, f: f64, g: f64) -> (f64, Task) {
        let ftest = self.finit + stp * self.gtest;
        if self.stage == 1 && f < ftest && g >= 0.0 {
            self.stage = 2;
        }
        let mut done = false;
        if self.bracket && (stp <= self.stmin || stp >= self.stmax) {
            done = true;
        }
        if self.bracket && self.stmax - self.stmin <= XTOL * self.stmax {
            done = true;
        }
        if stp == self.stpmax && f <= ftest && g <= self.gtest {
            done = true;
        }
        if stp == STPMIN && (f > ftest || g >= self.gtest) {
            done = true;
        }
        if f <= ftest && g.abs() <= self.c2 * (-self.ginit) {
            done = true;
        }
        if done {
            return (stp, Task::Done);
        }
        let mut stp = self.update(stp, f, g);
        if self.bracket {
            if (self.sty - self.stx).abs() >= 0.66 * self.width1 {
                stp = self.stx + 0.5 * (self.sty - self.stx);
            }
            self.width1 = self.width;
            self.width = (self.sty - self.stx).abs();
        }
        if self.bracket {
            self.stmin = py_min(self.stx, self.sty);
            self.stmax = py_max(self.stx, self.sty);
        } else {
            self.stmin = stp + XTRAPL * (stp - self.stx);
            self.stmax = stp + XTRAPU * (stp - self.stx);
        }
        stp = py_max(stp, STPMIN);
        stp = py_min(stp, self.stpmax);
        if self.stx == stp && stp == self.stpmax && self.stmin > self.stpmax {
            self.no_update = true;
        }
        if (self.bracket && stp < self.stmin)
            || stp >= self.stmax
            || (self.bracket && self.stmax - self.stmin < XTOL * self.stmax)
        {
            stp = self.stx;
        }
        (stp, Task::Fg)
    }

    /// ASE's `update`: the safeguarded More-Thuente step from the
    /// interval `[stx, sty]` and the trial `stp`, limited by
    /// [`Self::determine_step`].
    fn update(&mut self, stp: f64, fp: f64, gp: f64) -> f64 {
        let (stx, fx, gx) = (self.stx, self.fx, self.gx);
        let (sty, fy, gy) = (self.sty, self.fy, self.gy);
        let (stpmin, stpmax) = (self.stmin, self.stmax);
        let sign = gp * (gx / gx.abs());
        let stpf;
        if fp > fx {
            let theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp;
            let s = py_max(py_max(theta.abs(), gx.abs()), gp.abs());
            let mut gamma = s * ((theta / s).powi(2) - (gx / s) * (gp / s)).sqrt();
            if stp < stx {
                gamma = -gamma;
            }
            let p = (gamma - gx) + theta;
            let q = ((gamma - gx) + gamma) + gp;
            let r = p / q;
            let stpc = stx + r * (stp - stx);
            let stpq = stx + ((gx / ((fx - fp) / (stp - stx) + gx)) / 2.0) * (stp - stx);
            stpf = if (stpc - stx).abs() < (stpq - stx).abs() {
                stpc
            } else {
                stpc + (stpq - stpc) / 2.0
            };
            self.bracket = true;
        } else if sign < 0.0 {
            let theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp;
            let s = py_max(py_max(theta.abs(), gx.abs()), gp.abs());
            let mut gamma = s * ((theta / s).powi(2) - (gx / s) * (gp / s)).sqrt();
            if stp > stx {
                gamma = -gamma;
            }
            let p = (gamma - gp) + theta;
            let q = ((gamma - gp) + gamma) + gx;
            let r = p / q;
            let stpc = stp + r * (stx - stp);
            let stpq = stp + (gp / (gp - gx)) * (stx - stp);
            stpf = if (stpc - stp).abs() > (stpq - stp).abs() {
                stpc
            } else {
                stpq
            };
            self.bracket = true;
        } else if gp.abs() < gx.abs() {
            let theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp;
            let s = py_max(py_max(theta.abs(), gx.abs()), gp.abs());
            let mut gamma = s * py_max(0.0, (theta / s).powi(2) - (gx / s) * (gp / s)).sqrt();
            if stp > stx {
                gamma = -gamma;
            }
            let p = (gamma - gp) + theta;
            let q = (gamma + (gx - gp)) + gamma;
            let r = p / q;
            let stpc = if r < 0.0 && gamma != 0.0 {
                stp + r * (stx - stp)
            } else if stp > stx {
                stpmax
            } else {
                stpmin
            };
            let stpq = stp + (gp / (gp - gx)) * (stx - stp);
            if self.bracket {
                let mut f = if (stpc - stp).abs() < (stpq - stp).abs() {
                    stpc
                } else {
                    stpq
                };
                f = if stp > stx {
                    py_min(stp + 0.66 * (sty - stp), f)
                } else {
                    py_max(stp + 0.66 * (sty - stp), f)
                };
                stpf = f;
            } else {
                let mut f = if (stpc - stp).abs() > (stpq - stp).abs() {
                    stpc
                } else {
                    stpq
                };
                f = py_min(stpmax, f);
                f = py_max(stpmin, f);
                stpf = f;
            }
        } else if self.bracket {
            let theta = 3.0 * (fp - fy) / (sty - stp) + gy + gp;
            let s = py_max(py_max(theta.abs(), gy.abs()), gp.abs());
            let mut gamma = s * ((theta / s).powi(2) - (gy / s) * (gp / s)).sqrt();
            if stp > sty {
                gamma = -gamma;
            }
            let p = (gamma - gp) + theta;
            let q = ((gamma - gp) + gamma) + gy;
            let r = p / q;
            stpf = stp + r * (sty - stp);
        } else if stp > stx {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
        if fp > fx {
            self.sty = stp;
            self.fy = fp;
            self.gy = gp;
        } else {
            if sign < 0.0 {
                self.sty = stx;
                self.fy = fx;
                self.gy = gx;
            }
            self.stx = stp;
            self.fx = fp;
            self.gx = gp;
        }
        self.determine_step(stpf)
    }
}

/// `x0 + stp p` as NumPy forms `xk + stp * pk`.
fn trial_point(pos: ArrayView1<'_, f64>, dir: ArrayView1<'_, f64>, stp: f64) -> Array1<f64> {
    Array1::from_iter(pos.iter().zip(dir.iter()).map(|(x, p)| x + stp * p))
}

/// Run ASE's search from `pos` along `dir` with the known `f0` and `g0`.
///
/// `maxstep` limits the largest per-atom move of each increment; `None`
/// leaves the increments unlimited. `budget` bounds the evaluations.
/// A direction shorter than ASE's floor `sqrt(n / 3 * 1e-10)` is
/// lengthened to it first, as ASE's optimizers do before their search.
/// Returns the trial the search ended on, or `None` when the search
/// failed: an input error, a spent budget, or a non-finite trial.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search<F>(
    mut oracle: F,
    pos: ArrayView1<'_, f64>,
    f0: f64,
    g0: ArrayView1<'_, f64>,
    dir: ArrayView1<'_, f64>,
    maxstep: Option<f64>,
    search: AseSearch,
    c1: f64,
    c2: f64,
    budget: usize,
) -> Option<LineOutcome>
where
    F: FnMut(ArrayView1<'_, f64>) -> (f64, Array1<f64>),
{
    let floor = (pos.len() as f64 / 3.0 * 1e-10).sqrt();
    let size = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
    let dir = if size <= floor {
        dir.mapv(|v| v / (size / floor))
    } else {
        dir.to_owned()
    };
    let derphi0 = crate::vecops::dot(g0, dir.view());
    let mut state = State {
        dir: dir.view(),
        maxstep: maxstep.unwrap_or(f64::INFINITY),
        stpmax: search.stpmax,
        c1,
        c2,
        old_stp: 0.0,
        no_update: false,
        bracket: false,
        stage: 1,
        ginit: 0.0,
        gtest: 0.0,
        gx: 0.0,
        gy: 0.0,
        finit: 0.0,
        fx: 0.0,
        fy: 0.0,
        stx: 0.0,
        sty: 0.0,
        stmin: 0.0,
        stmax: 0.0,
        width: 0.0,
        width1: 0.0,
    };
    let (mut stp, task) = state.start(1.0, f0, derphi0);
    if task == Task::Error {
        return None;
    }
    let mut evals = 0;
    loop {
        if evals >= budget {
            return None;
        }
        let x = trial_point(pos, dir.view(), stp);
        let (f, g) = oracle(x.view());
        evals += 1;
        if !crate::step::finite(f, g.view()) {
            return None;
        }
        let dphi = crate::vecops::dot(g.view(), dir.view());
        state.old_stp = stp;
        let end = LineOutcome {
            x,
            f,
            g,
            alpha: stp,
            evals,
        };
        if state.no_update {
            return Some(end);
        }
        let (next, task) = state.next(stp, f, dphi);
        if task != Task::Fg {
            return Some(end);
        }
        stp = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::cell::RefCell;

    /// `(x - center)^2 / 2` on one atom. Center 1 is the ASE probe.
    fn bowl(center: f64, x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
        let z = x[0] - center;
        (0.5 * z * z, array![z, 0.0, 0.0])
    }

    fn run(
        center: f64,
        maxstep: Option<f64>,
        stpmax: f64,
        budget: usize,
    ) -> (Option<LineOutcome>, Vec<f64>) {
        let seen = RefCell::new(Vec::new());
        let oracle = |x: ArrayView1<'_, f64>| {
            seen.borrow_mut().push(x[0]);
            bowl(center, x)
        };
        let pos = array![0.0, 0.0, 0.0];
        let (f0, g0) = bowl(center, pos.view());
        let dir = array![1.0, 0.0, 0.0];
        let out = search(
            oracle,
            pos.view(),
            f0,
            g0.view(),
            dir.view(),
            maxstep,
            AseSearch { stpmax },
            0.23,
            0.46,
            budget,
        );
        (out, seen.into_inner())
    }

    #[test]
    fn maxstep_limits_each_increment_not_the_step() {
        let (out, seen) = run(1.0, Some(0.2), 50.0, 20);
        let out = out.expect("ASE converges at 0.6");
        assert_eq!(seen.len(), 3);
        for (got, want) in seen.iter().zip([0.2, 0.4, 0.6]) {
            assert!((got - want).abs() < 1e-12, "{seen:?}");
        }
        assert!((out.x[0] - 0.6).abs() < 1e-12);
        assert_eq!(out.evals, 3);
    }

    #[test]
    fn an_opening_step_above_stpmax_fails_before_any_evaluation() {
        let (out, seen) = run(1.0, Some(10.0), 0.5, 20);
        assert!(out.is_none());
        assert!(seen.is_empty());
    }

    #[test]
    fn a_descending_trial_at_stpmax_ends_the_search_there() {
        // ASE's STP = maxstep warning returns the trial, not a failure.
        let (out, seen) = run(10.0, None, 1.0, 20);
        let out = out.expect("the warning returns the trial");
        assert_eq!(seen, vec![1.0]);
        assert_eq!(out.x[0], 1.0);
    }

    #[test]
    fn a_spent_budget_fails_without_a_fallback() {
        let (out, seen) = run(1.0, Some(0.2), 50.0, 2);
        assert!(out.is_none());
        assert_eq!(seen.len(), 2);
    }
}
