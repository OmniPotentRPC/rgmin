//! The published FIRE 2.0 schedule stops when `vdfmax` is exceeded.
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Control, Error, FireConfig, FireKind, FirePreset, FireTrace, FireVariant, Method, Oracle,
    Solver,
};

struct Bowl;

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        use std::sync::OnceLock;
        static B: OnceLock<Bounds<f64>> = OnceLock::new();
        B.get_or_init(|| Bounds::new(array![-10.0], array![10.0], 0.0))
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0] * x[0]
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![2.0 * x[0]]
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn published() -> Solver {
    let mut solver = Solver::new(
        Method::fire(FireKind::V2),
        Control::default()
            .with_maxiter(5)
            .with_gtol(1e-12)
            .with_istep(0.1),
        1,
    );
    solver.set_fire_variant(FireVariant::Guenole2020);
    solver
}

#[test]
fn a_zero_vdfmax_stops_the_first_step_without_moving() {
    let mut solver = published();
    solver.set_fire_vdfmax(0);
    let mut x = array![1.0];
    let err = solver.step(&Bowl, &mut x).unwrap_err();
    assert!(matches!(err, Error::FireStalled { steps: 0 }), "{err}");
    assert_eq!(x[0], 1.0);
}

#[test]
fn forget_clears_the_negative_power_count() {
    let mut solver = published();
    solver.set_fire_vdfmax(1);
    let mut x = array![1.0];
    solver
        .step(&Bowl, &mut x)
        .expect("the first non-positive step is allowed");
    solver.forget();
    let again = solver.step(&Bowl, &mut x);
    assert!(again.is_ok(), "{again:?}");
}

fn lj_cluster(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d: [f64; 3] = std::array::from_fn(|a| x[3 * i + a] - x[3 * j + a]);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let s6 = r2.powi(-3);
            e += 4.0 * (s6 * s6 - s6);
            let du_over_r = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
            for a in 0..3 {
                g[3 * i + a] += du_over_r * d[a];
                g[3 * j + a] -= du_over_r * d[a];
            }
        }
    }
    (e, g)
}

fn lj7_start() -> Array1<f64> {
    let mut x = Vec::with_capacity(21);
    for k in 0..7 {
        let p = [(k % 2) as f64, ((k / 2) % 2) as f64, (k / 4) as f64];
        for (a, v) in p.iter().enumerate() {
            x.push(1.1 * v + 0.03 * ((3 * k + a) as f64).sin());
        }
    }
    Array1::from(x)
}

/// Reference transcription of Guénolé et al., Comput. Mater. Sci. 175,
/// 109584 (2020) Algorithm 2 / LAMMPS `min_style fire` (`integrator
/// eulerimplicit halfstepback yes initialdelay yes vdfmax 2000`),
/// recording `(dt, dt_max, alpha, power, n_pos, half_step_backs,
/// clipped_moves)`.
fn lammps_fire2_reference(
    mut x: Array1<f64>,
    dt0: f64,
    dmax: f64,
    gtol: f64,
    maxiter: usize,
) -> (Array1<f64>, usize, Vec<FireTrace>) {
    let n = x.len();
    let mut vel = Array1::<f64>::zeros(n);
    let mut dt = dt0;
    let dt_max = 10.0 * dt0;
    let dt_min = 0.02 * dt0;
    let n_delay = 20usize;
    let f_inc = 1.1;
    let f_dec = 0.5;
    let alpha_start = 0.25;
    let mut alpha = alpha_start;
    let f_alpha = 0.99;
    let vdfmax = 2_000usize;
    let mut n_pos = 0usize;
    let mut n_neg = 0usize;
    let mut hsb_total = 0usize;
    let mut clip_total = 0usize;
    let mut evals = 1usize;
    let (_, mut grad) = lj_cluster(x.view());
    let mut trace = Vec::new();

    for steps in 0..maxiter {
        let mut force = grad.mapv(|g| -g);
        let power = force.dot(&vel);
        let mut hsb = false;
        if power > 0.0 {
            n_pos += 1;
            n_neg = 0;
            if n_pos > n_delay {
                dt = (dt * f_inc).min(dt_max);
                alpha *= f_alpha;
            }
        } else {
            n_pos = 0;
            n_neg += 1;
            assert!(n_neg <= vdfmax, "exceeded vdfmax");
            if steps >= n_delay {
                if dt * f_dec >= dt_min {
                    dt *= f_dec;
                }
                alpha = alpha_start;
            }
            if vel.iter().any(|v| *v != 0.0) {
                hsb = true;
                x.scaled_add(-0.5 * dt, &vel);
                vel.fill(0.0);
                evals += 1;
                grad = lj_cluster(x.view()).1;
                force = grad.mapv(|g| -g);
            } else {
                vel.fill(0.0);
            }
        }
        vel.scaled_add(dt, &force);
        let vnorm = vel.dot(&vel).sqrt();
        let fnorm = force.dot(&force).sqrt();
        if fnorm > 0.0 {
            let scale = alpha * vnorm / fnorm;
            for i in 0..n {
                vel[i] = (1.0 - alpha) * vel[i] + scale * force[i];
            }
        }
        let dx = &vel * dt;
        let mut trial = &x + &dx;
        let mut max_atom = 0.0_f64;
        for chunk in dx.exact_chunks(3) {
            max_atom = max_atom.max(chunk.dot(&chunk).sqrt());
        }
        let mut clipped = false;
        if max_atom > dmax {
            clipped = true;
            let s = dmax / max_atom;
            trial = &x + &(&dx * s);
            let taken = &trial - &x;
            let factor = taken.dot(&taken).sqrt() / dx.dot(&dx).sqrt();
            if factor < 1.0 {
                vel.mapv_inplace(|v| v * factor);
            }
        }
        x = trial;
        evals += 1;
        grad = lj_cluster(x.view()).1;
        hsb_total += usize::from(hsb);
        clip_total += usize::from(clipped);
        trace.push(
            FireTrace::default()
                .with_dt(dt)
                .with_dt_max(dt_max)
                .with_alpha(alpha)
                .with_power(power)
                .with_n_pos(n_pos)
                .with_half_step_backs(hsb_total)
                .with_clipped_moves(clip_total),
        );
        if grad.dot(&grad).sqrt() < gtol {
            break;
        }
    }
    (x, evals, trace)
}

#[test]
fn fire2_trace_matches_lammps_reference_step_for_step_on_lj7() {
    let dt0 = 0.005;
    let dmax = 0.1;
    let gtol = 1.0 / 1_000_000.0;
    let maxiter = 1_000;
    let x0 = lj7_start();
    let (want_x, want_evals, want_trace) =
        lammps_fire2_reference(x0.clone(), dt0, dmax, gtol, maxiter);

    let evals = std::sync::atomic::AtomicUsize::new(0);
    let obj = Oracle::unbounded(21, |x: ArrayView1<f64>| {
        evals.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        lj_cluster(x)
    });
    let control = Control::default()
        .with_maxiter(maxiter)
        .with_gtol(gtol)
        .with_istep(dt0);
    let mut solver = Solver::new(Method::fire(FireKind::V2), control, 21);
    solver.set_fire_preset(FirePreset::Guenole2020).unwrap();
    solver.set_atom_maxmove(dmax);
    solver.start_fire_trace();

    let mut x = x0;
    for _ in 0..maxiter {
        let rep = solver.step(&obj, &mut x).unwrap();
        if rep.grad_norm < gtol {
            break;
        }
    }
    let got_trace = solver.take_fire_trace();
    let got_evals = evals.load(std::sync::atomic::Ordering::Relaxed);

    assert_eq!(got_trace.len(), want_trace.len());
    assert_eq!(got_evals, want_evals);
    let last = want_trace.last().unwrap();
    assert!(last.half_step_backs > 0);
    assert_eq!(want_evals, 1 + want_trace.len() + last.half_step_backs);
    let p_rel = 1.0 / 10_000_000_000.0;
    let x_tol = 1.0 / 1_000_000_000_000.0;
    for (step, (got, want)) in got_trace.iter().zip(want_trace.iter()).enumerate() {
        assert_eq!(got.dt, want.dt, "step {step} dt");
        assert_eq!(got.dt_max, want.dt_max, "step {step} dt_max");
        assert_eq!(got.alpha, want.alpha, "step {step} alpha");
        assert_eq!(got.n_pos, want.n_pos, "step {step} n_pos");
        assert_eq!(
            got.half_step_backs, want.half_step_backs,
            "step {step} half_step_backs"
        );
        assert_eq!(
            got.clipped_moves, want.clipped_moves,
            "step {step} clipped_moves"
        );
        assert!(
            (got.power - want.power).abs() <= p_rel * want.power.abs().max(1.0),
            "step {step} power {} vs {}",
            got.power,
            want.power
        );
    }
    for i in 0..x.len() {
        assert!(
            (x[i] - want_x[i]).abs() <= x_tol,
            "coord {i}: {} vs {}",
            x[i],
            want_x[i]
        );
    }
}

fn stiff_soft_overshoot(cap_two_step_overshoot: bool) -> (Array1<f64>, Vec<FireTrace>) {
    let dt = 0.1;
    let control = Control::default()
        .with_maxiter(180)
        .with_gtol(0.0)
        .with_istep(dt);
    let mut solver = Solver::new(Method::fire(FireKind::V2), control, 2);
    solver.set_fire_preset(FirePreset::Guenole2020).unwrap();
    solver
        .set_fire_config(
            FireConfig::ase()
                .with_dt(dt)
                .with_dt_max(1.0)
                .with_n_min(20)
                .with_alpha(0.25)
                .with_cap_two_step_overshoot(cap_two_step_overshoot),
        )
        .unwrap();
    solver.set_atom_maxmove(0.1);
    solver.start_fire_trace();
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        (
            7.0 * x[0] * x[0] + 0.02 * x[1] * x[1],
            array![14.0 * x[0], 0.04 * x[1]],
        )
    });
    let mut x = array![0.02, 0.05];
    for _ in 0..180 {
        solver.step(&obj, &mut x).unwrap();
    }
    (x, solver.take_fire_trace())
}

#[test]
fn two_step_overshoot_cap_engages_and_changes_the_stiff_soft_trajectory() {
    let (plain_x, plain) = stiff_soft_overshoot(false);
    let (capped_x, capped) = stiff_soft_overshoot(true);
    let cap_step = capped
        .iter()
        .position(|entry| entry.dt_max < 1.0)
        .expect("the stiff mode must trigger the cap");

    assert_eq!(capped[cap_step - 1].n_pos, 1);
    assert!(capped[cap_step].power <= 0.0);
    assert_eq!(plain[cap_step].dt_max, 1.0);
    assert_ne!(plain_x, capped_x);
    assert_ne!(plain, capped);
}
