//! Bitwise trajectory dump of the optimiser paths below.
//!
//! Prints one line per accepted iterate: the case name, the step, then
//! the IEEE-754 bits of the value, the gradient norm and every
//! coordinate. Reports append their own step count, stop norm, status
//! and retained-gradient state. Diff the output of two builds. An empty
//! diff means these recorded results are equal bit for bit; Python
//! bindings and test-only fixtures have their own suites.
//!
//! The oracles are the ones the test suite already uses: the Morse chain
//! of `tests/optimizer_trajectory.rs`, the Rosenbrock valley with its
//! analytic Hessian of `tests/hvp.rs`, and a Rayleigh quotient for the
//! manifold sessions. With `--features capi` the C session waist runs on
//! a generalized Rosenbrock, setter by setter.
//!
//! ```text
//! cargo run --release --example trajectory_dump > before.txt
//! ```

use std::fmt::Write as _;

use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::nlcg::{Conjugacy, Restart};
use rgmin::rtr::{RtrOptions, minimize_rtr};
use rgmin::{
    Accept, Control, DistanceConstraints, FireKind, HessianOracle, HvpOracle, LbfgsStep,
    LineSearch, ManifoldKind, Method, Oracle, Report, RigidMode, Solver, minimize_eon_cg,
    minimize_method, minimize_method_hess, minimize_method_hvp, minimize_scg,
};
use rgmin::{EonCg, ScgParams};

const D0: f64 = 0.7102;
const ALPHA: f64 = 1.6047;
const R0: f64 = 2.897;

fn morse_chain(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut energy = 0.0;
    let mut grad = Array1::zeros(x.len());
    for i in 0..n.saturating_sub(1) {
        let mut diff = [0.0; 3];
        let mut r2 = 0.0;
        for a in 0..3 {
            diff[a] = x[3 * i + a] - x[3 * (i + 1) + a];
            r2 += diff[a] * diff[a];
        }
        let r = r2.sqrt();
        if !r.is_finite() || r <= 0.0 {
            continue;
        }
        let ex = (-ALPHA * (r - R0)).exp();
        energy += D0 * (ex * ex - 2.0 * ex);
        let slope = (2.0 * D0 * ALPHA * ex * (1.0 - ex)) / r;
        for a in 0..3 {
            let piece = slope * diff[a];
            grad[3 * i + a] += piece;
            grad[3 * (i + 1) + a] -= piece;
        }
    }
    (energy, grad)
}

fn start_chain(natoms: usize) -> Array1<f64> {
    let mut x = Array1::zeros(natoms * 3);
    for i in 0..natoms {
        let t = i as f64;
        x[3 * i] = t * 3.15;
        x[3 * i + 1] = 0.20 * (t * 0.37).sin();
        x[3 * i + 2] = 0.12 * (t * 0.21).cos();
    }
    x
}

fn rosen_fg(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (a, b) = (1.0, 100.0);
    let f = (a - x[0]).powi(2) + b * (x[1] - x[0] * x[0]).powi(2);
    let g = array![
        -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0] * x[0]),
        2.0 * b * (x[1] - x[0] * x[0])
    ];
    (f, g)
}

fn rosen_hess(x: ArrayView1<f64>) -> Array2<f64> {
    let b = 100.0;
    let h11 = 2.0 - 4.0 * b * (x[1] - 3.0 * x[0] * x[0]);
    let h12 = -4.0 * b * x[0];
    let h22 = 2.0 * b;
    array![[h11, h12], [h12, h22]]
}

fn rosen_hvp(x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
    let h = rosen_hess(x);
    h.dot(&v)
}

fn rayleigh(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut g = Array1::zeros(n);
    let mut f = 0.0;
    for i in 0..n {
        let d = 1.0 + i as f64;
        f += 0.5 * d * x[i] * x[i];
        g[i] = d * x[i];
        if i + 1 < n {
            f += 0.25 * x[i] * x[i + 1];
            g[i] += 0.25 * x[i + 1];
            g[i + 1] += 0.25 * x[i];
        }
    }
    (f, g)
}

fn rayleigh_hvp(v: ArrayView1<f64>) -> Array1<f64> {
    rayleigh(v).1
}

fn rayleigh_start(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| 0.3 + 0.17 * ((i * 5 % 7) as f64))
}

struct Dump {
    out: String,
}

impl Dump {
    fn point_with_meta(
        &mut self,
        case: &str,
        step: usize,
        value: f64,
        grad_norm: f64,
        x: &Array1<f64>,
        meta: Option<&str>,
    ) {
        let mut line = format!(
            "{case} {step} {:016x} {:016x}",
            value.to_bits(),
            grad_norm.to_bits()
        );
        for c in x.iter() {
            let _ = write!(line, " {:016x}", c.to_bits());
        }
        if let Some(meta) = meta {
            let _ = write!(line, " {meta}");
        }
        self.out.push_str(&line);
        self.out.push('\n');
    }

    fn point(&mut self, case: &str, step: usize, value: f64, grad_norm: f64, x: &Array1<f64>) {
        self.point_with_meta(case, step, value, grad_norm, x, None);
    }

    fn report(&mut self, case: &str, step: usize, r: &Report) {
        let grad = r
            .grad
            .as_ref()
            .map_or_else(|| "none".to_string(), |g| format!("some:{}", g.len()));
        let meta = format!(
            "report_steps={} stop={:?} status={:?} grad={}",
            r.steps, r.stop_norm, r.status, grad
        );
        self.point_with_meta(case, step, r.value, r.grad_norm, &r.coords, Some(&meta));
    }

    fn note(&mut self, case: &str, text: &str) {
        let _ = writeln!(self.out, "{case} # {text}");
    }
}

const SESSION_METHODS: &[&str] = &[
    "sd", "cg", "prp", "eon_cg", "fr", "hs", "dy", "cd", "hz", "ls", "fr_pr", "bfgs", "lbfgs",
    "sr1", "sr2", "adam", "bb", "fire", "fire2", "quickmin", "ode12r", "diis",
];

struct Caps {
    tag: &'static str,
    istep: f64,
    maxmove: Option<f64>,
    atom: Option<f64>,
}

const CAPS: &[Caps] = &[
    Caps {
        tag: "plain",
        istep: 0.1,
        maxmove: None,
        atom: None,
    },
    Caps {
        tag: "atom",
        istep: 1.0,
        maxmove: None,
        atom: Some(0.05),
    },
    Caps {
        tag: "euclid",
        istep: 1.0,
        maxmove: Some(0.1),
        atom: None,
    },
    Caps {
        tag: "both",
        istep: 1.0,
        maxmove: Some(0.3),
        atom: Some(0.04),
    },
];

fn control(steps: usize, istep: f64, maxmove: Option<f64>) -> Control {
    Control::default()
        .with_maxiter(steps)
        .with_gtol(0.0)
        .with_istep(istep)
        .with_maxmove(maxmove)
}

fn wolfe() -> LineSearch {
    LineSearch::wolfe(1e-4, 0.9, 20)
}

fn session_cases(d: &mut Dump) {
    const N: usize = 12;
    const STEPS: usize = 14;
    let obj = Oracle::unbounded(N * 3, morse_chain);
    for name in SESSION_METHODS {
        for caps in CAPS {
            for accept in [Accept::None, Accept::Energy, Accept::Step] {
                let case = format!("session/{name}/{}/{accept:?}", caps.tag);
                let method = Method::from_name(name).unwrap();
                let mut solver =
                    Solver::new(method, control(STEPS, caps.istep, caps.maxmove), N * 3);
                solver.set_accept(accept);
                if let Some(atom) = caps.atom {
                    solver.set_atom_maxmove(atom);
                }
                solver.set_linesearch(wolfe());
                let mut x = start_chain(N);
                for step in 0..STEPS {
                    match solver.step(&obj, &mut x) {
                        Ok(r) => d.report(&case, step, &r),
                        Err(e) => {
                            d.note(&case, &format!("error at {step}: {e}"));
                            break;
                        }
                    }
                }
            }
        }
    }
}

fn lbfgs_step_cases(d: &mut Dump) {
    const N: usize = 12;
    const STEPS: usize = 14;
    let obj = Oracle::unbounded(N * 3, morse_chain);
    let kinds = [
        ("default", None),
        ("capped", Some(LbfgsStep::capped(0.1))),
        ("scaled", Some(LbfgsStep::scaled(0.1))),
        ("wolfe", Some(LbfgsStep::Wolfe)),
    ];
    for (tag, step_kind) in kinds {
        for caps in CAPS {
            let case = format!("lbfgs-step/{tag}/{}", caps.tag);
            let mut solver = Solver::new(
                Method::lbfgs(),
                control(STEPS, caps.istep, caps.maxmove),
                N * 3,
            );
            if let Some(k) = step_kind {
                solver.set_lbfgs_step(k).unwrap();
            }
            if let Some(atom) = caps.atom {
                solver.set_atom_maxmove(atom);
            }
            let mut x = start_chain(N);
            for step in 0..STEPS {
                match solver.step(&obj, &mut x) {
                    Ok(r) => d.report(&case, step, &r),
                    Err(e) => {
                        d.note(&case, &format!("error at {step}: {e}"));
                        break;
                    }
                }
            }
        }
    }
}

fn oneshot_cases(d: &mut Dump) {
    const N: usize = 12;
    let obj = Oracle::unbounded(N * 3, morse_chain);
    let mut methods: Vec<(String, Method)> = SESSION_METHODS
        .iter()
        .map(|n| ((*n).to_string(), Method::from_name(n).unwrap()))
        .collect();
    methods.push(("pso".into(), Method::pso()));
    methods.push(("fire-v1".into(), Method::fire(FireKind::V1)));
    methods.push(("fire-v2".into(), Method::fire(FireKind::V2)));
    for (name, method) in methods {
        for caps in CAPS {
            for (ls_tag, ls) in [("default", LineSearch::default()), ("wolfe", wolfe())] {
                for steps in [1usize, 9] {
                    let case = format!("oneshot/{name}/{}/{ls_tag}/{steps}", caps.tag);
                    let ctl = control(steps, caps.istep, caps.maxmove);
                    match minimize_method(&obj, start_chain(N), &ctl, method.clone(), ls) {
                        Ok(r) => d.report(&case, r.steps, &r),
                        Err(e) => d.note(&case, &format!("error: {e}")),
                    }
                }
            }
        }
    }
    for conj in [
        Conjugacy::FletcherReeves,
        Conjugacy::PolakRibiere,
        Conjugacy::PolakRibierePlus,
        Conjugacy::HestenesStiefel,
        Conjugacy::DaiYuan,
        Conjugacy::HagerZhang,
    ] {
        for restart in [Restart::Never, Restart::njws(), Restart::powell()] {
            let case = format!("nlcg/{conj:?}/{restart:?}");
            let ctl = control(20, 1.0, Some(0.2));
            match rgmin::minimize(&obj, start_chain(N), &ctl, conj.clone(), restart, wolfe()) {
                Ok(r) => d.report(&case, r.steps, &r),
                Err(e) => d.note(&case, &format!("error: {e}")),
            }
        }
    }
}

fn newton_cases(d: &mut Dump) {
    let hess = HessianOracle::unbounded(2, rosen_fg, rosen_hess);
    let hvp = HvpOracle::unbounded(2, rosen_fg, rosen_hvp);
    for (tag, maxmove) in [("none", None), ("euclid", Some(0.3)), ("tight", Some(0.05))] {
        for name in ["newton", "rfo", "dogleg"] {
            let method = Method::from_name(name).unwrap();
            for steps in [1usize, 4, 12] {
                let case = format!("hess/{name}/{tag}/{steps}");
                let ctl = control(steps, 1.0, maxmove);
                match minimize_method_hess(&hess, array![-1.2, 1.0], &ctl, method.clone(), wolfe())
                {
                    Ok(r) => d.report(&case, r.steps, &r),
                    Err(e) => d.note(&case, &format!("error: {e}")),
                }
            }
            let case = format!("hess-session/{name}/{tag}");
            let mut solver = Solver::new(method.clone(), control(12, 1.0, maxmove), 2);
            let mut x = array![-1.2, 1.0];
            for step in 0..8 {
                match solver.step_hess(&hess, &mut x) {
                    Ok(r) => d.report(&case, step, &r),
                    Err(e) => {
                        d.note(&case, &format!("error at {step}: {e}"));
                        break;
                    }
                }
            }
        }
        for name in ["newton_cg", "newton_pcg"] {
            let method = Method::from_name(name).unwrap();
            for steps in [1usize, 4, 12] {
                let case = format!("hvp/{name}/{tag}/{steps}");
                let ctl = control(steps, 1.0, maxmove);
                match minimize_method_hvp(&hvp, array![-1.2, 1.0], &ctl, method.clone(), wolfe()) {
                    Ok(r) => d.report(&case, r.steps, &r),
                    Err(e) => d.note(&case, &format!("error: {e}")),
                }
            }
            let case = format!("hvp-session/{name}/{tag}");
            let mut solver = Solver::new(method.clone(), control(12, 1.0, maxmove), 2);
            let mut x = array![-1.2, 1.0];
            for step in 0..8 {
                match solver.step_hvp(&hvp, &mut x) {
                    Ok(r) => d.report(&case, step, &r),
                    Err(e) => {
                        d.note(&case, &format!("error at {step}: {e}"));
                        break;
                    }
                }
            }
        }
    }
}

fn scg_and_eon_cases(d: &mut Dump) {
    const N: usize = 12;
    let obj = Oracle::unbounded(N * 3, morse_chain);
    for (tag, maxmove) in [("none", None), ("euclid", Some(0.2))] {
        for steps in [3usize, 12] {
            let case = format!("scg/{tag}/{steps}");
            let ctl = control(steps, 1.0, maxmove);
            match minimize_scg(
                &obj,
                start_chain(N),
                &ctl,
                &ScgParams::default(),
                Conjugacy::PolakRibiere,
                Restart::Never,
            ) {
                Ok(r) => d.report(&case, r.steps, &r),
                Err(e) => d.note(&case, &format!("error: {e}")),
            }
        }
    }
    for (tag, opts) in [
        ("default", EonCg::default()),
        ("cluster", EonCg::default().with_cluster(true)),
        (
            "atom-only",
            EonCg::default().with_atom_only(true).with_max_move(0.05),
        ),
        ("uncapped", EonCg::default().with_max_move(0.0)),
    ] {
        for iters in [2usize, 15] {
            let case = format!("eon-cg/{tag}/{iters}");
            let mut x = start_chain(N).to_vec();
            let opts = opts.with_max_iter(iters).with_gtol(0.0);
            let report = minimize_eon_cg(
                &mut x,
                |p, g| {
                    let (f, grad) = morse_chain(ArrayView1::from(p));
                    g.copy_from_slice(grad.as_slice().unwrap());
                    f
                },
                &opts,
            );
            let coords = Array1::from(x);
            d.point(&case, iters, report.energy, report.grad, &coords);
        }
    }
}

fn manifold_cases(d: &mut Dump) {
    let kinds: Vec<(&str, usize, ManifoldKind)> = vec![
        ("sphere", 6, ManifoldKind::Sphere),
        ("stiefel", 6, ManifoldKind::stiefel(3, 2)),
        ("grassmann", 6, ManifoldKind::GrassmannP { n: 3, p: 2 }),
        ("oblique", 6, ManifoldKind::oblique(3, 2)),
    ];
    for (tag, n, kind) in kinds {
        let obj = Oracle::unbounded(n, rayleigh);
        for name in [
            "sd", "cg", "bfgs", "lbfgs", "adam", "fire", "fire2", "bb", "sr1",
        ] {
            for caps in CAPS {
                for accept in [Accept::None, Accept::Energy, Accept::Step] {
                    let case = format!("manifold/{tag}/{name}/{}/{accept:?}", caps.tag);
                    let method = Method::from_name(name).unwrap();
                    let mut solver = Solver::new(method, control(10, caps.istep, caps.maxmove), n);
                    solver.set_manifold(kind.clone());
                    solver.set_accept(accept);
                    if let Some(atom) = caps.atom {
                        solver.set_atom_maxmove(atom);
                    }
                    let mut x = rayleigh_start(n);
                    for step in 0..10 {
                        match solver.step(&obj, &mut x) {
                            Ok(r) => d.report(&case, step, &r),
                            Err(e) => {
                                d.note(&case, &format!("error at {step}: {e}"));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn rtr_cases(d: &mut Dump) {
    const N: usize = 6;
    let manifold = ManifoldKind::Sphere;
    let mut start = rayleigh_start(N);
    start /= start.dot(&start).sqrt();
    let cases = [
        (
            "default",
            RtrOptions::default().with_grad_tol(0.0).with_max_iter(8),
        ),
        (
            "setters",
            RtrOptions::default()
                .with_radius_max(0.8)
                .with_grad_tol(1e9)
                .with_max_iter(3)
                .with_cg_max(4)
                .with_theta(0.7)
                .with_kappa(0.2)
                .with_rho_regularization(50.0),
        ),
    ];
    for (tag, opts) in cases {
        let configured = format!(
            "radius_max={:?} grad_tol={:016x} max_iter={} cg_max={:?} theta={:016x} \
             kappa={:016x} rho_regularization={:016x}",
            opts.radius_max.map(f64::to_bits),
            opts.grad_tol.to_bits(),
            opts.max_iter,
            opts.cg_max,
            opts.theta.to_bits(),
            opts.kappa.to_bits(),
            opts.rho_regularization.to_bits(),
        );
        let report = minimize_rtr(
            &manifold,
            start.clone(),
            |x| rayleigh(x.view()).0,
            |x| rayleigh(x.view()).1,
            |_x, v| rayleigh_hvp(v.view()),
            opts,
        );
        let case = format!("rtr/{tag}");
        let meta = format!(
            "{configured} accepted={} inner={}",
            report.accepted, report.inner_iterations,
        );
        d.point_with_meta(
            &case,
            report.iterations,
            report.value,
            report.grad_norm,
            &report.x,
            Some(&meta),
        );
    }
}

fn rigid_cases(d: &mut Dump) {
    const N: usize = 8;
    let obj = Oracle::unbounded(N * 3, morse_chain);
    let masses = Array1::from_shape_fn(N, |i| 1.0 + 0.5 * (i % 3) as f64);
    let mut frozen = vec![0u8; N * 3];
    frozen[..3].fill(1);
    for mode in [
        RigidMode::Off,
        RigidMode::Auto,
        RigidMode::Translation,
        RigidMode::TranslationRotation,
    ] {
        for variant in [
            "plain",
            "masses",
            "mass-metric",
            "mass-weighted",
            "frozen",
            "slab",
        ] {
            for name in ["lbfgs", "fire", "fire2", "sd", "cg", "bfgs"] {
                for caps in CAPS {
                    let case = format!("rigid/{mode:?}/{variant}/{name}/{}", caps.tag);
                    let method = Method::from_name(name).unwrap();
                    let mut solver =
                        Solver::new(method, control(10, caps.istep, caps.maxmove), N * 3);
                    solver.set_rigid_mode(mode);
                    solver.set_accept(Accept::Energy);
                    match variant {
                        "masses" => solver.set_masses(masses.clone()),
                        "mass-metric" => {
                            solver.set_masses(masses.clone());
                            solver.set_rigid_mass_metric(true);
                        }
                        "mass-weighted" => {
                            solver.set_masses(masses.clone());
                            solver.set_rigid_mass_weighted(true);
                        }
                        "frozen" => solver.set_frozen(&frozen),
                        "slab" => solver.set_pbc_axes(true, true, false),
                        _ => {}
                    }
                    if let Some(atom) = caps.atom {
                        solver.set_atom_maxmove(atom);
                    }
                    let mut x = start_chain(N);
                    for step in 0..10 {
                        match solver.step(&obj, &mut x) {
                            Ok(r) => d.report(&case, step, &r),
                            Err(e) => {
                                d.note(&case, &format!("error at {step}: {e}"));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn constrained_cases(d: &mut Dump) {
    const N: usize = 6;
    let obj = Oracle::unbounded(N * 3, morse_chain);
    let start = start_chain(N);
    for name in ["lbfgs", "fire", "fire2", "sd", "cg", "bfgs", "quickmin"] {
        for caps in CAPS {
            let case = format!("constrained/{name}/{}", caps.tag);
            let method = Method::from_name(name).unwrap();
            let mut solver = Solver::new(method, control(10, caps.istep, caps.maxmove), N * 3);
            let set = DistanceConstraints::at(vec![(0, 1), (2, 3)], start.view()).unwrap();
            solver.set_distance_constraints(set).unwrap();
            solver.set_accept(Accept::Energy);
            if let Some(atom) = caps.atom {
                solver.set_atom_maxmove(atom);
            }
            let mut x = start.clone();
            for step in 0..10 {
                match solver.step(&obj, &mut x) {
                    Ok(r) => d.report(&case, step, &r),
                    Err(e) => {
                        d.note(&case, &format!("error at {step}: {e}"));
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(feature = "capi")]
#[allow(unused_unsafe)]
mod c_waist {
    use std::ffi::CStr;
    use std::os::raw::c_void;

    use dlpk::sys::{DLDeviceType, DLManagedTensorVersioned};
    use rgmin::ffi::*;

    use super::Dump;

    const N: usize = 6;

    unsafe fn cpu_f64(t: *const DLManagedTensorVersioned) -> (*const f64, usize) {
        let dl = unsafe { &(*t).dl_tensor };
        let n = unsafe { *dl.shape as usize };
        let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
        (p, n)
    }

    unsafe extern "C" fn rosen_evalgrad(
        _user: *mut c_void,
        x: *const DLManagedTensorVersioned,
        value_out: *mut f64,
        g: *mut DLManagedTensorVersioned,
    ) -> rgmin_status_t {
        let (p, n) = unsafe { cpu_f64(x) };
        let (gp, _) = unsafe { cpu_f64(g as *const _) };
        let gp = gp as *mut f64;
        let mut f = 0.0;
        for i in 0..n {
            unsafe { *gp.add(i) = 0.0 };
        }
        for i in 0..n - 1 {
            let (a, b) = unsafe { (*p.add(i), *p.add(i + 1)) };
            let t = b - a * a;
            f += 100.0 * t * t + (1.0 - a) * (1.0 - a);
            unsafe {
                *gp.add(i) += -400.0 * a * t - 2.0 * (1.0 - a);
                *gp.add(i + 1) += 200.0 * t;
            }
        }
        unsafe { *value_out = f };
        rgmin_status_t::RGMIN_SUCCESS
    }

    fn last_error() -> String {
        let p = rgmin_last_error();
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }

    type Setup = Box<dyn Fn(*mut rgmin_solver_t) -> i32>;

    fn setups() -> Vec<(&'static str, Setup)> {
        let mut v: Vec<(&'static str, Setup)> = Vec::new();
        macro_rules! s {
            ($name:expr, |$s:ident| $body:expr) => {
                v.push(($name, Box::new(move |$s| unsafe { $body })));
            };
        }
        s!("none", |_s| 0);
        s!("forget", |s| {
            rgmin_solver_forget(s);
            0
        });
        s!("rebase", |s| {
            rgmin_solver_rebase(s);
            0
        });
        s!("maxmove", |s| {
            rgmin_solver_set_maxmove(s, 0.07);
            0
        });
        s!("maxmove-neg", |s| {
            rgmin_solver_set_maxmove(s, -1.0);
            0
        });
        s!("atom-maxmove", |s| {
            rgmin_solver_set_atom_maxmove(s, 0.03);
            0
        });
        s!("project-rigid-off", |s| {
            rgmin_solver_set_project_rigid(s, 0);
            0
        });
        s!("project-rigid-on", |s| {
            rgmin_solver_set_project_rigid(s, 3);
            0
        });
        s!("extra-updates", |s| {
            rgmin_solver_set_extra_updates(s, 2);
            0
        });
        s!("cautious", |s| {
            rgmin_solver_set_cautious(s, 1e-6, 1.0);
            0
        });
        s!("oblique", |s| {
            rgmin_solver_set_oblique(s, 3, 2);
            0
        });
        s!("stiefel", |s| {
            rgmin_solver_set_stiefel(s, 3, 2);
            0
        });
        s!("complex-circle", |s| {
            rgmin_solver_set_complex_circle(s, 3);
            0
        });
        s!("euclidean-complex", |s| {
            rgmin_solver_set_euclidean_complex(s, 3);
            0
        });
        s!("constant", |s| {
            rgmin_solver_set_constant(s, 6);
            0
        });
        s!("multinomial-ds", |s| {
            rgmin_solver_set_multinomial_ds(s, 2);
            0
        });
        s!("multinomial-sym", |s| {
            rgmin_solver_set_multinomial_sym(s, 2);
            0
        });
        s!("sphere-complex", |s| {
            rgmin_solver_set_sphere_complex(s, 3);
            0
        });
        s!("positive", |s| {
            rgmin_solver_set_positive(s, 6);
            0
        });
        s!("centered-matrix-cols", |s| {
            rgmin_solver_set_centered_matrix(s, 2, 3, 0);
            0
        });
        s!("centered-matrix-rows", |s| {
            rgmin_solver_set_centered_matrix(s, 2, 3, 1);
            0
        });
        s!("periodic", |s| {
            rgmin_solver_set_periodic(s, 1);
            0
        });
        s!("geometry-step", |s| {
            rgmin_solver_set_geometry_step(s, 1);
            0
        });
        s!("rigid-mass-metric", |s| {
            rgmin_solver_set_rigid_mass_metric(s, 1);
            0
        });
        s!("rigid-mass-weighted", |s| {
            rgmin_solver_set_rigid_mass_weighted(s, 1);
            0
        });
        s!("pbc-axes", |s| {
            rgmin_solver_set_pbc_axes(s, 1, 1, 0);
            0
        });
        s!("factor-shape", |s| {
            rgmin_solver_set_factor_shape(s, 3, 2);
            0
        });
        s!("unitary", |s| {
            rgmin_solver_set_unitary(s, 2);
            0
        });
        s!("lbfgs-neb-guards", |s| {
            rgmin_solver_set_lbfgs_neb_guards(s, 1);
            0
        });
        s!("forget-evaluation", |s| {
            rgmin_solver_forget_evaluation(s);
            0
        });
        s!("clear-exp", |s| {
            rgmin_solver_clear_exp(s);
            0
        });
        s!("exp-invalid", |s| {
            let mut p = rgmin_exp_default();
            p.c_stab = 0.0;
            rgmin_solver_set_exp(
                s,
                &p,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
            ) as i32
        });
        s!("exp-valid", |s| {
            let p = rgmin_exp_default();
            let code = rgmin_solver_set_exp(
                s,
                &p,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
            ) as i32;
            rgmin_solver_clear_exp(s);
            code
        });
        s!("clear-precon", |s| {
            rgmin_solver_clear_precon(s);
            0
        });
        s!("clear-distance-constraints", |s| {
            rgmin_solver_clear_distance_constraints(s);
            0
        });
        s!("stop-norm-max", |s| rgmin_solver_set_stop_norm(s, 1));
        s!("stop-norm-bad", |s| rgmin_solver_set_stop_norm(s, 9));
        s!("fire-preset", |s| rgmin_solver_set_fire_preset(s, 1));
        s!("fire-preset-bad", |s| rgmin_solver_set_fire_preset(s, 7));
        s!("fire-abc", |s| rgmin_solver_set_fire_abc(s, 1));
        s!("lbfgs-capped", |s| rgmin_solver_set_lbfgs_capped(s, 0.05));
        s!("lbfgs-capped-default", |s| rgmin_solver_set_lbfgs_capped(
            s, 0.0
        ));
        s!("lbfgs-wolfe", |s| rgmin_solver_set_lbfgs_wolfe(s));
        s!("lbfgs-trust", |s| rgmin_solver_set_lbfgs_trust(s, 0.2));
        v
    }

    pub fn run(d: &mut Dump) {
        let methods = [
            ("sd", rgmin_method_t::RGMIN_STEEPEST),
            ("pr", rgmin_method_t::RGMIN_POLAK_RIBIERE),
            ("bfgs", rgmin_method_t::RGMIN_BFGS),
            ("lbfgs", rgmin_method_t::RGMIN_LBFGS),
            ("sr1", rgmin_method_t::RGMIN_SR1),
            ("adam", rgmin_method_t::RGMIN_ADAM),
            ("bb", rgmin_method_t::RGMIN_BB),
            ("fire", rgmin_method_t::RGMIN_FIRE),
            ("fire2", rgmin_method_t::RGMIN_FIRE2),
            ("quickmin", rgmin_method_t::RGMIN_QUICKMIN),
            ("eon-cg", rgmin_method_t::RGMIN_EON_CG),
        ];
        let null_probe = setups();
        for (name, setup) in &null_probe {
            let code = setup(std::ptr::null_mut());
            d.note(
                &format!("capi/null/{name}"),
                &format!("{code} [{}]", last_error()),
            );
        }
        for (mname, method) in methods {
            for (sname, setup) in setups() {
                let case = format!("capi/{mname}/{sname}");
                let ctrl = rgmin_control_t {
                    maxiter: 1,
                    gtol: 0.0,
                    istep: 0.1,
                    memory: 6,
                    maxmove: 0.0,
                    noise_eps: 0.0,
                    noise_c: 0.0,

                    require_convergence: 0,
                    noise_eps_f: 0.0,
                };
                let solver = unsafe { rgmin_solver_create(method, &ctrl, N) };
                assert!(!solver.is_null());
                let code = setup(solver);
                d.note(&case, &format!("setup {code} [{}]", last_error()));
                let mut x = [-1.2, 1.0, -0.8, 0.9, 1.1, -0.4];
                let xt = unsafe { rgmin_tensor_borrow_cpu_f64(x.as_mut_ptr(), N) };
                for step in 0..6 {
                    let mut out = rgmin_report_t {
                        value: 0.0,
                        steps: 0,
                        grad_norm: 0.0,
                        stop: rgmin_stop_t::RGMIN_STOP_TOLERANCE,

                        converged: 0,
                    };
                    let st = unsafe {
                        rgmin_solver_step_fg(
                            solver,
                            Some(rosen_evalgrad),
                            std::ptr::null_mut(),
                            xt,
                            &mut out,
                        )
                    };
                    if st != rgmin_status_t::RGMIN_SUCCESS {
                        d.note(
                            &case,
                            &format!("step {step} status {} [{}]", st as i32, last_error()),
                        );
                        break;
                    }
                    d.point(
                        &case,
                        step,
                        out.value,
                        out.grad_norm,
                        &ndarray::Array1::from(x.to_vec()),
                    );
                }
                unsafe {
                    rgmin_solver_free(solver);
                    rgmin_tensor_free(xt);
                }
            }
        }
        let _ = DLDeviceType::kDLCPU;
    }
}

fn main() {
    let mut d = Dump { out: String::new() };
    session_cases(&mut d);
    lbfgs_step_cases(&mut d);
    oneshot_cases(&mut d);
    newton_cases(&mut d);
    scg_and_eon_cases(&mut d);
    manifold_cases(&mut d);
    rtr_cases(&mut d);
    rigid_cases(&mut d);
    constrained_cases(&mut d);
    #[cfg(feature = "capi")]
    c_waist::run(&mut d);
    print!("{}", d.out);
}
