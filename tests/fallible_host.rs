//! Fallible host callbacks: `Solver::try_step_hvp` and
//! `try_lowest_eigenpairs`.
//!
//! On callbacks that succeed, each fallible entry point takes the steps
//! of its infallible twin bit for bit. The first callback error ends the
//! call as `Error::Host` carrying that error, nothing is evaluated after
//! it, and `x` and the session stay as they were before the step.

use std::cell::Cell;
use std::fmt;
use std::sync::OnceLock;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgmin::manifold::{ManifoldKind, ProductFactor};
use rgmin::{
    ApplyBlockPreconditioner, ApplyHessian, Control, DistanceConstraints, Error, HessianVector,
    LbfgsStep, LevelSetConstraint, LineSearch, Method, OptimizerFamily, Report, Solver, Status,
    StopNorm, TryApplyHessian, TryHessianVector, lowest_eigenpairs, lowest_eigenpairs_precond,
    try_lowest_eigenpairs, try_lowest_eigenpairs_precond,
};

const DIM: usize = 4;

/// Two decoupled Rosenbrock valleys: nonconvex away from the minimum,
/// so the Steihaug path meets negative curvature and the trust
/// boundary.
struct Valleys;

fn bounds() -> &'static Bounds<f64> {
    static B: OnceLock<Bounds<f64>> = OnceLock::new();
    B.get_or_init(|| {
        Bounds::new(
            Array1::from_elem(DIM, -1e6),
            Array1::from_elem(DIM, 1e6),
            0.0,
        )
    })
}

impl Objective<f64> for Valleys {
    fn dim(&self) -> usize {
        DIM
    }

    fn bounds(&self) -> &Bounds<f64> {
        bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (0..DIM / 2)
            .map(|k| {
                let (a, b) = (x[2 * k], x[2 * k + 1]);
                (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
            })
            .sum()
    }
}

impl Gradient<f64> for Valleys {
    fn dim(&self) -> usize {
        DIM
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let mut g = Array1::zeros(DIM);
        for k in 0..DIM / 2 {
            let (a, b) = (x[2 * k], x[2 * k + 1]);
            g[2 * k] = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
            g[2 * k + 1] = 200.0 * (b - a * a);
        }
        g
    }
}

impl DifferentiableObjective<f64> for Valleys {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

impl HessianVector for Valleys {
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let mut hv = Array1::zeros(DIM);
        for k in 0..DIM / 2 {
            let (a, b) = (x[2 * k], x[2 * k + 1]);
            let haa = 2.0 - 400.0 * (b - a * a) + 800.0 * a * a;
            let hab = -400.0 * a;
            hv[2 * k] = haa * v[2 * k] + hab * v[2 * k + 1];
            hv[2 * k + 1] = hab * v[2 * k] + 200.0 * v[2 * k + 1];
        }
        hv
    }
}

#[derive(Debug, PartialEq)]
struct HostFailure(usize);

impl fmt::Display for HostFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "host call {} failed", self.0)
    }
}

impl std::error::Error for HostFailure {}

/// [`Valleys`] behind fallible callbacks that fail on call `fail_at`.
struct Host {
    calls: Cell<usize>,
    fail_at: Option<usize>,
    bounds: &'static Bounds<f64>,
    /// Coordinates of the first callback, when one has run.
    first: Cell<Option<[f64; DIM]>>,
}

impl Host {
    fn new(fail_at: Option<usize>) -> Self {
        Self::boxed(fail_at, bounds())
    }

    fn boxed(fail_at: Option<usize>, bounds: &'static Bounds<f64>) -> Self {
        Self {
            calls: Cell::new(0),
            fail_at,
            bounds,
            first: Cell::new(None),
        }
    }

    fn call(&self, x: ArrayView1<f64>) -> Result<(), HostFailure> {
        let n = self.calls.get() + 1;
        self.calls.set(n);
        if n == 1 {
            let mut first = [0.0; DIM];
            for (slot, value) in first.iter_mut().zip(x.iter()) {
                *slot = *value;
            }
            self.first.set(Some(first));
        }
        match self.fail_at {
            Some(k) if k == n => Err(HostFailure(n)),
            _ => Ok(()),
        }
    }
}

impl TryHessianVector for Host {
    type Error = HostFailure;

    fn dim(&self) -> usize {
        DIM
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.bounds
    }

    fn try_value_and_gradient(
        &self,
        x: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), HostFailure> {
        self.call(x)?;
        Ok(Valleys.value_and_gradient(x))
    }

    fn try_hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Array1<f64>, HostFailure> {
        self.call(x)?;
        Ok(Valleys.hessian_vector(x, v))
    }
}

impl TryApplyHessian for Host {
    type Error = HostFailure;

    fn try_apply_hessian(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Array1<f64>, HostFailure> {
        self.try_hessian_vector(x, v)
    }
}

fn start() -> Array1<f64> {
    array![-1.2, 1.0, 0.3, -0.7]
}

fn host_failure(err: &Error) -> &HostFailure {
    let Error::Host(inner) = err else {
        panic!("expected a host error, got {err:?}");
    };
    inner
        .downcast_ref::<HostFailure>()
        .expect("the host's own error")
}

fn methods() -> [Method; 2] {
    [Method::NewtonCg, Method::NewtonPcg]
}
const STEPS: usize = 40;

#[test]
fn try_step_hvp_takes_the_steps_of_step_hvp() {
    for method in methods() {
        let mut plain = Solver::new(method.clone(), Control::default(), DIM);
        let mut fallible = Solver::new(method.clone(), Control::default(), DIM);
        let host = Host::new(None);
        let mut x_plain = start();
        let mut x_fallible = start();
        for _ in 0..STEPS {
            let a = plain.step_hvp(&Valleys, &mut x_plain);
            let b = fallible.try_step_hvp(&host, &mut x_fallible);
            match (a, b) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.value.to_bits(), b.value.to_bits());
                    assert_eq!(a.grad_norm.to_bits(), b.grad_norm.to_bits());
                    assert_eq!(a.status, b.status);
                }
                (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                (a, b) => panic!("{method:?}: {a:?} against {b:?}"),
            }
            let bits = |x: &Array1<f64>| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&x_plain), bits(&x_fallible));
            assert_eq!(plain.hvp_calls(), fallible.hvp_calls());
            assert_eq!(plain.gradient_calls(), fallible.gradient_calls());
        }
        assert_eq!(
            host.calls.get(),
            fallible.hvp_calls() + fallible.gradient_calls()
        );
    }
}

#[test]
fn a_host_error_ends_the_step_where_it_happened() {
    for method in methods() {
        // The first step opens with a gradient and then spends products
        // and one trial gradient; failing on each of them in turn covers
        // the opening gradient, the sketch, CG and the trial.
        let mut reference = Solver::new(method.clone(), Control::default(), DIM);
        let counter = Host::new(None);
        let mut x_ref = start();
        let ref_report = reference
            .try_step_hvp(&counter, &mut x_ref)
            .expect("the reference step");
        let per_step = counter.calls.get();
        assert!(per_step >= 3);
        for fail_at in 1..=per_step {
            let mut solver = Solver::new(method.clone(), Control::default(), DIM);
            let host = Host::new(Some(fail_at));
            let mut x = start();
            let before = SessionState::of(&solver, &x);
            let err = solver
                .try_step_hvp(&host, &mut x)
                .expect_err("the host failed");
            assert_eq!(host_failure(&err), &HostFailure(fail_at));
            assert_eq!(host.calls.get(), fail_at, "no call after the failure");
            assert_eq!(SessionState::of(&solver, &x), before);
            assert_eq!(err.to_string(), format!("host: host call {fail_at} failed"));

            let retry_host = Host::new(None);
            let retry = solver
                .try_step_hvp(&retry_host, &mut x)
                .expect("retry after host failure");
            assert_eq!(ReportBits::of(&retry), ReportBits::of(&ref_report));
            assert_eq!(
                SessionState::of(&solver, &x),
                SessionState::of(&reference, &x_ref)
            );
        }
    }
}

#[test]
fn try_step_hvp_refuses_a_method_without_products() {
    let mut solver = Solver::new(Method::Bb, Control::default(), DIM);
    let host = Host::new(None);
    let mut x = start();
    let err = solver.try_step_hvp(&host, &mut x).expect_err("no products");
    assert!(matches!(err, Error::NeedHvp), "{err:?}");
    assert_eq!(host.calls.get(), 0);
}

fn eigen_seed() -> Array2<f64> {
    let mut seed = Array2::zeros((DIM, 2));
    for i in 0..DIM {
        seed[(i, 0)] = 1.0 + i as f64;
        seed[(i, 1)] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    seed
}

#[test]
fn try_lowest_eigenpairs_matches_lowest_eigenpairs() {
    let x = start();
    let seed = eigen_seed();
    let tol = 1e-10;
    let plain = lowest_eigenpairs(&Valleys, x.view(), seed.view(), tol, 50).expect("plain");
    let host = Host::new(None);
    let fallible = try_lowest_eigenpairs(&host, x.view(), seed.view(), tol, 50).expect("fallible");
    assert_eq!(plain.actions, fallible.actions);
    assert_eq!(plain.iterations, fallible.iterations);
    assert_eq!(host.calls.get(), fallible.actions);
    for (a, b) in plain.pairs.iter().zip(&fallible.pairs) {
        assert_eq!(a.value.to_bits(), b.value.to_bits());
        assert_eq!(a.residual.to_bits(), b.residual.to_bits());
        assert_eq!(a.converged, b.converged);
        assert!(
            a.vector
                .iter()
                .zip(&b.vector)
                .all(|(u, v)| u.to_bits() == v.to_bits())
        );
    }
}

#[test]
fn try_lowest_eigenpairs_stops_on_the_first_failed_product() {
    let x = start();
    let seed = eigen_seed();
    for fail_at in [1, 2, 5] {
        let host = Host::new(Some(fail_at));
        let err = try_lowest_eigenpairs(&host, x.view(), seed.view(), 1e-10, 50)
            .expect_err("the host failed");
        assert_eq!(host_failure(&err), &HostFailure(fail_at));
        assert_eq!(host.calls.get(), fail_at);
    }
}

fn unit_box() -> &'static Bounds<f64> {
    static B: OnceLock<Bounds<f64>> = OnceLock::new();
    B.get_or_init(|| {
        Bounds::new(
            Array1::from_elem(DIM, -1.0),
            Array1::from_elem(DIM, 1.0),
            0.0,
        )
    })
}

fn bits(x: &Array1<f64>) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[derive(Debug, PartialEq)]
struct SessionState {
    x: Vec<u64>,
    dim: usize,
    steps: usize,
    gradient_calls: usize,
    hvp_calls: usize,
    stop_norm: StopNorm,
    linesearch: LineSearch,
    optimizer_family: OptimizerFamily,
    pair_count: usize,
    lbfgs_step: LbfgsStep,
    lbfgs_trust_radius: Option<u64>,
    nlcg_descent_fallbacks: u64,
    precon_scale: Option<u64>,
    exp_mu: Option<u64>,
}

impl SessionState {
    fn of(solver: &Solver, x: &Array1<f64>) -> Self {
        Self {
            x: bits(x),
            dim: solver.dim(),
            steps: solver.steps(),
            gradient_calls: solver.gradient_calls(),
            hvp_calls: solver.hvp_calls(),
            stop_norm: solver.stop_norm(),
            linesearch: solver.linesearch(),
            optimizer_family: solver.optimizer_family(),
            pair_count: solver.pair_count(),
            lbfgs_step: solver.lbfgs_step(),
            lbfgs_trust_radius: solver.lbfgs_trust_radius().map(f64::to_bits),
            nlcg_descent_fallbacks: solver.nlcg_descent_fallbacks(),
            precon_scale: solver.precon_scale().map(f64::to_bits),
            exp_mu: solver.exp_mu().map(f64::to_bits),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ReportBits {
    value: u64,
    coords: Vec<u64>,
    steps: usize,
    grad_norm: u64,
    stop_norm: StopNorm,
    grad: Option<Vec<u64>>,
    status: Status,
}

impl ReportBits {
    fn of(report: &Report) -> Self {
        Self {
            value: report.value.to_bits(),
            coords: bits(&report.coords),
            steps: report.steps,
            grad_norm: report.grad_norm.to_bits(),
            stop_norm: report.stop_norm,
            grad: report.grad.as_ref().map(bits),
            status: report.status,
        }
    }
}

/// A point outside the box is clipped on a scratch copy for the callbacks;
/// a host error leaves the caller's unclipped point and the session untouched,
/// and a retry matches a fresh step bit for bit.
#[test]
fn a_host_error_leaves_an_unclipped_start_unchanged() {
    let outside = array![-1.2, 1.4, 0.3, -1.7];
    let clipped = unit_box().clip(outside.view());
    assert_ne!(bits(&outside), bits(&clipped));
    for method in methods() {
        let mut fresh = Solver::new(method.clone(), Control::default(), DIM);
        fresh.set_highs(true);
        fresh.set_box(Some(vec![-1.0; DIM]), Some(vec![1.0; DIM]));
        let ref_host = Host::boxed(None, unit_box());
        let mut x_fresh = outside.clone();
        let ref_report = fresh
            .try_step_hvp(&ref_host, &mut x_fresh)
            .expect("fresh boxed step");
        let per_step = ref_host.calls.get();
        assert!(per_step >= 3);

        for fail_at in 1..=per_step {
            let mut solver = Solver::new(method.clone(), Control::default(), DIM);
            solver.set_highs(true);
            solver.set_box(Some(vec![-1.0; DIM]), Some(vec![1.0; DIM]));
            let host = Host::boxed(Some(fail_at), unit_box());
            let mut x = outside.clone();
            let before = SessionState::of(&solver, &x);
            let err = solver
                .try_step_hvp(&host, &mut x)
                .expect_err("the host failed");
            assert_eq!(host_failure(&err), &HostFailure(fail_at));
            assert_eq!(host.calls.get(), fail_at);
            assert_eq!(SessionState::of(&solver, &x), before);
            assert_eq!(bits(&x), bits(&outside));
            let seen = host.first.get().expect("the first callback ran");
            assert_eq!(bits(&Array1::from_vec(seen.to_vec())), bits(&clipped));

            let good = Host::boxed(None, unit_box());
            let retry = solver
                .try_step_hvp(&good, &mut x)
                .expect("retry after unclipped failure");
            assert_eq!(good.calls.get(), per_step);
            assert_eq!(ReportBits::of(&retry), ReportBits::of(&ref_report));
            assert_eq!(
                SessionState::of(&solver, &x),
                SessionState::of(&fresh, &x_fresh)
            );
        }
    }
}

/// A failed step changes nothing in the session (counters, cached evaluation,
/// bounds or trust radius), so the next successful step matches a session that
/// never saw the failure.
#[test]
fn a_failed_step_does_not_change_the_next_one() {
    for method in methods() {
        // Cold session: fail on each callback of step 1, then retry and walk two steps.
        let mut fresh = Solver::new(method.clone(), Control::default(), DIM);
        let mut x_fresh = start();
        let ref_host = Host::new(None);
        let first = fresh
            .try_step_hvp(&ref_host, &mut x_fresh)
            .expect("fresh step 1");
        let per_step = ref_host.calls.get();
        let next_ref = fresh
            .try_step_hvp(&Host::new(None), &mut x_fresh)
            .expect("fresh step 2");

        for fail_at in 1..=per_step {
            let mut marked = Solver::new(method.clone(), Control::default(), DIM);
            let mut x_marked = start();
            let before = SessionState::of(&marked, &x_marked);
            let failed = Host::new(Some(fail_at));
            let err = marked
                .try_step_hvp(&failed, &mut x_marked)
                .expect_err("the host failed");
            assert_eq!(host_failure(&err), &HostFailure(fail_at));
            assert_eq!(failed.calls.get(), fail_at);
            assert_eq!(SessionState::of(&marked, &x_marked), before);

            let again = Host::new(None);
            let retry = marked
                .try_step_hvp(&again, &mut x_marked)
                .expect("the retry");
            assert_eq!(again.calls.get(), per_step);
            assert_eq!(ReportBits::of(&retry), ReportBits::of(&first));

            let second = marked
                .try_step_hvp(&Host::new(None), &mut x_marked)
                .expect("step after retry");
            assert_eq!(ReportBits::of(&second), ReportBits::of(&next_ref));
            assert_eq!(
                SessionState::of(&marked, &x_marked),
                SessionState::of(&fresh, &x_fresh)
            );
        }

        // Warm session: step 1 already populated the cached evaluation, counters
        // and trust radius; failing on step 2 (both at the cached point and at an
        // unclipped point on another box) leaves all of them intact.
        let mut warm_ref = Solver::new(method.clone(), Control::default(), DIM);
        warm_ref.set_highs(true);
        warm_ref.set_box(Some(vec![-1000.0; DIM]), Some(vec![1000.0; DIM]));
        let mut x_warm_ref = start();
        warm_ref
            .try_step_hvp(&Host::new(None), &mut x_warm_ref)
            .expect("warm ref step 1");
        let warm_counter = Host::new(None);
        let warm_second = warm_ref
            .try_step_hvp(&warm_counter, &mut x_warm_ref)
            .expect("warm ref step 2");
        let warm_calls = warm_counter.calls.get();

        for fail_at in 1..=warm_calls {
            let mut marked = Solver::new(method.clone(), Control::default(), DIM);
            marked.set_highs(true);
            marked.set_box(Some(vec![-1000.0; DIM]), Some(vec![1000.0; DIM]));
            let mut x_marked = start();
            marked
                .try_step_hvp(&Host::new(None), &mut x_marked)
                .expect("warm marked step 1");

            // Fail at the cached point.
            let before_cached = SessionState::of(&marked, &x_marked);
            let bad_cached = Host::new(Some(fail_at));
            let err = marked
                .try_step_hvp(&bad_cached, &mut x_marked)
                .expect_err("cached warm failure");
            assert_eq!(host_failure(&err), &HostFailure(fail_at));
            assert_eq!(SessionState::of(&marked, &x_marked), before_cached);

            // Fail at an unclipped point under unit_box as well.
            let mut x_outside = array![-1.2, 1.4, 0.3, -1.7];
            let before_outside = SessionState::of(&marked, &x_outside);
            let bad_outside = Host::boxed(Some(fail_at), unit_box());
            let err = marked
                .try_step_hvp(&bad_outside, &mut x_outside)
                .expect_err("unclipped warm failure");
            assert_eq!(host_failure(&err), &HostFailure(fail_at));
            assert_eq!(SessionState::of(&marked, &x_outside), before_outside);

            // Retry from x_marked still hits the cached point and matches warm_ref.
            let retry_host = Host::new(None);
            let retry = marked
                .try_step_hvp(&retry_host, &mut x_marked)
                .expect("warm retry");
            assert_eq!(retry_host.calls.get(), warm_calls);
            assert_eq!(ReportBits::of(&retry), ReportBits::of(&warm_second));
            assert_eq!(
                SessionState::of(&marked, &x_marked),
                SessionState::of(&warm_ref, &x_warm_ref)
            );
        }
    }
}

#[test]
fn level_set_step_rolls_back_on_host_failure() {
    fn level_set_start() -> Array1<f64> {
        array![2.0, -1.0, 0.5, -0.25]
    }

    struct LevelSetHost {
        calls: Cell<usize>,
        fail_at: Option<usize>,
    }

    impl LevelSetHost {
        fn new(fail_at: Option<usize>) -> Self {
            Self {
                calls: Cell::new(0),
                fail_at,
            }
        }

        fn call(&self) -> Result<(), HostFailure> {
            let n = self.calls.get() + 1;
            self.calls.set(n);
            if self.fail_at == Some(n) {
                return Err(HostFailure(n));
            }
            Ok(())
        }
    }

    impl TryHessianVector for LevelSetHost {
        type Error = HostFailure;

        fn dim(&self) -> usize {
            DIM
        }

        fn bounds(&self) -> &Bounds<f64> {
            bounds()
        }

        fn try_value_and_gradient(
            &self,
            x: ArrayView1<f64>,
        ) -> Result<(f64, Array1<f64>), HostFailure> {
            self.call()?;
            Ok((0.5 * x.dot(&x), x.to_owned()))
        }

        fn try_hessian_vector(
            &self,
            _x: ArrayView1<f64>,
            v: ArrayView1<f64>,
        ) -> Result<Array1<f64>, HostFailure> {
            self.call()?;
            Ok(v.to_owned())
        }
    }

    for method in methods() {
        let mut reference = Solver::new(method.clone(), Control::default(), DIM);
        reference
            .set_level_set(LevelSetConstraint::new(1.0))
            .expect("a finite target");
        let mut x_ref = level_set_start();
        let counter = LevelSetHost::new(None);
        let ref_report = reference
            .try_step_hvp(&counter, &mut x_ref)
            .expect("the reference constrained step");
        let per_step = counter.calls.get();
        assert!(
            per_step >= 4,
            "{method:?}: expected callbacks in the constrained step"
        );

        for fail_at in 1..=per_step {
            let mut solver = Solver::new(method.clone(), Control::default(), DIM);
            solver
                .set_level_set(LevelSetConstraint::new(1.0))
                .expect("a finite target");
            let mut x = level_set_start();
            let before = SessionState::of(&solver, &x);
            let host = LevelSetHost::new(Some(fail_at));
            let err = solver
                .try_step_hvp(&host, &mut x)
                .expect_err("the host failed");
            assert_eq!(host_failure(&err), &HostFailure(fail_at));
            assert_eq!(host.calls.get(), fail_at, "no call after the failure");
            assert_eq!(SessionState::of(&solver, &x), before);
            assert_eq!(bits(&x), bits(&level_set_start()));

            let retry_host = LevelSetHost::new(None);
            let retry = solver
                .try_step_hvp(&retry_host, &mut x)
                .expect("retry after constrained host failure");
            assert_eq!(ReportBits::of(&retry), ReportBits::of(&ref_report));
            assert_eq!(
                SessionState::of(&solver, &x),
                SessionState::of(&reference, &x_ref)
            );
        }
    }
}

#[test]
fn try_step_hvp_refuses_a_chart_and_distance_constraints() {
    let mut charted = Solver::new(Method::NewtonCg, Control::default(), DIM);
    charted
        .set_product([
            ProductFactor::new(ManifoldKind::Euclidean, 2).with_weight(2.0),
            ProductFactor::new(ManifoldKind::Euclidean, 2).with_weight(0.5),
        ])
        .expect("weighted product");
    let host = Host::new(None);
    let mut x = start();
    let err = charted
        .try_step_hvp(&host, &mut x)
        .expect_err("a chart is unsupported");
    assert!(
        matches!(err, Error::FallibleUnsupported { what: "a chart" }),
        "{err:?}"
    );
    assert_eq!(host.calls.get(), 0);
    assert_eq!(bits(&x), bits(&start()));

    let mut constrained = Solver::new(Method::NewtonPcg, Control::default(), 6);
    let anchor = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let set = DistanceConstraints::at(vec![(0, 1)], anchor.view()).expect("one bond");
    constrained
        .set_distance_constraints(set)
        .expect("the set fits");
    let bond = BondHost::new();
    let mut y = anchor.clone();
    let err = constrained
        .try_step_hvp(&bond, &mut y)
        .expect_err("constraints are unsupported");
    assert!(
        matches!(
            err,
            Error::FallibleUnsupported {
                what: "distance constraints"
            }
        ),
        "{err:?}"
    );
    assert_eq!(bond.calls.get(), 0);
    assert_eq!(bits(&y), bits(&anchor));

    let mut other = Solver::new(Method::Bb, Control::default(), 6);
    other
        .set_distance_constraints(
            DistanceConstraints::at(vec![(0, 1)], anchor.view()).expect("one bond"),
        )
        .expect("the set fits");
    let bond = BondHost::new();
    let err = other
        .try_step_hvp(&bond, &mut y)
        .expect_err("bb has no product");
    assert!(matches!(err, Error::NeedHvp), "{err:?}");
    assert_eq!(bond.calls.get(), 0);
}

#[test]
fn try_step_hvp_rejects_a_short_vector_before_any_callback() {
    let mut solver = Solver::new(Method::NewtonCg, Control::default(), DIM);
    let host = Host::new(None);
    let mut x = array![0.0];
    let err = solver.try_step_hvp(&host, &mut x).expect_err("short x");
    assert!(matches!(err, Error::Dim { got: 1, dim: 4 }), "{err:?}");
    assert_eq!(host.calls.get(), 0);
    assert_eq!(x, array![0.0]);
}

/// Two atoms, so a distance constraint has a coordinate vector to fix.
struct BondHost {
    calls: Cell<usize>,
    bounds: Bounds<f64>,
}

impl BondHost {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
            bounds: Bounds::new(Array1::from_elem(6, -1e6), Array1::from_elem(6, 1e6), 0.0),
        }
    }
}

impl TryHessianVector for BondHost {
    type Error = HostFailure;

    fn dim(&self) -> usize {
        6
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn try_value_and_gradient(
        &self,
        x: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), HostFailure> {
        self.calls.set(self.calls.get() + 1);
        Ok((x.dot(&x), x.to_owned()))
    }

    fn try_hessian_vector(
        &self,
        _x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Array1<f64>, HostFailure> {
        self.calls.set(self.calls.get() + 1);
        Ok(v.to_owned())
    }
}

/// Diagonal with a steep spread. A Jacobi `T` cuts the block iteration.
struct SpreadDiag {
    diag: Array1<f64>,
}

impl SpreadDiag {
    fn new() -> Self {
        Self {
            diag: Array1::from_iter((0..8).map(|i| 4.0_f64.powi(i))),
        }
    }

    fn apply(&self, v: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(self.diag.iter().zip(v.iter()).map(|(d, vi)| d * vi))
    }
}

impl ApplyHessian for SpreadDiag {
    fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        self.apply(v)
    }
}

struct SpreadHost<'a> {
    op: &'a SpreadDiag,
    calls: Cell<usize>,
    fail_at: Option<usize>,
}

impl TryApplyHessian for SpreadHost<'_> {
    type Error = HostFailure;

    fn try_apply_hessian(
        &self,
        _x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Array1<f64>, HostFailure> {
        let n = self.calls.get() + 1;
        self.calls.set(n);
        if self.fail_at == Some(n) {
            return Err(HostFailure(n));
        }
        Ok(self.op.apply(v))
    }
}

struct Jacobi(Array1<f64>);

impl ApplyBlockPreconditioner for Jacobi {
    fn apply_block_preconditioner(&self, block: ArrayView2<f64>) -> Array2<f64> {
        let mut out = block.to_owned();
        for mut col in out.columns_mut() {
            for (value, scale) in col.iter_mut().zip(self.0.iter()) {
                *value *= *scale;
            }
        }
        out
    }
}

#[test]
fn try_lowest_eigenpairs_precond_matches_and_stops() {
    let op = SpreadDiag::new();
    let jacobi = Jacobi(op.diag.mapv(|d| 1.0 / d));
    let n = op.diag.len();
    let x = Array1::zeros(n);
    let mut seed = Array2::zeros((n, 2));
    for i in 0..n {
        seed[(i, 0)] = ((i * 3 + 1) as f64) * 0.17;
        seed[(i, 1)] = ((i * 5 + 2) as f64) * -0.11;
    }
    let tol = 1e-8;
    let plain = lowest_eigenpairs_precond(&op, Some(&jacobi), x.view(), seed.view(), tol, 40)
        .expect("plain");
    let bare = lowest_eigenpairs(&op, x.view(), seed.view(), tol, 40).expect("bare");
    assert!(
        plain.iterations < bare.iterations,
        "preconditioned {} iterations, identity {}",
        plain.iterations,
        bare.iterations
    );
    let host = SpreadHost {
        op: &op,
        calls: Cell::new(0),
        fail_at: None,
    };
    let fallible =
        try_lowest_eigenpairs_precond(&host, Some(&jacobi), x.view(), seed.view(), tol, 40)
            .expect("fallible");
    assert_eq!(plain.actions, fallible.actions);
    assert_eq!(plain.iterations, fallible.iterations);
    assert_eq!(host.calls.get(), fallible.actions);
    for (a, b) in plain.pairs.iter().zip(&fallible.pairs) {
        assert_eq!(a.value.to_bits(), b.value.to_bits());
        assert_eq!(a.residual.to_bits(), b.residual.to_bits());
        assert_eq!(a.converged, b.converged);
        assert!(
            a.vector
                .iter()
                .zip(&b.vector)
                .all(|(u, v)| u.to_bits() == v.to_bits())
        );
    }
    let host = SpreadHost {
        op: &op,
        calls: Cell::new(0),
        fail_at: Some(3),
    };
    let err = try_lowest_eigenpairs_precond(&host, Some(&jacobi), x.view(), seed.view(), tol, 40)
        .expect_err("the host failed");
    assert_eq!(host_failure(&err), &HostFailure(3));
    assert_eq!(host.calls.get(), 3);
}

#[test]
fn try_lowest_eigenpairs_validates_before_any_product() {
    let x = start();
    let mut seed = eigen_seed();
    seed[(0, 0)] = f64::NAN;
    let host = Host::new(None);
    let err = try_lowest_eigenpairs(&host, x.view(), seed.view(), 1e-10, 50)
        .expect_err("the seed is not finite");
    assert!(matches!(err, Error::EigenInput { .. }), "{err:?}");
    assert_eq!(host.calls.get(), 0);
}
