//! With `etol` and `max_evals` left at `None` every driver takes the steps
//! develop took: step count, status and the bits of the final energy,
//! gradient norm and coordinates match the values recorded before the stops
//! existed.

mod common;

use std::sync::OnceLock;

use eindir_core::objectives::Rosenbrock;
use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{
    Conjugacy, Control, FireKind, LineSearch, Method, Report, Restart, ScgParams, minimize_method,
    minimize_pso, minimize_scg,
};

use common::{MORSE_R0, morse_value_grad};

struct MorseTrimer;

impl Objective<f64> for MorseTrimer {
    fn dim(&self) -> usize {
        9
    }

    fn bounds(&self) -> &Bounds<f64> {
        static BOX: OnceLock<Bounds<f64>> = OnceLock::new();
        BOX.get_or_init(|| Bounds::new(Array1::from_elem(9, -1e3), Array1::from_elem(9, 1e3), 0.0))
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for MorseTrimer {
    fn dim(&self) -> usize {
        9
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for MorseTrimer {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        morse_value_grad(x, [100.0, 100.0, 100.0])
    }
}

fn morse_start() -> Array1<f64> {
    let r = MORSE_R0 * 1.12;
    array![
        0.0,
        0.0,
        0.0,
        r,
        0.0,
        0.0,
        0.5 * r + 0.2,
        0.866 * r - 0.1,
        0.05
    ]
}

fn control() -> Control {
    Control::default()
        .with_maxiter(300)
        .with_gtol(1e-8)
        .with_istep(0.05)
}

fn fingerprint(name: &str, report: &Report) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for c in report.coords.iter() {
        for byte in c.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!(
        "{name} steps={} status={:?} value={:016x} grad={:016x} coords={hash:016x}",
        report.steps,
        report.status,
        report.value.to_bits(),
        report.grad_norm.to_bits(),
    )
}

fn observed() -> Vec<String> {
    let methods = [
        ("lbfgs", Method::lbfgs()),
        ("pr", Method::polak_ribiere()),
        ("sd", Method::Steepest),
        ("bfgs", Method::Bfgs),
        ("sr1", Method::Sr1),
        ("sr2", Method::Sr2),
        ("adam", Method::adam()),
        ("fire", Method::fire(FireKind::V1)),
        ("bb", Method::Bb),
        ("quickmin", Method::QuickMin),
        ("ode12r", Method::Ode12r),
        ("diis", Method::diis()),
        ("newton-cg", Method::NewtonCg),
    ];
    let mut rows = Vec::new();
    for (name, method) in methods {
        let report = minimize_method(
            &MorseTrimer,
            morse_start(),
            &control(),
            method.clone(),
            LineSearch::default(),
        )
        .unwrap();
        rows.push(fingerprint(&format!("morse/{name}"), &report));
        let report = minimize_method(
            &Rosenbrock::<2>::new(),
            array![-1.2, 1.0],
            &control().with_maxiter(120),
            method,
            LineSearch::default(),
        )
        .unwrap();
        rows.push(fingerprint(&format!("rosenbrock/{name}"), &report));
    }
    let report = minimize_pso(
        &Rosenbrock::<2>::new(),
        array![-1.2, 1.0],
        &control().with_maxiter(60),
        10,
        0.7,
        1.5,
        1.5,
    )
    .unwrap();
    rows.push(fingerprint("rosenbrock/pso", &report));
    let report = minimize_scg(
        &MorseTrimer,
        morse_start(),
        &control(),
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .unwrap();
    rows.push(fingerprint("morse/scg", &report));
    rows
}

const GOLDEN: &[&str] = &[
    "morse/lbfgs steps=7 status=Tolerance value=c000ff6a3210c7a6 grad=3dfa6e4a52001777 coords=6a8c1841aee2e43f",
    "rosenbrock/lbfgs steps=27 status=Tolerance value=3b22d0a1d0000000 grad=3ddfd1c1cf921387 coords=140bf99c08b6c4a5",
    "morse/pr steps=300 status=Budget value=c000ff6a3210c7a6 grad=3e5c0e56c87bdb53 coords=affd5abda9f0b255",
    "rosenbrock/pr steps=37 status=Tolerance value=3a99db3400000000 grad=3d9b09b5ffc09cba coords=eb01478ae710c2a4",
    "morse/sd steps=300 status=Budget value=c000ff6a3210c7a6 grad=3e54271f7fe84c07 coords=42c75a34b1e9d91e",
    "rosenbrock/sd steps=48 status=Tolerance value=3c2c6d0ed7c19000 grad=3e0af471fa6ba76a coords=a7fa18cdfe24ae77",
    "morse/bfgs steps=300 status=Budget value=c000ff6a3210c7a6 grad=3e582acc505626b1 coords=108f554976f2473a",
    "rosenbrock/bfgs steps=22 status=Tolerance value=3bd7a4a25e9a4000 grad=3e03d570a90b2abe coords=af7fbb3b6fb566bf",
    "morse/sr1 steps=23 status=Tolerance value=c000ff6a3210c7a6 grad=3e2febbc2ab91dba coords=b698b713b6533563",
    "rosenbrock/sr1 steps=86 status=Tolerance value=3bca3f45a3ae8000 grad=3e3230854353b118 coords=fad9604ec70b3eb3",
    "morse/sr2 steps=15 status=Tolerance value=c000ff6a3210c7a6 grad=3e2c915aa6be6f39 coords=cb9d956d8dc69a4b",
    "rosenbrock/sr2 steps=36 status=Tolerance value=3be065f1b0788000 grad=3e3b928c19b4fdf6 coords=656e2e11db4e0c71",
    "morse/adam steps=300 status=Budget value=c000ff6a3210c783 grad=3e9bdd05e0e91b58 coords=5612ea6f7d306439",
    "rosenbrock/adam steps=120 status=Budget value=3e7a6cddfab4474f grad=3f32e1b6c7001879 coords=62dfbf3f727d481b",
    "morse/fire steps=152 status=Tolerance value=c000ff6a3210c7a6 grad=3e3e7933aa99f073 coords=87534b056cf7b5f8",
    "rosenbrock/fire steps=120 status=Budget value=3fc3a3e83acee377 grad=3fe1061c66a06a9e coords=3658fc60a1634ed3",
    "morse/bb steps=11 status=Tolerance value=c000ff6a3210c7a6 grad=3e3dca7ce399bd97 coords=311ed040db7cc547",
    "rosenbrock/bb steps=120 status=Budget value=3d94212a06f4e46a grad=3f188ec79f13cd7f coords=6d3854795c5e0056",
    "morse/quickmin steps=90 status=Tolerance value=c000ff6a3210c7a6 grad=3e0cf6115665b5f2 coords=4d442c5143e3f0f1",
    "rosenbrock/quickmin steps=120 status=Budget value=40068b124c5f37f4 grad=3fffe25abcb6dad3 coords=4b4a184bda861de8",
    "morse/ode12r steps=10 status=Tolerance value=c000ff6a3210c7a6 grad=3e2fbacbee924d7f coords=75a74a1eb71b4ea7",
    "rosenbrock/ode12r steps=120 status=Budget value=3fac1e3189f228e3 grad=3fdb11bdea77a5c9 coords=8e0105cdddadade7",
    "morse/diis steps=21 status=Tolerance value=c000ff6a3210c7a6 grad=3e41d0003f84f7be coords=97704a13b6f369d6",
    "rosenbrock/diis steps=120 status=Budget value=3ed4d63044c3b4af grad=3f620ae705522f60 coords=d93b46d175e99264",
    "morse/newton-cg steps=8 status=Tolerance value=c000ff6a3210c7a6 grad=3cf4838b35a8ae65 coords=a8db5ce8d7503b0b",
    "rosenbrock/newton-cg steps=35 status=Tolerance value=3a885d0a00000000 grad=3d935d981c3d500e coords=450caff9540bd04d",
    "rosenbrock/pso steps=60 status=Budget value=3f5ce58875dd0e04 grad=3fb08df6f2c33387 coords=bb9028d813d76aeb",
    "morse/scg steps=10 status=StepTolerance value=c000ff6a3210c6e4 grad=3eb2e6ec4a05c8de coords=e45d85bf79340566",
];

#[test]
fn unset_stops_reproduce_the_recorded_runs() {
    let rows = observed();
    if std::env::var_os("RGMIN_PRINT_GOLDEN").is_some() {
        for row in &rows {
            println!("{row}");
        }
    }
    assert_eq!(rows.len(), GOLDEN.len(), "regenerate GOLDEN");
    for (got, want) in rows.iter().zip(GOLDEN) {
        assert_eq!(got, want);
    }
}
