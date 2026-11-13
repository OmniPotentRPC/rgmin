//! Final state of one fixed L-BFGS problem, bit for bit.
//!
//! The constants are the relaxation this file describes. A change that
//! moves an iterate, an energy, a gradient component, or the iteration
//! count fails here.

use ndarray::{Array1, ArrayView1};
use rgmin::Lbfgs;

fn coupled_quadratic(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut f = 0.0;
    let mut g = Array1::zeros(n);
    for i in 0..n {
        let c = (1 + (i % 9)) as f64;
        let t = x[i] - 0.25 * (i as f64);
        f += 0.5 * c * t * t;
        g[i] = c * t;
    }
    for i in 0..n - 1 {
        let d = x[i + 1] - x[i];
        f += 0.05 * d * d;
        g[i] -= 0.1 * d;
        g[i + 1] += 0.1 * d;
    }
    (f, g)
}

#[test]
fn fixed_problem_final_state_is_bit_stable() {
    let x0 = Array1::from_elem(24, 1.0);
    let mut opt = Lbfgs::with_capacity(5);
    opt.gtol = 1e-12;
    let mut steps = 0usize;
    let mut trail: u64 = 0;
    let (f, x, evals) = opt.minimize_watched(
        x0.view(),
        60,
        |v| Some(coupled_quadratic(v)),
        |it, fv| {
            steps = it + 1;
            trail = trail
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(fv.to_bits());
            true
        },
    );
    let (_value, g) = coupled_quadratic(x.view());
    assert_eq!(f.to_bits(), 4589819311624525405);
    assert_eq!(evals, 112);
    assert_eq!(steps, 30);
    assert_eq!(trail, 463538901992949837);
    let x_bits = [
        4582235132037054022,
        4598193933394498374,
        4602679111794636717,
        4604930625956723486,
        4607182418867064416,
        4608308318707957531,
        4609434218613638853,
        4610560118520506615,
        4611686018427286584,
        4612248968380677720,
        4612811918334095101,
        4613374868287399691,
        4613937818241064292,
        4614500768194592292,
        4615063718147977932,
        4615626668101135987,
        4616189618054675995,
        4616471093031384718,
        4616752568007743757,
        4617034042981698445,
        4617315517895096196,
        4617596990822294232,
        4617878379109045191,
        4618155327092086888,
    ];
    let g_bits = [
        4467216994839560192,
        4459345866108436480,
        4457005154968797184,
        4436378110673289216,
        4435353713728552960,
        4447845405991895040,
        13682287657701343232,
        13678091718660980736,
        13685180002897756160,
        13677710880421183488,
        13681847394294038528,
        13688742745311543296,
        13665404482710667264,
        4462707965299785728,
        4460858011216248832,
        13692649370690781184,
        13687422431092801536,
        13692694405134155776,
        13684150080880246784,
        13687573281551417344,
        13694123071445663744,
        13670816389537988608,
        4466899324227289088,
        4465908810346659840,
    ];
    for (i, (got, expect)) in x.iter().zip(x_bits).enumerate() {
        assert_eq!(got.to_bits(), expect, "x[{i}]");
    }
    for (i, (got, expect)) in g.iter().zip(g_bits).enumerate() {
        assert_eq!(got.to_bits(), expect, "g[{i}]");
    }
}
