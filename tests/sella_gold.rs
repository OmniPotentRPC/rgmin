//! Golden-master dest stepper and factory numbers against port sources.
//!
//! `tests/sella_manopt_gold.json` is minted from zadorlab/sella
//! `optimize/stepper.py`, `optimize/restricted_step.py`, and
//! `hessian_update.py`, and from manopt factory formulas dest comments
//! cite. Remint:
//! `SELLA_ROOT=/path/to/sella MANOPT_ROOT=/path/to/manopt python3 tests/sella_manopt_gold.py`.
//! When MANOPT_ROOT is set, remint records the factory files it used
//! (`spherefactory.m`, `positivefactory.m`, `symmetricfactory.m`,
//! `complexcirclefactory.m`, `multinomialfactory.m`,
//! `centeredmatrixfactory.m`, `sympositivedefinitefactory.m`).
//! dest tests load the frozen JSON; they do not import Sella or manopt.

use ndarray::{Array1, Array2, array};
use rgmin::manifold::{
    CenteredMatrix, ComplexCircle, Manifold, Multinomial, Positive, Spd, Sphere, Symmetric,
};
use rgmin::{qn_get_s, qn_irc_get_s, ras_clip, rfo_get_s, ts_bfgs_update};

const GOLD: &str = include_str!("sella_manopt_gold.json");
const REMINT: &str = include_str!("sella_manopt_gold.py");
const TOL: f64 = 1e-10;

fn case_slice(name: &str) -> &'static str {
    let key = format!("\"name\": \"{name}\"");
    let start = GOLD
        .find(&key)
        .unwrap_or_else(|| panic!("missing gold case {name}"));
    let rest = &GOLD[start..];
    let end = rest[1..]
        .find("\n    {")
        .or_else(|| rest.find("\n  ]"))
        .unwrap_or(rest.len());
    &rest[..end + 1]
}

fn json_nums(blob: &str, key: &str) -> Vec<f64> {
    let pat = format!("\"{key}\":");
    let start = blob
        .find(&pat)
        .unwrap_or_else(|| panic!("missing key {key}"));
    let after = &blob[start + pat.len()..];
    let lb = after.find('[').unwrap_or_else(|| panic!("{key} is not an array"));
    let mut depth = 0;
    let mut rb = lb;
    for (i, c) in after[lb..].char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    rb = lb + i;
                    break;
                }
            }
            _ => {}
        }
    }
    after[lb + 1..rb]
        .split(|c: char| c == ',' || c == '[' || c == ']' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f64>().unwrap_or_else(|e| panic!("{key} parse {s}: {e}")))
        .collect()
}

fn json_num(blob: &str, key: &str) -> f64 {
    let pat = format!("\"{key}\": ");
    let start = blob
        .find(&pat)
        .unwrap_or_else(|| panic!("missing number {key}"));
    let after = &blob[start + pat.len()..];
    let end = after
        .find(|c: char| c == ',' || c == '\n' || c == '}')
        .unwrap_or(after.len());
    after[..end]
        .trim()
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("{key} parse: {e}"))
}

fn json_usize(blob: &str, key: &str) -> usize {
    json_num(blob, key) as usize
}

fn mat2(vals: &[f64]) -> Array2<f64> {
    assert_eq!(vals.len(), 4, "expected 2x2, got {vals:?}");
    array![[vals[0], vals[1]], [vals[2], vals[3]]]
}

fn vecn(vals: &[f64]) -> Array1<f64> {
    Array1::from(vals.to_vec())
}

/// Analytic 2x2 symmetric eigendecomposition, ascending eigenvalues.
fn eigh2(h: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let a = h[(0, 0)];
    let b = h[(0, 1)];
    let c = h[(1, 1)];
    let tr = a + c;
    let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let l0 = 0.5 * (tr - disc);
    let l1 = 0.5 * (tr + disc);
    let mut v = Array2::<f64>::zeros((2, 2));
    if b.abs() < 1e-15 && (a - c).abs() < 1e-15 {
        v[(0, 0)] = 1.0;
        v[(1, 1)] = 1.0;
    } else if b.abs() >= (a - c).abs() {
        let n0 = ((l0 - c) * (l0 - c) + b * b).sqrt();
        v[(0, 0)] = (l0 - c) / n0;
        v[(1, 0)] = b / n0;
        let n1 = ((l1 - c) * (l1 - c) + b * b).sqrt();
        v[(0, 1)] = (l1 - c) / n1;
        v[(1, 1)] = b / n1;
    } else {
        let n0 = ((l0 - a) * (l0 - a) + b * b).sqrt();
        v[(0, 0)] = b / n0;
        v[(1, 0)] = (l0 - a) / n0;
        let n1 = ((l1 - a) * (l1 - a) + b * b).sqrt();
        v[(0, 1)] = b / n1;
        v[(1, 1)] = (l1 - a) / n1;
    }
    (array![l0, l1], v)
}

fn dest_prfo(h: &Array2<f64>, g: &Array1<f64>, order: usize, alpha: f64) -> Array1<f64> {
    let (evals, evecs) = eigh2(h);
    let n = g.len();
    let order = order.min(n);
    let mut s = Array1::zeros(n);
    if order > 0 {
        let mut gmax = Array1::zeros(order);
        let mut hmax = Array2::<f64>::zeros((order, order));
        for i in 0..order {
            hmax[(i, i)] = evals[i];
            for k in 0..n {
                gmax[i] += evecs[(k, i)] * g[k];
            }
        }
        let smax = rfo_get_s(&hmax, &gmax, order, alpha);
        for i in 0..order {
            for k in 0..n {
                s[k] += evecs[(k, i)] * smax[i];
            }
        }
    }
    let nmin = n - order;
    if nmin > 0 {
        let mut gmin = Array1::zeros(nmin);
        let mut hmin = Array2::<f64>::zeros((nmin, nmin));
        for i in 0..nmin {
            hmin[(i, i)] = evals[order + i];
            for k in 0..n {
                gmin[i] += evecs[(k, order + i)] * g[k];
            }
        }
        let smin = rfo_get_s(&hmin, &gmin, 0, alpha);
        for i in 0..nmin {
            for k in 0..n {
                s[k] += evecs[(k, order + i)] * smin[i];
            }
        }
    }
    s
}

fn assert_close(got: &Array1<f64>, gold: &[f64], name: &str) {
    assert_eq!(got.len(), gold.len(), "{name} len dest={} gold={}", got.len(), gold.len());
    for i in 0..got.len() {
        let err = (got[i] - gold[i]).abs();
        assert!(
            err <= TOL,
            "{name}[{i}] dest={} gold={} err={err}",
            got[i],
            gold[i]
        );
    }
}

#[test]
fn gold_json_was_minted_from_sella_stepper() {
    assert!(
        GOLD.contains("sella/optimize/stepper.py"),
        "gold source must name zadorlab/sella stepper.py"
    );
    assert!(
        GOLD.contains("sella/optimize/restricted_step.py"),
        "gold source must name zadorlab/sella restricted_step.py"
    );
    assert!(
        GOLD.contains("sella/hessian_update.py"),
        "gold source must name zadorlab/sella hessian_update.py"
    );
    assert!(GOLD.contains("\"kind\": \"rfo\""));
    assert!(GOLD.contains("\"kind\": \"qn\""));
    assert!(GOLD.contains("\"kind\": \"prfo\""));
    assert!(GOLD.contains("\"kind\": \"ts_bfgs\""));
    assert!(GOLD.contains("\"kind\": \"ras\""));
}

#[test]
fn remint_imports_restricted_atomic_step_cons() {
    assert!(
        REMINT.contains("restricted_step.py"),
        "remint must load sella/optimize/restricted_step.py"
    );
    assert!(
        REMINT.contains("RestrictedAtomicStep"),
        "remint must import RestrictedAtomicStep"
    );
    assert!(
        REMINT.contains("RestrictedAtomicStep.cons"),
        "remint must call RestrictedAtomicStep.cons"
    );
}

#[test]
fn remint_reads_manopt_root_dest_loads_frozen_json() {
    assert!(
        REMINT.contains("os.environ.get(\"MANOPT_ROOT\""),
        "remint must read MANOPT_ROOT"
    );
    assert!(
        REMINT.contains("spherefactory.m")
            && REMINT.contains("positivefactory.m")
            && REMINT.contains("symmetricfactory.m")
            && REMINT.contains("complexcirclefactory.m")
            && REMINT.contains("multinomialfactory.m")
            && REMINT.contains("centeredmatrixfactory.m")
            && REMINT.contains("sympositivedefinitefactory.m"),
        "remint must record manopt factory files"
    );
    assert!(
        GOLD.contains("\"kind\": \"sphere_proj\"")
            && GOLD.contains("\"kind\": \"sphere_retr\"")
            && GOLD.contains("\"kind\": \"sphere_transp\"")
            && GOLD.contains("\"kind\": \"positive_proj\"")
            && GOLD.contains("\"kind\": \"positive_retr\"")
            && GOLD.contains("\"kind\": \"positive_transp\"")
            && GOLD.contains("\"kind\": \"symmetric_proj\"")
            && GOLD.contains("\"kind\": \"symmetric_retr\"")
            && GOLD.contains("\"kind\": \"symmetric_transp\"")
            && GOLD.contains("\"kind\": \"complexcircle_proj\"")
            && GOLD.contains("\"kind\": \"complexcircle_retr\"")
            && GOLD.contains("\"kind\": \"complexcircle_transp\"")
            && GOLD.contains("\"kind\": \"multinomial_proj\"")
            && GOLD.contains("\"kind\": \"multinomial_retr\"")
            && GOLD.contains("\"kind\": \"multinomial_transp\"")
            && GOLD.contains("\"kind\": \"centered_proj\"")
            && GOLD.contains("\"kind\": \"centered_retr\"")
            && GOLD.contains("\"kind\": \"centered_transp\"")
            && GOLD.contains("\"kind\": \"spd_proj\"")
            && GOLD.contains("\"kind\": \"spd_retr\"")
            && GOLD.contains("\"kind\": \"spd_transp\""),
        "dest tests load frozen JSON factory proj/retr/transp for all seven kinds"
    );
}

#[test]
fn dest_rfo_matches_sella_stepper() {
    for name in [
        "rfo_eye_order0_alpha0.25",
        "rfo_eye_order0_alpha0.5",
        "rfo_eye_order0_alpha1.0",
        "rfo_saddle_order1_alpha1",
    ] {
        let blob = case_slice(name);
        let h = mat2(&json_nums(blob, "H"));
        let g = vecn(&json_nums(blob, "g"));
        let gold = json_nums(blob, "s");
        let dest = rfo_get_s(&h, &g, json_usize(blob, "order"), json_num(blob, "alpha"));
        assert_close(&dest, &gold, name);
    }
}

#[test]
fn dest_qn_matches_sella_stepper() {
    for name in [
        "qn_saddle_order1_alpha0.0",
        "qn_saddle_order1_alpha0.3",
        "qn_saddle_order1_alpha1.0",
    ] {
        let blob = case_slice(name);
        let h = mat2(&json_nums(blob, "H"));
        let g = vecn(&json_nums(blob, "g"));
        let (evals, evecs) = eigh2(&h);
        let (dest, _) = qn_get_s(
            &evals,
            &evecs,
            &g,
            json_usize(blob, "order"),
            json_num(blob, "alpha"),
        );
        assert_close(&dest, &json_nums(blob, "s"), name);
    }
}

#[test]
fn dest_qn_irc_matches_sella_stepper() {
    let name = "qn_irc_order1_alpha0.2";
    let blob = case_slice(name);
    let h = mat2(&json_nums(blob, "H"));
    let g = vecn(&json_nums(blob, "g"));
    let d1 = vecn(&json_nums(blob, "d1"));
    let (evals, evecs) = eigh2(&h);
    let (dest, _) = qn_irc_get_s(&evals, &evecs, &g, &d1, json_num(blob, "alpha"));
    assert_close(&dest, &json_nums(blob, "s"), name);
}

#[test]
fn dest_prfo_matches_sella_stepper() {
    let name = "prfo_saddle_order1_alpha1";
    let blob = case_slice(name);
    let h = mat2(&json_nums(blob, "H"));
    let g = vecn(&json_nums(blob, "g"));
    let dest = dest_prfo(&h, &g, json_usize(blob, "order"), json_num(blob, "alpha"));
    assert_close(&dest, &json_nums(blob, "s"), name);
}

#[test]
fn dest_ts_bfgs_matches_sella_hessian_update() {
    let name = "ts_bfgs_keep_saddle";
    let blob = case_slice(name);
    let mut b = mat2(&json_nums(blob, "B"));
    let step = vecn(&json_nums(blob, "step"));
    let y = vecn(&json_nums(blob, "y"));
    ts_bfgs_update(&mut b, &step, &y);
    let dest = Array1::from(vec![b[(0, 0)], b[(0, 1)], b[(1, 0)], b[(1, 1)]]);
    assert_close(&dest, &json_nums(blob, "s"), name);
}

#[test]
fn dest_ras_clip_matches_sella_max_atom() {
    let name = "ras_clip_max_atom";
    let blob = case_slice(name);
    let step = vecn(&json_nums(blob, "step"));
    let dest = ras_clip(&step, json_num(blob, "delta"));
    assert_close(&dest, &json_nums(blob, "s"), name);
}

fn factory_xv(name: &str) -> (Array1<f64>, Array1<f64>, Vec<f64>) {
    let blob = case_slice(name);
    (
        vecn(&json_nums(blob, "x")),
        vecn(&json_nums(blob, "v")),
        json_nums(blob, "s"),
    )
}

fn factory_xyv(name: &str) -> (Array1<f64>, Array1<f64>, Array1<f64>, Vec<f64>) {
    let blob = case_slice(name);
    (
        vecn(&json_nums(blob, "x")),
        vecn(&json_nums(blob, "y")),
        vecn(&json_nums(blob, "v")),
        json_nums(blob, "s"),
    )
}

#[test]
fn dest_sphere_matches_manopt_spherefactory() {
    let (x, v, gold) = factory_xv("sphere_proj_north");
    assert_close(&Sphere.project(&x, &v), &gold, "sphere_proj");
    let (xr, vr, goldr) = factory_xv("sphere_retr_north");
    assert_close(&Sphere.retract(&xr, &vr), &goldr, "sphere_retr");
    let (xt, yt, vt, goldt) = factory_xyv("sphere_transp_north");
    assert_close(&Sphere.transport(&xt, &yt, &vt), &goldt, "sphere_transp");
}

#[test]
fn dest_positive_matches_manopt_positivefactory() {
    let m = Positive::new(3);
    let (x, v, gold) = factory_xv("positive_proj_id");
    assert_close(&m.project(&x, &v), &gold, "positive_proj");
    let (xr, vr, goldr) = factory_xv("positive_retr_exp");
    assert_close(&m.retract(&xr, &vr), &goldr, "positive_retr");
    let (xt, yt, vt, goldt) = factory_xyv("positive_transp_id");
    assert_close(&m.transport(&xt, &yt, &vt), &goldt, "positive_transp");
}

#[test]
fn dest_symmetric_matches_manopt_symmetricfactory() {
    let (x, v, gold) = factory_xv("symmetric_multisym");
    assert_close(&Symmetric.project(&x, &v), &gold, "symmetric_proj");
    let (xr, vr, goldr) = factory_xv("symmetric_retr_plus");
    assert_close(&Symmetric.retract(&xr, &vr), &goldr, "symmetric_retr");
    let (xt, yt, vt, goldt) = factory_xyv("symmetric_transp_id");
    assert_close(&Symmetric.transport(&xt, &yt, &vt), &goldt, "symmetric_transp");
}

#[test]
fn dest_complexcircle_matches_manopt_complexcirclefactory() {
    let m = ComplexCircle::new(2);
    let (x, v, gold) = factory_xv("complexcircle_proj_pairs");
    assert_close(&m.project(&x, &v), &gold, "complexcircle_proj");
    let (xr, vr, goldr) = factory_xv("complexcircle_retr_sign");
    assert_close(&m.retract(&xr, &vr), &goldr, "complexcircle_retr");
    let (xt, yt, vt, goldt) = factory_xyv("complexcircle_transp_arrive");
    assert_close(&m.transport(&xt, &yt, &vt), &goldt, "complexcircle_transp");
}

#[test]
fn dest_multinomial_matches_manopt_multinomialfactory() {
    let (x, v, gold) = factory_xv("multinomial_proj_fisher");
    assert_close(&Multinomial.project(&x, &v), &gold, "multinomial_proj");
    let (xr, vr, goldr) = factory_xv("multinomial_retr_exp");
    assert_close(&Multinomial.retract(&xr, &vr), &goldr, "multinomial_retr");
    let (xt, yt, vt, goldt) = factory_xyv("multinomial_transp_arrive");
    assert_close(&Multinomial.transport(&xt, &yt, &vt), &goldt, "multinomial_transp");
}

#[test]
fn dest_centered_matches_manopt_centeredmatrixfactory() {
    let m = CenteredMatrix::new(2, 3, false);
    let (x, v, gold) = factory_xv("centered_proj_cols");
    assert_close(&m.project(&x, &v), &gold, "centered_proj");
    let (xr, vr, goldr) = factory_xv("centered_retr_plus");
    assert_close(&m.retract(&xr, &vr), &goldr, "centered_retr");
    let (xt, yt, vt, goldt) = factory_xyv("centered_transp_id");
    assert_close(&m.transport(&xt, &yt, &vt), &goldt, "centered_transp");
}

#[test]
fn dest_spd_matches_manopt_sympositivedefinitefactory() {
    let (x, v, gold) = factory_xv("spd_proj_symm");
    assert_close(&Spd.project(&x, &v), &gold, "spd_proj");
    let (xr, vr, goldr) = factory_xv("spd_retr_second");
    assert_close(&Spd.retract(&xr, &vr), &goldr, "spd_retr");
    let (xt, yt, vt, goldt) = factory_xyv("spd_transp_id");
    assert_close(&Spd.transport(&xt, &yt, &vt), &goldt, "spd_transp");
}
