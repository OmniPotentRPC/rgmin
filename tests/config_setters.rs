//! Every `with_*` setter on a configuration type replaces its own field
//! and leaves every other field as the starting value had it.
//!
//! A type's `Debug` output lists every field, so comparing it against
//! the starting value with only the target field assigned catches a
//! setter that writes the wrong field or more than one.
//! `every_setter_is_covered` reads the sources and fails when a setter
//! or a type has no case here or no literal doctest in `src/config.rs`.

use std::path::Path;

use ndarray::{Array1, array};
use rgmin::rtr::RtrRadius;
use rgmin::{
    AngleTerm, BondTerm, Control, DihedralTerm, DlaFutureParams, EigenExaAlgo, EigenExaParams,
    EigenParams, EigensolverKind, ElpaParams, EonCg, ExpParams, FfParams, FfTerms, Fire2Extras,
    FireConfig, GradNorm, HighsCrossover, HighsOptions, HighsSolverKind, IrcTrust, Lbfgs,
    LbfgsStep, LindhParams, LindhSpec, MorseTerm, Neighbours, NoiseFloor, QuickMinOptions, Report,
    RigidMode, RigidRequest, ScgOptions, ScgParams, ScgStepTolerance, SlepcParams, SlepcPmat,
    SlepcStKind, Status, StopNorm, TrustRegion,
};

/// `$base.$setter($value)` equals `$base` with only `$field` replaced by
/// `$stored` (`$value` when omitted), and that replacement changes the
/// value, so the case cannot pass on a no-op setter.
macro_rules! sets_only {
    ($base:expr, $setter:ident($value:expr), $field:ident = $stored:expr) => {{
        let base = $base;
        let mut want = base.clone();
        want.$field = $stored;
        let expected = format!("{want:?}");
        assert_ne!(
            format!("{base:?}"),
            expected,
            concat!(
                stringify!($setter),
                ": the case value is the starting value"
            )
        );
        let got = base.$setter($value);
        assert_eq!(format!("{got:?}"), expected, stringify!($setter));
    }};
    ($base:expr, $setter:ident($value:expr), $field:ident) => {
        sets_only!($base, $setter($value), $field = $value)
    };
}

/// An `Option` field: a bare value stores `Some`, and `None` clears a
/// stored value without touching the rest.
macro_rules! sets_only_opt {
    ($base:expr, $setter:ident($value:expr), $field:ident) => {{
        sets_only!($base, $setter($value), $field = Some($value));
        sets_only!($base.$setter($value), $setter(None), $field = None);
    }};
}

#[test]
fn control_setters_touch_one_field() {
    let base = Control::default;
    sets_only!(base(), with_maxiter(7), maxiter);
    sets_only!(base(), with_gtol(0.25), gtol);
    sets_only!(base(), with_istep(0.5), istep);
    sets_only_opt!(base(), with_maxmove(0.2), maxmove);
    sets_only!(base(), with_stop_norm(StopNorm::Rms), stop_norm);
    sets_only_opt!(base(), with_ftol_rel(1e-6), ftol_rel);
    sets_only_opt!(base(), with_noise_floor(NoiseFloor::new(1e-3)), noise_floor);
}

#[test]
fn noise_floor_setters_touch_one_field() {
    let base = || NoiseFloor::new(1e-3);
    sets_only!(base(), with_eps_g(2e-3), eps_g);
    sets_only!(base(), with_c(5.0), c);
    sets_only_opt!(base(), with_eps_f(1e-9), eps_f);
}

#[test]
fn scg_params_setters_touch_one_field() {
    let base = ScgParams::default;
    sets_only!(base(), with_sigma0(3e-5), sigma0);
    sets_only!(base(), with_lambda(2e-3), lambda);
    sets_only!(base(), with_lambda_limit(1e10), lambda_limit);
    sets_only!(base(), with_tol_sol(3e-7), tol_sol);
    sets_only!(base(), with_tol_func(3e-7), tol_func);
}

#[test]
fn scg_options_setters_touch_one_field() {
    sets_only!(
        ScgOptions::default(),
        with_step_tolerance(ScgStepTolerance::RelativeEuclidean),
        step_tolerance
    );
}

#[test]
fn eon_cg_setters_touch_one_field() {
    let base = EonCg::default;
    sets_only!(base(), with_max_move(0.05), max_move);
    sets_only!(base(), with_gtol(0.25), gtol);
    sets_only!(base(), with_max_iter(5), max_iter);
    sets_only!(base(), with_cluster(!base().cluster), cluster);
    sets_only!(base(), with_atom_only(!base().atom_only), atom_only);
}

#[test]
fn fire_config_setters_touch_one_field() {
    let base = FireConfig::ase;
    sets_only!(base(), with_dt(0.2), dt);
    sets_only!(base(), with_dt_max(2.0), dt_max);
    sets_only!(base(), with_n_min(3), n_min);
    sets_only!(base(), with_f_inc(1.2), f_inc);
    sets_only!(base(), with_f_dec(0.4), f_dec);
    sets_only!(base(), with_alpha(0.3), alpha);
    sets_only!(base(), with_f_alpha(0.95), f_alpha);
}

#[test]
fn fire2_extras_setters_touch_one_field() {
    let base = || Fire2Extras::new(0.1);
    sets_only!(base(), with_dt_min(1e-4), dt_min);
    sets_only!(base(), with_n_delay(5), n_delay);
    sets_only!(base(), with_initial_delay(false), initial_delay);
    sets_only!(base(), with_steps(4), steps);
    sets_only!(base(), with_n_neg(2), n_neg);
    sets_only!(base(), with_n_neg_max(10), n_neg_max);
}

#[test]
fn quick_min_options_setters_touch_one_field() {
    let base = QuickMinOptions::default;
    sets_only!(base(), with_adapt_dt(!base().adapt_dt), adapt_dt);
    sets_only!(
        base(),
        with_rescale_on_clamp(!base().rescale_on_clamp),
        rescale_on_clamp
    );
    sets_only!(base(), with_per_atom(!base().per_atom), per_atom);
    sets_only_opt!(base(), with_curvature_cap(0.7), curvature_cap);
}

#[cfg(feature = "highs")]
#[test]
fn highs_step_setters_touch_one_field() {
    use rgmin::HighsStep;
    let base = HighsStep::default;
    sets_only_opt!(base(), with_trust(0.3), trust);
    sets_only_opt!(base(), with_lo(-1.0), lo);
    sets_only_opt!(base(), with_hi(1.0), hi);
    sets_only!(
        base(),
        with_equalities(vec![(vec![(0, 1.0)], 0.5)]),
        equalities
    );
    sets_only_opt!(base(), with_center_axes((0, 3)), center_axes);
}

unsafe extern "C" fn quiet_highs(
    _kind: i32,
    _message: *const std::os::raw::c_char,
    _interrupt: *mut i32,
    _user: *mut std::os::raw::c_void,
) {
}

#[test]
fn highs_options_setters_touch_one_field() {
    let base = HighsOptions::default;
    sets_only!(base(), with_solver(HighsSolverKind::Ipm), solver);
    sets_only!(base(), with_crossover(HighsCrossover::Off), crossover);
    // SAFETY: `quiet_highs` reads none of its arguments, and the callback
    // is only compared, never installed.
    let callback = unsafe { rgmin::HighsCallback::new(quiet_highs, std::ptr::null_mut()) };
    sets_only_opt!(base(), with_callback(callback), callback);
}

#[test]
fn trust_region_setters_touch_one_field() {
    let base = || TrustRegion::new(0.1);
    sets_only!(base(), with_delta(0.3), delta);
    sets_only!(base(), with_tol(0.25), tol);
    sets_only!(base(), with_maxiter(7), maxiter);
    sets_only!(base(), with_order(base().order + 1), order);
}

#[test]
fn rtr_radius_setters_touch_one_field() {
    let base = || RtrRadius::new(1.0);
    sets_only!(base(), with_radius(0.3), radius);
    sets_only!(base(), with_radius_max(2.0), radius_max);
}

#[test]
fn irc_trust_setters_touch_one_field() {
    let base = || IrcTrust::new(Array1::zeros(3), Array1::ones(3), 0.1);
    sets_only!(base(), with_d1(array![1.0, 0.0, 0.0]), d1);
    sets_only!(base(), with_sqrtm(array![2.0, 2.0, 2.0]), sqrtm);
    sets_only!(base(), with_dx(0.2), dx);
}

#[test]
fn eigen_params_setters_touch_one_field() {
    let base = EigenParams::default;
    sets_only!(base(), with_kind(EigensolverKind::JacobiDavidson), kind);
    sets_only!(base(), with_nev(3), nev);
    sets_only!(base(), with_krylov(17), krylov);
    sets_only!(base(), with_max_iter(7), max_iter);
    sets_only!(base(), with_tol(0.25), tol);
    sets_only!(base(), with_degree(9), degree);
    sets_only!(base(), with_extra(5), extra);
}

#[test]
fn elpa_params_setters_touch_one_field() {
    sets_only!(ElpaParams::default(), with_nblk(48), nblk);
}

#[test]
fn eigen_exa_params_setters_touch_one_field() {
    let base = EigenExaParams::default;
    let other = match base().algo {
        EigenExaAlgo::S => EigenExaAlgo::Sx,
        EigenExaAlgo::Sx => EigenExaAlgo::S,
    };
    sets_only!(base(), with_algo(other), algo);
}

#[test]
fn slepc_params_setters_touch_one_field() {
    let base = SlepcParams::default;
    sets_only!(base(), with_st(SlepcStKind::Cayley), st);
    // SAFETY: the handle is only stored and compared; no SLEPc call
    // receives it.
    let pmat = unsafe { SlepcPmat::from_raw(std::ptr::dangling_mut()) };
    sets_only_opt!(base(), with_pmat(pmat), pmat);
}

#[test]
fn dla_future_params_setters_touch_one_field() {
    sets_only!(DlaFutureParams::default(), with_begin(3), begin);
}

#[test]
fn exp_params_setters_touch_one_field() {
    let base = ExpParams::default;
    sets_only!(base(), with_a(2.5), a);
    sets_only!(base(), with_c_stab(0.25), c_stab);
    sets_only_opt!(base(), with_mu(1.5), mu);
    sets_only_opt!(base(), with_r_nn(2.2), r_nn);
    sets_only_opt!(base(), with_r_cut(4.0), r_cut);
    sets_only!(base(), with_recalc_mu(!base().recalc_mu), recalc_mu);
    sets_only_opt!(base(), with_lengths([5.0, 6.0, 7.0]), lengths);
}

#[test]
fn neighbours_setters_touch_one_field() {
    let base = Neighbours::default;
    sets_only!(base(), with_i(vec![0, 1]), i);
    sets_only!(base(), with_j(vec![1, 2]), j);
    sets_only!(base(), with_r(vec![1.5, 2.5]), r);
    sets_only!(base(), with_anchor(vec![(0, 1.25)]), anchor);
}

#[test]
fn lindh_params_setters_touch_one_field() {
    let base = LindhParams::default;
    sets_only!(base(), with_c_stab(0.25), c_stab);
    sets_only_opt!(base(), with_r_cut(4.0), r_cut);
    sets_only!(base(), with_angles(!base().angles), angles);
    sets_only!(base(), with_dihedrals(!base().dihedrals), dihedrals);
    sets_only_opt!(base(), with_lengths([5.0, 6.0, 7.0]), lengths);
    sets_only_opt!(base(), with_mu(1.5), mu);
    sets_only!(base(), with_recalc_mu(!base().recalc_mu), recalc_mu);
}

#[test]
fn ff_params_setters_touch_one_field() {
    let base = FfParams::default;
    sets_only!(base(), with_c_stab(0.25), c_stab);
    sets_only_opt!(base(), with_r_cut(4.0), r_cut);
    sets_only_opt!(base(), with_lengths([5.0, 6.0, 7.0]), lengths);
    sets_only_opt!(base(), with_mu(1.5), mu);
    sets_only!(base(), with_recalc_mu(!base().recalc_mu), recalc_mu);
}

#[test]
fn ff_terms_setters_touch_one_field() {
    let bonds = [BondTerm { i: 0, j: 1, k: 1.0 }];
    let angles = [AngleTerm {
        i: 0,
        vertex: 1,
        k: 2,
        k_theta: 0.5,
    }];
    let morses = [MorseTerm {
        i: 0,
        j: 2,
        d: 1.0,
        alpha: 1.5,
        r0: 1.1,
    }];
    let dihedrals = [DihedralTerm {
        i: 0,
        j: 1,
        k: 2,
        l: 3,
        k_phi: 0.1,
    }];
    let base = FfTerms::default;
    sets_only!(base(), with_bonds(&bonds), bonds = &bonds[..]);
    sets_only!(base(), with_angles(&angles), angles = &angles[..]);
    sets_only!(base(), with_morses(&morses), morses = &morses[..]);
    sets_only!(
        base(),
        with_dihedrals(&dihedrals),
        dihedrals = &dihedrals[..]
    );
}

#[test]
fn lindh_spec_setters_touch_one_field() {
    let base = || LindhSpec::new(vec![1, 8], 0.1, [10.0; 3]);
    sets_only!(base(), with_z(vec![6, 1]), z);
    sets_only!(base(), with_c_stab(0.25), c_stab);
    sets_only!(base(), with_cell([5.0, 6.0, 7.0]), cell);
}

#[test]
fn rigid_request_setters_touch_one_field() {
    let pos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let moved = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let masses = [1.0, 16.0];
    let frozen = [0u8, 0, 0, 1, 1, 1];
    let base = || RigidRequest::cartesian(&pos, RigidMode::Auto);
    sets_only!(base(), with_pos(&moved), pos = &moved[..]);
    sets_only!(base(), with_masses(&masses[..]), masses = Some(&masses[..]));
    sets_only!(
        base().with_masses(&masses[..]),
        with_masses(None),
        masses = None
    );
    sets_only!(
        base(),
        with_mass_weighted_coords(true),
        mass_weighted_coords
    );
    sets_only!(base(), with_mass_metric(true), mass_metric);
    sets_only!(base(), with_mode(RigidMode::TranslationRotation), mode);
    sets_only!(base(), with_pbc([true, false, true]), pbc);
    sets_only!(base(), with_frozen(&frozen[..]), frozen = Some(&frozen[..]));
    sets_only!(
        base().with_frozen(&frozen[..]),
        with_frozen(None),
        frozen = None
    );
    sets_only!(
        base(),
        with_broken_translation([false, true, false]),
        broken_translation
    );
    sets_only!(
        base(),
        with_broken_rotation([true, false, false]),
        broken_rotation
    );
}

#[test]
fn report_setters_touch_one_field() {
    let base = || Report::new(1.0, array![0.5, -0.5]);
    sets_only!(base(), with_value(2.0), value);
    sets_only!(base(), with_coords(array![1.0, 2.0]), coords);
    sets_only!(base(), with_steps(9), steps);
    sets_only!(base(), with_grad_norm(0.125), grad_norm);
    sets_only!(base(), with_stop_norm(StopNorm::Rms), stop_norm);
    sets_only_opt!(base(), with_grad(array![0.1, 0.2]), grad);
    sets_only!(base(), with_status(Status::Budget), status);
}

/// `Lbfgs` holds curvature memory and is neither `Clone` nor `Debug`, so
/// the comparison is over its public fields.
fn lbfgs_fields(s: &Lbfgs) -> Vec<String> {
    let out = vec![
        format!("{:?}", s.max_pairs),
        format!("{:?}", s.gtol),
        format!("{:?}", s.armijo),
        format!("{:?}", s.curvature),
        format!("{:?}", s.max_line_evals),
        format!("{:?}", s.norm),
        format!("{:?}", s.ftol),
        format!("{:?}", s.step),
        format!("{:?}", s.fixed_h0),
        format!("{:?}", s.extra_updates),
        format!("{:?}", s.cautious_eps),
        format!("{:?}", s.cautious_alpha),
        format!("{:?}", s.distance_reset),
        format!("{:?}", s.angle_reset),
        format!("{:?}", s.curvature_reset),
        format!("{:?}", s.inverse_curvature),
    ];
    #[cfg(feature = "highs")]
    let out = {
        let mut out = out;
        out.push(format!("{:?}", s.highs));
        out.push(format!("{:?}", s.highs_options));
        out
    };
    out
}

macro_rules! lbfgs_sets_only {
    ($setter:ident($value:expr), $field:ident = $stored:expr) => {{
        let mut want = Lbfgs::default();
        want.$field = $stored;
        let expected = lbfgs_fields(&want);
        assert_ne!(
            lbfgs_fields(&Lbfgs::default()),
            expected,
            concat!(
                stringify!($setter),
                ": the case value is the starting value"
            )
        );
        let got = Lbfgs::default().$setter($value);
        assert_eq!(lbfgs_fields(&got), expected, stringify!($setter));
    }};
    ($setter:ident($value:expr), $field:ident) => {
        lbfgs_sets_only!($setter($value), $field = $value)
    };
}

#[test]
fn lbfgs_setters_touch_one_field() {
    let base = Lbfgs::default();
    lbfgs_sets_only!(with_max_pairs(3), max_pairs);
    lbfgs_sets_only!(with_gtol(0.25), gtol);
    lbfgs_sets_only!(with_armijo(1e-3), armijo);
    lbfgs_sets_only!(with_curvature(0.5), curvature);
    lbfgs_sets_only!(with_max_line_evals(7), max_line_evals);
    lbfgs_sets_only!(with_norm(GradNorm::Euclidean), norm);
    lbfgs_sets_only!(with_ftol(1e-9), ftol);
    lbfgs_sets_only!(with_step(LbfgsStep::Wolfe), step);
    lbfgs_sets_only!(with_fixed_h0(1.0 / 70.0), fixed_h0 = Some(1.0 / 70.0));
    lbfgs_sets_only!(with_extra_updates(2), extra_updates);
    lbfgs_sets_only!(with_cautious_eps(1e-6), cautious_eps);
    lbfgs_sets_only!(with_cautious_alpha(0.5), cautious_alpha);
    lbfgs_sets_only!(with_distance_reset(!base.distance_reset), distance_reset);
    lbfgs_sets_only!(with_angle_reset(!base.angle_reset), angle_reset);
    lbfgs_sets_only!(with_curvature_reset(!base.curvature_reset), curvature_reset);
    lbfgs_sets_only!(with_inverse_curvature(0.5), inverse_curvature);

    let cleared = Lbfgs::default().with_fixed_h0(0.5).with_fixed_h0(None);
    assert_eq!(lbfgs_fields(&cleared), lbfgs_fields(&base));
}

#[cfg(feature = "highs")]
#[test]
fn lbfgs_highs_setters_touch_one_field() {
    use rgmin::HighsStep;
    let step = HighsStep::default().with_trust(0.3);
    lbfgs_sets_only!(with_highs(step.clone()), highs = Some(step.clone()));
    let options = HighsOptions::default().with_solver(HighsSolverKind::Ipm);
    lbfgs_sets_only!(with_highs_options(options), highs_options);

    let cleared = Lbfgs::default().with_highs(step).with_highs(None);
    assert_eq!(lbfgs_fields(&cleared), lbfgs_fields(&Lbfgs::default()));
}

/// `CamelCase` to `snake_case`, the case-function naming above.
fn snake(name: &str) -> String {
    let mut out = String::new();
    for (k, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if k > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

fn rust_sources(dir: &Path, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(std::fs::read_to_string(&path).unwrap());
        }
    }
}

/// What one source file declares: `with_setters!` entries, hand-written
/// `pub fn with_*(mut self, ...)` in an inherent impl, and public fields.
#[derive(Default)]
struct Declared {
    generated: Vec<(String, String)>,
    hand: Vec<(String, String)>,
    fields: Vec<(String, String)>,
}

fn ident(text: &str) -> String {
    text.chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

fn declared_in(source: &str, out: &mut Declared) {
    let mut in_macro: Option<String> = None;
    let mut in_impl: Option<String> = None;
    let mut in_struct: Option<String> = None;
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("with_setters!(") {
            let name = ident(rest);
            if !name.is_empty() {
                in_macro = Some(name);
            }
            continue;
        }
        if let Some(ty) = &in_macro {
            if trimmed.starts_with("});") {
                in_macro = None;
            } else if let Some((setter, _)) = trimmed.split_once("=>") {
                out.generated.push((ty.clone(), setter.trim().to_string()));
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("pub struct ") {
            if trimmed.ends_with('{') {
                in_struct = Some(ident(rest));
            }
            continue;
        }
        if let Some(ty) = &in_struct {
            if trimmed == "}" {
                in_struct = None;
            } else if let Some((name, _)) =
                trimmed.strip_prefix("pub ").and_then(|r| r.split_once(':'))
            {
                out.fields.push((ty.clone(), name.to_string()));
            }
            continue;
        }
        if line.starts_with("impl") && !line.contains(" for ") {
            let head = line.trim_end_matches('{').trim();
            in_impl = Some(ident(head.rsplit(' ').next().unwrap_or("")));
        } else if line.starts_with('}') {
            in_impl = None;
        }
        if let (Some(ty), Some(rest)) = (&in_impl, trimmed.strip_prefix("pub fn with_"))
            && let Some((name, args)) = rest.split_once('(')
            && args.starts_with("mut self")
        {
            out.hand.push((ty.clone(), name.to_string()));
        }
    }
}

/// `(type, setter)` for every setter of a public field: each
/// `with_setters!` entry, and each hand-written `with_x` whose type has
/// a public field `x`. A `with_*` builder over private state is not one.
fn field_setters(sources: &[String]) -> Vec<(String, String)> {
    let mut declared = Declared::default();
    for source in sources {
        declared_in(source, &mut declared);
    }
    let mut out = declared.generated;
    for (ty, field) in declared.hand {
        if declared.fields.iter().any(|(t, f)| *t == ty && *f == field) {
            out.push((ty, format!("with_{field}")));
        }
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn every_setter_is_covered() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = Vec::new();
    rust_sources(&root.join("src"), &mut sources);
    let setters = field_setters(&sources);
    assert!(setters.len() > 100, "found only {} setters", setters.len());

    let here = std::fs::read_to_string(root.join("tests/config_setters.rs")).unwrap();
    let guards = std::fs::read_to_string(root.join("src/config.rs")).unwrap();
    let mut missing = Vec::new();
    let mut types: Vec<&str> = setters.iter().map(|(t, _)| t.as_str()).collect();
    types.dedup();
    for ty in types {
        let header = format!("fn {}_setters_touch_one_field()", snake(ty));
        let Some(start) = here.find(&header) else {
            missing.push(format!("{ty}: no `{header}`"));
            continue;
        };
        let body = &here[start..];
        let body = &body[..body.find("\n}\n").unwrap_or(body.len())];
        let body = if ty == "Lbfgs" {
            let highs = "fn lbfgs_highs_setters_touch_one_field()";
            let extra = &here[here.find(highs).unwrap()..];
            format!(
                "{body}{}",
                &extra[..extra.find("\n}\n").unwrap_or(extra.len())]
            )
        } else {
            body.to_string()
        };
        for (_, setter) in setters.iter().filter(|(t, _)| t == ty) {
            if !body.contains(&format!("{setter}(")) {
                missing.push(format!("{ty}::{setter}: no case"));
            }
        }
        if !guards.contains(&format!("literal_fails!({ty},"))
            && !guards.contains(&format!("literal_fails!(\n        {ty},"))
        {
            missing.push(format!("{ty}: no literal doctest in src/config.rs"));
        }
    }
    assert!(missing.is_empty(), "{missing:#?}");
}
