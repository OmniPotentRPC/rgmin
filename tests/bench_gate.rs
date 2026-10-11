//! Solve test, profiles, and the paired call-count gate.

use rgmin::{
    BenchRow, CallClaim, GateError, GateOpts, Trial, data_profile, gate_bench, more_wild_solved,
    paired_gate, parse_bench_csv, performance_profile, wilcoxon_tails,
};

fn trial(calls: f64, converged: bool) -> Trial {
    Trial { calls, converged }
}

fn series(calls: &[f64]) -> Vec<Trial> {
    calls
        .iter()
        .copied()
        .map(|calls| trial(calls, true))
        .collect()
}

#[test]
fn more_wild_boundary_matches_the_decrease_test() {
    assert!(more_wild_solved(10.0, 1.0, 0.0, 0.1));
    assert!(!more_wild_solved(10.0, 1.0 + 1e-12, 0.0, 0.1));
    assert!(more_wild_solved(10.0, 0.1, 0.0, 0.01));
    assert!(!more_wild_solved(10.0, 0.1 + 1e-9, 0.0, 0.01));
    assert!(more_wild_solved(5.0, 5.0, 5.0, 1.0));
    assert!(!more_wild_solved(5.0, 5.1, 5.0, 1.0));
    assert!(!more_wild_solved(1.0, 0.0, 2.0, 0.1));
    assert!(!more_wild_solved(1.0, 0.0, 0.0, 0.0));
    assert!(!more_wild_solved(1.0, f64::NAN, 0.0, 0.1));
}

#[test]
fn performance_profile_splits_ties_and_keeps_unsolved_problems() {
    let calls = vec![
        vec![Some(2.0), Some(4.0), None],
        vec![Some(4.0), Some(2.0), None],
    ];
    let at_one = performance_profile(&calls, 1.0).unwrap();
    assert_eq!(at_one, vec![1.0 / 3.0, 1.0 / 3.0]);
    let at_two = performance_profile(&calls, 2.0).unwrap();
    assert_eq!(at_two, vec![2.0 / 3.0, 2.0 / 3.0]);
    assert_eq!(performance_profile(&calls, 0.5).unwrap(), vec![0.0, 0.0]);
    assert!(matches!(
        performance_profile(&[vec![Some(1.0)], vec![Some(1.0), None]], 1.0),
        Err(GateError::Shape)
    ));
}

#[test]
fn data_profile_uses_simplex_gradient_units() {
    let calls = vec![
        vec![Some(4.0), Some(8.0), None],
        vec![None, Some(4.0), Some(1.0)],
    ];
    let dim = [3, 3, 0];
    let at_one = data_profile(&calls, &dim, 1.0).unwrap();
    assert_eq!(at_one, vec![1.0 / 3.0, 2.0 / 3.0]);
    let at_two = data_profile(&calls, &dim, 2.0).unwrap();
    assert_eq!(at_two[0], 2.0 / 3.0);
    assert!(matches!(
        data_profile(&calls, &[3, 3], 1.0),
        Err(GateError::Dim {
            got: 2,
            problems: 3
        })
    ));
}

#[test]
fn wilcoxon_exact_tails_match_the_sign_distribution() {
    let (fewer, more) = wilcoxon_tails(&[1.0, 2.0, 3.0]);
    assert_eq!(fewer, 1.0);
    assert_eq!(more, 1.0 / 8.0);
    let (fewer, more) = wilcoxon_tails(&[-1.0, -1.0, -1.0, -1.0, -1.0]);
    assert_eq!(fewer, 1.0 / 32.0);
    assert_eq!(more, 1.0);
    let (fewer, more) = wilcoxon_tails(&[0.0, f64::NAN]);
    assert_eq!((fewer, more), (1.0, 1.0));
}

#[test]
fn equal_shift_of_six_is_a_call_claim() {
    let opts = GateOpts {
        bootstrap: 200,
        ..GateOpts::default()
    };
    let fewer = paired_gate(&series(&[10.0; 6]), &series(&[7.0; 6]), &opts).unwrap();
    assert_eq!(fewer.claim, CallClaim::Fewer);
    assert!(fewer.passed);
    assert_eq!(fewer.median_delta, -3.0);
    assert!(fewer.ci_high < 0.0);
    assert_eq!(fewer.p_fewer, 1.0 / 64.0);
    let more = paired_gate(&series(&[7.0; 6]), &series(&[10.0; 6]), &opts).unwrap();
    assert_eq!(more.claim, CallClaim::More);
    assert!(!more.passed);
    assert_eq!(more.p_more, 1.0 / 64.0);
}

#[test]
fn normal_approximation_flags_a_long_equal_shift() {
    let (fewer, more) = wilcoxon_tails(&[1.0; 17]);
    assert!(fewer > 0.99, "lower tail {fewer}");
    assert!(more < 1e-3, "upper tail {more}");
    let opts = GateOpts {
        bootstrap: 100,
        ..GateOpts::default()
    };
    let report = paired_gate(&series(&[1.0; 17]), &series(&[2.0; 17]), &opts).unwrap();
    assert_eq!(report.claim, CallClaim::More);
    assert!(!report.passed);
    assert!(report.ci_low > 0.0);
}

#[test]
fn four_equal_pairs_stay_undecided() {
    let report = paired_gate(&series(&[1.0; 4]), &series(&[2.0; 4]), &GateOpts::default()).unwrap();
    assert_eq!(report.claim, CallClaim::Undecided);
    assert!(report.passed);
    assert_eq!(report.p_more, 1.0 / 16.0);
}

#[test]
fn identical_calls_are_not_a_regression() {
    let report = paired_gate(
        &series(&[4.0, 5.0, 6.0]),
        &series(&[4.0, 5.0, 6.0]),
        &GateOpts::default(),
    )
    .unwrap();
    assert_eq!(report.claim, CallClaim::Undecided);
    assert!(report.passed);
    assert_eq!(report.median_delta, 0.0);
    assert_eq!(report.n_call, 3);
    assert_eq!(report.p_fewer, 1.0);
}

#[test]
fn a_new_failure_blocks_the_gate() {
    let baseline = series(&[10.0; 6]);
    let mut candidate = series(&[5.0; 6]);
    candidate[3].converged = false;
    let report = paired_gate(&baseline, &candidate, &GateOpts::default()).unwrap();
    assert_eq!(report.claim, CallClaim::Fewer);
    assert_eq!(report.new_failures, 1);
    assert!(!report.passed);
    assert_eq!(report.n_call, 5);
}

#[test]
fn non_finite_calls_count_as_unfinished() {
    let baseline = series(&[3.0, 3.0]);
    let candidate = vec![trial(f64::NAN, true), trial(-1.0, true)];
    let report = paired_gate(&baseline, &candidate, &GateOpts::default()).unwrap();
    assert_eq!(report.new_failures, 2);
    assert!(!report.passed);
    assert_eq!(report.n_call, 0);
}

#[test]
fn length_and_alpha_are_refused() {
    assert_eq!(
        paired_gate(&series(&[1.0]), &series(&[1.0, 2.0]), &GateOpts::default()).unwrap_err(),
        GateError::Length
    );
    let opts = GateOpts {
        alpha: 0.0,
        ..GateOpts::default()
    };
    assert_eq!(
        paired_gate(&series(&[1.0]), &series(&[1.0]), &opts).unwrap_err(),
        GateError::Alpha
    );
}

#[test]
fn bench_csv_pairs_seeds_and_refuses_a_gap() {
    let text = "\
system,method,seed,converged,iters,calls,energy,fmax,wall_s,pot_s,overhead_us_per_iter
lj38,lbfgs,0,1,10,12,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs,1,1,11,14,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs,2,1,9,11,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs,3,1,12,13,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs,4,1,10,12,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs,5,1,10,12,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs-exp,5,1,8,8,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs-exp,1,1,7,7,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs-exp,0,1,6,6,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs-exp,2,1,6,6,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs-exp,3,1,7,7,1.0,1e-3,0.1,0.1,1.0
lj38,lbfgs-exp,4,1,6,6,1.0,1e-3,0.1,0.1,1.0
pt7,lbfgs,0,1,3,4,1.0,1e-3,0.1,0.1,1.0
";
    let rows = parse_bench_csv(text).unwrap();
    assert_eq!(rows.len(), 13);
    assert_eq!(
        rows[0],
        BenchRow {
            system: "lj38".into(),
            method: "lbfgs".into(),
            seed: 0,
            converged: true,
            calls: 12.0,
        }
    );
    let report = gate_bench(&rows, "lj38", "lbfgs", "lbfgs-exp", &GateOpts::default()).unwrap();
    assert_eq!(report.claim, CallClaim::Fewer);
    assert!(report.passed);
    assert_eq!(report.n_pairs, 6);
    let broken = "\
lj38,lbfgs,0,1,1,4,0,0,0,0,0
lj38,other,1,1,1,4,0,0,0,0,0
";
    let rows = parse_bench_csv(broken).unwrap();
    assert_eq!(
        gate_bench(&rows, "lj38", "lbfgs", "other", &GateOpts::default()).unwrap_err(),
        GateError::Pair { seed: 0 }
    );
    assert_eq!(
        parse_bench_csv("lj38,lbfgs,0,yes,1,4\n").unwrap_err(),
        GateError::Csv { line: 1 }
    );
}
