//! Fifty perturbed starts per system and method, and the rule for keeping
//! a change against that sample.
//!
//! Each start adds `N(0, 0.1^2)` to every free coordinate of seed `0..50`.
//! A change is kept when the sum of the twelve median call counts falls,
//! no median rises, a two-sided Mann-Whitney test does not find higher
//! final energies at `p < 0.05`, the rate of reaching the known lowest
//! minimum does not fall by more than the baseline binomial standard
//! error, every start the baseline converged still converges, and each
//! Pt7 energy stays within `1e-8` relative of the baseline energy on
//! that same start.
//!
//! Lennard-Jones reference energies are the Cambridge Cluster Database
//! table (pair well depth, six decimals): LJ13 `Ih` -44.326801, LJ38
//! `Oh` -173.928427, LJ75 `D5h` -397.492331. A hit lies within `1e-4`
//! relative of that value, inside the gap to the next isomer. The Pt7
//! reference is the unperturbed L-BFGS energy of `pt_heptamer.con`, and
//! a hit lies within `1e-8` relative of it.

use super::{build, lj_cluster, lj38, read_con, run_session};

const STARTS: u64 = 50;
const SIGMA: f64 = 0.1;
const LJ_HIT: f64 = 1e-4;
const PT_HIT: f64 = 1e-8;

/// Cambridge Cluster Database global minima, and the relaxed Pt7 heptamer.
fn known_minimum(system: &str) -> f64 {
    match system {
        "lj13" => -44.326801,
        "lj38" => -173.928427,
        "lj75" => -397.492331,
        "pt7" => -1776.0160052124,
        _ => panic!("no published minimum for {system}"),
    }
}

fn hit_tolerance(system: &str) -> f64 {
    if system == "pt7" { PT_HIT } else { LJ_HIT }
}

/// Type-7 quantile: linear interpolation between adjacent order statistics.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    debug_assert!(n > 0);
    let x = p * (n - 1) as f64;
    let i = x.floor() as usize;
    let frac = x - i as f64;
    if i + 1 >= n {
        sorted[n - 1]
    } else {
        sorted[i] * (1.0 - frac) + sorted[i + 1] * frac
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    quantile(&sorted, 0.5)
}

struct Start {
    calls: f64,
    energy: f64,
    converged: bool,
}

/// One system and method, seeds `0..STARTS` in order.
struct Sample {
    system: &'static str,
    method: &'static str,
    starts: Vec<Start>,
}

fn is_hit(system: &str, energy: f64) -> bool {
    let known = known_minimum(system);
    energy.is_finite() && (energy - known).abs() / known.abs() <= hit_tolerance(system)
}

/// Two-sided Mann–Whitney `p`, the `U` statistic of `new`, and its null mean.
///
/// Ranks use the average of tied values. The variance uses the tie
/// correction `Σ(t³ − t) / (n(n−1))`. The normal approximation applies
/// the continuity correction of one half. A zero variance (every value
/// equal) has `p = 1`.
fn mann_whitney(new: &[f64], old: &[f64]) -> (f64, f64, f64) {
    let n1 = new.len();
    let n2 = old.len();
    let n = n1 + n2;
    let mut pooled: Vec<(f64, usize)> = new
        .iter()
        .chain(old.iter())
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pooled.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut ranks = vec![0.0; n];
    let mut tie_sum = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && pooled[j + 1].0 == pooled[i].0 {
            j += 1;
        }
        let average = 0.5 * ((i + 1) + (j + 1)) as f64;
        for item in pooled.iter().take(j + 1).skip(i) {
            ranks[item.1] = average;
        }
        let t = (j - i + 1) as f64;
        tie_sum += t * t * t - t;
        i = j + 1;
    }
    let rank_sum: f64 = ranks.iter().take(n1).sum();
    let u = rank_sum - n1 as f64 * (n1 as f64 + 1.0) / 2.0;
    let mu = n1 as f64 * n2 as f64 / 2.0;
    if n < 2 {
        return (1.0, u, mu);
    }
    let var =
        n1 as f64 * n2 as f64 / 12.0 * ((n as f64 + 1.0) - tie_sum / (n as f64 * (n as f64 - 1.0)));
    if var.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return (1.0, u, mu);
    }
    let z = ((u - mu).abs() - 0.5).max(0.0) / var.sqrt();
    let p = libm::erfc(z / std::f64::consts::SQRT_2);
    (p, u, mu)
}

struct CellDecision {
    median_old: f64,
    median_new: f64,
    energy_worsened: bool,
    hit_rate_fell: bool,
    pt7_moved: bool,
    convergence_lost: bool,
}

fn judge_cell(old: &Sample, new: &Sample) -> CellDecision {
    assert_eq!(old.system, new.system);
    assert_eq!(old.starts.len(), new.starts.len());
    let median_old = median(&old.starts.iter().map(|s| s.calls).collect::<Vec<_>>());
    let median_new = median(&new.starts.iter().map(|s| s.calls).collect::<Vec<_>>());
    let old_e: Vec<f64> = old.starts.iter().map(|s| s.energy).collect();
    let new_e: Vec<f64> = new.starts.iter().map(|s| s.energy).collect();
    let (p, u, mu) = mann_whitney(&new_e, &old_e);
    let hits_old = old
        .starts
        .iter()
        .filter(|s| is_hit(old.system, s.energy))
        .count();
    let hits_new = new
        .starts
        .iter()
        .filter(|s| is_hit(new.system, s.energy))
        .count();
    let n = old.starts.len() as f64;
    let rate = hits_old as f64 / n;
    let noise = (rate * (1.0 - rate) / n).sqrt();
    let mut pt7_moved = false;
    if old.system == "pt7" {
        pt7_moved = old.starts.iter().zip(new.starts.iter()).any(|(a, b)| {
            let scale = a.energy.abs().max(1e-300);
            (b.energy - a.energy).abs() / scale > PT_HIT
        });
    }
    let convergence_lost = old
        .starts
        .iter()
        .zip(new.starts.iter())
        .any(|(a, b)| a.converged && !b.converged);
    CellDecision {
        median_old,
        median_new,
        energy_worsened: p < 0.05 && u > mu,
        hit_rate_fell: hits_new as f64 / n < rate - noise,
        pt7_moved,
        convergence_lost,
    }
}

/// The twelve-cell keep rule. `old` and `new` are the same systems and
/// methods, in the same order.
fn keep(old: &[Sample], new: &[Sample]) -> bool {
    assert_eq!(old.len(), new.len());
    let cells: Vec<CellDecision> = old
        .iter()
        .zip(new.iter())
        .map(|(a, b)| judge_cell(a, b))
        .collect();
    let sum_old: f64 = cells.iter().map(|c| c.median_old).sum();
    let sum_new: f64 = cells.iter().map(|c| c.median_new).sum();
    let median_rose = cells.iter().any(|c| c.median_new > c.median_old);
    let clean = cells
        .iter()
        .all(|c| !c.energy_worsened && !c.hit_rate_fell && !c.pt7_moved && !c.convergence_lost);
    sum_new < sum_old && !median_rose && clean
}

fn run_cell(system: &'static str, method: &'static str) -> Sample {
    let mut starts = Vec::with_capacity(STARTS as usize);
    for seed in 0..STARTS {
        let sys = match system {
            "lj13" => lj_cluster(13, 2.0, seed),
            "lj38" => lj38(seed),
            "lj75" => lj_cluster(75, 3.2, seed),
            "pt7" => read_con("data/pt_heptamer.con", 78, 1),
            _ => unreachable!(),
        };
        let (obj, x0) = build(&sys, SIGMA, seed);
        let out = run_session(&obj, x0, &sys, method, 0.01, 2000, 0.2);
        starts.push(Start {
            calls: obj.calls() as f64,
            energy: out.energy,
            converged: out.converged,
        });
    }
    Sample {
        system,
        method,
        starts,
    }
}

fn suite() -> Vec<Sample> {
    let mut cells = Vec::new();
    for system in ["lj13", "lj38", "lj75", "pt7"] {
        for method in ["lbfgs", "cg", "fire"] {
            cells.push(run_cell(system, method));
        }
    }
    cells
}

fn describe(cell: &Sample) -> String {
    let calls: Vec<f64> = cell.starts.iter().map(|s| s.calls).collect();
    let mut energy: Vec<f64> = cell.starts.iter().map(|s| s.energy).collect();
    energy.sort_by(f64::total_cmp);
    let hits = cell
        .starts
        .iter()
        .filter(|s| is_hit(cell.system, s.energy))
        .count();
    let converged = cell.starts.iter().filter(|s| s.converged).count();
    format!(
        "{} {} med {:.1} conv {converged} hits {hits} e {:.10} {:.10} {:.10} {:.10} {:.10}",
        cell.system,
        cell.method,
        median(&calls),
        quantile(&energy, 0.0),
        quantile(&energy, 0.25),
        quantile(&energy, 0.5),
        quantile(&energy, 0.75),
        quantile(&energy, 1.0),
    )
}

#[test]
fn mann_whitney_matches_the_tie_corrected_normal_approximation() {
    let (p, u, mu) = mann_whitney(&[4.0, 5.0, 6.0], &[1.0, 2.0, 3.0]);
    assert!((u - 9.0).abs() < 1e-12, "{u}");
    assert!((mu - 4.5).abs() < 1e-12, "{mu}");
    assert!((p - 0.0808555983700523).abs() < 1e-12, "{p}");

    let (p, u, mu) = mann_whitney(&[0.0, 0.0, 0.0, 1.0], &[2.0, 3.0, 3.0, 4.0]);
    assert!((u - 0.0).abs() < 1e-12, "{u}");
    assert!((mu - 8.0).abs() < 1e-12, "{mu}");
    assert!((p - 0.02558014672695405).abs() < 1e-12, "{p}");
    // The low sample is better, not worse: p < 0.05 and U < mean.
    assert!(p < 0.05 && u < mu);

    let (p, u, mu) = mann_whitney(&[2.0, 3.0, 3.0, 4.0], &[0.0, 0.0, 0.0, 1.0]);
    assert!(p < 0.05 && u > mu, "higher sample p {p} U {u} mean {mu}");

    let (p, u, _) = mann_whitney(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert_eq!(p, 1.0);
    assert!((u - 4.5).abs() < 1e-12);
}

fn flat(system: &'static str, method: &'static str, calls: f64, energy: f64, n: usize) -> Sample {
    Sample {
        system,
        method,
        starts: (0..n)
            .map(|_| Start {
                calls,
                energy,
                converged: true,
            })
            .collect(),
    }
}

#[test]
fn the_keep_rule_requires_fewer_calls_and_no_worse_energies() {
    let old = vec![
        flat("lj13", "lbfgs", 10.0, -1.0, 50),
        flat("pt7", "cg", 20.0, -1776.0160052124, 50),
    ];
    let fewer = vec![
        flat("lj13", "lbfgs", 8.0, -1.0, 50),
        flat("pt7", "cg", 20.0, -1776.0160052124, 50),
    ];
    assert!(keep(&old, &fewer));

    let slower = vec![
        flat("lj13", "lbfgs", 12.0, -1.0, 50),
        flat("pt7", "cg", 19.0, -1776.0160052124, 50),
    ];
    assert!(!keep(&old, &slower), "a risen median is not kept");

    let worse = vec![
        flat("lj13", "lbfgs", 8.0, -0.5, 50),
        flat("pt7", "cg", 20.0, -1776.0160052124, 50),
    ];
    assert!(
        !keep(&old, &worse),
        "stochastically higher energy is not kept"
    );

    let mut moved = fewer;
    moved[1].starts[0].energy *= 1.0 + 1e-6;
    assert!(!keep(&old, &moved), "Pt7 must match to 1e-8 relative");

    let mut missed = flat("lj13", "lbfgs", 8.0, known_minimum("lj13"), 50);
    let baseline_hits = flat("lj13", "lbfgs", 10.0, known_minimum("lj13"), 50);
    // Ten starts sit just outside the 1e-4 window. The rest match, so the
    // rank test does not have to be the reason the hit rate fails.
    for start in missed.starts.iter_mut().take(10) {
        start.energy = known_minimum("lj13") * (1.0 - 2e-4);
    }
    let old_hits = vec![baseline_hits];
    let new_hits = vec![missed];
    assert!(
        !keep(&old_hits, &new_hits),
        "hit rate fell by more than its noise"
    );
}

#[test]
fn perturbed_starts_record_calls_energies_and_hit_rates() {
    let cells = suite();
    let got = cells.iter().map(describe).collect::<Vec<_>>().join("\n");
    assert_eq!(got, EXPECT);
    // The recorded sample compared with itself does not drop any calls.
    assert!(!keep(&cells, &cells));
}

const EXPECT: &str = "\
lj13 lbfgs med 78.5 conv 50 hits 7 e -44.3267739636 -41.3943702892 -39.9017885716 -39.4804896386 -37.3620814708
lj13 cg med 147.5 conv 50 hits 7 e -44.3267739994 -41.3943700725 -40.0201992191 -39.4863679874 -37.3620804261
lj13 fire med 174.5 conv 50 hits 6 e -44.3267721578 -41.3943683190 -39.8174022629 -39.4317778566 -37.3620399021
lj38 lbfgs med 167.5 conv 50 hits 0 e -170.4713683953 -168.4169135461 -167.6403162857 -166.5276243934 -162.3740739132
lj38 cg med 403.5 conv 50 hits 0 e -170.3097331341 -168.4023705753 -167.4954911250 -166.0014298660 -163.7652533613
lj38 fire med 348.5 conv 50 hits 0 e -169.9610637047 -168.1542020253 -167.4729067303 -165.9733192576 -162.5557850730
lj75 lbfgs med 287.0 conv 50 hits 0 e -392.7852850634 -380.6434365962 -378.8226488826 -377.0567140985 -372.4082347086
lj75 cg med 731.5 conv 50 hits 0 e -389.6543849073 -381.2000275428 -378.6614932289 -377.1934474493 -372.2393931513
lj75 fire med 733.5 conv 47 hits 0 e -384.7250691061 -381.7804035145 -379.1756621304 -377.1257014681 -369.9238342993
pt7 lbfgs med 10.0 conv 50 hits 50 e -1776.0160119942 -1776.0160102886 -1776.0160065347 -1776.0160012052 -1776.0159875696
pt7 cg med 19.0 conv 50 hits 49 e -1776.0160116224 -1776.0160091544 -1776.0160064785 -1776.0160027824 -1776.0159867616
pt7 fire med 53.0 conv 50 hits 16 e -1776.0160092987 -1776.0159921917 -1776.0159821079 -1776.0159723772 -1776.0159333592";
