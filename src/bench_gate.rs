//! Statistical benchmark gate for repeated solver runs.
//!
//! A run is solved when the objective decrease meets the Moré and Wild
//! test (SIAM J. Optim. 20, 172, 2009, doi:10.1137/080724083):
//!
//! ```text
//! f(x0) - f(x) >= (1 - τ) (f(x0) - f_L)
//! ```
//!
//! [`performance_profile`] is the Dolan and Moré fraction of problems
//! solved within a factor of the best solver (Math. Program. 91, 201,
//! 2002, doi:10.1007/s101070100263). [`data_profile`] is the Moré and
//! Wild fraction solved within a budget of `α (n + 1)` calls.
//!
//! [`paired_gate`] compares force-call counts on paired seeds. The
//! candidate is reported as using fewer calls only when the one-sided
//! Wilcoxon signed-rank test (Biometrics Bulletin 1, 80, 1945,
//! doi:10.2307/3001968) and the percentile bootstrap interval (Efron,
//! Ann. Statist. 7, 1, 1979, doi:10.1214/aos/1176344552) agree. The
//! gate does not pass when the candidate fails a trial the baseline
//! solved. The atomistic bench CSV
//! (`system,method,seed,converged,iters,calls,...`) is the row format
//! [`parse_bench_csv`] reads.

use thiserror::Error;

/// Why a gate comparison was refused.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GateError {
    /// `alpha` was outside `(0, 1)`.
    #[error("alpha must lie in (0, 1)")]
    Alpha,
    /// The bootstrap drew no resamples.
    #[error("bootstrap count must be at least 1")]
    Bootstrap,
    /// The two series do not have the same length.
    #[error("paired series have different lengths")]
    Length,
    /// The requested system and methods have no rows.
    #[error("no paired trials")]
    Empty,
    /// A solver's problem list has a different length, or the profile
    /// has no solvers.
    #[error("profile shape is not rectangular")]
    Shape,
    /// `data_profile` was given a dimension per problem that does not
    /// match the problem count.
    #[error("dimension list length {got} != problem count {problems}")]
    Dim {
        /// Length of the supplied dimension list.
        got: usize,
        /// Number of problems in the call table.
        problems: usize,
    },
    /// One seed is missing, duplicated, or present for only one method.
    #[error("seed {seed} is not paired")]
    Pair {
        /// The seed that broke the pairing.
        seed: u64,
    },
    /// A CSV line does not have the bench columns.
    #[error("bench csv line {line} is not a data row")]
    Csv {
        /// 1-based line number in the text.
        line: usize,
    },
}

/// One finished trial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trial {
    /// Force calls, or objective evaluations, recorded for the trial.
    pub calls: f64,
    /// The run met its convergence test.
    pub converged: bool,
}

/// One row of the atomistic bench CSV.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchRow {
    /// System name (`lj38`, `pt7`, `al`, ...).
    pub system: String,
    /// Method name.
    pub method: String,
    /// Perturbation seed.
    pub seed: u64,
    /// `1` in the CSV when the run converged.
    pub converged: bool,
    /// Force calls.
    pub calls: f64,
}

/// Options for [`paired_gate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateOpts {
    /// One-sided significance level. The call claim needs the Wilcoxon
    /// tail and the bootstrap interval.
    pub alpha: f64,
    /// Bootstrap resamples of the paired median difference.
    pub bootstrap: usize,
    /// Seed for the splitmix64 resampler. The same seed repeats the interval.
    pub seed: u64,
}

impl Default for GateOpts {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            bootstrap: 999,
            seed: 0x00C0_FFEE,
        }
    }
}

/// What the call counts support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallClaim {
    /// Candidate used fewer calls.
    Fewer,
    /// Candidate used more calls.
    More,
    /// The samples do not separate the two series.
    Undecided,
}

/// Paired comparison of a candidate against a baseline.
#[derive(Clone, Debug, PartialEq)]
pub struct GateReport {
    /// Call-count claim on the trials both sides finished.
    pub claim: CallClaim,
    /// No new failure, and the call claim is not [`CallClaim::More`].
    pub passed: bool,
    /// Trials in the pair, including failures.
    pub n_pairs: usize,
    /// Trials where both sides finished with a finite non-negative call count.
    pub n_call: usize,
    /// Trials the baseline finished and the candidate did not.
    pub new_failures: usize,
    /// Median of `candidate.calls - baseline.calls` on [`Self::n_call`] pairs.
    /// NaN when `n_call` is 0.
    pub median_delta: f64,
    /// Lower end of the percentile bootstrap interval. NaN when `n_call` is 0.
    pub ci_low: f64,
    /// Upper end of the percentile bootstrap interval. NaN when `n_call` is 0.
    pub ci_high: f64,
    /// One-sided Wilcoxon probability of a candidate-minus-baseline
    /// signed-rank sum this small or smaller. 1 when there is no nonzero pair.
    pub p_fewer: f64,
    /// One-sided Wilcoxon probability of a signed-rank sum this large or larger.
    pub p_more: f64,
}

/// Moré-Wild convergence test.
///
/// Returns whether `f0 - f >= (1 - tau) * (f0 - f_low)`. `tau` must lie
/// in `(0, 1]`. A non-finite value, a lower bound above `f0`, or a `tau`
/// outside that interval is not a solved run.
pub fn more_wild_solved(f0: f64, f: f64, f_low: f64, tau: f64) -> bool {
    if !f0.is_finite() || !f.is_finite() || !f_low.is_finite() || !tau.is_finite() {
        return false;
    }
    if tau <= 0.0 || tau > 1.0 || f_low > f0 {
        return false;
    }
    let gap = f0 - f_low;
    f0 - f >= (1.0 - tau) * gap
}

/// Fraction of problems each solver solves within `ratio` times the best
/// finite call count on that problem.
///
/// `calls[solver][problem]` is `Some(count)` when that solver solved the
/// problem and `None` when it did not. A problem nobody solved stays in
/// the denominator and in nobody's numerator. A negative or non-finite
/// count is an unsolved run. `ratio < 1` solves nothing, because no
/// count is strictly better than the best by a factor below one. Tied
/// best counts both meet ratio 1.
pub fn performance_profile(calls: &[Vec<Option<f64>>], ratio: f64) -> Result<Vec<f64>, GateError> {
    let n_solver = calls.len();
    if n_solver == 0 {
        return Err(GateError::Shape);
    }
    let n_prob = calls[0].len();
    if calls.iter().any(|row| row.len() != n_prob) {
        return Err(GateError::Shape);
    }
    let mut solved = vec![0.0; n_solver];
    if n_prob == 0 || !ratio.is_finite() || ratio < 1.0 {
        return Ok(solved);
    }
    for problem in 0..n_prob {
        let mut best = f64::INFINITY;
        for row in calls {
            if let Some(count) = finite_calls(row[problem]) {
                best = best.min(count);
            }
        }
        if !best.is_finite() {
            continue;
        }
        let limit = ratio * best;
        for (solver, row) in calls.iter().enumerate() {
            if let Some(count) = finite_calls(row[problem]) {
                if count <= limit {
                    solved[solver] += 1.0;
                }
            }
        }
    }
    for count in &mut solved {
        *count /= n_prob as f64;
    }
    Ok(solved)
}

/// Fraction of problems each solver solves within `alpha * (n + 1)` calls.
///
/// `dim[problem]` is `n`, the number of coordinates. An unsolved entry
/// (`None`, negative, or non-finite) does not count. `alpha < 0` solves
/// nothing.
pub fn data_profile(
    calls: &[Vec<Option<f64>>],
    dim: &[usize],
    alpha: f64,
) -> Result<Vec<f64>, GateError> {
    let n_solver = calls.len();
    if n_solver == 0 {
        return Err(GateError::Shape);
    }
    let n_prob = calls[0].len();
    if calls.iter().any(|row| row.len() != n_prob) {
        return Err(GateError::Shape);
    }
    if dim.len() != n_prob {
        return Err(GateError::Dim {
            got: dim.len(),
            problems: n_prob,
        });
    }
    let mut solved = vec![0.0; n_solver];
    if n_prob == 0 || !alpha.is_finite() || alpha < 0.0 {
        return Ok(solved);
    }
    for problem in 0..n_prob {
        let budget = alpha * (dim[problem] as f64 + 1.0);
        for (solver, row) in calls.iter().enumerate() {
            if let Some(count) = finite_calls(row[problem]) {
                if count <= budget {
                    solved[solver] += 1.0;
                }
            }
        }
    }
    for count in &mut solved {
        *count /= n_prob as f64;
    }
    Ok(solved)
}

/// One-sided Wilcoxon tail probabilities for paired differences.
///
/// `delta[i]` is candidate minus baseline. Zeros and non-finite entries
/// are dropped. Ranks of the absolute values use the average rank on
/// ties. The returned pair is `(P(W+ <= observed), P(W+ >= observed))`
/// under the exact sign distribution when at most 16 nonzero pairs
/// remain, and under the normal approximation with the tie correction
/// and a continuity correction otherwise. No nonzero pairs returns `(1, 1)`.
pub fn wilcoxon_tails(delta: &[f64]) -> (f64, f64) {
    let values: Vec<f64> = delta
        .iter()
        .copied()
        .filter(|d| d.is_finite() && *d != 0.0)
        .collect();
    let n = values.len();
    if n == 0 {
        return (1.0, 1.0);
    }
    let (rank2, positive) = rank_twice(&values);
    let w_plus2: u64 = rank2
        .iter()
        .zip(&positive)
        .filter(|(_, pos)| **pos)
        .map(|(rank, _)| *rank)
        .sum();
    if n <= 16 {
        exact_tails(&rank2, w_plus2)
    } else {
        normal_tails(&values, w_plus2)
    }
}

/// Compare `candidate` with `baseline` on paired trials.
///
/// The series must have the same length. Call counts are compared only
/// on trials both sides finished (`converged`, finite, non-negative
/// calls). The median difference is candidate minus baseline. Fewer
/// calls requires the lower Wilcoxon tail below `opts.alpha` and a
/// bootstrap interval lying entirely below zero. More calls is the
/// symmetric upper statement. Otherwise the claim is undecided.
/// [`GateReport::passed`] is false when the claim is more calls or the
/// candidate fails a trial the baseline finished.
pub fn paired_gate(
    baseline: &[Trial],
    candidate: &[Trial],
    opts: &GateOpts,
) -> Result<GateReport, GateError> {
    if !(opts.alpha > 0.0 && opts.alpha < 1.0) {
        return Err(GateError::Alpha);
    }
    if opts.bootstrap == 0 {
        return Err(GateError::Bootstrap);
    }
    if baseline.len() != candidate.len() {
        return Err(GateError::Length);
    }
    if baseline.is_empty() {
        return Err(GateError::Empty);
    }
    let n_pairs = baseline.len();
    let mut new_failures = 0usize;
    let mut delta = Vec::new();
    for (base, cand) in baseline.iter().zip(candidate) {
        let base_ok = finished(base);
        let cand_ok = finished(cand);
        if base_ok && !cand_ok {
            new_failures += 1;
        }
        if base_ok && cand_ok {
            delta.push(cand.calls - base.calls);
        }
    }
    let n_call = delta.len();
    let (median_delta, ci_low, ci_high, p_fewer, p_more) = if n_call == 0 {
        (f64::NAN, f64::NAN, f64::NAN, 1.0, 1.0)
    } else {
        let (p_fewer, p_more) = wilcoxon_tails(&delta);
        let median_delta = median_of(&delta);
        let (ci_low, ci_high) = bootstrap_median_ci(&delta, opts);
        (median_delta, ci_low, ci_high, p_fewer, p_more)
    };
    let fewer = n_call > 0 && p_fewer < opts.alpha && ci_high < 0.0;
    let more = n_call > 0 && p_more < opts.alpha && ci_low > 0.0;
    let claim = if fewer && !more {
        CallClaim::Fewer
    } else if more && !fewer {
        CallClaim::More
    } else {
        CallClaim::Undecided
    };
    let passed = new_failures == 0 && claim != CallClaim::More;
    Ok(GateReport {
        claim,
        passed,
        n_pairs,
        n_call,
        new_failures,
        median_delta,
        ci_low,
        ci_high,
        p_fewer,
        p_more,
    })
}

/// Parse the atomistic bench CSV.
///
/// The header is optional. A data row is
/// `system,method,seed,converged,iters,calls` with further columns
/// ignored. `converged` is `0` or `1`. Blank lines are skipped.
pub fn parse_bench_csv(text: &str) -> Result<Vec<BenchRow>, GateError> {
    let mut rows = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if idx == 0 && line.starts_with("system,") {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 6 {
            return Err(GateError::Csv { line: idx + 1 });
        }
        let seed = fields[2]
            .parse::<u64>()
            .map_err(|_| GateError::Csv { line: idx + 1 })?;
        let converged = match fields[3] {
            "1" => true,
            "0" => false,
            _ => return Err(GateError::Csv { line: idx + 1 }),
        };
        let calls = fields[5]
            .parse::<f64>()
            .map_err(|_| GateError::Csv { line: idx + 1 })?;
        rows.push(BenchRow {
            system: fields[0].to_string(),
            method: fields[1].to_string(),
            seed,
            converged,
            calls,
        });
    }
    Ok(rows)
}

/// Pair two methods on one system by seed and run [`paired_gate`].
///
/// Every seed must appear once for each method. Seeds are sorted before
/// pairing.
pub fn gate_bench(
    rows: &[BenchRow],
    system: &str,
    baseline: &str,
    candidate: &str,
    opts: &GateOpts,
) -> Result<GateReport, GateError> {
    let mut base = Vec::new();
    let mut cand = Vec::new();
    for row in rows {
        if row.system != system {
            continue;
        }
        let trial = Trial {
            calls: row.calls,
            converged: row.converged,
        };
        if row.method == baseline {
            base.push((row.seed, trial));
        } else if row.method == candidate {
            cand.push((row.seed, trial));
        }
    }
    if base.is_empty() && cand.is_empty() {
        return Err(GateError::Empty);
    }
    pair_seeds(&mut base)?;
    pair_seeds(&mut cand)?;
    if base.len() != cand.len() || base.iter().zip(&cand).any(|(a, b)| a.0 != b.0) {
        let seed = unmatched_seed(&base, &cand);
        return Err(GateError::Pair { seed });
    }
    let baseline: Vec<Trial> = base.into_iter().map(|(_, trial)| trial).collect();
    let candidate: Vec<Trial> = cand.into_iter().map(|(_, trial)| trial).collect();
    paired_gate(&baseline, &candidate, opts)
}

fn finished(trial: &Trial) -> bool {
    trial.converged && trial.calls.is_finite() && trial.calls >= 0.0
}

fn finite_calls(count: Option<f64>) -> Option<f64> {
    count.filter(|count| count.is_finite() && *count >= 0.0)
}

/// Average ranks of absolute differences, stored as twice the rank so
/// ties stay integral. The boolean is the sign of the original difference.
fn rank_twice(values: &[f64]) -> (Vec<u64>, Vec<bool>) {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].abs().total_cmp(&values[b].abs()));
    let mut rank2 = vec![0u64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[order[j + 1]].abs() == values[order[i]].abs() {
            j += 1;
        }
        let twice = (i + j + 2) as u64;
        for slot in order.iter().take(j + 1).skip(i) {
            rank2[*slot] = twice;
        }
        i = j + 1;
    }
    let positive = values.iter().map(|value| *value > 0.0).collect();
    (rank2, positive)
}

fn exact_tails(rank2: &[u64], w_plus2: u64) -> (f64, f64) {
    let n = rank2.len();
    let total = 1u32 << n;
    let mut lower = 0u32;
    let mut upper = 0u32;
    for mask in 0..total {
        let mut w = 0u64;
        for (bit, rank) in rank2.iter().enumerate() {
            if (mask >> bit) & 1 == 1 {
                w += *rank;
            }
        }
        if w <= w_plus2 {
            lower += 1;
        }
        if w >= w_plus2 {
            upper += 1;
        }
    }
    let scale = f64::from(total);
    (f64::from(lower) / scale, f64::from(upper) / scale)
}

fn normal_tails(values: &[f64], w_plus2: u64) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = n * (n + 1.0) / 4.0;
    let mut var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0;
    let mut i = 0;
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].abs().total_cmp(&values[b].abs()));
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && values[order[j + 1]].abs() == values[order[i]].abs() {
            j += 1;
        }
        let t = (j - i + 1) as f64;
        var -= (t * t * t - t) / 48.0;
        i = j + 1;
    }
    let w = w_plus2 as f64 / 2.0;
    if var <= 0.0 {
        let p = if w == mean { 1.0 } else { 0.0 };
        return (p, p);
    }
    let sd = var.sqrt();
    let p_fewer = norm_cdf((w + 0.5 - mean) / sd);
    let p_more = 1.0 - norm_cdf((w - 0.5 - mean) / sd);
    (p_fewer.clamp(0.0, 1.0), p_more.clamp(0.0, 1.0))
}

fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Abramowitz and Stegun 7.1.26, absolute error about `1.5e-7`.
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn median_of(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

fn bootstrap_median_ci(delta: &[f64], opts: &GateOpts) -> (f64, f64) {
    let n = delta.len();
    let mut rng = SplitMix(opts.seed);
    let mut samples = Vec::with_capacity(opts.bootstrap);
    let mut draw = vec![0.0; n];
    for _ in 0..opts.bootstrap {
        for slot in &mut draw {
            *slot = delta[rng.index(n)];
        }
        samples.push(median_of(&draw));
    }
    samples.sort_by(f64::total_cmp);
    let low = percentile(&samples, opts.alpha / 2.0);
    let high = percentile(&samples, 1.0 - opts.alpha / 2.0);
    (low, high)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let x = p * (sorted.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    let weight = x - lo as f64;
    sorted[lo] * (1.0 - weight) + sorted[hi] * weight
}

/// splitmix64. The same generator seeds the atomistic bench perturbations.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn index(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

fn pair_seeds(rows: &mut [(u64, Trial)]) -> Result<(), GateError> {
    rows.sort_by_key(|(seed, _)| *seed);
    if let Some(window) = rows.windows(2).find(|window| window[0].0 == window[1].0) {
        return Err(GateError::Pair { seed: window[0].0 });
    }
    Ok(())
}

fn unmatched_seed(base: &[(u64, Trial)], cand: &[(u64, Trial)]) -> u64 {
    let mut i = 0;
    let mut j = 0;
    while i < base.len() && j < cand.len() {
        if base[i].0 == cand[j].0 {
            i += 1;
            j += 1;
        } else if base[i].0 < cand[j].0 {
            return base[i].0;
        } else {
            return cand[j].0;
        }
    }
    if i < base.len() { base[i].0 } else { cand[j].0 }
}
