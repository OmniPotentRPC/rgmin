# r2 mask-precond cost

Base `0df557c2355011419b4fa60ead390f300940b361` against candidate `839364e0ac2c33abe113841e1997f83f7babfe80`. Same machine, one core, serial. Arm order `A B B A`, five cycles, so each arm has 10 wall-clock repeats. Seeds 2000-2011. Budget 2000. A miss scores 2001. `fmax` 0.01. Rattle width 0.02. Atom cap 0.2.

`fire2-exp` is preconditioned FIRE 2.0 with the metric kick left off. `fire2-exp-kick` is that run on the base, and the same run on the candidate after `set_fire_metric_kick(true)`. `eon` is the capped eOn step. `fire2` is FIRE 2.0 with no metric. The metric neighbour list is `linkcell::pairs_open` 0.6.0. The repo energy is the pairwise Lennard-Jones in `tests/gpmin.rs` (`epsilon = 1`, `sigma = 1`, no cutoff). The rgpot energy is `rgpot::LJClusterPot` 3.7.0 constructed with its defaults.

The paired ratio uses the first A pass and the first B pass of each seed. The 95% interval is the 2.5 and 97.5 percentiles of 10000 medians drawn by numpy's default generator with seed 20261123. Convergence is a two-sided Fisher exact test. Evaluations are a two-sided Mann-Whitney U test. The A/A band compares the first two base passes. `IDENTICAL` means every repeat matches on evaluation count, iteration count, and energy bits. `PASS` means the candidate lost no convergence and the upper end of the interval is at most 1.10. Anything else is `FLAG`.

| cell | arm | base evals median | cand evals median | ratio | CI | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| repo/lj38 | fire2-exp | 84.5 | 84.5 | 1.0 | 1.0, 1.0 | IDENTICAL |
| repo/lj75 | fire2-exp | 85.5 | 85.5 | 1.0 | 1.0, 1.0 | IDENTICAL |
| repo/lj38 | fire2-exp-kick | 84.5 | 113.0 | 1.3416666666666668 | 1.293654990085922, 1.3717948717948718 | FLAG |
| repo/lj75 | fire2-exp-kick | 85.5 | 111.0 | 1.306599509234147 | 1.2675865800865802, 1.348449979694057 | FLAG |
| repo/lj38 | eon | 22.5 | 22.5 | 1.0 | 1.0, 1.0 | IDENTICAL |
| repo/lj75 | eon | 20.0 | 20.0 | 1.0 | 1.0, 1.0 | IDENTICAL |
| repo/lj38 | fire2 | 83.5 | 83.5 | 1.0 | 1.0, 1.0 | IDENTICAL |
| repo/lj75 | fire2 | 84.5 | 84.5 | 1.0 | 1.0, 1.0 | IDENTICAL |
| rgpot/lj38 | fire2-exp | 84.5 | 84.5 | 1.0 | 1.0, 1.0 | IDENTICAL |
| rgpot/lj75 | fire2-exp | 85.5 | 85.5 | 1.0 | 1.0, 1.0 | IDENTICAL |
| rgpot/lj38 | fire2-exp-kick | 84.5 | 113.0 | 1.3416666666666668 | 1.293654990085922, 1.3717948717948718 | FLAG |
| rgpot/lj75 | fire2-exp-kick | 85.5 | 111.0 | 1.306599509234147 | 1.2675865800865802, 1.348449979694057 | FLAG |
| rgpot/lj38 | eon | 22.5 | 22.5 | 1.0 | 1.0, 1.0 | IDENTICAL |
| rgpot/lj75 | eon | 20.0 | 20.0 | 1.0 | 1.0, 1.0 | IDENTICAL |
| rgpot/lj38 | fire2 | 83.5 | 83.5 | 1.0 | 1.0, 1.0 | IDENTICAL |
| rgpot/lj75 | fire2 | 84.5 | 84.5 | 1.0 | 1.0, 1.0 | IDENTICAL |

Wall-clock medians, Fisher and Mann-Whitney results, and the A/A band are in `summary.json`. Every cell converged 12/12. Repeats of one arm and seed agreed on evaluations, iterations, and energy bits. On the cells judged `IDENTICAL`, coordinate and trajectory hashes agree as well.

The develop checkout has no `set_fire_metric_kick`. `build-A-kick.log` exits 101. The candidate build with `--features metric-kick,rgpot` exits 0.

```
git merge-base --is-ancestor 0df557c2355011419b4fa60ead390f300940b361 839364e0ac2c33abe113841e1997f83f7babfe80
git worktree add --detach bench/mask-precond-cost/r2/arm-b 839364e0ac2c33abe113841e1997f83f7babfe80
RUSTUP_TOOLCHAIN=stable CC=gcc CXX=g++ \
  RGPOT_PREFIX=bench/mask-precond-cost/logs/rgpot-prefix \
  cargo build --release --manifest-path bench/mask-precond-cost/r2/runner/Cargo.toml --features rgpot
RUSTUP_TOOLCHAIN=stable CC=gcc CXX=g++ \
  RGPOT_PREFIX=bench/mask-precond-cost/logs/rgpot-prefix \
  cargo build --release --manifest-path bench/mask-precond-cost/r2/arm-b/bench/mask-precond-cost/r2/runner/Cargo.toml --features rgpot
RUSTUP_TOOLCHAIN=stable CC=gcc CXX=g++ \
  RGPOT_PREFIX=bench/mask-precond-cost/logs/rgpot-prefix \
  cargo build --release --manifest-path bench/mask-precond-cost/r2/arm-b/bench/mask-precond-cost/r2/runner/Cargo.toml --features metric-kick,rgpot
RUSTUP_TOOLCHAIN=stable CC=gcc CXX=g++ \
  RGPOT_PREFIX=bench/mask-precond-cost/logs/rgpot-prefix \
  python3 bench/mask-precond-cost/r2/drive.py
python3 bench/mask-precond-cost/r2/analyze.py \
  --rows bench/mask-precond-cost/r2/rows.json \
  --summary bench/mask-precond-cost/r2/summary.json \
  --budget 2000 \
  --protocol bench/mask-precond-cost/r2/logs/protocol.json
```
