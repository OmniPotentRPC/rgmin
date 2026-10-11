//! Strong scaling of Exp and Lindh on bulk diamond silicon.
//!
//! Ignored in the default suite. Release, with the parallel feature:
//!
//! ```text
//! cargo test --release --features par --test precon_scaling -- --ignored --nocapture
//! ```
//!
//! Sizes are 160, 2,000, and 32,000 atoms. A 100,000-atom Lindh matrix
//! reached about 20 GB and one repetition of the eight solves did not
//! finish, so that size is not in the table. Threads are 1, 2, 4, and
//! the machine maximum. Five repetitions
//! are interleaved. Assembly, apply, and solve are timed separately.
//! The parallel path inside the crate stays serial below [`rgmin::N_PAR`].

#![cfg(feature = "par")]

use std::time::Instant;

use ndarray::Array1;
use rgmin::{Exp, ExpParams, Lindh, LindhParams, Neighbours};

const A_SI: f64 = 5.431;
const CUTOFF: f64 = 4.0;
const REPS: usize = 5;
const APPLIES: usize = 8;
const SOLVES: usize = 8;

const BASIS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.25],
    [0.75, 0.25, 0.75],
    [0.25, 0.75, 0.75],
];

struct Crystal {
    pos: Array1<f64>,
    z: Vec<i32>,
    lengths: [f64; 3],
    neigh: Neighbours,
}

fn diamond(nx: usize, ny: usize, nz: usize) -> Crystal {
    let n = 8 * nx * ny * nz;
    let mut pos = Array1::zeros(3 * n);
    let mut k = 0;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                for b in BASIS {
                    pos[k] = (ix as f64 + b[0]) * A_SI;
                    pos[k + 1] = (iy as f64 + b[1]) * A_SI;
                    pos[k + 2] = (iz as f64 + b[2]) * A_SI;
                    k += 3;
                }
            }
        }
    }
    let lengths = [nx as f64 * A_SI, ny as f64 * A_SI, nz as f64 * A_SI];
    let neigh = neighbours(&pos, lengths);
    Crystal {
        pos,
        z: vec![14; n],
        lengths,
        neigh,
    }
}

fn neighbours(pos: &Array1<f64>, lengths: [f64; 3]) -> Neighbours {
    let n = pos.len() / 3;
    let coords: Vec<[f64; 3]> = (0..n)
        .map(|i| [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]])
        .collect();
    let pairs = cell_pairs(&coords, CUTOFF, lengths);
    let mut neigh = Neighbours::default();
    for (i, j, r) in pairs {
        neigh.i.push(i);
        neigh.j.push(j);
        neigh.r.push(r);
    }
    neigh
}

fn cell_pairs(pos: &[[f64; 3]], cutoff: f64, lengths: [f64; 3]) -> Vec<(usize, usize, f64)> {
    let n = pos.len();
    let origin = [0.0, 0.0, 0.0];
    let nbin = [
        (lengths[0] / cutoff).floor().max(1.0) as usize,
        (lengths[1] / cutoff).floor().max(1.0) as usize,
        (lengths[2] / cutoff).floor().max(1.0) as usize,
    ];
    let mut cells: std::collections::BTreeMap<(i32, i32, i32), Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        let key = bin_of(pos[i], origin, cutoff, nbin, lengths);
        cells.entry(key).or_default().push(i);
    }
    let mut pairs = Vec::new();
    for (&key, atoms) in &cells {
        let mut seen = Vec::new();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nb = wrap(key, [dx, dy, dz], nbin);
                    if seen.contains(&nb) {
                        continue;
                    }
                    seen.push(nb);
                    let Some(others) = cells.get(&nb) else {
                        continue;
                    };
                    for &i in atoms {
                        for &j in others {
                            if j <= i {
                                continue;
                            }
                            let mut d = [
                                pos[i][0] - pos[j][0],
                                pos[i][1] - pos[j][1],
                                pos[i][2] - pos[j][2],
                            ];
                            for k in 0..3 {
                                d[k] -= lengths[k] * (d[k] / lengths[k]).round();
                            }
                            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                            if r > 1e-8 && r < cutoff {
                                pairs.push((i, j, r));
                            }
                        }
                    }
                }
            }
        }
    }
    pairs.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    pairs
}

fn bin_of(
    p: [f64; 3],
    origin: [f64; 3],
    cutoff: f64,
    nbin: [usize; 3],
    lengths: [f64; 3],
) -> (i32, i32, i32) {
    let _ = (origin, cutoff);
    let mut key = [0i32; 3];
    for k in 0..3 {
        let n = nbin[k] as i32;
        let mut f = p[k] / lengths[k];
        f -= f.floor();
        if f >= 1.0 {
            f = 0.0;
        }
        let idx = (f * n as f64).floor() as i32;
        key[k] = idx.clamp(0, n - 1);
    }
    (key[0], key[1], key[2])
}

fn wrap(key: (i32, i32, i32), delta: [i32; 3], nbin: [usize; 3]) -> (i32, i32, i32) {
    let cur = [key.0, key.1, key.2];
    let mut out = [0i32; 3];
    for k in 0..3 {
        let n = nbin[k] as i32;
        let mut w = (cur[k] + delta[k]) % n;
        if w < 0 {
            w += n;
        }
        out[k] = w;
    }
    (out[0], out[1], out[2])
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

struct Sample {
    assemble: f64,
    apply: f64,
    solve: f64,
}

fn once_lindh(threads: usize, crystal: &Crystal, rhs: &Array1<f64>) -> Sample {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let mut params = LindhParams::default();
        params.mu = Some(1.0);
        params.r_cut = Some(CUTOFF);
        params.lengths = Some(crystal.lengths);
        let t0 = Instant::now();
        let metric = Lindh::assemble(crystal.pos.view(), &crystal.z, &[], &params).unwrap();
        let assemble = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for _ in 0..APPLIES {
            std::hint::black_box(metric.apply(rhs.view()));
        }
        let apply = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        for _ in 0..SOLVES {
            let solved = metric.solve(rhs.view()).expect("lindh pcg");
            std::hint::black_box(&solved);
        }
        let solve = t2.elapsed().as_secs_f64();
        Sample {
            assemble,
            apply,
            solve,
        }
    })
}

fn once_exp(threads: usize, crystal: &Crystal, rhs: &Array1<f64>) -> Sample {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let mut params = ExpParams::default();
        params.mu = Some(1.0);
        params.r_cut = Some(CUTOFF);
        params.r_nn = Some(2.35);
        params.lengths = Some(crystal.lengths);
        let t0 = Instant::now();
        let metric = Exp::assemble(crystal.pos.view(), &crystal.neigh, &params, 1.0, 2.35).unwrap();
        let assemble = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for _ in 0..APPLIES {
            std::hint::black_box(metric.apply(rhs.view()));
        }
        let apply = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        for _ in 0..SOLVES {
            let solved = metric.solve(rhs.view()).expect("exp pcg");
            std::hint::black_box(&solved);
        }
        let solve = t2.elapsed().as_secs_f64();
        Sample {
            assemble,
            apply,
            solve,
        }
    })
}

fn report(label: &str, natoms: usize, threads: &[usize], rows: &[Vec<Sample>]) {
    eprintln!(
        "{label},natoms={natoms},threads,assemble_median_s,assemble_min_s,assemble_max_s,assemble_eff,apply_median_s,apply_min_s,apply_max_s,apply_eff,solve_median_s,solve_min_s,solve_max_s,solve_eff"
    );
    let mut a1s = rows[0].iter().map(|s| s.assemble).collect::<Vec<_>>();
    let mut p1s = rows[0].iter().map(|s| s.apply).collect::<Vec<_>>();
    let mut s1s = rows[0].iter().map(|s| s.solve).collect::<Vec<_>>();
    let a1 = median(&mut a1s);
    let p1 = median(&mut p1s);
    let s1 = median(&mut s1s);
    for (k, &n) in threads.iter().enumerate() {
        let mut a = rows[k].iter().map(|s| s.assemble).collect::<Vec<_>>();
        let mut p = rows[k].iter().map(|s| s.apply).collect::<Vec<_>>();
        let mut s = rows[k].iter().map(|s| s.solve).collect::<Vec<_>>();
        let am = median(&mut a);
        let pm = median(&mut p);
        let sm = median(&mut s);
        let amin = a.iter().copied().fold(f64::INFINITY, f64::min);
        let amax = a.iter().copied().fold(0.0, f64::max);
        let pmin = p.iter().copied().fold(f64::INFINITY, f64::min);
        let pmax = p.iter().copied().fold(0.0, f64::max);
        let smin = s.iter().copied().fold(f64::INFINITY, f64::min);
        let smax = s.iter().copied().fold(0.0, f64::max);
        eprintln!(
            "{label},{natoms},{n},{am:.6},{amin:.6},{amax:.6},{:.3},{pm:.6},{pmin:.6},{pmax:.6},{:.3},{sm:.6},{smin:.6},{smax:.6},{:.3}",
            a1 / (n as f64 * am),
            p1 / (n as f64 * pm),
            s1 / (n as f64 * sm),
        );
    }
    if rows.len() >= 2 {
        let a = rows[0].iter().map(|s| s.assemble).collect::<Vec<_>>();
        let b = rows[1].iter().map(|s| s.assemble).collect::<Vec<_>>();
        let a_med = median(&mut a.clone());
        let b_med = median(&mut b.clone());
        let a_range =
            a.iter().copied().fold(0.0, f64::max) - a.iter().copied().fold(f64::INFINITY, f64::min);
        let b_range =
            b.iter().copied().fold(0.0, f64::max) - b.iter().copied().fold(f64::INFINITY, f64::min);
        let margin = a_med - b_med;
        let noise = a_range.max(b_range);
        eprintln!(
            "{label},natoms={natoms},two_vs_one_assemble,median_gap_s={margin:.6},range_s={noise:.6},beats={}",
            margin > noise
        );
    }
}

fn time_set(
    threads: &[usize],
    crystal: &Crystal,
    rhs: &Array1<f64>,
    exp: bool,
) -> Vec<Vec<Sample>> {
    let _ = if exp {
        once_exp(1, crystal, rhs)
    } else {
        once_lindh(1, crystal, rhs)
    };
    let mut rows = (0..threads.len()).map(|_| Vec::new()).collect::<Vec<_>>();
    for _rep in 0..REPS {
        for (k, &n) in threads.iter().enumerate() {
            let sample = if exp {
                once_exp(n, crystal, rhs)
            } else {
                once_lindh(n, crystal, rhs)
            };
            rows[k].push(sample);
        }
    }
    rows
}

#[test]
#[ignore]
fn strong_scaling_table() {
    let max = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let mut threads = vec![1usize, 2, 4];
    if !threads.contains(&max) {
        threads.push(max);
    }
    threads.retain(|&n| n <= max);
    if threads.is_empty() {
        threads.push(1);
    }
    let shapes = [
        (2usize, 2, 5, "160"),
        (5, 5, 10, "2000"),
        (20, 20, 10, "32000"),
    ];
    for (nx, ny, nz, name) in shapes {
        let natoms = 8 * nx * ny * nz;
        eprintln!("building {name} ({natoms} atoms)");
        let crystal = diamond(nx, ny, nz);
        let rhs = Array1::from_iter((0..crystal.pos.len()).map(|i| ((i % 7) as f64 - 3.0) * 0.05));
        let lindh = time_set(&threads, &crystal, &rhs, false);
        report(&format!("lindh-{name}"), natoms, &threads, &lindh);
        let exp = time_set(&threads, &crystal, &rhs, true);
        report(&format!("exp-{name}"), natoms, &threads, &exp);
    }
}
