//! Criterion-style smoke benchmark for geometry preconditioners (`Exp` and `Lindh`).
//!
//! Covers the Makri, Ortner, and Kermode (2019) Cu vacancy FCC supercells
//! (`3x3x3` N=107, `4x4x4` N=255, `6x6x6` N=863, `r0 = 2.55 A`, `r_cut = 5.3 A`),
//! a `8x8x8` (N=2047) Cu vacancy crystal for single-system thread scaling,
//! and a 16-interior-image `6x6x6` NEB band batch smoke row.
//! `cargo bench --bench precon --features par`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::Array1;
use rgmin::{Exp, ExpParams, Lindh, LindhParams, Neighbours};

struct CountingAlloc;

thread_local! {
    static TRACK_THREAD: Cell<bool> = const { Cell::new(false) };
}

static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

struct TrackScope;

impl TrackScope {
    fn enter() -> Self {
        TRACK_THREAD.with(|flag| flag.set(true));
        Self
    }
}

impl Drop for TrackScope {
    fn drop(&mut self) {
        TRACK_THREAD.with(|flag| flag.set(false));
    }
}

#[inline]
fn is_tracked() -> bool {
    matches!(TRACK_THREAD.try_with(Cell::get), Ok(true))
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && is_tracked() {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && is_tracked() {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        if !out.is_null() && is_tracked() {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            let delta = new_size as isize - layout.size() as isize;
            LIVE_BYTES.fetch_add(delta, Ordering::Relaxed);
        }
        out
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        if is_tracked() {
            LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

const R0_CU: f64 = 2.55;
const RCUT_CU: f64 = 5.3;
const WARMUP: usize = 2;
const SAMPLES: usize = 7;
const WARM_STEPS: usize = 8;

const FCC_BASIS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
];

struct VacancyCell {
    natoms: usize,
    lengths: [f64; 3],
    pos: Array1<f64>,
    neigh: Neighbours,
    rhs: Array1<f64>,
}

fn cu_vacancy(cells: usize) -> VacancyCell {
    let a = R0_CU * std::f64::consts::SQRT_2;
    let l = cells as f64 * a;
    let lengths = [l, l, l];
    let mut coords = Vec::with_capacity(4 * cells * cells * cells);
    let mut skipped = false;
    for ix in 0..cells {
        for iy in 0..cells {
            for iz in 0..cells {
                for b in FCC_BASIS {
                    if !skipped {
                        skipped = true;
                        continue;
                    }
                    let idx = coords.len();
                    let jitter = 0.015 * ((idx * 13 + 7) % 19) as f64 / 19.0;
                    coords.push([
                        (ix as f64 + b[0]) * a + jitter,
                        (iy as f64 + b[1]) * a - 0.7 * jitter,
                        (iz as f64 + b[2]) * a + 0.4 * jitter,
                    ]);
                }
            }
        }
    }
    let natoms = coords.len();
    let mut pos = Array1::zeros(3 * natoms);
    for (i, p) in coords.iter().enumerate() {
        pos[3 * i] = p[0];
        pos[3 * i + 1] = p[1];
        pos[3 * i + 2] = p[2];
    }
    let mut neigh = Neighbours::default();
    for i in 0..natoms {
        for j in (i + 1)..natoms {
            let mut d2 = 0.0;
            for k in 0..3 {
                let mut d = coords[i][k] - coords[j][k];
                d -= lengths[k] * (d / lengths[k]).round();
                d2 += d * d;
            }
            let r = d2.sqrt();
            if r < RCUT_CU {
                neigh.i.push(i);
                neigh.j.push(j);
                neigh.r.push(r);
            }
        }
    }
    let rhs = Array1::from_iter((0..3 * natoms).map(|k| {
        let s = ((k * 31 + 11) % 97) as f64 / 97.0 - 0.5;
        let wave = ((k / 3) as f64 * 0.17).sin();
        0.25 * s + 0.15 * wave
    }));
    VacancyCell {
        natoms,
        lengths,
        pos,
        neigh,
        rhs,
    }
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

fn bench_us(mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..WARMUP {
        f();
    }
    let mut us = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        f();
        us.push(t0.elapsed().as_secs_f64() * 1.0e6);
    }
    us.sort_by(f64::total_cmp);
    let med = quantile(&us, 0.5);
    let iqr = quantile(&us, 0.75) - quantile(&us, 0.25);
    (med, iqr)
}

fn main() {
    let _track = TrackScope::enter();
    println!(
        "| cell | N | pairs | CSR KiB | held KiB | 62-img MiB | assemble us (IQR) | apply us (IQR) | cold solve us (IQR) | cold iters | warm8 solve us/step (IQR) | solve allocs | dot allocs |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");

    for cells in [3usize, 4, 6] {
        let sys = cu_vacancy(cells);
        let params = ExpParams::default()
            .with_mu(1.0)
            .with_r_nn(R0_CU)
            .with_lengths(sys.lengths);

        let live_before = LIVE_BYTES.load(Ordering::Relaxed);
        let held = Exp::assemble(sys.pos.view(), &sys.neigh, &params, 1.0, R0_CU).expect("exp");
        let live_after = LIVE_BYTES.load(Ordering::Relaxed);
        let held_bytes = (live_after - live_before).max(0) as usize;
        let csr_kib = held.csr_bytes() as f64 / 1024.0;
        let held_kib = held_bytes as f64 / 1024.0;
        let band62_mib = (held_bytes as f64 * 62.0) / (1024.0 * 1024.0);

        let cold_iters = held.solve_iterations(sys.rhs.view()).expect("iters");

        let probe = Exp::assemble(sys.pos.view(), &sys.neigh, &params, 1.0, R0_CU).expect("exp");
        let _ = probe.solve(sys.rhs.view());
        let rhs2 = sys.rhs.mapv(|v| v * 0.97 + 1.0e-3);
        let alloc0 = ALLOC_CALLS.load(Ordering::Relaxed);
        let sol = probe.solve(rhs2.view()).expect("solve");
        std::hint::black_box(&sol);
        let solve_allocs = ALLOC_CALLS.load(Ordering::Relaxed).saturating_sub(alloc0);

        let slice_a = sys.rhs.as_slice().expect("contiguous");
        let slice_b = rhs2.as_slice().expect("contiguous");
        let dot_alloc0 = ALLOC_CALLS.load(Ordering::Relaxed);
        let dval = rgmin::precon::dot(slice_a, slice_b);
        std::hint::black_box(dval);
        let dot_allocs = ALLOC_CALLS
            .load(Ordering::Relaxed)
            .saturating_sub(dot_alloc0);

        let (asm_med, asm_iqr) = bench_us(|| {
            let m = Exp::assemble(sys.pos.view(), &sys.neigh, &params, 1.0, R0_CU).expect("exp");
            std::hint::black_box(&m);
        });

        let (app_med, app_iqr) = bench_us(|| {
            let y = held.apply(sys.rhs.view());
            std::hint::black_box(&y);
        });

        let (cold_med, cold_iqr) = {
            let fresh: Vec<Exp> = (0..(WARMUP + SAMPLES))
                .map(|_| {
                    Exp::assemble(sys.pos.view(), &sys.neigh, &params, 1.0, R0_CU).expect("exp")
                })
                .collect();
            let mut idx = 0usize;
            bench_us(|| {
                let s = fresh[idx].solve(sys.rhs.view()).expect("solve");
                idx += 1;
                std::hint::black_box(&s);
            })
        };

        let rhs_seq: Vec<Array1<f64>> = (0..WARM_STEPS)
            .map(|step| {
                let decay = 1.0 - 0.04 * (step as f64);
                Array1::from_iter(
                    sys.rhs
                        .iter()
                        .enumerate()
                        .map(|(k, &v)| decay * v + 0.005 * ((k + step) as f64 * 0.31).cos()),
                )
            })
            .collect();
        let (warm_total_med, warm_total_iqr) = bench_us(|| {
            let m = Exp::assemble(sys.pos.view(), &sys.neigh, &params, 1.0, R0_CU).expect("exp");
            for r in &rhs_seq {
                let s = m.solve(r.view()).expect("warm solve");
                std::hint::black_box(&s);
            }
        });
        let warm_step_med = (warm_total_med - asm_med).max(0.0) / (WARM_STEPS as f64);
        let warm_step_iqr = warm_total_iqr / (WARM_STEPS as f64);

        println!(
            "| {cells}x{cells}x{cells} | {} | {} | {:.1} | {:.1} | {:.2} | {:.1} ({:.1}) | {:.2} ({:.2}) | {:.1} ({:.1}) | {} | {:.1} ({:.1}) | {} | {} |",
            sys.natoms,
            sys.neigh.i.len(),
            csr_kib,
            held_kib,
            band62_mib,
            asm_med,
            asm_iqr,
            app_med,
            app_iqr,
            cold_med,
            cold_iqr,
            cold_iters,
            warm_step_med,
            warm_step_iqr,
            solve_allocs,
            dot_allocs,
        );
    }

    // Lindh small smoke row (covalent Si 2x2x2 supercell, 64 atoms)
    {
        let a_si = 5.431;
        let mut pos = Vec::new();
        for ix in 0..2 {
            for iy in 0..2 {
                for iz in 0..2 {
                    for b in [
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.0],
                        [0.5, 0.0, 0.5],
                        [0.0, 0.5, 0.5],
                        [0.25, 0.25, 0.25],
                        [0.75, 0.75, 0.25],
                        [0.75, 0.25, 0.75],
                        [0.25, 0.75, 0.75],
                    ] {
                        pos.push((ix as f64 + b[0]) * a_si);
                        pos.push((iy as f64 + b[1]) * a_si);
                        pos.push((iz as f64 + b[2]) * a_si);
                    }
                }
            }
        }
        let pos = Array1::from_vec(pos);
        let z = vec![14i32; 64];
        let lp = LindhParams::default()
            .with_mu(1.0)
            .with_r_cut(4.0)
            .with_lengths([2.0 * a_si; 3]);
        let lindh = Lindh::assemble(pos.view(), &z, &[], &lp).expect("lindh");
        let rhs = Array1::from_iter((0..pos.len()).map(|i| ((i % 7) as f64 - 3.0) * 0.05));
        let iters = lindh.solve_iterations(rhs.view()).expect("iters");
        let (sol_med, sol_iqr) = bench_us(|| {
            let s = lindh.solve(rhs.view()).expect("lindh solve");
            std::hint::black_box(&s);
        });
        println!();
        println!("Lindh Si64 smoke: iters={iters}, solve_us={sol_med:.1} (IQR {sol_iqr:.1})");
    }

    #[cfg(feature = "par")]
    {
        let sys8 = cu_vacancy(8); // N = 2047
        let params = ExpParams::default()
            .with_mu(1.0)
            .with_r_nn(R0_CU)
            .with_lengths(sys8.lengths);
        println!();
        println!("| Cu 8x8x8 (N=2047) threads | assemble+4xsolve us (IQR) | speedup |");
        println!("|---:|---:|---:|");
        let mut t1_med = 0.0;
        for threads in [1usize, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let (med, iqr) = pool.install(|| {
                bench_us(|| {
                    let m = Exp::assemble(sys8.pos.view(), &sys8.neigh, &params, 1.0, R0_CU)
                        .expect("exp");
                    for s in 0..4 {
                        let rhs = sys8.rhs.mapv(|v| v * (1.0 - 0.03 * s as f64));
                        let out = m.solve(rhs.view()).expect("solve");
                        std::hint::black_box(&out);
                    }
                })
            });
            if threads == 1 {
                t1_med = med;
            }
            println!("| {threads} | {med:.1} ({iqr:.1}) | {:.2}x |", t1_med / med);
        }

        // 16-interior-image 6x6x6 NEB band batch assemble + solve scaling
        use ndarray::ArrayView1;
        let sys6 = cu_vacancy(6);
        let imgs_pos: Vec<Array1<f64>> = (0..16)
            .map(|m| sys6.pos.mapv(|v| v + 0.002 * m as f64))
            .collect();
        let imgs_rhs: Vec<Array1<f64>> = (0..16)
            .map(|m| sys6.rhs.mapv(|v| (1.0 - 0.02 * m as f64) * v))
            .collect();
        let batch_in: Vec<(ArrayView1<f64>, &Neighbours)> =
            imgs_pos.iter().map(|p| (p.view(), &sys6.neigh)).collect();
        let rhs_views: Vec<ArrayView1<f64>> = imgs_rhs.iter().map(|r| r.view()).collect();

        println!();
        println!(
            "| Cu 6x6x6 x 16-img batch threads | assemble_batch+solve_batch us (IQR) | speedup |"
        );
        println!("|---:|---:|---:|");
        let mut b1_med = 0.0;
        for threads in [1usize, 4, 8] {
            let bparams = ExpParams::default()
                .with_mu(1.0)
                .with_r_nn(R0_CU)
                .with_lengths(sys6.lengths)
                .with_threads(threads);
            let (med, iqr) = bench_us(|| {
                let metrics =
                    Exp::assemble_batch(&batch_in, &bparams, 1.0, R0_CU).expect("batch asm");
                let sols = Exp::solve_batch(&metrics, &rhs_views, Some(threads))
                    .expect("batch ok")
                    .expect("batch spd");
                std::hint::black_box(&sols);
            });
            if threads == 1 {
                b1_med = med;
            }
            println!("| {threads} | {med:.1} ({iqr:.1}) | {:.2}x |", b1_med / med);
        }
    }
}
