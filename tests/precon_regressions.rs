//! Regression guards for preconditioner scaling:
//! - scalar-graph CSR memory size (3x nonzero reduction, ~3-4x byte reduction)
//! - zero heap allocations per `precon::dot` (serial and multi-threaded)
//! - bit-identical serial results and default capped eOn L-BFGS step
//! - multi-threaded assembly, PCG, session, and batch parity with serial

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::mem::size_of;
use std::sync::Mutex;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};
use rgmin::{
    BondTerm, Control, EON_MAX_MOVE, Exp, ExpParams, ExpPcKind, Ff, FfParams, FfTerms, Lbfgs,
    LbfgsStep, Lindh, LindhParams, Method, Neighbours, Oracle, Solver,
};

struct CountingAlloc;

thread_local! {
    static TRACK_THREAD: Cell<bool> = const { Cell::new(false) };
}

static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

/// Serialize tests that toggle thread tracking on shared allocator counters.
static ALLOC_LOCK: Mutex<()> = Mutex::new(());

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

/// Stated tolerance for multi-threaded vs serial preconditioner parity.
const THREAD_PARITY_ATOL: f64 = 1.0e-13;
const THREAD_PARITY_RTOL: f64 = 1.0e-13;

const FCC_BASIS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
];

struct VacancyFixture {
    natoms: usize,
    lengths: [f64; 3],
    pos: Array1<f64>,
    neigh: Neighbours,
    rhs: Array1<f64>,
}

fn cu_vacancy(cells: usize) -> VacancyFixture {
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
    VacancyFixture {
        natoms,
        lengths,
        pos,
        neigh,
        rhs,
    }
}

fn fnv1a_words(xs: &[f64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &v in xs {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn assert_within_tol(got: &Array1<f64>, reference: &Array1<f64>, label: &str) {
    assert_eq!(got.len(), reference.len(), "{label}: length mismatch");
    for (idx, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
        let diff = (g - r).abs();
        let bound = THREAD_PARITY_ATOL + THREAD_PARITY_RTOL * r.abs();
        assert!(
            diff <= bound,
            "{label}[{idx}]: got {g}, ref {r}, diff {diff} > bound {bound}"
        );
    }
}

#[test]
fn csr_memory_size_stores_the_scalar_graph_once() {
    let _guard = ALLOC_LOCK.lock().expect("alloc lock");
    let _track = TrackScope::enter();
    for (cells, expected_atoms, expected_pairs) in [(3usize, 107usize, 2862usize), (4, 255, 6858), (6, 863, 23274)] {
        let fix = cu_vacancy(cells);
        assert_eq!(fix.natoms, expected_atoms);
        assert_eq!(fix.neigh.i.len(), expected_pairs);

        let params = ExpParams::default()
            .with_mu(1.0)
            .with_r_nn(R0_CU)
            .with_lengths(fix.lengths);

        let live_before = LIVE_BYTES.load(Ordering::SeqCst);
        let exp = Exp::assemble(fix.pos.view(), &fix.neigh, &params, 1.0, R0_CU).expect("assemble");
        let live_after = LIVE_BYTES.load(Ordering::SeqCst);
        let held_bytes = (live_after - live_before).max(0) as usize;

        let scalar_nnz = 2 * expected_pairs + expected_atoms;
        let legacy_expanded_nnz = 3 * scalar_nnz;
        assert_eq!(
            exp.stored_nonzeros(),
            scalar_nnz,
            "cells={cells}: stored nonzeros must equal scalar graph (1/3 of 3x expansion)"
        );
        assert_eq!(legacy_expanded_nnz, 3 * exp.stored_nonzeros());

        // Legacy 3N-expanded CSR held row_ptr[3N+1], col_idx[3*nnz], val[3*nnz], diag[3N].
        let legacy_csr_bytes = (3 * expected_atoms + 1) * size_of::<usize>()
            + legacy_expanded_nnz * (size_of::<usize>() + size_of::<f64>())
            + (3 * expected_atoms) * size_of::<f64>();
        let csr_bytes = exp.csr_bytes();
        assert!(
            3 * csr_bytes <= legacy_csr_bytes,
            "cells={cells}: csr_bytes {csr_bytes} did not drop >=3x vs legacy {legacy_csr_bytes}"
        );
        // Total live heap bytes (CSR + BlockIc0 factor + Metric handle) must also stay
        // well below half of the legacy CSR alone (~3x reduction at 6x6x6).
        assert!(
            held_bytes * 5 <= legacy_csr_bytes * 2,
            "cells={cells}: total held bytes {held_bytes} exceeded 0.40 * legacy {legacy_csr_bytes}"
        );
        std::hint::black_box(&exp);
    }
}

#[test]
fn precon_dot_and_steady_pcg_do_not_allocate_scratch_vecs() {
    let _guard = ALLOC_LOCK.lock().expect("alloc lock");
    let _track = TrackScope::enter();
    for len in [0usize, 17, 1024, 1536, 2589, 20_000] {
        let a: Vec<f64> = (0..len).map(|i| ((i * 19 + 3) % 29) as f64 * 0.1 - 1.4).collect();
        let b: Vec<f64> = (0..len).map(|i| ((i * 7 + 11) % 23) as f64 * 0.1 - 1.1).collect();

        let calls_before = ALLOC_CALLS.load(Ordering::SeqCst);
        let mut s = 0.0;
        for _ in 0..16 {
            s += rgmin::precon::dot(&a, &b);
        }
        let calls_after = ALLOC_CALLS.load(Ordering::SeqCst);
        std::hint::black_box(s);
        assert_eq!(
            calls_after - calls_before,
            0,
            "serial precon::dot allocated on len={len}"
        );

        #[cfg(feature = "par")]
        if len >= 4096 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .start_handler(|_| TRACK_THREAD.with(|flag| flag.set(true)))
                .build()
                .expect("rayon pool");
            let warmup = pool.install(|| rgmin::precon::dot(&a, &b));
            std::hint::black_box(warmup);

            let par_before = ALLOC_CALLS.load(Ordering::SeqCst);
            let par_val = pool.install(|| {
                let mut total = 0.0;
                for _ in 0..16 {
                    total += rgmin::precon::dot(&a, &b);
                }
                total
            });
            let par_after = ALLOC_CALLS.load(Ordering::SeqCst);
            assert_eq!(
                par_after - par_before,
                0,
                "parallel precon::dot allocated on len={len}"
            );
            assert_eq!(par_val.to_bits(), s.to_bits(), "parallel vs serial dot bits");
        }
    }

    // Verify steady-state Exp::solve allocates only the returned Array1 (1 allocation),
    // with 0 internal PCG workspace or dot Vec allocations.
    let fix = cu_vacancy(3);
    let params = ExpParams::default()
        .with_mu(1.0)
        .with_r_nn(R0_CU)
        .with_lengths(fix.lengths);
    let exp = Exp::assemble(fix.pos.view(), &fix.neigh, &params, 1.0, R0_CU).expect("exp");
    let first = exp.solve(fix.rhs.view()).expect("first solve");
    std::hint::black_box(&first);
    let rhs_next = fix.rhs.mapv(|v| 0.96 * v + 1.0e-3);

    let before_solve = ALLOC_CALLS.load(Ordering::SeqCst);
    let second = exp.solve(rhs_next.view()).expect("second solve");
    let after_solve = ALLOC_CALLS.load(Ordering::SeqCst);
    std::hint::black_box(&second);
    assert_eq!(
        after_solve - before_solve,
        1,
        "expected only 1 output Array1 allocation on warm solve, got {}",
        after_solve - before_solve
    );
}

#[test]
fn serial_results_are_bit_identical_and_default_lbfgs_is_capped_eon() {
    let _guard = ALLOC_LOCK.lock().expect("alloc lock");
    let lbfgs = Lbfgs::default();
    assert!(
        matches!(lbfgs.step, LbfgsStep::Capped { max_atom, .. } if max_atom == EON_MAX_MOVE),
        "default L-BFGS step must remain the capped eOn step, got {:?}",
        lbfgs.step
    );

    let fix = cu_vacancy(3);
    let run_serial = |explicit_one: bool| -> Vec<u64> {
        let mut params = ExpParams::default()
            .with_mu(1.25)
            .with_r_nn(R0_CU)
            .with_lengths(fix.lengths);
        if explicit_one {
            params = params.with_threads(1);
        }
        let exp = Exp::assemble(fix.pos.view(), &fix.neigh, &params, 1.25, R0_CU).expect("exp");
        let applied = exp.apply(fix.rhs.view());
        let solved1 = exp.solve(fix.rhs.view()).expect("solve 1");
        let rhs2 = fix.rhs.mapv(|v| 0.92 * v - 0.003);
        let solved2 = exp.solve(rhs2.view()).expect("solve 2");

        // Preconditioned capped-step L-BFGS session on the quadratic metric
        let p_eval = exp.clone();
        let centre = fix.pos.clone();
        let obj = Oracle::unbounded(fix.pos.len(), move |x: ArrayView1<f64>| {
            let d = &x - &centre;
            let g = p_eval.apply(d.view());
            (0.5 * d.dot(&g), g)
        });
        let mut x = &fix.pos + &fix.rhs.mapv(|v| 0.05 * v);
        let control = Control::default().with_maxiter(1).with_gtol(1.0e-10);
        let mut solver = Solver::new(Method::lbfgs(), control, x.len());
        assert_eq!(solver.threads(), 1);
        if explicit_one {
            solver.set_threads(1).expect("set_threads(1)");
        }
        solver
            .set_exp(fix.pos.view(), &fix.neigh, &params)
            .expect("install exp");
        for _ in 0..4 {
            solver.step(&obj, &mut x).expect("step");
        }

        let mut words = Vec::new();
        words.extend(applied.iter().map(|v| v.to_bits()));
        words.extend(solved1.iter().map(|v| v.to_bits()));
        words.extend(solved2.iter().map(|v| v.to_bits()));
        words.extend(x.iter().map(|v| v.to_bits()));
        words
    };

    let default_serial_a = run_serial(false);
    let default_serial_b = run_serial(false);
    let explicit_one = run_serial(true);
    assert_eq!(default_serial_a, default_serial_b, "serial repeat drifted");
    assert_eq!(default_serial_a, explicit_one, "threads=None vs threads=1 drifted");

    // BlockIc0 vs Jacobi must solve the same SPD system within tolerance and cut iterations.
    let fix4 = cu_vacancy(4);
    let p_ic0 = Exp::assemble(
        fix4.pos.view(),
        &fix4.neigh,
        &ExpParams::default().with_pc(ExpPcKind::BlockIc0),
        1.0,
        R0_CU,
    )
    .expect("ic0");
    let p_jac = Exp::assemble(
        fix4.pos.view(),
        &fix4.neigh,
        &ExpParams::default().with_pc(ExpPcKind::Jacobi),
        1.0,
        R0_CU,
    )
    .expect("jacobi");
    let it_ic0 = p_ic0.solve_iterations(fix4.rhs.view()).expect("ic0 iters");
    let it_jac = p_jac.solve_iterations(fix4.rhs.view()).expect("jacobi iters");
    assert!(
        it_ic0 < it_jac,
        "BlockIc0 ({it_ic0}) should take fewer PCG iterations than Jacobi ({it_jac})"
    );
    let s_ic0 = p_ic0.solve(fix4.rhs.view()).expect("s_ic0");
    let s_jac = p_jac.solve(fix4.rhs.view()).expect("s_jac");
    let max_diff = (&s_ic0 - &s_jac)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1.0e-7, "BlockIc0 vs Jacobi max diff {max_diff}");

    // Warm start on a smoothly perturbed RHS must take fewer iterations than cold start.
    let next_rhs = fix4.rhs.mapv(|v| 0.98 * v + 1.0e-4);
    let (_, warm_iters) = p_ic0.solve_counted(next_rhs.view()).expect("warm");
    let cold_iters = p_ic0.solve_iterations(next_rhs.view()).expect("cold");
    assert!(
        warm_iters < cold_iters,
        "warm start ({warm_iters}) did not beat cold start ({cold_iters})"
    );
    let _ = fnv1a_words(s_ic0.as_slice().expect("contiguous"));
}

#[test]
fn threaded_build_pcg_and_batch_match_serial_within_stated_tolerance() {
    let _guard = ALLOC_LOCK.lock().expect("alloc lock");
    let fix = cu_vacancy(7); // N = 1371, crosses 512-atom gate and spans 2 1024-atom IC(0) blocks
    assert!(Exp::assemble(
        fix.pos.view(),
        &fix.neigh,
        &ExpParams::default().with_threads(0),
        1.0,
        R0_CU
    )
    .is_err());

    let serial_params = ExpParams::default()
        .with_mu(1.0)
        .with_r_nn(R0_CU)
        .with_lengths(fix.lengths)
        .with_threads(1);
    let serial_exp =
        Exp::assemble(fix.pos.view(), &fix.neigh, &serial_params, 1.0, R0_CU).expect("serial exp");
    let ref_apply = serial_exp.apply(fix.rhs.view());
    let ref_cold = serial_exp.solve(fix.rhs.view()).expect("serial cold");
    let rhs2 = fix.rhs.mapv(|v| 0.94 * v + 2.0e-3);
    let ref_warm = serial_exp.solve(rhs2.view()).expect("serial warm");

    for threads in [2usize, 3, 4, 8] {
        let par_params = ExpParams::default()
            .with_mu(1.0)
            .with_r_nn(R0_CU)
            .with_lengths(fix.lengths)
            .with_threads(threads);
        let par_exp =
            Exp::assemble(fix.pos.view(), &fix.neigh, &par_params, 1.0, R0_CU).expect("par exp");
        let got_apply = par_exp.apply(fix.rhs.view());
        let got_cold = par_exp.solve(fix.rhs.view()).expect("par cold");
        let got_warm = par_exp.solve(rhs2.view()).expect("par warm");

        assert_within_tol(&got_apply, &ref_apply, &format!("Exp::apply T={threads}"));
        assert_within_tol(&got_cold, &ref_cold, &format!("Exp::solve cold T={threads}"));
        assert_within_tol(&got_warm, &ref_warm, &format!("Exp::solve warm T={threads}"));
    }

    // Batch multi-image assembly & solve parity (6 images)
    let images_pos: Vec<Array1<f64>> = (0..6)
        .map(|m| fix.pos.mapv(|v| v + 0.002 * (m as f64)))
        .collect();
    let images_rhs: Vec<Array1<f64>> = (0..6)
        .map(|m| fix.rhs.mapv(|v| (1.0 - 0.05 * m as f64) * v))
        .collect();
    let batch_inputs: Vec<(ArrayView1<f64>, &Neighbours)> = images_pos
        .iter()
        .map(|p| (p.view(), &fix.neigh))
        .collect();
    let rhs_views: Vec<ArrayView1<f64>> = images_rhs.iter().map(|r| r.view()).collect();

    let batch_serial =
        Exp::assemble_batch(&batch_inputs, &serial_params, 1.0, R0_CU).expect("batch 1");
    let sol_serial = Exp::solve_batch(&batch_serial, &rhs_views, Some(1))
        .expect("batch ok")
        .expect("batch spd");

    let par4_params = serial_params.clone().with_threads(4);
    let batch_par = Exp::assemble_batch(&batch_inputs, &par4_params, 1.0, R0_CU).expect("batch 4");
    let sol_par = Exp::solve_batch(&batch_par, &rhs_views, Some(4))
        .expect("batch par ok")
        .expect("batch par spd");

    for m in 0..6 {
        assert_within_tol(&sol_par[m], &sol_serial[m], &format!("batch image {m}"));
    }

    // Lindh, FF, and Solver::set_threads parity on a covalent lattice (128 atoms)
    let n_cov = 128;
    let mut cov_pos = Array1::zeros(3 * n_cov);
    for i in 0..n_cov {
        cov_pos[3 * i] = (i % 4) as f64 * 1.48;
        cov_pos[3 * i + 1] = ((i / 4) % 4) as f64 * 1.50;
        cov_pos[3 * i + 2] = (i / 16) as f64 * 1.52;
    }
    let z = vec![6i32; n_cov];
    let cov_rhs = Array1::from_iter((0..3 * n_cov).map(|k| ((k % 11) as f64 - 5.0) * 0.04));

    let lindh1 = Lindh::assemble(
        cov_pos.view(),
        &z,
        &[],
        &LindhParams::default().with_r_cut(3.0).with_threads(1),
    )
    .expect("lindh 1");
    let lindh4 = Lindh::assemble(
        cov_pos.view(),
        &z,
        &[],
        &LindhParams::default().with_r_cut(3.0).with_threads(4),
    )
    .expect("lindh 4");
    assert_within_tol(
        &lindh4.solve(cov_rhs.view()).expect("l4"),
        &lindh1.solve(cov_rhs.view()).expect("l1"),
        "Lindh T=4 vs T=1",
    );

    let bonds: Vec<BondTerm> = (0..n_cov - 1)
        .map(|i| BondTerm {
            i,
            j: i + 1,
            k: 2.0,
        })
        .collect();
    let terms = FfTerms {
        bonds: &bonds,
        ..FfTerms::default()
    };
    let ff1 = Ff::assemble(cov_pos.view(), terms, &FfParams::default().with_threads(1))
        .expect("ff 1");
    let ff4 = Ff::assemble(cov_pos.view(), terms, &FfParams::default().with_threads(4))
        .expect("ff 4");
    assert_within_tol(
        &ff4.solve(cov_rhs.view()).expect("ff4"),
        &ff1.solve(cov_rhs.view()).expect("ff1"),
        "Ff T=4 vs T=1",
    );

    // Solver::set_threads(0) fails and set_threads(4) matches set_threads(1).
    let mut s_bad = Solver::new(Method::lbfgs(), Control::default(), cov_pos.len());
    assert!(s_bad.set_threads(0).is_err());

    let obj = Oracle::unbounded(cov_pos.len(), |x: ArrayView1<f64>| {
        let g = x.mapv(|v| 2.0 * v);
        (x.dot(&x), g)
    });
    let _ = obj.value_and_gradient(cov_pos.view());
    let mut x1 = cov_pos.clone();
    let mut x4 = cov_pos.clone();
    let mut s1 = Solver::new(Method::lbfgs(), Control::default(), cov_pos.len());
    let mut s4 = Solver::new(Method::lbfgs(), Control::default(), cov_pos.len());
    s1.set_threads(1).expect("t1");
    s4.set_threads(4).expect("t4");
    assert_eq!(s4.threads(), 4);
    let lp = LindhParams::default().with_mu(1.0).with_r_cut(3.0);
    s1.set_lindh(cov_pos.view(), &z, &[], &lp).expect("s1 lindh");
    s4.set_lindh(cov_pos.view(), &z, &[], &lp).expect("s4 lindh");
    s1.step(&obj, &mut x1).expect("step 1");
    s4.step(&obj, &mut x4).expect("step 4");
    assert_within_tol(&x4, &x1, "Solver::step T=4 vs T=1");
}
