// Copyright (c) rgmin developers
//! The allocation-free loop keeps a recorded trajectory bit for bit,
//! and a warm L-BFGS iteration does not allocate outside the objective.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::objectives::Rosenbrock;
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Accept, Control, FireKind, Lbfgs, Method, Solver};

thread_local! {
    static LOCAL_ALLOCS: Cell<usize> = const { Cell::new(0) };
}
static GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());
struct Counting;

fn bump() {
    let _ = LOCAL_ALLOCS.try_with(|c| c.set(c.get().wrapping_add(1)));
}

fn allocs() -> usize {
    LOCAL_ALLOCS.with(|c| c.get())
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn bits(x: f64) -> u64 {
    x.to_bits()
}

fn rosen(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (a, b) = (x[0], x[1]);
    let t = b - a * a;
    let u = 1.0 - a;
    let f = 100.0 * t * t + u * u;
    let g = array![-400.0 * a * t - 2.0 * u, 200.0 * t];
    (f, g)
}

/// Probes of `Lbfgs::minimize` on Rosenbrock from `(-1.2, 1)`.
/// Recorded from the merged eOn cap, which is the default step.
const LBFGS_PROBES: &[(u64, u64, u64)] = &[
    (0x4038333333333332, 0xbff3333333333333, 0x3ff0000000000000),
    (0x4035f233d82d51f6, 0xbff30d46fbdb1480, 0x3ff00f7a89873b8d),
    (0x40121a9d5ff6f9bd, 0xbff0e2fb118a75c6, 0x3ff0f1b8deb7cfee),
    (0x40108f2590ca106a, 0xbff08d2ef34c67fb, 0x3ff1142c13830a67),
    (0x40107fffab3b91b5, 0xbff078c0abb489fe, 0x3ff11555ebbe9469),
    (0x40107c3e8facc6ea, 0xbff07210eb65958e, 0x3ff10e8509ec99fb),
    (0x4010598fe9e2e0a7, 0xbff032de44c9c0f5, 0x3ff0b5d3c59c1212),
    (0x40100a0611bdc512, 0xbfef4991c0cf5881, 0x3fef9822b38bce04),
    (0x400fa933242b9557, 0xbfef7b400fb0caf3, 0x3fef72a32beda0af),
    (0x400f9653089565c4, 0xbfef8b350d03391f, 0x3fef534b7ca94fc2),
    (0x400f8c3c6a4797ea, 0xbfef8691c50477de, 0x3fef38488a0e607e),
    (0x400f437a535f668d, 0xbfef3cae42e1cebb, 0x3fee530e44e79ca8),
    (0x400ebd7de8d59570, 0xbfee777737225e89, 0x3fec6f6f4a2d5352),
    (0x400dc1e9cff0dd30, 0xbfec85f2ae4caabe, 0x3fe83762788c4cd0),
];

const SESSION_LBFGS: &[(u64, u64, u64)] = &[
    (0x40121a9d5ff6f9f2, 0xbff0e2fb118a75cd, 0x3ff0f1b8deb7cfeb),
    (0x40108f2590ca106a, 0xbff08d2ef34c67fc, 0x3ff1142c13830a67),
    (0x40107fffab3b91b5, 0xbff078c0abb489fe, 0x3ff11555ebbe9469),
    (0x40107c3e8facc6eb, 0xbff07210eb65958f, 0x3ff10e8509ec99fb),
    (0x4010598fe9e2e098, 0xbff032de44c9c0d8, 0x3ff0b5d3c59c11e5),
    (0x40100a0611bdc4f0, 0xbfef4991c0cf5814, 0x3fef9822b38bcd46),
    (0x400fa933242b9505, 0xbfef7b400fb0ca7e, 0x3fef72a32bed9ff8),
    (0x400f965308956560, 0xbfef8b350d0338bb, 0x3fef534b7ca94efe),
];

const SESSION_FIRE2: &[(u64, u64, u64)] = &[
    (0x4057551d19f6891e, 0x3fee978d4fdf3b66, 0x3ffe147ae147ae14),
    (0x40938fa881a061c8, 0x4000624dd2f1a9fc, 0x3fe50b308e289c7e),
    (0x407d5f3cfd571b53, 0xc000624dd2f1a9fc, 0x4000624dd2f1a9fc),
    (0x407cdc2a8ebf8e03, 0x4000624dd2f1a9fc, 0x4000624dd2f1a9fc),
    (0x407d5f3cfd571b53, 0xc000624dd2f1a9fc, 0x4000624dd2f1a9fc),
    (0x406121d7c53e9f9f, 0xbfee3fc0c66a6e42, 0x4000624dd2f1a9fc),
    (0x40381ff672c7490a, 0xbff4346512b5b55b, 0x40003ea3cd188f27),
    (0x4023a00d6932be5a, 0xbff569d5e06a7456, 0x3ffffe53f5b9fde5),
];

const GEOM_LBFGS: &[(u64, u64, u64)] = &[
    (0x40121a9d5ff6f9f2, 0xbff0e2fb118a75cd, 0x3ff0f1b8deb7cfeb),
    (0x40108f2590ca106a, 0xbff08d2ef34c67fc, 0x3ff1142c13830a67),
    (0x40107fffab3b91b5, 0xbff078c0abb489fe, 0x3ff11555ebbe9469),
    (0x40107c3e8facc6eb, 0xbff07210eb65958f, 0x3ff10e8509ec99fb),
    (0x4010598fe9e2e098, 0xbff032de44c9c0d8, 0x3ff0b5d3c59c11e5),
    (0x40100a0611bdc4f0, 0xbfef4991c0cf5814, 0x3fef9822b38bcd46),
    (0x400fa933242b9505, 0xbfef7b400fb0ca7e, 0x3fef72a32bed9ff8),
    (0x400f965308956560, 0xbfef8b350d0338bb, 0x3fef534b7ca94efe),
];

fn assert_row(tag: &str, k: usize, f: f64, x: ArrayView1<f64>, row: (u64, u64, u64)) {
    assert_eq!(bits(f), row.0, "{tag} step {k} value");
    assert_eq!(bits(x[0]), row.1, "{tag} step {k} x0");
    assert_eq!(bits(x[1]), row.2, "{tag} step {k} x1");
}

#[test]
fn recorded_trajectories_stay_bit_identical() {
    let _hold = GATE.lock().unwrap();
    let mut seen = Vec::new();
    let mut lbfgs = Lbfgs::with_capacity(8);
    lbfgs.gtol = 1e-12;
    let _ = lbfgs.minimize_watched(
        array![-1.2, 1.0].view(),
        12,
        |x| {
            let (f, g) = rosen(x);
            seen.push((f, x.to_owned()));
            Some((f, g))
        },
        |_, _| true,
    );
    assert_eq!(seen.len(), LBFGS_PROBES.len());
    for (k, ((f, x), row)) in seen.iter().zip(LBFGS_PROBES).enumerate() {
        assert_row("lbfgs", k, *f, x.view(), *row);
    }

    let obj = Rosenbrock::<2>::new();
    let control = Control {
        maxiter: 20,
        gtol: 1e-14,
        istep: 0.1,
        maxmove: None,
        ..Default::default()
    };
    let mut solver = Solver::new(Method::lbfgs(), control.clone(), 2);
    solver.set_accept(Accept::Energy);
    let mut x = array![-1.2, 1.0];
    for (k, row) in SESSION_LBFGS.iter().enumerate() {
        let rep = solver.step(&obj, &mut x).unwrap();
        assert_row("session lbfgs", k, rep.value, x.view(), *row);
    }
    let mut solver = Solver::new(Method::Fire { kind: FireKind::V2 }, control.clone(), 2);
    let mut x = array![-1.2, 1.0];
    for (k, row) in SESSION_FIRE2.iter().enumerate() {
        let rep = solver.step(&obj, &mut x).unwrap();
        assert_row("fire2", k, rep.value, x.view(), *row);
    }
    let mut solver = Solver::new(Method::lbfgs(), control, 2);
    solver.set_atom_maxmove(0.2);
    let mut x = array![-1.2, 1.0];
    for (k, row) in GEOM_LBFGS.iter().enumerate() {
        let rep = solver.step(&obj, &mut x).unwrap();
        assert_row("geometry lbfgs", k, rep.value, x.view(), *row);
    }
}

#[test]
fn warm_lbfgs_iteration_allocates_only_inside_the_objective() {
    let _hold = GATE.lock().unwrap();
    let mut lbfgs = Lbfgs::with_capacity(8);
    lbfgs.gtol = 0.0;
    let mut obj_at = Vec::with_capacity(32);
    let mut total_at = Vec::with_capacity(32);
    let obj_allocs = AtomicUsize::new(0);
    let _ = lbfgs.minimize_watched(
        array![-1.2, 1.0].view(),
        16,
        |x| {
            let before = allocs();
            let (f, g) = rosen(x);
            obj_allocs.fetch_add(allocs().saturating_sub(before), Ordering::Relaxed);
            Some((f, g))
        },
        |_it, _| {
            total_at.push(allocs());
            obj_at.push(obj_allocs.load(Ordering::Relaxed));
            true
        },
    );
    assert!(total_at.len() > 10, "iterations {}", total_at.len());
    let mut warm = Vec::new();
    for i in 1..total_at.len() {
        let total = total_at[i] - total_at[i - 1];
        let obj = obj_at[i] - obj_at[i - 1];
        let solver = total.saturating_sub(obj);
        warm.push(solver);
    }
    let tail = &warm[warm.len().saturating_sub(4)..];
    assert!(
        tail.iter().all(|n| *n == 0),
        "solver allocations per iteration after warm-up: {warm:?}"
    );
}

fn session_fire(preset: rgmin::FirePreset, abc: bool, cap: Option<f64>) -> Vec<usize> {
    let obj_allocs = AtomicUsize::new(0);
    let obj = rgmin::Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let before = allocs();
        let (f, g) = rosen(x);
        obj_allocs.fetch_add(allocs().saturating_sub(before), Ordering::Relaxed);
        (f, g)
    });
    let control = Control {
        maxiter: 20,
        gtol: 0.0,
        istep: 0.1,
        maxmove: None,
        ..Default::default()
    };
    let mut solver = Solver::new(Method::Fire { kind: FireKind::V2 }, control, 2);
    solver.set_fire_preset(preset).unwrap();
    solver.set_fire_abc(abc).unwrap();
    if let Some(cap) = cap {
        solver.set_atom_maxmove(cap);
    }
    let mut x = array![-1.2, 1.0];
    let mut counts = Vec::new();
    for _ in 0..6 {
        let obj0 = obj_allocs.load(Ordering::Relaxed);
        let before = allocs();
        let rep = solver.step(&obj, &mut x).unwrap();
        drop(rep);
        let total = allocs() - before;
        let objective = obj_allocs.load(Ordering::Relaxed) - obj0;
        counts.push(total.saturating_sub(objective));
    }
    counts
}

fn session_solver_allocs(method: Method, accept: Accept) -> Vec<usize> {
    let obj_allocs = AtomicUsize::new(0);
    let obj = rgmin::Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let before = allocs();
        let (f, g) = rosen(x);
        obj_allocs.fetch_add(allocs().saturating_sub(before), Ordering::Relaxed);
        (f, g)
    });
    let control = Control {
        maxiter: 20,
        gtol: 0.0,
        istep: 0.1,
        maxmove: None,
        ..Default::default()
    };
    let mut solver = Solver::new(method, control, 2);
    if accept == Accept::Energy {
        solver.set_accept(Accept::Energy);
    }
    let mut x = array![-1.2, 1.0];
    let mut counts = Vec::new();
    for _ in 0..6 {
        let obj0 = obj_allocs.load(Ordering::Relaxed);
        let before = allocs();
        let rep = solver.step(&obj, &mut x).unwrap();
        drop(rep);
        let total = allocs() - before;
        let objective = obj_allocs.load(Ordering::Relaxed) - obj0;
        counts.push(total.saturating_sub(objective));
    }
    counts
}

/// Counts recorded before the scratch fix, objective subtracted, six steps:
/// L-BFGS `Accept::Energy` `[35, 0, 0, 0, 0, 0]`,
/// FIRE 2.0 default `[2, 0, 0, 0, 0, 0]`,
/// Guénolé 2020 `[3, 1, 1, 1, 1, 1]`,
/// Guénolé 2020 with a 0.2 atom cap `[3, 1, 1, 1, 1, 1]`,
/// Guénolé 2020 with ABC and that cap `[4, 2, 2, 2, 2, 2]`,
/// FIRE 1.0 `[2, 0, 0, 0, 0, 0]`.
#[test]
fn steady_solver_steps_allocate_nothing_outside_the_objective() {
    let _hold = GATE.lock().unwrap();
    let cases = [
        session_solver_allocs(Method::lbfgs(), Accept::Energy),
        session_solver_allocs(Method::Fire { kind: FireKind::V2 }, Accept::None),
        session_fire(rgmin::FirePreset::Guenole2020, false, None),
        session_fire(rgmin::FirePreset::Guenole2020, false, Some(0.2)),
        session_fire(rgmin::FirePreset::Guenole2020, true, Some(0.2)),
        session_solver_allocs(Method::Fire { kind: FireKind::V1 }, Accept::None),
    ];
    for counts in cases {
        assert!(
            counts[1..].iter().all(|n| *n == 0),
            "solver allocations from step 0: {counts:?}"
        );
    }
}

fn rosen_chain(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len();
    let mut f = 0.0;
    let mut g = vec![0.0; n];
    for i in 0..n - 1 {
        let t = x[i + 1] - x[i] * x[i];
        let u = 1.0 - x[i];
        f += 100.0 * t * t + u * u;
        g[i] += -400.0 * x[i] * t - 2.0 * u;
        g[i + 1] += 200.0 * t;
    }
    (f, Array1::from(g))
}

/// `m = 20` keeps 40 pair vectors, more than the old 32-slot pool.
#[test]
fn a_long_history_stays_allocation_free_after_the_first_step() {
    let _hold = GATE.lock().unwrap();
    rgmin::note_buffer_need(40);
    let fresh: Vec<Array1<f64>> = (0..40).map(|_| Array1::zeros(8)).collect();
    let round = allocs();
    for buf in fresh {
        rgmin::recycle_gradient(buf);
    }
    for _ in 0..40 {
        let buf = rgmin::take_gradient(8);
        assert_eq!(buf.len(), 8);
    }
    assert_eq!(allocs(), round, "cold recycle of 40 then take allocated");
    let obj_allocs = AtomicUsize::new(0);
    let obj = rgmin::Oracle::unbounded(8, |x: ArrayView1<f64>| {
        let before = allocs();
        let (f, g) = rosen_chain(x);
        obj_allocs.fetch_add(allocs().saturating_sub(before), Ordering::Relaxed);
        (f, g)
    });
    let control = Control {
        maxiter: 80,
        gtol: 0.0,
        istep: 0.1,
        maxmove: None,
        ..Default::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 20 }, control, 8);
    solver.set_accept(Accept::Energy);
    assert!(
        rgmin::buffer_keep() >= 40,
        "pool keep {}",
        rgmin::buffer_keep()
    );
    let mut x = Array1::from(vec![-1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0]);
    let mut counts = Vec::new();
    for _ in 0..40 {
        let obj0 = obj_allocs.load(Ordering::Relaxed);
        let before = allocs();
        let rep = solver.step(&obj, &mut x).unwrap();
        drop(rep);
        let total = allocs() - before;
        let objective = obj_allocs.load(Ordering::Relaxed) - obj0;
        counts.push(total.saturating_sub(objective));
    }
    assert!(
        solver.retained_vectors() > 32,
        "retained {}",
        solver.retained_vectors()
    );
    assert!(
        counts[1..].iter().all(|n| *n == 0),
        "solver allocations from step 0: {counts:?}"
    );
    loop {
        let mark = allocs();
        let buf = rgmin::take_gradient(8);
        if allocs() != mark {
            drop(buf);
            break;
        }
    }
    let held: Vec<Array1<f64>> = (0..40).map(|_| Array1::zeros(8)).collect();
    let mut back = Vec::with_capacity(40);
    let after_build = allocs();
    for buf in held {
        rgmin::recycle_gradient(buf);
    }
    let after_recycle = allocs();
    for _ in 0..40 {
        back.push(rgmin::take_gradient(8));
    }
    let after_take = allocs();
    assert_eq!(
        (after_recycle, after_take),
        (after_build, after_build),
        "recycle {after_recycle} take {after_take} build {after_build}"
    );
    assert!(back.iter().all(|buf| buf.len() == 8));
}
