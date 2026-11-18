//! The C entry points of the variable-cell layout: the same relaxation as
//! the Rust session, bit for bit, through `rgmin_solver_step_cell`.

#![cfg(feature = "capi")]

mod cell_support;

use std::os::raw::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use cell_support::{evaluate, fcc, neighbours, rattle, strain_positions, strained};
use dlpk::sys::DLManagedTensorVersioned;
use ndarray::{Array1, ArrayView1};
use rgmin::ffi::{
    rgmin_cell_default, rgmin_cell_t, rgmin_control_default, rgmin_exp_default, rgmin_method_t,
    rgmin_report_t, rgmin_solver_clear_cell_filter, rgmin_solver_create, rgmin_solver_exp_mu_c,
    rgmin_solver_free, rgmin_solver_set_cell_filter, rgmin_solver_set_exp,
    rgmin_solver_set_exp_mu_c, rgmin_solver_step_cell, rgmin_status_t, rgmin_stop_t,
    rgmin_tensor_borrow_cpu_f64, rgmin_tensor_free,
};
use rgmin::{
    Accept, Cell, CellEvaluation, CellFilter, CellFilterConfig, Control, ExpParams, Method,
    PreconSpec, Solver,
};

const STRAIN: [[f64; 3]; 3] = [
    [0.03, 0.01, -0.006],
    [0.01, -0.025, 0.008],
    [-0.006, 0.008, 0.02],
];
const GTOL: f64 = 1.0e-5;
const MEMORY: usize = 20;
const R_CUT: f64 = 3.0;
const STEPS: usize = 80;

struct Crystal {
    positions: Array1<f64>,
    cell: Cell,
    orig: Cell,
    natoms: usize,
}

fn crystal() -> Crystal {
    let (mut positions, orig) = fcc(2, 3.61);
    rattle(&mut positions, 0.03, 11);
    let cell = strained(&orig, STRAIN);
    let positions = strain_positions(&positions, STRAIN);
    let natoms = positions.len() / 3;
    Crystal {
        positions,
        cell,
        orig,
        natoms,
    }
}

unsafe extern "C" fn lj_cell(
    user: *mut c_void,
    positions: *const DLManagedTensorVersioned,
    cell: *const f64,
    energy_out: *mut f64,
    forces_out: *mut DLManagedTensorVersioned,
    stress_out: *mut f64,
) -> rgmin_status_t {
    let calls = unsafe { &mut *user.cast::<usize>() };
    *calls += 1;
    let (p, n) = unsafe { tensor(positions) };
    let (f, fn_) = unsafe { tensor(forces_out as *const DLManagedTensorVersioned) };
    assert_eq!(n, fn_);
    let pos = unsafe { std::slice::from_raw_parts(p, n) };
    let flat = unsafe { std::slice::from_raw_parts(cell, 9) };
    let lattice = [
        [flat[0], flat[1], flat[2]],
        [flat[3], flat[4], flat[5]],
        [flat[6], flat[7], flat[8]],
    ];
    let ev = evaluate(ArrayView1::from(pos), &lattice);
    unsafe {
        *energy_out = ev.energy;
        std::slice::from_raw_parts_mut(f as *mut f64, n)
            .copy_from_slice(ev.forces.as_slice().unwrap());
        for (k, v) in ev.stress.iter().flatten().enumerate() {
            *stress_out.add(k) = *v;
        }
    }
    rgmin_status_t::RGMIN_SUCCESS
}

unsafe extern "C" fn failing_cell(
    _user: *mut c_void,
    _positions: *const DLManagedTensorVersioned,
    _cell: *const f64,
    _energy_out: *mut f64,
    _forces_out: *mut DLManagedTensorVersioned,
    _stress_out: *mut f64,
) -> rgmin_status_t {
    rgmin_status_t::RGMIN_INTERNAL_ERROR
}

unsafe fn tensor(t: *const DLManagedTensorVersioned) -> (*const f64, usize) {
    let dl = unsafe { &(*t).dl_tensor };
    let n = unsafe { *dl.shape as usize };
    let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
    (p, n)
}

fn c_control() -> rgmin::ffi::rgmin_control_t {
    let mut ctrl = rgmin_control_default();
    ctrl.maxiter = STEPS;
    ctrl.gtol = GTOL;
    ctrl.istep = 1.0;
    ctrl.memory = MEMORY;
    ctrl
}

struct CRun {
    positions: Vec<f64>,
    cell: [f64; 9],
    grad_norm: f64,
    steps: usize,
    calls: usize,
    mu_c: f64,
}

fn run_c(c: &Crystal, config: Option<&rgmin_cell_t>, exp: bool) -> CRun {
    let dim = 3 * c.natoms + 9;
    let ctrl = c_control();
    let solver = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, dim) };
    assert!(!solver.is_null());
    let orig: Vec<f64> = c.orig.iter().flatten().copied().collect();
    let status = unsafe {
        rgmin_solver_set_cell_filter(
            solver,
            config.map_or(std::ptr::null(), |r| r as *const rgmin_cell_t),
            orig.as_ptr(),
            c.natoms,
        )
    };
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    let nb = neighbours(c.positions.view(), &c.cell, R_CUT);
    if exp {
        let i: Vec<u32> = nb.i.iter().map(|&v| v as u32).collect();
        let j: Vec<u32> = nb.j.iter().map(|&v| v as u32).collect();
        let mut params = rgmin_exp_default();
        params.r_cut = R_CUT;
        let status = unsafe {
            rgmin_solver_set_exp(
                solver,
                &params,
                i.as_ptr(),
                j.as_ptr(),
                nb.r.as_ptr(),
                i.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
            )
        };
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    }
    let mut positions = c.positions.to_vec();
    let mut cell: [f64; 9] = std::array::from_fn(|k| c.cell[k / 3][k % 3]);
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(positions.as_mut_ptr(), positions.len()) };
    let mut calls = 0usize;
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: f64::INFINITY,
        stop: rgmin_stop_t::RGMIN_STOP_BUDGET,
    };
    for _ in 0..STEPS {
        let status = unsafe {
            rgmin_solver_step_cell(
                solver,
                Some(lj_cell),
                (&mut calls as *mut usize).cast(),
                tensor,
                cell.as_mut_ptr(),
                &mut out,
            )
        };
        assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
        if out.grad_norm < GTOL {
            break;
        }
    }
    let mu_c = unsafe { rgmin_solver_exp_mu_c(solver) };
    unsafe {
        rgmin_solver_free(solver);
        rgmin_tensor_free(tensor);
    }
    CRun {
        positions,
        cell,
        grad_norm: out.grad_norm,
        steps: out.steps,
        calls,
        mu_c,
    }
}

fn run_rust(c: &Crystal, config: CellFilterConfig, exp: bool) -> (Vec<f64>, [f64; 9], usize) {
    let filter = CellFilter::new(config, c.orig, c.natoms).unwrap();
    let control = Control::default()
        .with_maxiter(STEPS)
        .with_gtol(GTOL)
        .with_istep(1.0);
    let mut solver =
        Solver::new(Method::lbfgs_with_memory(MEMORY), control, filter.dim()).with_gtol(GTOL);
    solver.set_accept(Accept::Energy);
    solver.set_cell_filter(filter.clone()).unwrap();
    let mut positions = c.positions.clone();
    let mut cell = c.cell;
    if exp {
        let x = filter.pack(positions.view(), &cell).unwrap();
        let nb = neighbours(positions.view(), &cell, R_CUT);
        let params = ExpParams::default().with_r_cut(R_CUT);
        solver
            .set_precon(PreconSpec::Exp {
                positions: x.view(),
                neighbours: &nb,
                params: &params,
            })
            .unwrap();
    }
    let calls = AtomicUsize::new(0);
    let eval = |p: ArrayView1<f64>, lattice: &Cell| -> CellEvaluation {
        calls.fetch_add(1, Ordering::Relaxed);
        evaluate(p, lattice)
    };
    for _ in 0..STEPS {
        let report = solver.step_cell(&eval, &mut positions, &mut cell).unwrap();
        if report.grad_norm < GTOL {
            break;
        }
    }
    let flat = std::array::from_fn(|k| cell[k / 3][k % 3]);
    (positions.to_vec(), flat, calls.load(Ordering::Relaxed))
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn the_defaults_are_the_rust_defaults() {
    let c = rgmin_cell_default();
    assert!(c.exp_cell_factor.is_nan());
    assert_eq!(c.pressure, 0.0);
    assert_eq!(c.mask, [1; 9]);
    assert_eq!((c.hydrostatic, c.constant_volume), (0, 0));
}

#[test]
fn the_c_session_is_the_rust_session() {
    let c = crystal();
    let got = run_c(&c, None, false);
    let (positions, cell, calls) = run_rust(&c, CellFilterConfig::default(), false);
    assert!(got.grad_norm < GTOL, "grad {}", got.grad_norm);
    assert!(same_bits(&got.positions, &positions));
    assert!(same_bits(&got.cell, &cell));
    assert_eq!(got.calls, calls);
    eprintln!("c session: {} steps, {} evaluations", got.steps, got.calls);
}

#[test]
fn the_c_options_are_the_rust_options() {
    let c = crystal();
    let mut config = rgmin_cell_default();
    config.pressure = 0.002;
    config.hydrostatic = 1;
    let got = run_c(&c, Some(&config), false);
    let want = CellFilterConfig::default()
        .with_pressure(0.002)
        .with_hydrostatic(true);
    let (positions, cell, _) = run_rust(&c, want, false);
    assert!(same_bits(&got.positions, &positions));
    assert!(same_bits(&got.cell, &cell));
}

#[test]
fn the_c_exponential_cell_block_is_the_rust_one() {
    let c = crystal();
    let got = run_c(&c, None, true);
    let (positions, cell, _) = run_rust(&c, CellFilterConfig::default(), true);
    assert!(got.grad_norm < GTOL);
    assert!(same_bits(&got.positions, &positions));
    assert!(same_bits(&got.cell, &cell));
    assert!(got.mu_c >= 1.0, "mu_c {}", got.mu_c);
}

#[test]
fn mu_c_can_be_fixed_through_the_c_entry() {
    let c = crystal();
    let dim = 3 * c.natoms + 9;
    let ctrl = c_control();
    let solver = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, dim) };
    let orig: Vec<f64> = c.orig.iter().flatten().copied().collect();
    assert_eq!(
        unsafe { rgmin_solver_set_exp_mu_c(solver, 2.0) },
        rgmin_status_t::RGMIN_INVALID_PARAMETER,
        "no filter yet"
    );
    let status =
        unsafe { rgmin_solver_set_cell_filter(solver, std::ptr::null(), orig.as_ptr(), c.natoms) };
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(
        unsafe { rgmin_solver_set_exp_mu_c(solver, 2.0) },
        rgmin_status_t::RGMIN_INVALID_PARAMETER,
        "no preconditioner yet"
    );
    assert!(unsafe { rgmin_solver_exp_mu_c(solver) }.is_nan());
    let nb = neighbours(c.positions.view(), &c.cell, R_CUT);
    let i: Vec<u32> = nb.i.iter().map(|&v| v as u32).collect();
    let j: Vec<u32> = nb.j.iter().map(|&v| v as u32).collect();
    let mut params = rgmin_exp_default();
    params.r_cut = R_CUT;
    let installed = unsafe {
        rgmin_solver_set_exp(
            solver,
            &params,
            i.as_ptr(),
            j.as_ptr(),
            nb.r.as_ptr(),
            i.len(),
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };
    assert_eq!(installed, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(
        unsafe { rgmin_solver_set_exp_mu_c(solver, 2.0) },
        rgmin_status_t::RGMIN_SUCCESS
    );
    assert_eq!(unsafe { rgmin_solver_exp_mu_c(solver) }, 2.0);
    let mut positions = c.positions.to_vec();
    let mut cell: [f64; 9] = std::array::from_fn(|k| c.cell[k / 3][k % 3]);
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(positions.as_mut_ptr(), positions.len()) };
    let mut calls = 0usize;
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
        stop: rgmin_stop_t::RGMIN_STOP_BUDGET,
    };
    let step = |out: &mut rgmin_report_t, calls: &mut usize, cell: &mut [f64; 9]| unsafe {
        rgmin_solver_step_cell(
            solver,
            Some(lj_cell),
            (calls as *mut usize).cast(),
            tensor,
            cell.as_mut_ptr(),
            out,
        )
    };
    assert_eq!(
        step(&mut out, &mut calls, &mut cell),
        rgmin_status_t::RGMIN_SUCCESS
    );
    assert_eq!(unsafe { rgmin_solver_exp_mu_c(solver) }, 2.0);
    assert_eq!(
        unsafe { rgmin_solver_set_exp_mu_c(solver, f64::NAN) },
        rgmin_status_t::RGMIN_SUCCESS
    );
    assert!(unsafe { rgmin_solver_exp_mu_c(solver) }.is_nan());
    assert_eq!(
        step(&mut out, &mut calls, &mut cell),
        rgmin_status_t::RGMIN_SUCCESS
    );
    assert!(unsafe { rgmin_solver_exp_mu_c(solver) } >= 1.0);
    unsafe {
        rgmin_solver_clear_cell_filter(solver);
        rgmin_solver_free(solver);
        rgmin_tensor_free(tensor);
    }
}

#[test]
fn bad_input_is_a_status_not_a_panic() {
    let c = crystal();
    let dim = 3 * c.natoms + 9;
    let ctrl = c_control();
    let solver = unsafe { rgmin_solver_create(rgmin_method_t::RGMIN_LBFGS, &ctrl, dim) };
    let orig: Vec<f64> = c.orig.iter().flatten().copied().collect();
    let wrong = unsafe {
        rgmin_solver_set_cell_filter(solver, std::ptr::null(), orig.as_ptr(), c.natoms + 1)
    };
    assert_eq!(wrong, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    for huge in [usize::MAX, usize::MAX / 3] {
        let overflow =
            unsafe { rgmin_solver_set_cell_filter(solver, std::ptr::null(), orig.as_ptr(), huge) };
        assert_eq!(overflow, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    }
    let null = unsafe {
        rgmin_solver_set_cell_filter(std::ptr::null_mut(), std::ptr::null(), orig.as_ptr(), 1)
    };
    assert_eq!(null, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert_eq!(
        unsafe { rgmin_solver_set_cell_filter(solver, std::ptr::null(), orig.as_ptr(), c.natoms) },
        rgmin_status_t::RGMIN_SUCCESS
    );
    let mut positions = c.positions.to_vec();
    let mut cell: [f64; 9] = std::array::from_fn(|k| c.cell[k / 3][k % 3]);
    let tensor = unsafe { rgmin_tensor_borrow_cpu_f64(positions.as_mut_ptr(), positions.len()) };
    let mut out = rgmin_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
        stop: rgmin_stop_t::RGMIN_STOP_BUDGET,
    };
    let missing = unsafe {
        rgmin_solver_step_cell(
            solver,
            None,
            std::ptr::null_mut(),
            tensor,
            cell.as_mut_ptr(),
            &mut out,
        )
    };
    assert_eq!(missing, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    let failed = unsafe {
        rgmin_solver_step_cell(
            solver,
            Some(failing_cell),
            std::ptr::null_mut(),
            tensor,
            cell.as_mut_ptr(),
            &mut out,
        )
    };
    assert_ne!(failed, rgmin_status_t::RGMIN_SUCCESS);
    unsafe {
        rgmin_solver_free(solver);
        rgmin_tensor_free(tensor);
    }
}
