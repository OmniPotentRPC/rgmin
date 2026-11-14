#ifndef RGMIN_H
#define RGMIN_H

/* Crate version. cargo cinstall ships this header. */
#define RGMIN_H_MAJOR 0
#define RGMIN_H_MINOR 6
#define RGMIN_H_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <dlpack/dlpack.h>

/* dlpack names the struct and does not typedef it. C requires the alias.
 * C++ already accepts the struct name as a type. */
typedef struct DLManagedTensorVersioned DLManagedTensorVersioned;

/* The direct eindir entry point only needs opaque handles here. Consumers
 * that use eindir_core's constructors can include its full header first. */
#if defined(__has_include)
#  if __has_include(<eindir-core.h>)
#    include <eindir-core.h>
#  else
typedef struct eindir_objective_t eindir_objective_t;
typedef struct eindir_abi_stamp_t eindir_abi_stamp_t;
#  endif
#else
#  include <eindir-core.h>
#endif

/** \file rgmin.h
 *  \brief C ABI for dest rgmin.
 *
 *  Solvers live in Rust. This header is the C entry:
 *  \ref rgmin_minimize over dlpk \c DLManagedTensorVersioned tensors.
 *  include/xts.h is a compatibility alias for old hosts.
 */

/** Status of an ABI call. */
typedef enum rgmin_status_t {
    RGMIN_SUCCESS = 0,
    RGMIN_INVALID_PARAMETER = 1,
    RGMIN_INTERNAL_ERROR = 2,
    /** Tensor device is not CPU. The ABI stays stable for a later CUDA path. */
    RGMIN_UNSUPPORTED_DEVICE = 3,
    /** Named eigensolver is not linked in this build. */
    RGMIN_UNAVAILABLE = 4,
    /** A distance constraint set was refused or could not be met.
     *  \ref rgmin_last_constraint_error names the kind. ABI minor 31. */
    RGMIN_CONSTRAINT_ERROR = 5
} rgmin_status_t;

/** Compatibility identity for this C ABI. */
typedef struct rgmin_abi_stamp_t {
    uint16_t abi_major;
    uint16_t abi_minor;
    uint16_t layout_revision;
} rgmin_abi_stamp_t;

#define RGMIN_ABI_VERSION_MAJOR 1
#define RGMIN_ABI_VERSION_MINOR 41
/* 37 linesearch, 38 lbfgs_direction, 39 rgmin_lowest_eigenpairs.
 * 40 is the 0.6.0 union with the 0.5.2 C surface.
 * 41 rgmin_manifold_handle_t. */
#define RGMIN_ABI_LAYOUT_REVISION 5
/* Layout revision 5. rgmin_control_t is maxiter, gtol, istep, memory,
 * maxmove, noise_eps, noise_c. maxmove > 0 is the Euclidean cap; 0 keeps
 * the defaults, including the default per-atom cap; a negative value
 * leaves every step cap off. rgmin_report_t is value, steps, grad_norm,
 * stop. Both sides shipped these structs as revision 4 with different
 * fields, so the union is a new revision. */

/** Solver selector. \c RGMIN_LBFGS is the production unconstrained method. */
typedef enum rgmin_method_t {
    RGMIN_POLAK_RIBIERE = 0,
    RGMIN_FLETCHER_REEVES = 1,
    RGMIN_BFGS = 2,
    RGMIN_LBFGS = 3,
    RGMIN_SR1 = 4,
    RGMIN_ADAM = 5,
    RGMIN_STEEPEST = 6,
    RGMIN_SR2 = 7,
    RGMIN_PSO = 8,
    RGMIN_HESTENES_STIEFEL = 9,
    RGMIN_DAI_YUAN = 10,
    RGMIN_CONJUGATE_DESCENT = 11,
    RGMIN_HAGER_ZHANG = 12,
    RGMIN_LIU_STOREY = 13,
    RGMIN_FR_PR = 14,
    RGMIN_NEWTON = 15,
    RGMIN_RFO = 16,
    RGMIN_FIRE = 17,
    RGMIN_BB = 18,
    RGMIN_DOGLEG = 19,
    RGMIN_FIRE2 = 20,
    /** Quick-min. Released 0.5.2 discriminant. */
    RGMIN_QUICKMIN = 21,
    /** Pulay residual subspace. memory 0 means 6 stored pairs. */
    RGMIN_DIIS = 22,
    /** ODE12r: adaptive Euler on the (preconditioned) force. */
    RGMIN_ODE12R = 23,
    /** Steihaug-Toint Newton-CG. Released 0.5.2 discriminant. */
    RGMIN_NEWTON_CG = 24,
    /** Newton-CG with a Nyström preconditioner. Released 0.5.2 discriminant. */
    RGMIN_NEWTON_PCG = 25,
    /** Polak-Ribiere-Polyak, max(0, beta). Develop surface, past the released slots. */
    RGMIN_POLAK_RIBIERE_PLUS = 26,
    /** Polak-Ribiere with the eOn Powell restart at 0.5. Develop surface. */
    RGMIN_EON_CG = 27
} rgmin_method_t;

/** Parse a method token. Case is ignored; hyphens and spaces match underscores.
 *  \c newton-cg and \c newton_cg are the same method. Returns
 *  \c RGMIN_INVALID_PARAMETER on an unknown or non-UTF-8 name. */
rgmin_status_t rgmin_method_from_name(const char *name, rgmin_method_t *out);

/** Conjugacy coefficient β. Closed leaf subset of dest Conjugacy
 *  (src/nlcg). Integers are dest declaration order. Hybrid stays
 *  Rust-only. This is not rgmin_method_t (that enum is the solver axis). */
typedef enum rgmin_conjugacy_t {
    RGMIN_CONJUGACY_FLETCHER_REEVES = 0,
    RGMIN_CONJUGACY_POLAK_RIBIERE = 1,
    RGMIN_CONJUGACY_HESTENES_STIEFEL = 2,
    RGMIN_CONJUGACY_DAI_YUAN = 3,
    RGMIN_CONJUGACY_CONJUGATE_DESCENT = 4,
    RGMIN_CONJUGACY_HAGER_ZHANG = 5,
    RGMIN_CONJUGACY_LIU_STOREY = 6,
    RGMIN_CONJUGACY_FR_PR = 7,
    /** max(0, Polak-Ribiere). Leaf 8. ABI minor 35. */
    RGMIN_CONJUGACY_POLAK_RIBIERE_PLUS = 8
} rgmin_conjugacy_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_conjugacy_t) == sizeof(int32_t),
               "rgmin_conjugacy_t is i32-wide; do not build this header with -fshort-enums");
#endif

/** Why a finished minimization stopped. */
typedef enum rgmin_stop_t {
    RGMIN_STOP_TOLERANCE = 0,
    RGMIN_STOP_NOISE_FLOOR = 1,
    RGMIN_STOP_STEP_TOLERANCE = 2,
    RGMIN_STOP_BUDGET = 3
} rgmin_stop_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_stop_t) == sizeof(int32_t),
               "rgmin_stop_t is i32-wide; do not build this header with -fshort-enums");
#endif

/** Outer-loop controls. \c memory is the L-BFGS pair cap. \c istep <= 0
 *  selects 1. An energy-tested steepest, BFGS, SR1, SR2, BB or
 *  unpreconditioned CG session then takes min(1, 1/||g||) at its first
 *  step as the step scale, until rgmin_solver_forget or
 *  rgmin_solver_rebase. */
typedef struct rgmin_control_t {
    size_t maxiter;
    double gtol;
    double istep;
    size_t memory;
    /** > 0 Euclidean cap; 0 keeps defaults (including the per-atom cap);
     *  < 0 leaves every step cap off. Layout revision 5. */
    double maxmove;
    /** Gradient-projection noise. <= 0 leaves the floor off. */
    double noise_eps;
    /** Floor factor. <= 0 selects 3. */
    double noise_c;
} rgmin_control_t;

/** Result of \ref rgmin_minimize. */
typedef struct rgmin_report_t {
    double value;
    size_t steps;
    double grad_norm;
    /** Which stop produced this report. Layout revision 5. */
    rgmin_stop_t stop;
} rgmin_report_t;

typedef rgmin_status_t (*rgmin_eval_fn)(void *user, const DLManagedTensorVersioned *x,
                                    double *value_out);
typedef rgmin_status_t (*rgmin_grad_fn)(void *user, const DLManagedTensorVersioned *x,
                                    DLManagedTensorVersioned *grad_out);
typedef rgmin_status_t (*rgmin_hess_fn)(void *user, const DLManagedTensorVersioned *x,
                                    DLManagedTensorVersioned *hess_out);
/** Fused (f, grad). One geometry, one host potential call. */
typedef rgmin_status_t (*rgmin_evalgrad_fn)(void *user,
                                        const DLManagedTensorVersioned *x,
                                        double *value_out,
                                        DLManagedTensorVersioned *grad_out);

/** Crate version string. */
const char *rgmin_version(void);
/** Return the C ABI compatibility identity for this build. */
rgmin_abi_stamp_t rgmin_abi_stamp(void);
/** Return nonzero when a stamp is compatible with this build. */
int32_t rgmin_abi_compatible(const rgmin_abi_stamp_t *stamp);
/** Thread-local detail of the last failing call on this thread.
 *  The pointer is borrowed. It stays valid until the next rgmin call
 *  on this thread, and the caller does not free it. Null when this
 *  thread has not stored an error. */
const char *rgmin_last_error(void);
/** The same borrowed string as \ref rgmin_last_error. */
const char *rgmin_last_error_message(void);
/** Borrow a CPU f64 buffer as a dlpk tensor. Rust: rgmin_tensor_borrow_cpu_f64. */
DLManagedTensorVersioned *rgmin_tensor_borrow_cpu_f64(double *data, size_t n);
/** Free a tensor from rgmin_tensor_borrow_cpu_f64. */
void rgmin_tensor_free(DLManagedTensorVersioned *tensor);
/**
 * Minimize from \a x in place.
 *
 * \param eval  Value callback.
 * \param grad  Gradient callback.
 * \param user  Passed through to both callbacks.
 * \param x     CPU f64 state (updated on success).
 * \param ctrl  Iteration / L-BFGS memory controls.
 * \param method Solver. \c RGMIN_LBFGS is the production choice.
 * \param out   Filled on success.
 */
rgmin_status_t rgmin_minimize(rgmin_eval_fn eval, rgmin_grad_fn grad, void *user,
                            DLManagedTensorVersioned *x, const rgmin_control_t *ctrl,
                            rgmin_method_t method, rgmin_report_t *out);
/**
 * Newton / RFO. \a hess writes a length-\c n*n row-major Hessian.
 * \c method is \c RGMIN_NEWTON or \c RGMIN_RFO.
 */
rgmin_status_t rgmin_minimize_hess(rgmin_eval_fn eval, rgmin_grad_fn grad,
                               rgmin_hess_fn hess, void *user,
                               DLManagedTensorVersioned *x,
                               const rgmin_control_t *ctrl, rgmin_method_t method,
                               rgmin_report_t *out);

/**
 * Minimize an eindir-compatible objective without taking ownership of it.
 * The stamp must be compatible with this build and include an analytic
 * gradient. The caller retains ownership of an objective.
 */
rgmin_status_t rgmin_minimize_eindir(
    const eindir_objective_t *objective, const eindir_abi_stamp_t *stamp,
    DLManagedTensorVersioned *x, const rgmin_control_t *ctrl, rgmin_method_t method,
    rgmin_report_t *out);

/**
 * Opaque session. Algorithm memory (L-BFGS pairs, NLCG directions,
 * dense H, Adam moments, PSO swarm) lives here. \c x stays a dlpk
 * tensor. Callbacks are arguments of each step, not stored.
 */
typedef struct rgmin_solver_t rgmin_solver_t;

/** Allocate a session. \a dim is the length of \c x. Null on error.
 *  The exported symbol is \c rgmin_solver_create. */
rgmin_solver_t *rgmin_solver_create(rgmin_method_t method, const rgmin_control_t *ctrl,
                                size_t dim);
/** Release a session from \ref rgmin_solver_create. */
void rgmin_solver_free(rgmin_solver_t *solver);
/** Drop method memory. The next step is a cold start from the current \c x. */
void rgmin_solver_forget(rgmin_solver_t *solver);

/// Keep the method memory, drop the point: the next step evaluates the
/// objective afresh at the current `x` with the initial step scale and an
/// empty acceptance window, preconditioned by the retained curvature. The
/// restart for a sequence of related objectives.
void rgmin_solver_rebase(rgmin_solver_t *solver);
/** Drop only the cached evaluation; method memory, step scale and acceptance
 *  window stay. For an oracle that changes between steps at the same point.
 *  ABI minor 28. */
void rgmin_solver_forget_evaluation(rgmin_solver_t *solver);

/** Euclidean step cap for the next \ref rgmin_solver_step (saddle \c max_move). */
void rgmin_solver_set_maxmove(rgmin_solver_t *solver, double maxmove);
/** How an L-BFGS session uses a caller Hessian (eOn \c lbfgs_step). */
typedef enum rgmin_qn_step_t {
    RGMIN_QN_LBFGS = 0,
    RGMIN_QN_NEWTON = 1,
    RGMIN_QN_RFO = 2
} rgmin_qn_step_t;
/** Two-loop + H0, or Newton/RFO on P. Legal with \ref rgmin_solver_step_hess. */
void rgmin_solver_set_qn_step(rgmin_solver_t *solver, rgmin_qn_step_t step);
/** Exponential preconditioner parameters. NaN selects a default or an estimate.
 *  ABI minor 28. */
typedef struct rgmin_exp_t {
    double a;
    double c_stab;
    double mu;
    double r_nn;
    double r_cut;
    double length_x;
    double length_y;
    double length_z;
    int32_t recalc_mu;
} rgmin_exp_t;
/** Install the Packwood exponential preconditioner.
 *  \a distance is null for Cartesian pairs. \a anchor_i is null when
 *  \a nanchor is 0. \c mu NaN estimates the energy scale on the next step. */
rgmin_status_t rgmin_solver_set_exp(rgmin_solver_t *solver, const rgmin_exp_t *params,
                                    const uint32_t *pair_i, const uint32_t *pair_j,
                                    const double *distance, size_t npair,
                                    const uint32_t *anchor_i, const double *anchor_r,
                                    size_t nanchor);
/** Drop the preconditioner installed by \ref rgmin_solver_set_exp. */
void rgmin_solver_clear_exp(rgmin_solver_t *solver);
/** Energy scale, or NaN when none is installed. */
double rgmin_solver_exp_mu(const rgmin_solver_t *solver);

/** Which geometry preconditioner \ref rgmin_solver_set_precon installs.
 *  ABI minor 32. */
typedef enum rgmin_precon_kind_t {
    RGMIN_PRECON_NONE = 0,
    RGMIN_PRECON_EXP = 1,
    RGMIN_PRECON_LINDH = 2,
    RGMIN_PRECON_FF = 3,
    RGMIN_PRECON_AUTO = 4
} rgmin_precon_kind_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_precon_kind_t) == sizeof(int32_t),
               "rgmin_precon_kind_t is i32-wide; do not build this header with -fshort-enums");
#endif
/** \ref rgmin_precon_t::geometry_kind for \c RGMIN_PRECON_AUTO. */
#define RGMIN_GEOM_BULK 0
#define RGMIN_GEOM_SURFACE 1
#define RGMIN_GEOM_MOLECULE 2
#define RGMIN_GEOM_CLUSTER 3
/** One harmonic bond. Indices are \c uint32_t free-atom numbers. */
typedef struct rgmin_bond_t {
    uint32_t i, j;
    double k;
} rgmin_bond_t;
/** One harmonic angle at \c vertex. Indices are \c uint32_t. */
typedef struct rgmin_angle_t {
    uint32_t i, vertex, k;
    double k_theta;
} rgmin_angle_t;
/** Parameters for every geometry preconditioner.
 *
 *  \c struct_size is \c sizeof(rgmin_precon_t). This revision rejects a
 *  smaller value. A larger future struct is accepted; bytes past the
 *  fields this build knows stay unread.
 *  \c kind is \c uint32_t and holds a \c RGMIN_PRECON_* value.
 *  \c c_stab, \c mu, \c r_cut, and \c exp_a are NaN or <= 0 for the
 *  default or an estimate. \c exp_a is the Exp exponent A (default 3).
 *  \c cell holds orthorhombic lengths; a component <= 0 leaves that
 *  axis open. \c numbers and \c natoms are the free atoms (Lindh, AUTO).
 *  Ghost pointers are null when \c nghost is 0. Neighbour indices
 *  (\c nbr_i, \c nbr_j) and anchor indices (\c anchor_i) are
 *  \c size_t. Bond and angle indices are \c uint32_t. \c nbr_r null
 *  takes Cartesian distances on the next step. \c anchor_i bonds a
 *  free atom to a frozen atom at distance \c anchor_r. \c dihedrals
 *  nonzero adds the Lindh torsion term.
 *
 *  Morse bonds and FF harmonic torsions are Rust-only
 *  (\c MorseTerm, \c DihedralTerm). They are not fields of this struct.
 *  Every index is checked against \c dim/3 inside
 *  \ref rgmin_solver_set_precon. */
typedef struct rgmin_precon_t {
    uint32_t struct_size;
    uint32_t kind;
    double c_stab, mu, r_cut;
    double cell[3];
    const int32_t *numbers;
    size_t natoms;
    const int32_t *ghost_z;
    const double *ghost_pos;
    size_t nghost;
    const size_t *nbr_i;
    const size_t *nbr_j;
    const double *nbr_r;
    size_t npairs;
    const size_t *anchor_i;
    const double *anchor_r;
    size_t nanchor;
    double exp_a;
    const rgmin_bond_t *bonds;
    size_t nbonds;
    const rgmin_angle_t *angles;
    size_t nangles;
    int32_t dihedrals;
    int32_t geometry_kind;
} rgmin_precon_t;
/** Zero \a p, set \c struct_size, and select \a kind. Optional doubles
 *  are NaN so the defaults apply. */
void rgmin_precon_init(rgmin_precon_t *p, rgmin_precon_kind_t kind);
/** Install, replace, or clear the geometry preconditioner.
 *  \c RGMIN_PRECON_NONE clears it. \c RGMIN_PRECON_AUTO calls Rust
 *  \c select_precon with this session's method, so the caller does
 *  not pass an optimiser family. FIRE resolves to the identity.
 *  A force field is present when \c nbonds or \c nangles is nonzero,
 *  and it is used only for a covalent molecule or cluster.
 *  Indices are rejected here. */
rgmin_status_t rgmin_solver_set_precon(rgmin_solver_t *s, const rgmin_precon_t *p);
/** Drop whichever geometry preconditioner is installed. */
void rgmin_solver_clear_precon(rgmin_solver_t *s);
/** Energy scale of the installed geometry preconditioner.
 *  NaN when none is installed or the scale is not estimated yet. */
double rgmin_solver_precon_scale(const rgmin_solver_t *s);

/** How a session takes a proposed step (eOn lbfgs_accept). */
typedef enum rgmin_accept_t {
    RGMIN_ACCEPT_NONE = 0,
    RGMIN_ACCEPT_ENERGY = 1,
    RGMIN_ACCEPT_NONMONOTONE = 2,
    /** Clipped step, one oracle, no line search. Released 0.5.2. */
    RGMIN_ACCEPT_STEP = 3
} rgmin_accept_t;
void rgmin_solver_set_accept(rgmin_solver_t *solver, rgmin_accept_t accept);

/** FIRE schedule. */
typedef enum rgmin_fire_variant_t {
    RGMIN_FIRE_RGMIN = 0,
    RGMIN_FIRE_GUENOLE2020 = 1
} rgmin_fire_variant_t;
void rgmin_solver_set_fire_variant(rgmin_solver_t *solver,
                                   rgmin_fire_variant_t variant);
/** Split a quick-min session at `at`: atoms `[0, at)`, cell `[at, n)`.
 *  Returns RGMIN_INVALID_PARAMETER when the session is not quick-min or
 *  `at` is not an interior index. ABI minor 29. */
rgmin_status_t rgmin_solver_set_quickmin_cell(rgmin_solver_t *solver, size_t at);
/** Quick-min options: the rgmin default, the eOn step (fixed time step,
 *  no clamp rescale), or one projection per atom. A no-op on other
 *  sessions. ABI minor 33. */
typedef enum rgmin_quickmin_variant_t {
    RGMIN_QUICKMIN_RGMIN = 0,
    RGMIN_QUICKMIN_EON = 1,
    RGMIN_QUICKMIN_PER_ATOM = 2
} rgmin_quickmin_variant_t;
void rgmin_solver_set_quickmin_variant(rgmin_solver_t *solver,
                                       rgmin_quickmin_variant_t variant);

/** Approximate Wolfe value window for Euclidean RGMIN_ACCEPT_NONE sessions.
 * Reject negative or nonfinite values without changing the session.
 * Zero permits no rise. Gradient stopping and physical acceptance are unchanged.
 */
rgmin_status_t rgmin_solver_set_objective_roundoff(rgmin_solver_t *solver,
                                                  double relative);

/** Gradient norm a session compares to \c gtol and writes to
 *  \c rgmin_report_t.grad_norm. MAX_ATOM is the largest norm over
 *  (x, y, z) triples (ASE fmax, eOn max_atom, LAMMPS max). INF is the
 *  largest absolute component (eOn max_component, LAMMPS inf). RMS is
 *  the two-norm over sqrt(n) (eOn rms), ABI minor 32. */
typedef enum rgmin_stop_norm_t {
    RGMIN_STOP_NORM_TWO = 0,
    RGMIN_STOP_NORM_MAX_ATOM = 1,
    RGMIN_STOP_NORM_INF = 2,
    RGMIN_STOP_NORM_RMS = 3
} rgmin_stop_norm_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_stop_norm_t) == sizeof(int32_t),
               "rgmin_stop_norm_t is i32-wide; do not build this header with -fshort-enums");
#endif
/** Select the stop norm. TWO is the default. Returns 0, or 1 on a null
 *  session or an unknown integer. ABI minor 29. */
int32_t rgmin_solver_set_stop_norm(rgmin_solver_t *solver, rgmin_stop_norm_t norm);
/** FIRE 2.0 parameter set. GUENOLE2020 is Table 1 of Guenole et al.,
 *  Comput. Mater. Sci. 175, 109584 (2020): dt_max 10 dt, dt_min
 *  0.02 dt, N_delay 20, alpha_start 0.25, f_inc 1.1, f_dec 0.5,
 *  f_alpha 0.99, 2000 non-positive-power steps. */
typedef enum rgmin_fire_preset_t {
    RGMIN_FIRE_PRESET_DEFAULT = 0,
    RGMIN_FIRE_PRESET_GUENOLE2020 = 1
} rgmin_fire_preset_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_fire_preset_t) == sizeof(int32_t),
               "rgmin_fire_preset_t is i32-wide; do not build this header with -fshort-enums");
#endif
/** Select the FIRE 2.0 parameter set and restart the integrator from
 *  \c istep. Returns 0, or 1 on a null session, an unknown integer or
 *  a session that is not RGMIN_FIRE2. ABI minor 30. */
int32_t rgmin_solver_set_fire_preset(rgmin_solver_t *solver, rgmin_fire_preset_t preset);
/** ABC-FIRE (Echeverri Restrepo and Andric, Comput. Mater. Sci. 218,
 *  111978 (2023)) on when \c on is nonzero. The per-atom cap, else
 *  \c maxmove, bounds each displacement component. Returns 0, or 1 on
 *  a null session or a session that is not RGMIN_FIRE2. ABI minor 30. */
int32_t rgmin_solver_set_fire_abc(rgmin_solver_t *solver, int32_t on);
/** Per-atom cap of the default eOn L-BFGS step. The first step probes
 *  curvature. A proposal at least as long as the cap clears the memory
 *  and steps along H0 times the force. A positive \c max_atom sets the
 *  cap. A non-positive value restores 0.2. Returns 0, or 1 on a null
 *  session, a session that is not RGMIN_LBFGS, a non-finite cap, a
 *  non-Euclidean manifold, a distance-constraint set, or HiGHS.
 *  ABI minor 35. */
int32_t rgmin_solver_set_lbfgs_capped(rgmin_solver_t *solver, double max_atom);
/** ASE length control: shorten the two-loop step and keep the pairs.
 *  A non-positive cap restores 0.2. Same refusal rules as
 *  rgmin_solver_set_lbfgs_capped. ABI minor 35. */
int32_t rgmin_solver_set_lbfgs_scaled(rgmin_solver_t *solver, double max_atom);
/** Opt into the strong Wolfe line search. The default L-BFGS session
 *  does not search a line. Returns 0, or 1 on a null session or a
 *  session that is not RGMIN_LBFGS. ABI minor 34. */
int32_t rgmin_solver_set_lbfgs_wolfe(rgmin_solver_t *solver);
/** Per-atom cap (eOn \c maxAtomMotionAppliedV). While the L-BFGS step
 *  is the default cap, this replaces that cap. After
 *  rgmin_solver_set_lbfgs_wolfe, and without a distance constraint set,
 *  this selects the geometry step and ignores \ref rgmin_solver_set_accept.
 *  With a set, L-BFGS keeps its
 *  two-loop step under the acceptance mode and the cap scales each trial
 *  before it returns to the set. A nominal steepest trial at least 64 times the cap takes one
 *  finite-difference curvature probe, clipped to the objective bounds.
 *  Ordinary starts use 0.1 I and a five-value nonmonotone window.
 *  An installed exponential preconditioner replaces that scalar seed,
 *  including the steepest fallbacks. Newton and dogleg scale a saddle
 *  kick to this length. Non-positive clears it. */
void rgmin_solver_set_atom_maxmove(rgmin_solver_t *solver, double maxmove);
/** Capped Cartesian L-BFGS takes the geometry step: eOn's L-BFGS with a
 *  finite-difference or sy/yy initial curvature, resets on negative
 *  curvature, and an energy test with halvings. Nonzero turns it on; off,
 *  the per-atom cap only clips the two-loop trial. Needs a per-atom cap. */
void rgmin_solver_set_geometry_step(rgmin_solver_t *solver, int32_t enabled);
void rgmin_solver_set_project_rigid(rgmin_solver_t *solver, int32_t enabled);
/** 0 off, 1 auto, 2 translation, 3 translation and rotation.
 *  Nonzero on \ref rgmin_solver_set_project_rigid is auto. */
typedef enum rgmin_rigid_mode_t {
    RGMIN_RIGID_OFF = 0,
    RGMIN_RIGID_AUTO = 1,
    RGMIN_RIGID_TRANSLATION = 2,
    RGMIN_RIGID_TRANSLATION_ROTATION = 3
} rgmin_rigid_mode_t;
void rgmin_solver_set_rigid_mode(rgmin_solver_t *solver, int32_t mode);
void rgmin_solver_set_rigid_mass_metric(rgmin_solver_t *solver, int32_t enabled);
void rgmin_solver_set_rigid_mass_weighted(rgmin_solver_t *solver, int32_t enabled);
/** Periodic axes. A slab is (1, 1, 0). */
void rgmin_solver_set_pbc_axes(rgmin_solver_t *solver, int32_t x, int32_t y, int32_t z);
/** Length 3N. A nonzero byte freezes that coordinate. NULL clears the mask. */
void rgmin_solver_set_frozen(rgmin_solver_t *solver, const uint8_t *frozen, size_t len);
/** Broken translations, then broken rotations. The host sets both. */
void rgmin_solver_set_broken_axes(rgmin_solver_t *solver, int32_t tx, int32_t ty,
                                int32_t tz, int32_t rx, int32_t ry, int32_t rz);
void rgmin_solver_set_extra_updates(rgmin_solver_t *solver, size_t extra);
void rgmin_solver_set_cautious(rgmin_solver_t *solver, double eps, double alpha);
/** Opt-in L-BFGS NEB guards: distance, angle and curvature resets and an
 *  empty-memory scale of 0.01. Nonzero enables; other methods ignore it.
 *  ABI minor 28. */
void rgmin_solver_set_lbfgs_neb_guards(rgmin_solver_t *solver, int32_t enabled);

/** HiGHS feasible-set step. Nonzero enables it. Returns 0, or 1 if this
 *  build has no highs feature. */
int32_t rgmin_solver_set_highs(rgmin_solver_t *solver, int32_t enabled);
/** Per-coordinate box on \c x+p. A NULL side is unbounded. \a n is the
 *  length of each non-NULL side and must match the session dimension.
 *  Returns 0, or 1 if this build has no highs feature. */
int32_t rgmin_solver_set_box(rgmin_solver_t *solver, const double *lower,
                           const double *upper, size_t n);
/** L_inf trust radius on the HiGHS step. Non-positive clears it.
 *  Returns 0, or 1 if this build has no highs feature. */
int32_t rgmin_solver_set_trust(rgmin_solver_t *solver, double radius);
/** Alias with the same radius and feature contract. */
int32_t rgmin_solver_set_highs_trust(rgmin_solver_t *solver, double radius);

/** Append one sparse equality \c a.p = rhs. \a idx / \a coef length \a nnz.
 *  Returns 0, or 1 if this build has no highs feature. */
int32_t rgmin_solver_add_equality(rgmin_solver_t *solver, const size_t *idx,
                                const double *coef, size_t nnz, double rhs);
/** Drop every stored HiGHS equality. Returns 0, or 1 without highs. */
int32_t rgmin_solver_clear_equalities(rgmin_solver_t *solver);
/** HiGHS `solver` token. Constrained dest defaults to
 *  \c RGMIN_HIGHS_IPM. Unknown integers return 1. Returns 0, or 1
 *  without highs. */
typedef enum rgmin_highs_solver_t {
    RGMIN_HIGHS_CHOOSE = 0,
    RGMIN_HIGHS_SIMPLEX = 1,
    RGMIN_HIGHS_IPM = 2,
    RGMIN_HIGHS_IPX = 3,
    RGMIN_HIGHS_HIPO = 4,
    RGMIN_HIGHS_PDLP = 5,
    RGMIN_HIGHS_HIPDLP = 6,
    RGMIN_HIGHS_QPASM = 7
} rgmin_highs_solver_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_highs_solver_t) == sizeof(int32_t),
               "rgmin_highs_solver_t is i32-wide; do not build this header with -fshort-enums");
#endif
int32_t rgmin_solver_set_highs_solver(rgmin_solver_t *solver,
                                      rgmin_highs_solver_t kind);
/** HiGHS `run_crossover`. Constrained dest defaults to
 *  \c RGMIN_HIGHS_CROSSOVER_OFF. Unknown integers return 1. */
typedef enum rgmin_highs_crossover_t {
    RGMIN_HIGHS_CROSSOVER_CHOOSE = 0,
    RGMIN_HIGHS_CROSSOVER_ON = 1,
    RGMIN_HIGHS_CROSSOVER_OFF = 2
} rgmin_highs_crossover_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_highs_crossover_t) == sizeof(int32_t),
               "rgmin_highs_crossover_t is i32-wide; do not build this header with -fshort-enums");
#endif
int32_t rgmin_solver_set_highs_crossover(rgmin_solver_t *solver,
                                         rgmin_highs_crossover_t kind);
/** HiGHS callback kind. Integers match HiGHS `HighsCallbackType`. */
typedef enum rgmin_highs_cb_kind_t {
    RGMIN_HIGHS_CB_LOGGING = 0,
    RGMIN_HIGHS_CB_SIMPLEX_INTERRUPT = 1,
    RGMIN_HIGHS_CB_IPM_INTERRUPT = 2
} rgmin_highs_cb_kind_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_highs_cb_kind_t) == sizeof(int32_t),
               "rgmin_highs_cb_kind_t is i32-wide; do not build this header with -fshort-enums");
#endif
/** HiGHS user callback. \a interrupt nonzero stops the solve.
 *  Stored on the session for every later constrained step. */
typedef void (*rgmin_highs_callback_t)(int32_t kind, const char *message,
                                       int32_t *interrupt, void *user);
int32_t rgmin_solver_set_highs_callback(rgmin_solver_t *solver,
                                        rgmin_highs_callback_t cb, void *user);
/** Embedded manifold. Euclidean is the default.
 *  Molecular clusters use RIGID_QUOTIENT (Sella Cartesian T+R,
 *  R^{3N}/SE(3)) or MW_RIGID (Page-McIver / Sella IRC Eckart).
 *  SO3 is length 9; SE3 is length 12. Oblique is n-by-m via
 *  rgmin_solver_set_oblique. Stiefel p>1 is rgmin_solver_set_stiefel.
 *  Reserved: 7 SPD, 8 Grassmann, 9 Hyperbolic, 10 Poincare.
 *  Token 15 is skew-symmetric n-by-n, n >= 2.
 *  Token 16 is complex Euclidean C^n, packed interleaved, length 2n.
 *  Token 17 is the singleton {A} of packed length n (constantfactory).
 *  Token 18 is doubly-stochastic n-by-n (multinomialdoublystochasticfactory).
 *  Token 19 is symmetric doubly-stochastic n-by-n
 *  (multinomialsymmetricfactory).
 *  Token 20 is the complex unit sphere C^n, packed interleaved, length 2n.
 *  Token 21 is the positive orthant of packed length n (positivefactory).
 *  Token 22 is centered m-by-n matrices (centeredmatrixfactory).
 *  Reserved 7-10 unused. */
typedef enum rgmin_manifold_t {
    RGMIN_MANIFOLD_EUCLIDEAN = 0,
    RGMIN_MANIFOLD_SPHERE = 1,
    RGMIN_MANIFOLD_SO3 = 2,
    RGMIN_MANIFOLD_STIEFEL = 3,
    RGMIN_MANIFOLD_SE3 = 4,
    RGMIN_MANIFOLD_RIGID_QUOTIENT = 5,
    RGMIN_MANIFOLD_MW_RIGID = 6,
    RGMIN_MANIFOLD_OBLIQUE = 11,
    RGMIN_MANIFOLD_MULTINOMIAL = 12,
    RGMIN_MANIFOLD_COMPLEX_CIRCLE = 13,
    RGMIN_MANIFOLD_SYMMETRIC = 14,
    RGMIN_MANIFOLD_SKEWSYMMETRIC = 15,
    RGMIN_MANIFOLD_EUCLIDEAN_COMPLEX = 16,
    RGMIN_MANIFOLD_CONSTANT = 17,
    RGMIN_MANIFOLD_MULTINOMIAL_DS = 18,
    RGMIN_MANIFOLD_MULTINOMIAL_SYM = 19,
    RGMIN_MANIFOLD_SPHERE_COMPLEX = 20,
    RGMIN_MANIFOLD_SPD = 7,
    RGMIN_MANIFOLD_GRASSMANN = 8,
    RGMIN_MANIFOLD_HYPERBOLIC = 9,
    RGMIN_MANIFOLD_POINCARE = 10,
    RGMIN_MANIFOLD_POSITIVE = 21,
    RGMIN_MANIFOLD_CENTERED_MATRIX = 22,
    RGMIN_MANIFOLD_UNITARY = 23
} rgmin_manifold_t;
void rgmin_solver_set_manifold(rgmin_solver_t *solver, rgmin_manifold_t manifold);
/** Oblique OB(n,m): product of m unit spheres in R^n, column-major. */
void rgmin_solver_set_oblique(rgmin_solver_t *solver, size_t n, size_t m);
/** Stiefel St(n,p). p = 1 is the sphere packing. p > 1 is n*p. */
void rgmin_solver_set_stiefel(rgmin_solver_t *solver, size_t n, size_t p);
/** n unit-modulus complex numbers. Packed interleaved, length 2n. */
void rgmin_solver_set_complex_circle(rgmin_solver_t *solver, size_t n);
/** Complex Euclidean C^n. Packed interleaved, length 2n. */
void rgmin_solver_set_euclidean_complex(rgmin_solver_t *solver, size_t n);
/** Singleton of packed length n. manopt constantfactory. */
void rgmin_solver_set_constant(rgmin_solver_t *solver, size_t n);
/** Doubly-stochastic n-by-n, packed n*n. manopt
 *  multinomialdoublystochasticfactory. Token 18 defaults to n = 2. */
void rgmin_solver_set_multinomial_ds(rgmin_solver_t *solver, size_t n);
/** Symmetric doubly-stochastic n-by-n, packed n*n. manopt
 *  multinomialsymmetricfactory. Token 19 defaults to n = 2. */
void rgmin_solver_set_multinomial_sym(rgmin_solver_t *solver, size_t n);
/** Complex unit sphere in C^n. Packed interleaved, length 2n.
 *  manopt spherecomplexfactory. Token 20 defaults to n = 1. */
void rgmin_solver_set_sphere_complex(rgmin_solver_t *solver, size_t n);
/** Positive orthant of packed length n. manopt positivefactory.
 *  Token 21 defaults to n = 1. Reserved 7-10 unused. */
void rgmin_solver_set_positive(rgmin_solver_t *solver, size_t n);
/** Centered m-by-n matrices, packed m*n. manopt centeredmatrixfactory.
 *  Token 22 defaults to 2-by-2 centered columns. center_rows != 0
 *  centers rows. Reserved 7-10 unused. */
void rgmin_solver_set_centered_matrix(rgmin_solver_t *solver, size_t m, size_t n,
                                    int32_t center_rows);
void rgmin_solver_set_factor_shape(rgmin_solver_t *solver, size_t n, size_t p);
void rgmin_solver_set_unitary(rgmin_solver_t *solver, size_t n);

/** Per-atom masses for MW_RIGID. n_atoms == 0 or masses == NULL
 *  restores unit mass. */
void rgmin_solver_set_masses(rgmin_solver_t *solver, const double *masses,
                           size_t n_atoms);
/** Periodic cell. Nonzero drops rotation (Sella proj_rot): R^{3N}/T(3). */
void rgmin_solver_set_periodic(rgmin_solver_t *solver, int32_t enabled);

/** Why a distance constraint set was refused or could not be met.
 *  Integers follow the Rust ConstraintError variants. ABI minor 31. */
typedef enum rgmin_constraint_error_t {
    /** The last error on this thread was not a constraint error. */
    RGMIN_CONSTRAINT_NONE = 0,
    /** Pairs, lengths or options that describe no set. */
    RGMIN_CONSTRAINT_MALFORMED = 1,
    /** A pair names an atom past the end of the coordinates. */
    RGMIN_CONSTRAINT_OUT_OF_RANGE = 2,
    /** A mass that is not positive and finite. */
    RGMIN_CONSTRAINT_MASS = 3,
    /** A session setting the set cannot run with: a manifold other than
     *  Euclidean, the exponential preconditioner, the HiGHS step, a
     *  periodic cell, PSO, or rigid projection at an active bound. */
    RGMIN_CONSTRAINT_UNSUPPORTED = 4,
    /** SHAKE, LINCS or the tangent solve spent its derived budget. */
    RGMIN_CONSTRAINT_NOT_CONVERGED = 5,
    /** A bond turned too far from its anchor direction. */
    RGMIN_CONSTRAINT_REVERSED = 6,
    /** The constraints of a block are linearly dependent. */
    RGMIN_CONSTRAINT_DEPENDENT = 7,
    /** LINCS was requested on a block where its expansion diverges. */
    RGMIN_CONSTRAINT_LINCS_UNSAFE = 8,
    /** SETTLE has no solution for a rigid triangle. */
    RGMIN_CONSTRAINT_SETTLE_DEGENERATE = 9,
    /** A constrained atom reached its coordinate bounds. */
    RGMIN_CONSTRAINT_BOUND = 10,
    /** A coordinate, mass or residual is not finite. */
    RGMIN_CONSTRAINT_NON_FINITE = 11
} rgmin_constraint_error_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_constraint_error_t) == sizeof(int32_t),
               "rgmin_constraint_error_t is i32-wide; do not build this header with -fshort-enums");
#endif
/** Kind of the last error on this thread when it was a constraint error,
 *  else RGMIN_CONSTRAINT_NONE. ABI minor 31. */
rgmin_constraint_error_t rgmin_last_constraint_error(void);
/** Solver for the blocks SETTLE does not take. ABI minor 31. */
typedef enum rgmin_constraint_solver_t {
    /** LINCS where its expansion converges at least as fast per pass as
     *  the Gauss-Seidel bound of SHAKE, SHAKE elsewhere. */
    RGMIN_CONSTRAINT_SOLVER_AUTO = 0,
    /** Gauss-Seidel SHAKE on every block. */
    RGMIN_CONSTRAINT_SOLVER_SHAKE = 1,
    /** LINCS on every block. A diverging block is RGMIN_CONSTRAINT_LINCS_UNSAFE. */
    RGMIN_CONSTRAINT_SOLVER_LINCS = 2
} rgmin_constraint_solver_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_constraint_solver_t) == sizeof(int32_t),
               "rgmin_constraint_solver_t is i32-wide; do not build this header with -fshort-enums");
#endif
/** Opaque set of distances |x_i - x_j| = d over packed 3 N coordinates.
 *  A session copies it, so the caller frees it whenever it likes. */
typedef struct rgmin_constraints_t rgmin_constraints_t;
/** Fix |x_i - x_j| = lengths[k] for pairs (pair_i[k], pair_j[k]), with unit
 *  masses, the automatic solver, SETTLE on rigid triangles and the
 *  rounding floor as tolerance. Null on bad arguments, with
 *  rgmin_last_error and rgmin_last_constraint_error set. ABI minor 31. */
rgmin_constraints_t *rgmin_constraints_create(const uint32_t *pair_i, const uint32_t *pair_j,
                                              const double *lengths, size_t npair);
/** Fix each pair at its distance in \a x of length \a n. ABI minor 31. */
rgmin_constraints_t *rgmin_constraints_create_at(const uint32_t *pair_i,
                                                 const uint32_t *pair_j, size_t npair,
                                                 const double *x, size_t n);
/** Release a set. Null is a no-op. */
void rgmin_constraints_free(rgmin_constraints_t *set);
/** One mass per atom; corrections and projections use the metric M^-1. */
rgmin_status_t rgmin_constraints_set_masses(rgmin_constraints_t *set, const double *masses,
                                            size_t n_atoms);
/** Solver for the blocks SETTLE does not take. An unknown integer is
 *  RGMIN_INVALID_PARAMETER. */
rgmin_status_t rgmin_constraints_set_solver(rgmin_constraints_t *set,
                                            rgmin_constraint_solver_t solver);
/** SETTLE on rigid triangles when \a on is nonzero (the default). */
rgmin_status_t rgmin_constraints_set_settle(rgmin_constraints_t *set, int32_t on);
/** Relative tolerance on |d^2 - r^2| / (2 d^2), the GROMACS shake-tol
 *  measure. Zero (the default) is the rounding floor of each constraint,
 *  and a smaller tolerance is raised to it. */
rgmin_status_t rgmin_constraints_set_tolerance(rgmin_constraints_t *set, double tolerance);
/** Fix the LINCS expansion order (GROMACS lincs-order). Zero derives it. */
rgmin_status_t rgmin_constraints_set_lincs_order(rgmin_constraints_t *set, size_t order);
/** Largest ratio of a relative residual to its tolerance at \a x: at most
 *  one when every constraint is met. NaN on a null argument, a length the
 *  set does not fit, or a non-finite residual. */
double rgmin_constraints_excess(const rgmin_constraints_t *set, const double *x, size_t n);
/** Return \a x to the set by corrections along the bonds of \a anchor, the
 *  point \a x was stepped from. On an error \a x is unchanged. */
rgmin_status_t rgmin_constraints_constrain(const rgmin_constraints_t *set,
                                           const double *anchor, double *x, size_t n);
/** Copy \a set into the session. Every later step keeps each constraint to
 *  its tolerance. Methods and Accept modes all run on the set; a setting
 *  the set cannot run with is RGMIN_CONSTRAINT_ERROR here or at the next
 *  step, and a step that cannot meet the set leaves \a x unchanged.
 *  ABI minor 31. */
rgmin_status_t rgmin_solver_set_distance_constraints(rgmin_solver_t *solver,
                                                     const rgmin_constraints_t *set);
/** Drop the session's set and its method memory. ABI minor 31. */
void rgmin_solver_clear_distance_constraints(rgmin_solver_t *solver);
/**
 * One outer iteration: direction, line search, curvature update.
 * \a eval and \a grad are valid for this call only. \a x is in/out.
 */
rgmin_status_t rgmin_solver_step(rgmin_solver_t *solver, rgmin_eval_fn eval,
                             rgmin_grad_fn grad, void *user,
                             DLManagedTensorVersioned *x, rgmin_report_t *out);
/** One Newton / RFO iteration. \a hess writes a length-\c n*n Hessian. */
rgmin_status_t rgmin_solver_step_hess(rgmin_solver_t *solver, rgmin_eval_fn eval,
                                  rgmin_grad_fn grad, rgmin_hess_fn hess,
                                  void *user, DLManagedTensorVersioned *x,
                                  rgmin_report_t *out);
/** Like \ref rgmin_solver_step with one fused (f, g) callback. */
rgmin_status_t rgmin_solver_step_fg(rgmin_solver_t *solver, rgmin_evalgrad_fn evalgrad,
                                void *user, DLManagedTensorVersioned *x,
                                rgmin_report_t *out);
/** Like \ref rgmin_solver_step_hess with one fused (f, g) callback. */
rgmin_status_t rgmin_solver_step_hess_fg(rgmin_solver_t *solver,
                                     rgmin_evalgrad_fn evalgrad, rgmin_hess_fn hess,
                                     void *user, DLManagedTensorVersioned *x,
                                     rgmin_report_t *out);
/** Record `s = x+ - x`, `y = g+ - g` from the caller's previous outer.
 *  Does not evaluate. Returns 0 on success. A near-zero `y` is refused
 *  by the L-BFGS pair gate and still returns 0. */
int32_t rgmin_solver_push_pair(rgmin_solver_t *solver, const double *s,
                             const double *y, size_t n);
/** Packed length of x. A null session returns zero. */
size_t rgmin_solver_dim(const rgmin_solver_t *solver);
/** Outer steps completed. A null session returns zero. */
size_t rgmin_solver_steps(const rgmin_solver_t *solver);
/** Gradient evaluations on this session. Null returns zero.
 *  A finite-difference product is \ref rgmin_solver_hvp_calls, not a gradient. */
size_t rgmin_solver_gradient_calls(const rgmin_solver_t *solver);
/** Hessian-vector products on this session. Null returns zero.
 *  Analytic and forward-difference products each count as one. */
size_t rgmin_solver_hvp_calls(const rgmin_solver_t *solver);
/** Number of accepted L-BFGS pairs. Null and other methods return zero. */
size_t rgmin_solver_pair_count(const rgmin_solver_t *solver);
/** Copy the gradient behind the last successful step report into
 *  \a out. \a n must equal the session dimension. Fails before the
 *  first step. ABI minor 29. */
rgmin_status_t rgmin_solver_report_gradient(const rgmin_solver_t *solver,
                                            double *out, size_t n);
/** Line-search rule for \ref rgmin_linesearch. Integers are stable. */
typedef enum rgmin_linesearch_kind_t {
    /** Brent on a golden-section bracket. */
    RGMIN_LINESEARCH_BRENT = 0,
    /** Armijo backtracking. */
    RGMIN_LINESEARCH_BACKTRACKING = 1,
    /** Goldstein two-sided condition. */
    RGMIN_LINESEARCH_GOLDSTEIN = 2,
    /** Strong Wolfe with zoom. */
    RGMIN_LINESEARCH_WOLFE = 3
} rgmin_linesearch_kind_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_linesearch_kind_t) == sizeof(int32_t),
               "rgmin_linesearch_kind_t is i32-wide; do not build this header with -fshort-enums");
#endif
/** Reject invalid constants or a zero trial limit without changing the session.
 *  The 0.5.2 header spelled this kind \c rgmin_linesearch_t. That name is
 *  the opaque search handle added with \ref rgmin_linesearch. */
rgmin_status_t rgmin_solver_set_linesearch(rgmin_solver_t *solver,
                                         rgmin_linesearch_kind_t kind,
                                         double c1, double c2, size_t maxiter);


/** Opaque caller-driven line search. The search does not call an objective. */
typedef struct rgmin_linesearch rgmin_linesearch_t;

/** Start a line search from `x` along `dir`.
 *  The first \ref rgmin_linesearch call supplies the energy and gradient
 *  at `x`. Later calls supply them at the trial the previous call wrote.
 *  Null on a null pointer, a zero length, or an unknown kind. */
rgmin_linesearch_t *rgmin_linesearch_create(rgmin_linesearch_kind_t kind,
                                            const double *x, const double *dir,
                                            size_t n, double istep, double c1,
                                            double c2, double beta, size_t maxiter,
                                            double tol);
/** Advance the search with the caller's energy and gradient.
 *  Returns 0 and sets `*need_eval` to 1 when `x_out` is the next trial,
 *  or to 0 when `x_out` is the accepted point (`*value` and `*alpha` are
 *  then the accepted energy and absolute step). Returns 1 on a null
 *  argument or a gradient whose length is not the search dimension.
 *  The search never calls an objective. */
int32_t rgmin_linesearch(rgmin_linesearch_t *search, double f, const double *g,
                         double *x_out, double *alpha, double *value,
                         int32_t *need_eval);
/** Release a search from \ref rgmin_linesearch_create. Null is a no-op. */
void rgmin_linesearch_free(rgmin_linesearch_t *search);

/** Opaque `(s, y)` memory for \ref rgmin_lbfgs_direction.
 *  The handle accumulates pairs and does not evaluate an objective. */
typedef struct rgmin_lbfgs_pairs rgmin_lbfgs_pairs_t;

/** Apply H0 to `v`, writing `n` doubles to `out`. Return 0 on success.
 *  The callback is not an objective and must not call back into the
 *  same pair handle. */
typedef int32_t (*rgmin_h0_fn)(void *user, const double *v, double *out, size_t n);

/** Pair memory of dimension `n`. `capacity` 0 keeps ten pairs.
 *  Null when `n` is 0. */
rgmin_lbfgs_pairs_t *rgmin_lbfgs_pairs_create(size_t n, size_t capacity);
/** Append one pair. On success, `*stored` is 1 when the pair was kept
 *  and 0 when the curvature gate declined it. `stored` may be NULL.
 *  A length other than the handle dimension is \c RGMIN_INVALID_PARAMETER. */
rgmin_status_t rgmin_lbfgs_pairs_push(rgmin_lbfgs_pairs_t *pairs, const double *s,
                                      const double *y, size_t n, int32_t *stored);
/** Number of stored pairs. A null handle returns zero. */
size_t rgmin_lbfgs_pairs_count(const rgmin_lbfgs_pairs_t *pairs);
/** Drop every stored pair. A null handle is a no-op. */
void rgmin_lbfgs_pairs_clear(rgmin_lbfgs_pairs_t *pairs);
/** Release a handle from \ref rgmin_lbfgs_pairs_create. Null is a no-op. */
void rgmin_lbfgs_pairs_free(rgmin_lbfgs_pairs_t *pairs);
/** Direction `-H g`. Pass `h0_diag` or `apply_h0`, not both and not
 *  neither. An empty memory writes `-H0 g`. Does not evaluate an
 *  objective. ABI minor 38. */
rgmin_status_t rgmin_lbfgs_direction(const rgmin_lbfgs_pairs_t *pairs,
                                     const double *grad, const double *h0_diag,
                                     rgmin_h0_fn apply_h0, void *user, double *dir,
                                     size_t n);

/** Two-loop `d = -H g` with no evaluation and no push. Empty memory
 *  is steepest descent (`-g`). Writes \a n entries to \a dir. */
rgmin_status_t rgmin_solver_search_direction(rgmin_solver_t *solver,
                                          const double *grad, double *dir,
                                          size_t n);

/** Directional curvature `d^T H(x) d`. Non-success falls back to the probe. */
typedef rgmin_status_t (*rgmin_curv_fn)(void *user, const DLManagedTensorVersioned *x,
                                    const DLManagedTensorVersioned *d,
                                    double *curv_out);

typedef struct rgmin_scg_params_t {
    double sigma0;
    double lambda;
    double lambda_limit;
    double tol_sol;
    double tol_func;
    /** Leaf conjugacy. Literal: 0 is Fletcher-Reeves. Values outside
     *  0..8 are \c RGMIN_INVALID_PARAMETER. 8 is Polak-Ribiere-Plus.
     *  Ignored when \a params is NULL (Netlab Polak-Ribiere). */
    rgmin_conjugacy_t conjugacy;
} rgmin_scg_params_t;

/**
 * Møller SCG. \a curv may be NULL (finite-difference probe).
 * The exported symbol is \c rgmin_minimize_scg.
 *
 * \a params NULL selects ScgParams defaults, Polak-Ribiere, and
 * Restart::Never. A filled \a params takes \c conjugacy literally
 * (0 is Fletcher-Reeves). Restart is not a C token; the waist
 * passes Never. Møller n-success reset stays inside the Rust loop.
 * gpr_optim RgminScg.inl writes \c RGMIN_CONJUGACY_LIU_STOREY. In-tree
 * SCG.inl does not call this entry.
 *
 * A positive \c ctrl->maxmove is the Euclidean cap. Zero keeps the
 * defaults. A negative value leaves every step cap off.
 */
rgmin_status_t rgmin_minimize_scg(rgmin_eval_fn eval, rgmin_grad_fn grad, rgmin_curv_fn curv,
                                void *user, DLManagedTensorVersioned *x,
                                const rgmin_control_t *ctrl,
                                const rgmin_scg_params_t *params, rgmin_report_t *out);

/** Accepted-step SCG convergence rule. Both require objective convergence. */
typedef enum rgmin_scg_step_tolerance_t {
    /** ||alpha d||_inf < tol_sol. */
    RGMIN_SCG_ABSOLUTE_INFINITY = 0,
    /** ||alpha d||_2 < tol_sol * (1 + ||accepted x||_2). */
    RGMIN_SCG_RELATIVE_EUCLIDEAN = 1
} rgmin_scg_step_tolerance_t;

/** Optional SCG policies, separate from the stable parameter record. */
typedef struct rgmin_scg_options_t {
    /** Literal rgmin_scg_step_tolerance_t. Unknown integers are rejected. */
    int32_t step_tolerance;
} rgmin_scg_options_t;

/** SCG with explicit convergence policy. NULL options selects the legacy rule. */
rgmin_status_t rgmin_minimize_scg_with_options(
    rgmin_eval_fn eval, rgmin_grad_fn grad, rgmin_curv_fn curv, void *user,
    DLManagedTensorVersioned *x, const rgmin_control_t *ctrl,
    const rgmin_scg_params_t *params, const rgmin_scg_options_t *options,
    rgmin_report_t *out);




/** Closed eigensolver tag. Integers match schema/eigen.capnp. */
typedef enum rgmin_eigen_kind_t {
    RGMIN_EIGEN_LANCZOS = 0,
    RGMIN_EIGEN_RAYLEIGH_RITZ = 1,
    RGMIN_EIGEN_JACOBI_DAVIDSON = 2,
    RGMIN_EIGEN_LOBPCG = 3,
    RGMIN_EIGEN_PRIMME = 4,
    RGMIN_EIGEN_SLEPC = 5,
    RGMIN_EIGEN_CHASE = 6,
    RGMIN_EIGEN_ELPA = 7,
    RGMIN_EIGEN_ELPA2 = 8,
    RGMIN_EIGEN_SLATE = 9,
    RGMIN_EIGEN_MAGMA = 10,
    RGMIN_EIGEN_CUSOLVER = 11,
    RGMIN_EIGEN_DLA_FUTURE = 12,
    RGMIN_EIGEN_EIGENEXA = 13,
    RGMIN_EIGEN_DIMER = 14,
    RGMIN_EIGEN_LIBKRYLOV = 15
} rgmin_eigen_kind_t;

typedef struct rgmin_eigen_params_t {
    /** Literal rgmin_eigen_kind_t. Unknown integers are RGMIN_INVALID_PARAMETER. */
    int32_t kind;
    uint32_t nev;
    uint32_t krylov;
    uint32_t max_iter;
    double tol;
} rgmin_eigen_params_t;

typedef struct rgmin_lowest_mode_t {
    double value;
    size_t actions;
} rgmin_lowest_mode_t;

typedef rgmin_status_t (*rgmin_hvp_fn)(void *user, const DLManagedTensorVersioned *x,
                                   const DLManagedTensorVersioned *v,
                                   DLManagedTensorVersioned *hv_out);

/**
 * Matrix-free lowest Hessian eigenpair. \a params NULL is Lanczos.
 * Unlinked kinds return \c RGMIN_UNAVAILABLE. The vector is written
 * to \a mode_out.
 */
rgmin_status_t rgmin_lowest_eigenpair(rgmin_hvp_fn hvp, void *user,
                                    const DLManagedTensorVersioned *x,
                                    const DLManagedTensorVersioned *seed,
                                    DLManagedTensorVersioned *mode_out,
                                    const rgmin_eigen_params_t *params,
                                    rgmin_lowest_mode_t *out);

/** Steihaug-Toint Newton-CG. \a hvp writes \c H(x) v. \c method is
 *  \c RGMIN_NEWTON_CG or \c RGMIN_NEWTON_PCG. No dense Hessian is formed. */
rgmin_status_t rgmin_minimize_hvp(rgmin_eval_fn eval, rgmin_grad_fn grad,
                              rgmin_hvp_fn hvp, void *user,
                              DLManagedTensorVersioned *x,
                              const rgmin_control_t *ctrl, rgmin_method_t method,
                              rgmin_report_t *out);
/** One Steihaug-Toint iteration. Legal on \c RGMIN_NEWTON_CG and
 *  \c RGMIN_NEWTON_PCG. \a hvp is a product, not a matrix. */
rgmin_status_t rgmin_solver_step_hvp(rgmin_solver_t *solver, rgmin_eval_fn eval,
                                 rgmin_grad_fn grad, rgmin_hvp_fn hvp, void *user,
                                 DLManagedTensorVersioned *x, rgmin_report_t *out);



/** Separate backend settings. ChASE reads both fields. Block LOBPCG
 *  ignores them. A null pointer selects the defaults. The base
 *  parameter record retains its layout. */
typedef struct rgmin_eigen_options_t {
    /** ChASE filter degree. Zero selects the backend default. */
    uint32_t degree;
    /** ChASE extra search dimensions. Zero selects the backend default. */
    uint32_t extra;
} rgmin_eigen_options_t;

/** One pair from \ref rgmin_lowest_eigenpairs. \c actions is the product
 *  count of the whole solve. ABI minor 39. */
typedef struct rgmin_eigenpair_t {
    double value;
    size_t actions;
} rgmin_eigenpair_t;

/**
 * k lowest eigenpairs of a symmetric product (block LOBPCG, Knyazev 2001).
 * \a seed and \a modes_out are row-major n-by-k: rank 2 with shape (n, k),
 * or rank 1 of length n*k. They may alias. \a values_out and \a pairs_out
 * have length k, values ascending. Each \c actions field is the product
 * count of the solve. \a params NULL selects tol 1e-8 and
 * max_iter = max(n, 8). \a options NULL is accepted; this matrix-free
 * solve ignores ChASE degree and extra. An unlinked \a params->kind
 * returns \c RGMIN_UNAVAILABLE. k = 0, k > n, and non-finite input return
 * \c RGMIN_INVALID_PARAMETER. ABI minor 39.
 */
rgmin_status_t rgmin_lowest_eigenpairs(
    rgmin_hvp_fn hvp, void *user, const DLManagedTensorVersioned *x,
    const DLManagedTensorVersioned *seed, DLManagedTensorVersioned *modes_out,
    DLManagedTensorVersioned *values_out, const rgmin_eigen_params_t *params,
    const rgmin_eigen_options_t *options, rgmin_eigenpair_t *pairs_out, size_t k);

/**
 * Lowest Hessian eigenpair with separate optional backend settings.
 * Matrix-free backends ignore the ChASE degree and extra dimensions.
 * A null options pointer selects the backend defaults. The callback and
 * tensor requirements match rgmin_lowest_eigenpair.
 */
rgmin_status_t rgmin_lowest_eigenpair_with_options(
    rgmin_hvp_fn hvp, void *user, const DLManagedTensorVersioned *x,
    const DLManagedTensorVersioned *seed, DLManagedTensorVersioned *mode_out,
    const rgmin_eigen_params_t *params, const rgmin_eigen_options_t *options,
    rgmin_lowest_mode_t *out);

/** Caller-owned Eckart buffers. One per image. */
typedef struct rgmin_rigid_workspace rgmin_rigid_workspace_t;
rgmin_rigid_workspace_t *rgmin_rigid_workspace_new(void);
void rgmin_rigid_workspace_free(rgmin_rigid_workspace_t *ws);
size_t rgmin_rigid_workspace_rank(const rgmin_rigid_workspace_t *ws);
/** P v in place. n_mass == 0 is unit mass. frozen == NULL is no freeze. */
rgmin_status_t rgmin_rigid_project(
    rgmin_rigid_workspace_t *ws, double *v, size_t n, const double *pos,
    const double *masses, size_t n_mass, int32_t mode,
    int32_t mass_weighted_coords, int32_t mass_metric, int32_t pbc_x, int32_t pbc_y,
    int32_t pbc_z, const uint8_t *frozen, int32_t broken_tx, int32_t broken_ty,
    int32_t broken_tz, int32_t broken_rx, int32_t broken_ry, int32_t broken_rz);
/** Row-major H of length n*n becomes P H P. */
rgmin_status_t rgmin_rigid_project_hessian(
    rgmin_rigid_workspace_t *ws, double *h, size_t n, const double *pos,
    const double *masses, size_t n_mass, int32_t mode,
    int32_t mass_weighted_coords, int32_t mass_metric, int32_t pbc_x, int32_t pbc_y,
    int32_t pbc_z, const uint8_t *frozen, int32_t broken_tx, int32_t broken_ty,
    int32_t broken_tz, int32_t broken_rx, int32_t broken_ry, int32_t broken_rz);
/** Horn / Coutsias. Any periodic axis skips the rotation. */
rgmin_status_t rgmin_rigid_align(double *mobile, const double *reference, size_t n,
                             const double *masses, size_t n_mass, int32_t pbc_x,
                             int32_t pbc_y, int32_t pbc_z, double *rmsd);

/** \defgroup manifold Riemannian manifold handles (ABI minor 41)
 *
 *  An \ref rgmin_manifold_handle_t is a Stiefel St(n, p), an SPD(p) or a
 *  product of them. \ref rgmin_manifold_t is the unrelated session token
 *  of \ref rgmin_solver_set_manifold.
 *
 *  Every buffer is caller-owned, column-major, and \c len doubles long,
 *  where \c len is \ref rgmin_manifold_len:
 *  - Stiefel: an n-by-p frame with X^T X = I, \c len = n*p.
 *  - SPD: a p-by-p symmetric positive definite matrix, \c len = p*p.
 *  - Product: the factors' blocks one after another, in factor order.
 *  Tangent vectors and Euclidean gradients use the packing of the point.
 *
 *  Each call returns \c RGMIN_SUCCESS, or \c RGMIN_INVALID_PARAMETER on a
 *  null pointer, a \c len other than the packed length, a non-finite
 *  entry, an SPD point that is not symmetric positive definite, or a
 *  rank-deficient Stiefel retraction; \ref rgmin_last_error names the
 *  cause. Inputs are copied first and outputs are written only on
 *  success, so an output may alias an input.
 *  @{ */

/** Opaque manifold. Create with \ref rgmin_manifold_stiefel,
 *  \ref rgmin_manifold_spd or \ref rgmin_manifold_product; release with
 *  \ref rgmin_manifold_free. */
typedef struct rgmin_manifold_handle rgmin_manifold_handle_t;

/** Stiefel retraction for \ref rgmin_manifold_stiefel. */
typedef enum rgmin_stiefel_retraction_t {
    /** Q factor of X + V with a positive diagonal R. First order. */
    RGMIN_STIEFEL_QR = 0,
    /** Polar factor (X + V) ((X + V)^T (X + V))^{-1/2}. Second order. */
    RGMIN_STIEFEL_POLAR = 1
} rgmin_stiefel_retraction_t;

/** SPD metric for \ref rgmin_manifold_spd. */
typedef enum rgmin_spd_metric_t {
    /** <U, V>_X = tr(X^{-1} U X^{-1} V). */
    RGMIN_SPD_AFFINE_INVARIANT = 0,
    /** <U, V>_X = <Dlog(X)[U], Dlog(X)[V]>_F. */
    RGMIN_SPD_LOG_EUCLIDEAN = 1
} rgmin_spd_metric_t;

/** St(n, p), 1 <= p <= n, under the embedded metric. \a retraction is an
 *  \ref rgmin_stiefel_retraction_t value. Null on a bad shape or an
 *  unknown retraction. */
rgmin_manifold_handle_t *rgmin_manifold_stiefel(size_t n, size_t p, int32_t retraction);
/** SPD(p), p >= 1. \a metric is an \ref rgmin_spd_metric_t value. The
 *  retraction is the exponential of the metric. Null on p = 0 or an
 *  unknown metric. */
rgmin_manifold_handle_t *rgmin_manifold_spd(size_t p, int32_t metric);
/** Product of \a count handles in order. The factors are copied; the
 *  caller still frees each one. Null on a null array or entry, or
 *  \a count = 0. */
rgmin_manifold_handle_t *rgmin_manifold_product(const rgmin_manifold_handle_t *const *factors,
                                                size_t count);
/** Release a handle. Null is a no-op. */
void rgmin_manifold_free(rgmin_manifold_handle_t *manifold);
/** Packed length of one point. A null handle returns 0. */
size_t rgmin_manifold_len(const rgmin_manifold_handle_t *manifold);

/** Tangent projection of the ambient \a v at \a x.
 *  Stiefel: V - X sym(X^T V). SPD: sym(V). */
rgmin_status_t rgmin_manifold_project(const rgmin_manifold_handle_t *manifold, const double *x,
                                      const double *v, double *out, size_t len);
/** Riemannian gradient at \a x from the Euclidean gradient \a egrad.
 *  Stiefel: the projection. SPD affine-invariant: X sym(G) X.
 *  SPD log-Euclidean: Dlog(X)^{-2}[sym(G)]. */
rgmin_status_t rgmin_manifold_egrad_to_rgrad(const rgmin_manifold_handle_t *manifold,
                                             const double *x, const double *egrad, double *out,
                                             size_t len);
/** Retraction of the step \a v at \a x. Stiefel: the handle's QR or polar
 *  retraction. SPD: the exponential of the metric. */
rgmin_status_t rgmin_manifold_retract(const rgmin_manifold_handle_t *manifold, const double *x,
                                      const double *v, double *out, size_t len);
/** Riemannian exponential at \a x; \a v is projected to the tangent space
 *  first. Stiefel: the embedded-metric geodesic (Edelman, Arias, Smith
 *  1998). SPD: the geodesic of the metric. */
rgmin_status_t rgmin_manifold_exp(const rgmin_manifold_handle_t *manifold, const double *x,
                                  const double *v, double *out, size_t len);
/** Riemannian logarithm of \a y at \a x, the inverse of
 *  \ref rgmin_manifold_exp. A handle with a Stiefel factor returns
 *  \c RGMIN_INVALID_PARAMETER. */
rgmin_status_t rgmin_manifold_log(const rgmin_manifold_handle_t *manifold, const double *x,
                                  const double *y, double *out, size_t len);
/** Vector transport of \a v from \a x_from to \a x_to. Stiefel: projection
 *  at \a x_to. SPD: parallel transport of the metric. */
rgmin_status_t rgmin_manifold_transport(const rgmin_manifold_handle_t *manifold,
                                        const double *x_from, const double *x_to,
                                        const double *v, double *out, size_t len);
/** Riemannian inner product of the tangents \a u and \a v at \a x, written
 *  to \a *out. The product metric is the sum of the factor metrics. */
rgmin_status_t rgmin_manifold_inner(const rgmin_manifold_handle_t *manifold, const double *x,
                                    const double *u, const double *v, double *out, size_t len);
/** One Riemannian gradient step. \a rgrad_out is the Riemannian gradient
 *  of \a egrad at \a x; \a x_out is the retraction of -t * rgrad_out at
 *  \a x. \a t must be finite. \a rgrad_out and \a x_out must not alias
 *  each other. */
rgmin_status_t rgmin_manifold_gradient_step(const rgmin_manifold_handle_t *manifold,
                                            const double *x, const double *egrad, double t,
                                            double *rgrad_out, double *x_out, size_t len);
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* RGMIN_H */
