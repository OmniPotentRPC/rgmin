#ifndef RGMIN_H
#define RGMIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <dlpack/dlpack.h>

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
 *  \brief C ABI for the Rust rgminci-optimize hourglass.
 *
 *  Solvers live in Rust. This header is the only C entry:
 *  \ref rgmin_minimize over dlpk \c DLManagedTensorVersioned tensors.
 */

/** Status of an ABI call. */
typedef enum rgmin_status_t {
    RGMIN_SUCCESS = 0,
    RGMIN_INVALID_PARAMETER = 1,
    RGMIN_INTERNAL_ERROR = 2,
    /** Tensor device is not CPU. The ABI stays stable for a later CUDA path. */
    RGMIN_UNSUPPORTED_DEVICE = 3,
    RGMIN_UNAVAILABLE = 4
} rgmin_status_t;

/** Compatibility identity for the rgminci-optimize C ABI. */
typedef struct rgmin_abi_stamp_t {
    uint16_t abi_major;
    uint16_t abi_minor;
    uint16_t layout_revision;
} rgmin_abi_stamp_t;

#define RGMIN_ABI_VERSION_MAJOR 1
#define RGMIN_ABI_VERSION_MINOR 27
#define RGMIN_ABI_LAYOUT_REVISION 2

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
    RGMIN_FIRE2 = 20
} rgmin_method_t;

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
    RGMIN_CONJUGACY_FR_PR = 7
} rgmin_conjugacy_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(rgmin_conjugacy_t) == sizeof(int32_t),
               "rgmin_conjugacy_t is i32-wide; do not build this header with -fshort-enums");
#endif

/** Outer-loop controls. \c memory is the L-BFGS pair cap. */
typedef struct rgmin_control_t {
    size_t maxiter;
    double gtol;
    double istep;
    size_t memory;
    double maxmove;
} rgmin_control_t;

/** Result of \ref rgmin_minimize. */
typedef struct rgmin_report_t {
    double value;
    size_t steps;
    double grad_norm;
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
/** Thread-local last error message after a non-success status. */
const char *rgmin_last_error(void);
/** Borrow a CPU f64 buffer as a dlpk tensor. */
DLManagedTensorVersioned *rgmin_tensor_borrow_cpu_f64(double *data, size_t n);
/** Free a tensor allocated by this ABI. */
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

/** Allocate a session. \a dim is the length of \c x. Null on error. */
rgmin_solver_t *rgmin_solver_create(rgmin_method_t method, const rgmin_control_t *ctrl,
                                size_t dim);
/** Release a session from \ref rgmin_solver_create. */
void rgmin_solver_free(rgmin_solver_t *solver);
/** Drop method memory. The next step is a cold start from the current \c x. */
void rgmin_solver_forget(rgmin_solver_t *solver);
/** Retain method memory and reset the cached point and acceptance window. */
void rgmin_solver_rebase(rgmin_solver_t *solver);
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
/** How a session takes a proposed step (eOn lbfgs_accept). */
typedef enum rgmin_accept_t {
    RGMIN_ACCEPT_NONE = 0,
    RGMIN_ACCEPT_ENERGY = 1,
    RGMIN_ACCEPT_NONMONOTONE = 2,
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
/** Line search for the session arms that use one. */
typedef enum rgmin_linesearch_t {
    RGMIN_LINESEARCH_BRENT = 0,
    RGMIN_LINESEARCH_BACKTRACKING = 1,
    RGMIN_LINESEARCH_GOLDSTEIN = 2,
    RGMIN_LINESEARCH_WOLFE = 3
} rgmin_linesearch_t;
/** Reject invalid constants or a zero trial limit without changing the session. */
rgmin_status_t rgmin_solver_set_linesearch(rgmin_solver_t *solver,
                                         rgmin_linesearch_t kind,
                                         double c1, double c2, size_t maxiter);
void rgmin_solver_set_atom_maxmove(rgmin_solver_t *solver, double maxmove);
void rgmin_solver_set_project_rigid(rgmin_solver_t *solver, int32_t enabled);
void rgmin_solver_set_extra_updates(rgmin_solver_t *solver, size_t extra);
void rgmin_solver_set_cautious(rgmin_solver_t *solver, double eps, double alpha);
/** HiGHS feasible-set step. Nonzero enables it. Returns 0, or 1 if this
 *  build has no highs feature. */
int32_t rgmin_solver_set_highs(rgmin_solver_t *solver, int32_t enabled);
/** Coordinate bounds for enabled HiGHS constraints. Null sides are unbounded.
 *  Each non-null side has n values, and n must match the session dimension.
 *  Returns zero on a supported build; malformed domains fail before evaluation. */
int32_t rgmin_solver_set_box(rgmin_solver_t *solver, const double *lower,
                           const double *upper, size_t n);
/** Model-step trust radius. Non-positive disables the radius. */
int32_t rgmin_solver_set_trust(rgmin_solver_t *solver, double radius);
/** Alias with the same radius and feature contract. */
int32_t rgmin_solver_set_highs_trust(rgmin_solver_t *solver, double radius);
/** Append a sparse equality a.p = rhs on the model and accepted displacement.
 *  A violating accepted displacement returns RGMIN_INVALID_PARAMETER, restores the start,
 *  and clears method memory. Residual tolerance is 1e-7 in row units plus
 *  floating-point summation error. */
int32_t rgmin_solver_add_equality(rgmin_solver_t *solver, const size_t *idx,
                                const double *coef, size_t nnz, double rhs);
/** Remove every model-step equality. */
int32_t rgmin_solver_clear_equalities(rgmin_solver_t *solver);

/** Explicit HiGHS engine; CHOOSE retains the library default. */
typedef enum rgmin_highs_solver_t {
    RGMIN_HIGHS_CHOOSE = 0, RGMIN_HIGHS_SIMPLEX = 1,
    RGMIN_HIGHS_IPM = 2, RGMIN_HIGHS_IPX = 3, RGMIN_HIGHS_HIPO = 4,
    RGMIN_HIGHS_PDLP = 5, RGMIN_HIGHS_HIPDLP = 6, RGMIN_HIGHS_QPASM = 7
} rgmin_highs_solver_t;
/** Explicit crossover policy; CHOOSE retains the library default. */
typedef enum rgmin_highs_crossover_t {
    RGMIN_HIGHS_CROSSOVER_CHOOSE = 0, RGMIN_HIGHS_CROSSOVER_ON = 1,
    RGMIN_HIGHS_CROSSOVER_OFF = 2
} rgmin_highs_crossover_t;
/** HiGHS callback events. */
typedef enum rgmin_highs_cb_kind_t {
    RGMIN_HIGHS_CB_LOGGING = 0, RGMIN_HIGHS_CB_SIMPLEX_INTERRUPT = 1,
    RGMIN_HIGHS_CB_IPM_INTERRUPT = 2
} rgmin_highs_cb_kind_t;
/** Setting interrupt nonzero requests termination of the solve. */
typedef void (*rgmin_highs_callback_t)(int32_t kind, const char *message,
                                     int32_t *interrupt, void *user);
/** Return 0 on success, 1 for an unknown token, null handle, or absent HiGHS. */
int32_t rgmin_solver_set_highs_solver(rgmin_solver_t *solver, int32_t kind);
int32_t rgmin_solver_set_highs_crossover(rgmin_solver_t *solver, int32_t kind);
/** NULL callback clears it. The host retains context ownership through clear/free. */
int32_t rgmin_solver_set_highs_callback(rgmin_solver_t *solver,
                                      rgmin_highs_callback_t callback, void *user);

/** Embedded manifold. Euclidean is the default.
 *  Molecular clusters use RIGID_QUOTIENT (Sella Cartesian T+R,
 *  R^{3N}/SE(3)) or MW_RIGID (Page-McIver / Sella IRC Eckart).
 *  SO3 is length 9; SE3 is length 12. */
typedef enum rgmin_manifold_t {
    RGMIN_MANIFOLD_EUCLIDEAN = 0,
    RGMIN_MANIFOLD_SPHERE = 1,
    RGMIN_MANIFOLD_SO3 = 2,
    RGMIN_MANIFOLD_STIEFEL = 3,
    RGMIN_MANIFOLD_SE3 = 4,
    RGMIN_MANIFOLD_RIGID_QUOTIENT = 5,
    RGMIN_MANIFOLD_MW_RIGID = 6,
    RGMIN_MANIFOLD_SPD = 7,
    RGMIN_MANIFOLD_GRASSMANN = 8,
    RGMIN_MANIFOLD_HYPERBOLIC = 9,
    RGMIN_MANIFOLD_POINCARE = 10,
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
    RGMIN_MANIFOLD_POSITIVE = 21,
    RGMIN_MANIFOLD_CENTERED_MATRIX = 22,
    RGMIN_MANIFOLD_UNITARY = 23
} rgmin_manifold_t;
void rgmin_solver_set_manifold(rgmin_solver_t *solver, rgmin_manifold_t manifold);
/** Geometry dimensions are checked by the next solver step. */
void rgmin_solver_set_stiefel(rgmin_solver_t *solver, size_t n, size_t p);
void rgmin_solver_set_oblique(rgmin_solver_t *solver, size_t n, size_t m);
void rgmin_solver_set_complex_circle(rgmin_solver_t *solver, size_t n);
void rgmin_solver_set_euclidean_complex(rgmin_solver_t *solver, size_t n);
void rgmin_solver_set_constant(rgmin_solver_t *solver, size_t n);
void rgmin_solver_set_multinomial_ds(rgmin_solver_t *solver, size_t n);
void rgmin_solver_set_multinomial_sym(rgmin_solver_t *solver, size_t n);
void rgmin_solver_set_sphere_complex(rgmin_solver_t *solver, size_t n);
void rgmin_solver_set_positive(rgmin_solver_t *solver, size_t n);
void rgmin_solver_set_centered_matrix(rgmin_solver_t *solver, size_t m, size_t n, int32_t center_rows);
void rgmin_solver_set_factor_shape(rgmin_solver_t *solver, size_t n, size_t p);
void rgmin_solver_set_unitary(rgmin_solver_t *solver, size_t n);
/** Per-atom masses for MW_RIGID. n_atoms == 0 or masses == NULL
 *  restores unit mass. */
void rgmin_solver_set_masses(rgmin_solver_t *solver, const double *masses,
                           size_t n_atoms);
/** Periodic cell. Nonzero drops rotation (Sella proj_rot): R^{3N}/T(3). */
void rgmin_solver_set_periodic(rgmin_solver_t *solver, int32_t enabled);
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
/** Number of accepted L-BFGS pairs. Null and other methods return zero. */
size_t rgmin_solver_pair_count(const rgmin_solver_t *solver);
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
     *  0..7 are \c RGMIN_INVALID_PARAMETER. Ignored when \a params is
     *  NULL (Netlab Polak-Ribiere). */
    int32_t conjugacy;
} rgmin_scg_params_t;

/**
 * Møller SCG. \a curv may be NULL (finite-difference probe).
 * The exported symbol is \c rgmin_minimize_scg.
 *
 * \a params NULL selects ScgParams defaults, Polak-Ribiere, and
 * Restart::Never. A filled \a params takes \c conjugacy literally
 * (0 is Fletcher-Reeves). Restart is not a C token; the waist
 * passes Never. Møller n-success reset stays inside the Rust loop.
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

/** Separate backend settings; the base parameter record retains its layout. */
typedef struct rgmin_eigen_options_t {
    /** ChASE filter degree. Zero selects the backend default. */
    uint32_t degree;
    /** ChASE extra search dimensions. Zero selects the backend default. */
    uint32_t extra;
} rgmin_eigen_options_t;

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
 * to \a mode_out, which may share storage with either input.
 */
rgmin_status_t rgmin_lowest_eigenpair(rgmin_hvp_fn hvp, void *user,
                                    const DLManagedTensorVersioned *x,
                                    const DLManagedTensorVersioned *seed,
                                    DLManagedTensorVersioned *mode_out,
                                    const rgmin_eigen_params_t *params,
                                    rgmin_lowest_mode_t *out);


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


#ifdef __cplusplus
}
#endif

#endif /* RGMIN_H */
