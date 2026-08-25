#ifndef XTS_OPTIMIZE_H
#define XTS_OPTIMIZE_H

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

/** \file xts.h
 *  \brief C ABI for the Rust xtsci-optimize hourglass.
 *
 *  Solvers live in Rust. This header is the only C entry:
 *  \ref xts_minimize over dlpk \c DLManagedTensorVersioned tensors.
 */

/** Status of an ABI call. */
typedef enum xts_status_t {
    XTS_SUCCESS = 0,
    XTS_INVALID_PARAMETER = 1,
    XTS_INTERNAL_ERROR = 2,
    /** Tensor device is not CPU. The ABI stays stable for a later CUDA path. */
    XTS_UNSUPPORTED_DEVICE = 3,
    /** Named eigensolver is not linked in this build. */
    XTS_UNAVAILABLE = 4
} xts_status_t;

/** Compatibility identity for the xtsci-optimize C ABI. */
typedef struct xts_abi_stamp_t {
    uint16_t abi_major;
    uint16_t abi_minor;
    uint16_t layout_revision;
} xts_abi_stamp_t;

#define XTS_ABI_VERSION_MAJOR 1
#define XTS_ABI_VERSION_MINOR 19
#define XTS_ABI_LAYOUT_REVISION 4

/** Solver selector. \c XTS_LBFGS is the production unconstrained method. */
typedef enum xts_method_t {
    XTS_POLAK_RIBIERE = 0,
    XTS_FLETCHER_REEVES = 1,
    XTS_BFGS = 2,
    XTS_LBFGS = 3,
    XTS_SR1 = 4,
    XTS_ADAM = 5,
    XTS_STEEPEST = 6,
    XTS_SR2 = 7,
    XTS_PSO = 8,
    XTS_HESTENES_STIEFEL = 9,
    XTS_DAI_YUAN = 10,
    XTS_CONJUGATE_DESCENT = 11,
    XTS_HAGER_ZHANG = 12,
    XTS_LIU_STOREY = 13,
    XTS_FR_PR = 14,
    XTS_NEWTON = 15,
    XTS_RFO = 16,
    XTS_FIRE = 17,
    XTS_BB = 18,
    XTS_DOGLEG = 19,
    XTS_FIRE2 = 20
} xts_method_t;

/** Conjugacy coefficient β. Closed leaf subset of dest Conjugacy
 *  (src/nlcg). Integers are dest declaration order. Hybrid stays
 *  Rust-only. This is not xts_method_t (that enum is the solver axis). */
typedef enum xts_conjugacy_t {
    XTS_CONJUGACY_FLETCHER_REEVES = 0,
    XTS_CONJUGACY_POLAK_RIBIERE = 1,
    XTS_CONJUGACY_HESTENES_STIEFEL = 2,
    XTS_CONJUGACY_DAI_YUAN = 3,
    XTS_CONJUGACY_CONJUGATE_DESCENT = 4,
    XTS_CONJUGACY_HAGER_ZHANG = 5,
    XTS_CONJUGACY_LIU_STOREY = 6,
    XTS_CONJUGACY_FR_PR = 7
} xts_conjugacy_t;
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(xts_conjugacy_t) == sizeof(int32_t),
               "xts_conjugacy_t is i32-wide; do not build this header with -fshort-enums");
#endif

/** Outer-loop controls. \c memory is the L-BFGS pair cap. */
typedef struct xts_control_t {
    size_t maxiter;
    double gtol;
    double istep;
    size_t memory;
    double maxmove;
} xts_control_t;

/** Result of \ref xts_minimize. */
typedef struct xts_report_t {
    double value;
    size_t steps;
    double grad_norm;
} xts_report_t;

typedef xts_status_t (*xts_eval_fn)(void *user, const DLManagedTensorVersioned *x,
                                    double *value_out);
typedef xts_status_t (*xts_grad_fn)(void *user, const DLManagedTensorVersioned *x,
                                    DLManagedTensorVersioned *grad_out);
typedef xts_status_t (*xts_hess_fn)(void *user, const DLManagedTensorVersioned *x,
                                    DLManagedTensorVersioned *hess_out);
/** Fused (f, grad). One geometry, one host potential call. */
typedef xts_status_t (*xts_evalgrad_fn)(void *user,
                                        const DLManagedTensorVersioned *x,
                                        double *value_out,
                                        DLManagedTensorVersioned *grad_out);

/** Crate version string. */
const char *xts_version(void);
/** Return the C ABI compatibility identity for this build. */
xts_abi_stamp_t xts_abi_stamp(void);
/** Return nonzero when a stamp is compatible with this build. */
int32_t xts_abi_compatible(const xts_abi_stamp_t *stamp);
/** Thread-local last error. Rust symbol is rgmin_last_error. */
const char *rgmin_last_error(void);
/** Borrow a CPU f64 buffer as a dlpk tensor. Rust: rgmin_tensor_borrow_cpu_f64. */
DLManagedTensorVersioned *rgmin_tensor_borrow_cpu_f64(double *data, size_t n);
/** Free a tensor from rgmin_tensor_borrow_cpu_f64. */
void rgmin_tensor_free(DLManagedTensorVersioned *tensor);
#define xts_last_error rgmin_last_error
#define xts_tensor_borrow_cpu_f64 rgmin_tensor_borrow_cpu_f64
#define xts_tensor_free rgmin_tensor_free
/**
 * Minimize from \a x in place.
 *
 * \param eval  Value callback.
 * \param grad  Gradient callback.
 * \param user  Passed through to both callbacks.
 * \param x     CPU f64 state (updated on success).
 * \param ctrl  Iteration / L-BFGS memory controls.
 * \param method Solver. \c XTS_LBFGS is the production choice.
 * \param out   Filled on success.
 */
xts_status_t rgmin_minimize(xts_eval_fn eval, xts_grad_fn grad, void *user,
                            DLManagedTensorVersioned *x, const xts_control_t *ctrl,
                            xts_method_t method, xts_report_t *out);
#ifndef xts_minimize
#define xts_minimize rgmin_minimize
#endif
/**
 * Newton / RFO. \a hess writes a length-\c n*n row-major Hessian.
 * \c method is \c XTS_NEWTON or \c XTS_RFO.
 */
xts_status_t xts_minimize_hess(xts_eval_fn eval, xts_grad_fn grad,
                               xts_hess_fn hess, void *user,
                               DLManagedTensorVersioned *x,
                               const xts_control_t *ctrl, xts_method_t method,
                               xts_report_t *out);

/**
 * Minimize an eindir-compatible objective without taking ownership of it.
 * The stamp must be compatible with this build and include an analytic
 * gradient. The caller retains ownership of an objective.
 */
xts_status_t xts_minimize_eindir(
    const eindir_objective_t *objective, const eindir_abi_stamp_t *stamp,
    DLManagedTensorVersioned *x, const xts_control_t *ctrl, xts_method_t method,
    xts_report_t *out);

/**
 * Opaque session. Algorithm memory (L-BFGS pairs, NLCG directions,
 * dense H, Adam moments, PSO swarm) lives here. \c x stays a dlpk
 * tensor. Callbacks are arguments of each step, not stored.
 */
typedef struct xts_solver_t xts_solver_t;

#ifndef xts_solver_create
#define xts_solver_create rgmin_solver_create
#define xts_solver_free rgmin_solver_free
#define xts_solver_forget rgmin_solver_forget
#define xts_solver_set_maxmove rgmin_solver_set_maxmove
#define xts_solver_set_qn_step rgmin_solver_set_qn_step
#define xts_solver_set_accept rgmin_solver_set_accept
#define xts_solver_set_atom_maxmove rgmin_solver_set_atom_maxmove
#define xts_solver_set_project_rigid rgmin_solver_set_project_rigid
#define xts_solver_set_extra_updates rgmin_solver_set_extra_updates
#define xts_solver_set_cautious rgmin_solver_set_cautious
#define xts_solver_set_highs rgmin_solver_set_highs
#define xts_solver_set_manifold rgmin_solver_set_manifold
#define xts_solver_set_oblique rgmin_solver_set_oblique
#define xts_solver_set_stiefel rgmin_solver_set_stiefel
#define xts_solver_set_complex_circle rgmin_solver_set_complex_circle
#define xts_solver_set_euclidean_complex rgmin_solver_set_euclidean_complex
#define xts_solver_set_constant rgmin_solver_set_constant
#define xts_solver_set_multinomial_ds rgmin_solver_set_multinomial_ds
#define xts_solver_set_multinomial_sym rgmin_solver_set_multinomial_sym
#define xts_solver_set_sphere_complex rgmin_solver_set_sphere_complex
#define xts_solver_set_positive rgmin_solver_set_positive
#define xts_solver_set_masses rgmin_solver_set_masses
#define xts_solver_set_periodic rgmin_solver_set_periodic
#define xts_solver_step rgmin_solver_step
#define xts_solver_step_hess rgmin_solver_step_hess
#define xts_solver_step_fg rgmin_solver_step_fg
#define xts_solver_step_hess_fg rgmin_solver_step_hess_fg
#endif
/** Allocate a session. \a dim is the length of \c x. Null on error.
 *  The exported symbol is \c rgmin_solver_create. */
xts_solver_t *xts_solver_create(xts_method_t method, const xts_control_t *ctrl,
                                size_t dim);
/** Release a session from \ref xts_solver_create. */
void xts_solver_free(xts_solver_t *solver);
/** Drop method memory. The next step is a cold start from the current \c x. */
void xts_solver_forget(xts_solver_t *solver);
/** Euclidean step cap for the next \ref xts_solver_step (saddle \c max_move). */
void xts_solver_set_maxmove(xts_solver_t *solver, double maxmove);
/** How an L-BFGS session uses a caller Hessian (eOn \c lbfgs_step). */
typedef enum xts_qn_step_t {
    XTS_QN_LBFGS = 0,
    XTS_QN_NEWTON = 1,
    XTS_QN_RFO = 2
} xts_qn_step_t;
/** Two-loop + H0, or Newton/RFO on P. Legal with \ref xts_solver_step_hess. */
void xts_solver_set_qn_step(xts_solver_t *solver, xts_qn_step_t step);
/** How a session takes a proposed step (eOn lbfgs_accept). */
typedef enum xts_accept_t {
    XTS_ACCEPT_NONE = 0,
    XTS_ACCEPT_ENERGY = 1,
    XTS_ACCEPT_NONMONOTONE = 2
} xts_accept_t;
void xts_solver_set_accept(xts_solver_t *solver, xts_accept_t accept);
void xts_solver_set_atom_maxmove(xts_solver_t *solver, double maxmove);
void xts_solver_set_project_rigid(xts_solver_t *solver, int32_t enabled);
void xts_solver_set_extra_updates(xts_solver_t *solver, size_t extra);
void xts_solver_set_cautious(xts_solver_t *solver, double eps, double alpha);
/** HiGHS feasible-set step. Nonzero enables it. Returns 0, or 1 if this
 *  build has no highs feature. */
int32_t xts_solver_set_highs(xts_solver_t *solver, int32_t enabled);
/** Embedded manifold. Euclidean is the default.
 *  Molecular clusters use RIGID_QUOTIENT (Sella Cartesian T+R,
 *  R^{3N}/SE(3)) or MW_RIGID (Page-McIver / Sella IRC Eckart).
 *  SO3 is length 9; SE3 is length 12. Oblique is n-by-m via
 *  xts_solver_set_oblique. Stiefel p>1 is xts_solver_set_stiefel.
 *  Reserved: 7 SPD, 8 Grassmann, 9 Hyperbolic, 10 Poincare.
 *  Token 15 is skew-symmetric n-by-n, n >= 2.
 *  Token 16 is complex Euclidean C^n, packed interleaved, length 2n.
 *  Token 17 is the singleton {A} of packed length n (constantfactory).
 *  Token 18 is doubly-stochastic n-by-n (multinomialdoublystochasticfactory).
 *  Token 19 is symmetric doubly-stochastic n-by-n
 *  (multinomialsymmetricfactory).
 *  Token 20 is the complex unit sphere C^n, packed interleaved, length 2n.
 *  Token 21 is the positive orthant of packed length n (positivefactory).
 *  Reserved 7-10 unused. */
typedef enum xts_manifold_t {
    XTS_MANIFOLD_EUCLIDEAN = 0,
    XTS_MANIFOLD_SPHERE = 1,
    XTS_MANIFOLD_SO3 = 2,
    XTS_MANIFOLD_STIEFEL = 3,
    XTS_MANIFOLD_SE3 = 4,
    XTS_MANIFOLD_RIGID_QUOTIENT = 5,
    XTS_MANIFOLD_MW_RIGID = 6,
    XTS_MANIFOLD_OBLIQUE = 11,
    XTS_MANIFOLD_MULTINOMIAL = 12,
    XTS_MANIFOLD_COMPLEX_CIRCLE = 13,
    XTS_MANIFOLD_SYMMETRIC = 14,
    XTS_MANIFOLD_SKEWSYMMETRIC = 15,
    XTS_MANIFOLD_EUCLIDEAN_COMPLEX = 16,
    XTS_MANIFOLD_CONSTANT = 17,
    XTS_MANIFOLD_MULTINOMIAL_DS = 18,
    XTS_MANIFOLD_MULTINOMIAL_SYM = 19,
    XTS_MANIFOLD_SPHERE_COMPLEX = 20,
    XTS_MANIFOLD_POSITIVE = 21
} xts_manifold_t;
void xts_solver_set_manifold(xts_solver_t *solver, xts_manifold_t manifold);
/** Oblique OB(n,m): product of m unit spheres in R^n, column-major. */
void xts_solver_set_oblique(xts_solver_t *solver, size_t n, size_t m);
/** Stiefel St(n,p). p = 1 is the sphere packing. p > 1 is n*p. */
void xts_solver_set_stiefel(xts_solver_t *solver, size_t n, size_t p);
/** n unit-modulus complex numbers. Packed interleaved, length 2n. */
void xts_solver_set_complex_circle(xts_solver_t *solver, size_t n);
/** Complex Euclidean C^n. Packed interleaved, length 2n. */
void xts_solver_set_euclidean_complex(xts_solver_t *solver, size_t n);
/** Singleton of packed length n. manopt constantfactory. */
void xts_solver_set_constant(xts_solver_t *solver, size_t n);
/** Doubly-stochastic n-by-n, packed n*n. manopt
 *  multinomialdoublystochasticfactory. Token 18 defaults to n = 2. */
void xts_solver_set_multinomial_ds(xts_solver_t *solver, size_t n);
/** Symmetric doubly-stochastic n-by-n, packed n*n. manopt
 *  multinomialsymmetricfactory. Token 19 defaults to n = 2. */
void xts_solver_set_multinomial_sym(xts_solver_t *solver, size_t n);
/** Complex unit sphere in C^n. Packed interleaved, length 2n.
 *  manopt spherecomplexfactory. Token 20 defaults to n = 1. */
void xts_solver_set_sphere_complex(xts_solver_t *solver, size_t n);
/** Positive orthant of packed length n. manopt positivefactory.
 *  Token 21 defaults to n = 1. Reserved 7-10 unused. */
void xts_solver_set_positive(xts_solver_t *solver, size_t n);
/** Per-atom masses for MW_RIGID. n_atoms == 0 or masses == NULL
 *  restores unit mass. */
void xts_solver_set_masses(xts_solver_t *solver, const double *masses,
                           size_t n_atoms);
/** Periodic cell. Nonzero drops rotation (Sella proj_rot): R^{3N}/T(3). */
void xts_solver_set_periodic(xts_solver_t *solver, int32_t enabled);
/**
 * One outer iteration: direction, line search, curvature update.
 * \a eval and \a grad are valid for this call only. \a x is in/out.
 */
xts_status_t xts_solver_step(xts_solver_t *solver, xts_eval_fn eval,
                             xts_grad_fn grad, void *user,
                             DLManagedTensorVersioned *x, xts_report_t *out);
/** One Newton / RFO iteration. \a hess writes a length-\c n*n Hessian. */
xts_status_t xts_solver_step_hess(xts_solver_t *solver, xts_eval_fn eval,
                                  xts_grad_fn grad, xts_hess_fn hess,
                                  void *user, DLManagedTensorVersioned *x,
                                  xts_report_t *out);
/** Like \ref xts_solver_step with one fused (f, g) callback. */
xts_status_t xts_solver_step_fg(xts_solver_t *solver, xts_evalgrad_fn evalgrad,
                                void *user, DLManagedTensorVersioned *x,
                                xts_report_t *out);
/** Like \ref xts_solver_step_hess with one fused (f, g) callback. */
xts_status_t xts_solver_step_hess_fg(xts_solver_t *solver,
                                     xts_evalgrad_fn evalgrad, xts_hess_fn hess,
                                     void *user, DLManagedTensorVersioned *x,
                                     xts_report_t *out);

/** Directional curvature `d^T H(x) d`. Non-success falls back to the probe. */
typedef xts_status_t (*xts_curv_fn)(void *user, const DLManagedTensorVersioned *x,
                                    const DLManagedTensorVersioned *d,
                                    double *curv_out);

typedef struct xts_scg_params_t {
    double sigma0;
    double lambda;
    double lambda_limit;
    double tol_sol;
    double tol_func;
    /** Leaf conjugacy. Literal: 0 is Fletcher-Reeves. Values outside
     *  0..7 are \c XTS_INVALID_PARAMETER. Ignored when \a params is
     *  NULL (Netlab Polak-Ribiere). */
    xts_conjugacy_t conjugacy;
} xts_scg_params_t;

/**
 * Møller SCG. \a curv may be NULL (finite-difference probe).
 * The exported symbol is \c rgmin_minimize_scg.
 *
 * \a params NULL selects ScgParams defaults, Polak-Ribiere, and
 * Restart::Never. A filled \a params takes \c conjugacy literally
 * (0 is Fletcher-Reeves). Restart is not a C token; the waist
 * passes Never. Møller n-success reset stays inside the Rust loop.
 * gpr_optim RgminScg.inl writes \c XTS_CONJUGACY_LIU_STOREY. In-tree
 * SCG.inl does not call this entry.
 */
xts_status_t rgmin_minimize_scg(xts_eval_fn eval, xts_grad_fn grad, xts_curv_fn curv,
                                void *user, DLManagedTensorVersioned *x,
                                const xts_control_t *ctrl,
                                const xts_scg_params_t *params, xts_report_t *out);

#ifndef xts_minimize_scg
#define xts_minimize_scg rgmin_minimize_scg
#endif
#ifndef rgmin_curv_fn
#define rgmin_curv_fn xts_curv_fn
#define rgmin_scg_params_t xts_scg_params_t
#define rgmin_status_t xts_status_t
#define rgmin_eval_fn xts_eval_fn
#define rgmin_grad_fn xts_grad_fn
#define rgmin_control_t xts_control_t
#define rgmin_report_t xts_report_t
#define RGMIN_SUCCESS XTS_SUCCESS
#define RGMIN_INVALID_PARAMETER XTS_INVALID_PARAMETER
#endif
#ifndef rgmin_conjugacy_t
#define rgmin_conjugacy_t xts_conjugacy_t
#define RGMIN_CONJUGACY_FLETCHER_REEVES XTS_CONJUGACY_FLETCHER_REEVES
#define RGMIN_CONJUGACY_POLAK_RIBIERE XTS_CONJUGACY_POLAK_RIBIERE
#define RGMIN_CONJUGACY_HESTENES_STIEFEL XTS_CONJUGACY_HESTENES_STIEFEL
#define RGMIN_CONJUGACY_DAI_YUAN XTS_CONJUGACY_DAI_YUAN
#define RGMIN_CONJUGACY_CONJUGATE_DESCENT XTS_CONJUGACY_CONJUGATE_DESCENT
#define RGMIN_CONJUGACY_HAGER_ZHANG XTS_CONJUGACY_HAGER_ZHANG
#define RGMIN_CONJUGACY_LIU_STOREY XTS_CONJUGACY_LIU_STOREY
#define RGMIN_CONJUGACY_FR_PR XTS_CONJUGACY_FR_PR
#endif

/** Closed eigensolver tag. Integers match schema/eigen.capnp. */
typedef enum xts_eigen_kind_t {
    XTS_EIGEN_LANCZOS = 0,
    XTS_EIGEN_RAYLEIGH_RITZ = 1,
    XTS_EIGEN_JACOBI_DAVIDSON = 2,
    XTS_EIGEN_LOBPCG = 3,
    XTS_EIGEN_PRIMME = 4,
    XTS_EIGEN_SLEPC = 5,
    XTS_EIGEN_CHASE = 6,
    XTS_EIGEN_ELPA = 7,
    XTS_EIGEN_ELPA2 = 8,
    XTS_EIGEN_SLATE = 9,
    XTS_EIGEN_MAGMA = 10,
    XTS_EIGEN_CUSOLVER = 11,
    XTS_EIGEN_DLA_FUTURE = 12,
    XTS_EIGEN_EIGENEXA = 13
} xts_eigen_kind_t;

typedef struct xts_eigen_params_t {
    /** Literal xts_eigen_kind_t. Unknown integers are XTS_INVALID_PARAMETER. */
    int32_t kind;
    uint32_t nev;
    uint32_t krylov;
    uint32_t max_iter;
    double tol;
} xts_eigen_params_t;

typedef struct xts_lowest_mode_t {
    double value;
    size_t actions;
} xts_lowest_mode_t;

typedef xts_status_t (*xts_hvp_fn)(void *user, const DLManagedTensorVersioned *x,
                                   const DLManagedTensorVersioned *v,
                                   DLManagedTensorVersioned *hv_out);

/**
 * Matrix-free lowest Hessian eigenpair. \a params NULL is Lanczos.
 * Unlinked kinds return \c XTS_UNAVAILABLE. The vector is written
 * to \a mode_out.
 */
xts_status_t rgmin_lowest_eigenpair(xts_hvp_fn hvp, void *user,
                                    const DLManagedTensorVersioned *x,
                                    const DLManagedTensorVersioned *seed,
                                    DLManagedTensorVersioned *mode_out,
                                    const xts_eigen_params_t *params,
                                    xts_lowest_mode_t *out);

#ifndef rgmin_eigen_kind_t
#define rgmin_eigen_kind_t xts_eigen_kind_t
#define rgmin_eigen_params_t xts_eigen_params_t
#define rgmin_lowest_mode_t xts_lowest_mode_t
#define rgmin_hvp_fn xts_hvp_fn
#define RGMIN_EIGEN_LANCZOS XTS_EIGEN_LANCZOS
#define RGMIN_EIGEN_RAYLEIGH_RITZ XTS_EIGEN_RAYLEIGH_RITZ
#define RGMIN_EIGEN_JACOBI_DAVIDSON XTS_EIGEN_JACOBI_DAVIDSON
#define RGMIN_EIGEN_LOBPCG XTS_EIGEN_LOBPCG
#define RGMIN_EIGEN_PRIMME XTS_EIGEN_PRIMME
#define RGMIN_EIGEN_SLEPC XTS_EIGEN_SLEPC
#define RGMIN_EIGEN_CHASE XTS_EIGEN_CHASE
#define RGMIN_EIGEN_ELPA XTS_EIGEN_ELPA
#define RGMIN_EIGEN_ELPA2 XTS_EIGEN_ELPA2
#define RGMIN_EIGEN_SLATE XTS_EIGEN_SLATE
#define RGMIN_EIGEN_MAGMA XTS_EIGEN_MAGMA
#define RGMIN_EIGEN_CUSOLVER XTS_EIGEN_CUSOLVER
#define RGMIN_EIGEN_DLA_FUTURE XTS_EIGEN_DLA_FUTURE
#define RGMIN_EIGEN_EIGENEXA XTS_EIGEN_EIGENEXA
#define RGMIN_UNAVAILABLE XTS_UNAVAILABLE
#endif

#ifdef __cplusplus
}
#endif

#endif /* XTS_OPTIMIZE_H */
