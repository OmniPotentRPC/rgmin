#pragma once

/**
 * \file rgmin/optimize.hpp
 * \brief C++ API for the Rust rgmin hourglass.
 *
 * Solvers live in Rust. This header wraps rgmin_minimize over dlpk tensors.
 * It does not reimplement BFGS, L-BFGS, NLCG, or line search in C++.
 * New C++ includes this header. include/xts/optimize.hpp aliases the names.
 */

#include "../rgmin.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rgmin {

inline const char* version() noexcept { return rgmin_version(); }

enum class Method {
    PolakRibiere = RGMIN_POLAK_RIBIERE,
    FletcherReeves = RGMIN_FLETCHER_REEVES,
    Bfgs = RGMIN_BFGS,
    Lbfgs = RGMIN_LBFGS,
    Sr1 = RGMIN_SR1,
    Adam = RGMIN_ADAM,
    Steepest = RGMIN_STEEPEST,
    Sr2 = RGMIN_SR2,
    Pso = RGMIN_PSO,
    HestenesStiefel = RGMIN_HESTENES_STIEFEL,
    DaiYuan = RGMIN_DAI_YUAN,
    ConjugateDescent = RGMIN_CONJUGATE_DESCENT,
    HagerZhang = RGMIN_HAGER_ZHANG,
    LiuStorey = RGMIN_LIU_STOREY,
    FrPr = RGMIN_FR_PR,
    Newton = RGMIN_NEWTON,
    Rfo = RGMIN_RFO,
    Fire = RGMIN_FIRE,
    Bb = RGMIN_BB,
    Dogleg = RGMIN_DOGLEG,
    Fire2 = RGMIN_FIRE2,
    NewtonCg = RGMIN_NEWTON_CG,
    NewtonPcg = RGMIN_NEWTON_PCG,
    PolakRibierePlus = RGMIN_POLAK_RIBIERE_PLUS,
    EonCg = RGMIN_EON_CG,
};

/** Parse a method token. \c newton-cg and \c newton_cg are the same method. */
inline Method method_from_name(std::string const& name) {
    rgmin_method_t tag{};
    if (rgmin_method_from_name(name.c_str(), &tag) != RGMIN_SUCCESS) {
        char const* msg = rgmin_last_error();
        throw std::invalid_argument(msg ? msg : "unknown method");
    }
    return static_cast<Method>(static_cast<int>(tag));
}

struct Control {
    std::size_t maxiter = 100;
    double gtol = 1e-8;
    double istep = 0.1;
    std::size_t memory = 10;
    double maxmove = 0.0;
};

struct Report {
    double value = 0.0;
    std::size_t steps = 0;
    double grad_norm = 0.0;
};

using ScalarType = double;

struct OptimizeControl {
    std::size_t max_iterations = 100;
    ScalarType gtol = 1e-8;
    ScalarType tol = 1e-8;
    ScalarType istep = 0.1;
    std::size_t memory = 10;
    bool verbose = false;
    ScalarType maxmove = 1000;
    ScalarType xtol = 1e-6;
    ScalarType ftol = 1e-6;

    OptimizeControl() = default;
    OptimizeControl(std::size_t miter_val, ScalarType tol_val, bool verb_val)
        : max_iterations{miter_val}, gtol{tol_val}, tol{tol_val}, verbose{verb_val} {}

    Control to_control() const {
        return Control{max_iterations, gtol, istep, memory, maxmove};
    }
};

struct OptimizeResult {
    ScalarType fun = 0.0;
    std::size_t nit = 0;
    ScalarType grad_norm = 0.0;
    bool success = true;
    int status = 0;

    static OptimizeResult from_report(Report const& r) {
        OptimizeResult out;
        out.fun = r.value;
        out.nit = r.steps;
        out.grad_norm = r.grad_norm;
        return out;
    }
};

inline Report minimize_fn(rgmin_eval_fn eval, rgmin_grad_fn grad, void* user,
                          DLManagedTensorVersioned* x, Control const& ctrl,
                          Method method) {
    rgmin_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory,
                      ctrl.maxmove};
    rgmin_report_t out{};
    rgmin_status_t st =
        rgmin_minimize(eval, grad, user, x, &c, static_cast<rgmin_method_t>(method), &out);
    if (st != RGMIN_SUCCESS) {
        char const* msg = rgmin_last_error();
        throw std::runtime_error(msg ? msg : "rgmin_minimize failed");
    }
    return Report{out.value, out.steps, out.grad_norm};
}

inline Report minimize_hess_fn(rgmin_eval_fn eval, rgmin_grad_fn grad, rgmin_hess_fn hess,
                               void* user, DLManagedTensorVersioned* x,
                               Control const& ctrl, Method method) {
    rgmin_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory,
                      ctrl.maxmove};
    rgmin_report_t out{};
    rgmin_status_t st = rgmin_minimize_hess(
        eval, grad, hess, user, x, &c, static_cast<rgmin_method_t>(method), &out);
    if (st != RGMIN_SUCCESS) {
        char const* msg = rgmin_last_error();
        throw std::runtime_error(msg ? msg : "rgmin_minimize_hess failed");
    }
    return Report{out.value, out.steps, out.grad_norm};
}

inline Report minimize_eindir(const eindir_objective_t* objective,
                              const eindir_abi_stamp_t* stamp,
                              DLManagedTensorVersioned* x, Control const& ctrl,
                              Method method) {
    rgmin_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory,
                      ctrl.maxmove};
    rgmin_report_t out{};
    rgmin_status_t st = rgmin_minimize_eindir(
        objective, stamp, x, &c, static_cast<rgmin_method_t>(method), &out);
    if (st != RGMIN_SUCCESS) {
        char const* msg = rgmin_last_error();
        throw std::runtime_error(msg ? msg : "rgmin_minimize_eindir failed");
    }
    return Report{out.value, out.steps, out.grad_norm};
}

/// One-shot solve. Named for the header rather than `minimize`, which
/// the compat namespace below already takes: a namespace and a function
/// cannot share a name in one scope.
inline OptimizeResult optimize(rgmin_eval_fn eval, rgmin_grad_fn grad, void* user,
                               DLManagedTensorVersioned* x,
                               OptimizeControl const& ctrl, Method method) {
    return OptimizeResult::from_report(
        minimize_fn(eval, grad, user, x, ctrl.to_control(), method));
}

inline DLManagedTensorVersioned* borrow_cpu_f64(double* data, std::size_t n) {
    return rgmin_tensor_borrow_cpu_f64(data, n);
}

/// k lowest eigenpairs of a symmetric product. `seed` and `modes` are
/// row-major n-by-k. `values` and `pairs` have length k. Block LOBPCG
/// (Knyazev 2001). Throws on a refused call. ABI minor 39.
inline void lowest_eigenpairs(rgmin_hvp_fn hvp, void* user, double const* x,
                              double const* seed, double* modes, double* values,
                              std::size_t n, std::size_t k,
                              rgmin_eigen_params_t const* params,
                              rgmin_eigen_options_t const* options,
                              rgmin_eigenpair_t* pairs) {
    if (hvp == nullptr || x == nullptr || seed == nullptr || modes == nullptr ||
        values == nullptr || pairs == nullptr || k == 0 || n == 0 ||
        n > static_cast<std::size_t>(-1) / k) {
        throw std::invalid_argument("rgmin_lowest_eigenpairs: bad block");
    }
    auto* xt = rgmin_tensor_borrow_cpu_f64(const_cast<double*>(x), n);
    auto* st = rgmin_tensor_borrow_cpu_f64(const_cast<double*>(seed), n * k);
    auto* mt = rgmin_tensor_borrow_cpu_f64(modes, n * k);
    auto* vt = rgmin_tensor_borrow_cpu_f64(values, k);
    if (xt == nullptr || st == nullptr || mt == nullptr || vt == nullptr) {
        rgmin_tensor_free(xt);
        rgmin_tensor_free(st);
        rgmin_tensor_free(mt);
        rgmin_tensor_free(vt);
        throw std::invalid_argument("rgmin_lowest_eigenpairs: null buffer");
    }
    rgmin_status_t status = rgmin_lowest_eigenpairs(
        hvp, user, xt, st, mt, vt, params, options, pairs, k);
    rgmin_tensor_free(xt);
    rgmin_tensor_free(st);
    rgmin_tensor_free(mt);
    rgmin_tensor_free(vt);
    if (status != RGMIN_SUCCESS) {
        char const* msg = rgmin_last_error();
        throw std::invalid_argument(msg != nullptr ? msg
                                                   : "rgmin_lowest_eigenpairs failed");
    }
}

/// Thrown when a distance constraint set is refused or cannot be met.
/// kind() names why; the message is rgmin_last_error.
class ConstraintError : public std::runtime_error {
    rgmin_constraint_error_t kind_;

public:
    ConstraintError(char const* msg, rgmin_constraint_error_t kind)
        : std::runtime_error(msg), kind_(kind) {}
    rgmin_constraint_error_t kind() const noexcept { return kind_; }
};

namespace detail {

/// ConstraintError for RGMIN_CONSTRAINT_ERROR, std::runtime_error otherwise.
[[noreturn]] inline void raise(rgmin_status_t st, char const* fallback) {
    char const* msg = rgmin_last_error();
    msg = (msg != nullptr && msg[0] != '\0') ? msg : fallback;
    if (st == RGMIN_CONSTRAINT_ERROR) {
        throw ConstraintError(msg, rgmin_last_constraint_error());
    }
    throw std::runtime_error(msg);
}

}  // namespace detail

enum class ConstraintSolver {
    Auto = RGMIN_CONSTRAINT_SOLVER_AUTO,
    Shake = RGMIN_CONSTRAINT_SOLVER_SHAKE,
    Lincs = RGMIN_CONSTRAINT_SOLVER_LINCS,
};

/// RAII set of distances |x_i - x_j| = d over packed 3 N coordinates.
/// The with_* setters throw ConstraintError on a refused value.
class DistanceConstraints {
    rgmin_constraints_t* ptr_ = nullptr;

    explicit DistanceConstraints(rgmin_constraints_t* ptr) : ptr_(ptr) {
        if (ptr_ == nullptr) {
            detail::raise(rgmin_last_constraint_error() == RGMIN_CONSTRAINT_NONE
                              ? RGMIN_INVALID_PARAMETER
                              : RGMIN_CONSTRAINT_ERROR,
                          "rgmin_constraints_create failed");
        }
    }
    DistanceConstraints& check(rgmin_status_t st, char const* what) {
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, what);
        }
        return *this;
    }

public:
    /// Fix pair (pair_i[k], pair_j[k]) at lengths[k].
    DistanceConstraints(std::uint32_t const* pair_i, std::uint32_t const* pair_j,
                        double const* lengths, std::size_t npair)
        : DistanceConstraints(rgmin_constraints_create(pair_i, pair_j, lengths, npair)) {}
    /// Fix each pair at its distance in x of length n.
    static DistanceConstraints at(std::uint32_t const* pair_i, std::uint32_t const* pair_j,
                                  std::size_t npair, double const* x, std::size_t n) {
        return DistanceConstraints(rgmin_constraints_create_at(pair_i, pair_j, npair, x, n));
    }
    ~DistanceConstraints() { rgmin_constraints_free(ptr_); }
    DistanceConstraints(DistanceConstraints const&) = delete;
    DistanceConstraints& operator=(DistanceConstraints const&) = delete;
    DistanceConstraints(DistanceConstraints&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
    DistanceConstraints& operator=(DistanceConstraints&& o) noexcept {
        if (this != &o) {
            rgmin_constraints_free(ptr_);
            ptr_ = o.ptr_;
            o.ptr_ = nullptr;
        }
        return *this;
    }

    /// One mass per atom; corrections and projections use M^-1.
    DistanceConstraints& with_masses(double const* masses, std::size_t n_atoms) {
        return check(rgmin_constraints_set_masses(ptr_, masses, n_atoms), "set_masses failed");
    }
    DistanceConstraints& with_solver(ConstraintSolver solver) {
        return check(rgmin_constraints_set_solver(
                         ptr_, static_cast<rgmin_constraint_solver_t>(solver)),
                     "set_solver failed");
    }
    DistanceConstraints& with_settle(bool on) {
        return check(rgmin_constraints_set_settle(ptr_, on ? 1 : 0), "set_settle failed");
    }
    /// GROMACS shake-tol measure; zero is the rounding floor.
    DistanceConstraints& with_tolerance(double tolerance) {
        return check(rgmin_constraints_set_tolerance(ptr_, tolerance), "set_tolerance failed");
    }
    /// GROMACS lincs-order; zero derives it.
    DistanceConstraints& with_lincs_order(std::size_t order) {
        return check(rgmin_constraints_set_lincs_order(ptr_, order), "set_lincs_order failed");
    }
    /// At most one when every constraint is met at x.
    double excess(double const* x, std::size_t n) const noexcept {
        return rgmin_constraints_excess(ptr_, x, n);
    }
    /// Return x to the set from anchor. Throws ConstraintError and leaves x on failure.
    void constrain(double const* anchor, double* x, std::size_t n) const {
        rgmin_status_t st = rgmin_constraints_constrain(ptr_, anchor, x, n);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_constraints_constrain failed");
        }
    }
    rgmin_constraints_t const* get() const noexcept { return ptr_; }
};

namespace minimize {

inline constexpr Method BFGSOptimizer = Method::Bfgs;
inline constexpr Method LBFGSOptimizer = Method::Lbfgs;
inline constexpr Method SR1Optimizer = Method::Sr1;
inline constexpr Method SR2Optimizer = Method::Sr2;
inline constexpr Method ADAMOptimizer = Method::Adam;
inline constexpr Method SteepestDescentOptimizer = Method::Steepest;
inline constexpr Method ConjugateGradientOptimizer = Method::PolakRibiere;
inline constexpr Method PSOptim = Method::Pso;

}  // namespace minimize

/// RAII session. One step() is one outer iteration.
class Solver {
    rgmin_solver_t* ptr_ = nullptr;

public:
    Solver(Method method, Control const& ctrl, std::size_t dim) {
        rgmin_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory,
                          ctrl.maxmove};
        ptr_ = rgmin_solver_create(static_cast<rgmin_method_t>(method), &c, dim);
        if (ptr_ == nullptr) {
            char const* msg = rgmin_last_error();
            throw std::runtime_error(msg ? msg : "rgmin_solver_create failed");
        }
    }
    ~Solver() { rgmin_solver_free(ptr_); }
    Solver(Solver const&) = delete;
    Solver& operator=(Solver const&) = delete;
    Solver(Solver&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
    Solver& operator=(Solver&& o) noexcept {
        if (this != &o) {
            rgmin_solver_free(ptr_);
            ptr_ = o.ptr_;
            o.ptr_ = nullptr;
        }
        return *this;
    }

    void forget() { rgmin_solver_forget(ptr_); }
    /// Keep the method memory, drop the point (see rgmin_solver_rebase).
    void rebase() { rgmin_solver_rebase(ptr_); }
    std::size_t pair_count() const noexcept { return rgmin_solver_pair_count(ptr_); }
    /** Gradient evaluations. Finite-difference products are hvp_calls(). */
    std::size_t gradient_calls() const noexcept {
        return rgmin_solver_gradient_calls(ptr_);
    }
    /** Hessian-vector products, analytic or forward-difference. */
    std::size_t hvp_calls() const noexcept { return rgmin_solver_hvp_calls(ptr_); }
    void set_maxmove(double m) { rgmin_solver_set_maxmove(ptr_, m); }
    void set_qn_step(rgmin_qn_step_t step) { rgmin_solver_set_qn_step(ptr_, step); }
    /// Install the Packwood exponential preconditioner.
    /// A null distance pointer means Cartesian pairs. mu NaN estimates
    /// the energy scale on the next step.
    rgmin_status_t set_exp(rgmin_exp_t const* params, std::uint32_t const* pair_i,
                           std::uint32_t const* pair_j, double const* distance,
                           std::size_t npair, std::uint32_t const* anchor_i,
                           double const* anchor_r, std::size_t nanchor) {
        return rgmin_solver_set_exp(ptr_, params, pair_i, pair_j, distance, npair, anchor_i,
                                    anchor_r, nanchor);
    }
    /// Drop the preconditioner installed by set_exp.
    void clear_exp() { rgmin_solver_clear_exp(ptr_); }
    /// Energy scale, or NaN when none is installed.
    double exp_mu() const noexcept { return rgmin_solver_exp_mu(ptr_); }
    void set_accept(rgmin_accept_t accept) { rgmin_solver_set_accept(ptr_, accept); }
    /// Stop norm for gtol and Report::grad_norm. Returns 0, or 1 on an unknown norm.
    int set_stop_norm(rgmin_stop_norm_t norm) { return rgmin_solver_set_stop_norm(ptr_, norm); }
    /// FIRE 2.0 parameter set. Returns 0, or 1 on an unknown preset or a non-FIRE 2.0 session.
    int set_fire_preset(rgmin_fire_preset_t preset) {
        return rgmin_solver_set_fire_preset(ptr_, preset);
    }
    /// ABC-FIRE on a FIRE 2.0 session. Returns 0, or 1 on a non-FIRE 2.0 session.
    int set_fire_abc(bool on) { return rgmin_solver_set_fire_abc(ptr_, on ? 1 : 0); }
    /// eOn L-BFGS cap. A non-positive cap restores 0.2.
    /// Returns 0, or 1 when the session cannot run the capped step.
    int set_lbfgs_capped(double max_atom) { return rgmin_solver_set_lbfgs_capped(ptr_, max_atom); }
    /// ASE length control. A non-positive cap restores 0.2.
    /// Returns 0, or 1 when the session cannot run the scaled step.
    int set_lbfgs_scaled(double max_atom) { return rgmin_solver_set_lbfgs_scaled(ptr_, max_atom); }
    /// Strong Wolfe line search. The default session does not search a line.
    int set_lbfgs_wolfe() { return rgmin_solver_set_lbfgs_wolfe(ptr_); }
    /// Copy the gradient behind the last step report. n is the session dimension.
    rgmin_status_t report_gradient(double* out, std::size_t n) const {
        return rgmin_solver_report_gradient(ptr_, out, n);
    }
    /// Per-atom cap. Euclidean L-BFGS without a distance constraint set
    /// selects the geometry step and ignores set_accept. With a set,
    /// L-BFGS keeps its two-loop step under set_accept and the cap
    /// scales each trial before it returns to the set. A nominal
    /// steepest trial at least 64 times the cap takes one
    /// finite-difference curvature probe, clipped to the objective
    /// bounds. Ordinary starts use 0.1 I and a five-value nonmonotone
    /// window. An installed exponential preconditioner replaces that
    /// scalar seed, including the steepest fallbacks. Non-positive
    /// clears it.
    void set_atom_maxmove(double m) { rgmin_solver_set_atom_maxmove(ptr_, m); }
    void set_project_rigid(bool on) {
        rgmin_solver_set_project_rigid(ptr_, on ? 1 : 0);
    }
    void set_extra_updates(std::size_t n) {
        rgmin_solver_set_extra_updates(ptr_, n);
    }
    void set_cautious(double eps, double alpha) {
        rgmin_solver_set_cautious(ptr_, eps, alpha);
    }
    int set_highs(bool on) { return rgmin_solver_set_highs(ptr_, on ? 1 : 0); }
    int set_box(double const* lower, double const* upper, std::size_t n) {
        return rgmin_solver_set_box(ptr_, lower, upper, n);
    }
    int push_pair(double const* s, double const* y, std::size_t n) {
        return rgmin_solver_push_pair(ptr_, s, y, n);
    }
    int search_direction(double const* grad, double* dir, std::size_t n) {
        return rgmin_solver_search_direction(ptr_, grad, dir, n);
    }
    int set_trust(double radius) { return rgmin_solver_set_trust(ptr_, radius); }
    int add_equality(std::size_t const* idx, double const* coef, std::size_t nnz,
                     double rhs) {
        return rgmin_solver_add_equality(ptr_, idx, coef, nnz, rhs);
    }
    int clear_equalities() { return rgmin_solver_clear_equalities(ptr_); }
    int set_highs_solver(rgmin_highs_solver_t kind) {
        return rgmin_solver_set_highs_solver(ptr_, kind);
    }
    int set_highs_crossover(rgmin_highs_crossover_t kind) {
        return rgmin_solver_set_highs_crossover(ptr_, kind);
    }
    int set_highs_callback(rgmin_highs_callback_t cb, void* user) {
        return rgmin_solver_set_highs_callback(ptr_, cb, user);
    }
    void set_manifold(rgmin_manifold_t m) { rgmin_solver_set_manifold(ptr_, m); }
    void set_oblique(std::size_t n, std::size_t m) {
        rgmin_solver_set_oblique(ptr_, n, m);
    }
    void set_stiefel(std::size_t n, std::size_t p) {
        rgmin_solver_set_stiefel(ptr_, n, p);
    }
    void set_complex_circle(std::size_t n) {
        rgmin_solver_set_complex_circle(ptr_, n);
    }
    void set_euclidean_complex(std::size_t n) {
        rgmin_solver_set_euclidean_complex(ptr_, n);
    }
    void set_constant(std::size_t n) { rgmin_solver_set_constant(ptr_, n); }
    void set_multinomial_ds(std::size_t n) {
        rgmin_solver_set_multinomial_ds(ptr_, n);
    }
    void set_multinomial_sym(std::size_t n) {
        rgmin_solver_set_multinomial_sym(ptr_, n);
    }
    void set_sphere_complex(std::size_t n) {
        rgmin_solver_set_sphere_complex(ptr_, n);
    }
    void set_positive(std::size_t n) { rgmin_solver_set_positive(ptr_, n); }
    void set_centered_matrix(std::size_t m, std::size_t n, bool center_rows) {
        rgmin_solver_set_centered_matrix(ptr_, m, n, center_rows ? 1 : 0);
    }
    /// Copy set into the session. Throws ConstraintError on a refused setting.
    void set_distance_constraints(DistanceConstraints const& set) {
        rgmin_status_t st = rgmin_solver_set_distance_constraints(ptr_, set.get());
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_solver_set_distance_constraints failed");
        }
    }
    void clear_distance_constraints() { rgmin_solver_clear_distance_constraints(ptr_); }

    Report step(rgmin_eval_fn eval, rgmin_grad_fn grad, void* user,
                DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st = rgmin_solver_step(ptr_, eval, grad, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_solver_step failed");
        }
        return Report{out.value, out.steps, out.grad_norm};
    }

    Report step_fg(rgmin_evalgrad_fn evalgrad, void* user,
                   DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_fg(ptr_, evalgrad, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_solver_step_fg failed");
        }
        return Report{out.value, out.steps, out.grad_norm};
    }

    Report step_hvp(rgmin_eval_fn eval, rgmin_grad_fn grad, rgmin_hvp_fn hvp,
                    void* user, DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_hvp(ptr_, eval, grad, hvp, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_solver_step_hvp failed");
        }
        return Report{out.value, out.steps, out.grad_norm};
    }

    Report step_hess(rgmin_eval_fn eval, rgmin_grad_fn grad, rgmin_hess_fn hess,
                     void* user, DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_hess(ptr_, eval, grad, hess, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_solver_step_hess failed");
        }
        return Report{out.value, out.steps, out.grad_norm};
    }

    Report step_hess_fg(rgmin_evalgrad_fn evalgrad, rgmin_hess_fn hess,
                        void* user, DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_hess_fg(ptr_, evalgrad, hess, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_solver_step_hess_fg failed");
        }
        return Report{out.value, out.steps, out.grad_norm};
    }
};

/// Stored `(s, y)` pairs for one L-BFGS direction. The handle does not
/// evaluate an objective. `capacity` 0 keeps ten pairs.
class LbfgsPairs {
    rgmin_lbfgs_pairs_t* ptr_ = nullptr;

    explicit LbfgsPairs(rgmin_lbfgs_pairs_t* ptr) : ptr_(ptr) {
        if (ptr_ == nullptr) {
            detail::raise(RGMIN_INVALID_PARAMETER, "rgmin_lbfgs_pairs_create failed");
        }
    }

public:
    LbfgsPairs(std::size_t n, std::size_t capacity)
        : LbfgsPairs(rgmin_lbfgs_pairs_create(n, capacity)) {}
    ~LbfgsPairs() { rgmin_lbfgs_pairs_free(ptr_); }
    LbfgsPairs(LbfgsPairs const&) = delete;
    LbfgsPairs& operator=(LbfgsPairs const&) = delete;
    LbfgsPairs(LbfgsPairs&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    LbfgsPairs& operator=(LbfgsPairs&& other) noexcept {
        if (this != &other) {
            rgmin_lbfgs_pairs_free(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    /// True when the pair was stored. A non-positive curvature pair
    /// returns false. A bad length throws.
    bool push(double const* s, double const* y, std::size_t n) {
        int32_t stored = 0;
        rgmin_status_t st = rgmin_lbfgs_pairs_push(ptr_, s, y, n, &stored);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_lbfgs_pairs_push failed");
        }
        return stored != 0;
    }
    std::size_t count() const noexcept { return rgmin_lbfgs_pairs_count(ptr_); }
    void clear() noexcept { rgmin_lbfgs_pairs_clear(ptr_); }

    /// Write `-H g` for a diagonal `H0`. An empty memory writes `-H0 g`.
    void direction_diagonal(double const* grad, double const* h0, double* dir,
                            std::size_t n) const {
        rgmin_status_t st = rgmin_lbfgs_direction(ptr_, grad, h0, nullptr, nullptr, dir, n);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_lbfgs_direction failed");
        }
    }
    /// Write `-H g` for a callback `H0`. The callback is not an objective.
    void direction_apply(double const* grad, rgmin_h0_fn apply, void* user, double* dir,
                         std::size_t n) const {
        rgmin_status_t st = rgmin_lbfgs_direction(ptr_, grad, nullptr, apply, user, dir, n);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_lbfgs_direction failed");
        }
    }
};

/// RAII Stiefel St(n, p), SPD(p) or product manifold. Every pointer
/// argument is a caller-owned column-major buffer of len() doubles; a
/// product packs its factors in order. A refused call throws
/// std::runtime_error with rgmin_last_error. ABI minor 41.
class Manifold {
    rgmin_manifold_handle_t* ptr_ = nullptr;

    explicit Manifold(rgmin_manifold_handle_t* ptr) : ptr_(ptr) {
        if (ptr_ == nullptr) {
            detail::raise(RGMIN_INVALID_PARAMETER, "rgmin_manifold create failed");
        }
    }
    static void check(rgmin_status_t st, char const* what) {
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, what);
        }
    }

public:
    static Manifold stiefel(std::size_t n, std::size_t p,
                            rgmin_stiefel_retraction_t retraction = RGMIN_STIEFEL_QR) {
        return Manifold(rgmin_manifold_stiefel(n, p, static_cast<int32_t>(retraction)));
    }
    static Manifold spd(std::size_t p, rgmin_spd_metric_t metric = RGMIN_SPD_AFFINE_INVARIANT) {
        return Manifold(rgmin_manifold_spd(p, static_cast<int32_t>(metric)));
    }
    /// Product of the factors in order. The factors stay usable.
    template <class... Factors>
    static Manifold product(Manifold const& first, Factors const&... rest) {
        rgmin_manifold_handle_t const* handles[] = {first.get(), rest.get()...};
        return Manifold(rgmin_manifold_product(handles, sizeof...(Factors) + 1));
    }

    ~Manifold() { rgmin_manifold_free(ptr_); }
    Manifold(Manifold const&) = delete;
    Manifold& operator=(Manifold const&) = delete;
    Manifold(Manifold&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    Manifold& operator=(Manifold&& other) noexcept {
        if (this != &other) {
            rgmin_manifold_free(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    rgmin_manifold_handle_t const* get() const noexcept { return ptr_; }
    /// Packed length of one point.
    std::size_t len() const noexcept { return rgmin_manifold_len(ptr_); }

    void project(double const* x, double const* v, double* out) const {
        check(rgmin_manifold_project(ptr_, x, v, out, len()), "rgmin_manifold_project failed");
    }
    void egrad_to_rgrad(double const* x, double const* egrad, double* out) const {
        check(rgmin_manifold_egrad_to_rgrad(ptr_, x, egrad, out, len()),
              "rgmin_manifold_egrad_to_rgrad failed");
    }
    void retract(double const* x, double const* v, double* out) const {
        check(rgmin_manifold_retract(ptr_, x, v, out, len()), "rgmin_manifold_retract failed");
    }
    void exp(double const* x, double const* v, double* out) const {
        check(rgmin_manifold_exp(ptr_, x, v, out, len()), "rgmin_manifold_exp failed");
    }
    void log(double const* x, double const* y, double* out) const {
        check(rgmin_manifold_log(ptr_, x, y, out, len()), "rgmin_manifold_log failed");
    }
    void transport(double const* x_from, double const* x_to, double const* v, double* out) const {
        check(rgmin_manifold_transport(ptr_, x_from, x_to, v, out, len()),
              "rgmin_manifold_transport failed");
    }
    double inner(double const* x, double const* u, double const* v) const {
        double out = 0.0;
        check(rgmin_manifold_inner(ptr_, x, u, v, &out, len()), "rgmin_manifold_inner failed");
        return out;
    }
    /// `rgrad_out` is the Riemannian gradient of `egrad` at `x`; `x_out`
    /// is the retraction of `-t rgrad_out`.
    void gradient_step(double const* x, double const* egrad, double t, double* rgrad_out,
                       double* x_out) const {
        check(rgmin_manifold_gradient_step(ptr_, x, egrad, t, rgrad_out, x_out, len()),
              "rgmin_manifold_gradient_step failed");
    }
};

namespace nlcg {
namespace conjugacy {

inline constexpr Method PolakRibiere = Method::PolakRibiere;
inline constexpr Method FletcherReeves = Method::FletcherReeves;
inline constexpr Method HestenesStiefel = Method::HestenesStiefel;
inline constexpr Method DaiYuan = Method::DaiYuan;
inline constexpr Method ConjugateDescent = Method::ConjugateDescent;
inline constexpr Method HagerZhang = Method::HagerZhang;
inline constexpr Method LiuStorey = Method::LiuStorey;
inline constexpr Method FrPr = Method::FrPr;

}  // namespace conjugacy
}  // namespace nlcg

}  // namespace rgmin
