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
};

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
    std::size_t pair_count() const noexcept { return rgmin_solver_pair_count(ptr_); }
    void set_maxmove(double m) { rgmin_solver_set_maxmove(ptr_, m); }
    void set_qn_step(rgmin_qn_step_t step) { rgmin_solver_set_qn_step(ptr_, step); }
    void set_accept(rgmin_accept_t accept) { rgmin_solver_set_accept(ptr_, accept); }
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

    Report step(rgmin_eval_fn eval, rgmin_grad_fn grad, void* user,
                DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st = rgmin_solver_step(ptr_, eval, grad, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            char const* msg = rgmin_last_error();
            throw std::runtime_error(msg ? msg : "rgmin_solver_step failed");
        }
        return Report{out.value, out.steps, out.grad_norm};
    }

    Report step_fg(rgmin_evalgrad_fn evalgrad, void* user,
                   DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_fg(ptr_, evalgrad, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            char const* msg = rgmin_last_error();
            throw std::runtime_error(msg ? msg : "rgmin_solver_step_fg failed");
        }
        return Report{out.value, out.steps, out.grad_norm};
    }

    Report step_hess(rgmin_eval_fn eval, rgmin_grad_fn grad, rgmin_hess_fn hess,
                     void* user, DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_hess(ptr_, eval, grad, hess, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            char const* msg = rgmin_last_error();
            throw std::runtime_error(msg ? msg : "rgmin_solver_step_hess failed");
        }
        return Report{out.value, out.steps, out.grad_norm};
    }

    Report step_hess_fg(rgmin_evalgrad_fn evalgrad, rgmin_hess_fn hess,
                        void* user, DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_hess_fg(ptr_, evalgrad, hess, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            char const* msg = rgmin_last_error();
            throw std::runtime_error(msg ? msg : "rgmin_solver_step_hess_fg failed");
        }
        return Report{out.value, out.steps, out.grad_norm};
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
