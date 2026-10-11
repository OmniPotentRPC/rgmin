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

#include <atomic>
#include <cctype>
#include <cstddef>
#include <exception>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
    QuickMin = RGMIN_QUICKMIN,
    Diis = RGMIN_DIIS,
};

enum class EigenKind {
    Lanczos = RGMIN_EIGEN_LANCZOS,
    RayleighRitz = RGMIN_EIGEN_RAYLEIGH_RITZ,
    JacobiDavidson = RGMIN_EIGEN_JACOBI_DAVIDSON,
    Lobpcg = RGMIN_EIGEN_LOBPCG,
    Primme = RGMIN_EIGEN_PRIMME,
    Slepc = RGMIN_EIGEN_SLEPC,
    Chase = RGMIN_EIGEN_CHASE,
    Elpa = RGMIN_EIGEN_ELPA,
    Elpa2 = RGMIN_EIGEN_ELPA2,
    Slate = RGMIN_EIGEN_SLATE,
    Magma = RGMIN_EIGEN_MAGMA,
    Cusolver = RGMIN_EIGEN_CUSOLVER,
    DlaFuture = RGMIN_EIGEN_DLA_FUTURE,
    EigenExa = RGMIN_EIGEN_EIGENEXA,
    Dimer = RGMIN_EIGEN_DIMER,
    Libkrylov = RGMIN_EIGEN_LIBKRYLOV,
};

struct EigenParams {
    EigenKind kind = EigenKind::Lanczos;
    uint32_t nev = 1;
    uint32_t krylov = 0;
    uint32_t max_iter = 0;
    double tol = 0.0;
};

struct EigenOptions {
    uint32_t degree = 0;
    uint32_t extra = 0;
};

inline rgmin_lowest_mode_t lowest_eigenpair(
    rgmin_hvp_fn hvp, void* user, const DLManagedTensorVersioned* x,
    const DLManagedTensorVersioned* seed, DLManagedTensorVersioned* mode,
    EigenParams const& params = {}, EigenOptions const& options = {}) {
    rgmin_eigen_params_t c{static_cast<int32_t>(params.kind), params.nev,
                           params.krylov, params.max_iter, params.tol};
    rgmin_eigen_options_t extra{options.degree, options.extra};
    rgmin_lowest_mode_t result{};
    rgmin_status_t st = rgmin_lowest_eigenpair_with_options(
        hvp, user, x, seed, mode, &c, &extra, &result);
    if (st != RGMIN_SUCCESS) {
        char const* msg = rgmin_last_error();
        throw std::runtime_error(msg ? msg : "rgmin_lowest_eigenpair failed");
    }
    return result;
}

struct Control {
    std::size_t maxiter = 100;
    double gtol = 1e-8;
    double istep = 0.1;
    std::size_t memory = 10;
    double maxmove = 0.0;
    double noise_eps = 0.0;
    double noise_c = 0.0;
};

struct Report {
    double value = 0.0;
    std::size_t steps = 0;
    double grad_norm = 0.0;
    rgmin_stop_t stop = RGMIN_STOP_BUDGET;
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
        out.status = static_cast<int>(r.stop);
        return out;
    }
};

inline Report minimize_fn(rgmin_eval_fn eval, rgmin_grad_fn grad, void* user,
                          DLManagedTensorVersioned* x, Control const& ctrl,
                          Method method) {
    rgmin_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory,
                      ctrl.maxmove, ctrl.noise_eps, ctrl.noise_c};
    rgmin_report_t out{};
    rgmin_status_t st =
        rgmin_minimize(eval, grad, user, x, &c, static_cast<rgmin_method_t>(method), &out);
    if (st != RGMIN_SUCCESS) {
        char const* msg = rgmin_last_error();
        throw std::runtime_error(msg ? msg : "rgmin_minimize failed");
    }
    return Report{out.value, out.steps, out.grad_norm, out.stop};
}

inline Report minimize_hess_fn(rgmin_eval_fn eval, rgmin_grad_fn grad, rgmin_hess_fn hess,
                               void* user, DLManagedTensorVersioned* x,
                               Control const& ctrl, Method method) {
    rgmin_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory,
                      ctrl.maxmove, ctrl.noise_eps, ctrl.noise_c};
    rgmin_report_t out{};
    rgmin_status_t st = rgmin_minimize_hess(
        eval, grad, hess, user, x, &c, static_cast<rgmin_method_t>(method), &out);
    if (st != RGMIN_SUCCESS) {
        char const* msg = rgmin_last_error();
        throw std::runtime_error(msg ? msg : "rgmin_minimize_hess failed");
    }
    return Report{out.value, out.steps, out.grad_norm, out.stop};
}

inline Report minimize_eindir(const eindir_objective_t* objective,
                              const eindir_abi_stamp_t* stamp,
                              DLManagedTensorVersioned* x, Control const& ctrl,
                              Method method) {
    rgmin_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory,
                      ctrl.maxmove, ctrl.noise_eps, ctrl.noise_c};
    rgmin_report_t out{};
    rgmin_status_t st = rgmin_minimize_eindir(
        objective, stamp, x, &c, static_cast<rgmin_method_t>(method), &out);
    if (st != RGMIN_SUCCESS) {
        char const* msg = rgmin_last_error();
        throw std::runtime_error(msg ? msg : "rgmin_minimize_eindir failed");
    }
    return Report{out.value, out.steps, out.grad_norm, out.stop};
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

/// Fills \c rgmin_precon_t. The owned arrays keep the pointers valid.
/// Morse bonds and FF torsions are Rust-only.
class Precon {
    rgmin_precon_t raw_{};
    std::vector<int32_t> numbers_;
    std::vector<int32_t> ghost_z_;
    std::vector<double> ghost_pos_;
    std::vector<std::size_t> nbr_i_;
    std::vector<std::size_t> nbr_j_;
    std::vector<double> nbr_r_;
    std::vector<std::size_t> anchor_i_;
    std::vector<double> anchor_r_;
    std::vector<rgmin_bond_t> bonds_;
    std::vector<rgmin_angle_t> angles_;

    void bind() noexcept {
        raw_.numbers = numbers_.empty() ? nullptr : numbers_.data();
        raw_.natoms = numbers_.size();
        raw_.ghost_z = ghost_z_.empty() ? nullptr : ghost_z_.data();
        raw_.ghost_pos = ghost_pos_.empty() ? nullptr : ghost_pos_.data();
        raw_.nghost = ghost_z_.size();
        raw_.nbr_i = nbr_i_.empty() ? nullptr : nbr_i_.data();
        raw_.nbr_j = nbr_j_.empty() ? nullptr : nbr_j_.data();
        raw_.nbr_r = nbr_r_.empty() ? nullptr : nbr_r_.data();
        raw_.npairs = nbr_i_.size();
        raw_.anchor_i = anchor_i_.empty() ? nullptr : anchor_i_.data();
        raw_.anchor_r = anchor_r_.empty() ? nullptr : anchor_r_.data();
        raw_.nanchor = anchor_i_.size();
        raw_.bonds = bonds_.empty() ? nullptr : bonds_.data();
        raw_.nbonds = bonds_.size();
        raw_.angles = angles_.empty() ? nullptr : angles_.data();
        raw_.nangles = angles_.size();
    }

    Precon() = default;

public:
    Precon(Precon const&) = delete;
    Precon& operator=(Precon const&) = delete;
    Precon(Precon&& o) noexcept
        : raw_(o.raw_),
          numbers_(std::move(o.numbers_)),
          ghost_z_(std::move(o.ghost_z_)),
          ghost_pos_(std::move(o.ghost_pos_)),
          nbr_i_(std::move(o.nbr_i_)),
          nbr_j_(std::move(o.nbr_j_)),
          nbr_r_(std::move(o.nbr_r_)),
          anchor_i_(std::move(o.anchor_i_)),
          anchor_r_(std::move(o.anchor_r_)),
          bonds_(std::move(o.bonds_)),
          angles_(std::move(o.angles_)) {
        bind();
    }
    Precon& operator=(Precon&& o) noexcept {
        if (this != &o) {
            raw_ = o.raw_;
            numbers_ = std::move(o.numbers_);
            ghost_z_ = std::move(o.ghost_z_);
            ghost_pos_ = std::move(o.ghost_pos_);
            nbr_i_ = std::move(o.nbr_i_);
            nbr_j_ = std::move(o.nbr_j_);
            nbr_r_ = std::move(o.nbr_r_);
            anchor_i_ = std::move(o.anchor_i_);
            anchor_r_ = std::move(o.anchor_r_);
            bonds_ = std::move(o.bonds_);
            angles_ = std::move(o.angles_);
            bind();
        }
        return *this;
    }

    /// Lindh model Hessian. \c numbers has one atomic number per free atom.
    static Precon lindh(std::vector<int32_t> numbers) {
        Precon p;
        rgmin_precon_init(&p.raw_, RGMIN_PRECON_LINDH);
        p.numbers_ = std::move(numbers);
        p.bind();
        return p;
    }
    /// Exponential metric. Pass neighbours before \c Solver::set_precon.
    static Precon exp() {
        Precon p;
        rgmin_precon_init(&p.raw_, RGMIN_PRECON_EXP);
        return p;
    }
    /// Bonded metric. Harmonic bonds only. Morse and torsions are Rust-only.
    static Precon ff(std::vector<rgmin_bond_t> bonds) {
        Precon p;
        rgmin_precon_init(&p.raw_, RGMIN_PRECON_FF);
        p.bonds_ = std::move(bonds);
        p.bind();
        return p;
    }
    /// \c select_precon. \c geometry_kind is \c RGMIN_GEOM_BULK and companions.
    static Precon automatic(std::vector<int32_t> numbers, int32_t geometry_kind) {
        Precon p;
        rgmin_precon_init(&p.raw_, RGMIN_PRECON_AUTO);
        p.numbers_ = std::move(numbers);
        p.raw_.geometry_kind = geometry_kind;
        p.bind();
        return p;
    }

    Precon& cell(double a, double b, double c) & {
        raw_.cell[0] = a;
        raw_.cell[1] = b;
        raw_.cell[2] = c;
        return *this;
    }
    Precon cell(double a, double b, double c) && {
        raw_.cell[0] = a;
        raw_.cell[1] = b;
        raw_.cell[2] = c;
        return std::move(*this);
    }
    Precon& mu(double v) {
        raw_.mu = v;
        return *this;
    }
    Precon& c_stab(double v) {
        raw_.c_stab = v;
        return *this;
    }
    Precon& r_cut(double v) {
        raw_.r_cut = v;
        return *this;
    }
    Precon& exp_a(double v) {
        raw_.exp_a = v;
        return *this;
    }
    Precon& dihedrals(bool on) {
        raw_.dihedrals = on ? 1 : 0;
        return *this;
    }
    Precon& neighbours(std::vector<std::size_t> i, std::vector<std::size_t> j,
                       std::vector<double> r) {
        nbr_i_ = std::move(i);
        nbr_j_ = std::move(j);
        nbr_r_ = std::move(r);
        bind();
        return *this;
    }
    Precon& angles(std::vector<rgmin_angle_t> angles) {
        angles_ = std::move(angles);
        bind();
        return *this;
    }
    rgmin_precon_t const* get() const noexcept { return &raw_; }
};

/// RAII session. One step() is one outer iteration.
/// F3: mutable. Python bindings claim \c busy() with \c rgbind::Claim
/// at the start of every method. A second caller gets
/// "Solver is in use by another thread".
class Solver {
    rgmin_solver_t* ptr_ = nullptr;
    std::size_t dim_ = 0;
    Method method_ = Method::Lbfgs;
    std::atomic<bool> busy_{false};

public:
    Solver(Method method, Control const& ctrl, std::size_t dim)
        : dim_(dim), method_(method) {
        rgmin_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory,
                          ctrl.maxmove, ctrl.noise_eps, ctrl.noise_c};
        ptr_ = rgmin_solver_create(static_cast<rgmin_method_t>(method), &c, dim);
        if (ptr_ == nullptr) {
            char const* msg = rgmin_last_error();
            throw std::runtime_error(msg ? msg : "rgmin_solver_create failed");
        }
    }
    ~Solver() { rgmin_solver_free(ptr_); }
    Solver(Solver const&) = delete;
    Solver& operator=(Solver const&) = delete;
    Solver(Solver&& o) noexcept : ptr_(o.ptr_), dim_(o.dim_), method_(o.method_) {
        o.ptr_ = nullptr;
        o.dim_ = 0;
    }
    Solver& operator=(Solver&& o) noexcept {
        if (this != &o) {
            rgmin_solver_free(ptr_);
            ptr_ = o.ptr_;
            dim_ = o.dim_;
            method_ = o.method_;
            o.ptr_ = nullptr;
            o.dim_ = 0;
        }
        return *this;
    }

    /// F3 flag. Bindings claim it with rgbind::Claim. The flag starts clear.
    std::atomic<bool>& busy() noexcept { return busy_; }
    std::atomic<bool> const& busy() const noexcept { return busy_; }
    std::size_t dim() const noexcept { return dim_; }
    Method method() const noexcept { return method_; }
    rgmin_solver_t* raw() noexcept { return ptr_; }

    void forget() { rgmin_solver_forget(ptr_); }
    /// Keep the method memory, drop the point (see rgmin_solver_rebase).
    void rebase() { rgmin_solver_rebase(ptr_); }
    /// Drop only the cached evaluation (see rgmin_solver_forget_evaluation).
    void forget_evaluation() { rgmin_solver_forget_evaluation(ptr_); }
    std::size_t pair_count() const noexcept { return rgmin_solver_pair_count(ptr_); }
    void set_maxmove(double m) { rgmin_solver_set_maxmove(ptr_, m); }
    void set_qn_step(rgmin_qn_step_t step) { rgmin_solver_set_qn_step(ptr_, step); }
    rgmin_status_t set_precon(Precon const& p) {
        return rgmin_solver_set_precon(ptr_, p.get());
    }
    void clear_precon() { rgmin_solver_clear_precon(ptr_); }
    double precon_scale() const { return rgmin_solver_precon_scale(ptr_); }
    void set_accept(rgmin_accept_t accept) { rgmin_solver_set_accept(ptr_, accept); }
    void set_fire_variant(rgmin_fire_variant_t variant) {
        rgmin_solver_set_fire_variant(ptr_, variant);
    }
    rgmin_status_t set_quickmin_cell(std::size_t at) {
        return rgmin_solver_set_quickmin_cell(ptr_, at);
    }
    rgmin_status_t set_linesearch(rgmin_linesearch_t kind, double c1,
                                 double c2, std::size_t maxiter) {
        return rgmin_solver_set_linesearch(ptr_, kind, c1, c2, maxiter);
    }
    rgmin_status_t set_objective_roundoff(double relative) {
        return rgmin_solver_set_objective_roundoff(ptr_, relative);
    }
    void set_masses(double const* masses, std::size_t n_atoms) {
        rgmin_solver_set_masses(ptr_, masses, n_atoms);
    }
    void set_periodic(bool periodic) {
        rgmin_solver_set_periodic(ptr_, periodic ? 1 : 0);
    }
    void set_atom_maxmove(double m) { rgmin_solver_set_atom_maxmove(ptr_, m); }
    void set_lbfgs_neb_guards(bool on) {
        rgmin_solver_set_lbfgs_neb_guards(ptr_, on ? 1 : 0);
    }
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
    int set_highs_solver(rgmin_highs_solver_t kind) {
        return rgmin_solver_set_highs_solver(ptr_, static_cast<int32_t>(kind));
    }
    int set_highs_crossover(rgmin_highs_crossover_t kind) {
        return rgmin_solver_set_highs_crossover(ptr_, static_cast<int32_t>(kind));
    }
    int set_highs_callback(rgmin_highs_callback_t callback, void* user) {
        return rgmin_solver_set_highs_callback(ptr_, callback, user);
    }
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
    int set_highs_trust(double radius) { return rgmin_solver_set_highs_trust(ptr_, radius); }
    int add_equality(std::size_t const* idx, double const* coef, std::size_t nnz,
                     double rhs) {
        return rgmin_solver_add_equality(ptr_, idx, coef, nnz, rhs);
    }
    int clear_equalities() { return rgmin_solver_clear_equalities(ptr_); }
    void set_manifold(rgmin_manifold_t m) { rgmin_solver_set_manifold(ptr_, m); }
    void set_stiefel(std::size_t n, std::size_t p) { rgmin_solver_set_stiefel(ptr_, n, p); }
    void set_oblique(std::size_t n, std::size_t m) { rgmin_solver_set_oblique(ptr_, n, m); }
    void set_complex_circle(std::size_t n) { rgmin_solver_set_complex_circle(ptr_, n); }
    void set_euclidean_complex(std::size_t n) { rgmin_solver_set_euclidean_complex(ptr_, n); }
    void set_constant(std::size_t n) { rgmin_solver_set_constant(ptr_, n); }
    void set_multinomial_ds(std::size_t n) { rgmin_solver_set_multinomial_ds(ptr_, n); }
    void set_multinomial_sym(std::size_t n) { rgmin_solver_set_multinomial_sym(ptr_, n); }
    void set_sphere_complex(std::size_t n) { rgmin_solver_set_sphere_complex(ptr_, n); }
    void set_positive(std::size_t n) { rgmin_solver_set_positive(ptr_, n); }
    void set_centered_matrix(std::size_t m, std::size_t n, bool center_rows) { rgmin_solver_set_centered_matrix(ptr_, m, n, center_rows); }
    void set_factor_shape(std::size_t n, std::size_t p) { rgmin_solver_set_factor_shape(ptr_, n, p); }
    void set_unitary(std::size_t n) { rgmin_solver_set_unitary(ptr_, n); }

    Report step(rgmin_eval_fn eval, rgmin_grad_fn grad, void* user,
                DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st = rgmin_solver_step(ptr_, eval, grad, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            char const* msg = rgmin_last_error();
            throw std::runtime_error(msg ? msg : "rgmin_solver_step failed");
        }
        return Report{out.value, out.steps, out.grad_norm, out.stop};
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
        return Report{out.value, out.steps, out.grad_norm, out.stop};
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
        return Report{out.value, out.steps, out.grad_norm, out.stop};
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
        return Report{out.value, out.steps, out.grad_norm, out.stop};
    }

    /// One outer iteration. \a evalgrad is `(x, n, grad) -> value`.
    /// The C call runs to completion before any stored exception is
    /// rethrown, so a callback error does not unwind through C or Rust.
    /// \a x is in/out and has length \c dim().
    template <class Fn>
    Report step(Fn&& evalgrad, double* x) {
        if (x == nullptr || dim_ == 0) {
            throw std::invalid_argument("Solver::step: x is empty");
        }
        struct Call {
            Fn* fn;
            std::exception_ptr error;
            std::size_t n;
        };
        Call call{&evalgrad, nullptr, dim_};
        auto entry = [](void* user, DLManagedTensorVersioned const* xt, double* value_out,
                        DLManagedTensorVersioned* grad_out) -> rgmin_status_t {
            auto* self = static_cast<Call*>(user);
            try {
                if (xt == nullptr || grad_out == nullptr || value_out == nullptr ||
                    xt->dl_tensor.data == nullptr || grad_out->dl_tensor.data == nullptr) {
                    throw std::invalid_argument("Solver::step: null tensor");
                }
                auto const* xp = reinterpret_cast<double const*>(
                    static_cast<char const*>(xt->dl_tensor.data) + xt->dl_tensor.byte_offset);
                auto* gp = reinterpret_cast<double*>(
                    static_cast<char*>(grad_out->dl_tensor.data) + grad_out->dl_tensor.byte_offset);
                *value_out = (*self->fn)(xp, self->n, gp);
                return RGMIN_SUCCESS;
            } catch (...) {
                self->error = std::current_exception();
                return RGMIN_INVALID_PARAMETER;
            }
        };
        DLManagedTensorVersioned* tensor = rgmin_tensor_borrow_cpu_f64(x, dim_);
        if (tensor == nullptr) {
            char const* msg = rgmin_last_error();
            throw std::invalid_argument(msg ? msg : "rgmin_tensor_borrow_cpu_f64 failed");
        }
        rgmin_report_t out{};
        rgmin_status_t st = rgmin_solver_step_fg(ptr_, entry, &call, tensor, &out);
        rgmin_tensor_free(tensor);
        if (call.error) {
            std::rethrow_exception(call.error);
        }
        if (st != RGMIN_SUCCESS) {
            char const* msg = rgmin_last_error();
            throw std::runtime_error(msg ? msg : "rgmin_solver_step_fg failed");
        }
        return Report{out.value, out.steps, out.grad_norm, out.stop};
    }
};

/// Immutable session configuration. Unknown option names throw
/// \c std::invalid_argument listing the valid names. The Python binding
/// maps that exception to ValueError.
class SolverConfig {
public:
    static std::string valid_options();
    static std::string valid_methods();

    static SolverConfig build(std::string_view method,
                              std::vector<std::pair<std::string, std::string>> const& text,
                              std::vector<std::pair<std::string, double>> const& numbers,
                              std::vector<std::pair<std::string, bool>> const& flags);

    Method method() const noexcept { return method_; }
    std::string const& method_name() const noexcept { return method_name_; }
    Control const& control() const noexcept { return control_; }
    void apply(Solver& solver) const;

private:
    Method method_ = Method::Lbfgs;
    std::string method_name_ = "lbfgs";
    Control control_{};
    std::optional<rgmin_accept_t> accept_;
    std::optional<double> atom_maxmove_;
    std::optional<bool> neb_guards_;
    std::optional<rgmin_fire_variant_t> fire_variant_;
    std::optional<rgmin_linesearch_t> linesearch_;
    double c1_ = 1e-4;
    double c2_ = 0.9;
    std::size_t linesearch_maxiter_ = 20;
    std::optional<rgmin_qn_step_t> qn_step_;
    std::optional<bool> periodic_;
    std::optional<bool> project_rigid_;
    std::optional<std::size_t> extra_updates_;
    std::optional<double> cautious_eps_;
    std::optional<double> cautious_alpha_;
    std::optional<double> objective_roundoff_;
};

inline std::string SolverConfig::valid_options() {
    return "accept, atom_maxmove, beta, c1, c2, cautious_alpha, cautious_eps, "
           "extra_updates, fire_variant, gtol, istep, linesearch, "
           "linesearch_maxiter, maxiter, maxmove, maxstep, memory, neb_guards, "
           "noise_c, noise_eps, objective_roundoff, periodic, project_rigid, qn_step";
}

inline std::string SolverConfig::valid_methods() {
    return "adam, bb, bfgs, cg, conjugate_descent, dai_yuan, diis, dogleg, "
           "fire, fire2, fletcher_reeves, fr_pr, hager_zhang, hestenes_stiefel, "
           "lbfgs, liu_storey, newton, polak_ribiere, pso, quickmin, rfo, sd, "
           "sr1, sr2, steepest";
}

namespace detail {

inline std::string canon_token(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (char ch : text) {
        unsigned char u = static_cast<unsigned char>(ch);
        if (ch == '-' || ch == ' ') {
            out.push_back('_');
        } else {
            out.push_back(static_cast<char>(std::tolower(u)));
        }
    }
    return out;
}

inline Method method_from_name(std::string const& name) {
    if (name == "lbfgs") return Method::Lbfgs;
    if (name == "bfgs") return Method::Bfgs;
    if (name == "fire") return Method::Fire;
    if (name == "fire2") return Method::Fire2;
    if (name == "quickmin") return Method::QuickMin;
    if (name == "sd" || name == "steepest" || name == "steepest_descent") return Method::Steepest;
    if (name == "cg" || name == "pr" || name == "pr+" || name == "polak_ribiere") {
        return Method::PolakRibiere;
    }
    if (name == "fletcher_reeves" || name == "fr") return Method::FletcherReeves;
    if (name == "hestenes_stiefel" || name == "hs") return Method::HestenesStiefel;
    if (name == "dai_yuan" || name == "dy") return Method::DaiYuan;
    if (name == "conjugate_descent" || name == "cd") return Method::ConjugateDescent;
    if (name == "hager_zhang" || name == "hz") return Method::HagerZhang;
    if (name == "liu_storey" || name == "ls") return Method::LiuStorey;
    if (name == "fr_pr") return Method::FrPr;
    if (name == "sr1") return Method::Sr1;
    if (name == "sr2") return Method::Sr2;
    if (name == "adam") return Method::Adam;
    if (name == "pso") return Method::Pso;
    if (name == "newton") return Method::Newton;
    if (name == "rfo") return Method::Rfo;
    if (name == "bb" || name == "barzilai_borwein") return Method::Bb;
    if (name == "dogleg") return Method::Dogleg;
    if (name == "diis") return Method::Diis;
    std::ostringstream msg;
    msg << "unknown method '" << name << "'; valid methods: " << SolverConfig::valid_methods();
    throw std::invalid_argument(msg.str());
}

inline bool is_cg(Method method) {
    switch (method) {
    case Method::PolakRibiere:
    case Method::FletcherReeves:
    case Method::HestenesStiefel:
    case Method::DaiYuan:
    case Method::ConjugateDescent:
    case Method::HagerZhang:
    case Method::LiuStorey:
    case Method::FrPr:
        return true;
    default:
        return false;
    }
}

inline bool known_option(std::string const& key) {
    static char const* keys[] = {
        "accept", "atom_maxmove", "beta", "c1", "c2", "cautious_alpha",
        "cautious_eps", "extra_updates", "fire_variant", "gtol", "istep",
        "linesearch", "linesearch_maxiter", "maxiter", "maxmove", "maxstep",
        "memory", "neb_guards", "noise_c", "noise_eps", "objective_roundoff",
        "periodic", "project_rigid", "qn_step"};
    for (char const* item : keys) {
        if (key == item) return true;
    }
    return false;
}

inline void require_known(std::string const& key) {
    if (!known_option(key)) {
        std::ostringstream msg;
        msg << "unknown option '" << key << "'; valid options: " << SolverConfig::valid_options();
        throw std::invalid_argument(msg.str());
    }
}

inline void require_finite(std::string const& key, double value) {
    if (!(value == value) || value > 1e300 || value < -1e300) {
        throw std::invalid_argument(key + " must be finite");
    }
}

}  // namespace detail

inline SolverConfig SolverConfig::build(
    std::string_view method, std::vector<std::pair<std::string, std::string>> const& text,
    std::vector<std::pair<std::string, double>> const& numbers,
    std::vector<std::pair<std::string, bool>> const& flags) {
    SolverConfig cfg;
    cfg.control_.maxiter = 100;
    cfg.control_.gtol = 1e-8;
    cfg.control_.istep = 0.1;
    cfg.control_.memory = 10;
    cfg.control_.maxmove = 0.0;
    cfg.control_.noise_eps = 0.0;
    cfg.control_.noise_c = 0.0;
    std::string method_name = detail::canon_token(method);
    std::optional<std::string> beta;
    for (auto const& item : text) {
        std::string key = detail::canon_token(item.first);
        detail::require_known(key);
        std::string value = detail::canon_token(item.second);
        if (key == "beta") {
            beta = value;
        } else if (key == "accept") {
            if (value == "none") cfg.accept_ = RGMIN_ACCEPT_NONE;
            else if (value == "energy") cfg.accept_ = RGMIN_ACCEPT_ENERGY;
            else if (value == "nonmonotone") cfg.accept_ = RGMIN_ACCEPT_NONMONOTONE;
            else if (value == "step") cfg.accept_ = RGMIN_ACCEPT_STEP;
            else {
                throw std::invalid_argument(
                    "accept must be none, energy, nonmonotone, or step");
            }
        } else if (key == "linesearch") {
            if (value == "brent") cfg.linesearch_ = RGMIN_LINESEARCH_BRENT;
            else if (value == "backtracking" || value == "armijo") {
                cfg.linesearch_ = RGMIN_LINESEARCH_BACKTRACKING;
            } else if (value == "goldstein") cfg.linesearch_ = RGMIN_LINESEARCH_GOLDSTEIN;
            else if (value == "wolfe") cfg.linesearch_ = RGMIN_LINESEARCH_WOLFE;
            else {
                throw std::invalid_argument(
                    "linesearch must be brent, backtracking, goldstein, or wolfe");
            }
        } else if (key == "fire_variant") {
            if (value == "rgmin") cfg.fire_variant_ = RGMIN_FIRE_RGMIN;
            else if (value == "guenole2020" || value == "fire2") {
                cfg.fire_variant_ = RGMIN_FIRE_GUENOLE2020;
            } else {
                throw std::invalid_argument("fire_variant must be rgmin or guenole2020");
            }
        } else if (key == "qn_step") {
            if (value == "lbfgs") cfg.qn_step_ = RGMIN_QN_LBFGS;
            else if (value == "newton") cfg.qn_step_ = RGMIN_QN_NEWTON;
            else if (value == "rfo") cfg.qn_step_ = RGMIN_QN_RFO;
            else {
                throw std::invalid_argument("qn_step must be lbfgs, newton, or rfo");
            }
        } else {
            throw std::invalid_argument(key + " must be a number or a flag");
        }
    }
    for (auto const& item : flags) {
        std::string key = detail::canon_token(item.first);
        detail::require_known(key);
        if (key == "neb_guards") cfg.neb_guards_ = item.second;
        else if (key == "periodic") cfg.periodic_ = item.second;
        else if (key == "project_rigid") cfg.project_rigid_ = item.second;
        else {
            throw std::invalid_argument(key + " must be a number or a string");
        }
    }
    for (auto const& item : numbers) {
        std::string key = detail::canon_token(item.first);
        detail::require_known(key);
        detail::require_finite(key, item.second);
        if (key == "memory") {
            if (item.second < 0.0) throw std::invalid_argument("memory must be >= 0");
            cfg.control_.memory = static_cast<std::size_t>(item.second);
        } else if (key == "maxiter") {
            if (item.second < 0.0) throw std::invalid_argument("maxiter must be >= 0");
            cfg.control_.maxiter = static_cast<std::size_t>(item.second);
        } else if (key == "linesearch_maxiter") {
            if (item.second < 1.0) throw std::invalid_argument("linesearch_maxiter must be >= 1");
            cfg.linesearch_maxiter_ = static_cast<std::size_t>(item.second);
        } else if (key == "extra_updates") {
            if (item.second < 0.0) throw std::invalid_argument("extra_updates must be >= 0");
            cfg.extra_updates_ = static_cast<std::size_t>(item.second);
        } else if (key == "gtol") cfg.control_.gtol = item.second;
        else if (key == "istep") cfg.control_.istep = item.second;
        else if (key == "maxmove") cfg.control_.maxmove = item.second;
        else if (key == "noise_eps") cfg.control_.noise_eps = item.second;
        else if (key == "noise_c") cfg.control_.noise_c = item.second;
        else if (key == "c1") cfg.c1_ = item.second;
        else if (key == "c2") cfg.c2_ = item.second;
        else if (key == "atom_maxmove" || key == "maxstep") cfg.atom_maxmove_ = item.second;
        else if (key == "cautious_eps") cfg.cautious_eps_ = item.second;
        else if (key == "cautious_alpha") cfg.cautious_alpha_ = item.second;
        else if (key == "objective_roundoff") cfg.objective_roundoff_ = item.second;
        else {
            throw std::invalid_argument(key + " must be a string or a flag");
        }
    }
    if (beta) {
        Method selected = detail::method_from_name(*beta);
        if (!detail::is_cg(selected) && *beta != "cg") {
            throw std::invalid_argument("beta must name a conjugate-gradient variant");
        }
        if (!(method_name == "cg" || detail::is_cg(detail::method_from_name(method_name)))) {
            throw std::invalid_argument("beta is only valid for a conjugate-gradient method");
        }
        method_name = *beta == "cg" ? "polak_ribiere" : *beta;
    }
    cfg.method_ = detail::method_from_name(method_name);
    cfg.method_name_ = method_name == "cg" ? "polak_ribiere" : method_name;
    if (cfg.cautious_eps_.has_value() != cfg.cautious_alpha_.has_value()) {
        throw std::invalid_argument("cautious_eps and cautious_alpha must be set together");
    }
    return cfg;
}

inline void SolverConfig::apply(Solver& solver) const {
    if (accept_) solver.set_accept(*accept_);
    if (atom_maxmove_) solver.set_atom_maxmove(*atom_maxmove_);
    if (neb_guards_) solver.set_lbfgs_neb_guards(*neb_guards_);
    if (fire_variant_) solver.set_fire_variant(*fire_variant_);
    if (linesearch_) {
        rgmin_status_t st = solver.set_linesearch(*linesearch_, c1_, c2_, linesearch_maxiter_);
        if (st != RGMIN_SUCCESS) {
            char const* msg = rgmin_last_error();
            throw std::invalid_argument(msg ? msg : "set_linesearch failed");
        }
    } else if (c1_ != 1e-4 || c2_ != 0.9 || linesearch_maxiter_ != 20) {
        rgmin_status_t st =
            solver.set_linesearch(RGMIN_LINESEARCH_WOLFE, c1_, c2_, linesearch_maxiter_);
        if (st != RGMIN_SUCCESS) {
            char const* msg = rgmin_last_error();
            throw std::invalid_argument(msg ? msg : "set_linesearch failed");
        }
    }
    if (qn_step_) solver.set_qn_step(*qn_step_);
    if (periodic_) solver.set_periodic(*periodic_);
    if (project_rigid_) solver.set_project_rigid(*project_rigid_);
    if (extra_updates_) solver.set_extra_updates(*extra_updates_);
    if (cautious_eps_ && cautious_alpha_) solver.set_cautious(*cautious_eps_, *cautious_alpha_);
    if (objective_roundoff_) {
        rgmin_status_t st = solver.set_objective_roundoff(*objective_roundoff_);
        if (st != RGMIN_SUCCESS) {
            char const* msg = rgmin_last_error();
            throw std::invalid_argument(msg ? msg : "set_objective_roundoff failed");
        }
    }
}

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
