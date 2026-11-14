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

// rgmin_minimize_scg_with_options and rgmin_solver_set_objective_roundoff
// are declared by that header. Solver::set_objective_roundoff wraps the
// second. A null options pointer on the SCG entry keeps the absolute
// infinity-norm step rule.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <atomic>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
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
    Ode12r = RGMIN_ODE12R,
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
    /// Positive: Euclidean step cap. Zero: default caps. Negative: every cap off.
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

/// \ref rgmin_lowest_eigenpairs_ex. `precond` null is the identity.
/// `pairs` is length k and carries residual and converged. ABI minor 42.
inline void lowest_eigenpairs_ex(rgmin_hvp_fn hvp, void* user, rgmin_precond_fn precond,
                                 void* precond_user, double const* x, double const* seed,
                                 double* modes, double* values, std::size_t n, std::size_t k,
                                 rgmin_eigen_params_t const* params,
                                 rgmin_eigen_options_t const* options,
                                 rgmin_eigenpair_ex_t* pairs) {
    if (hvp == nullptr || x == nullptr || seed == nullptr || modes == nullptr ||
        values == nullptr || pairs == nullptr || k == 0 || n == 0 ||
        n > static_cast<std::size_t>(-1) / k) {
        throw std::invalid_argument("rgmin_lowest_eigenpairs_ex: bad block");
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
        throw std::invalid_argument("rgmin_lowest_eigenpairs_ex: null buffer");
    }
    rgmin_status_t status = rgmin_lowest_eigenpairs_ex(
        hvp, user, precond, precond_user, xt, st, mt, vt, params, options, pairs, k);
    rgmin_tensor_free(xt);
    rgmin_tensor_free(st);
    rgmin_tensor_free(mt);
    rgmin_tensor_free(vt);
    if (status != RGMIN_SUCCESS) {
        char const* msg = rgmin_last_error();
        throw std::invalid_argument(msg != nullptr ? msg
                                                   : "rgmin_lowest_eigenpairs_ex failed");
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
/// Mutable. Python bindings claim busy() for the duration of a step.
/// A second caller gets "Solver is in use by another thread".
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

    /// Busy flag. Bindings claim it for the duration of a step. The flag starts clear.
    std::atomic<bool>& busy() noexcept { return busy_; }
    std::atomic<bool> const& busy() const noexcept { return busy_; }
    std::size_t dim() const noexcept { return dim_; }
    Method method() const noexcept { return method_; }
    rgmin_solver_t* raw() noexcept { return ptr_; }

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

    void forget_evaluation() { rgmin_solver_forget_evaluation(ptr_); }
    rgmin_status_t set_precon(Precon const& p) { return rgmin_solver_set_precon(ptr_, p.get()); }
    void clear_precon() { rgmin_solver_clear_precon(ptr_); }
    double precon_scale() const { return rgmin_solver_precon_scale(ptr_); }
    void set_fire_variant(rgmin_fire_variant_t variant) {
        rgmin_solver_set_fire_variant(ptr_, variant);
    }
    rgmin_status_t set_quickmin_cell(std::size_t at) {
        return rgmin_solver_set_quickmin_cell(ptr_, at);
    }
    void set_quickmin_variant(rgmin_quickmin_variant_t variant) {
        rgmin_solver_set_quickmin_variant(ptr_, variant);
    }
    rgmin_status_t set_linesearch(rgmin_linesearch_kind_t kind, double c1, double c2,
                                 std::size_t maxiter) {
        return rgmin_solver_set_linesearch(ptr_, kind, c1, c2, maxiter);
    }
    rgmin_status_t set_objective_roundoff(double relative) {
        return rgmin_solver_set_objective_roundoff(ptr_, relative);
    }
    void set_geometry_step(bool on) { rgmin_solver_set_geometry_step(ptr_, on ? 1 : 0); }
    void set_lbfgs_neb_guards(bool on) { rgmin_solver_set_lbfgs_neb_guards(ptr_, on ? 1 : 0); }
    void set_rigid_mode(int32_t mode) { rgmin_solver_set_rigid_mode(ptr_, mode); }
    void set_rigid_mass_metric(bool on) { rgmin_solver_set_rigid_mass_metric(ptr_, on ? 1 : 0); }
    void set_rigid_mass_weighted(bool on) { rgmin_solver_set_rigid_mass_weighted(ptr_, on ? 1 : 0); }
    void set_pbc_axes(bool x, bool y, bool z) {
        rgmin_solver_set_pbc_axes(ptr_, x ? 1 : 0, y ? 1 : 0, z ? 1 : 0);
    }
    void set_frozen(std::uint8_t const* frozen, std::size_t len) {
        rgmin_solver_set_frozen(ptr_, frozen, len);
    }
    void set_broken_axes(bool tx, bool ty, bool tz, bool rx, bool ry, bool rz) {
        rgmin_solver_set_broken_axes(ptr_, tx ? 1 : 0, ty ? 1 : 0, tz ? 1 : 0, rx ? 1 : 0,
                                     ry ? 1 : 0, rz ? 1 : 0);
    }
    int set_highs_trust(double radius) { return rgmin_solver_set_highs_trust(ptr_, radius); }
    void set_factor_shape(std::size_t n, std::size_t p) { rgmin_solver_set_factor_shape(ptr_, n, p); }
    void set_unitary(std::size_t n) { rgmin_solver_set_unitary(ptr_, n); }
    void set_masses(double const* masses, std::size_t n_atoms) {
        rgmin_solver_set_masses(ptr_, masses, n_atoms);
    }
    void set_periodic(bool periodic) { rgmin_solver_set_periodic(ptr_, periodic ? 1 : 0); }
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
        return Report{out.value, out.steps, out.grad_norm, out.stop};
    }

    Report step_fg(rgmin_evalgrad_fn evalgrad, void* user,
                   DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_fg(ptr_, evalgrad, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_solver_step_fg failed");
        }
        return Report{out.value, out.steps, out.grad_norm, out.stop};
    }

    Report step_hvp(rgmin_eval_fn eval, rgmin_grad_fn grad, rgmin_hvp_fn hvp,
                    void* user, DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_hvp(ptr_, eval, grad, hvp, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_solver_step_hvp failed");
        }
        return Report{out.value, out.steps, out.grad_norm, out.stop};
    }

    Report step_hess(rgmin_eval_fn eval, rgmin_grad_fn grad, rgmin_hess_fn hess,
                     void* user, DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_hess(ptr_, eval, grad, hess, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_solver_step_hess failed");
        }
        return Report{out.value, out.steps, out.grad_norm, out.stop};
    }

    Report step_hess_fg(rgmin_evalgrad_fn evalgrad, rgmin_hess_fn hess,
                        void* user, DLManagedTensorVersioned* x) {
        rgmin_report_t out{};
        rgmin_status_t st =
            rgmin_solver_step_hess_fg(ptr_, evalgrad, hess, user, x, &out);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_solver_step_hess_fg failed");
        }
        return Report{out.value, out.steps, out.grad_norm, out.stop};
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
    /// push() with the caller's `rho` in place of `1 / (s . y)`. False
    /// when the curvature gate declines the pair or `rho` is not finite
    /// and positive. A bad length throws. ABI minor 43.
    bool push_rho(double const* s, double const* y, double rho, std::size_t n) {
        int32_t stored = 0;
        rgmin_status_t st = rgmin_lbfgs_pairs_push_rho(ptr_, s, y, rho, n, &stored);
        if (st != RGMIN_SUCCESS) {
            detail::raise(st, "rgmin_lbfgs_pairs_push_rho failed");
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
                              std::vector<std::pair<std::string, long long>> const& integers,
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
    std::optional<rgmin_linesearch_kind_t> linesearch_;
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
    if (name == "ode12r") return Method::Ode12r;
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
    if (name == "newton_cg" || name == "newton-cg") return Method::NewtonCg;
    if (name == "newton_pcg" || name == "newton-pcg") return Method::NewtonPcg;
    if (name == "eon_cg" || name == "eon-cg") return Method::EonCg;
    if (name == "prp" || name == "polak_ribiere_plus" || name == "polak-ribiere-plus") {
        return Method::PolakRibierePlus;
    }
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
    case Method::PolakRibierePlus:
    case Method::EonCg:
        return true;
    default:
        return false;
    }
}

inline bool known_option(std::string const& key) {
    static constexpr char const* const keys[] = {
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
    if (!std::isfinite(value)) {
        throw std::invalid_argument(key + " must be finite");
    }
}

inline bool is_count_option(std::string const& key) {
    return key == "memory" || key == "maxiter" || key == "linesearch_maxiter" ||
           key == "extra_updates";
}

inline std::size_t require_count(std::string const& key, long long value, long long min_value) {
    if (value < min_value) {
        throw std::invalid_argument(key + " must be >= " + std::to_string(min_value));
    }
    if (static_cast<unsigned long long>(value) >
        static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(key + " is out of range");
    }
    return static_cast<std::size_t>(value);
}

}  // namespace detail

inline SolverConfig SolverConfig::build(
    std::string_view method, std::vector<std::pair<std::string, std::string>> const& text,
    std::vector<std::pair<std::string, double>> const& numbers,
    std::vector<std::pair<std::string, long long>> const& integers,
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
        else if (detail::is_count_option(key)) {
            throw std::invalid_argument(key + " must be an int");
        } else {
            throw std::invalid_argument(key + " must be a number or a string");
        }
    }
    for (auto const& item : integers) {
        std::string key = detail::canon_token(item.first);
        detail::require_known(key);
        if (key == "memory") {
            cfg.control_.memory = detail::require_count(key, item.second, 0);
        } else if (key == "maxiter") {
            cfg.control_.maxiter = detail::require_count(key, item.second, 0);
        } else if (key == "linesearch_maxiter") {
            cfg.linesearch_maxiter_ = detail::require_count(key, item.second, 1);
        } else if (key == "extra_updates") {
            cfg.extra_updates_ = detail::require_count(key, item.second, 0);
        } else if (key == "gtol" || key == "istep" || key == "maxmove" || key == "noise_eps" ||
                   key == "noise_c" || key == "c1" || key == "c2" || key == "atom_maxmove" ||
                   key == "maxstep" || key == "cautious_eps" || key == "cautious_alpha" ||
                   key == "objective_roundoff") {
            double value = static_cast<double>(item.second);
            detail::require_finite(key, value);
            if (key == "gtol") cfg.control_.gtol = value;
            else if (key == "istep") cfg.control_.istep = value;
            else if (key == "maxmove") cfg.control_.maxmove = value;
            else if (key == "noise_eps") cfg.control_.noise_eps = value;
            else if (key == "noise_c") cfg.control_.noise_c = value;
            else if (key == "c1") cfg.c1_ = value;
            else if (key == "c2") cfg.c2_ = value;
            else if (key == "atom_maxmove" || key == "maxstep") cfg.atom_maxmove_ = value;
            else if (key == "cautious_eps") cfg.cautious_eps_ = value;
            else if (key == "cautious_alpha") cfg.cautious_alpha_ = value;
            else cfg.objective_roundoff_ = value;
        } else {
            throw std::invalid_argument(key + " must be a string or a flag");
        }
    }
    for (auto const& item : numbers) {
        std::string key = detail::canon_token(item.first);
        detail::require_known(key);
        detail::require_finite(key, item.second);
        if (detail::is_count_option(key)) {
            throw std::invalid_argument(key + " must be an int");
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
    // c1, c2, and linesearch_maxiter apply only with an explicit linesearch.
    if (!cfg.linesearch_.has_value() &&
        (cfg.c1_ != 1e-4 || cfg.c2_ != 0.9 || cfg.linesearch_maxiter_ != 20)) {
        throw std::invalid_argument(
            "c1, c2, and linesearch_maxiter require linesearch "
            "(brent, backtracking, goldstein, or wolfe)");
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

}  // namespace rgmin
