#include "rgmin/optimize.hpp"
#include "xts.h"
#include "xts/optimize.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" int32_t scale_h0(void* user, double const* v, double* out, std::size_t n) {
    double const gamma = *static_cast<double const*>(user);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = gamma * v[i];
    }
    return 0;
}

namespace {

rgmin_status_t quad_eval(void* /*user*/, DLManagedTensorVersioned const* x, double* value) {
    auto const* p = reinterpret_cast<double const*>(
        static_cast<char const*>(x->dl_tensor.data) + x->dl_tensor.byte_offset);
    *value = p[0] * p[0] + p[1] * p[1];
    return RGMIN_SUCCESS;
}

rgmin_status_t quad_grad(void* /*user*/, DLManagedTensorVersioned const* x,
                         DLManagedTensorVersioned* g) {
    auto const* p = reinterpret_cast<double const*>(
        static_cast<char const*>(x->dl_tensor.data) + x->dl_tensor.byte_offset);
    auto* gp = reinterpret_cast<double*>(
        static_cast<char*>(g->dl_tensor.data) + g->dl_tensor.byte_offset);
    gp[0] = 2.0 * p[0];
    gp[1] = 2.0 * p[1];
    return RGMIN_SUCCESS;
}

double const* data(DLManagedTensorVersioned const* t) {
    return reinterpret_cast<double const*>(static_cast<char const*>(t->dl_tensor.data) +
                                           t->dl_tensor.byte_offset);
}

double* data(DLManagedTensorVersioned* t) {
    return reinterpret_cast<double*>(static_cast<char*>(t->dl_tensor.data) +
                                     t->dl_tensor.byte_offset);
}

// f = sum_k (x_k - t_k)^2 over two atoms, tethers t in user.
rgmin_status_t tether_eval(void* user, DLManagedTensorVersioned const* x, double* value) {
    auto const* t = static_cast<double const*>(user);
    double const* p = data(x);
    double acc = 0.0;
    for (int k = 0; k < 6; ++k) {
        acc += (p[k] - t[k]) * (p[k] - t[k]);
    }
    *value = acc;
    return RGMIN_SUCCESS;
}

rgmin_status_t tether_grad(void* user, DLManagedTensorVersioned const* x,
                           DLManagedTensorVersioned* g) {
    auto const* t = static_cast<double const*>(user);
    double const* p = data(x);
    double* gp = data(g);
    for (int k = 0; k < 6; ++k) {
        gp[k] = 2.0 * (p[k] - t[k]);
    }
    return RGMIN_SUCCESS;
}

// Bond of length 1.2 between atoms of mass 1 and 7 tethered at t: the
// minimum is the midpoint of t plus or minus 0.6 along t_0 - t_1.
int constrained() {
    double tethers[6] = {0.3, -0.1, 0.2, -0.1, 0.2, -0.05};
    double const c = std::cos(0.5), s = std::sin(0.5);
    std::vector<double> start{0.5 + 0.6 * c, 0.4 + 0.6 * s, 0.1,
                              0.5 - 0.6 * c, 0.4 - 0.6 * s, 0.1};
    std::uint32_t const pi[1] = {0};
    std::uint32_t const pj[1] = {1};
    double const masses[2] = {1.0, 7.0};
    auto set = xts::optimize::DistanceConstraints::at(pi, pj, 1, start.data(), 6);
    set.with_masses(masses, 2).with_solver(rgmin::ConstraintSolver::Shake);
    double ab[3], dist = 0.0;
    for (int k = 0; k < 3; ++k) {
        ab[k] = tethers[k] - tethers[3 + k];
        dist += ab[k] * ab[k];
    }
    dist = std::sqrt(dist);
    rgmin::Control ctrl;
    ctrl.maxiter = 500;
    ctrl.gtol = std::cbrt(2.220446049250313e-16);
    ctrl.maxmove = 0.2;
    rgmin::Method const methods[] = {rgmin::Method::Lbfgs, rgmin::Method::Fire2,
                                     rgmin::Method::NewtonCg};
    for (rgmin::Method method : methods) {
        rgmin::Solver solver(method, ctrl, 6);
        solver.set_accept(RGMIN_ACCEPT_ENERGY);
        solver.set_distance_constraints(set);
        std::vector<double> x = start;
        rgmin::Report r{};
        for (std::size_t it = 0; it < ctrl.maxiter && !(r.grad_norm < ctrl.gtol && it > 0);
             ++it) {
            DLManagedTensorVersioned* t = rgmin::borrow_cpu_f64(x.data(), 6);
            r = solver.step(tether_eval, tether_grad, tethers, t);
            rgmin_tensor_free(t);
            if (!(set.excess(x.data(), 6) <= 1.0)) {
                std::fprintf(stderr, "method %d left the set\n", static_cast<int>(method));
                return 10;
            }
        }
        double off = 0.0;
        for (int k = 0; k < 6; ++k) {
            double const mid = 0.5 * (tethers[k % 3] + tethers[3 + k % 3]);
            double const want = mid + (k < 3 ? 0.6 : -0.6) * ab[k % 3] / dist;
            off += (x[k] - want) * (x[k] - want);
        }
        // k = 2 and k_min = min(k, k D / L) bound |x - x*| by
        // sqrt(2) |g| / k_min; forming x and x - x* rounds each
        // coordinate at most 9.5 times at the largest |x|.
        double const k_min = std::fmin(2.0, 2.0 * dist / 1.2);
        double scale = 0.0;
        for (double v : x) {
            scale = std::fmax(scale, std::fabs(v));
        }
        double const m = 5.0 * 2.220446049250313e-16;
        double const tol =
            std::sqrt(2.0) * r.grad_norm / k_min + std::sqrt(6.0) * m / (1.0 - m) * scale;
        if (!(r.grad_norm < ctrl.gtol) || std::sqrt(off) > tol) {
            std::fprintf(stderr, "method %d: grad=%g off=%g\n", static_cast<int>(method),
                         r.grad_norm, std::sqrt(off));
            return 11;
        }
    }

    std::uint32_t const same[1] = {0};
    double const one[1] = {1.0};
    try {
        rgmin::DistanceConstraints bad(same, same, one, 1);
        return 12;
    } catch (rgmin::ConstraintError const& e) {
        if (e.kind() != RGMIN_CONSTRAINT_MALFORMED) {
            return 13;
        }
    }
    double const negative[2] = {1.0, -1.0};
    try {
        set.with_masses(negative, 2);
        return 14;
    } catch (xts::optimize::ConstraintError const& e) {
        if (e.kind() != XTS_CONSTRAINT_MASS) {
            return 15;
        }
    }
    // A sideways pull of 3 per atom under a unit step: the old bond
    // cannot close the gap, so the step throws and leaves x.
    double const flat[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    auto bond = rgmin::DistanceConstraints::at(pi, pj, 1, flat, 6);
    double pull[6] = {0.0, 3.0, 0.0, 0.0, -3.0, 0.0};
    rgmin::Control unit;
    unit.istep = 1.0;
    rgmin::Solver steepest(rgmin::Method::Steepest, unit, 6);
    steepest.set_accept(RGMIN_ACCEPT_NONE);
    steepest.set_distance_constraints(bond);
    std::vector<double> y(flat, flat + 6);
    for (int k = 0; k < 6; ++k) {
        pull[k] = flat[k] + pull[k] / 2.0;
    }
    DLManagedTensorVersioned* ty = rgmin::borrow_cpu_f64(y.data(), 6);
    try {
        steepest.step(tether_eval, tether_grad, pull, ty);
        rgmin_tensor_free(ty);
        return 16;
    } catch (rgmin::ConstraintError const& e) {
        rgmin_tensor_free(ty);
        if (e.kind() != RGMIN_CONSTRAINT_REVERSED && e.kind() != RGMIN_CONSTRAINT_NOT_CONVERGED) {
            return 17;
        }
    }
    for (int k = 0; k < 6; ++k) {
        if (y[k] != flat[k]) {
            return 18;
        }
    }
    std::printf("constraints ok\n");
    return 0;
}

rgmin_status_t diag3_hvp(void*, DLManagedTensorVersioned const*,
                         DLManagedTensorVersioned const* v, DLManagedTensorVersioned* hv) {
    double const lam[3] = {-3.0, 1.0, 4.0};
    double const* vp = data(v);
    double* hp = data(hv);
    auto const n = v->dl_tensor.shape[0];
    for (int64_t i = 0; i < n && i < 3; ++i) {
        hp[i] = lam[i] * vp[i];
    }
    return RGMIN_SUCCESS;
}

int eigen_block() {
    double x[3] = {};
    double seed[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    double modes[6] = {};
    double values[2] = {};
    rgmin_eigenpair_t pairs[2] = {};
    try {
        xts::optimize::lowest_eigenpairs(diag3_hvp, nullptr, x, seed, modes, values, 3, 2,
                                         nullptr, nullptr, pairs);
    } catch (std::invalid_argument const& err) {
        std::fprintf(stderr, "eigenpairs: %s\n", err.what());
        return 6;
    }
    if (std::fabs(values[0] + 3.0) > 1e-6 || std::fabs(values[1] - 1.0) > 1e-6 ||
        pairs[0].actions == 0) {
        std::fprintf(stderr, "eigenpairs values %g %g\n", values[0], values[1]);
        return 7;
    }
    rgmin_eigenpair_ex_t extended[2] = {};
    double modes_ex[6] = {};
    double values_ex[2] = {};
    try {
        xts::optimize::lowest_eigenpairs_ex(diag3_hvp, nullptr, nullptr, nullptr, x, seed,
                                            modes_ex, values_ex, 3, 2, nullptr, nullptr, extended);
    } catch (std::invalid_argument const& err) {
        std::fprintf(stderr, "eigenpairs ex: %s\n", err.what());
        return 10;
    }
    if (extended[0].converged != 1 || extended[1].converged != 1 || extended[0].residual > 1e-6 ||
        std::fabs(values_ex[0] - values[0]) > 0.0 || std::fabs(modes_ex[0] - modes[0]) > 0.0) {
        std::fprintf(stderr, "extended pair residual %g converged %d\n", extended[0].residual,
                     extended[0].converged);
        return 11;
    }
    return 0;
}

}  // namespace

int main() {
    std::vector<double> x{1.0, 1.0};
    DLManagedTensorVersioned* t = rgmin::borrow_cpu_f64(x.data(), 2);
    if (t == nullptr) {
        std::fprintf(stderr, "borrow_cpu_f64 returned null\n");
        return 1;
    }
    rgmin::OptimizeControl ctrl;
    ctrl.max_iterations = 80;
    ctrl.gtol = 1e-10;
    rgmin::OptimizeResult r =
        rgmin::optimize(quad_eval, quad_grad, nullptr, t, ctrl, rgmin::Method::Lbfgs);
    if (r.grad_norm > 1e-6 || std::hypot(x[0], x[1]) > 1e-5) {
        std::fprintf(stderr, "optimize stalled: grad=%g x=(%g,%g)\n", r.grad_norm, x[0],
                     x[1]);
        return 2;
    }
    xts::optimize::Solver solver(xts::optimize::Method::Lbfgs, xts::optimize::Control{}, 2);
    double lo[2] = {-1.0, -2.0};
    double hi[2] = {1.0, 2.0};
    int st = solver.set_box(lo, hi, 2);
    int st_ipm = solver.set_highs_solver(RGMIN_HIGHS_IPM);
    int st_x = solver.set_highs_crossover(RGMIN_HIGHS_CROSSOVER_OFF);
    std::printf("optimize nit=%zu grad=%g set_box=%d ipm=%d xover=%d\n", r.nit,
                r.grad_norm, st, st_ipm, st_x);

    // Gradient (0.6, 0.8): two-norm 1, infinity norm 0.8. At gtol 0.9 the
    // infinity-norm session stops at the start.
    rgmin::Control stop_ctrl;
    stop_ctrl.gtol = 0.9;
    rgmin::Solver stopper(rgmin::Method::Fire, stop_ctrl, 2);
    double g[2] = {0.0, 0.0};
    if (stopper.report_gradient(g, 2) == RGMIN_SUCCESS ||
        stopper.set_stop_norm(RGMIN_STOP_NORM_INF) != 0) {
        std::fprintf(stderr, "stop norm setup failed\n");
        return 3;
    }
    std::vector<double> y{0.3, 0.4};
    DLManagedTensorVersioned* ty = rgmin::borrow_cpu_f64(y.data(), 2);
    rgmin::Report sr = stopper.step(quad_eval, quad_grad, nullptr, ty);
    rgmin_tensor_free(ty);
    if (stopper.report_gradient(g, 2) != RGMIN_SUCCESS || sr.steps != 0 ||
        std::fabs(sr.grad_norm - 0.8) > 1e-15 || std::fabs(g[0] - 0.6) > 1e-15 ||
        std::fabs(g[1] - 0.8) > 1e-15 || y[0] != 0.3 || y[1] != 0.4) {
        std::fprintf(stderr, "inf stop: steps=%zu grad=%g g=(%g,%g)\n", sr.steps,
                     sr.grad_norm, g[0], g[1]);
        return 4;
    }
    rgmin::Solver fire2(rgmin::Method::Fire2, stop_ctrl, 2);
    if (stopper.set_fire_abc(true) != 1 ||
        fire2.set_fire_preset(RGMIN_FIRE_PRESET_GUENOLE2020) != 0 ||
        fire2.set_fire_abc(true) != 0) {
        std::fprintf(stderr, "fire preset setup failed\n");
        return 5;
    }
    static_assert(static_cast<int>(rgmin::Method::PolakRibierePlus) == RGMIN_POLAK_RIBIERE_PLUS,
                  "PolakRibierePlus keeps the C enumerator");
    static_assert(static_cast<int>(rgmin::Method::EonCg) == RGMIN_EON_CG,
                  "EonCg keeps the C enumerator");
    if (rgmin::method_from_name("eon-cg") != rgmin::Method::EonCg ||
        rgmin::method_from_name("polak-ribiere-plus") != rgmin::Method::PolakRibierePlus) {
        std::fprintf(stderr, "method_from_name missed a minor-35 method\n");
        return 6;
    }
    bool refused = false;
    try {
        rgmin::LbfgsPairs empty(0, 4);
        (void)empty;
    } catch (std::runtime_error const&) {
        refused = true;
    }
    if (!refused) {
        std::fprintf(stderr, "a zero dimension must be refused\n");
        return 7;
    }
    rgmin::LbfgsPairs pairs(2, 4);
    double grad[2] = {1.0, -2.0};
    double diag[2] = {0.5, 0.25};
    double dir[2] = {0.0, 0.0};
    pairs.direction_diagonal(grad, diag, dir, 2);
    if (std::fabs(dir[0] + 0.5) > 1e-15 || std::fabs(dir[1] - 0.5) > 1e-15) {
        std::fprintf(stderr, "empty H0 direction (%g, %g)\n", dir[0], dir[1]);
        return 8;
    }
    double gamma = 0.5;
    double via_callback[2] = {0.0, 0.0};
    pairs.direction_apply(grad, scale_h0, &gamma, via_callback, 2);
    if (std::fabs(via_callback[0] + 0.5) > 1e-15 || std::fabs(via_callback[1] - 1.0) > 1e-15) {
        std::fprintf(stderr, "callback H0 direction (%g, %g)\n", via_callback[0], via_callback[1]);
        return 9;
    }
    int block = eigen_block();
    if (block != 0) {
        return block;
    }
    return constrained();
}
