#include "rgmin/optimize.hpp"
#include "xts/optimize.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

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
    return 0;
}
