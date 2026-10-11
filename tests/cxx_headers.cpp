#include "rgmin/optimize.hpp"
#include "xts/optimize.hpp"
#include "xts.h"

#include <cmath>
#include <cstddef>
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
    solver.set_accept(RGMIN_ACCEPT_STEP);
    solver.set_fire_variant(RGMIN_FIRE_GUENOLE2020);
    if (solver.set_linesearch(RGMIN_LINESEARCH_WOLFE, 1e-4, 0.9, 20) != RGMIN_SUCCESS) {
        return 3;
    }
    double s[2] = {1.0, 0.0};
    double y[2] = {2.0, 0.0};
    if (solver.push_pair(s, y, 2) != 0 || solver.pair_count() != 1) {
        return 4;
    }
    solver.rebase();
    double gradient[2] = {2.0, -4.0};
    double direction[2] = {};
    if (solver.search_direction(gradient, direction, 2) != RGMIN_SUCCESS ||
        std::abs(direction[0] + 1.0) > 1e-12 ||
        std::abs(direction[1] - 2.0) > 1e-12) {
        return 5;
    }
    if (ctrl.to_control().maxmove != ctrl.maxmove) {
        return 6;
    }
    auto stamp = xts_abi_stamp();
    if (stamp.abi_minor != RGMIN_ABI_VERSION_MINOR ||
        stamp.layout_revision != RGMIN_ABI_LAYOUT_REVISION) {
        return 7;
    }
    std::printf("optimize nit=%zu grad=%g pairs=%zu\n", r.nit,
                r.grad_norm, solver.pair_count());
    rgmin_tensor_free(t);

    std::vector<int32_t> z{1, 1};
    auto precon = rgmin::Precon::lindh(z).cell(10.0, 10.0, 10.0);
    rgmin::Solver geom(rgmin::Method::Lbfgs, rgmin::Control{}, 6);
    if (geom.set_precon(precon) != RGMIN_SUCCESS) {
        return 8;
    }
    if (sizeof(rgmin_precon_t) != precon.get()->struct_size) {
        std::fprintf(stderr, "precon struct %zu rust %u\n", sizeof(rgmin_precon_t),
                     precon.get()->struct_size);
        return 12;
    }
    if (offsetof(rgmin_bond_t, i) != 0 || offsetof(rgmin_bond_t, j) != 4 ||
        offsetof(rgmin_bond_t, k) != 8 || sizeof(rgmin_bond_t) != 16) {
        return 13;
    }
    if (offsetof(rgmin_angle_t, i) != 0 || offsetof(rgmin_angle_t, vertex) != 4 ||
        offsetof(rgmin_angle_t, k) != 8 || offsetof(rgmin_angle_t, k_theta) != 16 ||
        sizeof(rgmin_angle_t) != 24) {
        return 14;
    }
    if (offsetof(rgmin_precon_t, struct_size) != 0 || offsetof(rgmin_precon_t, kind) != 4 ||
        offsetof(rgmin_precon_t, c_stab) != 8 || offsetof(rgmin_precon_t, mu) != 16 ||
        offsetof(rgmin_precon_t, r_cut) != 24 || offsetof(rgmin_precon_t, cell) != 32 ||
        offsetof(rgmin_precon_t, numbers) != 56 || offsetof(rgmin_precon_t, natoms) != 64 ||
        offsetof(rgmin_precon_t, nbr_i) != 96 || offsetof(rgmin_precon_t, exp_a) != 152 ||
        offsetof(rgmin_precon_t, bonds) != 160 || offsetof(rgmin_precon_t, angles) != 176 ||
        offsetof(rgmin_precon_t, dihedrals) != 192 ||
        offsetof(rgmin_precon_t, geometry_kind) != 196 || sizeof(rgmin_precon_t) != 200) {
        std::fprintf(stderr, "precon layout kind=%zu size=%zu\n", offsetof(rgmin_precon_t, kind),
                     sizeof(rgmin_precon_t));
        return 15;
    }
    if (!std::isnan(geom.precon_scale())) {
        return 9;
    }
    rgmin_bond_t bad{9, 0, 1.0};
    auto ff = rgmin::Precon::ff({bad});
    if (geom.set_precon(ff) != RGMIN_INVALID_PARAMETER) {
        return 10;
    }
    geom.clear_precon();
    if (!std::isnan(geom.precon_scale())) {
        return 11;
    }
    return 0;
}
