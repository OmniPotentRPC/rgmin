/* Rigid quotient, Steihaug-Toint tCG, radius update,
 * and tangent-projected eigenpairs. argv[1] is sizeof(rgmin_rtr_tcg_result_t)
 * as Rust reports it. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "xts.h"

static int close_to(double got, double want) {
    return fabs(got - want) < 1e-9;
}

static rgmin_status_t scale_hvp(void *user, const DLManagedTensorVersioned *x,
                                 const DLManagedTensorVersioned *v, DLManagedTensorVersioned *hv) {
    const double scale = *(const double *)user;
    const double *vs = (const double *)v->dl_tensor.data;
    double *hs = (double *)hv->dl_tensor.data;
    size_t n = (size_t)v->dl_tensor.shape[0];
    size_t i;
    (void)x;
    for (i = 0; i < n; i++) {
        hs[i] = scale * vs[i];
    }
    return RGMIN_SUCCESS;
}

static rgmin_status_t stretch_hvp(void *user, const DLManagedTensorVersioned *x,
                                   const DLManagedTensorVersioned *v,
                                   DLManagedTensorVersioned *hv) {
    const double *vs = (const double *)v->dl_tensor.data;
    double *hs = (double *)hv->dl_tensor.data;
    const double u[6] = {-0.7071067811865476, 0.0, 0.0, 0.7071067811865476, 0.0, 0.0};
    const double t[6] = {0.7071067811865476, 0.0, 0.0, 0.7071067811865476, 0.0, 0.0};
    double du = 0.0;
    double dt = 0.0;
    size_t i;
    (void)user;
    (void)x;
    for (i = 0; i < 6; i++) {
        du += vs[i] * u[i];
        dt += vs[i] * t[i];
    }
    for (i = 0; i < 6; i++) {
        hs[i] = -2.0 * du * u[i] - 100.0 * dt * t[i];
    }
    return RGMIN_SUCCESS;
}

static int check_quadratics(void) {
    double scale = 2.0;
    double x = 0.0;
    double grad = 2.0;
    double eta = 0.0;
    rgmin_rtr_tcg_result_t out;
    rgmin_status_t st = rgmin_rtr_tcg_solve(NULL, scale_hvp, &scale, &x, &grad, 10.0, 1.0, 0.1, 5,
                                            &eta, 1, &out);
    if (st != RGMIN_SUCCESS || out.stop != RGMIN_TCG_RESIDUAL || out.iterations != 1 ||
        !close_to(eta, -1.0) || !close_to(out.model_decrease, 1.0)) {
        fprintf(stderr, "residual stop eta %g decrease %g stop %d\n", eta, out.model_decrease,
                out.stop);
        return 10;
    }
    scale = 4.0;
    grad = 2.0;
    st = rgmin_rtr_tcg_solve(NULL, scale_hvp, &scale, &x, &grad, 0.25, 1.0, 0.1, 5, &eta, 1, &out);
    if (st != RGMIN_SUCCESS || out.stop != RGMIN_TCG_BOUNDARY || !close_to(eta, -0.25) ||
        !close_to(out.model_decrease, 0.375)) {
        fprintf(stderr, "boundary stop eta %g decrease %g stop %d\n", eta, out.model_decrease,
                out.stop);
        return 11;
    }
    scale = -2.0;
    grad = 1.0;
    st = xts_rtr_tcg_solve(NULL, scale_hvp, &scale, &x, &grad, 0.4, 1.0, 0.1, 5, &eta, 1, &out);
    if (st != RGMIN_SUCCESS || out.stop != XTS_TCG_NEGATIVE_CURVATURE || !close_to(eta, -0.4) ||
        !close_to(out.model_decrease, 0.56)) {
        fprintf(stderr, "curvature stop eta %g decrease %g stop %d\n", eta, out.model_decrease,
                out.stop);
        return 12;
    }
    grad = 0.0;
    st = rgmin_rtr_tcg_solve(NULL, scale_hvp, &scale, &x, &grad, 1.0, 1.0, 0.1, 4, &eta, 1, &out);
    if (st != RGMIN_SUCCESS || out.stop != RGMIN_TCG_ZERO_GRADIENT || out.iterations != 0 ||
        !close_to(eta, 0.0)) {
        fprintf(stderr, "zero gradient stop %d eta %g\n", out.stop, eta);
        return 13;
    }
    return 0;
}

static int check_radius(void) {
    double radius = 1.0;
    int32_t accepted = -1;
    if (rgmin_rtr_radius_update(&radius, 8.0, 0.1, 1.0, 0.5, &accepted) != RGMIN_SUCCESS ||
        accepted != 0 || !close_to(radius, 0.25)) {
        fprintf(stderr, "shrink radius %g accepted %d\n", radius, accepted);
        return 20;
    }
    if (rgmin_rtr_radius_update(&radius, 8.0, 0.9, 1.0, 0.25, &accepted) != RGMIN_SUCCESS ||
        accepted != 1 || !close_to(radius, 0.5)) {
        fprintf(stderr, "grow radius %g accepted %d\n", radius, accepted);
        return 21;
    }
    if (rgmin_rtr_radius_update(&radius, 8.0, 0.5, 1.0, 0.1, &accepted) != RGMIN_SUCCESS ||
        accepted != 1 || !close_to(radius, 0.5)) {
        fprintf(stderr, "hold radius %g\n", radius);
        return 22;
    }
    if (xts_rtr_radius_update(&radius, 8.0, 1.0, 0.0, 0.1, &accepted) != RGMIN_SUCCESS ||
        accepted != 0 || !close_to(radius, 0.125)) {
        fprintf(stderr, "reject radius %g accepted %d\n", radius, accepted);
        return 23;
    }
    radius = 3.0;
    if (rgmin_rtr_radius_update(&radius, 4.0, 0.9, 1.0, 3.0, &accepted) != RGMIN_SUCCESS ||
        accepted != 1 || !close_to(radius, 4.0)) {
        fprintf(stderr, "cap radius %g\n", radius);
        return 24;
    }
    return 0;
}

static int check_radius_above_maximum(void) {
    double radius = 8.0;
    int32_t accepted = -1;
    if (rgmin_rtr_radius_update(&radius, 1.0, 0.5, 1.0, 0.0, &accepted) != RGMIN_SUCCESS ||
        accepted != 1 || !close_to(radius, 1.0)) {
        fprintf(stderr, "clamp radius %g accepted %d\n", radius, accepted);
        return 25;
    }
    radius = 8.0;
    if (rgmin_rtr_radius_update(&radius, 1.0, 0.1, 1.0, 0.0, &accepted) != RGMIN_SUCCESS ||
        accepted != 0 || !close_to(radius, 0.25)) {
        fprintf(stderr, "clamp then shrink radius %g accepted %d\n", radius, accepted);
        return 26;
    }
    return 0;
}

typedef struct counted_hvp_state {
    size_t calls;
    double value;
} counted_hvp_state;

static rgmin_status_t counted_hvp(void *user, const DLManagedTensorVersioned *x,
                                  const DLManagedTensorVersioned *v,
                                  DLManagedTensorVersioned *hv) {
    counted_hvp_state *state = (counted_hvp_state *)user;
    const double *vs = (const double *)v->dl_tensor.data;
    double *hs = (double *)hv->dl_tensor.data;
    size_t n = (size_t)v->dl_tensor.shape[0];
    size_t i;
    (void)x;
    state->calls++;
    for (i = 0; i < n; i++) {
        hs[i] = state->value * vs[i];
    }
    return RGMIN_SUCCESS;
}

static rgmin_status_t failing_hvp(void *user, const DLManagedTensorVersioned *x,
                                  const DLManagedTensorVersioned *v,
                                  DLManagedTensorVersioned *hv) {
    (void)x;
    (void)v;
    (void)hv;
    (*(size_t *)user)++;
    return RGMIN_INTERNAL_ERROR;
}

/* SPD(1) at 4 under the affine-invariant metric has <u, v> = u v / 16. With
 * gradient 2 and curvature 1 the step is -2, which is 0.5 long in the
 * manifold metric and 2 long in the Euclidean one. */
static int check_spd_tcg(void) {
    rgmin_manifold_handle_t *spd = rgmin_manifold_spd(1, RGMIN_SPD_AFFINE_INVARIANT);
    double scale = 1.0;
    const double x = 4.0;
    const double grad = 2.0;
    double eta = 0.0;
    rgmin_rtr_tcg_result_t out;
    rgmin_status_t st;
    int rc = 0;
    if (spd == NULL) {
        return 60;
    }
    st = rgmin_rtr_tcg_solve(spd, scale_hvp, &scale, &x, &grad, 10.0, 1.0, 0.1, 5, &eta, 1, &out);
    if (st != RGMIN_SUCCESS || out.stop != RGMIN_TCG_RESIDUAL || !close_to(eta, -2.0) ||
        !close_to(out.eta_norm, 0.5) || !close_to(out.model_decrease, 2.0)) {
        fprintf(stderr, "spd tcg eta %g norm %g decrease %g status %d\n", eta, out.eta_norm,
                out.model_decrease, st);
        rc = 61;
        goto done;
    }
    st = rgmin_rtr_tcg_solve(spd, scale_hvp, &scale, &x, &grad, 0.25, 1.0, 0.1, 5, &eta, 1, &out);
    if (st != RGMIN_SUCCESS || out.stop != RGMIN_TCG_BOUNDARY || !close_to(eta, -1.0) ||
        !close_to(out.eta_norm, 0.25)) {
        fprintf(stderr, "spd tcg boundary eta %g norm %g status %d\n", eta, out.eta_norm, st);
        rc = 62;
    }
done:
    rgmin_manifold_free(spd);
    return rc;
}

static int check_tcg_hessian_failure(void) {
    counted_hvp_state state;
    size_t calls = 0;
    const double x[2] = {0.0, 0.0};
    const double grad[2] = {1.0, 1.0};
    double eta[2] = {7.0, 7.0};
    rgmin_rtr_tcg_result_t out;
    rgmin_status_t st;
    out.eta_norm = -1.0;
    state.calls = 0;
    state.value = NAN;
    st = rgmin_rtr_tcg_solve(NULL, counted_hvp, &state, x, grad, 10.0, 1.0, 0.1, 50, eta, 2, &out);
    if (st != RGMIN_INTERNAL_ERROR || state.calls != 1 || eta[0] != 7.0 || eta[1] != 7.0 ||
        out.eta_norm != -1.0) {
        fprintf(stderr, "non-finite hvp status %d calls %zu eta %g\n", st, state.calls, eta[0]);
        return 63;
    }
    st = rgmin_rtr_tcg_solve(NULL, failing_hvp, &calls, x, grad, 10.0, 1.0, 0.1, 50, eta, 2, &out);
    if (st != RGMIN_INTERNAL_ERROR || calls != 1 || eta[0] != 7.0) {
        fprintf(stderr, "failing hvp status %d calls %zu\n", st, calls);
        return 64;
    }
    return 0;
}

static int check_quotient(void) {
    rgmin_manifold_handle_t *isolated = rgmin_manifold_rigid_quotient(2, 0);
    rgmin_manifold_handle_t *periodic = xts_manifold_rigid_quotient(2, 1);
    const double x[6] = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    const double trans[6] = {0.4, -0.2, 0.7, 0.4, -0.2, 0.7};
    const double rot[6] = {0.0, -1.0, 0.0, 0.0, 1.0, 0.0};
    const double stretch[6] = {-1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    double out[6];
    size_t i;
    int rc = 0;
    if (isolated == NULL || periodic == NULL || rgmin_manifold_len(isolated) != 6) {
        rc = 30;
        goto done;
    }
    if (rgmin_manifold_project(isolated, x, trans, out, 6) != RGMIN_SUCCESS) {
        rc = 31;
        goto done;
    }
    for (i = 0; i < 6; i++) {
        if (fabs(out[i]) > 1e-12) {
            rc = 32;
            goto done;
        }
    }
    if (rgmin_manifold_project(isolated, x, rot, out, 6) != RGMIN_SUCCESS) {
        rc = 33;
        goto done;
    }
    for (i = 0; i < 6; i++) {
        if (fabs(out[i]) > 1e-12) {
            rc = 34;
            goto done;
        }
    }
    if (rgmin_manifold_project(periodic, x, trans, out, 6) != RGMIN_SUCCESS) {
        rc = 35;
        goto done;
    }
    for (i = 0; i < 6; i++) {
        if (fabs(out[i]) > 1e-12) {
            rc = 36;
            goto done;
        }
    }
    if (rgmin_manifold_project(periodic, x, rot, out, 6) != RGMIN_SUCCESS) {
        rc = 37;
        goto done;
    }
    for (i = 0; i < 6; i++) {
        if (fabs(out[i] - rot[i]) > 1e-12) {
            rc = 38;
            goto done;
        }
    }
    if (rgmin_manifold_project(isolated, x, stretch, out, 6) != RGMIN_SUCCESS ||
        fabs(out[0] + 1.0) > 1e-12 || fabs(out[3] - 1.0) > 1e-12) {
        rc = 39;
    }
done:
    rgmin_manifold_free(isolated);
    rgmin_manifold_free(periodic);
    return rc;
}

static int check_projected_mode(void) {
    rgmin_manifold_handle_t *m = rgmin_manifold_rigid_quotient(2, 0);
    double x[6] = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    double seed[6] = {0.2, 0.1, -0.3, 0.4, 0.0, 0.2};
    double mode[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double values[1] = {0.0};
    DLManagedTensorVersioned *xt;
    DLManagedTensorVersioned *st;
    DLManagedTensorVersioned *mt;
    DLManagedTensorVersioned *vt;
    rgmin_eigen_params_t params = rgmin_eigen_params_default();
    rgmin_eigenpair_ex_t pair;
    rgmin_lowest_mode_t single;
    rgmin_status_t st_code;
    int rc = 0;
    if (m == NULL) {
        return 40;
    }
    params.kind = RGMIN_EIGEN_LOBPCG;
    params.tol = 1e-10;
    params.max_iter = 30;
    params.nev = 1;
    xt = rgmin_tensor_borrow_cpu_f64(x, 6);
    st = rgmin_tensor_borrow_cpu_f64(seed, 6);
    mt = rgmin_tensor_borrow_cpu_f64(mode, 6);
    vt = rgmin_tensor_borrow_cpu_f64(values, 1);
    st_code = rgmin_lowest_eigenpairs_projected(m, stretch_hvp, NULL, NULL, NULL, xt, st, mt, vt,
                                                &params, NULL, &pair, 1);
    if (st_code != RGMIN_SUCCESS || !close_to(pair.value, -2.0) || pair.converged != 1) {
        fprintf(stderr, "projected value %g status %d converged %d\n", pair.value, st_code,
                pair.converged);
        rc = 41;
    }
    x[0] = NAN;
    mode[0] = 7.0;
    st_code = rgmin_lowest_eigenpairs_projected(m, stretch_hvp, NULL, NULL, NULL, xt, st, mt, vt,
                                                &params, NULL, &pair, 1);
    if (st_code != RGMIN_INVALID_PARAMETER || mode[0] != 7.0) {
        fprintf(stderr, "non-finite projected input status %d output %g\n", st_code, mode[0]);
        rc = 42;
    }
    x[0] = 0.0;
    for (size_t i = 0; i < 6; i++) {
        seed[i] = i == 0 || i == 3 ? 0.7071067811865476 : 0.0;
    }
    params.kind = RGMIN_EIGEN_JACOBI_DAVIDSON;
    single.value = 0.0;
    single.actions = 0;
    st_code = rgmin_lowest_eigenpair_on_manifold(m, stretch_hvp, NULL, xt, st, st, &params, NULL,
                                                  &single);
    if (st_code != RGMIN_SUCCESS || !close_to(single.value, -2.0) ||
        fabs(seed[0] + seed[3]) > 1e-9) {
        fprintf(stderr, "single projected value %g status %d mode (%g, %g)\n", single.value,
                st_code, seed[0], seed[3]);
        rc = 43;
    }
    rgmin_tensor_free(xt);
    rgmin_tensor_free(st);
    rgmin_tensor_free(mt);
    rgmin_tensor_free(vt);
    rgmin_manifold_free(m);
    return rc;
}

int main(int argc, char **argv) {
    unsigned long want;
    int rc;
    rgmin_abi_stamp_t stamp;
    if (argc < 2) {
        return 2;
    }
    want = strtoul(argv[1], NULL, 10);
    if (sizeof(rgmin_rtr_tcg_result_t) != want) {
        fprintf(stderr, "rgmin_rtr_tcg_result_t is %zu bytes, Rust says %lu\n",
                sizeof(rgmin_rtr_tcg_result_t), want);
        return 3;
    }
    if (sizeof(rgmin_tcg_stop_t) != 4) {
        fprintf(stderr, "rgmin_tcg_stop_t is %zu bytes\n", sizeof(rgmin_tcg_stop_t));
        return 4;
    }
    stamp = rgmin_abi_stamp();
    if (stamp.abi_minor != 52 || RGMIN_ABI_VERSION_MINOR != 52 ||
        xts_abi_stamp().abi_minor != 52) {
        fprintf(stderr, "abi minor %u\n", stamp.abi_minor);
        return 5;
    }
    rc = check_quadratics();
    if (rc != 0) {
        return rc;
    }
    rc = check_radius();
    if (rc != 0) {
        return rc;
    }
    rc = check_radius_above_maximum();
    if (rc != 0) {
        return rc;
    }
    rc = check_spd_tcg();
    if (rc != 0) {
        return rc;
    }
    rc = check_tcg_hessian_failure();
    if (rc != 0) {
        return rc;
    }
    rc = check_quotient();
    if (rc != 0) {
        return rc;
    }
    return check_projected_mode();
}
