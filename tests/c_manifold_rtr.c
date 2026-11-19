/* ABI minor 53: rigid quotient, Steihaug-Toint tCG, radius update,
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
    if (stamp.abi_minor != 53 || RGMIN_ABI_VERSION_MINOR != 53 ||
        xts_abi_stamp().abi_minor != 53) {
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
    rc = check_quotient();
    if (rc != 0) {
        return rc;
    }
    return check_projected_mode();
}
