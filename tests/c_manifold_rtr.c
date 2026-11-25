/* ABI minor 56: rigid quotient, Steihaug-Toint tCG, radius update,
 * and tangent-projected eigenpairs. argv[1] is sizeof(rgmin_rtr_tcg_result_t)
 * as Rust reports it. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static rgmin_status_t identity_precond(void *user, const DLManagedTensorVersioned *block,
                                       DLManagedTensorVersioned *out) {
    const double *in = (const double *)block->dl_tensor.data;
    double *dst = (double *)out->dl_tensor.data;
    size_t n = 1;
    int32_t d;
    (void)user;
    for (d = 0; d < block->dl_tensor.ndim; d++) {
        n *= (size_t)block->dl_tensor.shape[d];
    }
    memcpy(dst, in, n * sizeof(double));
    return RGMIN_SUCCESS;
}

static int bytes_equal(const double *a, const double *b, size_t n) {
    return memcmp(a, b, n * sizeof(double)) == 0;
}

/* The stretch Hessian has eigenvalues -100 (translation), -2 (stretch) and
 * 0 elsewhere. A null manifold must match rgmin_lowest_eigenpairs_ex bit for
 * bit, since both run the same unprojected solver. */
static int check_null_manifold_parity(void) {
    const double x0[6] = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    const double seed0[6] = {0.2, 0.1, -0.3, 0.4, 0.0, 0.2};
    double x[6], seed[6], modes_a[6], modes_b[6], value_a[1] = {0.0}, value_b[1] = {0.0};
    DLManagedTensorVersioned *xt, *st, *ma, *mb, *va, *vb;
    rgmin_eigen_params_t params = rgmin_eigen_params_default();
    rgmin_eigenpair_ex_t pa, pb;
    rgmin_status_t sa, sb;
    int rc = 0;
    memcpy(x, x0, sizeof x);
    memcpy(seed, seed0, sizeof seed);
    memset(modes_a, 0, sizeof modes_a);
    memset(modes_b, 0, sizeof modes_b);
    params.kind = RGMIN_EIGEN_LOBPCG;
    params.tol = 1e-10;
    params.max_iter = 40;
    params.nev = 1;
    xt = rgmin_tensor_borrow_cpu_f64(x, 6);
    st = rgmin_tensor_borrow_cpu_f64(seed, 6);
    ma = rgmin_tensor_borrow_cpu_f64(modes_a, 6);
    mb = rgmin_tensor_borrow_cpu_f64(modes_b, 6);
    va = rgmin_tensor_borrow_cpu_f64(value_a, 1);
    vb = rgmin_tensor_borrow_cpu_f64(value_b, 1);
    sa = rgmin_lowest_eigenpairs_ex(stretch_hvp, NULL, NULL, NULL, xt, st, ma, va, &params, NULL,
                                    &pa, 1);
    sb = rgmin_lowest_eigenpairs_projected(NULL, stretch_hvp, NULL, NULL, NULL, xt, st, mb, vb,
                                           &params, NULL, &pb, 1);
    if (sa != RGMIN_SUCCESS || sb != RGMIN_SUCCESS || !bytes_equal(modes_a, modes_b, 6) ||
        !bytes_equal(value_a, value_b, 1) || pa.actions != pb.actions ||
        pa.converged != pb.converged || !close_to(value_a[0], -100.0)) {
        fprintf(stderr, "null manifold parity: status %d %d value %g %g actions %zu %zu\n", sa, sb,
                value_a[0], value_b[0], (size_t)pa.actions, (size_t)pb.actions);
        rc = 70;
    }
    rgmin_tensor_free(xt);
    rgmin_tensor_free(st);
    rgmin_tensor_free(ma);
    rgmin_tensor_free(mb);
    rgmin_tensor_free(va);
    rgmin_tensor_free(vb);
    return rc;
}

static int check_projected_in_place_and_precond(void) {
    rgmin_manifold_handle_t *m = rgmin_manifold_rigid_quotient(2, 0);
    double x[6] = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    double seed[6] = {0.2, 0.1, -0.3, 0.4, 0.0, 0.2};
    double values[1] = {0.0};
    DLManagedTensorVersioned *xt, *st, *vt;
    rgmin_eigen_params_t params = rgmin_eigen_params_default();
    rgmin_eigenpair_ex_t pair;
    rgmin_lowest_mode_t single;
    rgmin_status_t code;
    int rc = 0;
    if (m == NULL) {
        return 71;
    }
    params.kind = RGMIN_EIGEN_LOBPCG;
    params.tol = 1e-10;
    params.max_iter = 40;
    params.nev = 1;
    xt = rgmin_tensor_borrow_cpu_f64(x, 6);
    st = rgmin_tensor_borrow_cpu_f64(seed, 6);
    vt = rgmin_tensor_borrow_cpu_f64(values, 1);
    /* modes_out is x itself, with a non-null preconditioner. */
    code = rgmin_lowest_eigenpairs_projected(m, stretch_hvp, NULL, identity_precond, NULL, xt, st,
                                             xt, vt, &params, NULL, &pair, 1);
    if (code != RGMIN_SUCCESS || !close_to(pair.value, -2.0) || !close_to(values[0], -2.0) ||
        fabs(fabs(x[0]) - 0.7071067811865476) > 1e-6 || fabs(x[0] + x[3]) > 1e-6 ||
        fabs(x[1]) > 1e-6 || fabs(x[2]) > 1e-6) {
        fprintf(stderr, "in-place block status %d value %g x (%g, %g, %g, %g)\n", code,
                pair.value, x[0], x[1], x[2], x[3]);
        rc = 72;
        goto done;
    }
    /* mode_out is x itself on the single-pair entry. */
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 0.0;
    x[3] = 2.0;
    x[4] = 0.0;
    x[5] = 0.0;
    params.kind = RGMIN_EIGEN_LANCZOS;
    code = rgmin_lowest_eigenpair_on_manifold(m, stretch_hvp, NULL, xt, st, xt, &params, NULL,
                                              &single);
    if (code != RGMIN_SUCCESS || !close_to(single.value, -2.0) ||
        fabs(fabs(x[0]) - 0.7071067811865476) > 1e-6 || fabs(x[0] + x[3]) > 1e-6) {
        fprintf(stderr, "in-place single status %d value %g x (%g, %g)\n", code, single.value,
                x[0], x[3]);
        rc = 73;
    }
done:
    rgmin_tensor_free(xt);
    rgmin_tensor_free(st);
    rgmin_tensor_free(vt);
    rgmin_manifold_free(m);
    return rc;
}

static int check_projected_failures(void) {
    rgmin_manifold_handle_t *m = rgmin_manifold_rigid_quotient(2, 0);
    double x[6] = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    double seed[6] = {0.2, 0.1, -0.3, 0.4, 0.0, 0.2};
    double block_seed[12] = {0.2, 0.1, -0.3, 0.4, 0.0, 0.2, 0.1, 0.3, 0.2, -0.1, 0.5, 0.0};
    double mode[6] = {7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
    double modes2[12] = {7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
    double values1[1] = {7.0};
    double values2[2] = {7.0, 7.0};
    size_t calls = 0;
    size_t i;
    DLManagedTensorVersioned *xt, *st, *bt, *mt, *m2t, *v1t, *v2t;
    rgmin_eigen_params_t params = rgmin_eigen_params_default();
    rgmin_eigenpair_ex_t pairs[2];
    rgmin_lowest_mode_t single;
    rgmin_status_t code;
    int rc = 0;
    if (m == NULL) {
        return 74;
    }
    params.kind = RGMIN_EIGEN_LOBPCG;
    params.tol = 1e-10;
    params.max_iter = 30;
    params.nev = 1;
    xt = rgmin_tensor_borrow_cpu_f64(x, 6);
    st = rgmin_tensor_borrow_cpu_f64(seed, 6);
    bt = rgmin_tensor_borrow_cpu_f64(block_seed, 12);
    mt = rgmin_tensor_borrow_cpu_f64(mode, 6);
    m2t = rgmin_tensor_borrow_cpu_f64(modes2, 12);
    v1t = rgmin_tensor_borrow_cpu_f64(values1, 1);
    v2t = rgmin_tensor_borrow_cpu_f64(values2, 2);

    /* A failing Hessian callback is an internal error that names the
     * callback, and it leaves every output as it was. */
    code = rgmin_lowest_eigenpairs_projected(m, failing_hvp, &calls, NULL, NULL, xt, st, mt, v1t,
                                             &params, NULL, pairs, 1);
    if (code != RGMIN_INTERNAL_ERROR || strstr(rgmin_last_error(), "HVP callback") == NULL ||
        mode[0] != 7.0 || values1[0] != 7.0) {
        fprintf(stderr, "block hvp failure status %d message '%s'\n", code, rgmin_last_error());
        rc = 75;
        goto done;
    }
    code = rgmin_lowest_eigenpair_on_manifold(m, failing_hvp, &calls, xt, st, mt, &params, NULL,
                                              &single);
    if (code != RGMIN_INTERNAL_ERROR || strstr(rgmin_last_error(), "HVP callback") == NULL ||
        strstr(rgmin_last_error(), "projection failed") != NULL || mode[0] != 7.0) {
        fprintf(stderr, "manifold hvp failure status %d message '%s'\n", code,
                rgmin_last_error());
        rc = 76;
        goto done;
    }
    code = rgmin_lowest_eigenpair_on_manifold(NULL, failing_hvp, &calls, xt, st, mt, &params, NULL,
                                              &single);
    if (code != RGMIN_INTERNAL_ERROR || strstr(rgmin_last_error(), "HVP callback") == NULL ||
        strstr(rgmin_last_error(), "projection failed") != NULL || mode[0] != 7.0) {
        fprintf(stderr, "null manifold hvp failure status %d message '%s'\n", code,
                rgmin_last_error());
        rc = 77;
        goto done;
    }

    /* Two atoms leave one horizontal direction, so two pairs do not exist. */
    params.nev = 2;
    code = rgmin_lowest_eigenpairs_projected(m, stretch_hvp, NULL, NULL, NULL, xt, bt, m2t, v2t,
                                             &params, NULL, pairs, 2);
    if (code != RGMIN_INVALID_PARAMETER || modes2[0] != 7.0 || values2[0] != 7.0) {
        fprintf(stderr, "k above the horizontal rank status %d\n", code);
        rc = 78;
        goto done;
    }
    params.nev = 1;

    /* A non-finite seed is refused before any output is written. */
    seed[1] = NAN;
    code = rgmin_lowest_eigenpair_on_manifold(m, stretch_hvp, NULL, xt, st, mt, &params, NULL,
                                              &single);
    if (code != RGMIN_INVALID_PARAMETER || mode[0] != 7.0) {
        fprintf(stderr, "non-finite seed status %d\n", code);
        rc = 79;
        goto done;
    }
    seed[1] = 0.1;

    /* PRIMME, libkrylov and SLEPc cannot honor a projector. */
    {
        const int32_t external[3] = {RGMIN_EIGEN_PRIMME, RGMIN_EIGEN_LIBKRYLOV, RGMIN_EIGEN_SLEPC};
        params.max_iter = 64;
        for (i = 0; i < 3; i++) {
            params.kind = external[i];
            mode[0] = 7.0;
            code = rgmin_lowest_eigenpair_on_manifold(m, stretch_hvp, NULL, xt, st, mt, &params,
                                                      NULL, &single);
            if (code != RGMIN_INVALID_PARAMETER || strstr(rgmin_last_error(), "projector") == NULL ||
                mode[0] != 7.0) {
                fprintf(stderr, "external kind %d with a manifold: status %d message '%s'\n",
                        (int)external[i], code, rgmin_last_error());
                rc = 80;
                goto done;
            }
            code = rgmin_lowest_eigenpair_on_manifold(NULL, stretch_hvp, NULL, xt, st, mt, &params,
                                                      NULL, &single);
            if (code == RGMIN_INVALID_PARAMETER) {
                fprintf(stderr, "external kind %d without a manifold was refused\n",
                        (int)external[i]);
                rc = 81;
                goto done;
            }
        }
    }
done:
    rgmin_tensor_free(xt);
    rgmin_tensor_free(st);
    rgmin_tensor_free(bt);
    rgmin_tensor_free(mt);
    rgmin_tensor_free(m2t);
    rgmin_tensor_free(v1t);
    rgmin_tensor_free(v2t);
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
    if (stamp.abi_minor != RGMIN_ABI_VERSION_MINOR || stamp.abi_minor < 56 ||
        xts_abi_stamp().abi_minor != RGMIN_ABI_VERSION_MINOR) {
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
    rc = check_projected_mode();
    if (rc != 0) {
        return rc;
    }
    rc = check_null_manifold_parity();
    if (rc != 0) {
        return rc;
    }
    rc = check_projected_in_place_and_precond();
    if (rc != 0) {
        return rc;
    }
    return check_projected_failures();
}
