/* Session steps through the fused callback for the methods that keep no
 * overlay: every step must finish without reading a stack slot it never
 * wrote. The test passes without a checker. Run under valgrind with a
 * release library (RGMIN_C_TEST_RUNNER and cargo test --release), memcheck
 * reports a conditional jump on an uninitialised value from the step. */

#include <stdio.h>
#include "rgmin.h"

#define N 12

static rgmin_status_t bowl(void *user, const DLManagedTensorVersioned *x, double *value_out,
                           DLManagedTensorVersioned *grad_out) {
    const double *p = (const double *)x->dl_tensor.data;
    double *g = (double *)grad_out->dl_tensor.data;
    double sum = 0.0;
    size_t i;
    (void)user;
    for (i = 0; i < N; i++) {
        double k = 1.0 + (double)i;
        sum += 0.5 * k * p[i] * p[i];
        g[i] = k * p[i];
    }
    *value_out = sum;
    return RGMIN_SUCCESS;
}

static int drive(rgmin_method_t method, int etol) {
    rgmin_control_t ctrl = rgmin_control_default();
    rgmin_report_t out;
    rgmin_solver_t *session;
    double x[N];
    size_t i;
    int taken = 0;
    for (i = 0; i < N; i++) {
        x[i] = 0.3 + 0.1 * (double)i;
    }
    ctrl.maxiter = 5000;
    ctrl.gtol = 1e-6;
    session = rgmin_solver_create(method, &ctrl, N);
    if (session == NULL) {
        fprintf(stderr, "method %d: no session: %s\n", (int)method, rgmin_last_error());
        return 1;
    }
    if (etol && rgmin_solver_set_etol(session, 1e-12) != RGMIN_SUCCESS) {
        rgmin_solver_free(session);
        return 2;
    }
    out.stop = RGMIN_STOP_BUDGET;
    while (out.stop == RGMIN_STOP_BUDGET && taken < 5000) {
        DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(x, N);
        rgmin_status_t st;
        if (tensor == NULL) {
            rgmin_solver_free(session);
            return 3;
        }
        st = rgmin_solver_step_fg(session, bowl, NULL, tensor, &out);
        rgmin_tensor_free(tensor);
        if (st != RGMIN_SUCCESS) {
            fprintf(stderr, "method %d: step failed: %s\n", (int)method, rgmin_last_error());
            rgmin_solver_free(session);
            return 4;
        }
        taken++;
    }
    rgmin_solver_free(session);
    if (out.stop == RGMIN_STOP_BUDGET) {
        fprintf(stderr, "method %d: no stop after %d steps\n", (int)method, taken);
        return 5;
    }
    return 0;
}

int main(void) {
    const rgmin_method_t methods[] = {RGMIN_FIRE, RGMIN_FIRE2, RGMIN_BB, RGMIN_LBFGS};
    size_t m;
    for (m = 0; m < sizeof methods / sizeof methods[0]; m++) {
        int rc = drive(methods[m], 0);
        if (rc == 0) {
            rc = drive(methods[m], 1);
        }
        if (rc != 0) {
            return 10 + rc;
        }
    }
    return 0;
}
