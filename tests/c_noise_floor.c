/* Opt in to the noise floor and read RGMIN_STOP_NOISE_FLOOR.
 * A unit quadratic at x = 0.02 has |g| = 0.02, below 3 * 0.01 and above gtol. */

#include <stdio.h>
#include "rgmin.h"

static rgmin_status_t bowl_eval(void *user, const DLManagedTensorVersioned *x, double *value_out) {
    const double *p = (const double *)x->dl_tensor.data;
    (void)user;
    *value_out = 0.5 * p[0] * p[0];
    return RGMIN_SUCCESS;
}

static rgmin_status_t bowl_grad(void *user, const DLManagedTensorVersioned *x,
                                 DLManagedTensorVersioned *grad_out) {
    const double *p = (const double *)x->dl_tensor.data;
    double *g = (double *)grad_out->dl_tensor.data;
    (void)user;
    g[0] = p[0];
    return RGMIN_SUCCESS;
}

int main(void) {
    double x = 0.02;
    DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(&x, 1);
    rgmin_control_t ctrl;
    rgmin_report_t out;
    rgmin_status_t st;
    if (tensor == NULL) {
        return 2;
    }
    ctrl.maxiter = 20;
    ctrl.gtol = 1e-8;
    ctrl.istep = 1.0;
    ctrl.memory = 0;
    ctrl.maxmove = 0.0;
    ctrl.noise_eps = 0.01;
    ctrl.noise_c = 0.0; /* non-positive selects 3 */
    out.value = 0.0;
    out.steps = 0;
    out.grad_norm = 0.0;
    out.stop = RGMIN_STOP_BUDGET;
    st = rgmin_minimize(bowl_eval, bowl_grad, NULL, tensor, &ctrl, RGMIN_STEEPEST, &out);
    rgmin_tensor_free(tensor);
    if (st != RGMIN_SUCCESS) {
        fprintf(stderr, "minimize failed: %s\n", rgmin_last_error());
        return 3;
    }
    if (out.stop != RGMIN_STOP_NOISE_FLOOR) {
        fprintf(stderr, "stop %d grad %g steps %zu\n", (int)out.stop, out.grad_norm, out.steps);
        return 4;
    }
    return 0;
}
