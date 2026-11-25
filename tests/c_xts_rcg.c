/* The Riemannian CG names through xts.h alone: the method tag, the options
 * type and its default, and the one-call entry point. */

#include <math.h>
#include <stdio.h>
#include "xts.h"

_Static_assert(XTS_RCG == RGMIN_RCG, "XTS_RCG aliases RGMIN_RCG");

static xts_status_t bowl_eval(void *user, const DLManagedTensorVersioned *x, double *value_out) {
    const double *p = (const double *)x->dl_tensor.data;
    (void)user;
    *value_out = 0.5 * (p[0] * p[0] + 4.0 * p[1] * p[1]);
    return XTS_SUCCESS;
}

static xts_status_t bowl_grad(void *user, const DLManagedTensorVersioned *x,
                              DLManagedTensorVersioned *grad_out) {
    const double *p = (const double *)x->dl_tensor.data;
    double *g = (double *)grad_out->dl_tensor.data;
    (void)user;
    g[0] = p[0];
    g[1] = 4.0 * p[1];
    return XTS_SUCCESS;
}

int main(void) {
    xts_rcg_options_t options = xts_rcg_options_default();
    rgmin_rcg_options_t same = rgmin_rcg_options_default();
    xts_report_t out;
    double x[2] = {1.0, 1.0};
    DLManagedTensorVersioned *tensor;
    xts_status_t st;

    if (options.beta != same.beta || options.transport != same.transport ||
        options.linesearch != same.linesearch || options.max_iter != same.max_iter) {
        fprintf(stderr, "xts_rcg_options_default differs from rgmin_rcg_options_default\n");
        return 1;
    }
    tensor = xts_tensor_borrow_cpu_f64(x, 2);
    if (tensor == NULL) {
        return 2;
    }
    st = xts_minimize_rcg(bowl_eval, bowl_grad, NULL, tensor, &options, &out);
    xts_tensor_free(tensor);
    if (st != XTS_SUCCESS) {
        fprintf(stderr, "xts_minimize_rcg failed: %s\n", xts_last_error());
        return 3;
    }
    if (fabs(x[0]) > 1e-4 || fabs(x[1]) > 1e-4) {
        fprintf(stderr, "xts_minimize_rcg stopped at (%g, %g)\n", x[0], x[1]);
        return 4;
    }
    return 0;
}
