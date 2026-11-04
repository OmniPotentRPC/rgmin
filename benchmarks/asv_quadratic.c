#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "rgmin.h"

static const double *f64_base(const DLManagedTensorVersioned *tensor, int64_t *n_out) {
    const DLTensor *dl;
    const char *base;
    dl = &tensor->dl_tensor;
    base = (const char *)dl->data;
    *n_out = (dl->ndim > 0 && dl->shape != NULL) ? dl->shape[0] : 0;
    return (const double *)(base + dl->byte_offset);
}

static rgmin_status_t quadratic_value(void *user, const DLManagedTensorVersioned *x,
                                      double *value_out) {
    const double *data;
    int64_t n;
    int64_t i;
    double sum;
    (void)user;
    data = f64_base(x, &n);
    sum = 0.0;
    for (i = 0; i < n; i++) {
        sum += data[i] * data[i];
    }
    *value_out = 0.5 * sum;
    return RGMIN_SUCCESS;
}

static rgmin_status_t quadratic_gradient(void *user, const DLManagedTensorVersioned *x,
                                         DLManagedTensorVersioned *grad_out) {
    const double *data;
    double *grad;
    int64_t n;
    int64_t i;
    int64_t grad_n;
    (void)user;
    data = f64_base(x, &n);
    grad = (double *)f64_base(grad_out, &grad_n);
    if (grad_n < n) {
        n = grad_n;
    }
    for (i = 0; i < n; i++) {
        grad[i] = data[i];
    }
    return RGMIN_SUCCESS;
}

int rgmin_asv_quadratic(int n, int method) {
    double *x;
    int i;
    DLManagedTensorVersioned *tensor;
    rgmin_control_t ctrl;
    rgmin_report_t out;
    rgmin_status_t status;
    if (n < 1) {
        return RGMIN_INVALID_PARAMETER;
    }
    x = (double *)calloc((size_t)n, sizeof(double));
    if (x == NULL) {
        return RGMIN_INTERNAL_ERROR;
    }
    for (i = 0; i < n; i++) {
        x[i] = 0.2 + (n == 1 ? 0.0 : 0.8 * (double)i / (double)(n - 1));
    }
    tensor = rgmin_tensor_borrow_cpu_f64(x, (size_t)n);
    if (tensor == NULL) {
        free(x);
        return RGMIN_INTERNAL_ERROR;
    }
    ctrl.maxiter = 40;
    ctrl.gtol = 1e-8;
    ctrl.istep = 0.1;
    ctrl.memory = 10;
    ctrl.maxmove = 0.0;
    out.value = 0.0;
    out.steps = 0;
    out.grad_norm = 0.0;
    status = rgmin_minimize(quadratic_value, quadratic_gradient, NULL, tensor, &ctrl,
                            (rgmin_method_t)method, &out);
    rgmin_tensor_free(tensor);
    free(x);
    return (int)status;
}
