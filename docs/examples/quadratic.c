/* x^2 + y^2 from (1, -2). The value ends below 1e-8.
 *
 * The tensor from rgmin_tensor_borrow_cpu_f64 is freed with
 * rgmin_tensor_free. rgmin_minimize does not take that tensor.
 * rgmin_last_error_message is borrowed until the next call on this
 * thread. The caller does not free it.
 */
#include <rgmin.h>

#include <math.h>
#include <stdio.h>

static double *cpu(DLManagedTensorVersioned *t) {
    return (double *)((char *)t->dl_tensor.data + t->dl_tensor.byte_offset);
}

static const double *cpu_const(const DLManagedTensorVersioned *t) {
    return (const double *)((const char *)t->dl_tensor.data + t->dl_tensor.byte_offset);
}

static rgmin_status_t eval(void *user, const DLManagedTensorVersioned *x, double *value) {
    const double *p = cpu_const(x);
    (void)user;
    *value = p[0] * p[0] + p[1] * p[1];
    return RGMIN_SUCCESS;
}

static rgmin_status_t grad(void *user, const DLManagedTensorVersioned *x,
                           DLManagedTensorVersioned *g) {
    const double *p = cpu_const(x);
    double *gp = cpu(g);
    (void)user;
    gp[0] = 2.0 * p[0];
    gp[1] = 2.0 * p[1];
    return RGMIN_SUCCESS;
}

int main(void) {
    double x[2] = {1.0, -2.0};
    DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(x, 2);
    rgmin_control_t ctrl = {40, 1e-8, 0.1, 8, 0.2};
    rgmin_report_t out = {0.0, 0, 0.0};
    rgmin_status_t st = rgmin_minimize(eval, grad, 0, tensor, &ctrl, RGMIN_LBFGS, &out);
    rgmin_tensor_free(tensor);
    if (st != RGMIN_SUCCESS || !(out.value < 1e-8) || fabs(x[0]) > 1e-4 || fabs(x[1]) > 1e-4) {
        const char *msg = rgmin_last_error_message();
        fprintf(stderr, "quadratic f=%g x=(%g,%g) status=%d %s\n", out.value, x[0], x[1],
                (int)st, msg ? msg : "");
        return 1;
    }
    printf("quadratic %.3e\n", out.value);
    return 0;
}
