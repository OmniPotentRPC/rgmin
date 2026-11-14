/* Install a HiGHS callback through rgmin.h and take constrained steps.
 * f = x.x / 2 under x0 + x1 = 0 from (2, -1). The callback counts calls
 * through its user pointer, and a NULL callback clears it. */

#include <stdio.h>
#include "rgmin.h"

static rgmin_status_t half_square(void *user, const DLManagedTensorVersioned *x,
                                  double *value_out, DLManagedTensorVersioned *grad_out) {
    const double *p = (const double *)x->dl_tensor.data;
    double *g = (double *)grad_out->dl_tensor.data;
    (void)user;
    *value_out = 0.5 * (p[0] * p[0] + p[1] * p[1]);
    g[0] = p[0];
    g[1] = p[1];
    return RGMIN_SUCCESS;
}

static void count_calls(int32_t kind, const char *message, int32_t *interrupt, void *user) {
    (void)kind;
    (void)message;
    (void)interrupt;
    ++*(unsigned long *)user;
}

static int step(rgmin_solver_t *s, DLManagedTensorVersioned *tensor, rgmin_report_t *out) {
    rgmin_status_t st = rgmin_solver_step_fg(s, half_square, NULL, tensor, out);
    if (st != RGMIN_SUCCESS) {
        fprintf(stderr, "step failed: %s\n", rgmin_last_error());
        return 1;
    }
    return 0;
}

int main(void) {
    double x[2] = {2.0, -1.0};
    size_t idx[2] = {0, 1};
    double coef[2] = {1.0, 1.0};
    unsigned long hits = 0;
    unsigned long called;
    rgmin_control_t ctrl = {.maxiter = 100, .gtol = 1e-8, .istep = 1.0, .memory = 10};
    rgmin_report_t out = {.stop = RGMIN_STOP_BUDGET};
    rgmin_solver_t *s = rgmin_solver_create(RGMIN_LBFGS, &ctrl, 2);
    DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(x, 2);
    int failed = 0;
    if (s == NULL || tensor == NULL) {
        return 2;
    }
    rgmin_solver_set_accept(s, RGMIN_ACCEPT_STEP);
    if (rgmin_solver_set_highs_solver(s, RGMIN_HIGHS_IPM) != 0 ||
        rgmin_solver_set_highs_crossover(s, RGMIN_HIGHS_CROSSOVER_OFF) != 0 ||
        rgmin_solver_set_highs_callback(s, count_calls, &hits) != 0 ||
        rgmin_solver_set_highs(s, 1) != 0 ||
        rgmin_solver_add_equality(s, idx, coef, 2, 0.0) != 0) {
        fprintf(stderr, "setup failed: %s\n", rgmin_last_error());
        failed = 3;
    } else if (step(s, tensor, &out) != 0) {
        failed = 4;
    } else if (hits == 0 || x[0] + x[1] - 1.0 > 1e-7 || x[0] + x[1] - 1.0 < -1e-7) {
        fprintf(stderr, "hits %lu x (%g, %g)\n", hits, x[0], x[1]);
        failed = 5;
    } else {
        called = hits;
        x[0] = 2.0;
        x[1] = -1.0;
        if (rgmin_solver_set_highs_callback(s, NULL, NULL) != 0 || step(s, tensor, &out) != 0) {
            failed = 6;
        } else if (hits != called) {
            fprintf(stderr, "a cleared callback ran: %lu then %lu\n", called, hits);
            failed = 7;
        }
    }
    rgmin_tensor_free(tensor);
    rgmin_solver_free(s);
    return failed;
}
