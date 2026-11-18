/* The XTS_STOP_* aliases of the energy and evaluation stops, through
 * xts.h alone: their values, and the stops a session reports under them. */

#include <stdio.h>
#include "xts.h"

_Static_assert(XTS_STOP_ENERGY_TOLERANCE == RGMIN_STOP_ENERGY_TOLERANCE,
               "XTS_STOP_ENERGY_TOLERANCE aliases RGMIN_STOP_ENERGY_TOLERANCE");
_Static_assert(XTS_STOP_EVAL_BUDGET == RGMIN_STOP_EVAL_BUDGET,
               "XTS_STOP_EVAL_BUDGET aliases RGMIN_STOP_EVAL_BUDGET");
_Static_assert(XTS_STOP_ENERGY_TOLERANCE == 6, "energy tolerance is stop 6");
_Static_assert(XTS_STOP_EVAL_BUDGET == 7, "evaluation budget is stop 7");

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

/* Step a session until it reports something other than a budget stop.
 * A stop of RGMIN_STOP_BUDGET means the iteration goes on. */
static int run(xts_solver_t *session, xts_report_t *out, size_t *taken) {
    double x[2] = {1.0, 1.0};
    out->stop = RGMIN_STOP_BUDGET;
    *taken = 0;
    while (out->stop == RGMIN_STOP_BUDGET && *taken < 5000) {
        DLManagedTensorVersioned *tensor = xts_tensor_borrow_cpu_f64(x, 2);
        xts_status_t st;
        if (tensor == NULL) {
            return 1;
        }
        st = xts_solver_step(session, bowl_eval, bowl_grad, NULL, tensor, out);
        xts_tensor_free(tensor);
        if (st != XTS_SUCCESS) {
            fprintf(stderr, "step failed: %s\n", xts_last_error());
            return 2;
        }
        *taken += 1;
    }
    return 0;
}

int main(void) {
    xts_control_t ctrl = xts_control_default();
    xts_report_t out;
    xts_solver_t *session;
    size_t taken = 0;
    size_t plain_taken = 0;
    ctrl.maxiter = 5000;
    ctrl.gtol = 1e-12;
    ctrl.istep = 0.1;
    ctrl.memory = 10;

    session = xts_solver_create(XTS_LBFGS, &ctrl, 2);
    if (session == NULL || run(session, &out, &plain_taken) != 0) {
        return 3;
    }
    xts_solver_free(session);
    if (out.stop != RGMIN_STOP_TOLERANCE) {
        fprintf(stderr, "plain stop %d\n", (int)out.stop);
        return 4;
    }

    session = xts_solver_create(XTS_LBFGS, &ctrl, 2);
    if (session == NULL || xts_solver_set_etol(session, 1e-6) != XTS_SUCCESS) {
        return 5;
    }
    if (run(session, &out, &taken) != 0) {
        return 6;
    }
    if (out.stop != XTS_STOP_ENERGY_TOLERANCE || taken >= plain_taken) {
        fprintf(stderr, "etol stop %d after %zu steps, plain %zu\n", (int)out.stop, taken,
                plain_taken);
        return 7;
    }
    xts_solver_clear_etol(session);
    xts_solver_free(session);

    session = xts_solver_create(XTS_POLAK_RIBIERE, &ctrl, 2);
    if (session == NULL || xts_solver_set_max_evals(session, 6) != XTS_SUCCESS) {
        return 8;
    }
    if (run(session, &out, &taken) != 0) {
        return 9;
    }
    if (out.stop != XTS_STOP_EVAL_BUDGET || xts_solver_evals(session) < 6) {
        fprintf(stderr, "budget stop %d after %zu evaluations\n", (int)out.stop,
                xts_solver_evals(session));
        return 10;
    }
    xts_solver_clear_max_evals(session);
    xts_solver_free(session);
    return 0;
}
