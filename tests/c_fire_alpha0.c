/* A FIRE session with LAMMPS min_modify alpha0 0 starts and steps.
 * doc/src/min_modify.rst lists the fire defaults: tmax 10, delaystep 20,
 * dtgrow 1.1, dtshrink 0.5, alphashrink 0.99. dt_max is tmax times dt.
 * alpha0 is the one value changed, to zero. */

#include <stdio.h>

#include "rgmin.h"

static rgmin_status_t bowl(void *user, const DLManagedTensorVersioned *x, double *value_out,
                           DLManagedTensorVersioned *grad_out) {
    const double *p = (const double *)x->dl_tensor.data;
    double *g = (double *)grad_out->dl_tensor.data;
    (void)user;
    *value_out = 0.5 * (p[0] * p[0] + p[1] * p[1]);
    g[0] = p[0];
    g[1] = p[1];
    return RGMIN_SUCCESS;
}

static int finite_pair(const double *x) {
    return x[0] == x[0] && x[1] == x[1] && x[0] < 1e300 && x[0] > -1e300 && x[1] < 1e300 &&
           x[1] > -1e300;
}

static int drive(rgmin_method_t method) {
    rgmin_control_t ctrl = rgmin_control_default();
    rgmin_solver_t *session;
    double x[2] = {1.0, -0.5};
    double start0 = x[0];
    double start1 = x[1];
    int step;
    ctrl.istep = 0.1;
    ctrl.gtol = 1e-12;
    ctrl.maxiter = 8;
    session = rgmin_solver_create(method, &ctrl, 2);
    if (session == NULL) {
        fprintf(stderr, "no session: %s\n", rgmin_last_error());
        return 1;
    }
    if (method == RGMIN_FIRE2 &&
        rgmin_solver_set_fire_preset(session, RGMIN_FIRE_PRESET_GUENOLE2020) != 0) {
        fprintf(stderr, "preset: %s\n", rgmin_last_error());
        rgmin_solver_free(session);
        return 2;
    }
    if (rgmin_solver_set_fire_config(session, 0.1, 1.0, 20, 1.1, 0.5, 0.0, 0.99) != 0) {
        fprintf(stderr, "config: %s\n", rgmin_last_error());
        rgmin_solver_free(session);
        return 3;
    }
    for (step = 0; step < 4; step++) {
        DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(x, 2);
        rgmin_report_t out;
        rgmin_status_t st;
        if (tensor == NULL) {
            fprintf(stderr, "no tensor\n");
            rgmin_solver_free(session);
            return 4;
        }
        out.value = 0.0;
        out.steps = 0;
        out.grad_norm = 0.0;
        out.stop = RGMIN_STOP_TOLERANCE;
        out.converged = 0;
        st = rgmin_solver_step_fg(session, bowl, NULL, tensor, &out);
        rgmin_tensor_free(tensor);
        if (st != RGMIN_SUCCESS) {
            fprintf(stderr, "step %d: %s\n", step, rgmin_last_error());
            rgmin_solver_free(session);
            return 5;
        }
        if (!finite_pair(x)) {
            fprintf(stderr, "non-finite step %d: %g %g\n", step, x[0], x[1]);
            rgmin_solver_free(session);
            return 6;
        }
    }
    rgmin_solver_free(session);
    if (x[0] == start0 && x[1] == start1) {
        fprintf(stderr, "alpha0 0 did not move\n");
        return 7;
    }
    return 0;
}

int main(void) {
    int rc = drive(RGMIN_FIRE);
    if (rc != 0) {
        return 10 + rc;
    }
    rc = drive(RGMIN_FIRE2);
    if (rc != 0) {
        return 20 + rc;
    }
    return 0;
}
