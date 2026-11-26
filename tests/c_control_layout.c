/* The plain names in rgmin.h pass sizeof(rgmin_control_t), so a caller
 * built against this header has require_convergence read and converged
 * written. RGMIN_CONTROL_REVISION5_SIZE stands for a caller built
 * against layout revision 5: the bytes past noise_c are not read, and
 * the report is written only through stop. RGMIN_CONTROL_REVISION6_SIZE
 * stands for a caller built against layout revision 6: the bytes past
 * require_convergence are not read, and the whole report is written. */

#include <stdio.h>
#include <string.h>
#include "rgmin.h"

_Static_assert(RGMIN_CONTROL_REVISION5_SIZE == offsetof(rgmin_control_t, require_convergence),
               "revision 5 ends before require_convergence");
_Static_assert(RGMIN_CONTROL_REVISION6_SIZE == offsetof(rgmin_control_t, noise_eps_f),
               "revision 6 ends before noise_eps_f");
#if SIZE_MAX == UINT64_MAX
_Static_assert(RGMIN_CONTROL_REVISION5_SIZE == 56, "revision 5 control is 56 bytes");
_Static_assert(RGMIN_CONTROL_REVISION6_SIZE == 64, "revision 6 control is 64 bytes");
_Static_assert(sizeof(rgmin_control_t) == 72, "control is 72 bytes");
_Static_assert(offsetof(rgmin_report_t, converged) == 28, "converged follows stop");
_Static_assert(sizeof(rgmin_report_t) == 32, "report is 32 bytes");
#endif

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

static rgmin_status_t run(const rgmin_control_t *ctrl, size_t size, int plain,
                          rgmin_report_t *out) {
    double x = 4.0;
    DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(&x, 1);
    rgmin_status_t st;
    memset(out, 0xA5, sizeof *out);
    if (plain) {
        st = rgmin_minimize(bowl_eval, bowl_grad, NULL, tensor, ctrl, RGMIN_STEEPEST, out);
    } else {
        st = rgmin_minimize_opts_sized(bowl_eval, bowl_grad, NULL, tensor, ctrl, size,
                                       RGMIN_STEEPEST, 0, 0, out);
    }
    rgmin_tensor_free(tensor);
    return st;
}

int main(void) {
    rgmin_control_t ctrl = rgmin_control_default();
    rgmin_report_t out;
    rgmin_status_t st;
    unsigned char old[sizeof(rgmin_control_t)];
    unsigned char sentinel[sizeof(rgmin_report_t) - offsetof(rgmin_report_t, converged)];
    rgmin_solver_t *solver;

    if (ctrl.require_convergence != 0 || ctrl.maxiter != 100 || ctrl.noise_eps_f != 0.0) {
        fprintf(stderr, "default require_convergence %d maxiter %zu noise_eps_f %g\n",
                (int)ctrl.require_convergence, ctrl.maxiter, ctrl.noise_eps_f);
        return 2;
    }
    ctrl.maxiter = 0;
    ctrl.gtol = 1e-8;

    st = run(&ctrl, 0, 1, &out);
    if (st != RGMIN_SUCCESS || out.stop != RGMIN_STOP_BUDGET || out.converged != 0) {
        fprintf(stderr, "default run %d stop %d converged %d\n", (int)st, (int)out.stop,
                (int)out.converged);
        return 3;
    }
    ctrl.require_convergence = 1;
    st = run(&ctrl, 0, 1, &out);
    if (st != RGMIN_NOT_CONVERGED || out.stop != RGMIN_STOP_BUDGET || out.converged != 0) {
        fprintf(stderr, "strict run %d stop %d converged %d\n", (int)st, (int)out.stop,
                (int)out.converged);
        return 4;
    }

    memcpy(old, &ctrl, RGMIN_CONTROL_REVISION5_SIZE);
    memset(old + RGMIN_CONTROL_REVISION5_SIZE, 0xFF, sizeof old - RGMIN_CONTROL_REVISION5_SIZE);
    st = run((const rgmin_control_t *)(const void *)old, RGMIN_CONTROL_REVISION5_SIZE, 0, &out);
    memset(sentinel, 0xA5, sizeof sentinel);
    if (st != RGMIN_SUCCESS || out.stop != RGMIN_STOP_BUDGET ||
        memcmp((const unsigned char *)&out + offsetof(rgmin_report_t, converged), sentinel,
               sizeof sentinel) != 0) {
        fprintf(stderr, "revision 5 run %d stop %d\n", (int)st, (int)out.stop);
        return 5;
    }

    st = run(&ctrl, RGMIN_CONTROL_REVISION5_SIZE + 4, 0, &out);
    if (st != RGMIN_INVALID_PARAMETER || strstr(rgmin_last_error(), "ctrl_size") == NULL) {
        fprintf(stderr, "odd size %d: %s\n", (int)st, rgmin_last_error());
        return 6;
    }
    if (rgmin_control_init(&ctrl, RGMIN_CONTROL_REVISION5_SIZE + 4) != RGMIN_INVALID_PARAMETER) {
        return 7;
    }

    ctrl = rgmin_control_default();
    ctrl.require_convergence = 1;
    solver = rgmin_solver_create(RGMIN_STEEPEST, &ctrl, 1);
    if (solver == NULL) {
        fprintf(stderr, "create: %s\n", rgmin_last_error());
        return 8;
    }
    rgmin_solver_free(solver);
    if (rgmin_solver_create_sized(RGMIN_STEEPEST, &ctrl, 8, 1) != NULL) {
        return 9;
    }

    /* A revision 6 caller: require_convergence is read and converged is
     * written, and the bytes past require_convergence stay unread. */
    ctrl = rgmin_control_default();
    ctrl.maxiter = 0;
    ctrl.gtol = 1e-8;
    ctrl.require_convergence = 1;
    memcpy(old, &ctrl, RGMIN_CONTROL_REVISION6_SIZE);
    memset(old + RGMIN_CONTROL_REVISION6_SIZE, 0xFF, sizeof old - RGMIN_CONTROL_REVISION6_SIZE);
    st = run((const rgmin_control_t *)(const void *)old, RGMIN_CONTROL_REVISION6_SIZE, 0, &out);
    if (st != RGMIN_NOT_CONVERGED || out.stop != RGMIN_STOP_BUDGET || out.converged != 0) {
        fprintf(stderr, "revision 6 run %d stop %d converged %d\n", (int)st, (int)out.stop,
                (int)out.converged);
        return 10;
    }
    memset(&ctrl, 0xA5, sizeof ctrl);
    if (rgmin_control_init(&ctrl, RGMIN_CONTROL_REVISION6_SIZE) != RGMIN_SUCCESS ||
        ctrl.require_convergence != 0 || ctrl.maxiter != 100) {
        return 11;
    }
    {
        const unsigned char *bytes = (const unsigned char *)&ctrl;
        size_t i;
        for (i = RGMIN_CONTROL_REVISION6_SIZE; i < sizeof ctrl; ++i) {
            if (bytes[i] != 0xA5) {
                return 12;
            }
        }
    }
    return 0;
}
