/* A C caller of rgmin_minimize_eindir that links librgmin alone builds and
 * queries its objective through the eindir-core C ABI librgmin re-exports.
 * Each of those 13 functions is called here, so a build that hides one
 * fails to link. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "rgmin.h"

#if defined(__has_include)
#  if !__has_include(<eindir-core.h>)
#    define RGMIN_DECLARE_EINDIR 1
#  endif
#endif

#ifdef RGMIN_DECLARE_EINDIR
/* rgmin.h only forward declares the eindir types. These are eindir-core's
 * declarations as its header gives them. */
typedef enum {
    EINDIR_SUCCESS = 0,
    EINDIR_INVALID_PARAMETER = 1,
    EINDIR_INTERNAL_ERROR = 2
} eindir_status_t;
struct eindir_abi_stamp_t {
    uint32_t abi_major;
    uint32_t abi_minor;
    uint32_t objective_layout;
    size_t objective_size;
    size_t objective_align;
    uint32_t dlpack_major;
    uint32_t dlpack_minor;
    uint64_t features;
};
typedef eindir_status_t (*eindir_eval_fn)(void *, const DLManagedTensorVersioned *, double *);
typedef eindir_status_t (*eindir_grad_fn)(void *, const DLManagedTensorVersioned *,
                                          DLManagedTensorVersioned *);
const char *eindir_core_version(void);
const char *eindir_core_abi_family(void);
eindir_abi_stamp_t eindir_core_abi_stamp(void);
int32_t eindir_core_abi_compatible(const eindir_abi_stamp_t *stamp);
const char *eindir_last_error(void);
eindir_objective_t *eindir_objective_new(size_t dim, const DLManagedTensorVersioned *bounds_low,
                                         const DLManagedTensorVersioned *bounds_high,
                                         eindir_eval_fn eval_fn, eindir_grad_fn grad_fn,
                                         void *user_data, void (*free_fn)(void *));
void eindir_objective_free(eindir_objective_t *obj);
size_t eindir_objective_dim(const eindir_objective_t *obj);
eindir_status_t eindir_objective_bounds_low(const eindir_objective_t *obj,
                                            DLManagedTensorVersioned *out);
eindir_status_t eindir_objective_bounds_high(const eindir_objective_t *obj,
                                             DLManagedTensorVersioned *out);
eindir_status_t eindir_objective_eval(const eindir_objective_t *obj,
                                      const DLManagedTensorVersioned *x, double *value_out);
eindir_status_t eindir_objective_grad(const eindir_objective_t *obj,
                                      const DLManagedTensorVersioned *x,
                                      DLManagedTensorVersioned *grad_out);
int32_t eindir_objective_has_grad(const eindir_objective_t *obj);
#endif

static const double *in(const DLManagedTensorVersioned *t) {
    return (const double *)((const char *)t->dl_tensor.data + t->dl_tensor.byte_offset);
}

static double *out(DLManagedTensorVersioned *t) {
    return (double *)((char *)t->dl_tensor.data + t->dl_tensor.byte_offset);
}

static eindir_status_t rosen_eval(void *user, const DLManagedTensorVersioned *x, double *value) {
    const double *p = in(x);
    double a = p[0], b = p[1];
    (void)user;
    *value = 100.0 * (b - a * a) * (b - a * a) + (1.0 - a) * (1.0 - a);
    return EINDIR_SUCCESS;
}

static eindir_status_t rosen_grad(void *user, const DLManagedTensorVersioned *x,
                                  DLManagedTensorVersioned *g) {
    const double *p = in(x);
    double a = p[0], b = p[1];
    double *d = out(g);
    (void)user;
    d[0] = -400.0 * a * (b - a * a) + 2.0 * (a - 1.0);
    d[1] = 200.0 * (b - a * a);
    return EINDIR_SUCCESS;
}

static void mark_freed(void *user) { *(int *)user = 1; }

static int near(double got, double want) { return fabs(got - want) <= 1e-12 * (1.0 + fabs(want)); }

int main(void) {
    double low[2] = {-5.0, -5.0};
    double high[2] = {5.0, 5.0};
    double start[2] = {-1.2, 1.0};
    double grad[2] = {0.0, 0.0};
    double read_low[2] = {0.0, 0.0};
    double read_high[2] = {0.0, 0.0};
    double x[2] = {-1.2, 1.0};
    double value = 0.0;
    int freed = 0;
    const char *version = eindir_core_version();
    eindir_abi_stamp_t stamp = eindir_core_abi_stamp();
    eindir_abi_stamp_t newer = stamp;
    DLManagedTensorVersioned *tlow = rgmin_tensor_borrow_cpu_f64(low, 2);
    DLManagedTensorVersioned *thigh = rgmin_tensor_borrow_cpu_f64(high, 2);
    eindir_objective_t *objective;
    eindir_objective_t *value_only;
    DLManagedTensorVersioned *t;
    DLManagedTensorVersioned *tg;
    rgmin_control_t ctrl = rgmin_control_default();
    rgmin_report_t report;
    rgmin_status_t st;

    if (version == NULL || version[0] == '\0') {
        fprintf(stderr, "eindir_core_version is empty\n");
        return 2;
    }
    if (strcmp(eindir_core_abi_family(), "eindir.objective") != 0) {
        fprintf(stderr, "abi family %s\n", eindir_core_abi_family());
        return 3;
    }
    newer.abi_major += 1;
    if (!eindir_core_abi_compatible(&stamp) || eindir_core_abi_compatible(&newer) ||
        eindir_core_abi_compatible(NULL)) {
        fprintf(stderr, "abi stamp compatibility is wrong\n");
        return 4;
    }
    if (tlow == NULL || thigh == NULL) {
        return 5;
    }

    if (eindir_objective_new(3, tlow, thigh, rosen_eval, rosen_grad, NULL, NULL) != NULL ||
        strstr(eindir_last_error(), "length") == NULL) {
        fprintf(stderr, "short bounds accepted, last error: %s\n", eindir_last_error());
        return 6;
    }
    objective = eindir_objective_new(2, tlow, thigh, rosen_eval, rosen_grad, &freed, mark_freed);
    value_only = eindir_objective_new(2, tlow, thigh, rosen_eval, NULL, NULL, NULL);
    rgmin_tensor_free(tlow);
    rgmin_tensor_free(thigh);
    if (objective == NULL || value_only == NULL) {
        fprintf(stderr, "eindir_objective_new: %s\n", eindir_last_error());
        return 7;
    }

    if (eindir_objective_dim(objective) != 2 || eindir_objective_has_grad(objective) != 1 ||
        eindir_objective_has_grad(value_only) != 0) {
        fprintf(stderr, "dim or has_grad is wrong\n");
        return 8;
    }
    tlow = rgmin_tensor_borrow_cpu_f64(read_low, 2);
    thigh = rgmin_tensor_borrow_cpu_f64(read_high, 2);
    if (eindir_objective_bounds_low(objective, tlow) != EINDIR_SUCCESS ||
        eindir_objective_bounds_high(objective, thigh) != EINDIR_SUCCESS ||
        read_low[0] != -5.0 || read_low[1] != -5.0 || read_high[0] != 5.0 ||
        read_high[1] != 5.0) {
        fprintf(stderr, "bounds read back wrong\n");
        return 9;
    }
    rgmin_tensor_free(tlow);
    rgmin_tensor_free(thigh);

    t = rgmin_tensor_borrow_cpu_f64(start, 2);
    tg = rgmin_tensor_borrow_cpu_f64(grad, 2);
    if (eindir_objective_eval(objective, t, &value) != EINDIR_SUCCESS || !near(value, 24.2) ||
        eindir_objective_grad(objective, t, tg) != EINDIR_SUCCESS || !near(grad[0], -215.6) ||
        !near(grad[1], -88.0)) {
        fprintf(stderr, "eval %g grad %g %g\n", value, grad[0], grad[1]);
        return 10;
    }
    if (eindir_objective_grad(value_only, t, tg) != EINDIR_INVALID_PARAMETER ||
        strstr(eindir_last_error(), "no gradient") == NULL) {
        fprintf(stderr, "value-only grad, last error: %s\n", eindir_last_error());
        return 11;
    }
    rgmin_tensor_free(t);
    rgmin_tensor_free(tg);

    ctrl.maxiter = 2000;
    ctrl.gtol = 1e-8;
    t = rgmin_tensor_borrow_cpu_f64(x, 2);
    st = rgmin_minimize_eindir(objective, &stamp, t, &ctrl, RGMIN_LBFGS, &report);
    if (st != RGMIN_SUCCESS || report.stop != RGMIN_STOP_TOLERANCE ||
        fabs(x[0] - 1.0) > 1e-6 || fabs(x[1] - 1.0) > 1e-6) {
        fprintf(stderr, "minimize %d stop %d x %g %g: %s\n", (int)st, (int)report.stop, x[0],
                x[1], rgmin_last_error());
        rgmin_tensor_free(t);
        return 12;
    }
    if (rgmin_minimize_eindir(value_only, &stamp, t, &ctrl, RGMIN_LBFGS, &report) ==
            RGMIN_SUCCESS ||
        strstr(rgmin_last_error(), "no gradient") == NULL) {
        fprintf(stderr, "value-only objective accepted: %s\n", rgmin_last_error());
        rgmin_tensor_free(t);
        return 13;
    }
    rgmin_tensor_free(t);

    eindir_objective_free(value_only);
    eindir_objective_free(objective);
    if (!freed) {
        fprintf(stderr, "eindir_objective_free skipped the user destructor\n");
        return 14;
    }
    printf("eindir re-export ok: eindir-core %s\n", version);
    return 0;
}
