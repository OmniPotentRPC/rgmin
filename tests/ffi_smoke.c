#include <stdio.h>
#include "xts.h"

int main(void) {
    printf("%s\n", xts_version());
    xts_abi_stamp_t stamp = xts_abi_stamp();
    if (xts_abi_compatible(&stamp) == 0) {
        return 1;
    }
    if (stamp.abi_minor < 46) {
        return 2;
    }
    xts_control_t ctrl = xts_control_default();
    ctrl.maxiter = 1;
    ctrl.gtol = 1e-8;
    ctrl.istep = 0.1;
    ctrl.memory = 4;
    xts_solver_t *s = xts_solver_create(XTS_LBFGS, &ctrl, 2);
    if (s == NULL) {
        return 3;
    }
    double lo[2] = {-1.0, -2.0};
    double hi[2] = {1.0, 2.0};
    size_t idx[2] = {0, 1};
    double coef[2] = {1.0, -1.0};
    (void)xts_solver_set_highs(s, 1);
    (void)xts_solver_set_box(s, lo, hi, 2);
    (void)xts_solver_set_box(s, NULL, hi, 2);
    (void)xts_solver_set_trust(s, 0.25);
    (void)xts_solver_add_equality(s, idx, coef, 2, 0.0);
    (void)xts_solver_clear_equalities(s);
    xts_solver_free(s);
    return 0;
}
