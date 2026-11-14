#include <rgmin.h>

#include <stdint.h>
#include <stdio.h>

int main(void) {
    double x = 1.0;
    DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(&x, 1);
    rgmin_abi_stamp_t stamp = rgmin_abi_stamp();
    const char *version = rgmin_version();
    if (tensor == 0 || rgmin_abi_compatible(&stamp) == 0 || version == 0) {
        return 1;
    }
    rgmin_tensor_free(tensor);

    /* One bond, stretched by 0.02 and returned to length 1. */
    const uint32_t pi[1] = {0};
    const uint32_t pj[1] = {1};
    const double unit[1] = {1.0};
    const double anchor[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    double moved[6] = {-0.01, 0.0, 0.0, 1.01, 0.0, 0.0};
    rgmin_constraints_t *set = rgmin_constraints_create(pi, pj, unit, 1);
    if (set == 0 || rgmin_constraints_constrain(set, anchor, moved, 6) != RGMIN_SUCCESS ||
        !(rgmin_constraints_excess(set, moved, 6) <= 1.0) ||
        rgmin_constraints_create(pi, pi, unit, 1) != 0 ||
        rgmin_last_constraint_error() != RGMIN_CONSTRAINT_MALFORMED) {
        rgmin_constraints_free(set);
        return 2;
    }
    rgmin_constraints_free(set);

    /* One gradient step on St(2, 1) x SPD(1) stays on the product. */
    rgmin_manifold_handle_t *circle = rgmin_manifold_stiefel(2, 1, RGMIN_STIEFEL_QR);
    rgmin_manifold_handle_t *line = rgmin_manifold_spd(1, RGMIN_SPD_LOG_EUCLIDEAN);
    const rgmin_manifold_handle_t *factors[2] = {circle, line};
    rgmin_manifold_handle_t *both = rgmin_manifold_product(factors, 2);
    rgmin_manifold_free(circle);
    rgmin_manifold_free(line);
    const double point[3] = {1.0, 0.0, 2.0};
    const double egrad[3] = {0.0, -1.0, 1.0};
    double rgrad[3] = {0.0, 0.0, 0.0};
    double next[3] = {0.0, 0.0, 0.0};
    if (both == 0 || rgmin_manifold_len(both) != 3 ||
        rgmin_manifold_gradient_step(both, point, egrad, 0.5, rgrad, next, 3) != RGMIN_SUCCESS ||
        !(next[0] * next[0] + next[1] * next[1] - 1.0 < 1e-15) || !(next[2] > 0.0) ||
        !(next[2] < point[2]) || rgmin_manifold_stiefel(1, 2, RGMIN_STIEFEL_QR) != 0) {
        rgmin_manifold_free(both);
        return 3;
    }
    rgmin_manifold_free(both);
    printf("%d.%d.%d %s\n", RGMIN_H_MAJOR, RGMIN_H_MINOR, RGMIN_H_PATCH, version);
    return 0;
}
