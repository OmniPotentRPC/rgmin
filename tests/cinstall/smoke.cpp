#include <rgmin.h>

#include <cstdint>
#include <cstdio>

int main() {
    double x = 1.0;
    DLManagedTensorVersioned* tensor = rgmin_tensor_borrow_cpu_f64(&x, 1);
    rgmin_abi_stamp_t stamp = rgmin_abi_stamp();
    const char* version = rgmin_version();
    if (tensor == nullptr || rgmin_abi_compatible(&stamp) == 0 || version == nullptr) {
        return 1;
    }
    rgmin_tensor_free(tensor);

    // One bond, stretched by 0.02 and returned to length 1.
    const std::uint32_t pi[1] = {0};
    const std::uint32_t pj[1] = {1};
    const double unit[1] = {1.0};
    const double anchor[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    double moved[6] = {-0.01, 0.0, 0.0, 1.01, 0.0, 0.0};
    rgmin_constraints_t* set = rgmin_constraints_create(pi, pj, unit, 1);
    if (set == nullptr || rgmin_constraints_constrain(set, anchor, moved, 6) != RGMIN_SUCCESS ||
        !(rgmin_constraints_excess(set, moved, 6) <= 1.0) ||
        rgmin_constraints_create(pi, pi, unit, 1) != nullptr ||
        rgmin_last_constraint_error() != RGMIN_CONSTRAINT_MALFORMED) {
        rgmin_constraints_free(set);
        return 2;
    }
    rgmin_constraints_free(set);
    std::printf("%d.%d.%d %s\n", RGMIN_H_MAJOR, RGMIN_H_MINOR, RGMIN_H_PATCH, version);
    return 0;
}
