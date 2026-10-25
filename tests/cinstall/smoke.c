#include <rgmin.h>

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
    printf("%d.%d.%d %s\n", RGMIN_H_MAJOR, RGMIN_H_MINOR, RGMIN_H_PATCH, version);
    return 0;
}
