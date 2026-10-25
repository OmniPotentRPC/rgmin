#include <rgmin.h>

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
    std::printf("%d.%d.%d %s\n", RGMIN_H_MAJOR, RGMIN_H_MINOR, RGMIN_H_PATCH, version);
    return 0;
}
