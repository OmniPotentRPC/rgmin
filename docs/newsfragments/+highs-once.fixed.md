HiGHS thread setup serializes through `std::sync::Once` instead of racing on the environment.
