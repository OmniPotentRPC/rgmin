// Copyright (c) 2026 Rohit Goswami
// SPDX-License-Identifier: MIT
//
// No process-global mutable state. One atomic claim per solver. The GIL
// is dropped around the native step. Callback exceptions are captured
// and rethrown on the calling thread. Arrays must be float64.

#include <rgmin/optimize.hpp>
#include <rgbind/rgbind.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace rgbind {
template <>
struct status_traits<rgmin_status_t> {
    static int code(rgmin_status_t value) noexcept {
        switch (value) {
        case RGMIN_SUCCESS:
            return static_cast<int>(Status::ok);
        case RGMIN_INVALID_PARAMETER:
            return static_cast<int>(Status::build);
        case RGMIN_UNAVAILABLE:
            return static_cast<int>(Status::unavailable);
        default:
            return static_cast<int>(Status::other);
        }
    }
};
}  // namespace rgbind

namespace {

using Vector = nb::ndarray<double, nb::c_contig, nb::device::cpu>;
using ConstVector = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

template <class Array>
std::string shape_text(Array const& array) {
    std::ostringstream out;
    out << "(";
    for (std::size_t i = 0; i < static_cast<std::size_t>(array.ndim()); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << array.shape(i);
    }
    if (array.ndim() == 1) {
        out << ",";
    }
    out << ")";
    return out.str();
}

template <class Array>
void check_vector(Array const& array, std::size_t expected, char const* name) {
    if (array.ndim() != 1) {
        std::ostringstream message;
        message << name << ": expected shape rank 1, got rank " << array.ndim() << " "
                << shape_text(array);
        throw nb::value_error(message.str().c_str());
    }
    if (static_cast<std::size_t>(array.shape(0)) != expected) {
        std::ostringstream message;
        message << name << ": expected shape (" << expected << ",), got " << shape_text(array);
        throw nb::value_error(message.str().c_str());
    }
}

std::string method_name_from_code(int code) {
    switch (code) {
    case RGMIN_POLAK_RIBIERE:
        return "polak_ribiere";
    case RGMIN_FLETCHER_REEVES:
        return "fletcher_reeves";
    case RGMIN_BFGS:
        return "bfgs";
    case RGMIN_LBFGS:
        return "lbfgs";
    case RGMIN_SR1:
        return "sr1";
    case RGMIN_ADAM:
        return "adam";
    case RGMIN_STEEPEST:
        return "sd";
    case RGMIN_SR2:
        return "sr2";
    case RGMIN_PSO:
        return "pso";
    case RGMIN_HESTENES_STIEFEL:
        return "hestenes_stiefel";
    case RGMIN_DAI_YUAN:
        return "dai_yuan";
    case RGMIN_CONJUGATE_DESCENT:
        return "conjugate_descent";
    case RGMIN_HAGER_ZHANG:
        return "hager_zhang";
    case RGMIN_LIU_STOREY:
        return "liu_storey";
    case RGMIN_FR_PR:
        return "fr_pr";
    case RGMIN_NEWTON:
        return "newton";
    case RGMIN_RFO:
        return "rfo";
    case RGMIN_FIRE:
        return "fire";
    case RGMIN_BB:
        return "bb";
    case RGMIN_DOGLEG:
        return "dogleg";
    case RGMIN_FIRE2:
        return "fire2";
    case RGMIN_QUICKMIN:
        return "quickmin";
    case RGMIN_DIIS:
        return "diis";
    case RGMIN_ODE12R:
        return "ode12r";
    default:
        throw nb::value_error("unknown method");
    }
}

std::string method_name(nb::handle method) {
    if (!method || method.is_none()) {
        return "lbfgs";
    }
    if (nb::isinstance<nb::str>(method)) {
        return nb::cast<std::string>(method);
    }
    if (nb::isinstance<nb::int_>(method)) {
        return method_name_from_code(nb::cast<int>(method));
    }
    throw nb::type_error("method must be a string or Method");
}

struct Options {
    std::vector<std::pair<std::string, std::string>> text;
    std::vector<std::pair<std::string, double>> numbers;
    std::vector<std::pair<std::string, long long>> integers;
    std::vector<std::pair<std::string, bool>> flags;
};

Options parse_options(nb::kwargs const& kwargs) {
    Options out;
    for (auto item : kwargs) {
        std::string key = nb::cast<std::string>(item.first);
        nb::handle value = item.second;
        if (nb::isinstance<nb::bool_>(value)) {
            out.flags.emplace_back(key, nb::cast<bool>(value));
        } else if (nb::isinstance<nb::int_>(value)) {
            // nanobind 3.1 load_i64 returns false on overflow without
            // setting a Python error, so cast throws cast_error.
            long long integer = 0;
            try {
                integer = nb::cast<long long>(value);
            } catch (nb::cast_error const&) {
                throw nb::value_error((key + " is out of range").c_str());
            }
            out.integers.emplace_back(key, integer);
        } else if (nb::isinstance<nb::float_>(value)) {
            out.numbers.emplace_back(key, nb::cast<double>(value));
        } else if (nb::isinstance<nb::str>(value)) {
            out.text.emplace_back(key, nb::cast<std::string>(value));
        } else {
            throw nb::type_error(("option '" + key + "' has an unsupported type").c_str());
        }
    }
    return out;
}

rgmin::SolverConfig config_from(std::string const& method, Options const& options) {
    try {
        return rgmin::SolverConfig::build(method, options.text, options.numbers, options.integers,
                                          options.flags);
    } catch (std::invalid_argument const& error) {
        throw nb::value_error(error.what());
    }
}

void raise_for_status(rgmin_status_t status, rgbind::CallContext* context) {
    if (context != nullptr && context->has_error()) {
        context->rethrow_if_set();
    }
    if (status == RGMIN_SUCCESS) {
        return;
    }
    char const* raw = rgmin_last_error();
    std::string last = raw == nullptr || raw[0] == '\0' ? std::string("Solver step failed") : raw;
    // The C API reports every runtime failure as RGMIN_INVALID_PARAMETER.
    // The message is rgmin_last_error(), raised here as RuntimeError.
    throw std::runtime_error(last);
}

void construct_solver(rgmin::Solver* self, rgmin::SolverConfig const& config, std::size_t n) {
    auto* created = new (self) rgmin::Solver(config.method(), config.control(), n);
    try {
        config.apply(*created);
    } catch (std::invalid_argument const& error) {
        created->~Solver();
        throw nb::value_error(error.what());
    } catch (...) {
        created->~Solver();
        throw;
    }
}

nb::object owned_copy(double const* data, std::size_t n) {
    auto* copy = new double[n == 0 ? 1 : n];
    if (n != 0 && data != nullptr) {
        std::memcpy(copy, data, n * sizeof(double));
    }
    nb::capsule owner(copy, [](void* pointer) noexcept { delete[] static_cast<double*>(pointer); });
    nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> array(copy, {n}, owner);
    return nb::cast(std::move(array));
}

double const* tensor_f64(DLManagedTensorVersioned const* tensor) {
    return reinterpret_cast<double const*>(static_cast<char const*>(tensor->dl_tensor.data) +
                                           tensor->dl_tensor.byte_offset);
}

double* tensor_f64_mut(DLManagedTensorVersioned* tensor) {
    return reinterpret_cast<double*>(static_cast<char*>(tensor->dl_tensor.data) +
                                     tensor->dl_tensor.byte_offset);
}

rgmin_status_t map_trampoline(int code) noexcept {
    switch (static_cast<rgbind::TrampolineStatus>(code)) {
    case rgbind::TrampolineStatus::ok:
        return RGMIN_SUCCESS;
    case rgbind::TrampolineStatus::stopped:
    case rgbind::TrampolineStatus::python_error:
    case rgbind::TrampolineStatus::cxx_error:
    case rgbind::TrampolineStatus::bad_user:
        return RGMIN_INVALID_PARAMETER;
    case rgbind::TrampolineStatus::attach_failed:
        return RGMIN_INTERNAL_ERROR;
    }
    return RGMIN_INTERNAL_ERROR;
}

using EvalEntry = int (*)(void*, DLManagedTensorVersioned const*, double*, DLManagedTensorVersioned*);
using HessEntry = int (*)(void*, DLManagedTensorVersioned const*, DLManagedTensorVersioned*);

struct FgSlot {
    void* trampoline = nullptr;
    EvalEntry entry = nullptr;
};

struct HessSlot {
    FgSlot fg;
    void* hess_trampoline = nullptr;
    HessEntry hess_entry = nullptr;
};

extern "C" rgmin_status_t rgmin_py_evalgrad(void* user, DLManagedTensorVersioned const* coordinates,
                                            double* value, DLManagedTensorVersioned* gradient) {
    auto* slot = static_cast<FgSlot*>(user);
    if (slot == nullptr || slot->entry == nullptr) {
        return RGMIN_INVALID_PARAMETER;
    }
    return map_trampoline(slot->entry(slot->trampoline, coordinates, value, gradient));
}

extern "C" rgmin_status_t rgmin_py_evalgrad_hess(void* user, DLManagedTensorVersioned const* coordinates,
                                                 double* value, DLManagedTensorVersioned* gradient) {
    auto* slot = static_cast<HessSlot*>(user);
    if (slot == nullptr) {
        return RGMIN_INVALID_PARAMETER;
    }
    return rgmin_py_evalgrad(&slot->fg, coordinates, value, gradient);
}

extern "C" rgmin_status_t rgmin_py_hess(void* user, DLManagedTensorVersioned const* coordinates,
                                        DLManagedTensorVersioned* hessian) {
    auto* slot = static_cast<HessSlot*>(user);
    if (slot == nullptr || slot->hess_entry == nullptr) {
        return RGMIN_INVALID_PARAMETER;
    }
    return map_trampoline(slot->hess_entry(slot->hess_trampoline, coordinates, hessian));
}

void read_gradient(nb::handle gradient, std::size_t n, double* destination) {
    ConstVector grad = nb::cast<ConstVector>(gradient);
    check_vector(grad, n, "gradient");
    double const* source = grad.data();
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(source[i])) {
            throw rgbind::NumericalError("f_and_grad returned a non-finite gradient");
        }
    }
    std::memcpy(destination, source, n * sizeof(double));
}

void read_hessian(nb::handle value, std::size_t n, double* destination) {
    ConstVector hess = nb::cast<ConstVector>(value);
    std::size_t const nn = n * n;
    if (hess.ndim() == 1) {
        check_vector(hess, nn, "hessian");
    } else if (hess.ndim() == 2 && static_cast<std::size_t>(hess.shape(0)) == n &&
               static_cast<std::size_t>(hess.shape(1)) == n) {
        // Row-major n by n.
    } else {
        std::ostringstream message;
        message << "hessian: expected shape (" << n << ", " << n << ") or (" << nn
                << ",), got " << shape_text(hess);
        throw nb::value_error(message.str().c_str());
    }
    double const* source = hess.data();
    for (std::size_t i = 0; i < nn; ++i) {
        if (!std::isfinite(source[i])) {
            throw rgbind::NumericalError("hessian returned a non-finite entry");
        }
    }
    std::memcpy(destination, source, nn * sizeof(double));
}

rgmin::Report finish_step(rgmin_status_t status, rgmin_report_t const& out, rgbind::CallContext* context) {
    raise_for_status(status, context);
    if (!std::isfinite(out.value)) {
        throw rgbind::NumericalError("Solver produced a non-finite value");
    }
    return rgmin::Report{out.value, out.steps, out.grad_norm, out.stop};
}

DLManagedTensorVersioned* borrow_x(Vector x, std::size_t n) {
    DLManagedTensorVersioned* tensor = rgmin_tensor_borrow_cpu_f64(x.data(), n);
    if (tensor == nullptr) {
        char const* message = rgmin_last_error();
        throw nb::value_error(message != nullptr ? message : "rgmin_tensor_borrow_cpu_f64 failed");
    }
    return tensor;
}

rgmin::Report step_python(rgmin::Solver& solver, nb::callable function, Vector x) {
    std::size_t const n = solver.dim();
    check_vector(x, n, "x");
    rgbind::Claim claim(solver.busy(), "Solver");
    rgbind::CallContext context;
    auto callback = [&](DLManagedTensorVersioned const* coordinates, double* value,
                        DLManagedTensorVersioned* gradient) {
        if (coordinates == nullptr || gradient == nullptr || value == nullptr ||
            coordinates->dl_tensor.data == nullptr || gradient->dl_tensor.data == nullptr) {
            throw std::invalid_argument("f_and_grad received a null tensor");
        }
        nb::object point = owned_copy(tensor_f64(coordinates), n);
        nb::object result = function(point);
        if (!nb::isinstance<nb::tuple>(result) || nb::len(result) != 2) {
            throw nb::type_error("f_and_grad must return (value, gradient)");
        }
        double scalar = nb::cast<double>(result[0]);
        if (!std::isfinite(scalar)) {
            throw rgbind::NumericalError("f_and_grad returned a non-finite value");
        }
        *value = scalar;
        read_gradient(result[1], n, tensor_f64_mut(gradient));
    };
    using Callback = decltype(callback);
    rgbind::trampoline<Callback> tramp(std::move(callback), &context, true);
    FgSlot slot;
    slot.trampoline = &tramp;
    slot.entry = &rgbind::trampoline<Callback>::entry<DLManagedTensorVersioned const*, double*,
                                                      DLManagedTensorVersioned*>;
    DLManagedTensorVersioned* tensor = borrow_x(x, n);
    rgmin_report_t out{};
    rgmin_status_t status = RGMIN_SUCCESS;
    {
        nb::gil_scoped_release release;
        status = rgmin_solver_step_fg(solver.raw(), &rgmin_py_evalgrad, &slot, tensor, &out);
    }
    rgmin_tensor_free(tensor);
    return finish_step(status, out, &context);
}

rgmin::Report step_hess_python(rgmin::Solver& solver, nb::callable function, nb::callable hessian,
                               Vector x) {
    std::size_t const n = solver.dim();
    check_vector(x, n, "x");
    rgbind::Claim claim(solver.busy(), "Solver");
    rgbind::CallContext context;
    auto callback = [&](DLManagedTensorVersioned const* coordinates, double* value,
                        DLManagedTensorVersioned* gradient) {
        if (coordinates == nullptr || gradient == nullptr || value == nullptr ||
            coordinates->dl_tensor.data == nullptr || gradient->dl_tensor.data == nullptr) {
            throw std::invalid_argument("f_and_grad received a null tensor");
        }
        nb::object point = owned_copy(tensor_f64(coordinates), n);
        nb::object result = function(point);
        if (!nb::isinstance<nb::tuple>(result) || nb::len(result) != 2) {
            throw nb::type_error("f_and_grad must return (value, gradient)");
        }
        double scalar = nb::cast<double>(result[0]);
        if (!std::isfinite(scalar)) {
            throw rgbind::NumericalError("f_and_grad returned a non-finite value");
        }
        *value = scalar;
        read_gradient(result[1], n, tensor_f64_mut(gradient));
    };
    auto hess_callback = [&](DLManagedTensorVersioned const* coordinates,
                             DLManagedTensorVersioned* hessian_out) {
        if (coordinates == nullptr || hessian_out == nullptr ||
            coordinates->dl_tensor.data == nullptr || hessian_out->dl_tensor.data == nullptr) {
            throw std::invalid_argument("hessian received a null tensor");
        }
        nb::object point = owned_copy(tensor_f64(coordinates), n);
        nb::object result = hessian(point);
        read_hessian(result, n, tensor_f64_mut(hessian_out));
    };
    using Callback = decltype(callback);
    using HessCallback = decltype(hess_callback);
    rgbind::trampoline<Callback> tramp(std::move(callback), &context, true);
    rgbind::trampoline<HessCallback> hess_tramp(std::move(hess_callback), &context, true);
    HessSlot slot;
    slot.fg.trampoline = &tramp;
    slot.fg.entry = &rgbind::trampoline<Callback>::entry<DLManagedTensorVersioned const*, double*,
                                                         DLManagedTensorVersioned*>;
    slot.hess_trampoline = &hess_tramp;
    slot.hess_entry = &rgbind::trampoline<HessCallback>::entry<DLManagedTensorVersioned const*,
                                                               DLManagedTensorVersioned*>;
    DLManagedTensorVersioned* tensor = borrow_x(x, n);
    rgmin_report_t out{};
    rgmin_status_t status = RGMIN_SUCCESS;
    {
        nb::gil_scoped_release release;
        status = rgmin_solver_step_hess_fg(solver.raw(), &rgmin_py_evalgrad_hess, &rgmin_py_hess,
                                           &slot, tensor, &out);
    }
    rgmin_tensor_free(tensor);
    return finish_step(status, out, &context);
}

}  // namespace

NB_MODULE(_core, m) {
    m.attr("__version__") = rgmin_version();
    m.doc() = "Free-threaded bindings for the rgmin geometry optimizer.";
    rgbind::install_policy(m);

    nb::enum_<rgmin_method_t>(m, "Method", nb::is_arithmetic(),
                              "Geometry-optimization method. Members are UPPER_SNAKE. "
                              "CamelCase names are aliases of the same values.")
        .value("POLAK_RIBIERE", RGMIN_POLAK_RIBIERE)
        .value("FLETCHER_REEVES", RGMIN_FLETCHER_REEVES)
        .value("BFGS", RGMIN_BFGS)
        .value("LBFGS", RGMIN_LBFGS)
        .value("SR1", RGMIN_SR1)
        .value("ADAM", RGMIN_ADAM)
        .value("STEEPEST", RGMIN_STEEPEST)
        .value("SR2", RGMIN_SR2)
        .value("PSO", RGMIN_PSO)
        .value("HESTENES_STIEFEL", RGMIN_HESTENES_STIEFEL)
        .value("DAI_YUAN", RGMIN_DAI_YUAN)
        .value("CONJUGATE_DESCENT", RGMIN_CONJUGATE_DESCENT)
        .value("HAGER_ZHANG", RGMIN_HAGER_ZHANG)
        .value("LIU_STOREY", RGMIN_LIU_STOREY)
        .value("FR_PR", RGMIN_FR_PR)
        .value("NEWTON", RGMIN_NEWTON)
        .value("RFO", RGMIN_RFO)
        .value("FIRE", RGMIN_FIRE)
        .value("BB", RGMIN_BB)
        .value("DOGLEG", RGMIN_DOGLEG)
        .value("FIRE2", RGMIN_FIRE2)
        .value("QUICK_MIN", RGMIN_QUICKMIN)
        .value("DIIS", RGMIN_DIIS)
        .value("ODE12R", RGMIN_ODE12R)
        .value("PolakRibiere", RGMIN_POLAK_RIBIERE)
        .value("FletcherReeves", RGMIN_FLETCHER_REEVES)
        .value("Bfgs", RGMIN_BFGS)
        .value("Lbfgs", RGMIN_LBFGS)
        .value("Sr1", RGMIN_SR1)
        .value("Adam", RGMIN_ADAM)
        .value("Steepest", RGMIN_STEEPEST)
        .value("Sr2", RGMIN_SR2)
        .value("Pso", RGMIN_PSO)
        .value("HestenesStiefel", RGMIN_HESTENES_STIEFEL)
        .value("DaiYuan", RGMIN_DAI_YUAN)
        .value("ConjugateDescent", RGMIN_CONJUGATE_DESCENT)
        .value("HagerZhang", RGMIN_HAGER_ZHANG)
        .value("LiuStorey", RGMIN_LIU_STOREY)
        .value("FrPr", RGMIN_FR_PR)
        .value("Newton", RGMIN_NEWTON)
        .value("Rfo", RGMIN_RFO)
        .value("Fire", RGMIN_FIRE)
        .value("Bb", RGMIN_BB)
        .value("Dogleg", RGMIN_DOGLEG)
        .value("Fire2", RGMIN_FIRE2)
        .value("QuickMin", RGMIN_QUICKMIN)
        .value("Diis", RGMIN_DIIS)
        .value("Ode12r", RGMIN_ODE12R);

    nb::enum_<rgmin_stop_t>(m, "Stop", nb::is_arithmetic(),
                            "Why a step or a run ended. Members are UPPER_SNAKE. "
                            "CamelCase names are aliases of the same values.")
        .value("TOLERANCE", RGMIN_STOP_TOLERANCE)
        .value("NOISE_FLOOR", RGMIN_STOP_NOISE_FLOOR)
        .value("STEP_TOLERANCE", RGMIN_STOP_STEP_TOLERANCE)
        .value("BUDGET", RGMIN_STOP_BUDGET)
        .value("Tolerance", RGMIN_STOP_TOLERANCE)
        .value("NoiseFloor", RGMIN_STOP_NOISE_FLOOR)
        .value("StepTolerance", RGMIN_STOP_STEP_TOLERANCE)
        .value("Budget", RGMIN_STOP_BUDGET);

    nb::class_<rgmin::Report>(m, "Report", "Result of one optimizer step or of ``minimize``.")
        .def_prop_ro("value", [](rgmin::Report const& report) { return report.value; },
                     "Scalar value at the accepted point.")
        .def_prop_ro("steps", [](rgmin::Report const& report) { return report.steps; },
                     "Number of oracle evaluations used.")
        .def_prop_ro("grad_norm", [](rgmin::Report const& report) { return report.grad_norm; },
                     "Norm of the gradient.")
        .def_prop_ro("stop", [](rgmin::Report const& report) { return report.stop; },
                     "The ``Stop`` reason.");

    nb::class_<rgmin::SolverConfig>(m, "SolverConfig",
                                    "Validated, immutable optimizer options. Unknown keywords raise "
                                    "``ValueError`` listing the valid options. "
                                    "A Cartesian solver installs a per-atom cap of 0.2 "
                                    "(ASE maxstep, eOn max_move) unless the method already "
                                    "limits the step. BFGS, dogleg, and FIRE 2.0 start without "
                                    "that cap. ``maxstep`` or ``atom_maxmove`` replaces it on "
                                    "every method. A non-positive value clears it. A positive "
                                    "``maxmove`` is the Euclidean step limit and leaves the "
                                    "per-atom default off.")
        .def(
            "__init__",
            [](rgmin::SolverConfig* self, nb::object method, nb::kwargs kwargs) {
                new (self) rgmin::SolverConfig(config_from(method_name(method), parse_options(kwargs)));
            },
            nb::sig("def __init__(self, method: str | Method = 'lbfgs', **kwargs) -> None"))
        .def_prop_ro("method",
                     [](rgmin::SolverConfig const& config) {
                         return static_cast<rgmin_method_t>(config.method());
                     })
        .def_prop_ro("method_name",
                     [](rgmin::SolverConfig const& config) { return config.method_name(); })
        .def_static("valid_options", &rgmin::SolverConfig::valid_options)
        .def_static("valid_methods", &rgmin::SolverConfig::valid_methods);

    nb::class_<rgmin::Solver>(m, "Solver", "Move-only optimizer bound to one coordinate length.")
        .def(
            "__init__",
            [](rgmin::Solver* self, std::size_t n, rgmin::SolverConfig const& config) {
                construct_solver(self, config, n);
            },
            nb::arg("n"), nb::arg("config"))
        .def_prop_ro("dim", &rgmin::Solver::dim)
        .def_prop_ro("method",
                     [](rgmin::Solver const& solver) {
                         return static_cast<rgmin_method_t>(solver.method());
                     })
        .def("step", &step_python, nb::arg("f_and_grad"), nb::arg("x").noconvert(),
             nb::sig("def step(self, f_and_grad: collections.abc.Callable[[ndarray[dtype=float64]], "
                     "tuple[float, ndarray[dtype=float64]]], x: ndarray[dtype=float64]) -> Report"),
             "Take one step. ``f_and_grad(x)`` returns ``(value, gradient)``.")
        .def("step_hess", &step_hess_python, nb::arg("f_and_grad"), nb::arg("hessian"),
             nb::arg("x").noconvert(),
             nb::sig("def step_hess(self, f_and_grad: collections.abc.Callable[[ndarray[dtype=float64]], "
                     "tuple[float, ndarray[dtype=float64]]], hessian: collections.abc.Callable[["
                     "ndarray[dtype=float64]], ndarray[dtype=float64]], "
                     "x: ndarray[dtype=float64]) -> Report"),
             "One Newton, RFO, or dogleg step. ``hessian(x)`` returns an ``(n, n)`` matrix.");

    m.def(
        "minimize",
        [](nb::callable function, nb::handle x0_obj, nb::handle method, nb::kwargs kwargs) {
            Vector x0;
            try {
                x0 = nb::cast<Vector>(x0_obj, false);
            } catch (nb::cast_error const&) {
                throw nb::type_error("x0 must be a contiguous float64 array");
            }
            if (x0.ndim() != 1) {
                std::ostringstream message;
                message << "x0: expected shape rank 1, got rank " << x0.ndim() << " "
                        << shape_text(x0);
                throw nb::value_error(message.str().c_str());
            }
            rgmin::SolverConfig config = config_from(method_name(method), parse_options(kwargs));
            std::size_t const n = static_cast<std::size_t>(x0.shape(0));
            rgmin::Solver solver(config.method(), config.control(), n);
            try {
                config.apply(solver);
            } catch (std::invalid_argument const& error) {
                throw nb::value_error(error.what());
            }
            rgmin::Report report{};
            std::size_t const budget = config.control().maxiter;
            for (std::size_t i = 0; i < budget; ++i) {
                report = step_python(solver, function, x0);
                if (report.stop != RGMIN_STOP_BUDGET) {
                    break;
                }
            }
            return nb::make_tuple(report, x0);
        },
        nb::sig("def minimize(f_and_grad: collections.abc.Callable[[ndarray[dtype=float64]], "
                "tuple[float, ndarray[dtype=float64]]], x0: ndarray[dtype=float64], "
                "method: object, **kwargs) -> tuple[Report, ndarray[dtype=float64]]"),
        "Minimize ``f_and_grad`` in place and return ``(report, x)``.");
}
