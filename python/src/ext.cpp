// Copyright (c) 2026 Rohit Goswami
// SPDX-License-Identifier: BSD-3-Clause
//
// Free-threaded contract (rgbind F1-F9): no process-global mutable
// state, one atomic claim per solver, the GIL dropped around the native
// step, callback exceptions captured and rethrown on the calling thread,
// and strict float64 validation.

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
    std::vector<std::pair<std::string, bool>> flags;
};

Options parse_options(nb::kwargs const& kwargs) {
    Options out;
    for (auto item : kwargs) {
        std::string key = nb::cast<std::string>(item.first);
        nb::handle value = item.second;
        if (nb::isinstance<nb::bool_>(value)) {
            out.flags.emplace_back(key, nb::cast<bool>(value));
        } else if (nb::isinstance<nb::int_>(value) || nb::isinstance<nb::float_>(value)) {
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
        return rgmin::SolverConfig::build(method, options.text, options.numbers, options.flags);
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
    std::string last = raw == nullptr ? std::string{} : raw;
    std::string lower = last;
    for (char& character : lower) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    int code = rgbind::status_traits<rgmin_status_t>::code(status);
    if (lower.find("hessian") != std::string::npos || lower.find("not linked") != std::string::npos) {
        code = static_cast<int>(rgbind::Status::unavailable);
    } else if (lower.find("non-finite") != std::string::npos ||
               lower.find("singular") != std::string::npos ||
               lower.find("oracle:") != std::string::npos) {
        code = static_cast<int>(rgbind::Status::numerical);
    }
    rgbind::apply_status(rgbind::map_status_code(code, last, "Solver"), context);
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
        auto* data = reinterpret_cast<double*>(
            static_cast<char*>(const_cast<void*>(coordinates->dl_tensor.data)) +
            coordinates->dl_tensor.byte_offset);
        nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig> view(data, {n}, nb::handle());
        nb::object result = function(view);
        if (!nb::isinstance<nb::tuple>(result) || nb::len(result) != 2) {
            throw nb::value_error("f_and_grad must return (value, gradient)");
        }
        double scalar = nb::cast<double>(result[0]);
        if (!std::isfinite(scalar)) {
            throw rgbind::NumericalError("f_and_grad returned a non-finite value");
        }
        ConstVector grad = nb::cast<ConstVector>(result[1]);
        check_vector(grad, n, "gradient");
        double const* source = grad.data();
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(source[i])) {
                throw rgbind::NumericalError("f_and_grad returned a non-finite gradient");
            }
        }
        *value = scalar;
        auto* destination = reinterpret_cast<double*>(
            static_cast<char*>(gradient->dl_tensor.data) + gradient->dl_tensor.byte_offset);
        std::memmove(destination, source, n * sizeof(double));
    };
    using Callback = decltype(callback);
    rgbind::trampoline<Callback> tramp(std::move(callback), &context, true);
    using Entry = int (*)(void*, DLManagedTensorVersioned const*, double*, DLManagedTensorVersioned*);
    Entry entry = &rgbind::trampoline<Callback>::entry<DLManagedTensorVersioned const*, double*,
                                                       DLManagedTensorVersioned*>;
    DLManagedTensorVersioned* tensor = rgmin_tensor_borrow_cpu_f64(x.data(), n);
    if (tensor == nullptr) {
        char const* message = rgmin_last_error();
        throw nb::value_error(message != nullptr ? message : "rgmin_tensor_borrow_cpu_f64 failed");
    }
    rgmin_report_t out{};
    rgmin_status_t status = RGMIN_SUCCESS;
    {
        nb::gil_scoped_release release;
        status = rgmin_solver_step_fg(solver.raw(), reinterpret_cast<rgmin_evalgrad_fn>(entry),
                                      &tramp, tensor, &out);
    }
    rgmin_tensor_free(tensor);
    raise_for_status(status, &context);
    if (!std::isfinite(out.value)) {
        throw rgbind::NumericalError("Solver produced a non-finite value");
    }
    return rgmin::Report{out.value, out.steps, out.grad_norm, out.stop};
}

}  // namespace

NB_MODULE(_core, m) {
    m.attr("__version__") = rgmin_version();
    m.doc() = "Free-threaded bindings for the rgmin geometry optimizer.";
    rgbind::install_policy(m);

    nb::enum_<rgmin_method_t>(m, "Method", nb::is_arithmetic(),
                              "Geometry-optimization method. An ``enum.IntEnum`` of "
                              "``rgmin_method_t``.")
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
        .value("Diis", RGMIN_DIIS);

    nb::enum_<rgmin_stop_t>(m, "Stop", nb::is_arithmetic(), "Why a step or a run ended.")
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
                                    "``ValueError`` listing the valid options.")
        .def("__init__", [](rgmin::SolverConfig* self, nb::object method, nb::kwargs kwargs) {
            new (self) rgmin::SolverConfig(config_from(method_name(method), parse_options(kwargs)));
        })
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
             "Take one step. ``f_and_grad(x)`` returns ``(value, gradient)``.");

    m.def(
        "minimize",
        [](nb::callable function, nb::handle x0_obj, nb::handle method, nb::kwargs kwargs) {
            Vector x0 = nb::cast<Vector>(x0_obj);
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
            return report;
        },
        "Minimize ``f_and_grad`` in place, starting at ``x0``.");
}
