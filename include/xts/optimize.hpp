#pragma once

/**
 * \file xts/optimize.hpp
 * \brief Compatibility alias of rgmin/optimize.hpp.
 *
 * New C++ includes <rgmin/optimize.hpp> and uses namespace rgmin.
 */
#include "../rgmin/optimize.hpp"

namespace xts {
namespace optimize {

using rgmin::version;
using rgmin::Method;
using rgmin::method_from_name;
using rgmin::Control;
using rgmin::Report;
using rgmin::ScalarType;
using rgmin::OptimizeControl;
using rgmin::OptimizeResult;
using rgmin::minimize_fn;
using rgmin::minimize_hess_fn;
using rgmin::minimize_eindir;
using rgmin::optimize;
using rgmin::borrow_cpu_f64;
using rgmin::lowest_eigenpairs;
using rgmin::lowest_eigenpairs_ex;
using rgmin::Solver;
using rgmin::ConstraintError;
using rgmin::ConstraintSolver;
using rgmin::DistanceConstraints;
using rgmin::LbfgsPairs;

namespace minimize = rgmin::minimize;
namespace nlcg = rgmin::nlcg;

}  // namespace optimize
}  // namespace xts
