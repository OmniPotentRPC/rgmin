// MIT License
//
// One C entry over the rgpot C++ potentials the atomistic benchmark
// drives: the free-boundary Lennard-Jones cluster, the platinum Morse
// pair potential, and the Fortran-backed embedded-atom aluminium.

#include <cstddef>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/fortran/FortranPots.hpp"

namespace {

const rgpot::LJClusterPot &lj() {
  static const rgpot::LJClusterPot pot{};
  return pot;
}

const rgpot::MorsePot &morse() {
  static const rgpot::MorsePot pot{};
  return pot;
}

const rgpot::fortranpots::EAMAlPot &eam_al() {
  static const rgpot::fortranpots::EAMAlPot pot{};
  return pot;
}

} // namespace

extern "C" int bench_force(int kind, std::size_t natoms, const double *pos,
                           const int *atmnrs, const double *box,
                           double *forces, double *energy) {
  rgpot::ForceInput in{
      .nAtoms = natoms, .pos = pos, .atmnrs = atmnrs, .box = box};
  rgpot::ForceOut out{.F = forces,
                      .energy = 0.0,
                      .variance = 0.0,
                      .stress = {},
                      .has_stress = 0};
  try {
    switch (kind) {
    case 0:
      lj().forceImpl(in, &out);
      break;
    case 1:
      morse().forceImpl(in, &out);
      break;
    case 2:
      eam_al().forceImpl(in, &out);
      break;
    default:
      return 2;
    }
  } catch (...) {
    return 1;
  }
  *energy = out.energy;
  return 0;
}
