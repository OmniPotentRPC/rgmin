// Calls the installed rgpot Lennard-Jones cluster potential.
// The metric neighbour list is linkcell in the Rust runner.

#include <cstddef>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"

extern "C" int rgpot_lj_force(std::size_t natoms, const double *pos,
                              const int *atmnrs, const double *box,
                              double *forces, double *energy) {
  static const rgpot::LJClusterPot pot{};
  rgpot::ForceInput in{
      .nAtoms = natoms, .pos = pos, .atmnrs = atmnrs, .box = box};
  rgpot::ForceOut out{.F = forces,
                      .energy = 0.0,
                      .variance = 0.0,
                      .stress = {},
                      .has_stress = 0};
  try {
    pot.forceImpl(in, &out);
  } catch (...) {
    return 1;
  }
  *energy = out.energy;
  return 0;
}
