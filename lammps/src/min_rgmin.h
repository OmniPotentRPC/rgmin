/* -*- c++ -*- ----------------------------------------------------------
   min_style rgmin: LAMMPS minimizer driven through the rgmin C ABI.
   SPDX-License-Identifier: MIT
------------------------------------------------------------------------- */

#ifndef LMP_MIN_RGMIN_H
#define LMP_MIN_RGMIN_H

#include "min.h"

#include <rgmin.h>

#include <exception>
#include <string>
#include <vector>

namespace LAMMPS_NS {

// Every rank holds the whole 3N coordinate vector, ordered by atom ID,
// and runs the same rgmin session on it. Each evaluation scatters the
// owned atoms' coordinates, calls Min::energy_force (collective) and
// sums the owned -f into a 3N gradient with one MPI_Allreduce. The
// session is deterministic, so identical inputs keep the replicas in
// lockstep; iterate() checks that after every step.
class MinRgmin : public Min {
 public:
  MinRgmin(class LAMMPS *);
  ~MinRgmin() override;

  void init() override;
  void setup_style() override;
  void reset_vectors() override;
  int iterate(int) override;
  void run(int) override;
  int modify_param(int, char **) override;
  double memory_usage() override;

 private:
  // min_modify settings
  rgmin_method_t method;
  std::string method_name;
  size_t lbfgs_memory;
  double trust;
  int wolfe;
  int accept;    // -1 keeps the rgmin default (energy)
  double istep;  // <= 0 keeps the per-method default

  // session state, rebuilt by setup_style() for every minimize command
  rgmin_solver_t *solver;
  DLManagedTensorVersioned *xtensor;
  size_t ndof;
  double escale;   // natoms when thermo_modify norm yes, else 1
  std::vector<double> xg;       // rgmin's iterate
  std::vector<double> xprev;    // iterate before the current step
  std::vector<double> xcache;   // point of the last evaluation
  std::vector<double> gcache;   // gradient at xcache
  std::vector<double> gbuf;     // owned part of a gradient before the sum
  double ecache;                // unnormalized energy at xcache
  bigint cache_step;            // timestep of the evaluation in atom->f
  bool cache_valid;
  std::exception_ptr pending;
  std::string stop_message;

  void parse_method(const char *);
  void start_session();
  void free_session();
  void gather_positions(double *);
  void gather_gradient(double *);
  void scatter_positions(const double *);
  double evaluate(const double *, double *);
  bool atoms_hold(const std::vector<double> &) const;
  void sync_atoms(bool forces);
  void check_replicas(double);
  static rgmin_status_t evalgrad(void *, const DLManagedTensorVersioned *, double *,
                                 DLManagedTensorVersioned *);
};

}    // namespace LAMMPS_NS

#endif
