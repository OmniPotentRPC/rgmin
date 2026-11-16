// One relaxation through librgmin and the rgpot shim.
//
// The start, the fmax stop, and the potential match bench/atomistic.
// Link this file against the commit under test. Prints:
// converged iters calls energy fmax wall_s

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/fortran/FortranPots.hpp"

#include "rgmin.h"

namespace {

struct Rng {
  uint64_t s;
  uint64_t next() {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return static_cast<double>(next() >> 11) / static_cast<double>(1ull << 53); }
  double normal() {
    double u1 = std::max(uniform(), 1e-300);
    double u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
  }
};

double mic(double dx, double length) { return dx - length * std::round(dx / length); }

struct Sys {
  int kind;
  int z;
  std::vector<double> pos;
  std::vector<char> fixed;
  double cell[9];
};

struct Job {
  Sys sys;
  std::vector<int> free;
  std::vector<double> x;
  std::vector<double> full;
  std::vector<int> z;
  int calls = 0;
  double last_fmax = 0;
  std::vector<double> cached_x;
  std::vector<double> cached_f;
  double cached_e = 0;
  bool have_cache = false;
};

const rgpot::LJClusterPot &lj_pot() {
  static const rgpot::LJClusterPot pot{};
  return pot;
}
const rgpot::MorsePot &morse_pot() {
  static const rgpot::MorsePot pot{};
  return pot;
}
const rgpot::fortranpots::EAMAlPot &eam_pot() {
  static const rgpot::fortranpots::EAMAlPot pot{};
  return pot;
}
rgpot::EMTPot *&emt_slot() {
  static rgpot::EMTPot *pot = new rgpot::EMTPot{};
  return pot;
}

void reset_emt() {
  delete emt_slot();
  emt_slot() = new rgpot::EMTPot{};
}

int force(Job *job, double *energy, double *forces) {
  rgpot::ForceInput in{.nAtoms = job->sys.pos.size() / 3,
                       .pos = job->full.data(),
                       .atmnrs = job->z.data(),
                       .box = job->sys.cell};
  rgpot::ForceOut out{.F = forces, .energy = 0, .variance = 0, .stress = {}, .has_stress = 0};
  try {
    switch (job->sys.kind) {
    case 0:
      lj_pot().forceImpl(in, &out);
      break;
    case 1:
      morse_pot().forceImpl(in, &out);
      break;
    case 2:
      eam_pot().forceImpl(in, &out);
      break;
    case 4:
      emt_slot()->forceImpl(in, &out);
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

void place(Job *job) {
  if (job->free.size() == job->full.size()) {
    std::copy(job->x.begin(), job->x.end(), job->full.begin());
    return;
  }
  for (size_t k = 0; k < job->free.size(); ++k) job->full[job->free[k]] = job->x[k];
}

double fmax_of(const double *g, size_t n) {
  double m = 0;
  for (size_t i = 0; i + 2 < n; i += 3) {
    double s = g[i] * g[i] + g[i + 1] * g[i + 1] + g[i + 2] * g[i + 2];
    m = std::max(m, std::sqrt(s));
  }
  return m;
}

const double *tensor_data(const DLManagedTensorVersioned *t, size_t *n) {
  const DLTensor &dl = t->dl_tensor;
  *n = static_cast<size_t>(*dl.shape);
  void *raw = dl.data;
  char *bytes = static_cast<char *>(raw);
  void *shifted = static_cast<void *>(bytes + dl.byte_offset);
  return static_cast<const double *>(shifted);
}

double *tensor_data_mut(DLManagedTensorVersioned *t, size_t *n) {
  DLTensor &dl = t->dl_tensor;
  *n = static_cast<size_t>(*dl.shape);
  void *raw = dl.data;
  char *bytes = static_cast<char *>(raw);
  void *shifted = static_cast<void *>(bytes + dl.byte_offset);
  return static_cast<double *>(shifted);
}

bool same_x(const Job *job, const double *p, size_t n) {
  if (!job->have_cache || job->cached_x.size() != n) return false;
  for (size_t i = 0; i < n; ++i) {
    if (job->cached_x[i] != p[i]) return false;
  }
  return true;
}

void evaluate(Job *job, const double *p, size_t n) {
  std::copy(p, p + n, job->x.begin());
  place(job);
  job->cached_f.assign(job->full.size(), 0.0);
  if (force(job, &job->cached_e, job->cached_f.data()) != 0) {
    job->have_cache = false;
    return;
  }
  job->cached_x.assign(p, p + n);
  job->have_cache = true;
  job->calls += 1;
}

rgmin_status_t eval_cb(void *user, const DLManagedTensorVersioned *x, double *value) {
  auto *job = static_cast<Job *>(user);
  size_t n = 0;
  const double *p = tensor_data(x, &n);
  if (!same_x(job, p, n)) evaluate(job, p, n);
  if (!job->have_cache) return RGMIN_INVALID_PARAMETER;
  *value = job->cached_e;
  return RGMIN_SUCCESS;
}

rgmin_status_t grad_cb(void *user, const DLManagedTensorVersioned *x, DLManagedTensorVersioned *g) {
  auto *job = static_cast<Job *>(user);
  size_t n = 0;
  const double *p = tensor_data(x, &n);
  if (!same_x(job, p, n)) evaluate(job, p, n);
  if (!job->have_cache) return RGMIN_INVALID_PARAMETER;
  size_t gn = 0;
  double *out = tensor_data_mut(g, &gn);
  for (size_t k = 0; k < job->free.size(); ++k) out[k] = -job->cached_f[job->free[k]];
  job->last_fmax = fmax_of(out, gn);
  return RGMIN_SUCCESS;
}

Sys lj_cluster(int n, double radius, uint64_t seed) {
  Rng rng{seed * 7919 + 17};
  std::vector<double> pos;
  while (static_cast<int>(pos.size()) < 3 * n) {
    double p[3] = {(2 * rng.uniform() - 1) * radius, (2 * rng.uniform() - 1) * radius,
                   (2 * rng.uniform() - 1) * radius};
    if (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] > radius * radius) continue;
    bool close = false;
    for (size_t i = 0; i < pos.size(); i += 3) {
      double d = 0;
      for (int k = 0; k < 3; ++k) {
        double c = pos[i + k] - p[k];
        d += c * c;
      }
      if (d < 0.81) {
        close = true;
        break;
      }
    }
    if (!close) pos.insert(pos.end(), p, p + 3);
  }
  Sys s;
  s.kind = 0;
  s.z = 18;
  s.pos = std::move(pos);
  s.fixed.assign(n, 0);
  double cell[9] = {100, 0, 0, 0, 100, 0, 0, 0, 100};
  std::copy(cell, cell + 9, s.cell);
  return s;
}

Sys packing(int kind, int z, double a, int ncell, double min_dist, uint64_t seed) {
  int n = ncell * ncell * ncell * 4;
  double length = ncell * a;
  Rng rng{seed * 7919 + 17};
  std::vector<double> placed;
  std::vector<double> pos(3 * n);
  double min2 = min_dist * min_dist;
  for (int i = 0; i < n; ++i) {
    bool got = false;
    for (int t = 0; t < 200000; ++t) {
      double trial[3] = {rng.uniform() * length, rng.uniform() * length, rng.uniform() * length};
      bool ok = true;
      for (size_t q = 0; q < placed.size(); q += 3) {
        double d2 = 0;
        for (int k = 0; k < 3; ++k) {
          double d = mic(trial[k] - placed[q + k], length);
          d2 += d * d;
        }
        if (d2 < min2) {
          ok = false;
          break;
        }
      }
      if (ok) {
        placed.insert(placed.end(), trial, trial + 3);
        pos[3 * i] = trial[0];
        pos[3 * i + 1] = trial[1];
        pos[3 * i + 2] = trial[2];
        got = true;
        break;
      }
    }
    if (!got) {
      std::fprintf(stderr, "packing stopped at %d\n", i);
      std::exit(1);
    }
  }
  Sys s;
  s.kind = kind;
  s.z = z;
  s.pos = std::move(pos);
  s.fixed.assign(n, 0);
  double cell[9] = {length, 0, 0, 0, length, 0, 0, 0, length};
  std::copy(cell, cell + 9, s.cell);
  return s;
}

Sys read_con(const std::string &path, int z, int kind) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot read coordinates\n");
    std::exit(1);
  }
  std::string line;
  std::vector<std::string> lines;
  while (std::getline(in, line)) lines.push_back(line);
  std::stringstream lens(lines[2]);
  double L[3];
  lens >> L[0] >> L[1] >> L[2];
  Sys s;
  s.kind = kind;
  s.z = z;
  for (size_t i = 10; i < lines.size(); ++i) {
    std::stringstream row(lines[i]);
    double x, y, zc;
    int fix, id;
    if (!(row >> x >> y >> zc >> fix >> id)) continue;
    s.pos.push_back(x);
    s.pos.push_back(y);
    s.pos.push_back(zc);
    s.fixed.push_back(fix == 1);
  }
  double cell[9] = {L[0], 0, 0, 0, L[1], 0, 0, 0, L[2]};
  std::copy(cell, cell + 9, s.cell);
  return s;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: relax_driver <lj13|lj38|lj75|pt7|cu|al> <lbfgs|fire|fire2|cg> <con-or-> <reps|session>\n");
    return 2;
  }
  const std::string system = argv[1];
  const std::string method = argv[2];
  const std::string con = argv[3];
  const int reps = std::atoi(argv[4]);

  Sys sys;
  double sigma = 0;
  double istep = 1;
  rgmin_method_t method_id = RGMIN_LBFGS;
  if (system == "lj13") sys = lj_cluster(13, 2.0, 0);
  else if (system == "lj38") sys = lj_cluster(38, 2.4, 0);
  else if (system == "lj75") sys = lj_cluster(75, 3.2, 0);
  else if (system == "pt7") {
    sys = read_con(con, 78, 1);
    sigma = 0.1;
  } else if (system == "cu") sys = packing(4, 29, 3.595, 6, 1.90, 0);
  else if (system == "al") sys = packing(2, 13, 4.038, 6, 2.15, 0);
  else {
    std::fprintf(stderr, "unknown system\n");
    return 2;
  }
  if (method == "fire2") {
    method_id = RGMIN_FIRE2;
    istep = 0.1;
  } else if (method == "fire") {
    method_id = RGMIN_FIRE;
    istep = 0.1;
  } else if (method == "cg") {
    method_id = RGMIN_POLAK_RIBIERE;
  } else if (method != "lbfgs") {
    std::fprintf(stderr, "unknown method\n");
    return 2;
  }

  Job job;
  job.sys = sys;
  int natoms = static_cast<int>(sys.fixed.size());
  job.full = sys.pos;
  job.z.assign(natoms, sys.z);
  for (int a = 0; a < natoms; ++a) {
    if (!sys.fixed[a]) {
      job.free.push_back(3 * a);
      job.free.push_back(3 * a + 1);
      job.free.push_back(3 * a + 2);
    }
  }
  std::vector<double> x0(job.free.size());
  Rng kick{3};
  for (size_t k = 0; k < job.free.size(); ++k) {
    double nrm = kick.normal();
    x0[k] = sys.pos[job.free[k]] + sigma * nrm;
  }

  const bool session = std::string(argv[4]) == "session";
  auto relax_once = [&]() -> int {
    reset_emt();
    job.x = x0;
    job.calls = 0;
    job.full = sys.pos;
    rgmin_control_t ctrl{};
    ctrl.maxiter = 1;
    ctrl.gtol = 0;
    ctrl.istep = istep;
    ctrl.memory = 0;
    ctrl.maxmove = 0;
    rgmin_solver_t *solver = rgmin_solver_create(method_id, &ctrl, job.x.size());
    if (!solver) {
      std::fprintf(stderr, "solver_create failed\n");
      return 2;
    }
    auto t0 = std::chrono::steady_clock::now();
    int iters = 0;
    double energy = 0;
    double fm = 1e300;
    int converged = 0;
    for (int it = 0; it < 8000; ++it) {
      DLManagedTensorVersioned *tensor = rgmin_tensor_borrow_cpu_f64(job.x.data(), job.x.size());
      rgmin_report_t rep_out{};
      rgmin_status_t st = rgmin_solver_step(solver, eval_cb, grad_cb, &job, tensor, &rep_out);
      size_t n = 0;
      const double *back = tensor_data(tensor, &n);
      std::copy(back, back + n, job.x.begin());
      rgmin_tensor_free(tensor);
      if (st != RGMIN_SUCCESS) {
        std::fprintf(stderr, "step failed\n");
        rgmin_solver_free(solver);
        return 2;
      }
      iters = it + 1;
      energy = rep_out.value;
      fm = job.last_fmax;
      if (fm < 0.01) {
        converged = 1;
        break;
      }
    }
    auto wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    rgmin_solver_free(solver);
    std::printf("%d %d %d %.10f %.3e %.6f\n", converged, iters, job.calls, energy, fm, wall);
    std::fflush(stdout);
    return converged ? 0 : 1;
  };

  if (session) {
    std::printf("ready\n");
    std::fflush(stdout);
    char line[64];
    while (std::fgets(line, sizeof line, stdin) != nullptr) {
      if (std::strncmp(line, "quit", 4) == 0) return 0;
      if (std::strncmp(line, "run", 3) == 0) {
        if (relax_once() == 2) return 1;
      }
    }
    return 0;
  }

  int failed = 0;
  for (int rep = 0; rep < reps; ++rep) {
    int rc = relax_once();
    if (rc == 2) return 1;
    if (rc != 0) failed = 1;
  }
  return failed;
}
