/* ----------------------------------------------------------------------
   min_style rgmin: LAMMPS minimizer driven through the rgmin C ABI.
   SPDX-License-Identifier: MIT
------------------------------------------------------------------------- */

#include "min_rgmin.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "output.h"
#include "thermo.h"
#include "timer.h"
#include "update.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

// Same relative-energy floor as min_cg, min_fire and the other LAMMPS styles.
static constexpr double EPS_ENERGY = 1.0e-8;

// LAMMPS min_style fire defaults (Guenole et al. 2020, Table 1), which
// rgmin's RGMIN_FIRE_PRESET_GUENOLE2020 also holds.
static constexpr int FIRE_DELAYSTEP = 20;
static constexpr double FIRE_DTGROW = 1.1;
static constexpr double FIRE_DTSHRINK = 0.5;
static constexpr double FIRE_ALPHA0 = 0.25;
static constexpr double FIRE_ALPHASHRINK = 0.99;
static constexpr double FIRE_TMAX = 10.0;
static constexpr double FIRE_TMIN = 0.02;
static constexpr int FIRE_VDFMAX = 2000;

// The rgmin FIRE step scale (time step) when min_modify istep is not given.
static constexpr double FIRE_ISTEP = 0.1;

static double *tensor_data(const DLManagedTensorVersioned *t)
{
  return reinterpret_cast<double *>(static_cast<char *>(t->dl_tensor.data) +
                                    t->dl_tensor.byte_offset);
}

static bool is_fire(rgmin_method_t m)
{
  return m == RGMIN_FIRE || m == RGMIN_FIRE2;
}

/* ---------------------------------------------------------------------- */

MinRgmin::MinRgmin(LAMMPS *lmp) :
    Min(lmp), method(RGMIN_LBFGS), method_name("lbfgs"), lbfgs_memory(10), trust(0.0), wolfe(0),
    accept(-1), istep(0.0), solver(nullptr), xtensor(nullptr), ndof(0), escale(1.0), ecache(0.0),
    cache_step(-1), cache_valid(false)
{
  searchflag = 1;
}

/* ---------------------------------------------------------------------- */

MinRgmin::~MinRgmin()
{
  free_session();
}

/* ---------------------------------------------------------------------- */

void MinRgmin::free_session()
{
  if (xtensor) rgmin_tensor_free(xtensor);
  if (solver) rgmin_solver_free(solver);
  xtensor = nullptr;
  solver = nullptr;
  cache_valid = false;
  cache_step = -1;
}

/* ---------------------------------------------------------------------- */

void MinRgmin::init()
{
  Min::init();
  free_session();
}

/* ---------------------------------------------------------------------- */

void MinRgmin::setup_style()
{
  if (nextra_global)
    error->all(FLERR,
               "min_style rgmin does not support fixes that add global degrees of freedom "
               "such as fix box/relax; use min_style cg for cell relaxations");
  if (nextra_atom)
    error->all(FLERR,
               "min_style rgmin does not support pair styles that add per-atom degrees of "
               "freedom (for example electron radii)");
  if (atom->tag_enable == 0) error->all(FLERR, "min_style rgmin requires atom IDs");
  if (atom->tag_consecutive() == 0)
    error->all(FLERR, "min_style rgmin requires consecutive atom IDs 1 to N; use reset_atoms id");
  if (3 * atom->natoms > INT_MAX)
    error->all(FLERR, "min_style rgmin supports at most {} atoms", INT_MAX / 3);

  if (linestyle != QUADRATIC)
    error->all(FLERR,
               "min_style rgmin does not take min_modify line; use min_modify linesearch "
               "none or wolfe");
  if (delaystep != FIRE_DELAYSTEP || dtgrow != FIRE_DTGROW || dtshrink != FIRE_DTSHRINK ||
      alpha0 != FIRE_ALPHA0 || alphashrink != FIRE_ALPHASHRINK || tmax != FIRE_TMAX ||
      tmin != FIRE_TMIN || integrator != EULERIMPLICIT || halfstepback_flag != 1 ||
      delaystep_start_flag != 1 || max_vdotf_negatif != FIRE_VDFMAX)
    error->all(FLERR,
               "min_style rgmin keeps the Guenole 2020 FIRE parameters; min_modify delaystep, "
               "dtgrow, dtshrink, alpha0, alphashrink, tmax, tmin, integrator, halfstepback, "
               "initialdelay and vdfmax must stay at their defaults");
  if (abcflag && method != RGMIN_FIRE2)
    error->all(FLERR, "min_modify abcfire yes needs min_modify method fire2 with min_style rgmin");
  if (wolfe && method != RGMIN_LBFGS)
    error->all(FLERR, "min_modify linesearch wolfe needs min_modify method lbfgs");
  if (trust > 0.0 && method != RGMIN_LBFGS)
    error->all(FLERR, "min_modify trust needs min_modify method lbfgs");
  if (wolfe && trust > 0.0)
    error->all(FLERR, "min_modify trust and linesearch wolfe are exclusive");

  free_session();
}

/* ----------------------------------------------------------------------
   Min::fnorm_sqr() and friends read the per-rank force through xvec/fvec
------------------------------------------------------------------------- */

void MinRgmin::reset_vectors()
{
  nvec = 3 * atom->nlocal;
  if (nvec) xvec = atom->x[0];
  if (nvec) fvec = atom->f[0];
}

/* ---------------------------------------------------------------------- */

void MinRgmin::parse_method(const char *name)
{
  rgmin_method_t m;
  if (rgmin_method_from_name(name, &m) != RGMIN_SUCCESS)
    error->all(FLERR, "Unknown rgmin method {} in min_modify method", name);
  switch (m) {
    case RGMIN_LBFGS:
    case RGMIN_FIRE:
    case RGMIN_FIRE2:
    case RGMIN_STEEPEST:
    case RGMIN_BB:
      break;
    case RGMIN_BFGS:
    case RGMIN_SR1:
    case RGMIN_SR2:
      error->all(FLERR, "rgmin method {} stores a dense 3N x 3N matrix; use lbfgs", name);
      break;
    case RGMIN_NEWTON:
    case RGMIN_RFO:
    case RGMIN_DOGLEG:
      error->all(FLERR, "rgmin method {} needs a dense Hessian, which LAMMPS does not supply",
                 name);
      break;
    default:
      error->all(FLERR,
                 "rgmin method {} is not available with min_style rgmin; use lbfgs, fire, "
                 "fire2, sd or bb",
                 name);
      break;
  }
  method = m;
  method_name = name;
}

/* ---------------------------------------------------------------------- */

int MinRgmin::modify_param(int narg, char **arg)
{
  const std::string key = arg[0];
  if (key != "method" && key != "memory" && key != "trust" && key != "linesearch" &&
      key != "accept" && key != "istep")
    error->all(FLERR, "Unknown min_modify keyword {} for min_style rgmin", key);
  if (narg < 2) error->all(FLERR, "min_modify {} needs a value", key);
  const std::string val = arg[1];

  if (key == "method") {
    parse_method(arg[1]);
  } else if (key == "memory") {
    const int m = utils::inumeric(FLERR, arg[1], false, lmp);
    if (m < 1) error->all(FLERR, "min_modify memory must be at least 1");
    lbfgs_memory = static_cast<size_t>(m);
  } else if (key == "trust") {
    const double r = utils::numeric(FLERR, arg[1], false, lmp);
    if (!(r >= 0.0) || !std::isfinite(r))
      error->all(FLERR, "min_modify trust must be a finite non-negative radius");
    trust = r;
  } else if (key == "linesearch") {
    if (val == "none") wolfe = 0;
    else if (val == "wolfe") wolfe = 1;
    else error->all(FLERR, "min_modify linesearch is none or wolfe, not {}", val);
  } else if (key == "accept") {
    if (val == "default") accept = -1;
    else if (val == "none") accept = RGMIN_ACCEPT_NONE;
    else if (val == "energy") accept = RGMIN_ACCEPT_ENERGY;
    else if (val == "nonmonotone") accept = RGMIN_ACCEPT_NONMONOTONE;
    else error->all(FLERR, "min_modify accept is default, none, energy or nonmonotone, not {}", val);
  } else if (key == "istep") {
    const double s = utils::numeric(FLERR, arg[1], false, lmp);
    if (!(s >= 0.0) || !std::isfinite(s))
      error->all(FLERR, "min_modify istep must be finite and non-negative (0 is the default)");
    istep = s;
  }
  return 2;
}

/* ---------------------------------------------------------------------- */

void MinRgmin::start_session()
{
  ndof = 3 * static_cast<size_t>(atom->natoms);
  escale = output->thermo->normflag ? static_cast<double>(atom->natoms) : 1.0;
  xg.assign(ndof, 0.0);
  xprev.assign(ndof, 0.0);
  xcache.assign(ndof, 0.0);
  gcache.assign(ndof, 0.0);
  gbuf.assign(ndof, 0.0);

  gather_positions(xg.data());
  // Min::setup already evaluated the start, fixes included; hand that to
  // the first rgmin call instead of evaluating again.
  gather_gradient(gcache.data());
  xcache = xg;
  double e = ecurrent;
  MPI_Bcast(&e, 1, MPI_DOUBLE, 0, world);
  ecache = e * escale;
  cache_step = update->ntimestep;
  cache_valid = true;

  rgmin_control_t ctrl = rgmin_control_default();
  ctrl.maxiter = static_cast<size_t>(std::max(update->nsteps, 1));
  ctrl.gtol = update->ftol > 0.0 ? update->ftol : 0.0;
  ctrl.istep = istep > 0.0 ? istep : (is_fire(method) ? FIRE_ISTEP : 0.0);
  ctrl.memory = lbfgs_memory;

  solver = rgmin_solver_create(method, &ctrl, ndof);
  if (!solver) {
    const char *why = rgmin_last_error();
    error->all(FLERR, "rgmin_solver_create failed: {}", why ? why : "unknown error");
  }

  rgmin_stop_norm_t norm = RGMIN_STOP_NORM_TWO;
  if (normstyle == MAX) norm = RGMIN_STOP_NORM_MAX_ATOM;
  else if (normstyle == INF) norm = RGMIN_STOP_NORM_INF;
  if (rgmin_solver_set_stop_norm(solver, norm) != 0)
    error->all(FLERR, "rgmin_solver_set_stop_norm failed: {}", rgmin_last_error());

  if (accept >= 0) rgmin_solver_set_accept(solver, static_cast<rgmin_accept_t>(accept));
  if (dmax > 0.0) rgmin_solver_set_atom_maxmove(solver, dmax);
  if (wolfe && rgmin_solver_set_lbfgs_wolfe(solver) != 0)
    error->all(FLERR, "rgmin_solver_set_lbfgs_wolfe failed: {}", rgmin_last_error());
  if (trust > 0.0 && rgmin_solver_set_lbfgs_trust(solver, trust) != 0)
    error->all(FLERR, "rgmin_solver_set_lbfgs_trust failed: {}", rgmin_last_error());
  if (method == RGMIN_FIRE2) {
    if (rgmin_solver_set_fire_preset(solver, RGMIN_FIRE_PRESET_GUENOLE2020) != 0)
      error->all(FLERR, "rgmin_solver_set_fire_preset failed: {}", rgmin_last_error());
    if (abcflag && rgmin_solver_set_fire_abc(solver, 1) != 0)
      error->all(FLERR, "rgmin_solver_set_fire_abc failed: {}", rgmin_last_error());
  }

  xtensor = rgmin_tensor_borrow_cpu_f64(xg.data(), ndof);
  if (!xtensor) error->all(FLERR, "rgmin_tensor_borrow_cpu_f64 failed: {}", rgmin_last_error());

  if (comm->me == 0)
    utils::logmesg(lmp,
                   "rgmin {} (C ABI {}.{}): method {}, {} coordinates on each of {} ranks, "
                   "dmax {}, memory {}\n",
                   rgmin_version(), RGMIN_ABI_VERSION_MAJOR, RGMIN_ABI_VERSION_MINOR,
                   method_name, ndof, comm->nprocs, dmax, lbfgs_memory);
}

/* ----------------------------------------------------------------------
   sum per-rank pieces of a 3N vector indexed by atom ID; every entry has
   one owner, so the sum adds zeros and is exact and identical on all ranks
------------------------------------------------------------------------- */

void MinRgmin::gather_positions(double *out)
{
  std::fill(gbuf.begin(), gbuf.end(), 0.0);
  double **x = atom->x;
  tagint *tag = atom->tag;
  for (int i = 0; i < atom->nlocal; i++) {
    const size_t k = 3 * static_cast<size_t>(tag[i] - 1);
    gbuf[k] = x[i][0];
    gbuf[k + 1] = x[i][1];
    gbuf[k + 2] = x[i][2];
  }
  MPI_Allreduce(gbuf.data(), out, static_cast<int>(ndof), MPI_DOUBLE, MPI_SUM, world);
}

void MinRgmin::gather_gradient(double *out)
{
  timer->stamp();
  std::fill(gbuf.begin(), gbuf.end(), 0.0);
  double **f = atom->f;
  tagint *tag = atom->tag;
  for (int i = 0; i < atom->nlocal; i++) {
    const size_t k = 3 * static_cast<size_t>(tag[i] - 1);
    gbuf[k] = -f[i][0];
    gbuf[k + 1] = -f[i][1];
    gbuf[k + 2] = -f[i][2];
  }
  MPI_Allreduce(gbuf.data(), out, static_cast<int>(ndof), MPI_DOUBLE, MPI_SUM, world);
  timer->stamp(Timer::COMM);
}

/* ----------------------------------------------------------------------
   move owned atoms to the periodic image of the rgmin point nearest to
   where they are, so wrapping and image flags stay with LAMMPS
------------------------------------------------------------------------- */

void MinRgmin::scatter_positions(const double *xnew)
{
  double **x = atom->x;
  tagint *tag = atom->tag;
  double img[3];
  for (int i = 0; i < atom->nlocal; i++) {
    const size_t k = 3 * static_cast<size_t>(tag[i] - 1);
    domain->closest_image(x[i], &xnew[k], img);
    x[i][0] = img[0];
    x[i][1] = img[1];
    x[i][2] = img[2];
  }
}

/* ----------------------------------------------------------------------
   energy and gradient at xnew; collective over all ranks
------------------------------------------------------------------------- */

double MinRgmin::evaluate(const double *xnew, double *grad)
{
  if (!(cache_valid && std::equal(xcache.begin(), xcache.end(), xnew))) {
    scatter_positions(xnew);
    double e = energy_force(0);
    neval++;
    // thermo_pe sums with MPI_Allreduce; take one rank's bits so the
    // replicated sessions cannot part on the last digit.
    MPI_Bcast(&e, 1, MPI_DOUBLE, 0, world);
    gather_gradient(gcache.data());
    std::copy(xnew, xnew + ndof, xcache.begin());
    ecache = e * escale;
    cache_step = update->ntimestep;
    cache_valid = true;
  }
  std::copy(gcache.begin(), gcache.end(), grad);
  return ecache;
}

rgmin_status_t MinRgmin::evalgrad(void *user, const DLManagedTensorVersioned *x, double *value,
                                  DLManagedTensorVersioned *grad)
{
  auto *self = static_cast<MinRgmin *>(user);
  // A LAMMPS error must not unwind through the Rust frames; it is
  // rethrown once rgmin_solver_step_fg returns.
  if (self->pending) return RGMIN_INTERNAL_ERROR;
  try {
    *value = self->evaluate(tensor_data(x), tensor_data(grad));
    return RGMIN_SUCCESS;
  } catch (...) {
    self->pending = std::current_exception();
    return RGMIN_INTERNAL_ERROR;
  }
}

/* ---------------------------------------------------------------------- */

bool MinRgmin::atoms_hold(const std::vector<double> &v) const
{
  return cache_valid && std::equal(xcache.begin(), xcache.end(), v.begin());
}

// Put the atoms at rgmin's iterate. With forces, also make atom->f and
// the tallied energy belong to the current timestep, as output needs.
void MinRgmin::sync_atoms(bool forces)
{
  if (atoms_hold(xg) && (!forces || cache_step == update->ntimestep)) return;
  if (forces) {
    cache_valid = false;
    evaluate(xg.data(), gbuf.data());
  } else {
    scatter_positions(xg.data());
    cache_valid = false;
  }
}

void MinRgmin::check_replicas(double value)
{
  if (comm->nprocs == 1 || !std::isfinite(value)) return;
  double in[2] = {value, -value}, out[2];
  MPI_Allreduce(in, out, 2, MPI_DOUBLE, MPI_MAX, world);
  if (out[0] != value || -out[1] != value)
    error->all(FLERR, "min_style rgmin: the replicated rgmin sessions diverged across ranks");
}

/* ---------------------------------------------------------------------- */

int MinRgmin::iterate(int maxiter)
{
  if (!solver) start_session();

  for (int iter = 0; iter < maxiter; iter++) {
    if (timer->check_timeout(niter)) {
      sync_atoms(false);
      return TIMEOUT;
    }

    const bigint ntimestep = ++update->ntimestep;
    niter++;

    eprevious = ecurrent;
    std::copy(xg.begin(), xg.end(), xprev.begin());

    rgmin_report_t rep;
    const rgmin_status_t status = rgmin_solver_step_fg(solver, &evalgrad, this, xtensor, &rep);
    if (pending) {
      std::exception_ptr e = pending;
      pending = nullptr;
      std::rethrow_exception(e);
    }
    if (status != RGMIN_SUCCESS) {
      const char *why = rgmin_last_error();
      stop_message = fmt::format("rgmin step failed: {}", why ? why : "unknown error");
      if (comm->me == 0) error->warning(FLERR, stop_message);
      sync_atoms(false);
      return INTERROR;
    }
    check_replicas(rep.value);
    ecurrent = rep.value / escale;
    const bool moved = !std::equal(xg.begin(), xg.end(), xprev.begin());

    if (neval >= update->max_eval) {
      sync_atoms(false);
      return MAXEVAL;
    }

    // min_cg's test, skipped when the session refused its trial and stayed put
    if (moved &&
        fabs(ecurrent - eprevious) <
            update->etol * 0.5 * (fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY)) {
      sync_atoms(false);
      return ETOL;
    }

    // rep.grad_norm is the LAMMPS force norm of min_modify norm (two, max
    // or inf) over the full 3N force after fixes
    if (update->ftol > 0.0 && rep.grad_norm < update->ftol) {
      sync_atoms(false);
      return FTOL;
    }

    if (output->next == ntimestep) {
      sync_atoms(true);
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  sync_atoms(true);
  return MAXITER;
}

/* ---------------------------------------------------------------------- */

void MinRgmin::run(int n)
{
  stop_message.clear();
  Min::run(n);
  if (!stop_message.empty()) stopstr = const_cast<char *>(stop_message.c_str());
}

/* ---------------------------------------------------------------------- */

double MinRgmin::memory_usage()
{
  double bytes = 5.0 * ndof * sizeof(double);
  if (method == RGMIN_LBFGS) bytes += 2.0 * lbfgs_memory * ndof * sizeof(double);
  return bytes;
}
