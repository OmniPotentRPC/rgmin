/* Thin libkrylov waist. String keys stay in this file. Public ABI is
 * rgmin_eigen_kind_t ordinal 15 only. */

#include "ckrylov.h"

#include <stdint.h>
#include <string.h>

typedef int (*rgmin_libkrylov_apply)(void *user, int64_t n, const double *v,
                                     double *hv);

typedef struct {
  rgmin_libkrylov_apply apply;
  void *user;
  int64_t n;
  int64_t actions;
} rgmin_libkrylov_ctx;

static _Thread_local rgmin_libkrylov_ctx *rgmin_libkrylov_tls;

static int_t rgmin_ckrylov_multiply(const int_t *full_dim,
                                    const int_t *subset_dim,
                                    const real_t *vectors, real_t *products) {
  rgmin_libkrylov_ctx *ctx = rgmin_libkrylov_tls;
  int_t n;
  int_t m;
  int_t i;
  if (ctx == NULL || ctx->apply == NULL || full_dim == NULL ||
      subset_dim == NULL || vectors == NULL || products == NULL) {
    return CKRYLOV_INVALID_INPUT;
  }
  n = *full_dim;
  m = *subset_dim;
  if (n != (int_t)ctx->n || n <= 0 || m <= 0) {
    return CKRYLOV_INVALID_DIMENSION;
  }
  for (i = 0; i < m; ++i) {
    if (ctx->apply(ctx->user, (int64_t)n, vectors + i * n, products + i * n) !=
        0) {
      return CKRYLOV_INVALID_INPUT;
    }
    ctx->actions += 1;
  }
  return CKRYLOV_OK;
}

int rgmin_libkrylov_lowest(int64_t n, const double *seed, int64_t nev,
                           int64_t maxit, int64_t krylov, double tol,
                           double *out_vec, double *out_value,
                           int64_t *out_actions, void *user,
                           rgmin_libkrylov_apply apply) {
  int_t error;
  int_t index;
  int_t full_dim;
  int_t solution_dim;
  int_t basis_dim;
  char kind[] = CKRYLOV_REAL_KIND;
  char structure[] = CKRYLOV_SYMMETRIC_STRUCTURE;
  char equation[] = CKRYLOV_EIGENVALUE_EQUATION;
  char precond_key[] = "preconditioner";
  char precond_val[] = "n";
  char ortho_key[] = "orthonormalizer";
  char ortho_val[] = "o";
  char maxit_key[] = "max_iterations";
  char tol_key[] = "max_residual_norm";
  rgmin_libkrylov_ctx ctx;
  rgmin_libkrylov_ctx *prev;

  if (n <= 0 || seed == NULL || out_vec == NULL || out_value == NULL ||
      apply == NULL) {
    return 1;
  }
  if (nev < 1) {
    nev = 1;
  }
  if (nev > n) {
    nev = n;
  }
  /* Initial basis is the seed only. A larger krylov would overrun `seed`. */
  (void)krylov;

  ctx.apply = apply;
  ctx.user = user;
  ctx.n = n;
  ctx.actions = 0;
  prev = rgmin_libkrylov_tls;
  rgmin_libkrylov_tls = &ctx;

  error = ckrylov_initialize();
  if (error != CKRYLOV_OK) {
    rgmin_libkrylov_tls = prev;
    return 3;
  }

  error = ckrylov_set_enum_option(precond_key, (int_t)strlen(precond_key),
                                  precond_val, (int_t)strlen(precond_val));
  if (error == CKRYLOV_OK) {
    error = ckrylov_set_enum_option(ortho_key, (int_t)strlen(ortho_key),
                                    ortho_val, (int_t)strlen(ortho_val));
  }
  if (error == CKRYLOV_OK && maxit > 0) {
    error = ckrylov_set_integer_option(maxit_key, (int_t)strlen(maxit_key),
                                       (int_t)maxit);
  }
  if (error == CKRYLOV_OK && tol > 0.0) {
    error = ckrylov_set_real_option(tol_key, (int_t)strlen(tol_key),
                                    (real_t)tol);
  }
  if (error != CKRYLOV_OK) {
    ckrylov_finalize();
    rgmin_libkrylov_tls = prev;
    return 3;
  }

  full_dim = (int_t)n;
  solution_dim = 1;
  basis_dim = 1;
  (void)nev;
  index = ckrylov_add_space(kind, (int_t)strlen(kind), structure,
                            (int_t)strlen(structure), equation,
                            (int_t)strlen(equation), full_dim, solution_dim,
                            basis_dim);
  if (index <= 0) {
    ckrylov_finalize();
    rgmin_libkrylov_tls = prev;
    return 3;
  }

  error = ckrylov_set_real_space_vectors(index, full_dim, basis_dim, seed);
  if (error != CKRYLOV_OK) {
    ckrylov_finalize();
    rgmin_libkrylov_tls = prev;
    return 3;
  }

  error = ckrylov_solve_real_equation(index, rgmin_ckrylov_multiply);
  if (error == CKRYLOV_OK) {
    error = ckrylov_get_space_eigenvalues(index, solution_dim, out_value);
  }
  if (error == CKRYLOV_OK) {
    /* Header declares the buffer const; the Fortran getter writes it. */
    error = ckrylov_get_real_space_solutions(index, full_dim, solution_dim,
                                             (const real_t *)out_vec);
  }

  if (out_actions != NULL) {
    *out_actions = ctx.actions;
  }
  ckrylov_finalize();
  rgmin_libkrylov_tls = prev;

  if (error == CKRYLOV_OK) {
    return 0;
  }
  if (error == CKRYLOV_NOT_CONVERGED || error == CKRYLOV_MAX_ITERATIONS_REACHED) {
    return 2;
  }
  return 3;
}
