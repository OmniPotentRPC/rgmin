/* Thin PRIMME waist. Typed primme_params fields only. No string keys. */

#include "primme.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int (*rgmin_primme_apply)(void *user, int64_t n, const double *v,
                                  double *hv);

typedef struct {
  rgmin_primme_apply apply;
  rgmin_primme_apply precond;
  void *user;
  int64_t actions;
} rgmin_primme_ctx;

static void rgmin_primme_matvec(void *x, PRIMME_INT *ldx, void *y,
                                PRIMME_INT *ldy, int *blockSize,
                                primme_params *primme, int *ierr) {
  rgmin_primme_ctx *ctx = (rgmin_primme_ctx *)primme->matrix;
  const double *xin = (const double *)x;
  double *yout = (double *)y;
  int b = *blockSize;
  int64_t n = primme->n;
  int i;
  *ierr = 0;
  if (ctx == NULL || ctx->apply == NULL) {
    *ierr = 1;
    return;
  }
  for (i = 0; i < b; i++) {
    if (ctx->apply(ctx->user, n, xin + i * (*ldx), yout + i * (*ldy)) != 0) {
      *ierr = 1;
      return;
    }
    ctx->actions += 1;
  }
}

static void rgmin_primme_precond(void *x, PRIMME_INT *ldx, void *y,
                                 PRIMME_INT *ldy, int *blockSize,
                                 primme_params *primme, int *ierr) {
  rgmin_primme_ctx *ctx = (rgmin_primme_ctx *)primme->matrix;
  const double *xin = (const double *)x;
  double *yout = (double *)y;
  int b = *blockSize;
  int64_t n = primme->n;
  int i;
  *ierr = 0;
  if (ctx == NULL || ctx->precond == NULL) {
    *ierr = 1;
    return;
  }
  for (i = 0; i < b; i++) {
    if (ctx->precond(ctx->user, n, xin + i * (*ldx), yout + i * (*ldy)) != 0) {
      *ierr = 1;
      return;
    }
  }
}

int rgmin_primme_lowest(int64_t n, const double *seed, int64_t nev,
                        int64_t maxit, double tol, double *out_vec,
                        double *out_value, int64_t *out_actions, void *user,
                        rgmin_primme_apply apply, rgmin_primme_apply precond) {
  primme_params primme;
  double *evals = NULL;
  double *evecs = NULL;
  double *rnorms = NULL;
  int err;
  rgmin_primme_ctx ctx;

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

  primme_initialize(&primme);
  primme.n = (PRIMME_INT)n;
  primme.numEvals = (int)nev;
  primme.target = primme_smallest;
  primme.eps = tol > 0.0 ? tol : 1e-8;
  if (maxit > 0) {
    primme.maxMatvecs = (PRIMME_INT)maxit;
  }
  primme.matrixMatvec = rgmin_primme_matvec;
  ctx.apply = apply;
  ctx.precond = precond;
  ctx.user = user;
  ctx.actions = 0;
  primme.matrix = &ctx;
  if (precond != NULL) {
    primme.applyPreconditioner = rgmin_primme_precond;
    primme.correctionParams.precondition = 1;
  }

  evals = (double *)calloc((size_t)nev, sizeof(double));
  evecs = (double *)calloc((size_t)n * (size_t)nev, sizeof(double));
  rnorms = (double *)calloc((size_t)nev, sizeof(double));
  if (evals == NULL || evecs == NULL || rnorms == NULL) {
    free(evals);
    free(evecs);
    free(rnorms);
    primme_free(&primme);
    return 1;
  }
  memcpy(evecs, seed, (size_t)n * sizeof(double));
  primme.initSize = 1;

  err = dprimme(evals, evecs, rnorms, &primme);
  if (err == 0) {
    memcpy(out_vec, evecs, (size_t)n * sizeof(double));
    *out_value = evals[0];
    if (out_actions != NULL) {
      *out_actions = ctx.actions;
    }
  }
  free(evals);
  free(evecs);
  free(rnorms);
  primme_free(&primme);
  if (err == 0) {
    return 0;
  }
  if (err == PRIMME_MAIN_ITER_FAILURE) {
    return 2;
  }
  return 3;
}
