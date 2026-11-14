/* Thin SLEPc EPS waist. Typed EPSSet* / STSet* only. No options database. */
#include <stdint.h>
#include <slepceps.h>
#include <slepcst.h>

typedef int (*rgmin_hess_apply)(void *user, int64_t n, const double *v, double *hv);

typedef struct {
  void *user;
  rgmin_hess_apply apply;
  int64_t n;
  int64_t actions;
} RgminShell;

/* Host already lives in PETSc. Do not create a world or options database. */
static PetscErrorCode rgmin_slepc_ensure(void)
{
  PetscBool ready = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialized(&ready));
  if (ready) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
}

static PetscErrorCode MatMult_Rgmin(Mat A, Vec x, Vec y)
{
  RgminShell *ctx = NULL;
  const PetscScalar *xx = NULL;
  PetscScalar *yy = NULL;
  PetscInt n = 0;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArray(y, &yy));
#if defined(PETSC_USE_COMPLEX)
  {
    double *vin;
    double *vout;
    PetscInt i;
    int rc;

    PetscCall(PetscMalloc1((size_t)n, &vin));
    PetscCall(PetscMalloc1((size_t)n, &vout));
    for (i = 0; i < n; ++i) {
      vin[i] = (double)PetscRealPart(xx[i]);
    }
    rc = ctx->apply(ctx->user, (int64_t)n, vin, vout);
    if (rc == 0) {
      for (i = 0; i < n; ++i) {
        yy[i] = vout[i];
      }
    }
    PetscCall(PetscFree(vin));
    PetscCall(PetscFree(vout));
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(y, &yy));
    PetscCheck(rc == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "ApplyHessian failed");
  }
#else
  {
    int rc = ctx->apply(ctx->user, (int64_t)n, (const double *)xx, (double *)yy);
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(y, &yy));
    PetscCheck(rc == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "ApplyHessian failed");
  }
#endif
  ctx->actions += 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 0 = ok, 1 = bad args, 2 = no pair, 3 = PETSc/SLEPc error */
int rgmin_slepc_lowest(int64_t n, const double *seed, int64_t nev, int64_t ncv,
                       int64_t maxit, double tol, void *pmat, int32_t st_kind,
                       double *out_vec, double *out_value, int64_t *out_actions,
                       void *user, rgmin_hess_apply apply)
{
  EPS eps = NULL;
  ST st = NULL;
  Mat A = NULL;
  Vec v0 = NULL;
  Vec xr = NULL;
  RgminShell ctx;
  PetscInt i;
  PetscInt nconv = 0;
  PetscScalar kr = 0.0, ki = 0.0;
  const PetscScalar *xx = NULL;
  PetscBool nonzero = PETSC_FALSE;
  PetscErrorCode ierr;

  if (n <= 0 || seed == NULL || out_vec == NULL || out_value == NULL ||
      out_actions == NULL || apply == NULL) {
    return 1;
  }
  if (nev < 1) {
    nev = 1;
  }
  if (ncv < nev + 1) {
    ncv = nev + 1;
  }
  if (ncv > n) {
    ncv = n;
  }
  if (maxit < 1) {
    maxit = n;
  }
  if (tol <= 0.0) {
    tol = 1.0e-8;
  }

  ierr = rgmin_slepc_ensure();
  if (ierr) {
    /* 4 = PETSc/SLEPc not initialized. Waist does not PetscInitialize. */
    return 4;
  }

  ctx.user = user;
  ctx.apply = apply;
  ctx.n = n;
  ctx.actions = 0;

  ierr = MatCreateShell(PETSC_COMM_SELF, (PetscInt)n, (PetscInt)n, (PetscInt)n,
                        (PetscInt)n, &ctx, &A);
  if (ierr) {
    return 3;
  }
  ierr = MatShellSetOperation(A, MATOP_MULT, (void (*)(void))MatMult_Rgmin);
  if (ierr) {
    MatDestroy(&A);
    return 3;
  }

  ierr = EPSCreate(PETSC_COMM_SELF, &eps);
  if (ierr) {
    MatDestroy(&A);
    return 3;
  }
  ierr = EPSSetOperators(eps, A, NULL);
  if (ierr) {
    goto fail;
  }
  ierr = EPSSetProblemType(eps, EPS_HEP);
  if (ierr) {
    goto fail;
  }
  ierr = EPSSetType(eps, EPSKRYLOVSCHUR);
  if (ierr) {
    goto fail;
  }
  ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
  if (ierr) {
    goto fail;
  }
  ierr = EPSSetDimensions(eps, (PetscInt)nev, (PetscInt)ncv, PETSC_DECIDE);
  if (ierr) {
    goto fail;
  }
  ierr = EPSSetTolerances(eps, (PetscReal)tol, (PetscInt)maxit);
  if (ierr) {
    goto fail;
  }

  ierr = EPSGetST(eps, &st);
  if (ierr) {
    goto fail;
  }
  switch (st_kind) {
  case 1:
    ierr = STSetType(st, STSHIFT);
    break;
  case 2:
    ierr = STSetType(st, STSINVERT);
    break;
  case 3:
    ierr = STSetType(st, STPRECOND);
    break;
  case 4:
    ierr = STSetType(st, STCAYLEY);
    break;
  default:
    break;
  }
  if (ierr) {
    goto fail;
  }
  if (pmat != NULL) {
    ierr = STSetPreconditionerMat(st, (Mat)pmat);
    if (ierr) {
      goto fail;
    }
  }

  for (i = 0; i < (PetscInt)n; ++i) {
    if (seed[i] != 0.0) {
      nonzero = PETSC_TRUE;
      break;
    }
  }
  if (nonzero) {
    ierr = MatCreateVecs(A, &v0, NULL);
    if (ierr) {
      goto fail;
    }
    for (i = 0; i < (PetscInt)n; ++i) {
      ierr = VecSetValue(v0, i, (PetscScalar)seed[i], INSERT_VALUES);
      if (ierr) {
        goto fail;
      }
    }
    ierr = VecAssemblyBegin(v0);
    if (ierr) {
      goto fail;
    }
    ierr = VecAssemblyEnd(v0);
    if (ierr) {
      goto fail;
    }
    ierr = EPSSetInitialSpace(eps, 1, &v0);
    if (ierr) {
      goto fail;
    }
  }

  ierr = EPSSolve(eps);
  if (ierr) {
    goto fail;
  }
  ierr = EPSGetConverged(eps, &nconv);
  if (ierr) {
    goto fail;
  }
  if (nconv < 1) {
    if (v0) {
      VecDestroy(&v0);
    }
    EPSDestroy(&eps);
    MatDestroy(&A);
    return 2;
  }

  ierr = MatCreateVecs(A, &xr, NULL);
  if (ierr) {
    goto fail;
  }
  ierr = EPSGetEigenpair(eps, 0, &kr, &ki, xr, NULL);
  if (ierr) {
    goto fail;
  }
  *out_value = (double)PetscRealPart(kr);
  ierr = VecGetArrayRead(xr, &xx);
  if (ierr) {
    goto fail;
  }
  for (i = 0; i < (PetscInt)n; ++i) {
    out_vec[i] = (double)PetscRealPart(xx[i]);
  }
  ierr = VecRestoreArrayRead(xr, &xx);
  if (ierr) {
    goto fail;
  }
  *out_actions = ctx.actions;

  VecDestroy(&xr);
  if (v0) {
    VecDestroy(&v0);
  }
  EPSDestroy(&eps);
  MatDestroy(&A);
  return 0;

fail:
  if (xr) {
    VecDestroy(&xr);
  }
  if (v0) {
    VecDestroy(&v0);
  }
  if (eps) {
    EPSDestroy(&eps);
  }
  if (A) {
    MatDestroy(&A);
  }
  return 3;
}
