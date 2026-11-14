/* Thin ChASE waist. Assembled dense H only. No public char mode/opt. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void dchase_init_(int *n, int *nev, int *nex, double *h, int *ldh, double *v,
                  double *ritzv, int *init);
void dchase_(int *deg, double *tol, char *mode, char *opt, char *qr);
void dchase_finalize_(int *init);

int rgmin_chase_lowest(int64_t n, const double *h, const double *seed,
                       int64_t nev, int64_t extra, int64_t degree, double tol,
                       double *out_vec, double *out_value) {
  int ni;
  int nevi;
  int nexi;
  int ldh;
  int init;
  int deg;
  char mode;
  char opt;
  char qr;
  int64_t cols;
  int64_t i;
  int64_t j;
  double *h_col = NULL;
  double *v = NULL;
  double *ritzv = NULL;

  if (n <= 0 || h == NULL || seed == NULL || out_vec == NULL ||
      out_value == NULL) {
    return 1;
  }
  if (nev < 1) {
    nev = 1;
  }
  if (extra < 1) {
    extra = 8;
  }
  if (degree < 1) {
    degree = 20;
  }
  ni = (int)n;
  nevi = (int)nev;
  nexi = (int)extra;
  ldh = ni;
  cols = nev + extra;
  if (cols < 2) {
    return 1;
  }
  h_col = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
  v = (double *)calloc((size_t)n * (size_t)cols, sizeof(double));
  ritzv = (double *)calloc((size_t)cols, sizeof(double));
  if (h_col == NULL || v == NULL || ritzv == NULL) {
    free(h_col);
    free(v);
    free(ritzv);
    return 1;
  }
  for (j = 0; j < n; j++) {
    for (i = 0; i < n; i++) {
      h_col[i + n * j] = h[i * n + j];
    }
  }
  memcpy(v, seed, (size_t)n * sizeof(double));
  init = 0;
  dchase_init_(&ni, &nevi, &nexi, h_col, &ldh, v, ritzv, &init);
  deg = (int)degree;
  mode = 'A';
  opt = 'S';
  qr = 'C';
  dchase_(&deg, &tol, &mode, &opt, &qr);
  memcpy(out_vec, v, (size_t)n * sizeof(double));
  *out_value = ritzv[0];
  dchase_finalize_(&init);
  free(h_col);
  free(v);
  free(ritzv);
  return 0;
}
