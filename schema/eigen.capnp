@0xbac2a72df87e1fed;

# Lowest-mode waist for the IRC kick and lambda_min. Integers are the
# public ABI: Rust EigensolverKind, C rgmin_eigen_kind_t, and this
# schema share the same ordinals. There is no Text kind and no
# elpa_set string table.

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("rgmin::schema");

enum EigensolverKind {
  lanczos @0;
  rayleighRitz @1;
  jacobiDavidson @2;
  lobpcg @3;
  primme @4;
  slepc @5;
  chase @6;
  elpa @7;
  elpa2 @8;
  slate @9;
  magma @10;
  cusolver @11;
  dlaFuture @12;
  eigenExa @13;
  # Jónsson dimer with Heyden plane rotations.
  dimer @14;
  # libkrylov Davidson / Jacobi-Davidson. Matrix-free.
  libkrylov @15;
}

struct EigenParams {
  kind @0 :EigensolverKind = lanczos;
  # Number of extremal pairs. IRC kick uses 1.
  nev @1 :UInt32 = 1;
  # Krylov / subspace cap. 0 selects min(n, 12).
  krylov @2 :UInt32 = 0;
  # Outer iterations. 0 selects n.
  maxIter @3 :UInt32 = 0;
  # Residual tolerance. 0 selects 1e-8.
  tol @4 :Float64 = 0;
  # ChASE filter degree. 0 selects 20 with per-vector optimization.
  degree @5 :UInt32 = 0;
  # ChASE search-space extra (nex). 0 selects max(8, ceil(0.2 * nev)).
  extra @6 :UInt32 = 0;
}
