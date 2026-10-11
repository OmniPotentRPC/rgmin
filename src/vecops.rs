//! The vector seam: every O(n) operation a solver performs, in one
//! place.
//!
//! TAO writes each algorithm against PETSc `Vec` -- `VecDot`,
//! `VecAXPY`, `VecNorm` -- and that seam is its entire scaling story:
//! one `blmvm.c` runs serial, distributed, or on device because the
//! solver never touches storage. Here the objective side already has
//! its seam (`eindir` supplies `f` and `g`); this module is the
//! matching one for the solver's own algebra. Algorithms call these
//! functions; storage-specific execution lives behind them.
//!
//! The default implementation is the serial one the whole test suite
//! pins bit-identically. The `par` feature keeps that fold below
//! `PAR_MIN_LEN`. Longer `dot` and `sum` (and therefore `nrm2`) write
//! one partial per fixed chunk, then add those partials on one binary
//! tree. The tree is a function of the length, so the bits do not
//! depend on the thread count. They can still differ from the serial
//! fold; the bound is the one on `combine_tree`. Decisions stay on
//! [`dot_ordered`].

use ndarray::{Array1, ArrayView1};

/// `x . y` summed from the first index upward.
///
/// Noise projections and any reduction that decides a step stay on this
/// loop. [`dot`] under `par` matches it below `PAR_MIN_LEN` and, above
/// that, matches `combine_tree` rather than this fold.
pub fn dot_ordered(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let mut acc = 0.0;
    for i in 0..x.len() {
        acc += x[i] * y[i];
    }
    acc
}

/// `x . y`.
#[cfg(not(feature = "par"))]
pub fn dot(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    x.dot(&y)
}

/// Below this length the par feature stays on the serial path: on the
/// reference host the rayon reductions measured 2.6x slower than
/// serial at n = 20000 (thread coordination outweighs O(n) arithmetic),
/// so parallel length-n algebra engages only where it has a chance.
#[cfg(feature = "par")]
pub(crate) const PAR_MIN_LEN: usize = 65_536;

/// Fixed `par` chunk length. Independent of the pool size.
#[cfg(feature = "par")]
pub(crate) const PAR_CHUNK: usize = 4096;

/// Levels of [`combine_tree`]. Sixty-four covers every slice that fits
/// in memory, and a level that does not shrink still stops.
#[cfg(feature = "par")]
const TREE_LEVELS: usize = 64;

/// Pairwise sum of `partials`.
///
/// Level `k + 1` stores `partials[2i] + partials[2i + 1]`, and an odd
/// tail is copied unchanged. The pairing depends only on
/// `partials.len()`.
///
/// Any parenthesization of `m` finite terms differs from their exact sum
/// by at most `γ_{m-1} Σ |p_i|`, with unit roundoff `u = 2^{-53}` and
/// `γ_k = k u / (1 - k u)` when `k u < 1` (Higham, Accuracy and
/// Stability of Numerical Algorithms, 2nd ed., ISBN 978-0-89871-802-7,
/// Theorem 4.2). A `par` dot is this tree of chunk sums. Against
/// ndarray's dot the absolute gap is at most twice that bound on the
/// rounded products, because both folds are some parenthesization.
#[cfg(feature = "par")]
fn combine_tree(partials: &[f64]) -> f64 {
    if partials.is_empty() {
        return 0.0;
    }
    let mut level = partials.to_vec();
    for _ in 0..TREE_LEVELS {
        if level.len() <= 1 {
            break;
        }
        let n = level.len();
        let pairs = n / 2;
        let mut next = Vec::with_capacity(pairs + (n % 2));
        for i in 0..pairs {
            next.push(level[2 * i] + level[2 * i + 1]);
        }
        if n % 2 == 1 {
            next.push(level[n - 1]);
        }
        level = next;
    }
    level[0]
}

/// Left-to-right sum of `xs[start..end]`. One chain, so the lane width
/// does not choose the parenthesization.
#[cfg(feature = "par")]
fn sum_range(xs: &[f64], start: usize, end: usize) -> f64 {
    let mut acc = 0.0;
    for value in &xs[start..end] {
        acc += *value;
    }
    acc
}

/// Left-to-right sum of products `xs[i] * ys[i]`. The product is rounded
/// before the add, so the chain does not contract into an fma.
#[cfg(feature = "par")]
fn dot_range(xs: &[f64], ys: &[f64], start: usize, end: usize) -> f64 {
    let mut acc = 0.0;
    for i in start..end {
        let prod = xs[i] * ys[i];
        acc += prod;
    }
    acc
}

/// One partial per chunk, written at the chunk index.
#[cfg(feature = "par")]
fn fill_partials(n: usize, body: impl Fn(usize, usize) -> f64 + Sync) -> Vec<f64> {
    use rayon::prelude::*;
    let nchunks = n.div_ceil(PAR_CHUNK);
    let mut partials = vec![0.0; nchunks];
    partials
        .par_iter_mut()
        .enumerate()
        .for_each(|(chunk, slot)| {
            let start = chunk * PAR_CHUNK;
            let end = (start + PAR_CHUNK).min(n);
            *slot = body(start, end);
        });
    partials
}

/// `x . y`, reduced by [`combine_tree`] over fixed chunks.
#[cfg(feature = "par")]
pub fn dot(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    if x.len() != y.len() || x.len() < PAR_MIN_LEN {
        return x.dot(&y);
    }
    let (Some(xs), Some(ys)) = (x.as_slice(), y.as_slice()) else {
        return x.dot(&y);
    };
    combine_tree(&fill_partials(xs.len(), |start, end| {
        dot_range(xs, ys, start, end)
    }))
}

/// `y += a x`.
#[cfg(not(feature = "par"))]
pub fn axpy(a: f64, x: ArrayView1<f64>, y: &mut Array1<f64>) {
    y.scaled_add(a, &x);
}

/// `y += a x`, updated in parallel chunks.
#[cfg(feature = "par")]
pub fn axpy(a: f64, x: ArrayView1<f64>, y: &mut Array1<f64>) {
    use rayon::prelude::*;
    if x.len() < PAR_MIN_LEN {
        y.scaled_add(a, &x);
        return;
    }
    match (x.as_slice(), y.as_slice_mut()) {
        (Some(xs), Some(ys)) => ys
            .par_chunks_mut(PAR_CHUNK)
            .zip(xs.par_chunks(PAR_CHUNK))
            .for_each(|(cy, cx)| {
                for (p, q) in cy.iter_mut().zip(cx) {
                    *p += a * q;
                }
            }),
        _ => y.scaled_add(a, &x),
    }
}

/// `sum_i x_i`.
#[cfg(not(feature = "par"))]
pub fn sum(x: ArrayView1<f64>) -> f64 {
    x.sum()
}

/// `sum_i x_i`, reduced by [`combine_tree`] over fixed chunks.
#[cfg(feature = "par")]
pub fn sum(x: ArrayView1<f64>) -> f64 {
    if x.len() < PAR_MIN_LEN {
        return x.sum();
    }
    let Some(xs) = x.as_slice() else {
        return x.sum();
    };
    combine_tree(&fill_partials(xs.len(), |start, end| {
        sum_range(xs, start, end)
    }))
}

/// `||x||_2`.
pub fn nrm2(x: ArrayView1<f64>) -> f64 {
    dot(x, x).sqrt()
}

/// `y *= a`.
#[cfg(not(feature = "par"))]
pub fn scale(a: f64, y: &mut Array1<f64>) {
    y.mapv_inplace(|v| v * a);
}

/// `y *= a`, updated in parallel chunks.
#[cfg(feature = "par")]
pub fn scale(a: f64, y: &mut Array1<f64>) {
    use rayon::prelude::*;
    if y.len() < PAR_MIN_LEN {
        y.mapv_inplace(|v| v * a);
        return;
    }
    match y.as_slice_mut() {
        Some(ys) => ys.par_chunks_mut(PAR_CHUNK).for_each(|cy| {
            for p in cy.iter_mut() {
                *p *= a;
            }
        }),
        None => y.mapv_inplace(|v| v * a),
    }
}

/// `y[i] *= x[i]`.
#[cfg(not(feature = "par"))]
pub fn mul_assign(x: ArrayView1<f64>, y: &mut Array1<f64>) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] *= x[i];
    }
}

/// `y[i] *= x[i]`, updated in parallel chunks.
#[cfg(feature = "par")]
pub fn mul_assign(x: ArrayView1<f64>, y: &mut Array1<f64>) {
    use rayon::prelude::*;
    let n = x.len().min(y.len());
    if n < PAR_MIN_LEN {
        for i in 0..n {
            y[i] *= x[i];
        }
        return;
    }
    match (x.as_slice(), y.as_slice_mut()) {
        (Some(xs), Some(ys)) => {
            let ys = &mut ys[..n];
            let xs = &xs[..n];
            ys.par_chunks_mut(PAR_CHUNK)
                .zip(xs.par_chunks(PAR_CHUNK))
                .for_each(|(cy, cx)| {
                    for (p, q) in cy.iter_mut().zip(cx) {
                        *p *= *q;
                    }
                });
        }
        _ => {
            for i in 0..n {
                y[i] *= x[i];
            }
        }
    }
}

/// `y[i] /= x[i].max(floor)`.
#[cfg(not(feature = "par"))]
pub fn div_assign_floor(x: ArrayView1<f64>, y: &mut Array1<f64>, floor: f64) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] /= x[i].max(floor);
    }
}

/// `y[i] /= x[i].max(floor)`, updated in parallel chunks.
#[cfg(feature = "par")]
pub fn div_assign_floor(x: ArrayView1<f64>, y: &mut Array1<f64>, floor: f64) {
    use rayon::prelude::*;
    let n = x.len().min(y.len());
    if n < PAR_MIN_LEN {
        for i in 0..n {
            y[i] /= x[i].max(floor);
        }
        return;
    }
    match (x.as_slice(), y.as_slice_mut()) {
        (Some(xs), Some(ys)) => {
            let ys = &mut ys[..n];
            let xs = &xs[..n];
            ys.par_chunks_mut(PAR_CHUNK)
                .zip(xs.par_chunks(PAR_CHUNK))
                .for_each(|(cy, cx)| {
                    for (p, q) in cy.iter_mut().zip(cx) {
                        *p /= q.max(floor);
                    }
                });
        }
        _ => {
            for i in 0..n {
                y[i] /= x[i].max(floor);
            }
        }
    }
}

/// `||x||_inf`, infinity on any non-finite component: a broken vector
/// is never a small one.
pub fn nrminf(x: ArrayView1<f64>) -> f64 {
    let mut m = 0.0_f64;
    for v in x.iter() {
        if !v.is_finite() {
            return f64::INFINITY;
        }
        m = m.max(v.abs());
    }
    m
}

/// A solver-owned vector: storage plus the DLPack device it lives on.
///
/// This is the seam's storage handle (nbgu step of the TAO `Vec`
/// story): solvers hold `Vector`s and call the `v*` operations, which
/// dispatch on the device tag. The CPU arm wraps the crate's ndarray
/// storage at zero cost, so it is bit-identical to calling the slice
/// operations directly. A non-CPU tag needs a kernel backend behind
/// this seam; without one, construction fails loudly rather than
/// staging device data through the host, which the single-copy waist
/// contract forbids.
#[derive(Debug)]
pub struct Vector {
    data: Array1<f64>,
    device: dlpk::sys::DLDevice,
}

const CPU: dlpk::sys::DLDevice = dlpk::sys::DLDevice {
    device_type: dlpk::sys::DLDeviceType::kDLCPU,
    device_id: 0,
};

impl Vector {
    /// A zero CPU vector of length `n`.
    pub fn zeros_cpu(n: usize) -> Self {
        Self {
            data: Array1::zeros(n),
            device: CPU,
        }
    }

    /// Wrap host storage as a CPU vector. No copy.
    pub fn from_host(data: Array1<f64>) -> Self {
        Self { data, device: CPU }
    }

    /// Claim `data` for `device`. Only the CPU has a backend today;
    /// any other tag is refused so device data is never silently
    /// staged through the host.
    pub fn try_on(
        device: dlpk::sys::DLDevice,
        data: Array1<f64>,
    ) -> std::result::Result<Self, UnsupportedDevice> {
        if device.device_type == dlpk::sys::DLDeviceType::kDLCPU {
            Ok(Self { data, device })
        } else {
            Err(UnsupportedDevice {
                device_type: device.device_type as i32,
            })
        }
    }

    /// The device this vector lives on.
    pub fn device(&self) -> dlpk::sys::DLDevice {
        self.device
    }

    /// Host view of the CPU arm.
    pub fn host_view(&self) -> ArrayView1<'_, f64> {
        self.data.view()
    }

    /// Mutable host storage of the CPU arm.
    pub fn host_mut(&mut self) -> &mut Array1<f64> {
        &mut self.data
    }

    /// Unwrap into host storage.
    pub fn into_host(self) -> Array1<f64> {
        self.data
    }
}

/// The device tag names a backend this build does not carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsupportedDevice {
    /// The DLPack `DLDeviceType` discriminant that was refused.
    pub device_type: i32,
}

impl std::fmt::Display for UnsupportedDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DLPack device {} has no kernel backend in this build (CPU only)",
            self.device_type
        )
    }
}

impl std::error::Error for UnsupportedDevice {}

/// `x . y` on the vectors' device.
pub fn vdot(x: &Vector, y: &Vector) -> f64 {
    dot(x.data.view(), y.data.view())
}

/// `y += a x` on the vectors' device.
pub fn vaxpy(a: f64, x: &Vector, y: &mut Vector) {
    axpy(a, x.data.view(), &mut y.data);
}

/// `||x||_2` on the vector's device.
pub fn vnrm2(x: &Vector) -> f64 {
    nrm2(x.data.view())
}

/// `||x||_inf` on the vector's device, infinity on non-finite.
pub fn vnrminf(x: &Vector) -> f64 {
    nrminf(x.data.view())
}

/// `sum_i x_i` on the vector's device.
pub fn vsum(x: &Vector) -> f64 {
    sum(x.data.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn the_seam_matches_ndarray_on_the_serial_path() {
        let x = Array1::from((0..1000).map(|i| (i as f64).sin()).collect::<Vec<_>>());
        let y = Array1::from((0..1000).map(|i| (i as f64).cos()).collect::<Vec<_>>());
        let d = dot(x.view(), y.view());
        let mut z = y.clone();
        axpy(0.5, x.view(), &mut z);
        let mut reference = y.clone();
        reference.scaled_add(0.5, &x);
        assert!((d - x.dot(&y)).abs() < 1e-12 * d.abs().max(1.0));
        assert_eq!(z, reference);
        assert!((nrm2(x.view()) - x.dot(&x).sqrt()).abs() < 1e-12);
        assert!((sum(x.view()) - x.sum()).abs() < 1e-12);
    }

    #[test]
    fn the_infinity_norm_never_calls_a_broken_vector_small() {
        let x = Array1::from(vec![0.0, f64::NAN, 1.0]);
        assert!(nrminf(x.view()).is_infinite());
    }

    #[test]
    fn the_cpu_vector_is_the_slice_path_verbatim() {
        let a = Array1::from((0..64).map(|i| (i as f64).sin()).collect::<Vec<_>>());
        let b = Array1::from((0..64).map(|i| (i as f64).cos()).collect::<Vec<_>>());
        let vx = Vector::from_host(a.clone());
        let mut vy = Vector::from_host(b.clone());
        assert_eq!(vdot(&vx, &vy), dot(a.view(), b.view()));
        assert_eq!(vnrm2(&vx), nrm2(a.view()));
        vaxpy(0.25, &vx, &mut vy);
        let mut reference = b.clone();
        axpy(0.25, a.view(), &mut reference);
        assert_eq!(vy.into_host(), reference);
    }

    #[test]
    fn a_device_without_a_backend_is_refused_not_staged() {
        let cuda = dlpk::sys::DLDevice {
            device_type: dlpk::sys::DLDeviceType::kDLCUDA,
            device_id: 0,
        };
        let err = Vector::try_on(cuda, Array1::zeros(4)).unwrap_err();
        assert_eq!(err.device_type, dlpk::sys::DLDeviceType::kDLCUDA as i32);
    }
}
