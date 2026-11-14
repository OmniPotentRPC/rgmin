// Copyright (c) rgmin developers
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
//! pins bit-identically. Above [`REDUCTION_MIN`] every build sums fixed
//! chunks and then a balanced pairwise tree of those chunks. The `par`
//! feature only decides which thread fills which chunk.

use ndarray::{Array1, ArrayView1};

/// Chunk width of the deterministic reduction.
pub(crate) const REDUCTION_CHUNK: usize = 4096;

/// Length at which reductions switch to the fixed-chunk pairwise tree.
pub(crate) const REDUCTION_MIN: usize = 65_536;

/// Below this length the par feature stays on the serial element loops.
#[cfg(feature = "par")]
const PAR_MIN_LEN: usize = REDUCTION_MIN;

fn pairwise_sum(v: &[f64]) -> f64 {
    fn rec(v: &[f64]) -> f64 {
        match v.len() {
            0 => 0.0,
            1 => v[0],
            n => {
                let mid = n / 2;
                rec(&v[..mid]) + rec(&v[mid..])
            }
        }
    }
    rec(v)
}

fn chunk_dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (x, y) in a.iter().zip(b) {
        acc += x * y;
    }
    acc
}

fn chunk_sum(a: &[f64]) -> f64 {
    let mut acc = 0.0;
    for x in a {
        acc += *x;
    }
    acc
}

fn fill_dot_parts(a: &[f64], b: &[f64], parts: &mut [f64]) {
    #[cfg(feature = "par")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let _join = crate::phase::Guard::rayon();
        parts.par_iter_mut().enumerate().for_each(|(c, slot)| {
            let start = c * REDUCTION_CHUNK;
            let end = (start + REDUCTION_CHUNK).min(a.len());
            *slot = chunk_dot(&a[start..end], &b[start..end]);
        });
        return;
    }
    for (c, slot) in parts.iter_mut().enumerate() {
        let start = c * REDUCTION_CHUNK;
        let end = (start + REDUCTION_CHUNK).min(a.len());
        *slot = chunk_dot(&a[start..end], &b[start..end]);
    }
}

fn fill_sum_parts(a: &[f64], parts: &mut [f64]) {
    #[cfg(feature = "par")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let _join = crate::phase::Guard::rayon();
        parts.par_iter_mut().enumerate().for_each(|(c, slot)| {
            let start = c * REDUCTION_CHUNK;
            let end = (start + REDUCTION_CHUNK).min(a.len());
            *slot = chunk_sum(&a[start..end]);
        });
        return;
    }
    for (c, slot) in parts.iter_mut().enumerate() {
        let start = c * REDUCTION_CHUNK;
        let end = (start + REDUCTION_CHUNK).min(a.len());
        *slot = chunk_sum(&a[start..end]);
    }
}

thread_local! {
    static PARTS: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn with_parts<R>(n: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
    let nchunks = n.div_ceil(REDUCTION_CHUNK).max(1);
    PARTS.with(|parts| {
        let mut parts = parts.borrow_mut();
        if parts.len() < nchunks {
            parts.resize(nchunks, 0.0);
        }
        f(&mut parts[..nchunks])
    })
}

/// `x . y`. At or above [`REDUCTION_MIN`] this is the fixed-chunk
/// pairwise sum, identical for every thread count.
pub fn dot(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    let _phase = crate::phase::Guard::dot();
    if x.len() < REDUCTION_MIN {
        return x.dot(&y);
    }
    match (x.as_slice(), y.as_slice()) {
        (Some(a), Some(b)) => with_parts(a.len(), |parts| {
            fill_dot_parts(a, b, parts);
            pairwise_sum(parts)
        }),
        _ => x.dot(&y),
    }
}

/// `y += a x`.
#[cfg(not(feature = "par"))]
pub fn axpy(a: f64, x: ArrayView1<f64>, y: &mut Array1<f64>) {
    let _phase = crate::phase::Guard::axpy();
    y.scaled_add(a, &x);
}

/// `y += a x`, updated in parallel chunks.
#[cfg(feature = "par")]
pub fn axpy(a: f64, x: ArrayView1<f64>, y: &mut Array1<f64>) {
    use rayon::prelude::*;
    let _phase = crate::phase::Guard::axpy();
    if x.len() < PAR_MIN_LEN {
        y.scaled_add(a, &x);
        return;
    }
    match (x.as_slice(), y.as_slice_mut()) {
        (Some(xs), Some(ys)) => {
            let _join = crate::phase::Guard::rayon();
            ys.par_chunks_mut(4096)
                .zip(xs.par_chunks(4096))
                .for_each(|(cy, cx)| {
                    for (p, q) in cy.iter_mut().zip(cx) {
                        *p += a * q;
                    }
                });
        }
        _ => y.scaled_add(a, &x),
    }
}

/// `sum_i x_i`. At or above [`REDUCTION_MIN`] this is the fixed-chunk
/// pairwise sum, identical for every thread count.
pub fn sum(x: ArrayView1<f64>) -> f64 {
    if x.len() < REDUCTION_MIN {
        return x.sum();
    }
    match x.as_slice() {
        Some(a) => with_parts(a.len(), |parts| {
            fill_sum_parts(a, parts);
            pairwise_sum(parts)
        }),
        None => x.sum(),
    }
}

/// `dst[i] = src[i]`. A memcpy, not a rayon join: at n = 96_000 a join
/// costs more than the copy.
pub fn copy_slice(dst: &mut [f64], src: &[f64]) {
    debug_assert_eq!(dst.len(), src.len());
    dst.copy_from_slice(src);
}

/// `dst[i] = op(src[i])`, one pass.
#[inline(always)]
pub fn apply1(dst: &mut [f64], src: &[f64], op: impl Fn(&mut f64, f64) + Sync) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(feature = "par")]
    if dst.len() >= PAR_MIN_LEN && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let _join = crate::phase::Guard::rayon();
        dst.par_chunks_mut(16_384)
            .zip(src.par_chunks(16_384))
            .for_each(|(d, s)| {
                for (di, &si) in d.iter_mut().zip(s) {
                    op(di, si);
                }
            });
        return;
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        op(d, s);
    }
}

/// `dst[i] = op(a[i], b[i])`, one pass.
#[inline(always)]
pub fn apply2(dst: &mut [f64], a: &[f64], b: &[f64], op: impl Fn(&mut f64, f64, f64) + Sync) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    #[cfg(feature = "par")]
    if dst.len() >= PAR_MIN_LEN && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let _join = crate::phase::Guard::rayon();
        dst.par_chunks_mut(16_384)
            .zip(a.par_chunks(16_384))
            .zip(b.par_chunks(16_384))
            .for_each(|((d, x), y)| {
                for ((di, &xi), &yi) in d.iter_mut().zip(x).zip(y) {
                    op(di, xi, yi);
                }
            });
        return;
    }
    for ((d, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        op(d, x, y);
    }
}

/// `dst[i] = keep * dst[i] + scale * src[i]`.
#[inline(always)]
pub fn blend(dst: &mut [f64], keep: f64, scale: f64, src: &[f64]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(feature = "par")]
    if dst.len() >= PAR_MIN_LEN && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let _join = crate::phase::Guard::rayon();
        dst.par_chunks_mut(16_384)
            .zip(src.par_chunks(16_384))
            .for_each(|(d, s)| {
                for (di, &si) in d.iter_mut().zip(s) {
                    *di = keep * *di + scale * si;
                }
            });
        return;
    }
    for (di, &si) in dst.iter_mut().zip(src) {
        *di = keep * *di + scale * si;
    }
}

/// `dst[i] += scale * src[i]`.
#[inline(always)]
pub fn add_scaled(dst: &mut [f64], src: &[f64], scale: f64) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(feature = "par")]
    if dst.len() >= PAR_MIN_LEN && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let _join = crate::phase::Guard::rayon();
        dst.par_chunks_mut(16_384)
            .zip(src.par_chunks(16_384))
            .for_each(|(d, s)| {
                for (di, &si) in d.iter_mut().zip(s) {
                    *di += scale * si;
                }
            });
        return;
    }
    for (di, &si) in dst.iter_mut().zip(src) {
        *di += scale * si;
    }
}

/// `dst[i] = pos[i] + t * dir[i]` in one pass.
///
/// A line-search trial used to copy `pos` and then axpy. That is two
/// trips over three vectors and, under `par`, a rayon join on the
/// second trip only. One pass writes the same bits.
pub fn lincomb(dst: &mut [f64], pos: &[f64], t: f64, dir: &[f64]) {
    debug_assert_eq!(dst.len(), pos.len());
    debug_assert_eq!(dst.len(), dir.len());
    #[cfg(feature = "par")]
    if dst.len() >= PAR_MIN_LEN {
        use rayon::prelude::*;
        let _join = crate::phase::Guard::rayon();
        dst.par_chunks_mut(4096)
            .zip(pos.par_chunks(4096))
            .zip(dir.par_chunks(4096))
            .for_each(|((d, p), r)| {
                for ((di, &pi), &ri) in d.iter_mut().zip(p).zip(r) {
                    *di = pi + t * ri;
                }
            });
        return;
    }
    for ((di, &pi), &ri) in dst.iter_mut().zip(pos).zip(dir) {
        *di = pi + t * ri;
    }
}

/// `||x||_2`.
pub fn nrm2(x: ArrayView1<f64>) -> f64 {
    dot(x, x).sqrt()
}

/// `||x - y||_2`, the same bits as [`nrm2`] of the stored difference.
///
/// Each element is subtracted once. The squares then fold the way
/// ndarray folds `d · d` on that contiguous difference. Above the
/// parallel cutoff the difference is stored and [`nrm2`] runs, so the
/// chunk fold stays the one `d · d` uses.
pub fn nrm2_diff(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    assert_eq!(x.len(), y.len());
    #[cfg(feature = "par")]
    if x.len() >= PAR_MIN_LEN {
        let diff = &x.to_owned() - &y;
        return nrm2(diff.view());
    }
    diff_sq(x, y).sqrt()
}

/// Eight-way sum of squares of `x - y`, matching ndarray's unrolled dot
/// of the stored difference with itself.
fn diff_sq(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    let n = x.len();
    let mut i = 0;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    while i + 8 <= n {
        let d0 = x[i] - y[i];
        let d1 = x[i + 1] - y[i + 1];
        let d2 = x[i + 2] - y[i + 2];
        let d3 = x[i + 3] - y[i + 3];
        let d4 = x[i + 4] - y[i + 4];
        let d5 = x[i + 5] - y[i + 5];
        let d6 = x[i + 6] - y[i + 6];
        let d7 = x[i + 7] - y[i + 7];
        p0 += d0 * d0;
        p1 += d1 * d1;
        p2 += d2 * d2;
        p3 += d3 * d3;
        p4 += d4 * d4;
        p5 += d5 * d5;
        p6 += d6 * d6;
        p7 += d7 * d7;
        i += 8;
    }
    let mut sum = 0.0;
    sum += p0 + p4;
    sum += p1 + p5;
    sum += p2 + p6;
    sum += p3 + p7;
    while i < n {
        let d = x[i] - y[i];
        sum += d * d;
        i += 1;
    }
    sum
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn a_difference_norm_matches_the_stored_vector() {
        for n in [0usize, 1, 7, 8, 9, 64, 3000] {
            let x = Array1::from((0..n).map(|i| (i as f64 * 0.17).sin()).collect::<Vec<_>>());
            let y = Array1::from((0..n).map(|i| (i as f64 * 0.09).cos()).collect::<Vec<_>>());
            let stored = &x - &y;
            assert_eq!(
                nrm2_diff(x.view(), y.view()).to_bits(),
                nrm2(stored.view()).to_bits(),
                "n={n}"
            );
        }
        let x = Array1::from(vec![-0.0, 1.0, f64::MIN_POSITIVE]);
        let y = Array1::from(vec![0.0, -1.0, 0.0]);
        let stored = &x - &y;
        assert_eq!(
            nrm2_diff(x.view(), y.view()).to_bits(),
            nrm2(stored.view()).to_bits()
        );
    }

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

    fn reference_dot(a: &[f64], b: &[f64]) -> f64 {
        let mut parts = Vec::new();
        let mut start = 0;
        while start < a.len() {
            let end = (start + REDUCTION_CHUNK).min(a.len());
            let mut acc = 0.0;
            for i in start..end {
                acc += a[i] * b[i];
            }
            parts.push(acc);
            start = end;
        }
        fn rec(v: &[f64]) -> f64 {
            match v.len() {
                0 => 0.0,
                1 => v[0],
                n => {
                    let mid = n / 2;
                    rec(&v[..mid]) + rec(&v[mid..])
                }
            }
        }
        rec(&parts)
    }

    #[test]
    fn a_long_dot_matches_the_fixed_chunk_pairwise_tree() {
        let n = 70_000;
        let a = Array1::from((0..n).map(|i| (i as f64 * 0.01).sin()).collect::<Vec<_>>());
        let b = Array1::from((0..n).map(|i| (i as f64 * 0.01).cos()).collect::<Vec<_>>());
        let got = dot(a.view(), b.view());
        let expect = reference_dot(a.as_slice().unwrap(), b.as_slice().unwrap());
        assert_eq!(got.to_bits(), expect.to_bits());
        let summed = sum(a.view());
        let mut parts = Vec::new();
        let slice = a.as_slice().unwrap();
        let mut start = 0;
        while start < n {
            let end = (start + REDUCTION_CHUNK).min(n);
            parts.push(slice[start..end].iter().sum::<f64>());
            start = end;
        }
        fn rec(v: &[f64]) -> f64 {
            match v.len() {
                1 => v[0],
                n => {
                    let mid = n / 2;
                    rec(&v[..mid]) + rec(&v[mid..])
                }
            }
        }
        assert_eq!(summed.to_bits(), rec(&parts).to_bits());
    }

    #[cfg(feature = "par")]
    #[test]
    fn a_long_dot_does_not_change_with_the_thread_count() {
        let n = 70_000;
        let a = Array1::from((0..n).map(|i| (i as f64 * 0.01).sin()).collect::<Vec<_>>());
        let b = Array1::from((0..n).map(|i| (i as f64 * 0.01).cos()).collect::<Vec<_>>());
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| dot(a.view(), b.view()));
        let many = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| dot(a.view(), b.view()));
        assert_eq!(one.to_bits(), many.to_bits());
    }
}
