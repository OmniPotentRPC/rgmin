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
//! pins bit-identically. The `par` feature swaps in rayon over slices
//! for the length-`n` reductions and updates; results may differ from
//! serial in the last bits because floating-point addition reorders,
//! which is why it is a feature and not a default, and why no test of
//! the serial path runs under it.

use ndarray::{Array1, ArrayView1};

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
const PAR_MIN_LEN: usize = 65_536;

/// `x . y`, reduced in parallel chunks.
#[cfg(feature = "par")]
pub fn dot(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    use rayon::prelude::*;
    if x.len() < PAR_MIN_LEN {
        return x.dot(&y);
    }
    match (x.as_slice(), y.as_slice()) {
        (Some(a), Some(b)) => a
            .par_chunks(4096)
            .zip(b.par_chunks(4096))
            .map(|(ca, cb)| ca.iter().zip(cb).map(|(p, q)| p * q).sum::<f64>())
            .sum(),
        _ => x.dot(&y),
    }
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
            .par_chunks_mut(4096)
            .zip(xs.par_chunks(4096))
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

/// `sum_i x_i`, reduced in parallel chunks.
#[cfg(feature = "par")]
pub fn sum(x: ArrayView1<f64>) -> f64 {
    use rayon::prelude::*;
    if x.len() < PAR_MIN_LEN {
        return x.sum();
    }
    match x.as_slice() {
        Some(a) => a.par_chunks(4096).map(|c| c.iter().sum::<f64>()).sum(),
        None => x.sum(),
    }
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
        Some(ys) => ys.par_chunks_mut(4096).for_each(|cy| {
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
            ys.par_chunks_mut(4096)
                .zip(xs.par_chunks(4096))
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
            ys.par_chunks_mut(4096)
                .zip(xs.par_chunks(4096))
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

    #[cfg(feature = "par")]
    #[test]
    fn par_dot_axpy_on_a_long_vector_uses_rayon() {
        let n = super::PAR_MIN_LEN;
        let x = Array1::from_elem(n, 0.5);
        let y = Array1::from_elem(n, 2.0);
        let d = dot(x.view(), y.view());
        assert!((d - n as f64).abs() < 1e-6, "par dot {d}");
        let mut z = y.clone();
        axpy(0.5, x.view(), &mut z);
        assert!((z[0] - 2.25).abs() < 1e-15);
        assert!((z[n - 1] - 2.25).abs() < 1e-15);
        let mut s = x.clone();
        scale(4.0, &mut s);
        assert!((sum(s.view()) - 2.0 * n as f64).abs() < 1e-6);
    }
}
