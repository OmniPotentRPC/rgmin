//! L-BFGS direction from stored pairs and a caller-supplied `H0`.
//!
//! The two-loop is the one gpridge evaluates: `q` starts at `-g`, `H0`
//! is applied to that intermediate, and the value returned is not
//! negated again. For a linear `H0` that is `-H g`. A scalar `H0`
//! multiplies each component. Nothing here evaluates an objective.

use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};

/// Curvature pairs for [`lbfgs_direction`].
///
/// Oldest first. A pair whose curvature is not strictly positive is
/// declined. `capacity` 0 keeps ten pairs.
pub struct LbfgsPairs {
    n: usize,
    max_pairs: usize,
    memory: Vec<Pair>,
}

struct Pair {
    s: Array1<f64>,
    y: Array1<f64>,
    rho: f64,
}

/// Initial inverse Hessian for one two-loop direction.
pub enum LbfgsH0<'a> {
    /// `H0 = γ I`.
    Scale(f64),
    /// `v ↦ H0 v`. Invoked once, on the intermediate vector.
    Apply(&'a dyn Fn(ArrayView1<'_, f64>) -> Result<Array1<f64>>),
}

impl LbfgsPairs {
    /// Pair memory of dimension `n`. `capacity` 0 selects 10.
    pub fn new(n: usize, capacity: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::LbfgsDirection {
                what: "dimension is zero",
            });
        }
        Ok(Self {
            n,
            max_pairs: if capacity == 0 { 10 } else { capacity },
            memory: Vec::new(),
        })
    }

    /// Dimension of every stored vector.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Store `(s, y)` when `s · y` is finite and strictly positive.
    pub fn push(&mut self, s: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> Result<bool> {
        if s.len() != self.n || y.len() != self.n {
            return Err(Error::LbfgsDirection {
                what: "pair length does not match the dimension",
            });
        }
        let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        if !sy.is_finite() || !(sy > 0.0) {
            return Ok(false);
        }
        self.push_rho(s, y, 1.0 / sy)?;
        Ok(true)
    }

    /// Store `(s, y)` with the caller's `rho`. A non-finite `rho` is declined.
    pub fn push_rho(
        &mut self,
        s: ArrayView1<'_, f64>,
        y: ArrayView1<'_, f64>,
        rho: f64,
    ) -> Result<bool> {
        if s.len() != self.n || y.len() != self.n {
            return Err(Error::LbfgsDirection {
                what: "pair length does not match the dimension",
            });
        }
        if !rho.is_finite() {
            return Ok(false);
        }
        self.memory.push(Pair {
            s: s.to_owned(),
            y: y.to_owned(),
            rho,
        });
        while self.memory.len() > self.max_pairs {
            self.memory.remove(0);
        }
        Ok(true)
    }

    /// Direction from `g` and `H0`.
    pub fn direction(&self, gradient: ArrayView1<'_, f64>, h0: LbfgsH0<'_>) -> Result<Array1<f64>> {
        if gradient.len() != self.n {
            return Err(Error::LbfgsDirection {
                what: "gradient length does not match the dimension",
            });
        }
        let g = gradient.as_slice().ok_or(Error::LbfgsDirection {
            what: "gradient is not contiguous",
        })?;
        let mut q = vec![0.0; self.n];
        for i in 0..self.n {
            q[i] = -g[i];
        }
        let m = self.memory.len();
        let mut alpha = vec![0.0; m];
        for k in (0..m).rev() {
            let p = &self.memory[k];
            let a = p.rho * dot(p.s.as_slice().unwrap(), &q);
            alpha[k] = a;
            axpy(-a, p.y.as_slice().unwrap(), &mut q);
        }
        match h0 {
            LbfgsH0::Scale(gamma) => {
                if !gamma.is_finite() {
                    return Err(Error::LbfgsDirection {
                        what: "H0 scale is not finite",
                    });
                }
                for v in &mut q {
                    *v *= gamma;
                }
            }
            LbfgsH0::Apply(apply) => {
                let applied = apply(ArrayView1::from(&q))?;
                if applied.len() != self.n {
                    return Err(Error::LbfgsDirection {
                        what: "H0 length does not match the dimension",
                    });
                }
                q.copy_from_slice(applied.as_slice().ok_or(Error::LbfgsDirection {
                    what: "H0 output is not contiguous",
                })?);
            }
        }
        for k in 0..m {
            let p = &self.memory[k];
            let b = p.rho * dot(p.y.as_slice().unwrap(), &q);
            axpy(alpha[k] - b, p.s.as_slice().unwrap(), &mut q);
        }
        Ok(Array1::from_vec(q))
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..y.len() {
        y[i] += alpha * x[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_memory_is_scaled_steepest() {
        let pairs = LbfgsPairs::new(2, 0).unwrap();
        let g = Array1::from_vec(vec![2.0, -4.0]);
        let d = pairs.direction(g.view(), LbfgsH0::Scale(0.01)).unwrap();
        assert_eq!(d[0], -0.02);
        assert_eq!(d[1], 0.04);
    }

    #[test]
    fn one_pair_matches_the_two_loop() {
        let mut pairs = LbfgsPairs::new(2, 4).unwrap();
        let s = Array1::from_vec(vec![1.0, 0.0]);
        let y = Array1::from_vec(vec![2.0, 0.0]);
        assert!(pairs.push(s.view(), y.view()).unwrap());
        let g = Array1::from_vec(vec![1.0, 3.0]);
        let d = pairs.direction(g.view(), LbfgsH0::Scale(0.5)).unwrap();
        // q = -g = [-1, -3]
        // alpha = (1/2) * s·q = (1/2)*(-1) = -0.5
        // q -= alpha y = [-1, -3] - (-0.5)*[2, 0] = [0, -3]
        // q *= 0.5 = [0, -1.5]
        // beta = (1/2) * y·q = (1/2)*0 = 0
        // q += (alpha - beta) s = [0, -1.5] + (-0.5)*[1, 0] = [-0.5, -1.5]
        assert_eq!(d[0], -0.5);
        assert_eq!(d[1], -1.5);
    }
}
