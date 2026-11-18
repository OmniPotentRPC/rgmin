//! SplitMix64 and Box–Muller, matching the hyperplane BAOAB stream.
//!
//! The uniform deviate is `(u64 >> 11 as f64 + 1/2) * 2^{-53}`, which lies
//! in `(0, 1)`. The spare normal is the sine of the same Box–Muller pair.

/// Deterministic normal generator.
#[derive(Clone, Debug)]
pub(super) struct SplitMix64 {
    state: u64,
    spare: f64,
    has_spare: bool,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: 0.0,
            has_spare: false,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    pub(super) fn next_normal(&mut self) -> f64 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = radius * theta.sin();
        self.has_spare = true;
        radius * theta.cos()
    }
}
