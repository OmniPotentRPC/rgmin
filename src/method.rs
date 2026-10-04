//! Which local method to run.

use crate::nlcg::{Conjugacy, Restart};

/// Local first-order, quasi-Newton, or swarm method over an eindir objective.
#[derive(Clone, Debug, PartialEq)]
pub enum Method {
    /// Nonlinear CG (Nocedal-Wright algorithm 5.4).
    ///
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Nlcg {
        /// β formula.
        conjugacy: Conjugacy,
        /// When to reset to steepest descent.
        restart: Restart,
    },
    /// Dense inverse-BFGS (Nocedal-Wright 6.17).
    ///
    /// Broyden, *The Convergence of a Class of Double-rank Minimization
    /// Algorithms 1. General Considerations*,
    /// <https://doi.org/10.1093/imamat/6.1.76>.
    /// Fletcher, *A new approach to variable metric algorithms*,
    /// <https://doi.org/10.1093/comjnl/13.3.317>.
    /// Shanno, *Conditioning of quasi-Newton methods for function
    /// minimization*, <https://doi.org/10.1090/s0025-5718-1970-0274029-x>.
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Bfgs,
    /// Limited-memory BFGS two-loop recursion (Nocedal-Wright 7.4).
    ///
    /// Nocedal, *Updating quasi-Newton matrices with limited storage*,
    /// <https://doi.org/10.1090/s0025-5718-1980-0572855-7>.
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Lbfgs {
        /// Correction pairs kept (`m`). Typical 5 to 20.
        memory: usize,
    },
    /// Inverse SR1 (Nocedal-Wright 6.24).
    ///
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Sr1,
    /// SR2 Hessian update (`rgmin` `sr2.hpp`).
    ///
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Sr2,
    /// Kingma and Ba Adam, with a line search along the preconditioned step.
    ///
    /// Kingma and Ba, *Adam: A Method for Stochastic Optimization*,
    /// <https://doi.org/10.48550/arXiv.1412.6980>.
    Adam {
        /// First-moment decay.
        beta1: f64,
        /// Second-moment decay.
        beta2: f64,
        /// Denominator floor.
        eps: f64,
    },
    /// Steepest descent: `d = -g` every step.
    ///
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Steepest,
    /// Shifted Newton on a caller-supplied dense Hessian.
    ///
    /// Banerjee, Adams, Simons, Shepard,
    /// <https://doi.org/10.1021/j100247a015>.
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Newton {
        /// Shifted Newton or Banerjee RFO.
        kind: crate::newton::NewtonKind,
    },
    /// Particle swarm (xtsci `PSOptim`) over the objective box.
    ///
    /// Kennedy and Eberhart, *Particle swarm optimization*,
    /// <https://doi.org/10.1109/ICNN.1995.488968>.
    Pso {
        /// Swarm size. Typical 10 to 40.
        n_particles: usize,
        /// Inertia weight `w`.
        inertia: f64,
        /// Cognitive coefficient.
        c1: f64,
        /// Social coefficient.
        c2: f64,
    },
    /// FIRE inertial first-order step (Bitzek 2006 or Guénolé 2020).
    ///
    /// Bitzek, Koskinen, Gähler, Moseler, Gumbsch,
    /// <https://doi.org/10.1103/PhysRevLett.97.170201>.
    /// Guénolé, Nöhring, Vaid, Houllé, Xie, Prakash, Bitzek,
    /// <https://doi.org/10.1016/j.commatsci.2020.109584>.
    Fire {
        /// FIRE 1.0 or FIRE 2.0.
        kind: crate::fire::FireKind,
    },
    /// Barzilai-Borwein spectral steepest descent.
    ///
    /// Barzilai and Borwein, <https://doi.org/10.1093/imanum/8.1.141>.
    /// Raydan, <https://doi.org/10.1137/S1052623494266365>.
    Bb,
    /// Quick-min: project the velocity on the force, then an Euler step.
    ///
    /// The reference rule and the session departures (step cap, FIRE
    /// time-step factors, per-block cell velocity) are in
    /// [`crate::quickmin`].
    QuickMin,
    /// Powell dogleg on a caller-supplied dense Hessian.
    ///
    /// Nocedal and Wright, algorithm 4.1,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Dogleg,
}

impl Method {
    /// Polak-Ribiere CG, no restart.
    pub fn polak_ribiere() -> Self {
        Self::Nlcg {
            conjugacy: Conjugacy::PolakRibiere,
            restart: Restart::Never,
        }
    }

    /// L-BFGS with `m = 10` correction pairs.
    pub fn lbfgs() -> Self {
        Self::Lbfgs { memory: 10 }
    }

    /// Adam with Kingma-Ba defaults.
    pub fn adam() -> Self {
        Self::Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// PSO with xtsci defaults: 10 particles, `w = 0.5`, `c1 = c2 = 1.5`.
    pub fn pso() -> Self {
        Self::Pso {
            n_particles: 10,
            inertia: 0.5,
            c1: 1.5,
            c2: 1.5,
        }
    }

    /// NLCG with the given conjugacy and no restart.
    pub fn nlcg(conjugacy: Conjugacy) -> Self {
        Self::Nlcg {
            conjugacy,
            restart: Restart::Never,
        }
    }
}

impl Default for Method {
    fn default() -> Self {
        Self::polak_ribiere()
    }
}
