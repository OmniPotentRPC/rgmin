//! Which local method to run.

use crate::nlcg::{Conjugacy, Restart};

/// Local first-order, quasi-Newton, or swarm method over an eindir objective.
///
/// Non-exhaustive: a match outside this crate needs a wildcard arm, so a
/// later method is an additive change.
#[non_exhaustive]
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
    /// SR1 in a trust region (Nocedal-Wright 6.24, algorithm 4.1).
    ///
    /// The model Hessian starts at `I`. A positive-definite model takes
    /// a dogleg step; an indefinite model takes the Cauchy point.
    ///
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Sr1,
    /// Powell symmetric Broyden in the SR1 trust region.
    ///
    /// The update is symmetric and satisfies the secant condition.
    /// Powell, *A new algorithm for unconstrained optimization*, in
    /// *Nonlinear Programming*, Academic Press, 1970.
    Sr2,
    /// Kingma and Ba Adam. The preconditioned direction is scaled to
    /// unit per-atom length and a line search sets the step.
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
    /// ODE12r: forward Euler on `P^{-1} F` with an embedded second-order
    /// error estimate, residual-controlled acceptance and an adaptive step.
    ///
    /// Makri, Ortner, Kermode, *A preconditioning scheme for minimum
    /// energy path finding methods*,
    /// <https://doi.org/10.1063/1.5064465>. The rule and the session
    /// clamps are in [`crate::ode12r`]; an installed geometry
    /// preconditioner is the metric. A Cartesian session keeps the
    /// default per-atom cap, and the stepper judges a capped move at
    /// the length it took.
    Ode12r,
    /// GDIIS geometry step.
    ///
    /// Pulay, *Convergence acceleration of iterative sequences. The
    /// case of SCF iteration*,
    /// <https://doi.org/10.1016/0009-2614(80)80396-4>.
    /// Császár and Pulay, J. Mol. Struct. 114, 31 (1984),
    /// <https://doi.org/10.1016/0022-2860(84)87198-7>.
    /// The image is `sum c_i (x_i - α g_i)`. A step past eight caps,
    /// an uphill image, or a step that does not lower the force drops
    /// the history and takes one steepest step. The line search chooses
    /// the length.
    Diis {
        /// Stored gradient and position pairs. At least 2.
        memory: usize,
    },
    /// Powell dogleg on a caller-supplied dense Hessian.
    ///
    /// Nocedal and Wright, algorithm 4.1,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Dogleg,
    /// Steihaug-Toint Newton-CG. Hessian access is a product `H v`,
    /// never a dense matrix.
    ///
    /// Steihaug, *The Conjugate Gradient Method and Trust Regions in
    /// Large Scale Optimization*,
    /// <https://doi.org/10.1137/0720042>.
    /// Nocedal and Wright, *Numerical Optimization*, chapter 7,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    NewtonCg,
    /// Steihaug-Toint Newton-CG with a Nyström preconditioner
    /// ([`crate::NystromPrecond`], rank 16) rebuilt at the current point.
    /// The trust ceiling stays Euclidean.
    ///
    /// Steihaug, <https://doi.org/10.1137/0720042>.
    /// Nocedal and Wright, chapter 7,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    /// Frangella, Tropp, Udell,
    /// <https://doi.org/10.1137/21M1466244>.
    NewtonPcg,
}

impl Method {
    /// Polak-Ribiere CG, no restart.
    pub fn polak_ribiere() -> Self {
        Self::Nlcg {
            conjugacy: Conjugacy::PolakRibiere,
            restart: Restart::Never,
        }
    }

    /// Pulay subspace of 6 gradient and position pairs.
    pub fn diis() -> Self {
        Self::Diis { memory: 6 }
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

    /// Whether a fresh Cartesian session installs [`crate::DEFAULT_ATOM_MAXMOVE`].
    ///
    /// BFGS takes its length from a strong Wolfe line search. Dogleg
    /// keeps the step inside a Euclidean trust region and accepts it
    /// from the model-agreement ratio. FIRE 2.0 sizes the move with
    /// its own `dt`. Those three leave the per-atom default off. Every
    /// other method installs 0.2.
    ///
    /// Nocedal and Wright, *Numerical Optimization*, algorithms 3.5,
    /// 3.6, and 4.1, <https://doi.org/10.1007/978-0-387-40065-5>.
    /// ASE `Optimizer.defaults` is `{"maxstep": 0.2}` in
    /// `ase/optimize/optimize.py` at tag 3.29.0,
    /// <https://gitlab.com/ase/ase/-/blob/3.29.0/ase/optimize/optimize.py>.
    /// eOn `OptimizerOptions.maxMove` is `0.2` in `schema/eon_params.capnp`
    /// at `4be157d4`,
    /// <https://github.com/TheochemUI/eOn/blob/4be157d43265af6b1140aaac0e52e42add62693f/schema/eon_params.capnp>.
    /// Guénolé et al., *Assessment and optimization of the fast inertial
    /// relaxation engine (FIRE)*,
    /// <https://doi.org/10.1016/j.commatsci.2020.109584>.
    ///
    /// An explicit cap still applies to every method. A non-positive
    /// value still clears it.
    pub(crate) fn installs_default_atom_cap(&self) -> bool {
        !matches!(
            self,
            Self::Bfgs
                | Self::Dogleg
                | Self::NewtonCg
                | Self::NewtonPcg
                | Self::Fire {
                    kind: crate::fire::FireKind::V2
                }
        )
    }

    /// Canonical method token. Hyphens, as in `newton-cg`.
    ///
    /// Parameterized variants report the family name (`lbfgs`, `adam`,
    /// `pso`, `cg`). [`Self::from_name`] rebuilds that family at its
    /// default parameters and with no NLCG restart.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Nlcg { conjugacy, .. } => match conjugacy {
                Conjugacy::FletcherReeves => "fletcher-reeves",
                Conjugacy::PolakRibiere => "cg",
                Conjugacy::HestenesStiefel => "hestenes-stiefel",
                Conjugacy::DaiYuan => "dai-yuan",
                Conjugacy::ConjugateDescent => "conjugate-descent",
                Conjugacy::HagerZhang => "hager-zhang",
                Conjugacy::LiuStorey => "liu-storey",
                Conjugacy::FrPr => "fr-pr",
                Conjugacy::PolakRibierePlus => "pr-plus",
                Conjugacy::Hybrid { .. } => "hybrid",
            },
            Self::Bfgs => "bfgs",
            Self::Lbfgs { .. } => "lbfgs",
            Self::Sr1 => "sr1",
            Self::Sr2 => "sr2",
            Self::Adam { .. } => "adam",
            Self::Steepest => "steepest",
            Self::Newton { kind } => match kind {
                crate::newton::NewtonKind::Shifted => "newton",
                crate::newton::NewtonKind::Rfo => "rfo",
            },
            Self::Pso { .. } => "pso",
            Self::Fire { kind } => match kind {
                crate::fire::FireKind::V1 => "fire",
                crate::fire::FireKind::V2 => "fire2",
            },
            Self::Bb => "bb",
            Self::Dogleg => "dogleg",
            Self::QuickMin => "quickmin",
            Self::Ode12r => "ode12r",
            Self::Diis { .. } => "diis",
            Self::NewtonCg => "newton-cg",
            Self::NewtonPcg => "newton-pcg",
        }
    }

    /// Parse a method token. Case is ignored. Hyphens and spaces are
    /// underscores, so `newton-cg` and `newton_cg` are the same method.
    ///
    /// `cg` is Polak-Ribiere. `newton` is shifted Newton. `sd` is
    /// steepest descent. L-BFGS memory, Adam, and PSO use the same
    /// defaults as [`Self::lbfgs`], [`Self::adam`], and [`Self::pso`].
    pub fn from_name(name: &str) -> crate::Result<Self> {
        let token = canon_method(name);
        let method = match token.as_str() {
            "lbfgs" => Self::lbfgs(),
            "bfgs" => Self::Bfgs,
            "sr1" => Self::Sr1,
            "sr2" => Self::Sr2,
            "adam" => Self::adam(),
            "sd" | "steepest" | "steepest_descent" => Self::Steepest,
            "cg" | "pr" | "polak_ribiere" => Self::polak_ribiere(),
            "fletcher_reeves" | "fr" => Self::nlcg(Conjugacy::FletcherReeves),
            "hestenes_stiefel" | "hs" => Self::nlcg(Conjugacy::HestenesStiefel),
            "dai_yuan" | "dy" => Self::nlcg(Conjugacy::DaiYuan),
            "conjugate_descent" | "cd" => Self::nlcg(Conjugacy::ConjugateDescent),
            "hager_zhang" | "hz" => Self::nlcg(Conjugacy::HagerZhang),
            "liu_storey" | "ls" => Self::nlcg(Conjugacy::LiuStorey),
            "fr_pr" => Self::nlcg(Conjugacy::FrPr),
            "pr_plus" | "polak_ribiere_plus" => Self::nlcg(Conjugacy::PolakRibierePlus),
            "newton" => Self::Newton {
                kind: crate::newton::NewtonKind::Shifted,
            },
            "rfo" => Self::Newton {
                kind: crate::newton::NewtonKind::Rfo,
            },
            "pso" => Self::pso(),
            "fire" => Self::Fire {
                kind: crate::fire::FireKind::V1,
            },
            "fire2" => Self::Fire {
                kind: crate::fire::FireKind::V2,
            },
            "bb" | "barzilai_borwein" => Self::Bb,
            "dogleg" => Self::Dogleg,
            "quickmin" | "quick_min" => Self::QuickMin,
            "ode12r" => Self::Ode12r,
            "diis" | "gdiis" => Self::diis(),
            "newton_cg" => Self::NewtonCg,
            "newton_pcg" => Self::NewtonPcg,
            _ => {
                return Err(crate::Error::UnknownMethod {
                    name: name.to_string(),
                });
            }
        };
        Ok(method)
    }
}

fn canon_method(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        let ch = if ch == '-' || ch == ' ' { '_' } else { ch };
        for lower in ch.to_lowercase() {
            out.push(lower);
        }
    }
    out
}

#[cfg(test)]
mod name_tests {
    use super::*;

    #[test]
    fn method_strings_cover_newton_cg_dogleg_and_dense_newton() {
        assert_eq!(Method::from_name("newton-cg").unwrap(), Method::NewtonCg);
        assert_eq!(Method::from_name("Newton-CG").unwrap(), Method::NewtonCg);
        assert_eq!(Method::from_name("newton_cg").unwrap(), Method::NewtonCg);
        assert_eq!(Method::from_name("newton-pcg").unwrap(), Method::NewtonPcg);
        assert_eq!(Method::NewtonCg.as_str(), "newton-cg");
        assert_eq!(Method::NewtonPcg.as_str(), "newton-pcg");
        assert_eq!(
            Method::from_name("newton").unwrap(),
            Method::Newton {
                kind: crate::newton::NewtonKind::Shifted,
            }
        );
        assert_eq!(
            Method::from_name("rfo").unwrap(),
            Method::Newton {
                kind: crate::newton::NewtonKind::Rfo,
            }
        );
        assert_eq!(Method::from_name("dogleg").unwrap(), Method::Dogleg);
        assert_eq!(Method::from_name("lbfgs").unwrap(), Method::lbfgs());
        assert_eq!(Method::from_name("CG").unwrap(), Method::polak_ribiere());
        assert_eq!(Method::from_name("sd").unwrap(), Method::Steepest);
        assert_eq!(Method::from_name("fire2").unwrap().as_str(), "fire2");
        let err = Method::from_name("not-a-solver").unwrap_err();
        assert!(err.to_string().contains("not-a-solver"), "{err}");
    }
}

impl Default for Method {
    fn default() -> Self {
        Self::polak_ribiere()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::FireKind;
    use crate::newton::NewtonKind;

    #[test]
    fn the_default_atom_cap_follows_the_step_control() {
        assert!(!Method::Bfgs.installs_default_atom_cap());
        assert!(!Method::Dogleg.installs_default_atom_cap());
        assert!(!Method::Fire { kind: FireKind::V2 }.installs_default_atom_cap());
        let capped = [
            Method::lbfgs(),
            Method::polak_ribiere(),
            Method::nlcg(crate::nlcg::Conjugacy::FletcherReeves),
            Method::Sr1,
            Method::Sr2,
            Method::adam(),
            Method::Steepest,
            Method::Newton {
                kind: NewtonKind::Shifted,
            },
            Method::Newton {
                kind: NewtonKind::Rfo,
            },
            Method::Fire { kind: FireKind::V1 },
            Method::Bb,
            Method::QuickMin,
            Method::diis(),
            Method::pso(),
        ];
        for method in capped {
            assert!(method.installs_default_atom_cap(), "{method:?}");
        }
    }
}
