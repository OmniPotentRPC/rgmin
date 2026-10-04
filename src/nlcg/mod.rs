//! Nonlinear CG: conjugacy coefficient and restart.

use ndarray::ArrayView1;

/// Shared vectors for a β evaluation.
#[derive(Clone, Debug)]
pub struct ConjugacyContext<'a> {
    /// `g_k`.
    pub current_gradient: ArrayView1<'a, f64>,
    /// `g_{k-1}`.
    pub previous_gradient: ArrayView1<'a, f64>,
    /// `d_{k-1}` (search direction, already a descent direction).
    pub previous_direction: ArrayView1<'a, f64>,
}

/// Conjugacy coefficient β. Formulas are Nocedal and Wright chapter 5
/// (<https://doi.org/10.1007/978-0-387-40065-5>).
#[derive(Clone, Debug, PartialEq)]
pub enum Conjugacy {
    /// Fletcher-Reeves, NJWS 5.41a: `||g||^2 / ||g_old||^2`.
    ///
    /// Fletcher and Reeves, *Function minimization by conjugate gradients*,
    /// <https://doi.org/10.1093/comjnl/7.2.149>.
    FletcherReeves,
    /// Polak-Ribiere, NJWS 5.44: `g·(g - g_old) / ||g_old||^2`.
    ///
    /// Polak and Ribiere, *Note sur la convergence de methodes de directions
    /// conjuguees* (1969).
    PolakRibiere,
    /// Hestenes-Stiefel, NJWS 5.46: `g·y / (y·d_old)`.
    ///
    /// Hestenes and Stiefel, *Methods of Conjugate Gradients for Solving
    /// Linear Systems* (1952).
    HestenesStiefel,
    /// Dai-Yuan, NJWS 5.49: `||g||^2 / (d_old·y)`.
    ///
    /// Dai and Yuan, *A Nonlinear Conjugate Gradient Method with a Strong
    /// Global Convergence Property*, <https://doi.org/10.1137/S1052623497318992>.
    DaiYuan,
    /// Fletcher conjugate descent: `||g||^2 / (- d_old·g_old)`.
    ///
    /// Same denominator as Liu-Storey. Hager and Zhang review the formula
    /// in *A New Conjugate Gradient Method with Guaranteed Descent and an
    /// Efficient Line Search*, <https://doi.org/10.1137/030601880>.
    ConjugateDescent,
    /// Hager-Zhang (2005) eq. 1.3 / NJWS 5.50.
    ///
    /// Hager and Zhang, *A New Conjugate Gradient Method with Guaranteed
    /// Descent and an Efficient Line Search*,
    /// <https://doi.org/10.1137/030601880>.
    HagerZhang,
    /// Liu-Storey (1991): `- g·y / (d_old·g_old)`.
    ///
    /// Liu and Storey, *Efficient generalized conjugate gradient algorithms,
    /// part 1: Theory*, <https://doi.org/10.1007/BF00940464>.
    LiuStorey,
    /// Gilbert-Nocedal FR-PR hybrid, NJWS 5.48.
    ///
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    FrPr,
    /// xtsci `HybridizedConj`: `max` or `min` of two child β formulas.
    ///
    /// This is not the Gilbert-Nocedal FR-PR hybrid ([`Self::FrPr`]).
    Hybrid {
        /// First formula.
        a: Box<Conjugacy>,
        /// Second formula.
        b: Box<Conjugacy>,
        /// `true` is `max(β_a, β_b)`; `false` is `min`.
        take_max: bool,
    },
}

/// Restart when conjugacy is lost.
///
/// Nocedal and Wright, *Numerical Optimization*,
/// <https://doi.org/10.1007/978-0-387-40065-5>.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Restart {
    /// Never force β = 0.
    Never,
    /// NJWS 5.52: restart when `|g·g_old| / ||g||^2 >= threshold`.
    ///
    /// Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    Njws {
        /// ν; book default is 0.1.
        threshold: f64,
    },
}

impl Restart {
    /// Default NJWS restart with ν = 0.1.
    pub fn njws() -> Self {
        Self::Njws { threshold: 0.1 }
    }

    /// True when the next step should reset to steepest descent.
    pub fn should_restart(&self, ctx: &ConjugacyContext<'_>) -> bool {
        match *self {
            Self::Never => false,
            Self::Njws { threshold } => {
                let gg = dot(ctx.current_gradient, ctx.current_gradient);
                if gg <= 0.0 {
                    return true;
                }
                let dev = dot(ctx.current_gradient, ctx.previous_gradient).abs() / gg;
                dev >= threshold
            }
        }
    }
}

impl Conjugacy {
    /// Combine two formulas with `max` (`take_max`) or `min`.
    pub fn hybrid(a: Conjugacy, b: Conjugacy, take_max: bool) -> Self {
        Self::Hybrid {
            a: Box::new(a),
            b: Box::new(b),
            take_max,
        }
    }

    /// β for this method. Degenerate denominators return 0.
    pub fn beta(&self, ctx: &ConjugacyContext<'_>) -> f64 {
        let g = ctx.current_gradient;
        let gold = ctx.previous_gradient;
        let d = ctx.previous_direction;
        let gg = dot(g, g);
        let gg_old = dot(gold, gold);
        let y = &g - &gold;
        let y_g = dot(g, y.view());
        match self {
            Self::FletcherReeves => div(gg, gg_old),
            Self::PolakRibiere => div(y_g, gg_old),
            Self::HestenesStiefel => {
                let y_d = dot(y.view(), d);
                div(y_g, y_d)
            }
            Self::DaiYuan => {
                let y_d = dot(y.view(), d);
                div(gg, y_d)
            }
            Self::ConjugateDescent => div(gg, -dot(d, gold)),
            Self::HagerZhang => {
                let y_d = dot(y.view(), d);
                if y_d.abs() <= f64::EPSILON {
                    return 0.0;
                }
                let yy = dot(y.view(), y.view());
                let d_g = dot(d, g);
                y_g / y_d - 2.0 * yy * d_g / (y_d * y_d)
            }
            Self::LiuStorey => div(-y_g, dot(d, gold)),
            Self::FrPr => {
                let beta_pr = Self::PolakRibiere.beta(ctx);
                let beta_fr = Self::FletcherReeves.beta(ctx);
                if beta_pr < -beta_fr {
                    -beta_fr
                } else if beta_pr.abs() <= beta_fr {
                    beta_pr
                } else {
                    beta_fr
                }
            }
            Self::Hybrid { a, b, take_max } => {
                let ba = a.beta(ctx);
                let bb = b.beta(ctx);
                if *take_max { ba.max(bb) } else { ba.min(bb) }
            }
        }
    }
}

fn dot(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn div(num: f64, den: f64) -> f64 {
    if den.abs() <= f64::EPSILON {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn textbook_betas_on_orthogonal_gradients() {
        let g = array![1.0, 0.0];
        let gold = array![0.0, 1.0];
        let d = array![-1.0, 0.0];
        let ctx = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold.view(),
            previous_direction: d.view(),
        };
        assert_eq!(Conjugacy::FletcherReeves.beta(&ctx), 1.0);
        assert_eq!(Conjugacy::PolakRibiere.beta(&ctx), 1.0);
        // y = [1, -1], y·d = -1, g·y = 1, β_HS = -1
        assert_eq!(Conjugacy::HestenesStiefel.beta(&ctx), -1.0);
        // β_DY = 1 / (d·y) = 1 / -1 = -1
        assert_eq!(Conjugacy::DaiYuan.beta(&ctx), -1.0);
        // d·g_old = 0, so CD and LS denominators vanish
        assert_eq!(Conjugacy::ConjugateDescent.beta(&ctx), 0.0);
        assert_eq!(Conjugacy::LiuStorey.beta(&ctx), 0.0);
        // ŷ form: y=[1,-1], ||y||^2=2, d·y=-1, d·g=-1
        // β_HZ = 1/(-1) - 2*2*(-1)/1 = -1 + 4 = 3
        assert_eq!(Conjugacy::HagerZhang.beta(&ctx), 3.0);
        // |β_PR| = |β_FR| = 1, so FR-PR returns β_PR
        assert_eq!(Conjugacy::FrPr.beta(&ctx), 1.0);
    }

    #[test]
    fn conjugate_descent_uses_minus_d_dot_g_old() {
        let g = array![2.0, 0.0];
        let gold = array![1.0, 0.0];
        let d = array![-1.0, 0.0];
        let ctx = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold.view(),
            previous_direction: d.view(),
        };
        // ||g||^2 = 4, d·g_old = -1, β_CD = 4 / (-(-1)) = 4
        assert_eq!(Conjugacy::ConjugateDescent.beta(&ctx), 4.0);
        // g·y = 2, β_LS = -2 / (-1) = 2
        assert_eq!(Conjugacy::LiuStorey.beta(&ctx), 2.0);
        // β_FR = 4, β_PR = 2, |PR| < FR so FR-PR returns PR
        assert_eq!(Conjugacy::FrPr.beta(&ctx), 2.0);
        // HybridizedConj(PR, FR, max) is 4, not the Gilbert-Nocedal clamp.
        assert_eq!(
            Conjugacy::hybrid(Conjugacy::PolakRibiere, Conjugacy::FletcherReeves, true).beta(&ctx),
            4.0
        );
        // y=[1,0], ||y||^2=1, d·y=-1, d·g=-2
        // β_HZ = 2/(-1) - 2*1*(-2)/1 = -2 + 4 = 2
        assert_eq!(Conjugacy::HagerZhang.beta(&ctx), 2.0);
    }

    #[test]
    fn hybrid_pr_fr_take_max_is_max_of_the_two_betas() {
        let g = array![1.0, 0.0];
        let gold = array![0.0, 1.0];
        let d = array![-1.0, 0.0];
        let ctx = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold.view(),
            previous_direction: d.view(),
        };
        let beta_pr = Conjugacy::PolakRibiere.beta(&ctx);
        let beta_fr = Conjugacy::FletcherReeves.beta(&ctx);
        let hybrid = Conjugacy::Hybrid {
            a: Box::new(Conjugacy::PolakRibiere),
            b: Box::new(Conjugacy::FletcherReeves),
            take_max: true,
        };
        assert_eq!(hybrid.beta(&ctx), beta_pr.max(beta_fr));
        assert_eq!(hybrid.beta(&ctx), 1.0);
        // Hybrid is max/min of the two formulas, not Gilbert-Nocedal FR-PR.
        let nested = Conjugacy::hybrid(
            Conjugacy::hybrid(Conjugacy::PolakRibiere, Conjugacy::FletcherReeves, true),
            Conjugacy::HestenesStiefel,
            false,
        );
        assert_eq!(nested.beta(&ctx), beta_pr.max(beta_fr).min(-1.0));
    }

    #[test]
    fn fr_pr_clamps_below_minus_fr_hybrid_does_not() {
        let g = array![1.0, 0.0];
        let gold = array![3.0, 1.0];
        let d = array![-3.0, -1.0];
        let ctx = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold.view(),
            previous_direction: d.view(),
        };
        // ||g||^2 = 1, ||g_old||^2 = 10, g·y = -2
        // β_FR = 0.1, β_PR = -0.2; Gilbert-Nocedal 5.48 returns -β_FR
        assert_eq!(Conjugacy::FletcherReeves.beta(&ctx), 0.1);
        assert_eq!(Conjugacy::PolakRibiere.beta(&ctx), -0.2);
        assert_eq!(Conjugacy::FrPr.beta(&ctx), -0.1);
        let take_max = Conjugacy::hybrid(Conjugacy::PolakRibiere, Conjugacy::FletcherReeves, true);
        let take_min = Conjugacy::hybrid(Conjugacy::PolakRibiere, Conjugacy::FletcherReeves, false);
        assert_eq!(take_max.beta(&ctx), 0.1);
        assert_eq!(take_min.beta(&ctx), -0.2);
        assert_ne!(take_max.beta(&ctx), Conjugacy::FrPr.beta(&ctx));
        assert_ne!(take_min.beta(&ctx), Conjugacy::FrPr.beta(&ctx));
    }

    #[test]
    fn liu_storey_is_the_gpr_optim_scg_inl_beta() {
        // SCG.inl: mu = p·r on the pre-step residual; after accept,
        // beta = (r_old - r).dot(r) / mu. That is Liu-Storey
        // -g·(g-g_old)/(d·g_old). Restart (d = -g_old) equals PR.
        let g = array![2.0, 0.0];
        let gold = array![1.0, 0.0];
        let d = -&gold;
        let ctx = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold.view(),
            previous_direction: d.view(),
        };
        assert_eq!(
            Conjugacy::LiuStorey.beta(&ctx),
            Conjugacy::PolakRibiere.beta(&ctx)
        );
        assert_eq!(Conjugacy::LiuStorey.beta(&ctx), 2.0);

        let g2 = array![2.0, 1.0];
        let gold2 = array![1.0, 0.0];
        let d2 = array![-0.5, 1.0];
        let ctx2 = ConjugacyContext {
            current_gradient: g2.view(),
            previous_gradient: gold2.view(),
            previous_direction: d2.view(),
        };
        assert_eq!(Conjugacy::LiuStorey.beta(&ctx2), 6.0);
        assert_ne!(
            Conjugacy::LiuStorey.beta(&ctx2),
            Conjugacy::PolakRibiere.beta(&ctx2)
        );
    }

    #[test]
    fn njws_5_52_uses_current_gradient_norm() {
        let g = array![10.0, 0.0];
        let gold = array![0.5, 0.0];
        let d = array![-0.5, 0.0];
        let ctx = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold.view(),
            previous_direction: d.view(),
        };
        // |g·g_old| / ||g||^2 = 5 / 100 = 0.05 < 0.1
        // The ||g_old||^2 form is 5 / 0.25 = 20, which is not 5.52.
        assert!(!Restart::njws().should_restart(&ctx));
        assert!(Restart::Njws { threshold: 0.04 }.should_restart(&ctx));

        let g2 = array![1.0, 0.0];
        let gold2 = array![0.2, 0.0];
        let ctx2 = ConjugacyContext {
            current_gradient: g2.view(),
            previous_gradient: gold2.view(),
            previous_direction: d.view(),
        };
        // |g·g_old| / ||g||^2 = 0.2 / 1 = 0.2 >= 0.1
        assert!(Restart::njws().should_restart(&ctx2));
        assert!(!Restart::Never.should_restart(&ctx2));
    }
}
