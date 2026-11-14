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
    /// Polak-Ribiere+, NJWS 5.45: `max(β_PR, 0)`.
    ///
    /// The same clamp applies in the Euclidean product and in the metric
    /// product. Nocedal and Wright, *Numerical Optimization*,
    /// <https://doi.org/10.1007/978-0-387-40065-5>.
    PolakRibierePlus,
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

    /// Powell restart: `|g · g_old| >= 0.2 ||g||^2`.
    ///
    /// The same normalized test as [`Self::njws`], with threshold 0.2.
    pub fn powell() -> Self {
        Self::Njws { threshold: 0.2 }
    }

    /// True when the next step should reset to steepest descent.
    pub fn should_restart(&self, ctx: &ConjugacyContext<'_>) -> bool {
        self.should_restart_metric(
            ctx.current_gradient,
            ctx.current_gradient,
            ctx.previous_gradient,
            ctx.previous_gradient,
        )
    }

    /// [`Self::should_restart`] in the metric `<u, v>_P`, with `z = P^{-1} g`.
    pub fn should_restart_metric(
        &self,
        g: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        g_old: ArrayView1<'_, f64>,
        z_old: ArrayView1<'_, f64>,
    ) -> bool {
        if g.len() != z.len() || g_old.len() != z_old.len() {
            return true;
        }
        match *self {
            Self::Never => false,
            Self::Njws { threshold } => {
                let gg = dot(g, z);
                if gg <= 0.0 {
                    return true;
                }
                let dev = dot(g, z_old).abs() / gg;
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
        match self {
            Self::FletcherReeves => div(dot(g, g), dot(gold, gold)),
            Self::PolakRibiere => {
                let (y_g, gg_old) = dot_diff_and_dot(g, g, gold, gold);
                div(y_g, gg_old)
            }
            Self::PolakRibierePlus => Self::PolakRibiere.beta(ctx).max(0.0),
            Self::HestenesStiefel => {
                let (y_g, y_d) = diff_dots(g, g, gold, d);
                div(y_g, y_d)
            }
            Self::DaiYuan => div(dot(g, g), diff_dot(g, gold, d)),
            Self::ConjugateDescent => div(dot(g, g), -dot(d, gold)),
            Self::HagerZhang => {
                let (y_g, y_d, yy) = diff_dots_self(g, g, gold, d);
                if y_d.abs() <= f64::EPSILON {
                    return 0.0;
                }
                let d_g = dot(d, g);
                y_g / y_d - 2.0 * yy * d_g / (y_d * y_d)
            }
            Self::LiuStorey => {
                let (y_g, _) = dot_diff_and_dot(g, g, gold, gold);
                div(-y_g, dot(d, gold))
            }
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

    /// β in the metric whose Riesz gradient is `z = P^{-1} g`.
    ///
    /// Gradient norms use `g · z`. The Polak-Ribiere numerator is
    /// `z · (g - g_old)`, the preconditioned form. It equals
    /// `g · (z - z_old)` when both maps use the same SPD matrix.
    /// Inner products of a gradient with a direction stay `g · d`.
    /// Passing `z = g` and `z_old = g_old` reproduces [`Self::beta`].
    pub fn beta_metric(
        &self,
        ctx: &ConjugacyContext<'_>,
        z: ArrayView1<'_, f64>,
        z_old: ArrayView1<'_, f64>,
    ) -> f64 {
        let g = ctx.current_gradient;
        let gold = ctx.previous_gradient;
        let d = ctx.previous_direction;
        // Each arm forms only the products it returns. A product that the
        // formula does not read is left out; the ones it does read stay in
        // the same association.
        match self {
            Self::FletcherReeves => {
                let gz = dot(g, z);
                div(gz, dot(gold, z_old))
            }
            Self::PolakRibiere => {
                let (zy, gz_old) = dot_diff_and_dot(z, g, gold, z_old);
                div(zy, gz_old)
            }
            Self::PolakRibierePlus => Self::PolakRibiere.beta_metric(ctx, z, z_old).max(0.0),
            Self::HestenesStiefel => {
                let (zy, y_d) = diff_dots(z, g, gold, d);
                div(zy, y_d)
            }
            Self::DaiYuan => div(dot(g, z), diff_dot(g, gold, d)),
            Self::ConjugateDescent => {
                let gz = dot(g, z);
                div(gz, -dot(d, gold))
            }
            Self::HagerZhang => {
                // y is the Euclidean gradient change. z · y is the
                // preconditioned Polak-Ribiere product. yy is y · (z - z_old).
                let (zy, y_d, yy) = diff_dots_cross(z, g, gold, d, z, z_old);
                if y_d.abs() <= f64::EPSILON {
                    return 0.0;
                }
                let d_g = dot(d, g);
                zy / y_d - 2.0 * yy * d_g / (y_d * y_d)
            }
            Self::LiuStorey => {
                let (zy, _) = dot_diff_and_dot(z, g, gold, z_old);
                div(-zy, dot(d, gold))
            }
            Self::FrPr => {
                let beta_pr = Self::PolakRibiere.beta_metric(ctx, z, z_old);
                let beta_fr = Self::FletcherReeves.beta_metric(ctx, z, z_old);
                if beta_pr < -beta_fr {
                    -beta_fr
                } else if beta_pr.abs() <= beta_fr {
                    beta_pr
                } else {
                    beta_fr
                }
            }
            Self::Hybrid { a, b, take_max } => {
                let ba = a.beta_metric(ctx, z, z_old);
                let bb = b.beta_metric(ctx, z, z_old);
                if *take_max { ba.max(bb) } else { ba.min(bb) }
            }
        }
    }
}

fn dot(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    sum_products(a.iter().zip(b.iter()).map(|(x, y)| x * y))
}

/// Left fold of products from the `f64` identity of [`Iterator::sum`].
///
/// On the toolchain this crate pins (Rust 1.88) that identity is `-0.0`,
/// not `+0.0`. An empty sum, or a sum of `-0.0` products, keeps the sign.
fn sum_products(products: impl Iterator<Item = f64>) -> f64 {
    products.fold(-0.0, |acc, x| acc + x)
}

fn sum_identity() -> f64 {
    sum_products(std::iter::empty())
}

/// `z · (g - g_old)` and `g_old · z_old`, each a left fold of those products.
///
/// The difference is formed once per index and multiplied, which is the
/// product a stored `g - g_old` feeds to [`dot`]. Both accumulators start
/// at the same identity [`sum_products`] uses.
fn dot_diff_and_dot(
    z: ArrayView1<'_, f64>,
    g: ArrayView1<'_, f64>,
    gold: ArrayView1<'_, f64>,
    z_old: ArrayView1<'_, f64>,
) -> (f64, f64) {
    let n = z.len().min(g.len()).min(gold.len()).min(z_old.len());
    (0..n).fold((sum_identity(), sum_identity()), |(zy, gz), i| {
        let y = g[i] - gold[i];
        (zy + z[i] * y, gz + gold[i] * z_old[i])
    })
}

/// `a · (u - v)` and `(u - v) · b`, each a left fold from [`sum_identity`].
///
/// `a · d` is the product [`dot`] forms with a stored difference on the
/// left. `(u - v) · b` is the product with that difference on the left of
/// the second factor, which is what `dot(y, b)` multiplies.
fn diff_dots(
    a: ArrayView1<'_, f64>,
    u: ArrayView1<'_, f64>,
    v: ArrayView1<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> (f64, f64) {
    let n = a.len().min(u.len()).min(v.len()).min(b.len());
    (0..n).fold((sum_identity(), sum_identity()), |(ad, db), i| {
        let y = u[i] - v[i];
        (ad + a[i] * y, db + y * b[i])
    })
}

/// `(u - v) · b` from [`sum_identity`].
fn diff_dot(u: ArrayView1<'_, f64>, v: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    let n = u.len().min(v.len()).min(b.len());
    (0..n).fold(sum_identity(), |acc, i| {
        let y = u[i] - v[i];
        acc + y * b[i]
    })
}

/// `a · (u - v)`, `(u - v) · b`, and `(u - v) · (u - v)`.
fn diff_dots_self(
    a: ArrayView1<'_, f64>,
    u: ArrayView1<'_, f64>,
    v: ArrayView1<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> (f64, f64, f64) {
    let n = a.len().min(u.len()).min(v.len()).min(b.len());
    (0..n).fold(
        (sum_identity(), sum_identity(), sum_identity()),
        |(ad, db, yy), i| {
            let y = u[i] - v[i];
            (ad + a[i] * y, db + y * b[i], yy + y * y)
        },
    )
}

/// `a · (u - v)`, `(u - v) · b`, and `(u - v) · (p - q)`.
fn diff_dots_cross(
    a: ArrayView1<'_, f64>,
    u: ArrayView1<'_, f64>,
    v: ArrayView1<'_, f64>,
    b: ArrayView1<'_, f64>,
    p: ArrayView1<'_, f64>,
    q: ArrayView1<'_, f64>,
) -> (f64, f64, f64) {
    let n = a
        .len()
        .min(u.len())
        .min(v.len())
        .min(b.len())
        .min(p.len())
        .min(q.len());
    (0..n).fold(
        (sum_identity(), sum_identity(), sum_identity()),
        |(ad, db, yy), i| {
            let y = u[i] - v[i];
            let yz = p[i] - q[i];
            (ad + a[i] * y, db + y * b[i], yy + y * yz)
        },
    )
}

/// `num/den`, or 0 when `den` is negligible beside `num`.
///
/// The test is `|den| <= eps (|num| + 1)`. The `+ 1` is an absolute
/// floor, in the same units as `num`. A denominator smaller than `eps`
/// is dropped even when `num` is zero, which keeps a vanishing inner
/// product from returning a huge β. When `|num|` is large the `|num|`
/// term dominates, so the test is relative and a genuine ratio is kept.
/// β is already dimensionless, so the floor does not have to be scaled
/// by a length unit.
fn div(num: f64, den: f64) -> f64 {
    if !num.is_finite() || !den.is_finite() || den.abs() <= f64::EPSILON * (num.abs() + 1.0) {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

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
    fn preconditioned_polak_ribiere_dots_z_with_the_gradient_change() {
        let g = array![1.0, 1.0];
        let gold = array![1.0, 0.0];
        let z = array![4.0, 0.5];
        let z_old = array![0.25, 2.0];
        let d = array![-1.0, 0.0];
        let ctx = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold.view(),
            previous_direction: d.view(),
        };
        // y = (0, 1), z · y = 0.5, g_old · z_old = 0.25, β = 2.
        assert!(
            (Conjugacy::PolakRibiere.beta_metric(&ctx, z.view(), z_old.view()) - 2.0).abs() < 1e-12
        );
    }

    #[test]
    fn the_metric_beta_is_the_euclidean_beta_when_p_is_the_identity() {
        let g = array![1.0, -2.0, 0.5];
        let gold = array![0.25, 0.5, -1.0];
        let d = array![-0.5, 0.1, 0.2];
        let ctx = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold.view(),
            previous_direction: d.view(),
        };
        for method in [
            Conjugacy::FletcherReeves,
            Conjugacy::PolakRibiere,
            Conjugacy::HestenesStiefel,
            Conjugacy::DaiYuan,
            Conjugacy::ConjugateDescent,
            Conjugacy::HagerZhang,
            Conjugacy::LiuStorey,
            Conjugacy::FrPr,
        ] {
            let plain = method.beta(&ctx);
            let metric = method.beta_metric(&ctx, g.view(), gold.view());
            assert_eq!(plain, metric, "{method:?}");
        }
        assert_eq!(
            Restart::njws().should_restart(&ctx),
            Restart::njws().should_restart_metric(g.view(), g.view(), gold.view(), gold.view())
        );
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
        assert_eq!(Conjugacy::PolakRibierePlus.beta(&ctx), 0.0);
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
    fn fused_polak_ribiere_products_match_sum_bit_for_bit() {
        fn reference(
            z: ArrayView1<f64>,
            g: ArrayView1<f64>,
            gold: ArrayView1<f64>,
            z_old: ArrayView1<f64>,
        ) -> (f64, f64) {
            let y = &g - &gold;
            let zy: f64 = z.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            let gz: f64 = gold.iter().zip(z_old.iter()).map(|(a, b)| a * b).sum();
            (zy, gz)
        }
        fn check(z: Array1<f64>, g: Array1<f64>, gold: Array1<f64>, z_old: Array1<f64>) {
            let (zy, gz) = dot_diff_and_dot(z.view(), g.view(), gold.view(), z_old.view());
            let (zy_old, gz_old) = reference(z.view(), g.view(), gold.view(), z_old.view());
            assert_eq!(zy.to_bits(), zy_old.to_bits(), "zy {z:?} {g:?} {gold:?}");
            assert_eq!(gz.to_bits(), gz_old.to_bits(), "gz {gold:?} {z_old:?}");
        }

        let empty = Array1::<f64>::zeros(0);
        let (zy, gz) = dot_diff_and_dot(empty.view(), empty.view(), empty.view(), empty.view());
        let neg0 = (-0.0f64).to_bits();
        assert_eq!(zy.to_bits(), neg0);
        assert_eq!(gz.to_bits(), neg0);
        check(empty.clone(), empty.clone(), empty.clone(), empty);

        // Every product is -0.0: (g - g_old) is -0, times +0; g_old is +0 times -0.
        let z = array![0.0, 0.0, 0.0];
        let g = array![-0.0, -0.0, -0.0];
        let gold = array![0.0, 0.0, 0.0];
        let z_old = array![-0.0, -0.0, -0.0];
        let (zy, gz) = dot_diff_and_dot(z.view(), g.view(), gold.view(), z_old.view());
        assert_eq!(zy.to_bits(), neg0);
        assert_eq!(gz.to_bits(), neg0);
        check(z, g, gold, z_old);

        check(
            array![f64::NAN, 1.0],
            array![2.0, f64::NAN],
            array![f64::INFINITY, -1.0],
            array![-0.0, f64::NEG_INFINITY],
        );
        check(
            array![f64::NEG_INFINITY, 0.0, -0.0],
            array![1.0, f64::INFINITY, -0.0],
            array![-0.0, 3.0, 0.0],
            array![f64::NAN, -0.0, f64::INFINITY],
        );

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(0x6c62_272e_07bb_0142)
                .wrapping_add(0x2f4a_8d6c_91e5_b3a7);
            f64::from_bits(state)
        };
        for n in [1usize, 2, 7, 8, 17, 72] {
            let z = Array1::from_iter((0..n).map(|_| next()));
            let g = Array1::from_iter((0..n).map(|_| next()));
            let gold = Array1::from_iter((0..n).map(|_| next()));
            let z_old = Array1::from_iter((0..n).map(|_| next()));
            check(z, g, gold, z_old);
        }
    }

    #[test]
    fn other_betas_match_the_allocated_difference_bit_for_bit() {
        fn sum_dot(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }
        fn check(
            z: &Array1<f64>,
            g: &Array1<f64>,
            gold: &Array1<f64>,
            d: &Array1<f64>,
            z_old: &Array1<f64>,
        ) {
            let y = g - gold;
            let yz = z - z_old;
            let yg: f64 = sum_dot(g.view(), y.view());
            let yd: f64 = sum_dot(y.view(), d.view());
            let yy: f64 = sum_dot(y.view(), y.view());
            let zy: f64 = sum_dot(z.view(), y.view());
            let y_yz: f64 = sum_dot(y.view(), yz.view());
            let (yg_f, yd_f) = diff_dots(g.view(), g.view(), gold.view(), d.view());
            let (zy_f, yd_m) = diff_dots(z.view(), g.view(), gold.view(), d.view());
            let yd_only = diff_dot(g.view(), gold.view(), d.view());
            let (yg_s, yd_s, yy_s) = diff_dots_self(g.view(), g.view(), gold.view(), d.view());
            let (zy_c, yd_c, yy_c) = diff_dots_cross(
                z.view(),
                g.view(),
                gold.view(),
                d.view(),
                z.view(),
                z_old.view(),
            );
            assert_eq!(yg_f.to_bits(), yg.to_bits());
            assert_eq!(yd_f.to_bits(), yd.to_bits());
            assert_eq!(zy_f.to_bits(), zy.to_bits());
            assert_eq!(yd_m.to_bits(), yd.to_bits());
            assert_eq!(yd_only.to_bits(), yd.to_bits());
            assert_eq!(yg_s.to_bits(), yg.to_bits());
            assert_eq!(yd_s.to_bits(), yd.to_bits());
            assert_eq!(yy_s.to_bits(), yy.to_bits());
            assert_eq!(zy_c.to_bits(), zy.to_bits());
            assert_eq!(yd_c.to_bits(), yd.to_bits());
            assert_eq!(yy_c.to_bits(), y_yz.to_bits());

            let ctx = ConjugacyContext {
                current_gradient: g.view(),
                previous_gradient: gold.view(),
                previous_direction: d.view(),
            };
            let gg: f64 = sum_dot(g.view(), g.view());
            let gg_old: f64 = sum_dot(gold.view(), gold.view());
            let gz: f64 = sum_dot(g.view(), z.view());
            let gz_old: f64 = sum_dot(gold.view(), z_old.view());
            assert_eq!(
                Conjugacy::FletcherReeves.beta(&ctx).to_bits(),
                div(gg, gg_old).to_bits()
            );
            assert_eq!(
                Conjugacy::HestenesStiefel.beta(&ctx).to_bits(),
                div(yg, yd).to_bits()
            );
            assert_eq!(
                Conjugacy::DaiYuan.beta(&ctx).to_bits(),
                div(gg, yd).to_bits()
            );
            assert_eq!(
                Conjugacy::FletcherReeves
                    .beta_metric(&ctx, z.view(), z_old.view())
                    .to_bits(),
                div(gz, gz_old).to_bits()
            );
            assert_eq!(
                Conjugacy::HestenesStiefel
                    .beta_metric(&ctx, z.view(), z_old.view())
                    .to_bits(),
                div(zy, yd).to_bits()
            );
            assert_eq!(
                Conjugacy::DaiYuan
                    .beta_metric(&ctx, z.view(), z_old.view())
                    .to_bits(),
                div(gz, yd).to_bits()
            );
        }

        let empty = Array1::<f64>::zeros(0);
        check(&empty, &empty, &empty, &empty, &empty);

        let z = array![0.0, 0.0];
        let g = array![-0.0, -0.0];
        let gold = array![0.0, 0.0];
        let d = array![0.0, 0.0];
        let z_old = array![-0.0, -0.0];
        check(&z, &g, &gold, &d, &z_old);
        let y = &g - &gold;
        let yd: f64 = y.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
        assert_eq!(yd.to_bits(), (-0.0f64).to_bits());

        check(
            &array![f64::NAN, 1.0, f64::INFINITY],
            &array![2.0, f64::NAN, -0.0],
            &array![f64::NEG_INFINITY, -1.0, 0.0],
            &array![-0.0, f64::INFINITY, 3.0],
            &array![0.0, f64::NAN, -4.0],
        );

        let mut state = 0x0f1e_2d3c_4b5a_6978u64;
        let mut next = || {
            state = state
                .wrapping_mul(0x6c62_272e_07bb_0142)
                .wrapping_add(0x2f4a_8d6c_91e5_b3a7);
            f64::from_bits(state)
        };
        for n in [1usize, 3, 8, 17] {
            let z = Array1::from_iter((0..n).map(|_| next()));
            let g = Array1::from_iter((0..n).map(|_| next()));
            let gold = Array1::from_iter((0..n).map(|_| next()));
            let d = Array1::from_iter((0..n).map(|_| next()));
            let z_old = Array1::from_iter((0..n).map(|_| next()));
            check(&z, &g, &gold, &d, &z_old);
        }
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

    #[test]
    fn powell_restarts_at_a_fifth_of_the_squared_gradient() {
        // |g·g_old| = 0.19 ||g||^2 restarts NJWS (0.1) and not Powell (0.2).
        let g = array![1.0, 0.0];
        let gold = array![0.19, 0.0];
        let d = array![-1.0, 0.0];
        let ctx = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold.view(),
            previous_direction: d.view(),
        };
        assert!(Restart::njws().should_restart(&ctx));
        assert!(!Restart::powell().should_restart(&ctx));
        let gold_at = array![0.2, 0.0];
        let at = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold_at.view(),
            previous_direction: d.view(),
        };
        assert!(Restart::powell().should_restart(&at));
        assert_eq!(Restart::powell(), Restart::Njws { threshold: 0.2 });
    }
}
