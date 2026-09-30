//! Kennedy-Eberhart particle swarm over the objective box.

use eindir_core::{Bounds, DifferentiableObjective, Objective};
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::control::Control;
use crate::error::{Error, Result};
use crate::report::Report;
use crate::step::l2;

/// Swarm RNG seed. Tests and the public API share this value.
pub(crate) const RNG_SEED: u64 = 1;
const STAGNANT_LIMIT: usize = 50;
const AVG_VEL_TOL: f64 = 1e-8;

pub(crate) struct Particle {
    pub(crate) position: Array1<f64>,
    pub(crate) velocity: Array1<f64>,
    pub(crate) best_position: Array1<f64>,
    pub(crate) best_value: f64,
}

/// Particle swarm (xtsci `PSOptim`) over the objective box. Line search is unused.
///
/// Velocity is Kennedy-Eberhart with the xtsci inertia weight:
/// `v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x)`, with one scalar
/// `r1`, `r2` in `[0, 1)` per particle, then `Vmax` clamp and a
/// reflective box. `gbest` updates inside the swarm pass. The swarm RNG
/// is `StdRng` seeded with `1`.
///
/// Kennedy and Eberhart, *Particle swarm optimization*,
/// <https://doi.org/10.1109/ICNN.1995.488968>.
pub fn minimize_pso<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    n_particles: usize,
    inertia: f64,
    c1: f64,
    c2: f64,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = init.into();
    if pos.len() != Objective::dim(obj) {
        return Err(Error::Dim {
            got: pos.len(),
            dim: Objective::dim(obj),
        });
    }
    let bounds = obj.bounds();
    pos = bounds.clip(pos.view());
    let n_particles = n_particles.max(1);
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let start_val = obj.eval(pos.view());
    let mut swarm = Vec::with_capacity(n_particles);
    swarm.push(Particle {
        velocity: random_velocity(bounds, &mut rng),
        best_position: pos.clone(),
        best_value: start_val,
        position: pos,
    });
    let mut gbest_position = swarm[0].best_position.clone();
    let mut gbest_value = start_val;
    for _ in 1..n_particles {
        let position = bounds.mkpoint(&mut rng);
        let value = obj.eval(position.view());
        let particle = Particle {
            velocity: random_velocity(bounds, &mut rng),
            best_position: position.clone(),
            best_value: value,
            position,
        };
        if particle.best_value < gbest_value {
            gbest_position = particle.best_position.clone();
            gbest_value = particle.best_value;
        }
        swarm.push(particle);
    }

    let mut prev_gbest_value = f64::INFINITY;
    let mut iteration = 0;
    let mut stagnant_iterations = 0;
    while iteration < control.maxiter {
        update_swarm(
            obj,
            bounds,
            &mut swarm,
            &mut gbest_position,
            &mut gbest_value,
            inertia,
            c1,
            c2,
            &mut rng,
        );
        let avg_velocity = average_velocity(&swarm);
        if gbest_value == prev_gbest_value {
            stagnant_iterations += 1;
        } else {
            stagnant_iterations = 0;
        }
        if stagnant_iterations > STAGNANT_LIMIT || avg_velocity < AVG_VEL_TOL {
            break;
        }
        prev_gbest_value = gbest_value;
        iteration += 1;
    }

    let (value, grad) = obj.value_and_gradient(gbest_position.view());
    Ok(Report {
        value,
        coords: gbest_position,
        steps: iteration,
        grad_norm: l2(&grad),
    })
}

pub(crate) fn update_swarm<O>(
    obj: &O,
    bounds: &Bounds<f64>,
    swarm: &mut [Particle],
    gbest_position: &mut Array1<f64>,
    gbest_value: &mut f64,
    inertia: f64,
    c1: f64,
    c2: f64,
    rng: &mut StdRng,
) where
    O: Objective<f64> + ?Sized,
{
    let span = &bounds.high - &bounds.low;
    let vmax = 0.5 * l2(&span);
    let n = bounds.dims;
    for particle in swarm.iter_mut() {
        let r1 = rng.random::<f64>();
        let r2 = rng.random::<f64>();
        for i in 0..n {
            let v = inertia * particle.velocity[i]
                + c1 * r1 * (particle.best_position[i] - particle.position[i])
                + c2 * r2 * (gbest_position[i] - particle.position[i]);
            particle.velocity[i] = v.clamp(-vmax, vmax);
            particle.position[i] += particle.velocity[i];
        }
        let clipped = bounds.clip(particle.position.view());
        for i in 0..n {
            if clipped[i] == bounds.low[i] || clipped[i] == bounds.high[i] {
                particle.velocity[i] = -particle.velocity[i];
            }
        }
        particle.position = clipped;
        let new_value = obj.eval(particle.position.view());
        if new_value < particle.best_value {
            particle.best_value = new_value;
            particle.best_position = particle.position.clone();
            if new_value < *gbest_value {
                *gbest_position = particle.best_position.clone();
                *gbest_value = new_value;
            }
        }
    }
}

pub(crate) fn random_velocity<R: Rng>(bounds: &Bounds<f64>, rng: &mut R) -> Array1<f64> {
    Array1::from_iter((0..bounds.dims).map(|i| {
        let span = (bounds.high[i] - bounds.low[i]).abs();
        rng.random_range(-span..=span)
    }))
}

fn average_velocity(swarm: &[Particle]) -> f64 {
    if swarm.is_empty() {
        return 0.0;
    }
    swarm.iter().map(|p| l2(&p.velocity)).sum::<f64>() / swarm.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::objectives::Rosenbrock;
    use ndarray::array;

    fn control() -> Control {
        Control {
            maxiter: 30,
            gtol: 1e-8,
            istep: 0.1,
            maxmove: None,
            ftol_rel: None,
        }
    }

    #[test]
    fn seed_one_is_deterministic() {
        let obj = Rosenbrock::<2>::new();
        let start = array![-1.2, 1.0];
        let a = minimize_pso(&obj, start.clone(), &control(), 10, 0.5, 1.5, 1.5).unwrap();
        let b = minimize_pso(&obj, start, &control(), 10, 0.5, 1.5, 1.5).unwrap();
        assert_eq!(a.value, b.value);
        assert_eq!(a.coords, b.coords);
        assert_eq!(a.steps, b.steps);
    }

    #[test]
    fn kennedy_eberhart_velocity_is_inertia_plus_pulls() {
        // c1 = c2 = 0 => v <- clip(w v, -Vmax, Vmax), independent of r1, r2.
        let bounds = eindir_core::Bounds::new(array![-2.0], array![2.0], 0.0);
        let obj = crate::oracle::Oracle::unbounded(1, |x| (x[0] * x[0], array![2.0 * x[0]]));
        let mut swarm = [Particle {
            position: array![0.5],
            velocity: array![0.4],
            best_position: array![0.5],
            best_value: 0.25,
        }];
        let mut gbest_position = array![0.5];
        let mut gbest_value = 0.25;
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        update_swarm(
            &obj,
            &bounds,
            &mut swarm,
            &mut gbest_position,
            &mut gbest_value,
            0.5,
            0.0,
            0.0,
            &mut rng,
        );
        // v = 0.5 * 0.4 = 0.2; x = 0.5 + 0.2 = 0.7
        assert!((swarm[0].velocity[0] - 0.2).abs() < 1e-15);
        assert!((swarm[0].position[0] - 0.7).abs() < 1e-15);
    }
}
