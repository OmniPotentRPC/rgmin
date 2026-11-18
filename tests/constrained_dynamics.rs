//! Invariant measure, equipartition, and constrained energy.
//!
//! The densities are the ones proved for constrained Langevin dynamics
//! by Lelièvre, Rousset and Stoltz (Math. Comp. 81, 2012,
//! doi:10.1090/S0025-5718-2012-02594-4) and sampled by the geodesic BAOAB
//! scheme of Leimkuhler and Matthews (Proc. R. Soc. A 472, 20160138, 2016,
//! doi:10.1098/rspa.2016.0138). The Kolmogorov threshold is the asymptotic
//! 5% value 1.36 / sqrt(n).

use ndarray::{Array1, array};

use rgmin::dynamics::{LangevinConfig, NveConfig, PhasePoint, baoab, rattle};
#[cfg(feature = "readcon")]
use rgmin::manifold::AffineHyperplane;
use rgmin::manifold::{Manifold, ManifoldKind, MassMetric, ProductFactor, ProductManifold, Sphere};

fn kolmogorov(samples: &[f64], cdf: impl Fn(f64) -> f64) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(|left, right| left.partial_cmp(right).expect("finite sample"));
    let n = ordered.len() as f64;
    let mut stat = 0.0_f64;
    for (index, value) in ordered.iter().enumerate() {
        let model = cdf(*value);
        let above = (index as f64 + 1.0) / n;
        let below = index as f64 / n;
        stat = stat.max((above - model).abs()).max((model - below).abs());
    }
    stat
}

fn circle_cdf(beta: f64, theta: f64, density_weight: impl Fn(f64) -> f64) -> f64 {
    let panels = 1024usize;
    let start = -std::f64::consts::PI;
    let width = (theta - start) / panels as f64;
    let full_width = 2.0 * std::f64::consts::PI / panels as f64;
    let unnormalized = |at: f64| (-beta * at.cos()).exp() * density_weight(at);
    let mut partition = 0.0;
    let mut partition_previous = unnormalized(start);
    for index in 1..=panels {
        let at = start + full_width * index as f64;
        let value = unnormalized(at);
        partition += 0.5 * (partition_previous + value) * full_width;
        partition_previous = value;
    }
    let density_at = |at: f64| unnormalized(at) / partition;
    let mut sum = 0.0;
    let mut previous = density_at(start);
    for index in 1..=panels {
        let at = start + width * index as f64;
        let density = density_at(at);
        sum += 0.5 * (previous + density) * width;
        previous = density;
    }
    sum
}

struct SampleWindow {
    burn: usize,
    stride: usize,
    count: usize,
}

fn sample_positions(
    manifold: &dyn Manifold,
    metric: &MassMetric,
    mut phase: PhasePoint,
    config: &LangevinConfig,
    window: SampleWindow,
    force: impl Fn(&[f64]) -> (Vec<f64>, f64),
) -> (Vec<Array1<f64>>, Vec<f64>) {
    let mut positions = Vec::with_capacity(window.count);
    let mut kinetic = Vec::with_capacity(window.count);
    let mut draws = 0u64;
    for sample in 0..window.burn + window.count {
        let report = baoab(
            manifold,
            metric,
            &mut phase,
            &config.with_seed(config.seed.wrapping_add(draws)),
            &[],
            window.stride,
            |x| Ok::<_, &'static str>(force(x)),
        )
        .expect("sample");
        draws = draws.wrapping_add(1);
        if sample >= window.burn {
            kinetic.push(report.kinetic);
            positions.push(phase.position.clone());
        }
    }
    (positions, kinetic)
}

#[test]
fn circle_samples_the_boltzmann_density() {
    let beta = 1.5;
    let metric = MassMetric::new(array![1.0, 3.0]).unwrap();
    let config = LangevinConfig::default()
        .with_thermal(1.0 / beta)
        .with_friction(4.0)
        .with_time_step(0.05)
        .with_seed(7);
    let (positions, kinetic) = sample_positions(
        &Sphere,
        &metric,
        PhasePoint::default()
            .with_position(array![1.0, 0.0])
            .with_velocity(array![0.0, 0.0]),
        &config,
        SampleWindow {
            burn: 400,
            stride: 40,
            count: 300,
        },
        |x| (vec![-1.0, 0.0], x[0]),
    );
    let angles: Vec<f64> = positions
        .iter()
        .map(|point| point[1].atan2(point[0]))
        .collect();
    let stat = kolmogorov(&angles, |theta| {
        circle_cdf(beta, theta, |angle| {
            (angle.cos().powi(2) + angle.sin().powi(2) / 3.0).sqrt()
        })
    });
    let threshold = 1.36 / (angles.len() as f64).sqrt();
    assert!(stat <= threshold, "Kolmogorov {stat} exceeds {threshold}");
    let mean = kinetic.iter().sum::<f64>() / kinetic.len() as f64;
    let expected = 0.5 / beta;
    let variance = kinetic
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / kinetic.len() as f64;
    let error = variance.sqrt() / (kinetic.len() as f64).sqrt();
    assert!((mean - expected).abs() <= 4.0 * error.max(expected * f64::EPSILON));
    let mut worst = 0.0_f64;
    for point in &positions {
        let norm = (point[0] * point[0] + point[1] * point[1]).sqrt();
        worst = worst.max((norm - 1.0).abs());
    }
    assert!(worst <= 64.0 * f64::EPSILON);
}

#[test]
fn sphere_coordinate_is_uniform_and_kinetic_energy_equipartitions() {
    let metric = MassMetric::uniform(3, 1.0).unwrap();
    let thermal = 0.8;
    let config = LangevinConfig::default()
        .with_thermal(thermal)
        .with_friction(4.0)
        .with_time_step(0.05)
        .with_seed(11);
    let (positions, kinetic) = sample_positions(
        &Sphere,
        &metric,
        PhasePoint::default()
            .with_position(array![0.0, 0.0, 1.0])
            .with_velocity(array![0.2, 0.0, 0.0]),
        &config,
        SampleWindow {
            burn: 400,
            stride: 40,
            count: 500,
        },
        |_| (vec![0.0, 0.0, 0.0], 0.0),
    );
    let heights: Vec<f64> = positions.iter().map(|point| point[2]).collect();
    let stat = kolmogorov(&heights, |z| 0.5 * (z + 1.0));
    let threshold = 1.36 / (heights.len() as f64).sqrt();
    assert!(stat <= threshold, "Kolmogorov {stat} exceeds {threshold}");
    let mean = kinetic.iter().sum::<f64>() / kinetic.len() as f64;
    let expected = thermal;
    let variance = kinetic
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / kinetic.len() as f64;
    let error = variance.sqrt() / (kinetic.len() as f64).sqrt();
    assert!((mean - expected).abs() <= 4.0 * error.max(expected * f64::EPSILON));
    let mut worst = 0.0_f64;
    for point in &positions {
        let norm = point.iter().map(|value| value * value).sum::<f64>().sqrt();
        worst = worst.max((norm - 1.0).abs());
    }
    assert!(worst <= 64.0 * f64::EPSILON);
    assert_eq!(
        baoab(
            &Sphere,
            &metric,
            &mut PhasePoint::default()
                .with_position(array![0.0, 0.0, 1.0])
                .with_velocity(array![0.0, 0.0, 0.0]),
            &config,
            &[],
            1,
            |_| Ok::<_, &'static str>((vec![0.0, 0.0, 0.0], 0.0)),
        )
        .unwrap()
        .degrees_of_freedom,
        2
    );
}

#[test]
fn product_of_circles_stays_on_each_factor_and_equipartitions() {
    let product = ProductManifold::new([
        ProductFactor::new(ManifoldKind::Sphere, 2),
        ProductFactor::new(ManifoldKind::Sphere, 2),
    ])
    .unwrap();
    let metric = MassMetric::uniform(4, 1.0).unwrap();
    let thermal = 0.5;
    let config = LangevinConfig::default()
        .with_thermal(thermal)
        .with_friction(4.0)
        .with_time_step(0.04)
        .with_seed(19);
    let (positions, kinetic) = sample_positions(
        &product,
        &metric,
        PhasePoint::default()
            .with_position(array![1.0, 0.0, 0.0, 1.0])
            .with_velocity(array![0.0, 0.1, 0.2, 0.0]),
        &config,
        SampleWindow {
            burn: 200,
            stride: 30,
            count: 400,
        },
        |_| (vec![0.0; 4], 0.0),
    );
    let mut worst = 0.0_f64;
    for point in &positions {
        for block in [[0, 1], [2, 3]] {
            let norm = (point[block[0]].powi(2) + point[block[1]].powi(2)).sqrt();
            worst = worst.max((norm - 1.0).abs());
        }
    }
    assert!(worst <= 64.0 * f64::EPSILON);
    let mean = kinetic.iter().sum::<f64>() / kinetic.len() as f64;
    let expected = thermal;
    let variance = kinetic
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / kinetic.len() as f64;
    let error = variance.sqrt() / (kinetic.len() as f64).sqrt();
    assert!((mean - expected).abs() <= 4.0 * error.max(expected * f64::EPSILON));
}

#[test]
fn rattle_energy_excursion_scales_as_dt_squared() {
    let metric = MassMetric::uniform(3, 1.0).unwrap();
    let band = |time_step: f64| {
        let steps = (1.0 / time_step).round() as usize;
        let mut phase = PhasePoint::default()
            .with_position(array![1.0, 0.0, 0.0])
            .with_velocity(array![0.0, 0.4, 0.0]);
        let config = NveConfig::default().with_time_step(time_step);
        let mut worst = 0.0_f64;
        let mut energy = 0.0;
        for index in 0..=steps {
            let report = rattle(&Sphere, &metric, &mut phase, &config, 1, |x| {
                Ok::<_, &'static str>((vec![0.0, 0.0, -1.0], x[2]))
            })
            .expect("rattle");
            if index == 0 {
                energy = report.potential + report.kinetic;
            } else {
                worst = worst.max((report.potential + report.kinetic - energy).abs());
            }
        }
        worst
    };
    let coarse = band(1.0 / 32.0);
    let fine = band(1.0 / 64.0);
    assert!(fine <= coarse * 0.5, "coarse {coarse} fine {fine}");
    assert!(coarse <= (1.0_f64 / 32.0).powi(2) * 4.0);
}

#[cfg(feature = "readcon")]
#[test]
fn argon_rattle_on_a_com_plane_conserves_energy() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/argon.con");
    let (position, _) = rgmin::read_checkpoint(path).expect("argon con");
    let count = position.len() / 3;
    let mut normal = Array1::zeros(position.len());
    let inv = 1.0 / (count as f64).sqrt();
    for atom in 0..count {
        normal[3 * atom] = inv;
    }
    let plane = AffineHyperplane::new(position.clone(), normal).unwrap();
    let metric = MassMetric::per_atom(&vec![39.948; count]).unwrap();
    let cell = minimage::Cell::ortho(20.0, 20.0, 20.0).unwrap();
    let sigma = 3.4;
    let epsilon = 0.0104;
    let scale = {
        let joule = 1.602176634e-19;
        let avogadro = 6.02214076e23;
        let kj_per_ev = joule * avogadro / 1000.0;
        kj_per_ev * 100.0
    };
    let force = |x: &[f64]| {
        let mut f = vec![0.0; x.len()];
        let mut energy = 0.0;
        for i in 0..count {
            for j in (i + 1)..count {
                let pi = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
                let pj = [x[3 * j], x[3 * j + 1], x[3 * j + 2]];
                let dr = cell.displacement(pj, pi);
                let r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                let inv_r2 = 1.0 / r2;
                let sr2 = sigma * sigma * inv_r2;
                let sr6 = sr2 * sr2 * sr2;
                let sr12 = sr6 * sr6;
                energy += 4.0 * epsilon * (sr12 - sr6);
                let magnitude = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r2;
                for axis in 0..3 {
                    let component = magnitude * dr[axis];
                    f[3 * i + axis] += component;
                    f[3 * j + axis] -= component;
                }
            }
        }
        (f, energy)
    };
    let mut velocity = Array1::zeros(position.len());
    velocity[1] = 0.02;
    velocity[4] = -0.01;
    velocity[7] = -0.01;
    let phase = PhasePoint::default()
        .with_position(position)
        .with_velocity(velocity);
    let band = |time_step: f64, phase: &PhasePoint| {
        let steps = (0.5 / time_step).round() as usize;
        let mut state = phase.clone();
        let config = NveConfig::default()
            .with_time_step(time_step)
            .with_kinetic_scale(scale);
        let mut worst_energy = 0.0_f64;
        let mut worst_gap = 0.0_f64;
        let mut energy = 0.0;
        for index in 0..=steps {
            let report = rattle(&plane, &metric, &mut state, &config, 1, |x| {
                Ok::<_, &'static str>(force(x))
            })
            .expect("argon rattle");
            let gap = plane.signed_distance(&state.position).unwrap().abs();
            worst_gap = worst_gap.max(gap);
            if index == 0 {
                energy = report.potential + report.kinetic;
            } else {
                worst_energy = worst_energy.max((report.potential + report.kinetic - energy).abs());
            }
        }
        (worst_energy, worst_gap)
    };
    let (coarse, gap) = band(1.0 / 64.0, &phase);
    let (fine, _) = band(1.0 / 128.0, &phase.clone());
    assert!(fine <= coarse * 0.5, "coarse {coarse} fine {fine}");
    assert!(gap <= 20.0 * f64::EPSILON * 64.0, "constraint gap {gap}");
    let _ = phase;
}

fn constraint_gap(point: &Array1<f64>) -> f64 {
    (point.iter().map(|value| value * value).sum::<f64>().sqrt() - 1.0).abs()
}

fn radial_speed(point: &Array1<f64>, velocity: &Array1<f64>) -> f64 {
    point
        .iter()
        .zip(velocity.iter())
        .map(|(q, u)| q * u)
        .sum::<f64>()
        .abs()
}

/// Gram-solve tolerance from `mass::solve_linear`: one normal, unit scale.
fn solver_tolerance(magnitude: f64) -> f64 {
    f64::EPSILON * magnitude.max(1.0) * 64.0
}

#[test]
fn constraint_drift_stays_at_solver_tolerance_over_a_long_run() {
    let steps = 20_000usize;
    let metric = MassMetric::uniform(3, 1.0).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![0.0, 0.0, 1.0])
        .with_velocity(array![0.35, -0.15, 0.0]);
    let config = LangevinConfig::default()
        .with_thermal(0.4)
        .with_friction(0.8)
        .with_time_step(0.02)
        .with_seed(23);
    let mut worst_gap = 0.0_f64;
    let mut worst_radial = 0.0_f64;
    for block in 0..steps / 100 {
        baoab(
            &Sphere,
            &metric,
            &mut phase,
            &config.with_seed(config.seed.wrapping_add(block as u64)),
            &[],
            100,
            |x| {
                Ok::<_, &'static str>((
                    vec![-x[0], 0.2, -0.5 * x[2]],
                    0.5 * x[0] * x[0] + 0.25 * x[2] * x[2],
                ))
            },
        )
        .expect("long sphere run");
        worst_gap = worst_gap.max(constraint_gap(&phase.position));
        worst_radial = worst_radial.max(radial_speed(&phase.position, &phase.velocity));
    }
    let tolerance = solver_tolerance(1.0);
    assert!(
        worst_gap <= tolerance,
        "position gap {worst_gap} exceeds solver tolerance {tolerance}"
    );
    assert!(
        worst_radial <= tolerance,
        "radial velocity {worst_radial} exceeds solver tolerance {tolerance}"
    );

    let plane = rgmin::manifold::AffineHyperplane::new(
        array![0.0, 0.0, 0.0, 0.0],
        array![1.0, -1.0, 0.5, 0.0],
    )
    .unwrap();
    let metric = MassMetric::new(array![2.0, 0.5, 3.0, 1.0]).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![0.2, 0.2, 0.0, 0.1])
        .with_velocity(array![0.1, -0.05, 0.2, -0.1]);
    let config = NveConfig::default().with_time_step(0.01);
    let mut worst = 0.0_f64;
    for _ in 0..200 {
        rattle(&plane, &metric, &mut phase, &config, 50, |x| {
            Ok::<_, &'static str>((vec![-x[0], -0.3 * x[1], -x[2], 0.0], 0.0))
        })
        .expect("long plane run");
        let gap = plane.signed_distance(&phase.position).unwrap().abs();
        worst = worst.max(gap);
    }
    let magnitude = phase
        .position
        .iter()
        .fold(0.0_f64, |acc, value| acc + value.abs());
    let tolerance = solver_tolerance(magnitude);
    assert!(
        worst <= tolerance,
        "hyperplane gap {worst} exceeds solver tolerance {tolerance}"
    );
}

fn free_kinetic_band(
    metric: &MassMetric,
    position: Array1<f64>,
    velocity: Array1<f64>,
    time_step: f64,
) -> f64 {
    let horizon = 1.0;
    let steps = (horizon / time_step).round() as usize;
    let mut phase = PhasePoint::default()
        .with_position(position)
        .with_velocity(velocity);
    let config = LangevinConfig::default()
        .with_friction(0.0)
        .with_thermal(1.0)
        .with_time_step(time_step)
        .with_seed(1);
    let mut energy = 0.0;
    let mut worst = 0.0_f64;
    let mut worst_gap = 0.0_f64;
    let width = phase.position.len();
    for index in 0..=steps {
        let report = baoab(&Sphere, metric, &mut phase, &config, &[], 1, |_| {
            Ok::<_, &'static str>((vec![0.0; width], 0.0))
        })
        .expect("free geodesic");
        worst_gap = worst_gap.max(constraint_gap(&phase.position));
        if index == 0 {
            energy = report.kinetic;
        } else {
            worst = worst.max((report.kinetic - energy).abs());
        }
    }
    assert!(
        worst_gap <= solver_tolerance(1.0),
        "constraint gap {worst_gap}"
    );
    worst
}

#[test]
fn unequal_mass_rattle_matches_the_oblique_step_and_conserves_energy() {
    let metric = MassMetric::new(array![1.0, 4.0]).unwrap();
    let position = array![1.0, 0.0];
    let velocity = array![0.0, 0.2];
    let step = 0.1;
    let mut phase = PhasePoint::default()
        .with_position(position.clone())
        .with_velocity(velocity.clone());
    let config = NveConfig::default().with_time_step(step);
    rattle(&Sphere, &metric, &mut phase, &config, 1, |_| {
        Ok::<_, &'static str>((vec![0.0, 0.0], 0.0))
    })
    .expect("one rattle step");
    // The straight step is (1, 0.02). The sphere gradient is (2, 0), so the
    // oblique correction changes only the first coordinate, and the result
    // sits on the circle.
    let straight_y = step * velocity[1];
    assert!((phase.position[1] - straight_y).abs() <= solver_tolerance(1.0));
    assert!(constraint_gap(&phase.position) <= solver_tolerance(1.0));
    let radial = radial_speed(&phase.position, &phase.velocity);
    assert!(radial <= solver_tolerance(1.0), "radial {radial}");

    let coarse = free_kinetic_band(&metric, array![1.0, 0.0], array![0.0, 0.25], 1.0 / 64.0);
    let fine = free_kinetic_band(&metric, array![1.0, 0.0], array![0.0, 0.25], 1.0 / 128.0);
    assert!(
        fine <= coarse * 0.5,
        "kinetic band is not second order: coarse {coarse} fine {fine}"
    );

    let product = ProductManifold::new([
        ProductFactor::new(ManifoldKind::Sphere, 2),
        ProductFactor::new(ManifoldKind::Sphere, 2),
    ])
    .unwrap();
    let metric = MassMetric::new(array![1.0, 3.0, 2.0, 5.0]).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![1.0, 0.0, 0.0, 1.0])
        .with_velocity(array![0.0, 0.2, 0.15, 0.0]);
    let config = LangevinConfig::default()
        .with_friction(0.0)
        .with_time_step(0.01)
        .with_seed(4);
    let mut worst = 0.0_f64;
    for _ in 0..400 {
        baoab(&product, &metric, &mut phase, &config, &[], 1, |_| {
            Ok::<_, &'static str>((vec![0.0; 4], 0.0))
        })
        .expect("product rattle");
        for block in [[0, 1], [2, 3]] {
            let norm = (phase.position[block[0]].powi(2) + phase.position[block[1]].powi(2)).sqrt();
            worst = worst.max((norm - 1.0).abs());
        }
    }
    assert!(worst <= solver_tolerance(1.0), "product gap {worst}");
}

#[test]
fn rattle_projects_the_final_velocity_to_the_tangent_space() {
    let metric = MassMetric::new(array![1.0, 4.0]).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![1.0, 0.0])
        .with_velocity(array![0.0, 0.2]);
    let config = NveConfig::default().with_time_step(0.1);
    rattle(&Sphere, &metric, &mut phase, &config, 1, |_| {
        Ok::<_, &'static str>((vec![1.0, 0.0], 0.0))
    })
    .expect("rattle");
    let radial = radial_speed(&phase.position, &phase.velocity);
    let tolerance = solver_tolerance(1.0);
    assert!(
        radial <= tolerance,
        "final RATTLE velocity has radial component {radial}, tolerance {tolerance}"
    );
}

#[test]
fn kinetic_scale_is_applied_without_readcon() {
    let metric = MassMetric::new(array![2.0]).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![0.0])
        .with_velocity(array![3.0]);
    let config = NveConfig::default()
        .with_time_step(0.1)
        .with_kinetic_scale(3.0);
    let report = rattle(
        &rgmin::manifold::Euclidean,
        &metric,
        &mut phase,
        &config,
        1,
        |_| Ok::<_, &'static str>((vec![0.0], 0.0)),
    )
    .expect("rattle");
    let expected = 3.0;
    assert!((report.kinetic - expected).abs() <= f64::EPSILON * expected);
}

#[test]
fn ou_stream_does_not_depend_on_which_masses_vanish() {
    let config = LangevinConfig::default()
        .with_thermal(0.3)
        .with_friction(1.5)
        .with_time_step(0.2)
        .with_seed(42)
        .with_kinetic_scale(2.0);
    let run = |masses: [f64; 3]| {
        let metric = MassMetric::new(array![masses[0], masses[1], masses[2]]).unwrap();
        let mut phase = PhasePoint::default()
            .with_position(array![0.0, 0.0, 0.0])
            .with_velocity(array![0.0, 0.0, 0.0]);
        baoab(
            &rgmin::manifold::Euclidean,
            &metric,
            &mut phase,
            &config,
            &[],
            1,
            |_| Ok::<_, &'static str>((vec![0.0, 0.0, 0.0], 0.0)),
        )
        .expect("ou");
        phase.velocity
    };
    let frozen = run([0.0, 1.0, 4.0]);
    let full = run([2.0, 1.0, 4.0]);
    assert_eq!(frozen[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(frozen[1].to_bits(), full[1].to_bits());
    assert_eq!(frozen[2].to_bits(), full[2].to_bits());
    assert_ne!(full[0].to_bits(), frozen[0].to_bits());
}

#[test]
fn ou_noise_samples_the_requested_thermal_kinetic_energy() {
    const SAMPLE_COUNT: usize = 1_000;
    const STANDARD_ERRORS: f64 = 5.0;
    let thermal = 0.7;
    let metric = MassMetric::new(array![2.0]).unwrap();
    let config = LangevinConfig::default()
        .with_thermal(thermal)
        .with_friction(100.0)
        .with_time_step(0.1)
        .with_kinetic_scale(2.0);
    let mut kinetic = Vec::with_capacity(SAMPLE_COUNT);
    for seed in 0..SAMPLE_COUNT as u64 {
        let mut phase = PhasePoint::default()
            .with_position(array![0.0])
            .with_velocity(array![0.0]);
        let report = baoab(
            &rgmin::manifold::Euclidean,
            &metric,
            &mut phase,
            &config.with_seed(seed),
            &[],
            1,
            |_| Ok::<_, &'static str>((vec![0.0], 0.0)),
        )
        .expect("OU sample");
        kinetic.push(report.kinetic);
    }
    let mean = kinetic.iter().sum::<f64>() / kinetic.len() as f64;
    let variance = kinetic
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / kinetic.len() as f64;
    let standard_error = (variance / kinetic.len() as f64).sqrt();
    let expected = 0.5 * thermal;
    assert!(
        (mean - expected).abs() <= STANDARD_ERRORS * standard_error,
        "OU kinetic mean {mean}, expected {expected}, standard error {standard_error}"
    );
}

#[test]
fn curved_dynamics_rejects_an_off_manifold_start() {
    let metric = MassMetric::uniform(2, 1.0).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![2.0, 0.0])
        .with_velocity(array![0.0, 0.0]);
    let error = rattle(
        &Sphere,
        &metric,
        &mut phase,
        &NveConfig::default(),
        1,
        |_| Ok::<_, &'static str>((vec![0.0, 0.0], 0.0)),
    )
    .expect_err("off-sphere start");
    assert!(matches!(
        error,
        rgmin::dynamics::DynamicsError::Geometry(rgmin::Error::Dynamics { .. })
    ));
}

#[test]
fn dynamics_rejects_a_nonzero_velocity_for_a_zero_mass_coordinate() {
    let metric = MassMetric::new(array![0.0, 1.0]).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![0.0, 0.0])
        .with_velocity(array![1.0, 0.0]);
    let error = baoab(
        &rgmin::manifold::Euclidean,
        &metric,
        &mut phase,
        &LangevinConfig::default(),
        &[],
        1,
        |_| Ok::<_, &'static str>((vec![0.0, 0.0], 0.0)),
    )
    .expect_err("moving zero-mass coordinate");
    assert!(matches!(
        error,
        rgmin::dynamics::DynamicsError::Geometry(rgmin::Error::Dynamics { .. })
    ));
}

#[test]
fn curved_wall_stays_on_the_sphere_and_the_allowed_side() {
    let metric = MassMetric::uniform(3, 1.0).unwrap();
    let wall = rgmin::PlanarWall::new(array![0.0, 0.0, 0.2], array![0.0, 0.0, 1.0]).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![1.0, 0.0, 0.0])
        .with_velocity(array![0.0, 0.2, 0.45]);
    let config = LangevinConfig::default()
        .with_thermal(0.05)
        .with_friction(0.0)
        .with_time_step(0.05)
        .with_seed(8);
    let mut hits = 0u64;
    let mut worst = 0.0_f64;
    let mut worst_outside = 0.0_f64;
    for _ in 0..80 {
        let report = baoab(
            &Sphere,
            &metric,
            &mut phase,
            &config,
            std::slice::from_ref(&wall),
            1,
            |_| Ok::<_, &'static str>((vec![0.0, 0.0, 0.0], 0.0)),
        )
        .expect("wall");
        hits += report.wall_hits;
        worst = worst.max(constraint_gap(&phase.position));
        let height = phase.position[2] - 0.2;
        if height > 0.0 {
            worst_outside = worst_outside.max(height);
        }
    }
    assert!(hits > 0, "the particle never met the wall");
    assert!(worst <= solver_tolerance(1.0), "left the sphere by {worst}");
    assert!(
        worst_outside <= solver_tolerance(1.0) * 64.0,
        "crossed the wall by {worst_outside}"
    );
}

#[test]
fn flat_wall_reflects_position_and_normal_velocity() {
    let metric = MassMetric::uniform(1, 1.0).unwrap();
    let wall = rgmin::PlanarWall::new(array![0.0], array![1.0]).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![-0.1])
        .with_velocity(array![1.0]);
    let config = LangevinConfig::default()
        .with_friction(0.0)
        .with_time_step(0.4);
    baoab(
        &rgmin::manifold::Euclidean,
        &metric,
        &mut phase,
        &config,
        std::slice::from_ref(&wall),
        1,
        |_| Ok::<_, &'static str>((vec![0.0], 0.0)),
    )
    .expect("flat wall");
    let tolerance = solver_tolerance(1.0);
    assert!(
        (phase.position[0] + 0.3).abs() <= tolerance,
        "flat wall position is {}, expected -0.3",
        phase.position[0]
    );
    assert!(
        (phase.velocity[0] + 1.0).abs() <= tolerance,
        "flat wall normal velocity is {}, expected -1",
        phase.velocity[0]
    );
}

#[test]
fn curved_wall_reflects_position_and_normal_velocity() {
    const STEP_ANGLE: f64 = 0.2;
    const EXPECTED_X: f64 = 0.979_933_380_147_167_4;
    const EXPECTED_Z: f64 = -0.199_325_288_092_974_45;
    let metric = MassMetric::uniform(3, 1.0).unwrap();
    let wall = rgmin::PlanarWall::new(array![0.0, 0.0, 0.0], array![0.0, 0.0, 1.0]).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![1.0, 0.0, 0.0])
        .with_velocity(array![0.0, 0.0, 1.0]);
    let config = LangevinConfig::default()
        .with_friction(0.0)
        .with_time_step(STEP_ANGLE);
    baoab(
        &Sphere,
        &metric,
        &mut phase,
        &config,
        std::slice::from_ref(&wall),
        1,
        |_| Ok::<_, &'static str>((vec![0.0; 3], 0.0)),
    )
    .expect("curved wall");
    let tolerance = solver_tolerance(1.0);
    assert!(
        (phase.position[0] - EXPECTED_X).abs() <= tolerance
            && phase.position[1].abs() <= tolerance
            && (phase.position[2] - EXPECTED_Z).abs() <= tolerance,
        "curved wall position is {:?}",
        phase.position
    );
    assert!(
        (phase.velocity[2] + EXPECTED_X).abs() <= tolerance,
        "curved wall normal velocity is {}, expected {}",
        phase.velocity[2],
        -EXPECTED_X
    );
}

fn hamiltonian(report_potential: f64, report_kinetic: f64) -> f64 {
    report_potential + report_kinetic
}

fn zero_friction_band(time_step: f64, horizon: f64) -> (f64, f64) {
    let steps = (horizon / time_step).round() as usize;
    let metric = MassMetric::uniform(3, 2.0).unwrap();
    let mut phase = PhasePoint::default()
        .with_position(array![0.0, 1.0, 0.0])
        .with_velocity(array![0.25, 0.0, -0.1]);
    let config = LangevinConfig::default()
        .with_thermal(1.0)
        .with_friction(0.0)
        .with_time_step(time_step)
        .with_seed(5)
        .with_kinetic_scale(1.0);
    let force = |x: &[f64]| (vec![0.0, 0.0, -1.0], x[2]);
    let mut energy = 0.0;
    let mut worst = 0.0_f64;
    let mut late = 0.0_f64;
    for index in 0..=steps {
        let report = baoab(&Sphere, &metric, &mut phase, &config, &[], 1, |x| {
            Ok::<_, &'static str>(force(x))
        })
        .expect("zero friction");
        let total = hamiltonian(report.potential, report.kinetic);
        if index == 0 {
            energy = total;
        } else {
            let gap = (total - energy).abs();
            worst = worst.max(gap);
            if index * 2 >= steps {
                late = late.max(gap);
            }
        }
        assert!(constraint_gap(&phase.position) <= solver_tolerance(1.0));
    }
    (worst, late)
}

#[test]
fn zero_friction_is_reversible_and_the_energy_stays_second_order() {
    let metric = MassMetric::uniform(2, 1.0).unwrap();
    let start = PhasePoint::default()
        .with_position(array![1.0, 0.0])
        .with_velocity(array![0.0, 0.4]);
    let config = LangevinConfig::default()
        .with_thermal(0.7)
        .with_friction(0.0)
        .with_time_step(0.05)
        .with_seed(9);
    let force = |x: &[f64]| (vec![-1.0, 0.0], x[0]);
    let mut forward = start.clone();
    let steps = 80usize;
    baoab(&Sphere, &metric, &mut forward, &config, &[], steps, |x| {
        Ok::<_, &'static str>(force(x))
    })
    .expect("forward");
    forward.velocity *= -1.0;
    baoab(&Sphere, &metric, &mut forward, &config, &[], steps, |x| {
        Ok::<_, &'static str>(force(x))
    })
    .expect("reverse");
    let position_gap = start
        .position
        .iter()
        .zip(forward.position.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let velocity_gap = start
        .velocity
        .iter()
        .zip(forward.velocity.iter())
        .map(|(a, b)| (a + b) * (a + b))
        .sum::<f64>()
        .sqrt();
    let roundoff = solver_tolerance(steps as f64);
    assert!(
        position_gap <= roundoff,
        "position did not reverse: {position_gap}"
    );
    assert!(
        velocity_gap <= roundoff,
        "velocity did not reverse: {velocity_gap}"
    );

    let mut rattled = start.clone();
    let nve = NveConfig::default().with_time_step(config.time_step);
    rattle(&Sphere, &metric, &mut rattled, &nve, steps, |x| {
        Ok::<_, &'static str>(force(x))
    })
    .expect("rattle");
    rattled.velocity *= -1.0;
    rattle(&Sphere, &metric, &mut rattled, &nve, steps, |x| {
        Ok::<_, &'static str>(force(x))
    })
    .expect("rattle reverse");
    let rattle_gap = start
        .position
        .iter()
        .zip(rattled.position.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        rattle_gap <= roundoff,
        "RATTLE position did not reverse: {rattle_gap}"
    );

    let (coarse, coarse_late) = zero_friction_band(1.0 / 32.0, 2.0);
    let (fine, fine_late) = zero_friction_band(1.0 / 64.0, 2.0);
    assert!(
        fine <= coarse * 0.5,
        "energy band is not second order: coarse {coarse} fine {fine}"
    );
    assert!(
        coarse_late <= coarse * 2.0,
        "energy drifted inside the coarse run: band {coarse} late {coarse_late}"
    );
    assert!(
        fine_late <= fine * 2.0,
        "energy drifted inside the fine run: band {fine} late {fine_late}"
    );
    let (longer, _) = zero_friction_band(1.0 / 64.0, 8.0);
    assert!(
        longer <= fine * 2.0,
        "energy grew with the horizon: short {fine} long {longer}"
    );
}
