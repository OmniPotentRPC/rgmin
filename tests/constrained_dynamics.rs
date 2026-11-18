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

fn i0(x: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = 1.0;
    let half = 0.5 * x;
    let mut k = 1.0;
    loop {
        term *= half * half / (k * k);
        let next = sum + term;
        if next == sum {
            return sum;
        }
        sum = next;
        k += 1.0;
    }
}

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

fn circle_cdf(beta: f64, theta: f64) -> f64 {
    let panels = 1024usize;
    let start = -std::f64::consts::PI;
    let width = (theta - start) / panels as f64;
    let partition = 2.0 * std::f64::consts::PI * i0(beta);
    let density_at = |at: f64| (-beta * at.cos()).exp() / partition;
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

fn sample_positions(
    manifold: &dyn Manifold,
    metric: &MassMetric,
    mut phase: PhasePoint,
    config: &LangevinConfig,
    burn: usize,
    stride: usize,
    count: usize,
    force: impl Fn(&[f64]) -> (Vec<f64>, f64),
) -> (Vec<Array1<f64>>, Vec<f64>) {
    let mut positions = Vec::with_capacity(count);
    let mut kinetic = Vec::with_capacity(count);
    let mut draws = 0u64;
    for sample in 0..burn + count {
        let report = baoab(
            manifold,
            metric,
            &mut phase,
            &config.with_seed(config.seed.wrapping_add(draws)),
            &[],
            stride,
            |x| Ok::<_, &'static str>(force(x)),
        )
        .expect("sample");
        draws = draws.wrapping_add(1);
        if sample >= burn {
            kinetic.push(report.kinetic);
            positions.push(phase.position.clone());
        }
    }
    (positions, kinetic)
}

#[test]
fn circle_samples_the_boltzmann_density() {
    let beta = 1.5;
    let metric = MassMetric::uniform(2, 1.0).unwrap();
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
        400,
        40,
        500,
        |x| (vec![-1.0, 0.0], x[0]),
    );
    let angles: Vec<f64> = positions
        .iter()
        .map(|point| point[1].atan2(point[0]))
        .collect();
    let stat = kolmogorov(&angles, |theta| circle_cdf(beta, theta));
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
        400,
        40,
        500,
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
        200,
        30,
        400,
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
