use ndarray::{Array1, array, s};
use rgmin::vecops::{Vector, div_assign_floor, mul_assign, scale, vsum};

#[test]
fn elementwise_updates_match_scalar_oracles_for_slices_and_strides() {
    for n in [0, 3, 65_537] {
        let storage = Array1::from_iter((0..(2 * n)).map(|i| 0.125 * (i % 17) as f64));
        for x in [storage.slice(s![..n]), storage.slice(s![..;2])] {
            let start = Array1::from_elem(n + 2, 2.0);
            let mut y = start.clone();
            mul_assign(x, &mut y);
            for i in 0..n {
                assert_eq!(y[i], 2.0 * x[i]);
            }
            assert_eq!(y[n], 2.0);
            assert_eq!(y[n + 1], 2.0);
            div_assign_floor(x, &mut y, 0.25);
            for i in 0..n {
                assert_eq!(y[i], (2.0 * x[i]) / x[i].max(0.25));
            }
            scale(-0.5, &mut y);
            for i in 0..n {
                assert_eq!(y[i], -0.5 * ((2.0 * x[i]) / x[i].max(0.25)));
            }
            assert_eq!(y[n], -1.0);
        }
    }
    assert_eq!(vsum(&Vector::from_host(array![1.0, 2.0, -0.5])), 2.5);
}
