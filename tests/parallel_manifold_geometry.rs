#![cfg(feature = "par")]

use ndarray::Array1;
use rgmin::manifold::{Euclidean, Manifold, MwRigid, RigidQuotient, Sphere};

#[test]
fn long_translations_preserve_each_coordinate() {
    let n=65_538;
    let x=Array1::from_iter((0..n).map(|i|(i%8) as f64/8.0));
    let v=Array1::from_iter((0..n).map(|i|if i%2==0 {0.5} else {-0.5}));
    for geometry in [&Euclidean as &dyn Manifold,&MwRigid,&RigidQuotient] {
        let y=geometry.retract(&x,&v);
        assert_eq!(y.len(),n);
        for i in 0..n { assert_eq!(y[i],(i%8) as f64/8.0+if i%2==0 {0.5} else {-0.5}); }
    }
}

#[test]
fn long_sphere_projection_and_retraction_match_the_exact_geometry() {
    let n=65_536;
    let x=Array1::from_elem(n,1.0/256.0);
    let v=Array1::from_iter((0..n).map(|i|if i%2==0 {1.0/256.0} else {-1.0/256.0}));
    let projected=Sphere.project(&x,&(&x+&v));
    assert_eq!(projected,v);
    let y=Sphere.retract(&x,&v);
    let positive=2.0/256.0/2.0_f64.sqrt();
    for i in 0..n {
        let expected=if i%2==0 {positive} else {0.0};
        assert!((y[i]-expected).abs()<1e-15);
    }
    assert!((y.dot(&y)-1.0).abs()<1e-12);
}
