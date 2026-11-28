

# Embed in anneal

Hopping chains in [anneal](https://github.com/HaoZeke/anneal) relax thousands of times from perturbations of
an already-relaxed structure. They do not own a second L-BFGS: they hold
an `rgmin::Lbfgs` through `anneal_core::methods::warm_lbfgs::WarmLbfgs`.


## Checkout layout

    Rust/
      eindir/
      rgmin/
      anneal/
      landfold/

`anneal` and `landfold` path-depend on the two siblings so there is one
`eindir-core` and one `rgmin` in the graph.


## Call site

The public hopping API is unchanged:

```rust
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::Array1;

let mut opt = WarmLbfgs::default();
let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
let (f, x, evals) = opt.minimize(x0.view(), 200, |v| Some(fg(v)));
```

`WarmLbfgs::forget` drops the pair history when the chain jumps to a
structurally different basin. `minimize_watched` still sits at the top of
each accepted iterate so a screening predictor can stop the relaxation.


## Why this crate, not a copy

A second two-loop in anneal drifted from the Nocedal-Wright 7.20 scale
once already. The production method and the pair history belong in
one place.
