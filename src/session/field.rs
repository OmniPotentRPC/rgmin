//! The field a session reports in.
//!
//! A session measures its stop and its report on [`Solver::horizontal_grad`]:
//! the manifold's Riemannian gradient, the rigid blocks of a product
//! projected with the session masses and periodic flag, and the declared
//! rigid motion removed at the point. A line search that reads another
//! field can descend along a direction the report never sees, or refuse a
//! direction the report calls a descent.
//! [`ProjectedField`] is that one field as a [`Manifold`], so Riemannian CG
//! forms its gradient, its Wolfe slopes, its coefficient and its transport
//! in the field the stop reads.

use std::cell::RefCell;

use ndarray::Array1;

use super::Solver;
use crate::manifold::Manifold;
use crate::rigid::RigidWorkspace;

/// The session manifold with the session's rigid-motion handling.
///
/// `retract`, `inner` and `norm` are the manifold's own. `project`,
/// `to_tangent`, `egrad2rgrad` and `transport` are the session's tangent
/// projection, its projection of a vector that is not a gradient, its
/// horizontal gradient and its vector transport.
pub(crate) struct ProjectedField<'a> {
    solver: &'a Solver,
    ws: RefCell<RigidWorkspace>,
}

impl<'a> ProjectedField<'a> {
    pub(crate) fn new(solver: &'a Solver) -> Self {
        Self {
            solver,
            ws: RefCell::new(RigidWorkspace::new()),
        }
    }
}

impl Manifold for ProjectedField<'_> {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        self.solver.manifold.required_dim(n)
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.solver
            .project_vec_into(&mut self.ws.borrow_mut(), x, v)
    }

    fn to_tangent(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.solver
            .tangent_vec_into(&mut self.ws.borrow_mut(), x, v)
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        self.solver
            .horizontal_grad_into(&mut self.ws.borrow_mut(), x, egrad)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.solver.manifold.retract(x, v)
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.solver
            .transport_vec_into(&mut self.ws.borrow_mut(), x_from, x_to, v)
    }

    fn inner(&self, x: &Array1<f64>, u: &Array1<f64>, v: &Array1<f64>) -> f64 {
        self.solver.manifold.inner(x, u, v)
    }

    fn norm(&self, x: &Array1<f64>, v: &Array1<f64>) -> f64 {
        self.solver.manifold.norm(x, v)
    }
}
