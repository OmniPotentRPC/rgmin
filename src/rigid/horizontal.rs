/// eOn `projectOutRotTrans`. Unit-mass Eckart, translation and rotation.
pub(crate) fn project_out_rot_trans(vec: &mut Array1<f64>, pos: ArrayView1<f64>) {
    project_horizontal(vec, pos, None, true);
}

/// Horizontal space of `R^{3N}/SE(3)` or `R^{3N}/T(3)`.
///
/// `rotate` false keeps translations and drops rotations. Masses, when
/// present, select the mass metric. A Cartesian force should pass `None`.
pub(crate) fn project_horizontal(
    vec: &mut Array1<f64>,
    pos: ArrayView1<f64>,
    masses: Option<&[f64]>,
    rotate: bool,
) {
    let Some(ps) = pos.as_slice() else {
        return;
    };
    let Some(vs) = vec.as_slice_mut() else {
        return;
    };
    thread_local! {
        static WS: RefCell<RigidWorkspace> = RefCell::new(RigidWorkspace::new());
    }
    WS.with(|ws| {
        let mut req = RigidRequest::cartesian(
            ps,
            if rotate {
                RigidMode::TranslationRotation
            } else {
                RigidMode::Translation
            },
        );
        req.masses = masses;
        req.mass_metric = masses.is_some();
        ws.borrow_mut().project(vs, &req);
    });
}
