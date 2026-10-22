    #[test]
    fn drops_a_pure_translation() {
        let pos = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut v = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        project_out_rot_trans(&mut v, pos.view());
        assert!(l2(&v) < 1e-12);
    }

    #[test]
    fn unequal_masses_still_kill_translation() {
        let pos = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut v = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let masses = [16.0, 1.0, 1.0];
        project_horizontal(&mut v, pos.view(), Some(&masses), true);
        assert!(l2(&v) < 1e-12);
    }

    #[test]
    fn unequal_masses_kill_mass_weighted_rotation() {
        let pos = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
        let mut v = array![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let masses = [12.0, 1.0, 1.0];
        project_horizontal(&mut v, pos.view(), Some(&masses), true);
        assert!(l2(&v) < 1e-10, "{v:?}");
    }

    #[test]
    fn periodic_keeps_rotation_drops_translation() {
        let pos = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0];
        let rot = array![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0];
        let mut v = rot.clone();
        project_horizontal(&mut v, pos.view(), None, false);
        assert!(l2(&v) > 0.5, "rotation was removed under PBC {v:?}");
        let mut t = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        project_horizontal(&mut t, pos.view(), None, false);
        assert!(l2(&t) < 1e-12, "translation survived under PBC {t:?}");
    }

    #[test]
    fn single_atom_drops_translation_only() {
        let pos = [0.2, -0.4, 0.7];
        let mut v = [0.3, -0.1, 0.5];
        let mut ws = RigidWorkspace::new();
        ws.project(&mut v, &RigidRequest::cartesian(&pos, RigidMode::Auto));
        assert!(nrm(&v) < 1e-12, "{v:?}");
        assert_eq!(ws.rank(), 3);
    }

    #[test]
    fn linear_molecule_has_rank_five() {
        let pos = [0.0, 0.0, -1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1];
        let mut ws = RigidWorkspace::new();
        let mut v = [0.0; 9];
        ws.project(&mut v, &RigidRequest::cartesian(&pos, RigidMode::Auto));
        assert_eq!(ws.rank(), 5, "linear rank {}", ws.rank());
        let mut bend = [0.0, 0.2, 0.0, 0.0, -0.1, 0.0, 0.0, 0.2, 0.0];
        ws.project(&mut bend, &RigidRequest::cartesian(&pos, RigidMode::Auto));
        assert!(nrm(&bend) > 1e-3, "bend vanished {bend:?}");
    }

    #[test]
    fn projector_is_idempotent_and_symmetric() {
        let pos = [0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.1, 1.3, 0.4, -0.4, 0.2, 1.1];
        let mut ws = RigidWorkspace::new();
        let req = RigidRequest::cartesian(&pos, RigidMode::TranslationRotation);
        let v0 = [
            0.2, -0.3, 0.1, 0.4, 0.1, -0.2, -0.3, 0.5, 0.2, 0.1, -0.4, 0.3,
        ];
        let w0 = [
            0.1, 0.2, -0.4, -0.2, 0.3, 0.1, 0.5, -0.1, 0.2, -0.3, 0.2, 0.4,
        ];
        let mut v = v0;
        ws.project(&mut v, &req);
        let once = v;
        ws.project(&mut v, &req);
        for i in 0..v.len() {
            assert!((v[i] - once[i]).abs() < 1e-10, "not idempotent at {i}");
        }
        let mut pw = w0;
        ws.project(&mut pw, &req);
        let left: f64 = once.iter().zip(w0).map(|(a, b)| a * b).sum();
        let right: f64 = v0.iter().zip(pw).map(|(a, b)| a * b).sum();
        assert!((left - right).abs() < 1e-9, "not symmetric {left} {right}");
    }

    #[test]
    fn rigid_motion_of_the_input_does_not_change_the_output() {
        let pos = [0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.2, 1.1, 0.4];
        let mut ws = RigidWorkspace::new();
        let req = RigidRequest::cartesian(&pos, RigidMode::Auto);
        let mut v = [0.1, -0.2, 0.3, -0.4, 0.2, 0.1, 0.2, 0.0, -0.3];
        ws.project(&mut v, &req);
        let mut moved = [0.1, -0.2, 0.3, -0.4, 0.2, 0.1, 0.2, 0.0, -0.3];
        for i in 0..3 {
            moved[3 * i] += 0.4;
            moved[3 * i + 1] += -0.2;
        }
        let com = [0.4, 0.4666666667, 0.1333333333];
        for i in 0..3 {
            let rx = pos[3 * i] - com[0];
            let ry = pos[3 * i + 1] - com[1];
            moved[3 * i] += -ry * 0.15;
            moved[3 * i + 1] += rx * 0.15;
        }
        ws.project(&mut moved, &req);
        for i in 0..v.len() {
            assert!((v[i] - moved[i]).abs() < 1e-8, "{i} {} {}", v[i], moved[i]);
        }
    }

    #[test]
    fn moving_the_geometry_moves_the_projected_vector() {
        let pos = [0.0, 0.0, 0.0, 1.2, 0.1, 0.2, 0.2, 1.1, -0.3, -0.4, 0.5, 0.8];
        let v = [
            0.2, -0.1, 0.3, -0.4, 0.2, 0.1, 0.1, 0.3, -0.2, 0.0, -0.2, 0.4,
        ];
        let mut ws = RigidWorkspace::new();
        let mut pv = v;
        ws.project(&mut pv, &RigidRequest::cartesian(&pos, RigidMode::Auto));
        let ang = 0.4_f64;
        let (c, s) = (ang.cos(), ang.sin());
        let mut com = [0.0; 3];
        for i in 0..4 {
            for d in 0..3 {
                com[d] += pos[3 * i + d];
            }
        }
        for d in 0..3 {
            com[d] /= 4.0;
        }
        let mut pos2 = [0.0; 12];
        let mut v2 = [0.0; 12];
        let mut rpv = [0.0; 12];
        for i in 0..4 {
            let x = pos[3 * i] - com[0];
            let y = pos[3 * i + 1] - com[1];
            pos2[3 * i] = c * x - s * y + com[0] + 0.7;
            pos2[3 * i + 1] = s * x + c * y + com[1] - 0.2;
            pos2[3 * i + 2] = pos[3 * i + 2] + 0.3;
            v2[3 * i] = c * v[3 * i] - s * v[3 * i + 1];
            v2[3 * i + 1] = s * v[3 * i] + c * v[3 * i + 1];
            v2[3 * i + 2] = v[3 * i + 2];
            rpv[3 * i] = c * pv[3 * i] - s * pv[3 * i + 1];
            rpv[3 * i + 1] = s * pv[3 * i] + c * pv[3 * i + 1];
            rpv[3 * i + 2] = pv[3 * i + 2];
        }
        ws.project(&mut v2, &RigidRequest::cartesian(&pos2, RigidMode::Auto));
        for i in 0..12 {
            assert!((v2[i] - rpv[i]).abs() < 1e-8, "{i} {} {}", v2[i], rpv[i]);
        }
    }
