
    fn scaled(pos: &[f64], s: f64) -> Vec<f64> {
        pos.iter().map(|p| p * s).collect()
    }

    fn rank_at(pos: &[f64]) -> usize {
        let mut ws = RigidWorkspace::new();
        let mut v = vec![0.0; pos.len()];
        ws.project(&mut v, &RigidRequest::cartesian(pos, RigidMode::Auto));
        ws.rank()
    }

    fn rigid_motions(pos: &[f64]) -> Vec<Vec<f64>> {
        let nat = pos.len() / 3;
        let mut com = [0.0; 3];
        for i in 0..nat {
            for d in 0..3 {
                com[d] += pos[3 * i + d] / nat as f64;
            }
        }
        let mut out = Vec::new();
        for d in 0..3 {
            let mut t = vec![0.0; pos.len()];
            for i in 0..nat {
                t[3 * i + d] = 1.0;
            }
            out.push(t);
        }
        for d in 0..3 {
            let mut omega = [0.0; 3];
            omega[d] = 1.0;
            let mut r = vec![0.0; pos.len()];
            for i in 0..nat {
                let p = [
                    pos[3 * i] - com[0],
                    pos[3 * i + 1] - com[1],
                    pos[3 * i + 2] - com[2],
                ];
                r[3 * i] = omega[1] * p[2] - omega[2] * p[1];
                r[3 * i + 1] = omega[2] * p[0] - omega[0] * p[2];
                r[3 * i + 2] = omega[0] * p[1] - omega[1] * p[0];
            }
            out.push(r);
        }
        out
    }

    const SCALES: [f64; 11] = [
        1e-150, 1e-60, 1e-24, 1e-12, 1e-6, 1.0, 1e6, 1e12, 1e24, 1e60, 1e150,
    ];

    const CLUSTER: [f64; 12] = [
        0.0, 0.0, 0.0, 1.3, 0.1, -0.2, 0.4, 1.1, 0.3, -0.5, 0.2, 0.9,
    ];

    #[test]
    fn non_collinear_cluster_keeps_every_rigid_mode_at_any_scale() {
        for s in SCALES {
            let pos = scaled(&CLUSTER, s);
            assert_eq!(rank_at(&pos), 6, "rank at scale {s:e}");
        }
    }

    #[test]
    fn shrinking_a_cluster_does_not_hide_the_rotations() {
        let mut ws = RigidWorkspace::new();
        for s in [1e-6, 1e-9, 1e-12, 1e-15, 1e-30, 1e-100] {
            let pos = scaled(&CLUSTER, s);
            for mut motion in rigid_motions(&pos) {
                let before = nrm(&motion);
                ws.project(&mut motion, &RigidRequest::cartesian(&pos, RigidMode::Auto));
                assert!(
                    nrm(&motion) <= 1e-10 * before,
                    "motion survived at scale {s:e}: {} of {before}",
                    nrm(&motion)
                );
            }
        }
    }

    #[test]
    fn growing_a_cluster_does_not_hide_the_translations() {
        let mut ws = RigidWorkspace::new();
        for s in [1e6, 1e9, 1e12, 1e15, 1e30, 1e100] {
            let pos = scaled(&CLUSTER, s);
            for mut motion in rigid_motions(&pos) {
                let before = nrm(&motion);
                ws.project(&mut motion, &RigidRequest::cartesian(&pos, RigidMode::Auto));
                assert!(
                    nrm(&motion) <= 1e-10 * before,
                    "motion survived at scale {s:e}: {} of {before}",
                    nrm(&motion)
                );
            }
        }
    }

    #[test]
    fn projection_commutes_with_uniform_scaling() {
        let v = [0.3, -0.2, 0.5, 0.1, 0.4, -0.6, -0.3, 0.2, 0.2, 0.7, -0.1, 0.0];
        let mut base = v;
        let mut ws = RigidWorkspace::new();
        ws.project(&mut base, &RigidRequest::cartesian(&CLUSTER, RigidMode::Auto));
        for s in [1e-12, 1e-6, 1e6, 1e12] {
            let pos = scaled(&CLUSTER, s);
            let mut w = v;
            ws.project(&mut w, &RigidRequest::cartesian(&pos, RigidMode::Auto));
            for (a, b) in w.iter().zip(&base) {
                assert!((a - b).abs() < 1e-9, "scale {s:e}: {w:?} vs {base:?}");
            }
        }
    }

    #[test]
    fn collinear_and_coincident_rank_does_not_depend_on_scale() {
        let line = [0.0, 0.0, -1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1];
        let stacked = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        for s in SCALES {
            assert_eq!(rank_at(&scaled(&line, s)), 5, "line at scale {s:e}");
            assert_eq!(rank_at(&scaled(&stacked, s)), 3, "stack at scale {s:e}");
            assert_eq!(rank_at(&scaled(&[0.2, -0.4, 0.7], s)), 3, "atom at {s:e}");
        }
    }

    #[test]
    fn near_collinear_rank_follows_the_geometry_not_the_scale() {
        for s in [1e-12, 1.0, 1e12] {
            let bent = scaled(&[0.0, 0.0, -1.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1.0], s);
            assert_eq!(rank_at(&bent), 6, "bent at scale {s:e}");
            let straight = scaled(&[0.0, 0.0, -1.0, 0.0, 1e-13, 0.0, 0.0, 0.0, 1.0], s);
            assert_eq!(rank_at(&straight), 5, "straight at scale {s:e}");
        }
    }

    #[test]
    fn non_collinear_triangle_has_horizontal_rank_three_exactly() {
        let tri = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.8, 0.1];
        let mut ws = RigidWorkspace::new();
        let mut v = [0.0; 9];
        ws.project(&mut v, &RigidRequest::cartesian(&tri, RigidMode::Auto));
        assert_eq!(ws.rank(), 6);
        let mut kept = Vec::<Vec<f64>>::new();
        for axis in 0..9 {
            let mut e = vec![0.0; 9];
            e[axis] = 1.0;
            ws.project(&mut e, &RigidRequest::cartesian(&tri, RigidMode::Auto));
            for q in &kept {
                let d: f64 = e.iter().zip(q).map(|(a, b)| a * b).sum();
                for (a, b) in e.iter_mut().zip(q) {
                    *a -= d * b;
                }
            }
            let len = nrm(&e);
            if len > 1e-10 {
                kept.push(e.iter().map(|a| a / len).collect());
            }
        }
        assert_eq!(kept.len(), 3, "horizontal dimension is 3N - 6");
    }

    #[test]
    fn pinned_atom_keeps_three_rotations_at_any_scale() {
        for s in [1e-12, 1e-6, 1.0, 1e6, 1e12] {
            let pos = scaled(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], s);
            let frozen = [1u8, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            let mut req = RigidRequest::cartesian(&pos, RigidMode::Auto);
            req.frozen = Some(&frozen);
            let mut ws = RigidWorkspace::new();
            let mut v = [0.0; 12];
            ws.project(&mut v, &req);
            assert_eq!(ws.rank(), 3, "pinned rank at scale {s:e}");
        }
    }
