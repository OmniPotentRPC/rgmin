    #[test]
    fn slab_keeps_the_open_axis() {
        let pos = [0.0, 0.0, 1.0, 1.0, 0.0, 1.1, 0.2, 1.0, 0.9, 1.1, 0.8, 1.2];
        let mut req = RigidRequest::cartesian(&pos, RigidMode::Auto);
        req.pbc = [true, true, false];
        let mut ws = RigidWorkspace::new();
        let mut tz = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        ws.project(&mut tz, &req);
        assert!(nrm(&tz) > 0.5, "open axis removed {tz:?}");
        let mut tx = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        ws.project(&mut tx, &req);
        assert!(nrm(&tx) < 1e-10, "periodic translation survived {tx:?}");
        assert_eq!(ws.rank(), 2);
    }

    #[test]
    fn translation_rotation_on_a_cell_keeps_rotations() {
        let pos = [0.0, 0.0, 0.0, 1.0, 0.2, 0.1, 0.1, 1.1, 0.4];
        let mut req = RigidRequest::cartesian(&pos, RigidMode::TranslationRotation);
        req.pbc = [true, true, true];
        let mut ws = RigidWorkspace::new();
        let mut v = [0.0; 9];
        ws.project(&mut v, &req);
        assert_eq!(ws.rank(), 6, "cell rank {}", ws.rank());
    }

    #[test]
    fn broken_translation_is_not_removed() {
        let pos = [0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.2, 1.0, 0.4];
        let mut req = RigidRequest::cartesian(&pos, RigidMode::Auto);
        req.broken_translation = [false, false, true];
        let mut ws = RigidWorkspace::new();
        let mut tz = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        ws.project(&mut tz, &req);
        assert!(nrm(&tz) > 0.5, "broken tz was removed {tz:?}");
        let mut tx = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        ws.project(&mut tx, &req);
        assert!(nrm(&tx) < 1e-8, "tx survived {tx:?}");
    }

    fn pin_cluster() -> [f64; 12] {
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn one_pinned_atom_keeps_three_rotations() {
        let pos = pin_cluster();
        let frozen = [1u8, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut req = RigidRequest::cartesian(&pos, RigidMode::Auto);
        req.frozen = Some(&frozen);
        let mut ws = RigidWorkspace::new();
        let mut v = [0.0; 12];
        ws.project(&mut v, &req);
        assert_eq!(ws.rank(), 3, "pin rank {}", ws.rank());
        // Rotation about the pin (the origin) along x: ω × r.
        let mut spin = [0.0; 12];
        spin[7] = 0.0;
        spin[8] = 1.0; // atom (0,1,0) -> (0,0,1)
        spin[10] = -1.0; // atom (0,0,1) -> (0,-1,0)
        ws.project(&mut spin, &req);
        assert!(nrm(&spin) < 1e-8, "spin about the pin survived {spin:?}");
        let mut stretch = [0.0; 12];
        stretch[3] = 1.0;
        ws.project(&mut stretch, &req);
        assert!(
            nrm(&stretch) > 1e-3,
            "radial stretch was removed {stretch:?}"
        );
    }

    #[test]
    fn two_pins_keep_one_rotation_and_three_leave_none() {
        let pos = pin_cluster();
        let mut two = RigidRequest::cartesian(&pos, RigidMode::Auto);
        let mask2 = [1u8, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        two.frozen = Some(&mask2);
        let mut ws = RigidWorkspace::new();
        let mut v = [0.0; 12];
        ws.project(&mut v, &two);
        assert_eq!(ws.rank(), 1, "two-pin rank {}", ws.rank());
        let mut three = two;
        let mask3 = [1u8, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0];
        three.frozen = Some(&mask3);
        ws.project(&mut v, &three);
        assert_eq!(ws.rank(), 0, "three-pin rank {}", ws.rank());
    }

    #[test]
    fn projected_hessian_kills_a_rigid_vector() {
        let pos = [0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.1, 1.2, 0.3];
        let n = pos.len();
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            h[i * n + i] = 2.0 + 0.1 * i as f64;
            if i + 1 < n {
                h[i * n + i + 1] = 0.3;
                h[(i + 1) * n + i] = 0.3;
            }
        }
        let mut ws = RigidWorkspace::new();
        let req = RigidRequest::cartesian(&pos, RigidMode::Auto);
        ws.project_hessian(&mut h, &req);
        let e = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut he = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                he[i] += h[i * n + j] * e[j];
            }
        }
        assert!(nrm(&he) < 1e-8, "PHP translation {he:?}");
    }

    #[test]
    fn align_rotates_a_free_triangle_and_not_a_slab() {
        let reference = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 1.1, 0.3];
        let mut mobile = reference;
        let ang = 0.7_f64;
        let (c, s) = (ang.cos(), ang.sin());
        for i in 0..3 {
            let x = mobile[3 * i];
            let y = mobile[3 * i + 1];
            mobile[3 * i] = c * x - s * y + 0.4;
            mobile[3 * i + 1] = s * x + c * y - 0.2;
            mobile[3 * i + 2] += 0.5;
        }
        let mut free = mobile;
        let rmsd = align_rigid(&mut free, &reference, None, [false; 3]);
        assert!(rmsd < 1e-8, "rmsd {rmsd} mobile {free:?}");
        let mut slab = mobile;
        let before = slab;
        let _ = align_rigid(&mut slab, &reference, None, [true, true, false]);
        let moved = before
            .iter()
            .zip(slab.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        assert!(moved > 0.1, "periodic align rotated the cloud");
        let mut cm = [0.0; 3];
        let mut cr = [0.0; 3];
        for i in 0..3 {
            for d in 0..3 {
                cm[d] += slab[3 * i + d];
                cr[d] += reference[3 * i + d];
            }
        }
        for d in 0..3 {
            assert!((cm[d] / 3.0 - cr[d] / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn lj38_force_has_no_rigid_part() {
        let mut pos = vec![0.0; 38 * 3];
        let mut placed = 0;
        'fill: for ix in 0..4 {
            for iy in 0..4 {
                for iz in 0..4 {
                    if placed == 38 {
                        break 'fill;
                    }
                    pos[3 * placed] = ix as f64 * 1.1;
                    pos[3 * placed + 1] = iy as f64 * 1.1;
                    pos[3 * placed + 2] = iz as f64 * 1.1;
                    placed += 1;
                }
            }
        }
        let mut g = lj_grad(&pos);
        let mut ws = RigidWorkspace::new();
        ws.project(&mut g, &RigidRequest::cartesian(&pos, RigidMode::Auto));
        let mut com = [0.0; 3];
        let mut torque = [0.0; 3];
        let mut cpos = [0.0; 3];
        for i in 0..38 {
            for d in 0..3 {
                cpos[d] += pos[3 * i + d];
            }
        }
        for d in 0..3 {
            cpos[d] /= 38.0;
        }
        for i in 0..38 {
            let r = [
                pos[3 * i] - cpos[0],
                pos[3 * i + 1] - cpos[1],
                pos[3 * i + 2] - cpos[2],
            ];
            let f = [g[3 * i], g[3 * i + 1], g[3 * i + 2]];
            for d in 0..3 {
                com[d] += f[d];
            }
            torque[0] += r[1] * f[2] - r[2] * f[1];
            torque[1] += r[2] * f[0] - r[0] * f[2];
            torque[2] += r[0] * f[1] - r[1] * f[0];
        }
        assert!(nrm(&com) < 1e-8, "com force {com:?}");
        assert!(nrm(&torque) < 1e-6, "torque {torque:?}");
    }
