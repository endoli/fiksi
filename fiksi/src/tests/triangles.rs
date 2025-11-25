// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{
    Decomposer, System, constraints, elements, tests::RESIDUAL_THRESHOLD, utils::root_mean_squares,
};

#[test]
fn single_triangle() {
    for decomposer in [
        Decomposer::None,
        Decomposer::SinglePass,
        Decomposer::RecursiveAssembly,
    ] {
        let mut s = System::new();

        let p0 = elements::Point::create(&mut s, 0., 0.);
        let p1 = elements::Point::create(&mut s, 1., 0.5);
        let p2 = elements::Point::create(&mut s, 2., 1.);

        constraints::PointPointDistance::create(&mut s, p0, p1, 1.);
        constraints::PointPointDistance::create(&mut s, p0, p2, 1.);
        constraints::PointPointDistance::create(&mut s, p1, p2, 1.);

        s.solve(crate::SolvingOptions {
            decomposer,
            ..crate::SolvingOptions::default()
        });

        let rms_residuals =
            root_mean_squares(s.get_constraint_handles().map(|c| c.calculate_residual(&s)));
        assert!(
            rms_residuals < RESIDUAL_THRESHOLD,
            "The system was not solved (root mean square residuals: {rms_residuals})"
        );
    }
}

#[test]
fn connected_triangles() {
    let mut s = System::new();

    let p0 = elements::Point::create(&mut s, 0., 0.);
    let p1 = elements::Point::create(&mut s, 1., 0.5);
    let p2 = elements::Point::create(&mut s, 2., 1.);
    let p3 = elements::Point::create(&mut s, 3., 1.5);
    let p4 = elements::Point::create(&mut s, 4., 2.);
    let p5 = elements::Point::create(&mut s, 5., 2.5);

    constraints::PointPointPointAngle::create(&mut s, p5, p0, p1, -135_f64.to_radians());
    constraints::PointPointPointAngle::create(&mut s, p1, p2, p3, -120_f64.to_radians());
    constraints::PointPointPointAngle::create(&mut s, p3, p4, p5, -115_f64.to_radians());

    constraints::PointPointDistance::create(&mut s, p0, p1, 7.);
    constraints::PointPointDistance::create(&mut s, p1, p2, 5.);
    constraints::PointPointDistance::create(&mut s, p2, p3, 9.);
    constraints::PointPointDistance::create(&mut s, p3, p4, 8.);
    constraints::PointPointDistance::create(&mut s, p4, p5, 6.);
    constraints::PointPointDistance::create(&mut s, p5, p0, 7.);

    s.solve(crate::SolvingOptions::DEFAULT);

    let rms_residuals =
        root_mean_squares(s.get_constraint_handles().map(|c| c.calculate_residual(&s)));
    assert!(
        rms_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (root mean square residuals: {rms_residuals})"
    );
}

/// Three rigid triangles hinged at a point (i.e., not connected rigidly).
#[test]
fn hinged_triangles() {
    let mut s = System::new();

    let p0 = elements::Point::create(&mut s, 0.5, 0.);
    let p1 = elements::Point::create(&mut s, 1.1, 0.5);
    let p2 = elements::Point::create(&mut s, 2.1, 1.);
    let p3 = elements::Point::create(&mut s, 3.1, 1.5);
    let p4 = elements::Point::create(&mut s, 4.1, 2.);
    let p5 = elements::Point::create(&mut s, 5.1, 2.5);
    let p6 = elements::Point::create(&mut s, 6.1, 3.);

    constraints::PointPointDistance::create(&mut s, p0, p1, 1.); // 0
    constraints::PointPointDistance::create(&mut s, p0, p2, 1.); // 1
    constraints::PointPointDistance::create(&mut s, p1, p2, 1.); // 2

    constraints::PointPointDistance::create(&mut s, p0, p3, 1.); // 3
    constraints::PointPointDistance::create(&mut s, p0, p4, 1.); // 4
    constraints::PointPointDistance::create(&mut s, p3, p4, 1.); // 5

    constraints::PointPointDistance::create(&mut s, p0, p5, 1.); // 6
    constraints::PointPointDistance::create(&mut s, p0, p6, 1.); // 7
    constraints::PointPointDistance::create(&mut s, p5, p6, 1.); // 8

    s.solve(crate::SolvingOptions::DEFAULT);

    let rms_residuals =
        root_mean_squares(s.get_constraint_handles().map(|c| c.calculate_residual(&s)));
    assert!(
        rms_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (root mean square residuals: {rms_residuals})"
    );
}
