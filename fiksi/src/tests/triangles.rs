// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{System, constraints, elements, utils::sum_squares};

#[test]
fn connected_triangles() {
    let mut s = System::new();

    let p1 = elements::Point::create(&mut s, 0., 0.);
    let p2 = elements::Point::create(&mut s, 1., 0.5);
    let p3 = elements::Point::create(&mut s, 2., 1.);
    let p4 = elements::Point::create(&mut s, 3., 1.5);
    let p5 = elements::Point::create(&mut s, 4., 2.);
    let p6 = elements::Point::create(&mut s, 5., 2.5);

    constraints::PointPointPointAngle::create(&mut s, p6, p1, p2, -135_f64.to_radians());
    constraints::PointPointPointAngle::create(&mut s, p2, p3, p4, -120_f64.to_radians());
    constraints::PointPointPointAngle::create(&mut s, p4, p5, p6, -115_f64.to_radians());

    constraints::PointPointDistance::create(&mut s, p1, p2, 7.);
    constraints::PointPointDistance::create(&mut s, p2, p3, 5.);
    constraints::PointPointDistance::create(&mut s, p3, p4, 9.);
    constraints::PointPointDistance::create(&mut s, p4, p5, 8.);
    constraints::PointPointDistance::create(&mut s, p5, p6, 6.);
    constraints::PointPointDistance::create(&mut s, p6, p1, 7.);

    s.solve(None, crate::SolvingOptions::default());

    let sum_squared_residuals =
        sum_squares(s.get_constraint_handles().map(|c| c.calculate_residual(&s)));
    assert!(
        sum_squared_residuals < 1e-8,
        "The system was not solved (sum of squared residuals: {sum_squared_residuals})"
    );
}
