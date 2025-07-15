// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{System, constraints, elements, utils::sum_squares};

/// A "good enough" sum of squared residuals that is considered to have solved the system.
///
/// This would normally depend on the domain, especially for things like distance constraints.
const RESIDUAL_THRESHOLD: f64 = 1e-5;

/// Tests whether an under-constrained triangle configuration gets solved.
#[test]
fn underconstrained_triangle() {
    let mut s = System::new();

    let p1 = elements::Point::create(&mut s, 0., 0.);
    let p2 = elements::Point::create(&mut s, 1., 0.5);
    let p3 = elements::Point::create(&mut s, 2., 1.);
    let angle1 = constraints::PointPointPointAngle::create(&mut s, p1, p2, p3, 40_f64.to_radians());
    let angle2 = constraints::PointPointPointAngle::create(&mut s, p2, p3, p1, 80_f64.to_radians());
    s.solve(None, crate::SolvingOptions::default());

    let sum_squared_residuals =
        sum_squares([angle1.calculate_residual(&s), angle2.calculate_residual(&s)]);
    assert!(
        sum_squared_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (sum of squared residuals: {sum_squared_residuals})"
    );
}

/// Tests whether in a partially overconstrained system the non-overconstrained part gets solved.
#[test]
fn overconstrained_triangle_line_incidence() {
    let mut s = System::new();

    let p1 = elements::Point::create(&mut s, 0., 0.);
    let p2 = elements::Point::create(&mut s, 1., 0.5);
    let p3 = elements::Point::create(&mut s, 2., 1.);
    let p4 = elements::Point::create(&mut s, 3., 1.5);
    let line1 = elements::Line::create(&mut s, p3, p4);
    // Overconstrain the triangle angles to something that's geometrically impossible.
    let angle1 = constraints::PointPointPointAngle::create(&mut s, p1, p2, p3, 40_f64.to_radians());
    let angle2 = constraints::PointPointPointAngle::create(&mut s, p2, p3, p1, 80_f64.to_radians());
    let angle3 =
        constraints::PointPointPointAngle::create(&mut s, p3, p1, p2, 100_f64.to_radians());
    let incidence = constraints::PointLineIncidence::create(&mut s, p2, line1);
    s.solve(None, crate::SolvingOptions::default());

    let sum_squared_residuals = sum_squares([
        angle1.calculate_residual(&s),
        angle2.calculate_residual(&s),
        angle3.calculate_residual(&s),
    ]);
    assert!(
        sum_squared_residuals >= RESIDUAL_THRESHOLD,
        "The angle constraints were unexpectedly solved (this shouldn't be possible geometrically)"
    );

    let squared_incidence_residual = sum_squares([incidence.calculate_residual(&s)]);
    assert!(
        squared_incidence_residual < RESIDUAL_THRESHOLD,
        "The point-line incidence was not solved (sum of squared residuals: {sum_squared_residuals})"
    );
}
