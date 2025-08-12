// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{System, constraints, elements, utils::sum_squares};

use super::RESIDUAL_THRESHOLD;

/// Tests whether an under-constrained triangle configuration gets solved.
#[test]
fn coincident_points() {
    let mut s = System::new();

    let p0 = elements::Point::create(&mut s, 0., 0.);
    let p1 = elements::Point::create(&mut s, 1., 0.5);
    let coincidence = constraints::PointPointCoincidence::create(&mut s, p0, p1);

    s.solve(None, crate::SolvingOptions::default());

    let sum_squared_residuals = sum_squares([coincidence.calculate_residual(&s)]);
    assert!(
        sum_squared_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (sum of squared residuals: {sum_squared_residuals})"
    );
}

/// Tests whether an under-constrained triangle configuration gets solved.
#[test]
fn underconstrained_triangle() {
    let mut s = System::new();

    let p0 = elements::Point::create(&mut s, 0., 0.);
    let p1 = elements::Point::create(&mut s, 1., 0.5);
    let p2 = elements::Point::create(&mut s, 2., 1.);
    let angle0 = constraints::PointPointPointAngle::create(&mut s, p0, p1, p2, 40_f64.to_radians());
    let angle1 = constraints::PointPointPointAngle::create(&mut s, p1, p2, p0, 80_f64.to_radians());
    s.solve(None, crate::SolvingOptions::default());

    let sum_squared_residuals =
        sum_squares([angle0.calculate_residual(&s), angle1.calculate_residual(&s)]);
    assert!(
        sum_squared_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (sum of squared residuals: {sum_squared_residuals})"
    );
}

/// Tests whether in a partially overconstrained system the non-overconstrained part gets solved.
#[test]
fn overconstrained_triangle_line_incidence() {
    let mut s = System::new();

    let p0 = elements::Point::create(&mut s, 0., 0.);
    let p1 = elements::Point::create(&mut s, 1., 0.5);
    let p2 = elements::Point::create(&mut s, 2., 1.);
    let p3 = elements::Point::create(&mut s, 3., 1.5);
    let line0 = elements::Line::create(&mut s, p2, p3);
    // Overconstrain the triangle angles to something that's geometrically impossible.
    let angle0 = constraints::PointPointPointAngle::create(&mut s, p0, p1, p2, 40_f64.to_radians());
    let angle1 = constraints::PointPointPointAngle::create(&mut s, p1, p2, p0, 80_f64.to_radians());
    let angle2 =
        constraints::PointPointPointAngle::create(&mut s, p2, p0, p1, 100_f64.to_radians());
    let incidence = constraints::PointLineIncidence::create(&mut s, p1, line0);
    s.solve(None, crate::SolvingOptions::default());

    let sum_squared_residuals = sum_squares([
        angle0.calculate_residual(&s),
        angle1.calculate_residual(&s),
        angle2.calculate_residual(&s),
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

#[test]
fn overconstrained() {
    let mut s = System::new();

    // A system of four points with some pairwise distance constraints. The system is
    // overconstrained. Dropping the distance constraint on e.g. p1p4 makes the system rigid.
    let p0 = elements::Point::create(&mut s, 0.123, 0.1);
    let p1 = elements::Point::create(&mut s, 1.2, 0.);
    let p2 = elements::Point::create(&mut s, -0.5, 1.1);
    let p3 = elements::Point::create(&mut s, 1.599, 1.2);

    constraints::PointPointDistance::create(&mut s, p0, p1, 1.);
    constraints::PointPointDistance::create(&mut s, p0, p2, 1.5);
    constraints::PointPointDistance::create(&mut s, p1, p3, 1.7);
    constraints::PointPointDistance::create(&mut s, p2, p3, 1.2);
    constraints::PointPointDistance::create(&mut s, p1, p2, 2.);
    let p0p3 = constraints::PointPointDistance::create(&mut s, p0, p3, 5.);

    let analysis = s.analyze(None);
    // Note we don't guarantee a specific ordering in which a constraint is designated as
    // causing overconstrainedness. Currently it's always the constraint that was added later,
    // though.
    assert_eq!(&analysis.overconstrained, &[p0p3.as_any_constraint()]);
}

/// A rigid triangle system with a rigid inscribed circle.
#[test]
fn triangle_inscribed_circle() {
    let mut s = System::new();

    let p0 = elements::Point::create(&mut s, 0., 0.);
    let p1 = elements::Point::create(&mut s, 1., 0.5);
    let p2 = elements::Point::create(&mut s, 1.5, 1.);
    let p3 = elements::Point::create(&mut s, 2.8, 1.5);

    constraints::PointPointDistance::create(&mut s, p0, p1, 1.);
    constraints::PointPointDistance::create(&mut s, p0, p2, 1.);
    constraints::PointPointDistance::create(&mut s, p1, p2, 1.);

    let line0 = elements::Line::create(&mut s, p0, p1);
    let line1 = elements::Line::create(&mut s, p0, p2);
    let line2 = elements::Line::create(&mut s, p1, p2);
    let circle = elements::Circle::create(&mut s, p3, 1.);

    constraints::LineCircleTangency::create(&mut s, line0, circle);
    constraints::LineCircleTangency::create(&mut s, line1, circle);
    constraints::LineCircleTangency::create(&mut s, line2, circle);

    s.solve(None, crate::SolvingOptions::default());

    let sum_squared_residuals = sum_squares(
        s.get_constraint_handles()
            .map(|constraint| constraint.calculate_residual(&s)),
    );
    assert!(
        sum_squared_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (sum of squared residuals: {sum_squared_residuals})"
    );
}

#[test]
fn two_connected_components() {
    let mut s = System::new();

    let p0 = elements::Point::create(&mut s, 0.123, 0.1);
    let p1 = elements::Point::create(&mut s, 1.2, 0.);
    let p2 = elements::Point::create(&mut s, -0.5, 1.1);
    let p3 = elements::Point::create(&mut s, 1.599, 1.2);

    let p0p1 = constraints::PointPointDistance::create(&mut s, p0, p1, 1.);
    let p2p3 = constraints::PointPointDistance::create(&mut s, p2, p3, 1.2);

    s.solve(None, crate::SolvingOptions::default());
    let sum_squared_residuals =
        sum_squares([p0p1.calculate_residual(&s), p2p3.calculate_residual(&s)]);
    assert!(
        sum_squared_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (sum of squared residuals: {sum_squared_residuals})"
    );
}
