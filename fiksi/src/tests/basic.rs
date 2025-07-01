// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{System, constraints, elements, utils::sum_squares};

/// A "good enough" sum of squared residuals that is considered to have solved the system.
///
/// This would normally depend on the domain, especially for things like distance constraints.
const RESIDUAL_THRESHOLD: f64 = 1e-5;

/// Tests whether an under-constrained triangle configuration gets solved.
#[test]
fn triangle() {
    let mut s = System::new();
    let element_set = s.add_element_set();
    let constraint_set = s.add_constraint_set();

    let p1 = s.add_element(&[&element_set], elements::Point { x: 0., y: 0. });
    let p2 = s.add_element(&[&element_set], elements::Point { x: 1., y: 0.5 });
    let p3 = s.add_element(&[&element_set], elements::Point { x: 2., y: 1. });
    let angle1 = s.add_constraint(
        &[&constraint_set],
        constraints::PointPointPointAngle::new(&p1, &p2, &p3, 40_f64.to_radians()),
    );
    let angle2 = s.add_constraint(
        &[&constraint_set],
        constraints::PointPointPointAngle::new(&p2, &p3, &p1, 80_f64.to_radians()),
    );
    s.solve(
        &element_set,
        &constraint_set,
        crate::SolvingOptions::default(),
    );

    let sum_squared_residuals = sum_squares(&[
        s.calculate_constraint_residual(&angle1),
        s.calculate_constraint_residual(&angle2),
    ]);
    assert!(
        sum_squared_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (sum of squared residuals: {sum_squared_residuals})"
    );
}

/// Tests whether in a partially overconstrained system the non-overconstrained part gets solved.
#[test]
fn overconstrained_triangle_line_incidence() {
    let mut s = System::new();
    let element_set = s.add_element_set();
    let constraint_set = s.add_constraint_set();

    let p1 = s.add_element(&[&element_set], elements::Point { x: 0., y: 0. });
    let p2 = s.add_element(&[&element_set], elements::Point { x: 1., y: 0.5 });
    let p3 = s.add_element(&[&element_set], elements::Point { x: 2., y: 1. });
    let p4 = s.add_element(&[&element_set], elements::Point { x: 3., y: 1.5 });
    let line1 = s.add_element(&[&element_set], elements::Line::new(&p3, &p4));
    // Overconstrain the triangle angles to something that's geometrically impossible.
    let angle1 = s.add_constraint(
        &[&constraint_set],
        constraints::PointPointPointAngle::new(&p1, &p2, &p3, 40_f64.to_radians()),
    );
    let angle2 = s.add_constraint(
        &[&constraint_set],
        constraints::PointPointPointAngle::new(&p2, &p3, &p1, 80_f64.to_radians()),
    );
    let angle3 = s.add_constraint(
        &[&constraint_set],
        constraints::PointPointPointAngle::new(&p3, &p1, &p2, 100_f64.to_radians()),
    );
    let incidence = s.add_constraint(
        &[&constraint_set],
        constraints::PointLineIncidence::new(&p2, &line1),
    );
    s.solve(
        &element_set,
        &constraint_set,
        crate::SolvingOptions::default(),
    );

    let sum_squared_residuals = sum_squares(&[
        s.calculate_constraint_residual(&angle1),
        s.calculate_constraint_residual(&angle2),
        s.calculate_constraint_residual(&angle3),
    ]);
    assert!(
        sum_squared_residuals >= RESIDUAL_THRESHOLD,
        "The angle constraints were unexpectedly solved (this shouldn't be possible geometrically)"
    );

    let squared_incidence_residual = sum_squares(&[s.calculate_constraint_residual(&incidence)]);
    assert!(
        squared_incidence_residual < RESIDUAL_THRESHOLD,
        "The point-line incidence was not solved (sum of squared residuals: {sum_squared_residuals})"
    );
}
