// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{System, constraints, elements, utils::root_mean_squares};

use super::RESIDUAL_THRESHOLD;

/// A rigid system with a large magnitude of metric point-point distance constraint values.
///
/// It is not necessarily numerically tricky, as all its elements and constraints have values
/// within the same order of magnitude.
#[test]
fn large_order_of_magnitude() {
    let mut s = System::new();

    const FACTOR: f64 = 1e20;

    let p0 = elements::Point::create(&mut s, 1.5 * FACTOR, 6.5 * FACTOR);
    let p1 = elements::Point::create(&mut s, 3.2 * FACTOR, 0.8 * FACTOR);
    let p2 = elements::Point::create(&mut s, 2.2 * FACTOR, -1.5 * FACTOR);

    constraints::PointPointDistance::create(&mut s, p0, p1, 5. * FACTOR);
    constraints::PointPointDistance::create(&mut s, p0, p2, 3. * FACTOR);
    constraints::PointPointDistance::create(&mut s, p1, p2, 4. * FACTOR);

    s.solve(crate::SolvingOptions::default());

    extern crate std;

    let rms_residuals = root_mean_squares(
        s.get_constraint_handles()
            .map(|constraint| constraint.calculate_residual(&s)),
    );
    assert!(
        rms_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (root mean square residuals: {rms_residuals})"
    );
}

/// A rigid system with a large magnitude of metric point-point distance constraint values and a
/// singular line-line parallelism constraint.
///
/// It can be hard to solve as the line-line parallelism residual value (a cross product) will
/// roughly be on the order of the square of the point-point distance constraints (a vector norm),
/// meaning it is at double the order of magnitude. This quickly introduces numerical issues.
#[test]
fn metric_and_singular() {
    let mut s = System::new();

    const FACTOR: f64 = 1e7;

    let p0 = elements::Point::create(&mut s, 1.5 * FACTOR, 6.5 * FACTOR);
    let p1 = elements::Point::create(&mut s, 3.2 * FACTOR, 0.8 * FACTOR);
    let p2 = elements::Point::create(&mut s, 2.2 * FACTOR, -1.5 * FACTOR);
    let p3 = elements::Point::create(&mut s, 1.2 * FACTOR, 0.5 * FACTOR);

    constraints::PointPointDistance::create(&mut s, p0, p1, 5. * FACTOR);
    constraints::PointPointDistance::create(&mut s, p1, p2, 4. * FACTOR);
    constraints::PointPointDistance::create(&mut s, p2, p3, 3. * FACTOR);
    constraints::PointPointDistance::create(&mut s, p3, p1, 1. * FACTOR);

    let line0 = elements::Line::create(&mut s, p0, p1);
    let line1 = elements::Line::create(&mut s, p2, p3);
    constraints::LineLineParallelism::create(&mut s, line0, line1);

    s.solve(crate::SolvingOptions::default());

    let rms_residuals = root_mean_squares(
        s.get_constraint_handles()
            .map(|constraint| constraint.calculate_residual(&s)),
    );
    assert!(
        rms_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (root mean square residuals: {rms_residuals})"
    );
}

// A near-degenerate isosceles triangle where the two equal sides have length of a relatively large
// magnitude, and the third side is relatively small.
//
// It can be hard to solve as the point-point distances span a large order of magnitude.
#[test]
fn near_degenerate_isosceles_triangle() {
    let mut s = System::new();

    const FACTOR: f64 = 1e13;

    let p0 = elements::Point::create(&mut s, 1.5 * FACTOR, 6.5 * FACTOR);
    let p1 = elements::Point::create(&mut s, 3.2 * FACTOR, 0.8 * FACTOR);
    let p2 = elements::Point::create(&mut s, 2.2, -1.5);

    constraints::PointPointDistance::create(&mut s, p0, p1, 4. * FACTOR + 1.);
    constraints::PointPointDistance::create(&mut s, p1, p2, 4. * FACTOR + 1.);
    constraints::PointPointDistance::create(&mut s, p0, p2, 1.);

    s.solve(crate::SolvingOptions::default());

    let rms_residuals = root_mean_squares(
        s.get_constraint_handles()
            .map(|constraint| constraint.calculate_residual(&s)),
    );
    assert!(
        rms_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (root mean square residuals: {rms_residuals})"
    );
}
