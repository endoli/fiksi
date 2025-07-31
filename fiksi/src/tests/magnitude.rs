// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{System, constraints, elements, utils::sum_squares};

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
