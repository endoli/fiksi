// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{System, constraints, elements, utils::sum_squares};

use super::RESIDUAL_THRESHOLD;

/// This configuration is rigidly solvable, but its starting configuration is singular.
///
/// Simply following the gradient won't reach the optimum solution.
#[test]
fn collinear_points() {
    let mut s = System::new();

    let p0 = elements::Point::create(&mut s, 0., 0.);
    let p1 = elements::Point::create(&mut s, 3., 0.);
    let p2 = elements::Point::create(&mut s, 6., 0.);

    constraints::PointPointDistance::create(&mut s, p0, p1, 1.);
    constraints::PointPointDistance::create(&mut s, p0, p2, 1.);
    constraints::PointPointDistance::create(&mut s, p1, p2, 1.);

    s.solve(
        None,
        crate::SolvingOptions {
            decompose: false,
            ..crate::SolvingOptions::default()
        },
    );

    let sum_squared_residuals = sum_squares(
        s.get_constraint_handles()
            .map(|constraint| constraint.calculate_residual(&s)),
    );
    assert!(
        sum_squared_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (sum of squared residuals: {sum_squared_residuals})"
    );
}
