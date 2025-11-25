// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{System, constraints, elements, tests::RESIDUAL_THRESHOLD, utils::root_mean_squares};

/// This configuration is rigidly solvable, but its starting configuration is numerically singular.
///
/// Following the gradients, the system converges to a saddle ridge: one or more of the points
/// needs to get a y-offset, but the y-gradients are precisely zero. Simply following the gradient
/// won't reach the optimum solution.
///
/// With decomposition and recognizing some common patterns, these can be solved analytically. The
/// more general solution is to introduce randomness to jitter systems out of local optima.
/// Randomness is also needed for numerical analysis (e.g., to numerically detect
/// overconstrainedness), for the same reason.
#[test]
fn collinear_points() {
    let mut s = System::new();

    let p0 = elements::Point::create(&mut s, 0., 0.);
    let p1 = elements::Point::create(&mut s, 3., 0.);
    let p2 = elements::Point::create(&mut s, 6., 0.);

    constraints::PointPointDistance::create(&mut s, p0, p1, 1.);
    constraints::PointPointDistance::create(&mut s, p0, p2, 1.);
    constraints::PointPointDistance::create(&mut s, p1, p2, 1.);

    s.solve(crate::SolvingOptions::DEFAULT);

    let rms_residuals = root_mean_squares(
        s.get_constraint_handles()
            .map(|constraint| constraint.calculate_residual(&s)),
    );
    assert!(
        rms_residuals < RESIDUAL_THRESHOLD,
        "The system was not solved (root mean square residuals: {rms_residuals})"
    );
}
