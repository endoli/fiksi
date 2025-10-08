// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{Decomposer, System, constraints, elements, utils::sum_squares};

/// A triangle where one point is kept fixed.
#[test]
fn single_triangle_with_fixed_point() {
    for decomposer in [Decomposer::None, Decomposer::SinglePass] {
        let mut s = System::new();

        let p0 = elements::Point::create(&mut s, 0., 0.);
        let p1 = elements::Point::create(&mut s, 1., 0.5);
        let p2 = elements::Point::create(&mut s, 2., 1.);

        p1.fix(&mut s);

        constraints::PointPointDistance::create(&mut s, p0, p1, 1.);
        constraints::PointPointDistance::create(&mut s, p0, p2, 1.);
        constraints::PointPointDistance::create(&mut s, p1, p2, 1.);

        s.solve(crate::SolvingOptions {
            decomposer,
            ..crate::SolvingOptions::default()
        });

        let sum_squared_residuals =
            sum_squares(s.get_constraint_handles().map(|c| c.calculate_residual(&s)));
        assert!(
            sum_squared_residuals < 1e-6,
            "The system was not solved (sum of squared residuals: {sum_squared_residuals})"
        );

        assert_eq!(
            p1.get_value(&s),
            kurbo::Point::new(1., 0.5),
            "The point that was to be kept fixed is no longer identical to its starting value"
        );
    }
}

/// A fully identified system of a point incident on a circle, where the point and the circle's
/// center are fixed; i.e., only the circle's radius is free.
#[test]
fn fixed_point_and_circle_center_incidence() {
    for decomposer in [Decomposer::None, Decomposer::SinglePass] {
        let mut s = System::new();

        let p0 = elements::Point::create(&mut s, 0., 0.);
        let center = elements::Point::create(&mut s, 4., 3.); // At a distance of 5 units from the origin
        let radius = elements::Length::create(&mut s, 1.);
        let circle = elements::Circle::create(&mut s, center, radius);

        p0.fix(&mut s);
        center.fix(&mut s);

        constraints::PointCircleIncidence::create(&mut s, p0, circle);

        s.solve(crate::SolvingOptions {
            decomposer,
            ..crate::SolvingOptions::default()
        });

        assert_eq!(
            p0.get_value(&s),
            kurbo::Point::new(0., 0.),
            "The point that was to be kept fixed is no longer identical to its starting value"
        );
        assert_eq!(
            center.get_value(&s),
            kurbo::Point::new(4., 3.),
            "The circle center that was to be kept fixed is no longer identical to its starting value"
        );
        assert!(
            (radius.get_value(&s) - 5.).abs() < 1e-6,
            "The circle's radius ({}) should be 5",
            radius.get_value(&s)
        );
    }
}
