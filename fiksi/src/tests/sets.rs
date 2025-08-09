// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests subsets of systems, by having free vs fixed elements or ignoring some constraints.

use crate::{System, constraints, elements};

use super::RESIDUAL_THRESHOLD;

/// Fixes two points (defining an implicit line), and makes a free line parallel to that).
#[test]
fn make_line_parallel_to_fixed_points() {
    let mut s = System::new();

    const P0: kurbo::Point = kurbo::Point::new(0., 0.);
    const P1: kurbo::Point = kurbo::Point::new(1., 1.);

    let p0 = elements::Point::create(&mut s, P0.x, P0.y);
    let p1 = elements::Point::create(&mut s, P1.x, P1.y);

    let p2 = elements::Point::create(&mut s, 0., 0.);
    let p3 = elements::Point::create(&mut s, 1., 0.);

    let line = elements::Line::create(&mut s, p2, p3);

    let solve_set = s.create_solve_set();
    let p0l = constraints::PointLineIncidence::create(&mut s, p0, line);
    let p1l = constraints::PointLineIncidence::create(&mut s, p1, line);
    let p2p3d = constraints::PointPointDistance::create(&mut s, p2, p3, P0.distance(P1));

    s.add_constraint_to_solve_set(&solve_set, &p0l);
    s.add_constraint_to_solve_set(&solve_set, &p1l);
    s.add_constraint_to_solve_set(&solve_set, &p2p3d);
    s.add_element_to_solve_set(&solve_set, &p2);
    s.add_element_to_solve_set(&solve_set, &p3);

    s.solve(Some(&solve_set), crate::SolvingOptions::default());

    let val_p2 = p2.get_value(&s);
    let val_p3 = p3.get_value(&s);
    let vec0 = P1 - P0;
    let vec1 = val_p3 - val_p2;

    assert!(vec1.cross(vec0) < RESIDUAL_THRESHOLD);
    assert!((vec1.hypot2() - vec0.hypot2()).abs() < RESIDUAL_THRESHOLD);
}

/// Adds two trivially inconsistent constraints. Solve twice, ignoring one at a time.
#[test]
fn ignore_one_constraint() {
    let mut s = System::new();

    let p0 = elements::Point::create(&mut s, 0., 0.);
    let p1 = elements::Point::create(&mut s, 0., 1.);

    let p0p1_10 = constraints::PointPointDistance::create(&mut s, p0, p1, 10.);
    let p0p1_20 = constraints::PointPointDistance::create(&mut s, p0, p1, 20.);

    let solve_set_10 = s.create_solve_set();
    s.add_constraint_to_solve_set(&solve_set_10, &p0p1_10);
    s.add_element_to_solve_set(&solve_set_10, &p0);
    s.add_element_to_solve_set(&solve_set_10, &p1);

    let solve_set_20 = s.create_solve_set();
    s.add_constraint_to_solve_set(&solve_set_20, &p0p1_20);
    s.add_element_to_solve_set(&solve_set_20, &p0);
    s.add_element_to_solve_set(&solve_set_20, &p1);

    s.solve(Some(&solve_set_10), crate::SolvingOptions::default());
    assert!((p1.get_value(&s).distance(p0.get_value(&s)) - 10.).abs() < RESIDUAL_THRESHOLD);

    s.solve(Some(&solve_set_20), crate::SolvingOptions::default());
    assert!((p1.get_value(&s).distance(p0.get_value(&s)) - 20.).abs() < RESIDUAL_THRESHOLD);
}
