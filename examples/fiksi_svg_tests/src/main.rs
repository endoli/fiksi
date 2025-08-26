// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! An example using the Fiksi SVG builder.

use fiksi::{System, constraints, elements};
use fiksi_svg::{SystemRenderer, color};

fn circle_triangle_line() -> std::io::Result<()> {
    let mut s = System::new();

    let p1 = elements::Point::create(&mut s, 10., 0.);
    let p2 = elements::Point::create(&mut s, 20., 10.);
    let p3 = elements::Point::create(&mut s, 30., -10.);
    let p4 = elements::Point::create(&mut s, -40., -50.);
    let p5 = elements::Point::create(&mut s, 40., -50.);

    // Configure points `(p1, p2, p3)` into a triangle with two given angles and one given side
    // length.
    constraints::PointPointPointAngle::create(&mut s, p1, p2, p3, 40_f64.to_radians());
    constraints::PointPointPointAngle::create(&mut s, p2, p3, p1, 70_f64.to_radians());
    constraints::PointPointDistance::create(&mut s, p1, p2, 70.);

    let triangle_side1 = elements::Line::create(&mut s, p1, p2);
    let triangle_side2 = elements::Line::create(&mut s, p2, p3);
    let triangle_side3 = elements::Line::create(&mut s, p1, p3);

    // A circle coincident with one of the triangle corners.
    let radius = elements::Length::create(&mut s, 5.);
    let circle = elements::Circle::create(&mut s, p3, radius);
    // The (boundless) line representing the opposite triangle side must be tangent on the circle.
    constraints::LineCircleTangency::create(&mut s, triangle_side1, circle);

    // A line.
    let line = elements::Line::create(&mut s, p4, p5);
    // There must be a (counterclockwise) 90 degree angle from one of the triangle sides to the
    // start of the line.
    constraints::LineLineAngle::create(&mut s, triangle_side3, line, -90_f64.to_radians());

    // The circle's center (also one of the triangle corners) must be incident on the line.
    constraints::PointLineIncidence::create(&mut s, p3, line);
    // One of the line's points must be a given distance from the circle center, and the line's
    // points must be a given distance from each other.
    constraints::PointPointDistance::create(&mut s, p3, p4, 40.);
    constraints::PointPointDistance::create(&mut s, p4, p5, 80.);

    let mut renderer = SystemRenderer::new();
    for el in [triangle_side1, triangle_side2, triangle_side3] {
        renderer.set_element_color(
            el,
            color::AlphaColor::<color::Oklch>::new([0.5, 0.1, 0., 1.]),
        );
    }
    renderer.set_element_color(
        line,
        color::AlphaColor::<color::Oklch>::new([0.5, 0.1, 80., 1.]),
    );
    renderer.set_element_color(
        circle,
        color::AlphaColor::<color::Oklch>::new([0.5, 0.1, 160., 1.]),
    );

    std::fs::write(
        "./circle-triangle-line-initial.svg",
        renderer.render_system(None, 1., &s),
    )?;

    s.solve(fiksi::SolvingOptions::DEFAULT);
    std::fs::write(
        "./circle-triangle-line-solved.svg",
        renderer.render_system(None, 1., &s),
    )?;

    Ok(())
}

fn main() -> std::io::Result<()> {
    circle_triangle_line()?;

    Ok(())
}
