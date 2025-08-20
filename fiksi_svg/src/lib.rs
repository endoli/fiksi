// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This renders Fiksi systems to an SVG.

// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET

use core::f64;
use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
};

pub use color;
pub use fiksi;

use color::{AlphaColor, ColorSpace, Oklab};
use fiksi::{
    AnyElementHandle, Element, ElementHandle, System,
    kurbo::{self, Shape},
};

const DEFAULT_ELEMENT_COLOR: AlphaColor<Oklab> = AlphaColor::<Oklab>::new([0., 0., 0., 1.]);

/// A Fiksi system renderer for rendering [Fiksi systems](System) into an SVG.
#[derive(Debug)]
pub struct SystemRenderer {
    colors: HashMap<AnyElementHandle, AlphaColor<Oklab>>,
    hidden: HashSet<AnyElementHandle>,
}

impl SystemRenderer {
    /// Construct a new SVG builder, to build an SVG from [Fiksi systems](System).
    pub fn new() -> Self {
        Self {
            colors: HashMap::new(),
            hidden: HashSet::new(),
        }
    }

    /// Set the draw color of an element.
    pub fn set_element_color<T: Element, CS: ColorSpace>(
        &mut self,
        element: ElementHandle<T>,
        color: AlphaColor<CS>,
    ) {
        let ok = color.convert::<Oklab>();
        self.colors.insert(element.as_any_element(), ok);
    }

    /// Hide element.
    ///
    /// The element doesn't get drawn, and doesn't get taken into account for viewbox calculation.
    pub fn hide_element<T: Element>(&mut self, element: ElementHandle<T>) {
        self.hidden.insert(element.as_any_element());
    }

    /// Render the elements of the [Fiksi system](System) to an SVG. The SVG is returned.
    ///
    /// - `viewbox` is the optional SVG viewbox to set. If not given, a bounding box is calculated
    ///   from the system's elements.
    /// - `stroke_width` is the width of strokes in the SVG.
    pub fn render_system(
        &mut self,
        viewbox: Option<kurbo::Rect>,
        stroke_width: f64,
        system: &System,
    ) -> String {
        use fiksi::ElementValue;

        let mut bbox = kurbo::Rect::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );

        // Two layers of drawing (we want to position points on top of lines, for example).
        let mut top = String::new();
        let mut bottom = String::new();

        for handle in system.get_element_handles() {
            if self.hidden.contains(&handle) {
                continue;
            }

            match handle.get_value(system) {
                ElementValue::Point(point) => {
                    let color = self.colors.get(&handle).unwrap_or(&DEFAULT_ELEMENT_COLOR);

                    bbox = bbox.union_pt(point);
                    write!(
                        &mut top,
                        r#"<circle cx="{}" cy="{}" r="{}" stroke="{:0X}" stroke-width="{}" fill="{:0X}" id="point-{}"/>"#,
                        point.x,
                        point.y,
                        stroke_width,
                        color.to_rgba8(),
                        stroke_width * 0.25,
                        color.map_lightness(|l| (l + 0.3).clamp(0., 1.)).to_rgba8(),
                        handle.as_id()
                    )
                    .unwrap();
                }
                ElementValue::Line(line) => {
                    let color = self.colors.get(&handle).unwrap_or(&DEFAULT_ELEMENT_COLOR);

                    bbox = bbox.union(line.bounding_box());
                    write!(
                        &mut bottom,
                        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{:0X}" stroke-width="{}" id="line-{}"/>"#,
                        line.p0.x,
                        line.p0.y,
                        line.p1.x,
                        line.p1.y,
                        color.to_rgba8(),
                        stroke_width,
                        handle.as_id()
                    )
                    .unwrap();
                }
                ElementValue::Circle(circle) => {
                    let color = self.colors.get(&handle).unwrap_or(&DEFAULT_ELEMENT_COLOR);

                    bbox = bbox.union(circle.bounding_box());
                    write!(
                        &mut bottom,
                        r#"<circle cx="{}" cy="{}" r="{}" stroke="{:0X}" stroke-width="{}" fill="none" id="circle-{}"/>"#,
                        circle.center.x,
                        circle.center.y,
                        circle.radius,
                        color.to_rgba8(),
                        stroke_width,
                        handle.as_id()
                    ).unwrap();
                }
            }
        }

        let inflate = bbox.width().max(bbox.height()) * 0.1;
        bbox = bbox.inflate(inflate, inflate);
        let viewbox = viewbox.unwrap_or(bbox);
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{} {} {} {}\">{}{}</svg>",
            viewbox.x0,
            viewbox.y0,
            viewbox.width(),
            viewbox.height(),
            bottom,
            top,
        )
    }
}

impl Default for SystemRenderer {
    fn default() -> Self {
        Self::new()
    }
}
