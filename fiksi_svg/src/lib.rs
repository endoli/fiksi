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

use std::{collections::HashMap, fmt::Write};

pub use color;
pub use fiksi;

use color::{AlphaColor, ColorSpace, Rgba8, Srgb};
use fiksi::{AnyElementHandle, Element, ElementHandle, System};

const DEFAULT_ELEMENT_COLOR: Rgba8 = Rgba8::from_u8_array([0, 0, 0, u8::MAX]);

/// An SVG builder for rendering [Fiksi systems](System) into an SVG.
#[derive(Debug)]
pub struct SvgBuilder {
    svg: String,
    colors: HashMap<AnyElementHandle, Rgba8>,
}

impl SvgBuilder {
    /// Construct a new SVG builder, to build an SVG from [Fiksi systems](System).
    pub fn new() -> Self {
        Self {
            svg: r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="-100 -100 200 200">"#.into(),
            colors: HashMap::new(),
        }
    }

    /// Set the draw color of an element.
    pub fn set_element_color<T: Element, CS: ColorSpace>(
        &mut self,
        element: ElementHandle<T>,
        color: AlphaColor<CS>,
    ) {
        let srgb = color.convert::<Srgb>().to_rgba8();
        self.colors.insert(element.as_any_element(), srgb);
    }

    /// Render the elements of the [Fiksi system](System) to the SVG.
    pub fn add_system_snapshot(&mut self, system: &System) {
        use fiksi::ElementValue;
        for handle in system.get_element_handles() {
            match handle.get_value(system) {
                ElementValue::Point(point) => {
                    let color = self.colors.get(&handle).unwrap_or(&DEFAULT_ELEMENT_COLOR);

                    write!(
                        &mut self.svg,
                        r#"<circle cx="{}" cy="{}" r="1" fill="{color:0X}" id="point-{}"/>"#,
                        point.x,
                        point.y,
                        handle.as_id()
                    )
                    .unwrap();
                }
                ElementValue::Line(line) => {
                    let color = self.colors.get(&handle).unwrap_or(&DEFAULT_ELEMENT_COLOR);

                    write!(
                        &mut self.svg,
                        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{color:0X}" id="line-{}"/>"#,
                        line.p0.x,
                        line.p0.y,
                        line.p1.x,
                        line.p1.y,
                        handle.as_id()
                    ).unwrap();
                }
                ElementValue::Circle(circle) => {
                    let color = self.colors.get(&handle).unwrap_or(&DEFAULT_ELEMENT_COLOR);

                    write!(
                        &mut self.svg,
                        r#"<circle cx="{}" cy="{}" r="{}" stroke="{color:0X}" fill="none" id="circle-{}"/>"#,
                        circle.center.x,
                        circle.center.y,
                        circle.radius,
                        handle.as_id()
                    ).unwrap();
                }
            }
        }
    }

    /// Get the SVG.
    pub fn finish(mut self) -> String {
        write!(&mut self.svg, "</svg>").unwrap();
        self.svg
    }
}
