// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Geometric elements like points and lines.

use alloc::vec::Vec;

pub(crate) mod element {
    use core::marker::PhantomData;

    /// A handle to an element within a [`System`](crate::System).
    #[derive(Debug)]
    pub struct ElementHandle<T> {
        /// The ID of the system the element belongs to.
        pub(crate) system_id: u32,
        /// The ID of the element within the system.
        pub(crate) id: u32,
        _t: PhantomData<T>,
    }

    impl<T> ElementHandle<T> {
        pub(crate) fn from_ids(system_id: u32, id: u32) -> Self {
            Self {
                system_id,
                id,
                _t: PhantomData,
            }
        }

        pub(crate) fn drop_system_id(&self) -> ElementId {
            ElementId { id: self.id }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub(crate) struct ElementId {
        /// The ID of the element within the system.
        pub(crate) id: u32,
    }

    impl ElementId {
        pub(crate) fn from_id(id: u32) -> Self {
            Self { id }
        }
    }
}

use element::ElementHandle;

use crate::Vertex;

/// A point given by a 2D coordinate.
#[derive(Debug)]
pub struct Point {
    /// The x-coordinate of the point.
    pub x: f64,
    /// The y-coordinate of the point.
    pub y: f64,
}

impl sealed::ElementInner for Point {
    type Output = kurbo::Point;

    fn add_into(&self, element_vertices: &mut Vec<Vertex>, variables: &mut Vec<f64>) {
        element_vertices.push(Vertex::Point {
            idx: variables
                .len()
                .try_into()
                .expect("less than 2^32 variables"),
        });
        variables.extend(&[self.x, self.y]);
    }

    fn from_vertex(vertex: &Vertex, variables: &[f64]) -> Self::Output {
        let &Vertex::Point { idx } = vertex else {
            unreachable!()
        };
        kurbo::Point {
            x: variables[idx as usize],
            y: variables[idx as usize + 1],
        }
    }
}

/// A line defined by two endpoints.
#[derive(Debug)]
pub struct Line<'el> {
    /// First point of the line.
    pub point1: &'el ElementHandle<Point>,
    /// Second point of the line.
    pub point2: &'el ElementHandle<Point>,
}

impl<'el> sealed::ElementInner for Line<'el> {
    type Output = kurbo::Line;

    fn add_into(&self, element_vertices: &mut Vec<Vertex>, _variables: &mut Vec<f64>) {
        let &Vertex::Point { idx: point1_idx } = &element_vertices[self.point1.id as usize] else {
            unreachable!()
        };
        let &Vertex::Point { idx: point2_idx } = &element_vertices[self.point2.id as usize] else {
            unreachable!()
        };
        element_vertices.push(Vertex::Line {
            point1_idx,
            point2_idx,
        });
    }

    fn from_vertex(vertex: &Vertex, variables: &[f64]) -> Self::Output {
        let &Vertex::Line {
            point1_idx,
            point2_idx,
        } = vertex
        else {
            unreachable!()
        };
        kurbo::Line {
            p0: kurbo::Point {
                x: variables[point1_idx as usize],
                y: variables[point1_idx as usize + 1],
            },
            p1: kurbo::Point {
                x: variables[point2_idx as usize],
                y: variables[point2_idx as usize + 1],
            },
        }
    }
}

/// A circle defined by a centerpoint and a radius.
#[derive(Debug)]
pub struct Circle<'el> {
    /// The center of the circle.
    pub center: &'el ElementHandle<Point>,

    /// The radius of the circle.
    pub radius: f64,
}

impl sealed::ElementInner for Circle<'_> {
    type Output = kurbo::Circle;

    fn add_into(&self, element_vertices: &mut Vec<Vertex>, variables: &mut Vec<f64>) {
        let &Vertex::Point { idx: center_idx } = &element_vertices[self.center.id as usize] else {
            unreachable!()
        };
        element_vertices.push(Vertex::Circle {
            center_idx,
            radius_idx: variables
                .len()
                .try_into()
                .expect("less than 2^32 variables"),
        });
        variables.extend(&[self.radius]);
    }

    fn from_vertex(vertex: &Vertex, variables: &[f64]) -> kurbo::Circle {
        let &Vertex::Circle {
            center_idx,
            radius_idx,
        } = vertex
        else {
            unreachable!()
        };
        kurbo::Circle {
            center: kurbo::Point {
                x: variables[center_idx as usize],
                y: variables[center_idx as usize + 1],
            },
            radius: variables[radius_idx as usize],
        }
    }
}

pub(crate) mod sealed {
    use alloc::vec::Vec;

    use crate::Vertex;

    pub(crate) trait ElementInner {
        /// The data type when retrieving an elements value.
        type Output;

        fn add_into(&self, element_vertices: &mut Vec<Vertex>, variables: &mut Vec<f64>);
        fn from_vertex(vertex: &Vertex, variables: &[f64]) -> Self::Output;
    }
}

/// A geometric element that can be [constrained](crate::Constraint).
///
/// These can be added to a [`System`](crate::System).
#[expect(private_bounds, reason = "Sealed inner trait")]
pub trait Element: sealed::ElementInner {
    /// The data type when retrieving an elements value.
    type Output: From<<Self as sealed::ElementInner>::Output>;
}

impl Element for Point {
    type Output = kurbo::Point;
}
impl<'el> Element for Line<'el> {
    // type Output = <Self as sealed::ElementInner>::Output;
    type Output = kurbo::Line;
}
impl<'el> Element for Circle<'el> {
    type Output = kurbo::Circle;
}
