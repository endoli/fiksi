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
    fn add_into(&self, element_vertices: &mut Vec<Vertex>, variables: &mut Vec<f64>) {
        element_vertices.push(Vertex::Point {
            idx: variables
                .len()
                .try_into()
                .expect("less than 2^32 variables"),
        });
        variables.extend(&[self.x, self.y]);
    }

    fn from_vertex(vertex: &Vertex, variables: &[f64]) -> Self {
        if let &Vertex::Point { idx } = vertex {
            Self {
                x: variables[idx as usize],
                y: variables[idx as usize + 1],
            }
        } else {
            unreachable!()
        }
    }
}

/// A line defined by two endpoints.
#[derive(Debug)]
pub struct Line {
    /// First point of the line.
    pub point1: ElementHandle<Point>,
    /// Second point of the line.
    pub point2: ElementHandle<Point>,
}

impl sealed::ElementInner for Line {
    fn add_into(&self, vertices: &mut Vec<Vertex>, _data: &mut Vec<f64>) {
        let &Vertex::Point { idx: point1_idx } = &vertices[self.point1.id as usize] else {
            unreachable!()
        };
        let &Vertex::Point { idx: point2_idx } = &vertices[self.point2.id as usize] else {
            unreachable!()
        };
        vertices.push(Vertex::Line {
            point1_idx,
            point2_idx,
        });
    }

    fn from_vertex(_vertex: &Vertex, _variables: &[f64]) -> Self {
        unimplemented!()
    }
}

/// A circle defined by a centerpoint and a radius.
///
/// TODO: use.
#[derive(Debug)]
pub struct Circle {
    /// The center of the circle.
    pub center: ElementHandle<Point>,

    /// The radius of the circle.
    pub radius: f64,
}

pub(crate) mod sealed {
    use alloc::vec::Vec;

    use crate::Vertex;

    pub(crate) trait ElementInner {
        fn add_into(&self, element_vertices: &mut Vec<Vertex>, variables: &mut Vec<f64>);
        fn from_vertex(vertex: &Vertex, variables: &[f64]) -> Self;
    }
}

/// A geometric element that can be [constrained](crate::Constraint).
///
/// These can be added to a [`System`](crate::System).
#[expect(private_bounds, reason = "Sealed inner trait")]
pub trait Element: sealed::ElementInner {}

impl Element for Point {}
impl Element for Line {}
