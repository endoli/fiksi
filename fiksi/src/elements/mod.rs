// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Geometric elements like points and lines.

use alloc::vec::Vec;

pub(crate) mod element {
    use super::{Element, sealed::ElementInner};

    /// Dynamically tagged, typed handles to elements.
    pub enum TaggedElementHandle {
        /// A handle to a [`Point`](super::Point) element.
        Point(ElementHandle<super::Point>),
        /// A handle to a [`Line`](super::Line) element.
        Line(ElementHandle<super::Line>),
        /// A handle to a [`Circle`](super::Circle) element.
        Circle(ElementHandle<super::Circle>),
    }

    /// A handle to an element within a [`System`](crate::System).
    pub struct ElementHandle<T: Element> {
        /// The ID of the system the element belongs to.
        pub(crate) system_id: u32,
        /// The ID of the element within the system.
        pub(crate) id: u32,
        /// Additional data stored in the handle.
        ///
        /// Currently only used by [`super::AnyElement`] to store the tag.
        pub(crate) data: <T as ElementInner>::HandleData,
    }

    impl<T: Element> ElementHandle<T> {
        pub(crate) fn from_ids(system_id: u32, id: u32) -> Self {
            Self {
                system_id,
                id,
                data: Default::default(),
            }
        }

        pub(crate) fn drop_system_id(self) -> ElementId {
            ElementId { id: self.id }
        }

        /// Get a type-erased handle to the element.
        ///
        /// To turn the returned handle back into a typed handle, use
        /// [`ElementHandle<AnyElement>::as_tagged_element`](ElementHandle<super::AnyElement>::as_tagged_element).
        pub fn as_any_element(self) -> ElementHandle<super::AnyElement> {
            ElementHandle::from_ids(self.system_id, self.id)
        }
    }

    impl ElementHandle<super::AnyElement> {
        /// Get a typed handle to the element.
        pub fn as_tagged_element(self) -> TaggedElementHandle {
            use super::AnyElementTag;

            match self.data {
                AnyElementTag::Point => {
                    TaggedElementHandle::Point(ElementHandle::from_ids(self.system_id, self.id))
                }
                AnyElementTag::Line => {
                    TaggedElementHandle::Line(ElementHandle::from_ids(self.system_id, self.id))
                }
                AnyElementTag::Circle => {
                    TaggedElementHandle::Circle(ElementHandle::from_ids(self.system_id, self.id))
                }
            }
        }
    }

    impl<T: Element> core::fmt::Debug for ElementHandle<T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let mut s = f.debug_struct("ElementHandle");
            s.field("system_id", &self.system_id);
            s.field("id", &self.id);
            s.field("debug", &self.data);
            s.finish()
        }
    }

    impl<T: Element> Clone for ElementHandle<T> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<T: Element> Copy for ElementHandle<T> {}

    impl<T: Element> PartialEq for ElementHandle<T> {
        fn eq(&self, other: &Self) -> bool {
            self.system_id == other.system_id && self.id == other.id
        }
    }
    impl<T: Element> Eq for ElementHandle<T> {}

    impl<T: Element> Ord for ElementHandle<T> {
        fn cmp(&self, other: &Self) -> core::cmp::Ordering {
            (self.system_id, self.id).cmp(&(other.system_id, other.id))
        }
    }
    impl<T: Element> PartialOrd for ElementHandle<T> {
        fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
            Some(self.cmp(other))
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

use crate::{ElementValue, Vertex};

/// A point given by a 2D coordinate.
#[derive(Debug)]
pub struct Point {
    /// The x-coordinate of the point.
    pub x: f64,
    /// The y-coordinate of the point.
    pub y: f64,
}

impl Point {
    /// Construct a new `Point` at the given coordinate.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl sealed::ElementInner for Point {
    type Output = kurbo::Point;
    type HandleData = ();

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
pub struct Line {
    /// First point of the line.
    pub point1: ElementHandle<Point>,
    /// Second point of the line.
    pub point2: ElementHandle<Point>,
}

impl Line {
    /// Construct a new `Line` with the given points.
    pub fn new(point1: ElementHandle<Point>, point2: ElementHandle<Point>) -> Self {
        Self { point1, point2 }
    }
}

impl sealed::ElementInner for Line {
    type Output = kurbo::Line;
    type HandleData = ();

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
pub struct Circle {
    /// The center of the circle.
    pub center: ElementHandle<Point>,

    /// The radius of the circle.
    pub radius: f64,
}

impl Circle {
    /// Construct a new `Circle` with the given point and radius.
    pub fn new(center: ElementHandle<Point>, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl sealed::ElementInner for Circle {
    type Output = kurbo::Circle;
    type HandleData = ();

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
        /// The data type when retrieving an element's value.
        type Output;

        /// Additional data stored in element handles of this element type.
        type HandleData: Copy + core::fmt::Debug + Default;

        fn add_into(&self, element_vertices: &mut Vec<Vertex>, variables: &mut Vec<f64>);
        fn from_vertex(vertex: &Vertex, variables: &[f64]) -> Self::Output;
    }
}

/// A type-erased element.
///
/// See [`ElementHandle::as_any_element`].
pub struct AnyElement {}

/// The actual type of the element.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum AnyElementTag {
    #[default]
    Point,
    Line,
    Circle,
}

impl sealed::ElementInner for AnyElement {
    type Output = ElementValue;
    type HandleData = AnyElementTag;

    fn add_into(&self, _element_vertices: &mut Vec<Vertex>, _variables: &mut Vec<f64>) {
        // This element can't be constructed manually, it's used for type-erased element handles.
        unreachable!()
    }
    fn from_vertex(vertex: &Vertex, variables: &[f64]) -> Self::Output {
        match vertex {
            Vertex::Point { .. } => ElementValue::Point(Point::from_vertex(vertex, variables)),
            Vertex::Line { .. } => ElementValue::Line(Line::from_vertex(vertex, variables)),
            Vertex::Circle { .. } => ElementValue::Circle(Circle::from_vertex(vertex, variables)),
        }
    }
}

/// A geometric element that can be [constrained](crate::Constraint).
///
/// These can be added to a [`System`](crate::System).
#[expect(private_bounds, reason = "Sealed inner trait")]
pub trait Element: sealed::ElementInner {
    /// The data type when retrieving an element's value.
    type Output: From<<Self as sealed::ElementInner>::Output>;
}

impl Element for Point {
    type Output = kurbo::Point;
}
impl Element for Line {
    type Output = kurbo::Line;
}
impl Element for Circle {
    type Output = kurbo::Circle;
}
impl Element for AnyElement {
    type Output = ElementValue;
}
