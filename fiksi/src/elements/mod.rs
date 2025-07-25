// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Geometric elements like points and lines.

pub(crate) mod element {
    use core::marker::PhantomData;

    use crate::{ElementValue, System};

    use super::{Element, ElementTag, sealed::ElementInner};

    /// Dynamically tagged, typed handles to elements.
    pub enum TaggedElementHandle {
        /// A handle to a [`Point`](super::Point) element.
        Point(ElementHandle<super::Point>),
        /// A handle to a [`Line`](super::Line) element.
        Line(ElementHandle<super::Line>),
        /// A handle to a [`Circle`](super::Circle) element.
        Circle(ElementHandle<super::Circle>),
    }

    /// A handle to an element within a [`System`].
    pub struct ElementHandle<T: Element> {
        /// The ID of the system the element belongs to.
        pub(crate) system_id: u32,
        /// The ID of the element within the system.
        pub(crate) id: u32,
        _t: PhantomData<T>,
    }

    impl<T: Element> ElementHandle<T> {
        pub(crate) fn from_ids(system_id: u32, id: u32) -> Self {
            Self {
                system_id,
                id,
                _t: PhantomData,
            }
        }

        pub(crate) fn drop_system_id(self) -> ElementId {
            ElementId { id: self.id }
        }

        /// Get the value of the element.
        pub fn get_value(&self, system: &System) -> <T as Element>::Output {
            // TODO: return `Result` instead of panicking?
            assert_eq!(
                self.system_id, system.id,
                "Tried to get an element that is not part of this `System`"
            );

            <T as ElementInner>::from_vertex(
                &system.element_vertices[self.drop_system_id().id as usize],
                &system.variables,
            )
            .into()
        }

        /// Get a type-erased handle to the element.
        ///
        /// To turn the returned handle back into a typed handle, use
        /// [`AnyElementHandle::as_tagged_element`].
        pub fn as_any_element(self) -> AnyElementHandle {
            AnyElementHandle {
                system_id: self.system_id,
                id: self.id,
                tag: T::tag(),
            }
        }
    }

    /// A type-erased handle to an element within a [`System`].
    #[derive(Copy, Clone, Debug)]
    pub struct AnyElementHandle {
        /// The ID of the system the element belongs to.
        pub(crate) system_id: u32,
        /// The ID of the element within the system.
        pub(crate) id: u32,
        tag: ElementTag,
    }

    impl AnyElementHandle {
        pub(crate) fn from_ids_and_tag(system_id: u32, id: u32, tag: ElementTag) -> Self {
            Self { system_id, id, tag }
        }

        /// Get the value of the element.
        pub fn get_value(&self, system: &System) -> ElementValue {
            // TODO: return `Result` instead of panicking?
            assert_eq!(
                self.system_id, system.id,
                "Tried to get an element that is not part of this `System`"
            );

            let vertex = &system.element_vertices[self.id as usize];
            match self.tag {
                ElementTag::Point => {
                    ElementValue::Point(super::Point::from_vertex(vertex, &system.variables))
                }
                ElementTag::Line => {
                    ElementValue::Line(super::Line::from_vertex(vertex, &system.variables))
                }
                ElementTag::Circle => {
                    ElementValue::Circle(super::Circle::from_vertex(vertex, &system.variables))
                }
            }
        }

        /// Get a typed handle to the element.
        pub fn as_tagged_element(self) -> TaggedElementHandle {
            match self.tag {
                ElementTag::Point => {
                    TaggedElementHandle::Point(ElementHandle::from_ids(self.system_id, self.id))
                }
                ElementTag::Line => {
                    TaggedElementHandle::Line(ElementHandle::from_ids(self.system_id, self.id))
                }
                ElementTag::Circle => {
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

    impl<T: Element> core::hash::Hash for ElementHandle<T> {
        fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
            self.system_id.hash(state);
            self.id.hash(state);
        }
    }

    impl PartialEq for AnyElementHandle {
        fn eq(&self, other: &Self) -> bool {
            // Element handle IDs are unique, so we don't need to compare the tags.
            self.system_id == other.system_id && self.id == other.id
        }
    }
    impl Eq for AnyElementHandle {}

    impl Ord for AnyElementHandle {
        fn cmp(&self, other: &Self) -> core::cmp::Ordering {
            // Element handle IDs are unique, so we don't need to compare the tags.
            (self.system_id, self.id).cmp(&(other.system_id, other.id))
        }
    }
    impl PartialOrd for AnyElementHandle {
        fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl core::hash::Hash for AnyElementHandle {
        fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
            // Element handle IDs are unique, so we don't need to hash the tags.
            self.system_id.hash(state);
            self.id.hash(state);
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

use crate::{System, Vertex};

/// A point given by a 2D coordinate.
#[derive(Debug)]
pub struct Point {
    /// The x-coordinate of the point.
    x: f64,
    /// The y-coordinate of the point.
    y: f64,
}

impl Point {
    /// Construct a new `Point` at the given coordinate.
    pub fn create(system: &mut System, x: f64, y: f64) -> ElementHandle<Self> {
        system.add_element([x, y], |variables_idx| Vertex::Point { idx: variables_idx })
    }
}

impl sealed::ElementInner for Point {
    type Output = kurbo::Point;
    type HandleData = ();

    fn tag() -> ElementTag {
        ElementTag::Point
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
    point1: ElementHandle<Point>,
    /// Second point of the line.
    point2: ElementHandle<Point>,
}

impl Line {
    /// Construct a new `Line` with the given points.
    pub fn create(
        system: &mut System,
        point1: ElementHandle<Point>,
        point2: ElementHandle<Point>,
    ) -> ElementHandle<Self> {
        let &Vertex::Point { idx: point1_idx } = &system.element_vertices[point1.id as usize]
        else {
            unreachable!()
        };
        let &Vertex::Point { idx: point2_idx } = &system.element_vertices[point2.id as usize]
        else {
            unreachable!()
        };
        system.add_element([], |_| Vertex::Line {
            point1_idx,
            point2_idx,
        })
    }
}

impl sealed::ElementInner for Line {
    type Output = kurbo::Line;
    type HandleData = ();

    fn tag() -> ElementTag {
        ElementTag::Line
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
    center: ElementHandle<Point>,

    /// The radius of the circle.
    radius: f64,
}

impl Circle {
    /// Construct a new `Circle` with the given point and radius.
    pub fn create(
        system: &mut System,
        center: ElementHandle<Point>,
        radius: f64,
    ) -> ElementHandle<Self> {
        let &Vertex::Point { idx: center_idx } = &system.element_vertices[center.id as usize]
        else {
            unreachable!()
        };
        system.add_element([radius], |radius_idx| Vertex::Circle {
            center_idx,
            radius_idx,
        })
    }
}

impl sealed::ElementInner for Circle {
    type Output = kurbo::Circle;
    type HandleData = ();

    fn tag() -> ElementTag {
        ElementTag::Circle
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

/// The actual type of the element.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ElementTag {
    Point,
    Line,
    Circle,
}

impl<'a> From<&'a Vertex> for ElementTag {
    fn from(vertex: &'a Vertex) -> Self {
        match vertex {
            Vertex::Point { .. } => Self::Point,
            Vertex::Line { .. } => Self::Line,
            Vertex::Circle { .. } => Self::Circle,
        }
    }
}

pub(crate) mod sealed {
    use crate::Vertex;

    pub(crate) trait ElementInner {
        /// The data type when retrieving an element's value.
        type Output;

        /// Additional data stored in element handles of this element type.
        type HandleData: Copy + core::fmt::Debug + Default;

        fn tag() -> super::ElementTag;
        fn from_vertex(vertex: &Vertex, variables: &[f64]) -> Self::Output;
    }
}

/// A geometric element that can be [constrained](crate::Constraint).
///
/// These can be added to a [`System`].
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
