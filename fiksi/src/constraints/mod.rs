// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Constraints between geometric elements.

#[cfg(feature = "libm")]
use crate::floatfuncs::FloatFuncs;

use crate::Edge;
use crate::Vertex;
use crate::elements;
use crate::{ElementHandle, ElementId};

trait FloatExt {
    /// Returns the square of `self`.
    ///
    /// Using `std`, you'd be able to do this using `self.powi(2)`, and have this be compiled to a
    /// single multiply op. However, when compiling using `libm`, there is no `powi` and libm's
    /// `self.powf(2.)` doesn't compile away.
    fn square(self) -> Self;
}

impl FloatExt for f64 {
    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }
}

pub(crate) mod constraint {
    use core::marker::PhantomData;

    /// A handle to a constraint within a [`System`](crate::System).
    #[derive(Debug)]
    pub struct ConstraintHandle<T> {
        /// The ID of the system the constraint belongs to.
        system_id: u32,
        /// The ID of the constraint within the system.
        id: u32,
        _t: PhantomData<T>,
    }

    impl<T> ConstraintHandle<T> {
        pub(crate) fn from_ids(system_id: u32, id: u32) -> Self {
            Self {
                system_id,
                id,
                _t: PhantomData::default(),
            }
        }

        pub(crate) fn drop_system_id(&self) -> ConstraintId {
            ConstraintId { id: self.id }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub(crate) struct ConstraintId {
        /// The ID of the constraint within the system.
        pub(crate) id: u32,
    }
}

/// Constrain two points to have a given straight-line distance between each other.
pub struct PointPointDistance {
    point1: ElementId,
    point2: ElementId,

    /// Euclidean distance.
    distance: f64,
}

impl sealed::ConstraintInner for PointPointDistance {
    fn as_edge(&self, vertices: &[Vertex]) -> Edge {
        let &Vertex::Point { idx: point1_idx } = &vertices[self.point1.id as usize] else {
            unreachable!()
        };
        let &Vertex::Point { idx: point2_idx } = &vertices[self.point2.id as usize] else {
            unreachable!()
        };
        Edge::PointPointDistance {
            point1_idx,
            point2_idx,
            distance: self.distance,
        }
    }
}

impl PointPointDistance {
    /// Construct a constraint between two points to have a given straight-line distance between each other.
    ///
    /// `distance` is the Euclidean distance between `point1` and `point2`.
    pub fn new(
        point1: &ElementHandle<elements::Point>,
        point2: &ElementHandle<elements::Point>,
        distance: f64,
    ) -> Self {
        Self {
            point1: point1.drop_system_id(),
            point2: point2.drop_system_id(),
            distance,
        }
    }
}

/// A representation of a [`PointPointDistance`] constraint within a [`crate::System`], allowing
/// evaluation.
///
/// TODO: can this be merged with [`PointPointDistance`]?
pub(crate) struct PointPointDistance_ {
    pub point1_idx: u32,
    pub point2_idx: u32,

    /// Euclidean distance.
    pub distance: f64,
}

impl PointPointDistance_ {
    pub(crate) fn compute_residual_and_partial_derivatives(
        &self,
        index_map: &alloc::collections::BTreeMap<u32, u32>,
        variables: &[f64],
        residual: &mut f64,
        first_derivative: &mut [f64],
    ) {
        let point1 = kurbo::Point {
            x: variables[self.point1_idx as usize],
            y: variables[self.point1_idx as usize + 1],
        };
        let point2 = kurbo::Point {
            x: variables[self.point2_idx as usize],
            y: variables[self.point2_idx as usize + 1],
        };

        let distance = ((point1.x - point2.x).square() + (point1.y - point2.y).square()).sqrt();
        *residual += distance - self.distance;

        let distance_recip = 1. / distance;
        let derivative = [
            (point1.x - point2.x) * distance_recip,
            (point1.y - point2.y) * distance_recip,
            -(point1.x - point2.x) * distance_recip,
            -(point1.y - point2.y) * distance_recip,
        ];

        if let Some(idx) = index_map.get(&self.point1_idx) {
            first_derivative[*idx as usize] += derivative[0];
        }
        if let Some(idx) = index_map.get(&(self.point1_idx + 1)) {
            first_derivative[*idx as usize] += derivative[1];
        }
        if let Some(idx) = index_map.get(&self.point2_idx) {
            first_derivative[*idx as usize] += derivative[2];
        }
        if let Some(idx) = index_map.get(&(self.point2_idx + 1)) {
            first_derivative[*idx as usize] += derivative[3];
        }
    }
}

/// Constrain three points to describe a given angle.
pub struct PointPointPointAngle {
    point1: ElementId,
    point2: ElementId,
    point3: ElementId,

    /// Angle in radians.
    angle: f64,
}

impl sealed::ConstraintInner for PointPointPointAngle {
    fn as_edge(&self, vertices: &[Vertex]) -> Edge {
        let &Vertex::Point { idx: point1_idx } = &vertices[self.point1.id as usize] else {
            unreachable!()
        };
        let &Vertex::Point { idx: point2_idx } = &vertices[self.point2.id as usize] else {
            unreachable!()
        };
        let &Vertex::Point { idx: point3_idx } = &vertices[self.point3.id as usize] else {
            unreachable!()
        };
        Edge::PointPointPointAngle {
            point1_idx,
            point2_idx,
            point3_idx,
            angle: self.angle,
        }
    }
}

impl PointPointPointAngle {
    /// Construct a constraint between three points to describe ag iven angle.
    ///
    /// `angle` is the angle in radians the points should describe.
    pub fn new(
        point1: &ElementHandle<elements::Point>,
        point2: &ElementHandle<elements::Point>,
        point3: &ElementHandle<elements::Point>,
        angle: f64,
    ) -> Self {
        Self {
            point1: point1.drop_system_id(),
            point2: point2.drop_system_id(),
            point3: point3.drop_system_id(),
            angle,
        }
    }
}

/// A representation of a [`PointPointPointAngle`] constraint within a [`crate::System`], allowing
/// evaluation.
///
/// TODO: can this be merged with [`PointPointPointAngle`]?
pub(crate) struct PointPointPointAngle_ {
    pub point1_idx: u32,
    pub point2_idx: u32,
    pub point3_idx: u32,

    /// Angle in radians.
    pub angle: f64,
}

impl PointPointPointAngle_ {
    pub(crate) fn compute_residual_and_partial_derivatives(
        &self,
        index_map: &alloc::collections::BTreeMap<u32, u32>,
        variables: &[f64],
        residual: &mut f64,
        first_derivative: &mut [f64],
    ) {
        let point1 = kurbo::Point {
            x: variables[self.point1_idx as usize],
            y: variables[self.point1_idx as usize + 1],
        };
        let point2 = kurbo::Point {
            x: variables[self.point2_idx as usize],
            y: variables[self.point2_idx as usize + 1],
        };
        let point3 = kurbo::Point {
            x: variables[self.point3_idx as usize],
            y: variables[self.point3_idx as usize + 1],
        };

        let u = point1 - point2;
        let v = point3 - point2;

        let angle = v.atan2() - u.atan2();
        let angle = if angle > core::f64::consts::PI {
            angle - 2.0 * core::f64::consts::PI
        } else if angle < -core::f64::consts::PI {
            angle + 2.0 * core::f64::consts::PI
        } else {
            angle
        };

        *residual += angle - self.angle;

        let u_squared_recip = u.length_squared().recip();
        let v_squared_recip = v.length_squared().recip();

        let dangle_dpoint1x = u.y * u_squared_recip;
        let dangle_dpoint1y = -u.x * u_squared_recip;
        let dangle_dpoint3x = -v.y * v_squared_recip;
        let dangle_dpoint3y = v.x * v_squared_recip;

        let dangle_dpoint2x = -dangle_dpoint1x - dangle_dpoint3x;
        let dangle_dpoint2y = -dangle_dpoint1y - dangle_dpoint3y;

        let derivative = [
            dangle_dpoint1x,
            dangle_dpoint1y,
            dangle_dpoint2x,
            dangle_dpoint2y,
            dangle_dpoint3x,
            dangle_dpoint3y,
        ];

        if let Some(idx) = index_map.get(&self.point1_idx) {
            first_derivative[*idx as usize] += derivative[0];
        }
        if let Some(idx) = index_map.get(&(self.point1_idx + 1)) {
            first_derivative[*idx as usize] += derivative[1];
        }
        if let Some(idx) = index_map.get(&self.point2_idx) {
            first_derivative[*idx as usize] += derivative[2];
        }
        if let Some(idx) = index_map.get(&(self.point2_idx + 1)) {
            first_derivative[*idx as usize] += derivative[3];
        }
        if let Some(idx) = index_map.get(&(self.point3_idx)) {
            first_derivative[*idx as usize] += derivative[4];
        }
        if let Some(idx) = index_map.get(&(self.point3_idx + 1)) {
            first_derivative[*idx as usize] += derivative[5];
        }
    }
}

/// Constrain two lines to describe a given angle.
///
/// TODO: actually implement this, or require using [`PointPointPointAngle`]?
pub struct LineLineAngle {
    line1: ElementHandle<elements::Line>,
    line2: ElementHandle<elements::Line>,

    /// Angle in radians.
    angle: f64,
}

impl sealed::ConstraintInner for LineLineAngle {
    fn as_edge(&self, vertices: &[Vertex]) -> Edge {
        let &Vertex::Line {
            point1_idx,
            point2_idx,
        } = &vertices[self.line1.id as usize]
        else {
            unreachable!()
        };
        let &Vertex::Line {
            point1_idx: point3_idx,
            point2_idx: point4_idx,
        } = &vertices[self.line2.id as usize]
        else {
            unreachable!()
        };
        Edge::LineLineAngle {
            point1_idx,
            point2_idx,
            point3_idx,
            point4_idx,
            angle: self.angle,
        }
    }
}

/// A representation of a [`LineLineAngle`] constraint within a [`crate::System`], allowing
/// evaluation.
///
/// TODO: implement
/// TODO: can this be merged with [`LineLineAngle`]?
pub(crate) struct LineLineAngle_ {
    pub point1_idx: u32,
    pub point2_idx: u32,
    pub point3_idx: u32,
    pub point4_idx: u32,

    /// Angle in radians.
    pub angle: f64,
}

pub(crate) mod sealed {
    pub(crate) trait ConstraintInner {
        fn as_edge(&self, vertices: &[crate::Vertex]) -> crate::Edge;
    }
}

/// A constraint between geometric [elements](crate::Element).
///
/// These can be added to a [`System`](crate::System).
#[expect(private_bounds, reason = "Sealed inner trait")]
pub trait Constraint: sealed::ConstraintInner {}
impl Constraint for PointPointDistance {}
impl Constraint for PointPointPointAngle {}
impl Constraint for LineLineAngle {}
