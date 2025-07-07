// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Constraints between geometric elements.

#[cfg(not(feature = "std"))]
use crate::floatfuncs::FloatFuncs;

use crate::{ConstraintHandle, Edge, ElementHandle, System, Vertex, elements};

trait FloatExt {
    /// Returns the square of `self`.
    ///
    /// Using `std`, you'd be able to do this using `self.powi(2)`, and have this be compiled to a
    /// `self * self`. However, when compiling using `libm`, there is no `powi` and libm's
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
    pub struct ConstraintHandle<T> {
        /// The ID of the system the constraint belongs to.
        pub(crate) system_id: u32,
        /// The ID of the constraint within the system.
        pub(crate) id: u32,
        _t: PhantomData<T>,
    }

    impl<T> ConstraintHandle<T> {
        pub(crate) fn from_ids(system_id: u32, id: u32) -> Self {
            Self {
                system_id,
                id,
                _t: PhantomData,
            }
        }

        pub(crate) fn drop_system_id(self) -> ConstraintId {
            ConstraintId { id: self.id }
        }
    }

    impl<T> core::fmt::Debug for ConstraintHandle<T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let mut s = f.debug_struct("ConstraintHandle");
            s.field("system_id", &self.system_id);
            s.field("id", &self.id);
            s.finish()
        }
    }

    impl<T> Clone for ConstraintHandle<T> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<T> Copy for ConstraintHandle<T> {}

    impl<T> PartialEq for ConstraintHandle<T> {
        fn eq(&self, other: &Self) -> bool {
            self.system_id == other.system_id && self.id == other.id
        }
    }
    impl<T> Eq for ConstraintHandle<T> {}

    impl<T> Ord for ConstraintHandle<T> {
        fn cmp(&self, other: &Self) -> core::cmp::Ordering {
            (self.system_id, self.id).cmp(&(other.system_id, other.id))
        }
    }
    impl<T> PartialOrd for ConstraintHandle<T> {
        fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
            Some(self.cmp(other))
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
    point1_idx: u32,
    point2_idx: u32,

    /// Euclidean distance.
    distance: f64,
}

impl PointPointDistance {
    /// Construct a constraint between two points to have a given straight-line distance between each other.
    ///
    /// `distance` is the Euclidean distance between `point1` and `point2`.
    pub fn create(
        system: &mut System,
        point1: ElementHandle<elements::Point>,
        point2: ElementHandle<elements::Point>,
        distance: f64,
    ) -> ConstraintHandle<Self> {
        let &Vertex::Point { idx: point1_idx } = &system.element_vertices[point1.id as usize]
        else {
            unreachable!()
        };
        let &Vertex::Point { idx: point2_idx } = &system.element_vertices[point2.id as usize]
        else {
            unreachable!()
        };

        let constraint = Self {
            point1_idx,
            point2_idx,
            distance,
        };

        system.add_constraint(Edge::PointPointDistance(constraint))
    }

    pub(crate) fn compute_residual_and_partial_derivatives(
        &self,
        free_variable_map: &alloc::collections::BTreeMap<u32, u32>,
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

        if let Some(idx) = free_variable_map.get(&self.point1_idx) {
            first_derivative[*idx as usize] += derivative[0];
        }
        if let Some(idx) = free_variable_map.get(&(self.point1_idx + 1)) {
            first_derivative[*idx as usize] += derivative[1];
        }
        if let Some(idx) = free_variable_map.get(&self.point2_idx) {
            first_derivative[*idx as usize] += derivative[2];
        }
        if let Some(idx) = free_variable_map.get(&(self.point2_idx + 1)) {
            first_derivative[*idx as usize] += derivative[3];
        }
    }
}

/// Constrain three points to describe a given angle.
pub struct PointPointPointAngle {
    point1_idx: u32,
    point2_idx: u32,
    point3_idx: u32,

    /// Angle in radians.
    angle: f64,
}

impl PointPointPointAngle {
    /// Construct a constraint between three points to describe a given angle.
    ///
    /// `angle` is the angle in radians the points should describe.
    pub fn create(
        system: &mut System,
        point1: ElementHandle<elements::Point>,
        point2: ElementHandle<elements::Point>,
        point3: ElementHandle<elements::Point>,
        angle: f64,
    ) -> ConstraintHandle<Self> {
        let &Vertex::Point { idx: point1_idx } = &system.element_vertices[point1.id as usize]
        else {
            unreachable!()
        };
        let &Vertex::Point { idx: point2_idx } = &system.element_vertices[point2.id as usize]
        else {
            unreachable!()
        };
        let &Vertex::Point { idx: point3_idx } = &system.element_vertices[point3.id as usize]
        else {
            unreachable!()
        };

        let constraint = Self {
            point1_idx,
            point2_idx,
            point3_idx,
            angle,
        };

        system.add_constraint(Edge::PointPointPointAngle(constraint))
    }

    pub(crate) fn compute_residual_and_partial_derivatives(
        &self,
        free_variable_map: &alloc::collections::BTreeMap<u32, u32>,
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

        if let Some(idx) = free_variable_map.get(&self.point1_idx) {
            first_derivative[*idx as usize] += derivative[0];
        }
        if let Some(idx) = free_variable_map.get(&(self.point1_idx + 1)) {
            first_derivative[*idx as usize] += derivative[1];
        }
        if let Some(idx) = free_variable_map.get(&self.point2_idx) {
            first_derivative[*idx as usize] += derivative[2];
        }
        if let Some(idx) = free_variable_map.get(&(self.point2_idx + 1)) {
            first_derivative[*idx as usize] += derivative[3];
        }
        if let Some(idx) = free_variable_map.get(&(self.point3_idx)) {
            first_derivative[*idx as usize] += derivative[4];
        }
        if let Some(idx) = free_variable_map.get(&(self.point3_idx + 1)) {
            first_derivative[*idx as usize] += derivative[5];
        }
    }
}

/// Constrain a point and a line such that the point lies on the (infinite) line.
///
/// Note this does not constrain the point to lie on the line *segment* defined by `line`. This is
/// equivalent to constraining the three points (the two points of the line and the point proper)
/// to be collinear.
pub struct PointLineIncidence {
    point_idx: u32,
    line_point1_idx: u32,
    line_point2_idx: u32,
}

impl PointLineIncidence {
    /// Construct a constraint between a point and a line such that the point lies on the
    /// (infinite) line.
    pub fn create(
        system: &mut System,
        point: ElementHandle<elements::Point>,
        line: ElementHandle<elements::Line>,
    ) -> ConstraintHandle<Self> {
        let &Vertex::Point { idx: point_idx } = &system.element_vertices[point.id as usize] else {
            unreachable!()
        };
        let &Vertex::Line {
            point1_idx: line_point1_idx,
            point2_idx: line_point2_idx,
        } = &system.element_vertices[line.id as usize]
        else {
            unreachable!()
        };

        let constraint = Self {
            point_idx,
            line_point1_idx,
            line_point2_idx,
        };

        system.add_constraint(Edge::PointLineIncidence(constraint))
    }

    pub(crate) fn compute_residual_and_partial_derivatives(
        &self,
        free_variable_map: &alloc::collections::BTreeMap<u32, u32>,
        variables: &[f64],
        residual: &mut f64,
        first_derivative: &mut [f64],
    ) {
        let point1 = kurbo::Point {
            x: variables[self.point_idx as usize],
            y: variables[self.point_idx as usize + 1],
        };
        let point2 = kurbo::Point {
            x: variables[self.line_point1_idx as usize],
            y: variables[self.line_point1_idx as usize + 1],
        };
        let point3 = kurbo::Point {
            x: variables[self.line_point2_idx as usize],
            y: variables[self.line_point2_idx as usize + 1],
        };

        // For collinear points, the triangle defined by those points has area 0.
        *residual += point1.x * (point2.y - point3.y)
            + point2.x * (point3.y - point1.y)
            + point3.x * (point1.y - point2.y);

        let derivative = [
            point2.y - point3.y,
            -point2.x + point3.x,
            point3.y - point1.y,
            point1.x - point3.x,
            point1.y - point2.y,
            -point1.x + point2.x,
        ];

        if let Some(idx) = free_variable_map.get(&self.point_idx) {
            first_derivative[*idx as usize] += derivative[0];
        }
        if let Some(idx) = free_variable_map.get(&(self.point_idx + 1)) {
            first_derivative[*idx as usize] += derivative[1];
        }
        if let Some(idx) = free_variable_map.get(&self.line_point1_idx) {
            first_derivative[*idx as usize] += derivative[2];
        }
        if let Some(idx) = free_variable_map.get(&(self.line_point1_idx + 1)) {
            first_derivative[*idx as usize] += derivative[3];
        }
        if let Some(idx) = free_variable_map.get(&(self.line_point2_idx)) {
            first_derivative[*idx as usize] += derivative[4];
        }
        if let Some(idx) = free_variable_map.get(&(self.line_point2_idx + 1)) {
            first_derivative[*idx as usize] += derivative[5];
        }
    }
}

/// Constrain two lines to describe a given angle.
pub struct LineLineAngle {
    line1_point1_idx: u32,
    line1_point2_idx: u32,
    line2_point1_idx: u32,
    line2_point2_idx: u32,

    /// Angle in radians.
    angle: f64,
}

impl LineLineAngle {
    /// Construct a constraint between two lines to describe a given angle.
    pub fn create(
        system: &mut System,
        line1: ElementHandle<elements::Line>,
        line2: ElementHandle<elements::Line>,
        angle: f64,
    ) -> ConstraintHandle<Self> {
        let &Vertex::Line {
            point1_idx: line1_point1_idx,
            point2_idx: line1_point2_idx,
        } = &system.element_vertices[line1.id as usize]
        else {
            unreachable!()
        };
        let &Vertex::Line {
            point1_idx: line2_point1_idx,
            point2_idx: line2_point2_idx,
        } = &system.element_vertices[line2.id as usize]
        else {
            unreachable!()
        };

        let constraint = Self {
            line1_point1_idx,
            line1_point2_idx,
            line2_point1_idx,
            line2_point2_idx,
            angle,
        };

        system.add_constraint(Edge::LineLineAngle(constraint))
    }

    pub(crate) fn compute_residual_and_partial_derivatives(
        &self,
        free_variable_map: &alloc::collections::BTreeMap<u32, u32>,
        variables: &[f64],
        residual: &mut f64,
        first_derivative: &mut [f64],
    ) {
        let line1_point1 = kurbo::Point {
            x: variables[self.line1_point1_idx as usize],
            y: variables[self.line1_point1_idx as usize + 1],
        };
        let line1_point2 = kurbo::Point {
            x: variables[self.line1_point2_idx as usize],
            y: variables[self.line1_point2_idx as usize + 1],
        };
        let line2_point1 = kurbo::Point {
            x: variables[self.line2_point1_idx as usize],
            y: variables[self.line2_point1_idx as usize + 1],
        };
        let line2_point2 = kurbo::Point {
            x: variables[self.line2_point2_idx as usize],
            y: variables[self.line2_point2_idx as usize + 1],
        };

        let u = line1_point2 - line1_point1;
        let v = line2_point2 - line2_point1;

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

        let dangle_dline1_point1x = -u.y * u_squared_recip;
        let dangle_dline1_point1y = u.x * u_squared_recip;
        let dangle_dline2_point1x = v.y * v_squared_recip;
        let dangle_dline2_point1y = -v.x * v_squared_recip;

        let derivative = [
            dangle_dline1_point1x,
            dangle_dline1_point1y,
            -dangle_dline1_point1x,
            -dangle_dline1_point1y,
            dangle_dline2_point1x,
            dangle_dline2_point1y,
            -dangle_dline2_point1x,
            -dangle_dline2_point1y,
        ];

        if let Some(idx) = free_variable_map.get(&self.line1_point1_idx) {
            first_derivative[*idx as usize] += derivative[0];
        }
        if let Some(idx) = free_variable_map.get(&(self.line1_point1_idx + 1)) {
            first_derivative[*idx as usize] += derivative[1];
        }
        if let Some(idx) = free_variable_map.get(&self.line1_point2_idx) {
            first_derivative[*idx as usize] += derivative[2];
        }
        if let Some(idx) = free_variable_map.get(&(self.line1_point2_idx + 1)) {
            first_derivative[*idx as usize] += derivative[3];
        }
        if let Some(idx) = free_variable_map.get(&(self.line2_point1_idx)) {
            first_derivative[*idx as usize] += derivative[4];
        }
        if let Some(idx) = free_variable_map.get(&(self.line2_point1_idx + 1)) {
            first_derivative[*idx as usize] += derivative[5];
        }
        if let Some(idx) = free_variable_map.get(&(self.line2_point2_idx)) {
            first_derivative[*idx as usize] += derivative[6];
        }
        if let Some(idx) = free_variable_map.get(&(self.line2_point2_idx + 1)) {
            first_derivative[*idx as usize] += derivative[7];
        }
    }
}

/// Constrain a line and a circle such that the line is tangent on the circle.
pub struct LineCircleTangency {
    line_point1_idx: u32,
    line_point2_idx: u32,
    circle_center_idx: u32,
    circle_radius_idx: u32,
}

impl LineCircleTangency {
    /// Construct a constraint between a line and a circle such that the line is tangent on the
    /// circle.
    pub fn create(
        system: &mut System,
        line: ElementHandle<elements::Line>,
        circle: ElementHandle<elements::Circle>,
    ) -> ConstraintHandle<Self> {
        let &Vertex::Line {
            point1_idx: line_point1_idx,
            point2_idx: line_point2_idx,
        } = &system.element_vertices[line.id as usize]
        else {
            unreachable!()
        };
        let &Vertex::Circle {
            center_idx: circle_center_idx,
            radius_idx: circle_radius_idx,
        } = &system.element_vertices[circle.id as usize]
        else {
            unreachable!()
        };

        let constraint = Self {
            line_point1_idx,
            line_point2_idx,
            circle_center_idx,
            circle_radius_idx,
        };

        system.add_constraint(Edge::LineCircleTangency(constraint))
    }

    pub(crate) fn compute_residual_and_partial_derivatives(
        &self,
        free_variable_map: &alloc::collections::BTreeMap<u32, u32>,
        variables: &[f64],
        residual: &mut f64,
        first_derivative: &mut [f64],
    ) {
        let line_point1 = kurbo::Point {
            x: variables[self.line_point1_idx as usize],
            y: variables[self.line_point1_idx as usize + 1],
        };
        let line_point2 = kurbo::Point {
            x: variables[self.line_point2_idx as usize],
            y: variables[self.line_point2_idx as usize + 1],
        };
        let circle_center = kurbo::Point {
            x: variables[self.circle_center_idx as usize],
            y: variables[self.circle_center_idx as usize + 1],
        };
        let circle_radius = variables[self.circle_radius_idx as usize];

        let length2 = line_point1.distance_squared(line_point2);
        let length = length2.sqrt();

        // TODO: better handle degenerate lines of length 0.
        if length == 0. {
            return;
        }

        let length_recip = 1. / length;
        let signed_area = line_point1.x * (line_point2.y - circle_center.y)
            + line_point2.x * (circle_center.y - line_point1.y)
            + circle_center.x * (line_point1.y - line_point2.y);

        // We are interested in the _unsigned_ area here, as it does not matter on which side of
        // the line the circle center lies. That does mean there is a cusp when the circle
        // center is exactly on the line.
        *residual += length_recip * signed_area.abs() - circle_radius;

        let sign = signed_area.signum();
        let length3_recip = 1. / (length2 * length);
        let derivative = [
            sign * length3_recip
                * (length2 * (line_point2.y - circle_center.y)
                    + signed_area * (line_point2.x - line_point1.x)),
            sign * length3_recip
                * (length2 * (-line_point2.x + circle_center.x)
                    + signed_area * (line_point2.y - line_point1.y)),
            sign * length3_recip
                * (length2 * (circle_center.y - line_point1.y)
                    - signed_area * (line_point2.x - line_point1.x)),
            sign * length3_recip
                * (length2 * (line_point1.x - circle_center.x)
                    - signed_area * (line_point2.y - line_point1.y)),
            sign * length_recip * (line_point1.y - line_point2.y),
            sign * length_recip * (-line_point1.x + line_point2.x),
        ];

        if let Some(idx) = free_variable_map.get(&self.line_point1_idx) {
            first_derivative[*idx as usize] += derivative[0];
        }
        if let Some(idx) = free_variable_map.get(&(self.line_point1_idx + 1)) {
            first_derivative[*idx as usize] += derivative[1];
        }
        if let Some(idx) = free_variable_map.get(&self.line_point2_idx) {
            first_derivative[*idx as usize] += derivative[2];
        }
        if let Some(idx) = free_variable_map.get(&(self.line_point2_idx + 1)) {
            first_derivative[*idx as usize] += derivative[3];
        }
        if let Some(idx) = free_variable_map.get(&self.circle_center_idx) {
            first_derivative[*idx as usize] += derivative[4];
        }
        if let Some(idx) = free_variable_map.get(&(self.circle_center_idx + 1)) {
            first_derivative[*idx as usize] += derivative[5];
        }
        if let Some(idx) = free_variable_map.get(&self.circle_radius_idx) {
            first_derivative[*idx as usize] += -1.;
        }
    }
}

/// A constraint between geometric [elements](crate::Element).
///
/// These can be added to a [`System`].
pub trait Constraint {}
impl Constraint for PointPointDistance {}
impl Constraint for PointPointPointAngle {}
impl Constraint for PointLineIncidence {}
impl Constraint for LineCircleTangency {}
impl Constraint for LineLineAngle {}
