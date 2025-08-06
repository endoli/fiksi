// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Constraints between geometric elements.

#[cfg(not(feature = "std"))]
use crate::floatfuncs::FloatFuncs;

use crate::{
    ConstraintHandle, Edge, ElementHandle, Subsystem, System, Vertex, elements,
    graph::IncidentElements,
};

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

    use crate::{System, utils};

    use super::{Constraint, ConstraintTag};

    /// Dynamically tagged, typed handles to constraints.
    pub enum TaggedConstraintHandle {
        /// A handle to a [`PointPointDistance`](super::PointPointDistance) constraint.
        PointPointDistance(ConstraintHandle<super::PointPointDistance>),

        /// A handle to a [`PointPointPointAngle`](super::PointPointPointAngle) constraint.
        PointPointPointAngle(ConstraintHandle<super::PointPointPointAngle>),

        /// A handle to a [`PointLineIncidence`](super::PointLineIncidence) constraint.
        PointLineIncidence(ConstraintHandle<super::PointLineIncidence>),

        /// A handle to a [`LineLineAngle`](super::LineLineAngle) constraint.
        LineLineAngle(ConstraintHandle<super::LineLineAngle>),

        /// A handle to a [`LineLineParallelism`](super::LineLineParallelism) constraint.
        LineLineParallelism(ConstraintHandle<super::LineLineParallelism>),

        /// A handle to a [`LineCircleTangency`](super::LineCircleTangency) constraint.
        LineCircleTangency(ConstraintHandle<super::LineCircleTangency>),
    }

    /// A handle to a constraint within a [`System`].
    pub struct ConstraintHandle<T> {
        /// The ID of the system the constraint belongs to.
        pub(crate) system_id: u32,
        /// The ID of the constraint within the system.
        pub(crate) id: u32,
        _t: PhantomData<T>,
    }

    impl<T: Constraint> ConstraintHandle<T> {
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

        /// Calculate the residual of this constraint.
        pub fn calculate_residual(self, system: &System) -> f64 {
            // TODO: return `Result` instead of panicking?
            assert_eq!(
                self.system_id, system.id,
                "Tried to evaluate a constraint that is not part of this `System`"
            );

            let constraint = &system.constraint_edges[self.id as usize];
            utils::calculate_residual(constraint, &system.variables)
        }

        /// Get a type-erased handle to the constraint.
        ///
        /// To turn the returned handle back into a typed handle, use
        /// [`AnyConstraintHandle::as_tagged_constraint`].
        pub fn as_any_constraint(self) -> AnyConstraintHandle {
            AnyConstraintHandle {
                system_id: self.system_id,
                id: self.id,
                tag: T::tag(),
            }
        }
    }

    /// A type-erased handle to a constraint within a [`System`].
    #[derive(Copy, Clone, Debug)]
    pub struct AnyConstraintHandle {
        /// The ID of the system the constraint belongs to.
        pub(crate) system_id: u32,
        /// The ID of the constraint within the system.
        pub(crate) id: u32,
        tag: ConstraintTag,
    }

    impl AnyConstraintHandle {
        pub(crate) fn from_ids_and_tag(system_id: u32, id: u32, tag: ConstraintTag) -> Self {
            Self { system_id, id, tag }
        }

        /// Get the value of the constraint.
        pub fn calculate_residual(&self, system: &System) -> f64 {
            // TODO: return `Result` instead of panicking?
            assert_eq!(
                self.system_id, system.id,
                "Tried to get a constraint that is not part of this `System`"
            );

            let constraint = &system.constraint_edges[self.id as usize];
            utils::calculate_residual(constraint, &system.variables)
        }

        /// Get a typed handle to the constraint.
        pub fn as_tagged_constraint(self) -> TaggedConstraintHandle {
            match self.tag {
                ConstraintTag::PointPointDistance => TaggedConstraintHandle::PointPointDistance(
                    ConstraintHandle::from_ids(self.system_id, self.id),
                ),
                ConstraintTag::PointPointPointAngle => {
                    TaggedConstraintHandle::PointPointPointAngle(ConstraintHandle::from_ids(
                        self.system_id,
                        self.id,
                    ))
                }
                ConstraintTag::PointLineIncidence => TaggedConstraintHandle::PointLineIncidence(
                    ConstraintHandle::from_ids(self.system_id, self.id),
                ),
                ConstraintTag::LineLineAngle => TaggedConstraintHandle::LineLineAngle(
                    ConstraintHandle::from_ids(self.system_id, self.id),
                ),
                ConstraintTag::LineLineParallelism => TaggedConstraintHandle::LineLineParallelism(
                    ConstraintHandle::from_ids(self.system_id, self.id),
                ),
                ConstraintTag::LineCircleTangency => TaggedConstraintHandle::LineCircleTangency(
                    ConstraintHandle::from_ids(self.system_id, self.id),
                ),
            }
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

    impl<T> core::hash::Hash for ConstraintHandle<T> {
        fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
            self.system_id.hash(state);
            self.id.hash(state);
        }
    }

    impl PartialEq for AnyConstraintHandle {
        fn eq(&self, other: &Self) -> bool {
            // Constraint handle IDs are unique, so we don't need to compare the tags.
            self.system_id == other.system_id && self.id == other.id
        }
    }
    impl Eq for AnyConstraintHandle {}

    impl Ord for AnyConstraintHandle {
        fn cmp(&self, other: &Self) -> core::cmp::Ordering {
            // Constraint handle IDs are unique, so we don't need to compare the tags.
            (self.system_id, self.id).cmp(&(other.system_id, other.id))
        }
    }
    impl PartialOrd for AnyConstraintHandle {
        fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl core::hash::Hash for AnyConstraintHandle {
        fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
            // Constraint handle IDs are unique, so we don't need to hash the tags.
            self.system_id.hash(state);
            self.id.hash(state);
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

impl sealed::ConstraintInner for PointPointDistance {
    fn tag() -> ConstraintTag {
        ConstraintTag::PointPointDistance
    }
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

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([point1.drop_system_id(), point2.drop_system_id()]),
        );
        system.add_constraint(Edge::PointPointDistance(constraint))
    }

    /// If only, say, the residual or some subset of the gradient entries are actually used, and
    /// that is clear from the code at the call site, the compiler should be able to correctly
    /// remove the dead calculations as this is marked as `#[inline(always)]`. That allows us not to
    /// to duplicate code unnecessarily for the calculations. Plus, calculating the residuals and
    /// the Jacobian at the same time is a common case, and for some constraints it's more efficient
    /// when they're calculated together.
    #[inline(always)]
    fn compute_residual_and_gradient_(
        variables: &[f64; 4],
        param_distance: f64,
    ) -> (f64, [f64; 4]) {
        let point1 = kurbo::Point {
            x: variables[0],
            y: variables[1],
        };
        let point2 = kurbo::Point {
            x: variables[2],
            y: variables[3],
        };

        let distance = ((point1.x - point2.x).square() + (point1.y - point2.y).square()).sqrt();
        let residual = distance - param_distance;

        let distance_recip = 1. / distance;
        let gradient = [
            (point1.x - point2.x) * distance_recip,
            (point1.y - point2.y) * distance_recip,
            -(point1.x - point2.x) * distance_recip,
            -(point1.y - point2.y) * distance_recip,
        ];

        (residual, gradient)
    }

    pub(crate) fn compute_residual(&self, variables: &[f64]) -> f64 {
        // The compiler should be able to optimize this such that only the residual is calculated.
        // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
        Self::compute_residual_and_gradient_(
            &[
                variables[self.point1_idx as usize],
                variables[self.point1_idx as usize + 1],
                variables[self.point2_idx as usize],
                variables[self.point2_idx as usize + 1],
            ],
            self.distance,
        )
        .0
    }

    pub(crate) fn compute_residual_and_gradient(
        &self,
        subsystem: &Subsystem<'_>,
        variables: &[f64],
        residual: &mut f64,
        gradient: &mut [f64],
    ) {
        let (r, g) = Self::compute_residual_and_gradient_(
            &[
                variables[self.point1_idx as usize],
                variables[self.point1_idx as usize + 1],
                variables[self.point2_idx as usize],
                variables[self.point2_idx as usize + 1],
            ],
            self.distance,
        );

        *residual += r;

        if let Some(idx) = subsystem.free_variable_index(self.point1_idx) {
            gradient[idx as usize] += g[0];
        }
        if let Some(idx) = subsystem.free_variable_index(self.point1_idx + 1) {
            gradient[idx as usize] += g[1];
        }
        if let Some(idx) = subsystem.free_variable_index(self.point2_idx) {
            gradient[idx as usize] += g[2];
        }
        if let Some(idx) = subsystem.free_variable_index(self.point2_idx + 1) {
            gradient[idx as usize] += g[3];
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

impl sealed::ConstraintInner for PointPointPointAngle {
    fn tag() -> ConstraintTag {
        ConstraintTag::PointPointPointAngle
    }
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

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([
                point1.drop_system_id(),
                point2.drop_system_id(),
                point3.drop_system_id(),
            ]),
        );
        system.add_constraint(Edge::PointPointPointAngle(constraint))
    }

    // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
    #[inline(always)]
    fn compute_residual_and_gradient_(variables: &[f64; 6], param_angle: f64) -> (f64, [f64; 6]) {
        let point1 = kurbo::Point {
            x: variables[0],
            y: variables[1],
        };
        let point2 = kurbo::Point {
            x: variables[2],
            y: variables[3],
        };
        let point3 = kurbo::Point {
            x: variables[4],
            y: variables[5],
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

        let residual = angle - param_angle;

        let u_squared_recip = u.length_squared().recip();
        let v_squared_recip = v.length_squared().recip();

        let dangle_dpoint1x = u.y * u_squared_recip;
        let dangle_dpoint1y = -u.x * u_squared_recip;
        let dangle_dpoint3x = -v.y * v_squared_recip;
        let dangle_dpoint3y = v.x * v_squared_recip;

        let dangle_dpoint2x = -dangle_dpoint1x - dangle_dpoint3x;
        let dangle_dpoint2y = -dangle_dpoint1y - dangle_dpoint3y;

        let gradient = [
            dangle_dpoint1x,
            dangle_dpoint1y,
            dangle_dpoint2x,
            dangle_dpoint2y,
            dangle_dpoint3x,
            dangle_dpoint3y,
        ];

        (residual, gradient)
    }

    pub(crate) fn compute_residual(&self, variables: &[f64]) -> f64 {
        // The compiler should be able to optimize this such that only the residual is calculated.
        // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
        Self::compute_residual_and_gradient_(
            &[
                variables[self.point1_idx as usize],
                variables[self.point1_idx as usize + 1],
                variables[self.point2_idx as usize],
                variables[self.point2_idx as usize + 1],
                variables[self.point3_idx as usize],
                variables[self.point3_idx as usize + 1],
            ],
            self.angle,
        )
        .0
    }

    pub(crate) fn compute_residual_and_gradient(
        &self,
        subsystem: &Subsystem<'_>,
        variables: &[f64],
        residual: &mut f64,
        gradient: &mut [f64],
    ) {
        let (r, g) = Self::compute_residual_and_gradient_(
            &[
                variables[self.point1_idx as usize],
                variables[self.point1_idx as usize + 1],
                variables[self.point2_idx as usize],
                variables[self.point2_idx as usize + 1],
                variables[self.point3_idx as usize],
                variables[self.point3_idx as usize + 1],
            ],
            self.angle,
        );

        *residual += r;

        if let Some(idx) = subsystem.free_variable_index(self.point1_idx) {
            gradient[idx as usize] += g[0];
        }
        if let Some(idx) = subsystem.free_variable_index(self.point1_idx + 1) {
            gradient[idx as usize] += g[1];
        }
        if let Some(idx) = subsystem.free_variable_index(self.point2_idx) {
            gradient[idx as usize] += g[2];
        }
        if let Some(idx) = subsystem.free_variable_index(self.point2_idx + 1) {
            gradient[idx as usize] += g[3];
        }
        if let Some(idx) = subsystem.free_variable_index(self.point3_idx) {
            gradient[idx as usize] += g[4];
        }
        if let Some(idx) = subsystem.free_variable_index(self.point3_idx + 1) {
            gradient[idx as usize] += g[5];
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

impl sealed::ConstraintInner for PointLineIncidence {
    fn tag() -> ConstraintTag {
        ConstraintTag::PointLineIncidence
    }
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

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([
                point.drop_system_id(),
                system.variable_to_primitive[line_point1_idx as usize],
                system.variable_to_primitive[line_point2_idx as usize],
            ]),
        );
        system.add_constraint(Edge::PointLineIncidence(constraint))
    }

    // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
    #[inline(always)]
    fn compute_residual_and_gradient_(variables: &[f64; 6]) -> (f64, [f64; 6]) {
        let point1 = kurbo::Point {
            x: variables[0],
            y: variables[1],
        };
        let point2 = kurbo::Point {
            x: variables[2],
            y: variables[3],
        };
        let point3 = kurbo::Point {
            x: variables[4],
            y: variables[5],
        };

        // For collinear points, the triangle defined by those points has area 0.
        let residual = point1.x * (point2.y - point3.y)
            + point2.x * (point3.y - point1.y)
            + point3.x * (point1.y - point2.y);

        let gradient = [
            point2.y - point3.y,
            -point2.x + point3.x,
            point3.y - point1.y,
            point1.x - point3.x,
            point1.y - point2.y,
            -point1.x + point2.x,
        ];

        (residual, gradient)
    }

    pub(crate) fn compute_residual(&self, variables: &[f64]) -> f64 {
        // The compiler should be able to optimize this such that only the residual is calculated.
        // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
        Self::compute_residual_and_gradient_(&[
            variables[self.point_idx as usize],
            variables[self.point_idx as usize + 1],
            variables[self.line_point1_idx as usize],
            variables[self.line_point1_idx as usize + 1],
            variables[self.line_point2_idx as usize],
            variables[self.line_point2_idx as usize + 1],
        ])
        .0
    }

    pub(crate) fn compute_residual_and_gradient(
        &self,
        subsystem: &Subsystem<'_>,
        variables: &[f64],
        residual: &mut f64,
        gradient: &mut [f64],
    ) {
        let (r, g) = Self::compute_residual_and_gradient_(&[
            variables[self.point_idx as usize],
            variables[self.point_idx as usize + 1],
            variables[self.line_point1_idx as usize],
            variables[self.line_point1_idx as usize + 1],
            variables[self.line_point2_idx as usize],
            variables[self.line_point2_idx as usize + 1],
        ]);

        *residual += r;

        if let Some(idx) = subsystem.free_variable_index(self.point_idx) {
            gradient[idx as usize] += g[0];
        }
        if let Some(idx) = subsystem.free_variable_index(self.point_idx + 1) {
            gradient[idx as usize] += g[1];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line_point1_idx) {
            gradient[idx as usize] += g[2];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line_point1_idx + 1) {
            gradient[idx as usize] += g[3];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line_point2_idx) {
            gradient[idx as usize] += g[4];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line_point2_idx + 1) {
            gradient[idx as usize] += g[5];
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

impl sealed::ConstraintInner for LineLineAngle {
    fn tag() -> ConstraintTag {
        ConstraintTag::LineLineAngle
    }
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

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([
                system.variable_to_primitive[line1_point1_idx as usize],
                system.variable_to_primitive[line1_point2_idx as usize],
                system.variable_to_primitive[line2_point1_idx as usize],
                system.variable_to_primitive[line2_point2_idx as usize],
            ]),
        );
        system.add_constraint(Edge::LineLineAngle(constraint))
    }

    // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
    #[inline(always)]
    fn compute_residual_and_gradient_(variables: &[f64; 8], param_angle: f64) -> (f64, [f64; 8]) {
        let line1_point1 = kurbo::Point {
            x: variables[0],
            y: variables[1],
        };
        let line1_point2 = kurbo::Point {
            x: variables[2],
            y: variables[3],
        };
        let line2_point1 = kurbo::Point {
            x: variables[4],
            y: variables[5],
        };
        let line2_point2 = kurbo::Point {
            x: variables[6],
            y: variables[7],
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

        let residual = angle - param_angle;

        let u_squared_recip = u.length_squared().recip();
        let v_squared_recip = v.length_squared().recip();

        let dangle_dline1_point1x = -u.y * u_squared_recip;
        let dangle_dline1_point1y = u.x * u_squared_recip;
        let dangle_dline2_point1x = v.y * v_squared_recip;
        let dangle_dline2_point1y = -v.x * v_squared_recip;

        let gradient = [
            dangle_dline1_point1x,
            dangle_dline1_point1y,
            -dangle_dline1_point1x,
            -dangle_dline1_point1y,
            dangle_dline2_point1x,
            dangle_dline2_point1y,
            -dangle_dline2_point1x,
            -dangle_dline2_point1y,
        ];

        (residual, gradient)
    }

    pub(crate) fn compute_residual(&self, variables: &[f64]) -> f64 {
        // The compiler should be able to optimize this such that only the residual is calculated.
        // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
        Self::compute_residual_and_gradient_(
            &[
                variables[self.line1_point1_idx as usize],
                variables[self.line1_point1_idx as usize + 1],
                variables[self.line1_point2_idx as usize],
                variables[self.line1_point2_idx as usize + 1],
                variables[self.line2_point1_idx as usize],
                variables[self.line2_point1_idx as usize + 1],
                variables[self.line2_point2_idx as usize],
                variables[self.line2_point2_idx as usize + 1],
            ],
            self.angle,
        )
        .0
    }

    pub(crate) fn compute_residual_and_gradient(
        &self,
        subsystem: &Subsystem<'_>,
        variables: &[f64],
        residual: &mut f64,
        gradient: &mut [f64],
    ) {
        let (r, g) = Self::compute_residual_and_gradient_(
            &[
                variables[self.line1_point1_idx as usize],
                variables[self.line1_point1_idx as usize + 1],
                variables[self.line1_point2_idx as usize],
                variables[self.line1_point2_idx as usize + 1],
                variables[self.line2_point1_idx as usize],
                variables[self.line2_point1_idx as usize + 1],
                variables[self.line2_point2_idx as usize],
                variables[self.line2_point2_idx as usize + 1],
            ],
            self.angle,
        );

        *residual += r;

        if let Some(idx) = subsystem.free_variable_index(self.line1_point1_idx) {
            gradient[idx as usize] += g[0];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line1_point1_idx + 1) {
            gradient[idx as usize] += g[1];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line1_point2_idx) {
            gradient[idx as usize] += g[2];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line1_point2_idx + 1) {
            gradient[idx as usize] += g[3];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line2_point1_idx) {
            gradient[idx as usize] += g[4];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line2_point1_idx + 1) {
            gradient[idx as usize] += g[5];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line2_point2_idx) {
            gradient[idx as usize] += g[6];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line2_point2_idx + 1) {
            gradient[idx as usize] += g[7];
        }
    }
}

/// Constrain two lines to be parallel to each other.
pub struct LineLineParallelism {
    line1_point1_idx: u32,
    line1_point2_idx: u32,
    line2_point1_idx: u32,
    line2_point2_idx: u32,
}

impl sealed::ConstraintInner for LineLineParallelism {
    fn tag() -> ConstraintTag {
        ConstraintTag::LineLineParallelism
    }
}

impl LineLineParallelism {
    /// Construct a constraint between two lines to be parallel to each other.
    pub fn create(
        system: &mut System,
        line1: ElementHandle<elements::Line>,
        line2: ElementHandle<elements::Line>,
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

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([
                system.variable_to_primitive[line1_point1_idx as usize],
                system.variable_to_primitive[line1_point2_idx as usize],
                system.variable_to_primitive[line2_point1_idx as usize],
                system.variable_to_primitive[line2_point2_idx as usize],
            ]),
        );
        system.add_constraint(Edge::LineLineParallelism(Self {
            line1_point1_idx,
            line1_point2_idx,
            line2_point1_idx,
            line2_point2_idx,
        }))
    }

    // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
    #[inline(always)]
    fn compute_residual_and_gradient_(variables: &[f64; 8]) -> (f64, [f64; 8]) {
        let line1_point1 = kurbo::Point {
            x: variables[0],
            y: variables[1],
        };
        let line1_point2 = kurbo::Point {
            x: variables[2],
            y: variables[3],
        };
        let line2_point1 = kurbo::Point {
            x: variables[4],
            y: variables[5],
        };
        let line2_point2 = kurbo::Point {
            x: variables[6],
            y: variables[7],
        };

        let u = line1_point2 - line1_point1;
        let v = line2_point2 - line2_point1;

        let residual = v.cross(u);

        let gradient = [
            v.y,  // l1p1x
            -v.x, // l1p1y
            -v.y, // l1p2x
            v.x,  // l1p2y
            -u.y, // l2p1x
            u.x,  // l2p1y
            u.y,  // l2p2x
            -u.x, // l2p2y
        ];

        (residual, gradient)
    }

    pub(crate) fn compute_residual(&self, variables: &[f64]) -> f64 {
        // The compiler should be able to optimize this such that only the residual is calculated.
        // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
        Self::compute_residual_and_gradient_(&[
            variables[self.line1_point1_idx as usize],
            variables[self.line1_point1_idx as usize + 1],
            variables[self.line1_point2_idx as usize],
            variables[self.line1_point2_idx as usize + 1],
            variables[self.line2_point1_idx as usize],
            variables[self.line2_point1_idx as usize + 1],
            variables[self.line2_point2_idx as usize],
            variables[self.line2_point2_idx as usize + 1],
        ])
        .0
    }

    pub(crate) fn compute_residual_and_gradient(
        &self,
        subsystem: &Subsystem<'_>,
        variables: &[f64],
        residual: &mut f64,
        gradient: &mut [f64],
    ) {
        let (r, g) = Self::compute_residual_and_gradient_(&[
            variables[self.line1_point1_idx as usize],
            variables[self.line1_point1_idx as usize + 1],
            variables[self.line1_point2_idx as usize],
            variables[self.line1_point2_idx as usize + 1],
            variables[self.line2_point1_idx as usize],
            variables[self.line2_point1_idx as usize + 1],
            variables[self.line2_point2_idx as usize],
            variables[self.line2_point2_idx as usize + 1],
        ]);

        *residual += r;

        if let Some(idx) = subsystem.free_variable_index(self.line1_point1_idx) {
            gradient[idx as usize] += g[0];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line1_point1_idx + 1) {
            gradient[idx as usize] += g[1];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line1_point2_idx) {
            gradient[idx as usize] += g[2];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line1_point2_idx + 1) {
            gradient[idx as usize] += g[3];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line2_point1_idx) {
            gradient[idx as usize] += g[4];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line2_point1_idx + 1) {
            gradient[idx as usize] += g[5];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line2_point2_idx) {
            gradient[idx as usize] += g[6];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line2_point2_idx + 1) {
            gradient[idx as usize] += g[7];
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

impl sealed::ConstraintInner for LineCircleTangency {
    fn tag() -> ConstraintTag {
        ConstraintTag::LineCircleTangency
    }
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

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([
                system.variable_to_primitive[line_point1_idx as usize],
                system.variable_to_primitive[line_point2_idx as usize],
                system.variable_to_primitive[circle_center_idx as usize],
                circle.drop_system_id(),
            ]),
        );
        system.add_constraint(Edge::LineCircleTangency(constraint))
    }

    // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
    #[inline(always)]
    fn compute_residual_and_gradient_(variables: &[f64; 7]) -> (f64, [f64; 7]) {
        let line_point1 = kurbo::Point {
            x: variables[0],
            y: variables[1],
        };
        let line_point2 = kurbo::Point {
            x: variables[2],
            y: variables[3],
        };
        let circle_center = kurbo::Point {
            x: variables[4],
            y: variables[5],
        };
        let circle_radius = variables[6];

        let length2 = line_point1.distance_squared(line_point2);
        let length = length2.sqrt();

        // TODO: better handle degenerate lines of length 0.
        if length == 0. {
            return (0., [0.; 7]);
        }

        let length_recip = 1. / length;
        let signed_area = line_point1.x * (line_point2.y - circle_center.y)
            + line_point2.x * (circle_center.y - line_point1.y)
            + circle_center.x * (line_point1.y - line_point2.y);

        // We are interested in the _unsigned_ area here, as it does not matter on which side of
        // the line the circle center lies. That does mean there is a cusp when the circle
        // center is exactly on the line.
        let residual = length_recip * signed_area.abs() - circle_radius;

        let sign = signed_area.signum();
        let length3_recip = 1. / (length2 * length);
        let gradient = [
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
            -1.,
        ];

        (residual, gradient)
    }

    pub(crate) fn compute_residual(&self, variables: &[f64]) -> f64 {
        // The compiler should be able to optimize this such that only the residual is calculated.
        // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
        Self::compute_residual_and_gradient_(&[
            variables[self.line_point1_idx as usize],
            variables[self.line_point1_idx as usize + 1],
            variables[self.line_point2_idx as usize],
            variables[self.line_point2_idx as usize + 1],
            variables[self.circle_center_idx as usize],
            variables[self.circle_center_idx as usize + 1],
            variables[self.circle_radius_idx as usize],
        ])
        .0
    }

    pub(crate) fn compute_residual_and_gradient(
        &self,
        subsystem: &Subsystem<'_>,
        variables: &[f64],
        residual: &mut f64,
        gradient: &mut [f64],
    ) {
        let (r, g) = Self::compute_residual_and_gradient_(&[
            variables[self.line_point1_idx as usize],
            variables[self.line_point1_idx as usize + 1],
            variables[self.line_point2_idx as usize],
            variables[self.line_point2_idx as usize + 1],
            variables[self.circle_center_idx as usize],
            variables[self.circle_center_idx as usize + 1],
            variables[self.circle_radius_idx as usize],
        ]);

        *residual += r;

        if let Some(idx) = subsystem.free_variable_index(self.line_point1_idx) {
            gradient[idx as usize] += g[0];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line_point1_idx + 1) {
            gradient[idx as usize] += g[1];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line_point2_idx) {
            gradient[idx as usize] += g[2];
        }
        if let Some(idx) = subsystem.free_variable_index(self.line_point2_idx + 1) {
            gradient[idx as usize] += g[3];
        }
        if let Some(idx) = subsystem.free_variable_index(self.circle_center_idx) {
            gradient[idx as usize] += g[4];
        }
        if let Some(idx) = subsystem.free_variable_index(self.circle_center_idx + 1) {
            gradient[idx as usize] += g[5];
        }
        if let Some(idx) = subsystem.free_variable_index(self.circle_radius_idx) {
            gradient[idx as usize] += g[6];
        }
    }
}

/// The actual type of the constraint.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ConstraintTag {
    PointPointDistance,
    PointPointPointAngle,
    PointLineIncidence,
    LineLineAngle,
    LineLineParallelism,
    LineCircleTangency,
}

impl<'a> From<&'a Edge> for ConstraintTag {
    fn from(edge: &'a Edge) -> Self {
        match edge {
            Edge::PointPointDistance { .. } => Self::PointPointDistance,
            Edge::PointPointPointAngle { .. } => Self::PointPointPointAngle,
            Edge::PointLineIncidence { .. } => Self::PointLineIncidence,
            Edge::LineLineAngle { .. } => Self::LineLineAngle,
            Edge::LineLineParallelism { .. } => Self::LineLineParallelism,
            Edge::LineCircleTangency { .. } => Self::LineCircleTangency,
        }
    }
}

pub(crate) mod sealed {
    pub(crate) trait ConstraintInner {
        fn tag() -> super::ConstraintTag;
    }
}

/// A constraint between geometric [elements](crate::Element).
///
/// These can be added to a [`System`].
#[expect(private_bounds, reason = "Sealed inner trait")]
pub trait Constraint: sealed::ConstraintInner {}

impl Constraint for PointPointDistance {}
impl Constraint for PointPointPointAngle {}
impl Constraint for PointLineIncidence {}
impl Constraint for LineLineAngle {}
impl Constraint for LineLineParallelism {}
impl Constraint for LineCircleTangency {}

#[cfg(test)]
mod tests {
    use core::array;

    use crate::Rng;

    use super::*;

    /// Generate an array of random floats between 0. and 1. inclusive.
    fn next_f64s<const N: usize>(rng: &mut Rng) -> [f64; N] {
        array::from_fn(|_| rng.next_f64())
    }

    /// Calculate component-wise `a + b`.
    fn add<const N: usize>(mut a: [f64; N], b: [f64; N]) -> [f64; N] {
        for n in 0..N {
            a[n] += b[n];
        }
        a
    }

    /// Calculate the dot product of a and b, `sum_n(a[n] * b[n])`.
    fn dot<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
        let mut dot = 0.;
        for n in 0..N {
            dot += a[n] * b[n];
        }
        dot
    }

    /// For a given residual r(variables) and small step `delta` we assume
    /// `r(variables + delta) ~= r(variables) + dot(gradient, delta)`
    /// where `gradient` is the vector of partial derivatives `r'(variables)`.
    ///
    /// This tests whether that approximation actually holds.
    fn test_first_finite_difference<const N: usize>(
        residual_and_gradient: impl Fn(&[f64; N]) -> (f64, [f64; N]),
        variable_and_delta_map: impl Fn([f64; N], [f64; N]) -> ([f64; N], [f64; N]),
    ) {
        const RELATIVE_EPSILON: f64 = 1e-3;
        let mut rng = Rng::from_seed(42);

        for _ in 0..5 {
            let (variables, delta) =
                variable_and_delta_map(next_f64s(&mut rng), next_f64s(&mut rng));
            let (r, gradient) = residual_and_gradient(&variables);
            let (r_plus, _) = residual_and_gradient(&add(variables, delta));

            let linearized_diff = dot(gradient, delta);
            let first_finite_diff = r_plus - r;

            assert!(
                (linearized_diff - first_finite_diff).abs()
                    / f64::max(linearized_diff.abs(), first_finite_diff.abs())
                    < RELATIVE_EPSILON,
                "Difference predicted based on linearized first derivatives and actual first finite difference do not match.\n\
                Variables: {variables:?}\n\
                Delta: {delta:?}\n\
                Predicted: {linearized_diff:?}\n\
                Actual: {first_finite_diff:?}"
            );
        }
    }

    #[test]
    fn point_point_distance_first_finite_difference() {
        test_first_finite_difference(
            |variables| PointPointDistance::compute_residual_and_gradient_(variables, 0.5e0),
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e0),
                    delta.map(|d| (d - 0.5) * 1e-4),
                )
            },
        );
        test_first_finite_difference(
            |variables| PointPointDistance::compute_residual_and_gradient_(variables, 0.5e-9),
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e-10),
                    delta.map(|d| (d - 0.5) * 1e-14),
                )
            },
        );
        test_first_finite_difference(
            |variables| PointPointDistance::compute_residual_and_gradient_(variables, 0.5e10),
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e10),
                    delta.map(|d| (d - 0.5) * 1e6),
                )
            },
        );
    }

    #[test]
    fn point_point_point_angle_first_finite_difference() {
        test_first_finite_difference(
            |variables| {
                PointPointPointAngle::compute_residual_and_gradient_(variables, 10_f64.to_radians())
            },
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e0),
                    delta.map(|d| (d - 0.5) * 1e-5),
                )
            },
        );
        test_first_finite_difference(
            |variables| {
                PointPointPointAngle::compute_residual_and_gradient_(
                    variables,
                    -40_f64.to_radians(),
                )
            },
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e-10),
                    delta.map(|d| (d - 0.5) * 1e-15),
                )
            },
        );
        test_first_finite_difference(
            |variables| PointPointDistance::compute_residual_and_gradient_(variables, 0.5e10),
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e10),
                    delta.map(|d| (d - 0.5) * 1e6),
                )
            },
        );
    }

    #[test]
    fn point_line_incidence_first_finite_difference() {
        test_first_finite_difference(
            PointLineIncidence::compute_residual_and_gradient_,
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e0),
                    delta.map(|d| (d - 0.5) * 1e-4),
                )
            },
        );
        test_first_finite_difference(
            PointLineIncidence::compute_residual_and_gradient_,
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e-10),
                    delta.map(|d| (d - 0.5) * 1e-14),
                )
            },
        );
    }

    #[test]
    fn line_line_angle_first_finite_difference() {
        test_first_finite_difference(
            |variables| {
                LineLineAngle::compute_residual_and_gradient_(variables, 10_f64.to_radians())
            },
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e0),
                    delta.map(|d| (d - 0.5) * 1e-4),
                )
            },
        );
        test_first_finite_difference(
            |variables| {
                LineLineAngle::compute_residual_and_gradient_(variables, -40_f64.to_radians())
            },
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e-10),
                    delta.map(|d| (d - 0.5) * 1e-14),
                )
            },
        );
    }

    #[test]
    fn line_line_parallelism_first_finite_difference() {
        test_first_finite_difference(
            LineLineParallelism::compute_residual_and_gradient_,
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e0),
                    delta.map(|d| (d - 0.5) * 1e-4),
                )
            },
        );
        test_first_finite_difference(
            LineLineParallelism::compute_residual_and_gradient_,
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e-10),
                    delta.map(|d| (d - 0.5) * 1e-14),
                )
            },
        );
    }

    #[test]
    fn line_circle_tangency_first_finite_difference() {
        test_first_finite_difference(
            LineCircleTangency::compute_residual_and_gradient_,
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e0),
                    delta.map(|d| (d - 0.5) * 1e-4),
                )
            },
        );
        test_first_finite_difference(
            LineCircleTangency::compute_residual_and_gradient_,
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e-10),
                    delta.map(|d| (d - 0.5) * 1e-14),
                )
            },
        );
    }
}
