// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Constraints between geometric elements.

use crate::{
    ConstraintHandle, ElementHandle, EncodedElement, System, elements, graph::IncidentElements,
};

pub(crate) mod expressions;

pub(crate) mod constraint {
    #[cfg(not(feature = "std"))]
    use crate::floatfuncs::FloatFuncs;

    use core::marker::PhantomData;

    use crate::{System, utils};

    use super::{Constraint, ConstraintTag};

    /// Dynamically tagged, typed handles to constraints.
    pub enum TaggedConstraintHandle {
        /// A handle to a [`PointPointCoincidence`](super::PointPointCoincidence) constraint.
        PointPointCoincidence(ConstraintHandle<super::PointPointCoincidence>),

        /// A handle to a [`PointPointDistance`](super::PointPointDistance) constraint.
        PointPointDistance(ConstraintHandle<super::PointPointDistance>),

        /// A handle to a [`PointPointPointAngle`](super::PointPointPointAngle) constraint.
        PointPointPointAngle(ConstraintHandle<super::PointPointPointAngle>),

        /// A handle to a [`PointLineIncidence`](super::PointLineIncidence) constraint.
        PointLineIncidence(ConstraintHandle<super::PointLineIncidence>),

        /// A handle to a [`PointLineDistance`](super::PointLineDistance) constraint.
        PointLineDistance(ConstraintHandle<super::PointLineDistance>),

        /// A handle to a [`PointCircleIncidence`](super::PointCircleIncidence) constraint.
        PointCircleIncidence(ConstraintHandle<super::PointCircleIncidence>),

        /// A handle to a [`SegmentSegmentLengthEquality`](super::SegmentSegmentLengthEquality) constraint.
        SegmentSegmentLengthEquality(ConstraintHandle<super::SegmentSegmentLengthEquality>),

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
        ///
        /// If the constraint referenced by the handle contributes more than one residual, this
        /// calculates the square root of the sum of its squared residuals.
        pub fn calculate_residual(self, system: &System) -> f64 {
            // TODO: return `Result` instead of panicking?
            assert_eq!(
                self.system_id, system.id,
                "Tried to evaluate a constraint that is not part of this `System`"
            );

            let constraint = &system.constraints[self.id as usize];
            if T::VALENCY > 1 {
                utils::sum_squares((0..T::VALENCY).map(|offset| {
                    utils::calculate_residual(
                        &system.expressions[constraint.expressions_idx as usize + offset as usize],
                        &system.variables,
                    )
                }))
                .sqrt()
            } else {
                let expression = &system.expressions[constraint.expressions_idx as usize];
                utils::calculate_residual(expression, &system.variables)
            }
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
        ///
        /// If the constraint referenced by the handle contributes more than one residual, this
        /// calculates the square root of the sum of its squared residuals.
        pub fn calculate_residual(&self, system: &System) -> f64 {
            // TODO: return `Result` instead of panicking?
            assert_eq!(
                self.system_id, system.id,
                "Tried to get a constraint that is not part of this `System`"
            );

            let valency = self.tag.valency();
            let constraint = &system.constraints[self.id as usize];
            if valency > 1 {
                utils::sum_squares((0..valency).map(|offset| {
                    utils::calculate_residual(
                        &system.expressions[constraint.expressions_idx as usize + offset as usize],
                        &system.variables,
                    )
                }))
                .sqrt()
            } else {
                let expression = &system.expressions[constraint.expressions_idx as usize];
                utils::calculate_residual(expression, &system.variables)
            }
        }

        /// Get a typed handle to the constraint.
        pub fn as_tagged_constraint(self) -> TaggedConstraintHandle {
            match self.tag {
                ConstraintTag::PointPointCoincidence => {
                    TaggedConstraintHandle::PointPointCoincidence(ConstraintHandle::from_ids(
                        self.system_id,
                        self.id,
                    ))
                }
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
                ConstraintTag::PointLineDistance => TaggedConstraintHandle::PointLineDistance(
                    ConstraintHandle::from_ids(self.system_id, self.id),
                ),
                ConstraintTag::PointCircleIncidence => {
                    TaggedConstraintHandle::PointCircleIncidence(ConstraintHandle::from_ids(
                        self.system_id,
                        self.id,
                    ))
                }
                ConstraintTag::SegmentSegmentLengthEquality => {
                    TaggedConstraintHandle::SegmentSegmentLengthEquality(
                        ConstraintHandle::from_ids(self.system_id, self.id),
                    )
                }
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

/// Constrain two points to be exactly at the same place.
///
/// If you want to constrain points such that they are a given distance from each other, use
/// [`PointPointDistance`].
pub struct PointPointCoincidence {
    point1_idx: u32,
    point2_idx: u32,
}

impl sealed::ConstraintInner for PointPointCoincidence {
    fn tag() -> ConstraintTag {
        ConstraintTag::PointPointCoincidence
    }
}

impl PointPointCoincidence {
    /// Construct a constraint between two points to be exactly at the same place.
    pub fn create(
        system: &mut System,
        point1: ElementHandle<elements::Point>,
        point2: ElementHandle<elements::Point>,
    ) -> ConstraintHandle<Self> {
        let &EncodedElement::Point { idx: point1_idx } = &system.elements[point1.id as usize]
        else {
            unreachable!()
        };
        let &EncodedElement::Point { idx: point2_idx } = &system.elements[point2.id as usize]
        else {
            unreachable!()
        };

        system.graph.add_constraint(
            2,
            IncidentElements::from_array([point1.drop_system_id(), point2.drop_system_id()]),
        );
        system.add_constraint(
            ConstraintTag::PointPointCoincidence,
            [
                // Linear difference along the x-axis
                expressions::VariableVariableEquality {
                    variable1_idx: point1_idx,
                    variable2_idx: point2_idx,
                }
                .into(),
                // Linear difference along the y-axis
                expressions::VariableVariableEquality {
                    variable1_idx: point1_idx + 1,
                    variable2_idx: point2_idx + 1,
                }
                .into(),
            ],
        )
    }
}

/// Constrain two points to have a given straight-line distance between each other.
///
/// If you want to constrain two points to be at the exact same location, i.e., having a distance
/// of 0, use [`PointPointCoincidence`] instead. Using [`PointPointDistance`] is numerically
/// singular at a distance of 0.
pub struct PointPointDistance {}

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
        let &EncodedElement::Point { idx: point1_idx } = &system.elements[point1.id as usize]
        else {
            unreachable!()
        };
        let &EncodedElement::Point { idx: point2_idx } = &system.elements[point2.id as usize]
        else {
            unreachable!()
        };

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([point1.drop_system_id(), point2.drop_system_id()]),
        );
        system.add_constraint(
            ConstraintTag::PointPointDistance,
            [expressions::PointPointDistance {
                point1_idx,
                point2_idx,
                distance,
            }
            .into()],
        )
    }
}

/// Constrain three points to describe a given angle.
pub struct PointPointPointAngle {}

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
        let &EncodedElement::Point { idx: point1_idx } = &system.elements[point1.id as usize]
        else {
            unreachable!()
        };
        let &EncodedElement::Point { idx: point2_idx } = &system.elements[point2.id as usize]
        else {
            unreachable!()
        };
        let &EncodedElement::Point { idx: point3_idx } = &system.elements[point3.id as usize]
        else {
            unreachable!()
        };

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([
                point1.drop_system_id(),
                point2.drop_system_id(),
                point3.drop_system_id(),
            ]),
        );
        system.add_constraint(
            ConstraintTag::PointPointPointAngle,
            [expressions::PointPointPointAngle {
                point1_idx,
                point2_idx,
                point3_idx,
                angle,
            }
            .into()],
        )
    }
}

/// Constrain a point and a line such that the point lies on the (infinite) line.
///
/// Note this does not constrain the point to lie on the line *segment* defined by `line`. This is
/// equivalent to constraining the three points (the two points of the line and the point proper)
/// to be collinear.
pub struct PointLineIncidence {}

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
        let &EncodedElement::Point { idx: point_idx } = &system.elements[point.id as usize] else {
            unreachable!()
        };
        let &EncodedElement::Line {
            point1_idx: line_point1_idx,
            point2_idx: line_point2_idx,
        } = &system.elements[line.id as usize]
        else {
            unreachable!()
        };

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([
                point.drop_system_id(),
                system.variable_to_primitive[line_point1_idx as usize],
                system.variable_to_primitive[line_point2_idx as usize],
            ]),
        );
        system.add_constraint(
            ConstraintTag::PointLineIncidence,
            [expressions::PointLineIncidence {
                point_idx,
                line_point1_idx,
                line_point2_idx,
            }
            .into()],
        )
    }
}

/// Constrain a point and a line such that the point is some signed distance from the (infinite)
/// line.
///
/// The distance is signed such that negative distances are on the left of the line, from the
/// perspective of the line's direction, and positive distances are on the right.
///
/// Note this does not constrain the point to lie some distance from the line *segment* defined by `line`.
pub struct PointLineDistance {}

impl sealed::ConstraintInner for PointLineDistance {
    fn tag() -> ConstraintTag {
        ConstraintTag::PointLineDistance
    }
}

impl PointLineDistance {
    /// Construct a constraint between a point and a line such that the point lies on the
    /// (infinite) line.
    pub fn create(
        system: &mut System,
        point: ElementHandle<elements::Point>,
        line: ElementHandle<elements::Line>,
        distance: f64,
    ) -> ConstraintHandle<Self> {
        let &EncodedElement::Point { idx: point_idx } = &system.elements[point.id as usize] else {
            unreachable!()
        };
        let &EncodedElement::Line {
            point1_idx: line_point1_idx,
            point2_idx: line_point2_idx,
        } = &system.elements[line.id as usize]
        else {
            unreachable!()
        };

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([
                point.drop_system_id(),
                system.variable_to_primitive[line_point1_idx as usize],
                system.variable_to_primitive[line_point2_idx as usize],
            ]),
        );
        system.add_constraint(
            ConstraintTag::PointLineDistance,
            [expressions::PointLineDistance {
                point_idx,
                line_point1_idx,
                line_point2_idx,
                distance,
            }
            .into()],
        )
    }
}

/// Constrain a point and a circle such that the point is on the circle.
pub struct PointCircleIncidence {}

impl sealed::ConstraintInner for PointCircleIncidence {
    fn tag() -> ConstraintTag {
        ConstraintTag::PointCircleIncidence
    }
}

impl PointCircleIncidence {
    /// Construct a constraint between a point and a circle such that the point is at the circle's
    /// center.
    pub fn create(
        system: &mut System,
        point: ElementHandle<elements::Point>,
        circle: ElementHandle<elements::Circle>,
    ) -> ConstraintHandle<Self> {
        let &EncodedElement::Point { idx: point_idx } = &system.elements[point.id as usize] else {
            unreachable!()
        };
        let &EncodedElement::Circle {
            center_idx: circle_center_idx,
            radius_idx: circle_radius_idx,
        } = &system.elements[circle.id as usize]
        else {
            unreachable!()
        };

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([
                point.drop_system_id(),
                system.variable_to_primitive[circle_center_idx as usize],
                system.variable_to_primitive[circle_radius_idx as usize],
            ]),
        );
        system.add_constraint(
            ConstraintTag::PointCircleIncidence,
            [expressions::PointCircleIncidence {
                point_idx,
                circle_center_idx,
                circle_radius_idx,
            }
            .into()],
        )
    }
}

/// Constrain two segments (defined by two points each) to have equal length.
pub struct SegmentSegmentLengthEquality {}

impl sealed::ConstraintInner for SegmentSegmentLengthEquality {
    fn tag() -> ConstraintTag {
        ConstraintTag::SegmentSegmentLengthEquality
    }
}

impl SegmentSegmentLengthEquality {
    /// Construct a constraint between two pairs of points such that the segments defined by the
    /// each point pair have equal length.
    pub fn create(
        system: &mut System,
        segment1_point1: ElementHandle<elements::Point>,
        segment1_point2: ElementHandle<elements::Point>,
        segment2_point1: ElementHandle<elements::Point>,
        segment2_point2: ElementHandle<elements::Point>,
    ) -> ConstraintHandle<Self> {
        let &EncodedElement::Point {
            idx: segment1_point1_idx,
        } = &system.elements[segment1_point1.id as usize]
        else {
            unreachable!()
        };
        let &EncodedElement::Point {
            idx: segment1_point2_idx,
        } = &system.elements[segment1_point2.id as usize]
        else {
            unreachable!()
        };
        let &EncodedElement::Point {
            idx: segment2_point1_idx,
        } = &system.elements[segment2_point1.id as usize]
        else {
            unreachable!()
        };
        let &EncodedElement::Point {
            idx: segment2_point2_idx,
        } = &system.elements[segment2_point2.id as usize]
        else {
            unreachable!()
        };

        system.graph.add_constraint(
            1,
            IncidentElements::from_array([
                system.variable_to_primitive[segment1_point1_idx as usize],
                system.variable_to_primitive[segment1_point2_idx as usize],
                system.variable_to_primitive[segment2_point1_idx as usize],
                system.variable_to_primitive[segment2_point2_idx as usize],
            ]),
        );
        system.add_constraint(
            ConstraintTag::SegmentSegmentLengthEquality,
            [expressions::SegmentSegmentLengthEquality {
                segment1_point1_idx,
                segment1_point2_idx,
                segment2_point1_idx,
                segment2_point2_idx,
            }
            .into()],
        )
    }
}

/// Constrain two lines to describe a given angle.
pub struct LineLineAngle {}

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
        let &EncodedElement::Line {
            point1_idx: line1_point1_idx,
            point2_idx: line1_point2_idx,
        } = &system.elements[line1.id as usize]
        else {
            unreachable!()
        };
        let &EncodedElement::Line {
            point1_idx: line2_point1_idx,
            point2_idx: line2_point2_idx,
        } = &system.elements[line2.id as usize]
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
        system.add_constraint(
            ConstraintTag::LineLineAngle,
            [expressions::LineLineAngle {
                line1_point1_idx,
                line1_point2_idx,
                line2_point1_idx,
                line2_point2_idx,
                angle,
            }
            .into()],
        )
    }
}

/// Constrain two lines to be parallel to each other.
pub struct LineLineParallelism {}

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
        let &EncodedElement::Line {
            point1_idx: line1_point1_idx,
            point2_idx: line1_point2_idx,
        } = &system.elements[line1.id as usize]
        else {
            unreachable!()
        };
        let &EncodedElement::Line {
            point1_idx: line2_point1_idx,
            point2_idx: line2_point2_idx,
        } = &system.elements[line2.id as usize]
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
        system.add_constraint(
            ConstraintTag::LineLineParallelism,
            [expressions::LineLineParallelism {
                line1_point1_idx,
                line1_point2_idx,
                line2_point1_idx,
                line2_point2_idx,
            }
            .into()],
        )
    }
}

/// Constrain a line and a circle such that the line is tangent on the circle.
pub struct LineCircleTangency {}

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
        let &EncodedElement::Line {
            point1_idx: line_point1_idx,
            point2_idx: line_point2_idx,
        } = &system.elements[line.id as usize]
        else {
            unreachable!()
        };
        let &EncodedElement::Circle {
            center_idx: circle_center_idx,
            radius_idx: circle_radius_idx,
        } = &system.elements[circle.id as usize]
        else {
            unreachable!()
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
        system.add_constraint(
            ConstraintTag::LineCircleTangency,
            [expressions::LineCircleTangency {
                line_point1_idx,
                line_point2_idx,
                circle_center_idx,
                circle_radius_idx,
            }
            .into()],
        )
    }
}

/// The actual type of the constraint.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ConstraintTag {
    PointPointCoincidence,
    PointPointDistance,
    PointPointPointAngle,
    PointLineIncidence,
    PointLineDistance,
    PointCircleIncidence,
    SegmentSegmentLengthEquality,
    LineLineAngle,
    LineLineParallelism,
    LineCircleTangency,
}

impl ConstraintTag {
    pub(crate) fn valency(self) -> u8 {
        match self {
            Self::PointPointCoincidence => PointPointCoincidence::VALENCY,
            Self::PointPointDistance => PointPointDistance::VALENCY,
            Self::PointPointPointAngle => PointPointPointAngle::VALENCY,
            Self::PointLineIncidence => PointLineIncidence::VALENCY,
            Self::PointLineDistance => PointLineDistance::VALENCY,
            Self::PointCircleIncidence => PointCircleIncidence::VALENCY,
            Self::SegmentSegmentLengthEquality => SegmentSegmentLengthEquality::VALENCY,
            Self::LineLineAngle => LineLineAngle::VALENCY,
            Self::LineLineParallelism => LineLineParallelism::VALENCY,
            Self::LineCircleTangency => LineCircleTangency::VALENCY,
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
pub trait Constraint: sealed::ConstraintInner {
    // Note: currently, a constraint's geometric valency is equal to the number of residual
    // expressions it provides. This might not always be the case, depending on exactly how a
    // constraint can be specified numerically.
    /// The valency of this constraint.
    ///
    /// This is the number of degrees of freedom taken away by the constraint.
    const VALENCY: u8;
}

impl Constraint for PointPointCoincidence {
    const VALENCY: u8 = 2;
}

impl Constraint for PointPointDistance {
    const VALENCY: u8 = 1;
}

impl Constraint for PointPointPointAngle {
    const VALENCY: u8 = 1;
}

impl Constraint for PointLineIncidence {
    const VALENCY: u8 = 1;
}

impl Constraint for PointLineDistance {
    const VALENCY: u8 = 1;
}

impl Constraint for PointCircleIncidence {
    const VALENCY: u8 = 1;
}

impl Constraint for SegmentSegmentLengthEquality {
    const VALENCY: u8 = 1;
}

impl Constraint for LineLineAngle {
    const VALENCY: u8 = 1;
}

impl Constraint for LineLineParallelism {
    const VALENCY: u8 = 1;
}

impl Constraint for LineCircleTangency {
    const VALENCY: u8 = 1;
}

impl ConstraintHandle<PointPointDistance> {
    /// Update the target distance of this point-point distance constraint.
    pub fn update_parameter(self, system: &mut System, distance: f64) {
        let constraint = &system.constraints[self.id as usize];
        let expressions::Expression::PointPointDistance(expression) =
            &mut system.expressions[constraint.expressions_idx as usize]
        else {
            unreachable!()
        };

        expression.distance = distance;
    }
}

impl ConstraintHandle<PointPointPointAngle> {
    /// Update the target angle of this point-point-point angle constraint.
    pub fn update_parameter(self, system: &mut System, angle: f64) {
        let constraint = &system.constraints[self.id as usize];
        let expressions::Expression::PointPointPointAngle(expression) =
            &mut system.expressions[constraint.expressions_idx as usize]
        else {
            unreachable!()
        };

        expression.angle = angle;
    }
}

impl ConstraintHandle<PointLineDistance> {
    /// Update the target distance of this point-line distance constraint.
    pub fn update_parameter(self, system: &mut System, distance: f64) {
        let constraint = &system.constraints[self.id as usize];
        let expressions::Expression::PointLineDistance(expression) =
            &mut system.expressions[constraint.expressions_idx as usize]
        else {
            unreachable!()
        };

        expression.distance = distance;
    }
}

impl ConstraintHandle<LineLineAngle> {
    /// Update the target angle of this line-line angle constraint.
    pub fn update_parameter(self, system: &mut System, angle: f64) {
        let constraint = &system.constraints[self.id as usize];
        let expressions::Expression::LineLineAngle(expression) =
            &mut system.expressions[constraint.expressions_idx as usize]
        else {
            unreachable!()
        };

        expression.angle = angle;
    }
}
