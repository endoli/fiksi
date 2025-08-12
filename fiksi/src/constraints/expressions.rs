// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(not(feature = "std"))]
use crate::floatfuncs::FloatFuncs;

use crate::Subsystem;

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

pub(crate) enum Expression {
    VariableVariableEquality(VariableVariableEquality),
    PointPointDistance(PointPointDistance),
    PointPointPointAngle(PointPointPointAngle),
    PointLineIncidence(PointLineIncidence),
    LineLineAngle(LineLineAngle),
    LineLineParallelism(LineLineParallelism),
    LineCircleTangency(LineCircleTangency),
}

pub(crate) struct VariableVariableEquality {
    pub(crate) variable1_idx: u32,
    pub(crate) variable2_idx: u32,
}

impl From<VariableVariableEquality> for Expression {
    fn from(expression: VariableVariableEquality) -> Self {
        Self::VariableVariableEquality(expression)
    }
}

impl VariableVariableEquality {
    /// See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
    #[inline(always)]
    fn compute_residual_and_gradient_(variables: [f64; 2]) -> (f64, [f64; 2]) {
        let residual = variables[1] - variables[0];

        let gradient = [-1., 1.];

        (residual, gradient)
    }

    pub(crate) fn compute_residual(&self, variables: &[f64]) -> f64 {
        // The compiler should be able to optimize this such that only the residual is calculated.
        // See the note about inlining on [`PointPointDistance::compute_residual_and_gradient_`].
        Self::compute_residual_and_gradient_([
            variables[self.variable1_idx as usize],
            variables[self.variable2_idx as usize],
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
        let (r, g) = Self::compute_residual_and_gradient_([
            variables[self.variable1_idx as usize],
            variables[self.variable2_idx as usize],
        ]);

        *residual += r;

        if let Some(idx) = subsystem.free_variable_index(self.variable1_idx) {
            gradient[idx as usize] += g[0];
        }
        if let Some(idx) = subsystem.free_variable_index(self.variable2_idx) {
            gradient[idx as usize] += g[1];
        }
    }
}

/// Constrain two points to have a given straight-line distance between each other.
pub(crate) struct PointPointDistance {
    pub(crate) point1_idx: u32,
    pub(crate) point2_idx: u32,

    /// Euclidean distance.
    pub(crate) distance: f64,
}

impl From<PointPointDistance> for Expression {
    fn from(expression: PointPointDistance) -> Self {
        Self::PointPointDistance(expression)
    }
}

impl PointPointDistance {
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
pub(crate) struct PointPointPointAngle {
    pub(crate) point1_idx: u32,
    pub(crate) point2_idx: u32,
    pub(crate) point3_idx: u32,

    /// Angle in radians.
    pub(crate) angle: f64,
}

impl From<PointPointPointAngle> for Expression {
    fn from(expression: PointPointPointAngle) -> Self {
        Self::PointPointPointAngle(expression)
    }
}

impl PointPointPointAngle {
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
pub(crate) struct PointLineIncidence {
    pub(crate) point_idx: u32,
    pub(crate) line_point1_idx: u32,
    pub(crate) line_point2_idx: u32,
}

impl From<PointLineIncidence> for Expression {
    fn from(expression: PointLineIncidence) -> Self {
        Self::PointLineIncidence(expression)
    }
}

impl PointLineIncidence {
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
pub(crate) struct LineLineAngle {
    pub(crate) line1_point1_idx: u32,
    pub(crate) line1_point2_idx: u32,
    pub(crate) line2_point1_idx: u32,
    pub(crate) line2_point2_idx: u32,

    /// Angle in radians.
    pub(crate) angle: f64,
}

impl From<LineLineAngle> for Expression {
    fn from(expression: LineLineAngle) -> Self {
        Self::LineLineAngle(expression)
    }
}

impl LineLineAngle {
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
pub(crate) struct LineLineParallelism {
    pub(crate) line1_point1_idx: u32,
    pub(crate) line1_point2_idx: u32,
    pub(crate) line2_point1_idx: u32,
    pub(crate) line2_point2_idx: u32,
}

impl From<LineLineParallelism> for Expression {
    fn from(expression: LineLineParallelism) -> Self {
        Self::LineLineParallelism(expression)
    }
}

impl LineLineParallelism {
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
pub(crate) struct LineCircleTangency {
    pub(crate) line_point1_idx: u32,
    pub(crate) line_point2_idx: u32,
    pub(crate) circle_center_idx: u32,
    pub(crate) circle_radius_idx: u32,
}

impl From<LineCircleTangency> for Expression {
    fn from(expression: LineCircleTangency) -> Self {
        Self::LineCircleTangency(expression)
    }
}

impl LineCircleTangency {
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
    fn variable_variable_equality_finite_difference() {
        test_first_finite_difference(
            |variables| VariableVariableEquality::compute_residual_and_gradient_(*variables),
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e0),
                    delta.map(|d| (d - 0.5) * 1e-4),
                )
            },
        );
        test_first_finite_difference(
            |variables| VariableVariableEquality::compute_residual_and_gradient_(*variables),
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e-10),
                    delta.map(|d| (d - 0.5) * 1e-14),
                )
            },
        );
        test_first_finite_difference(
            |variables| VariableVariableEquality::compute_residual_and_gradient_(*variables),
            |variables, delta| {
                (
                    variables.map(|d| (d - 0.5) * 1e10),
                    delta.map(|d| (d - 0.5) * 1e6),
                )
            },
        );
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
