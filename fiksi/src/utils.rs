// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use crate::{
    Edge,
    constraints::{
        LineCircleTangency_, PointLineIncidence_, PointPointDistance_, PointPointPointAngle_,
    },
};

/// Compute residuals and Jacobian for all constraints.
///
/// The Jacobian is relative to the free variables.
pub(crate) fn calculate_residuals_and_jacobian(
    constraints: &[&Edge],
    // Map from variable indices (in `variables`) to free variable indices.
    free_variable_map: &alloc::collections::BTreeMap<u32, u32>,
    variables: &[f64],
    residuals: &mut [f64],
    jacobian: &mut [f64],
) {
    jacobian.fill(0.);
    residuals.fill(0.);

    let num_free_variables = free_variable_map.len();

    for (constraint_idx, &constraint) in constraints.iter().enumerate() {
        match *constraint {
            Edge::PointPointDistance {
                point1_idx,
                point2_idx,
                distance,
            } => {
                PointPointDistance_ {
                    point1_idx,
                    point2_idx,
                    distance,
                }
                .compute_residual_and_partial_derivatives(
                    free_variable_map,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::PointPointPointAngle {
                point1_idx,
                point2_idx,
                point3_idx,
                angle,
            } => {
                PointPointPointAngle_ {
                    point1_idx,
                    point2_idx,
                    point3_idx,
                    angle,
                }
                .compute_residual_and_partial_derivatives(
                    free_variable_map,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::PointLineIncidence {
                point_idx,
                line_point1_idx,
                line_point2_idx,
            } => {
                PointLineIncidence_ {
                    point_idx,
                    line_point1_idx,
                    line_point2_idx,
                }
                .compute_residual_and_partial_derivatives(
                    free_variable_map,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::LineCircleTangency {
                line_point1_idx,
                line_point2_idx,
                circle_center_idx,
                circle_radius_idx,
            } => {
                LineCircleTangency_ {
                    line_point1_idx,
                    line_point2_idx,
                    circle_center_idx,
                    circle_radius_idx,
                }
                .compute_residual_and_partial_derivatives(
                    free_variable_map,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::LineLineAngle { .. } => {
                unimplemented!()
            }
        }
    }
}
