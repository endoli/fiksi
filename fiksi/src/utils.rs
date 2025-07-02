// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use crate::Edge;

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
        match constraint {
            Edge::PointPointDistance(constraint) => {
                constraint.compute_residual_and_partial_derivatives(
                    free_variable_map,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::PointPointPointAngle(constraint) => {
                constraint.compute_residual_and_partial_derivatives(
                    free_variable_map,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::PointLineIncidence(constraint) => {
                constraint.compute_residual_and_partial_derivatives(
                    free_variable_map,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::LineCircleTangency(constraint) => {
                constraint.compute_residual_and_partial_derivatives(
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

#[inline]
pub(crate) fn sum_squares(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum()
}
