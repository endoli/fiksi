// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use crate::{
    Edge,
    constraints::{PointPointDistance_, PointPointPointAngle_},
};

/// Compute residuals and Jacobian for all constraints.
///
/// The Jacobian is relative to the free variables.
pub(crate) fn calculate_residuals_and_jacobian(
    constraints: &[&Edge],
    index_map: &alloc::collections::BTreeMap<u32, u32>,
    variables: &[f64],
    residuals: &mut [f64],
    jacobian: &mut [f64],
) {
    jacobian.fill(0.);
    residuals.fill(0.);

    let num_free_variables = index_map.len();

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
                    index_map,
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
                    index_map,
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
