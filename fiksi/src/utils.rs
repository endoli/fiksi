// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use core::borrow::Borrow;

use crate::{Edge, Subsystem};

/// Compute residuals and Jacobian for all constraints.
///
/// The Jacobian is relative to the free variables.
pub(crate) fn calculate_residuals_and_jacobian(
    subsystem: &Subsystem<'_>,
    variables: &[f64],
    residuals: &mut [f64],
    jacobian: &mut [f64],
) {
    jacobian.fill(0.);
    residuals.fill(0.);

    let num_free_variables = subsystem.free_variables().len();

    for (constraint_idx, constraint) in subsystem.constraints().enumerate() {
        match constraint {
            Edge::PointPointDistance(constraint) => {
                constraint.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::PointPointPointAngle(constraint) => {
                constraint.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::PointLineIncidence(constraint) => {
                constraint.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::LineCircleTangency(constraint) => {
                constraint.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::LineLineAngle(constraint) => {
                constraint.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
            Edge::LineLineParallelism(constraint) => {
                constraint.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[constraint_idx],
                    &mut jacobian[constraint_idx * num_free_variables
                        ..(constraint_idx + 1) * num_free_variables],
                );
            }
        }
    }
}

#[inline]
pub(crate) fn sum_squares(values: impl IntoIterator<Item = impl Borrow<f64>>) -> f64 {
    values
        .into_iter()
        .map(|v| {
            let v = *v.borrow();
            v * v
        })
        .sum()
}
