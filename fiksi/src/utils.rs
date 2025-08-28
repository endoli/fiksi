// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use core::borrow::Borrow;

use crate::{Expression, Subsystem};

#[inline(always)]
pub(crate) fn calculate_residual(expression: &Expression, variables: &[f64]) -> f64 {
    match expression {
        Expression::VariableVariableEquality(expression) => expression.compute_residual(variables),
        Expression::PointPointDistance(expression) => expression.compute_residual(variables),
        Expression::PointPointPointAngle(expression) => expression.compute_residual(variables),
        Expression::PointLineIncidence(expression) => expression.compute_residual(variables),
        Expression::PointLineDistance(expression) => expression.compute_residual(variables),
        Expression::PointCircleIncidence(expression) => expression.compute_residual(variables),
        Expression::SegmentSegmentLengthEquality(expression) => {
            expression.compute_residual(variables)
        }
        Expression::LineCircleTangency(expression) => expression.compute_residual(variables),
        Expression::LineLineAngle(expression) => expression.compute_residual(variables),
        Expression::LineLineOrthogonality(expression) => expression.compute_residual(variables),
        Expression::LineLineParallelism(expression) => expression.compute_residual(variables),
    }
}

/// Compute residuals for all expressions.
pub(crate) fn calculate_residuals(
    subsystem: &Subsystem<'_>,
    variables: &[f64],
    residuals: &mut [f64],
) {
    residuals.fill(0.);

    for (expression_idx, expression) in subsystem.expressions().enumerate() {
        residuals[expression_idx] = calculate_residual(expression, variables);
    }
}

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

    for (expression_idx, expression) in subsystem.expressions().enumerate() {
        match expression {
            Expression::VariableVariableEquality(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
                );
            }
            Expression::PointPointDistance(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
                );
            }
            Expression::PointPointPointAngle(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
                );
            }
            Expression::PointLineIncidence(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
                );
            }
            Expression::PointLineDistance(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
                );
            }
            Expression::PointCircleIncidence(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
                );
            }
            Expression::SegmentSegmentLengthEquality(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
                );
            }
            Expression::LineCircleTangency(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
                );
            }
            Expression::LineLineAngle(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
                );
            }
            Expression::LineLineOrthogonality(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
                );
            }
            Expression::LineLineParallelism(expression) => {
                expression.compute_residual_and_gradient(
                    subsystem,
                    variables,
                    &mut residuals[expression_idx],
                    &mut jacobian[expression_idx * num_free_variables
                        ..(expression_idx + 1) * num_free_variables],
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
