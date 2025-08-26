// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{vec, vec::Vec};

use nalgebra::DMatrix;

use crate::{AnyConstraintHandle, System, VariableMap, collections::IndexSet};

const EPSILON: f64 = 1e-8;

/// Transform the matrix to reduced row echelon form through Gauss-Jordan elimination.
///
/// Rows that are linearly independent from preceding rows are returned. The number of linearly
/// independent rows is the rank of the matrix.
///
/// This can change the order of the matrix columns (see the paragraph below). Rather than swapping
/// around the matrix columns, the order is kept track of in `column_indices`, mapping from the
/// index in the resulting matrix to the original column index. This function does not assume any
/// particular starting order, i.e., `column_indices` does not have to be `0..num_columns` (it does
/// need to be some permutation of that).
///
/// This differs from "classical" Gauss-Jordan elimination in that it proceeds row-by-row. This
/// allows efficiently computing which rows are dependent (i.e., which rows are a linear
/// combination of other rows). To this end, rather than swapping rows, columns are swapped.
///
/// Further, by allowing for changed column ordering, it allows finding dependent columns by
/// (re)starting the elimination with different right-most columns. See Section 4 of "Using the
/// witness method to detect rigid subsystems of geometric constraints in CAD" (2010) by Michelucci
/// et al.
///
/// In the above, within the context of Fiksi, "rows" are constraints and "columns" are elements.
pub(crate) fn incremental_gauss_jordan_elimination(
    matrix: &mut DMatrix<f64>,
    column_indices: &mut [usize],
) -> Vec<bool> {
    let constraints = matrix.nrows();
    let variables = matrix.ncols();

    #[cfg(debug_assertions)]
    {
        let mut column_indices = column_indices.to_vec();
        column_indices.sort_unstable();
        column_indices.dedup();
        assert_eq!(
            column_indices,
            Vec::from_iter(0..variables),
            "`column_indices` must contain exactly all column indices (in any order): nothing more, nothing less"
        );
    }

    let mut constraint_increases_rank: Vec<bool> = vec![false; constraints];

    let mut current_col = 0;

    for row in 0..usize::min(constraints, variables) {
        let mut rank = 0;
        for row_idx in 0..row {
            // Only independent rows (i.e., nonzero rows, which increase the rank), take up a pivot
            // column. Hence we can index by rank.
            let column_idx = column_indices[rank];
            let factor = matrix[(row, column_idx)];
            for col in 0..variables {
                matrix[(row, col)] -= factor * matrix[(row_idx, col)];
            }
            if constraint_increases_rank[row_idx] {
                rank += 1;
            }
        }

        // first non-zero value in the row
        // TODO: find absolute largest element to improve numerical stability?
        let mut pivot_found = false;
        for idx in current_col..variables {
            let real_idx = column_indices[idx];
            if matrix[(row, real_idx)].abs() > EPSILON {
                column_indices.swap(current_col, idx);
                pivot_found = true;
                break;
            }
        }

        // Row is all-zero (i.e., dependent).
        if !pivot_found {
            continue;
        }

        let factor = matrix[(row, column_indices[current_col])];
        for col in 0..variables {
            matrix[(row, col)] *= 1. / factor;
        }

        // The above brings the matrix into echelon form. Back-substitute to get *reduced* row
        // echelon form.
        let column_idx = column_indices[current_col];
        for row_idx in 0..row {
            let factor = matrix[(row_idx, column_idx)];
            for col in 0..variables {
                matrix[(row_idx, col)] -= factor * matrix[(row, col)];
            }
        }

        // Note `current_col` is also the rank of the matrix up to the current row.
        current_col += 1;
        constraint_increases_rank[row] = true;
    }

    constraint_increases_rank
}

/// Find constraints causing the system to be overconstrained.
///
/// Note if, e.g., two distance constraints together cause the system to be overconstrained, either
/// one of those could be designated as causing the system to become overconstrained.
pub(crate) fn find_overconstraints(system: &System) -> Vec<AnyConstraintHandle> {
    // TODO: this currently assumes all variables are free.

    // The (non-squared) residuals of the constraints' expressions.
    let mut residuals = vec![0.; system.expressions.len()];
    // All first-order partial derivatives of the constraints' expressions, as expressions x free
    // variables. This is in row-major order.
    let mut jacobian = vec![0.; system.expressions.len() * system.variables.len()];

    // All variables are free.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "We don't allow this many variables."
    )]
    let variable_map = VariableMap {
        free_variables: &IndexSet::from_iter(0..system.variables.len() as u32),
        variable_values: &system.variables,
        free_variable_values: &system.variables,
    };
    for (row, expression) in system.expressions.iter().enumerate() {
        let gradient =
            &mut jacobian[row * system.variables.len()..(row + 1) * system.variables.len()];
        residuals[row] = expression.calculate_residual_and_gradient(variable_map, gradient);
    }

    let mut jacobian_ = nalgebra::DMatrix::zeros(system.expressions.len(), system.variables.len());
    for i in 0..system.expressions.len() {
        for j in 0..system.variables.len() {
            jacobian_[(i, j)] = jacobian[i * system.variables.len() + j];
        }
    }

    let mut column_pivots = Vec::from_iter(0..jacobian_.ncols());
    let independent_expressions =
        incremental_gauss_jordan_elimination(&mut jacobian_, &mut column_pivots);

    let mut dependent = vec![];
    for (expression_idx, independent) in independent_expressions.iter().enumerate() {
        if !independent {
            let constraint = system.expression_to_constraint[expression_idx];
            dependent.push(AnyConstraintHandle::from_ids_and_tag(
                system.id,
                constraint.id,
                system.constraints[constraint.id as usize].tag,
            ));
        }
    }

    dependent
}
