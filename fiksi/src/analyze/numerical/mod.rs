// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{vec, vec::Vec};

use nalgebra::DMatrix;

use crate::{
    AnyConstraintHandle, Edge, SolveSet, System, Vertex, constraints::ConstraintTag, utils,
};
const EPSILON: f64 = 1e-8;

/// Transform the matrix to reduced row echelon form through Gauss-Jordan elimination.
///
/// Rows which are linearly independent from preceding rows are returned. The number of linearly
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
/// Further, by allowing for changed column ordering, it allows finding dependent columns by
/// (re)starting the elimination with different right-most columns.
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
pub(crate) fn find_overconstraints(
    system: &System,
    solve_set: Option<&SolveSet>,
) -> Vec<AnyConstraintHandle> {
    let mut free_variables: Vec<u32> = vec![];
    for vertex in &system.element_vertices {
        match vertex {
            Vertex::Point { idx } => {
                free_variables.extend(&[*idx, idx + 1]);
            }
            Vertex::Circle { radius_idx, .. } => {
                free_variables.extend(&[*radius_idx]);
            }
            // In the current setup, not all vertices in the set contribute free variables.
            _ => {}
        }
    }
    let num_variables = free_variables.len();
    free_variables.sort_unstable();

    // Map from variable index into free variable index within the Jacobian matrix, gradient
    // vector, etc.
    let mut free_variable_map = alloc::collections::BTreeMap::new();
    for (idx, &free_variable) in free_variables.iter().enumerate() {
        free_variable_map.insert(
            free_variable,
            idx.try_into().expect("less than 2^32 elements"),
        );
    }
    let constraints: Vec<&Edge> = match solve_set {
        Some(solve_set) => solve_set
            .constraints
            .iter()
            .map(|id| &system.constraint_edges[id.id as usize])
            .collect(),
        None => system.constraint_edges.iter().collect(),
    };

    // The (non-squared) residuals of the constraints.
    let mut residuals = vec![0.; constraints.len()];
    // All first-order partial derivatives of the constraints, as constraints x free variables.
    // This is in row-major order.
    let mut jacobian = vec![0.; constraints.len() * num_variables];
    utils::calculate_residuals_and_jacobian(
        &constraints,
        &free_variable_map,
        &system.variables,
        &mut residuals,
        &mut jacobian,
    );

    let mut jacobian_ = nalgebra::DMatrix::zeros(constraints.len(), free_variables.len());
    for i in 0..constraints.len() {
        for j in 0..free_variables.len() {
            jacobian_[(i, j)] = jacobian[i * free_variables.len() + j];
        }
    }

    let mut column_pivots = Vec::from_iter(0..jacobian_.ncols());
    let independent_constraints =
        incremental_gauss_jordan_elimination(&mut jacobian_, &mut column_pivots);

    let mut dependent = vec![];
    for (constraint_idx, independent) in independent_constraints.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "there are fewer than 2^32 constraints"
        )]
        if !independent {
            let id_in_system = solve_set
                .map(|solve_set| {
                    solve_set
                        .constraints
                        .iter()
                        .position(|c| c.id == constraint_idx as u32)
                        .expect("constraint is present") as u32
                })
                .unwrap_or(constraint_idx as u32);
            dependent.push(AnyConstraintHandle::from_ids_and_tag(
                system.id,
                id_in_system,
                ConstraintTag::from(&system.constraint_edges[id_in_system as usize]),
            ));
        }
    }

    dependent
}
