// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::f64;

use alloc::{vec, vec::Vec};

use crate::{ConstraintId, Edge, ElementId, Vertex, utils::calculate_residuals_and_jacobian};

/// The Levenberg-Marquardt solver.
///
/// Solve for the free variables in `variables` minimizing the residuals of the constraints in
/// `constraint_set`. The variables given by the elements in `element_set` are seen as free, other
/// variables are seen as fixed parameters.
pub(crate) fn levenberg_marquardt(
    variables: &mut [f64],
    // TODO: actually use `element_set`
    _element_set: &[ElementId],
    constraint_set: &[ConstraintId],
    element_vertices: &[Vertex],
    constraint_edges: &[Edge],
) {
    // TODO: this is allocation-happy.
    // TODO: we don't use enough of nalgebra here to justify including that dependency.

    let mut lambda = 10.;
    let mut prev_residual = f64::INFINITY;

    let mut free_variables: Vec<u32> = vec![];
    for vertex in element_vertices {
        match vertex {
            Vertex::Point { idx } => {
                free_variables.extend(&[*idx, idx + 1]);
            }
            // In the current setup, not all vertices in the set contribute free variables. E.g.
            // `Vertex::Line` only refers to existing points, meaning it does not contribute its
            // own free variables. `Vertex::Circle` refers to a point and has a radius as free
            // variable.
            Vertex::Circle { radius_idx, .. } => {
                free_variables.extend(&[*radius_idx]);
            }
            _ => {}
        }
    }
    free_variables.sort_unstable();
    let mut free_variable_map = alloc::collections::BTreeMap::new();
    for (idx, &free_variable) in free_variables.iter().enumerate() {
        free_variable_map.insert(
            free_variable,
            idx.try_into().expect("less than 2^32 elements"),
        );
    }

    let constraints: Vec<&Edge> = constraint_set
        .iter()
        .map(|id| &constraint_edges[id.id as usize])
        .collect();

    // The (non-squared) residuals of the constraints.
    let mut residuals = vec![0.; constraints.len()];
    // All first-order partial derivatives of the constraints. This is a dense representation (each
    // pair of constraint and variable has a partial derivative represented here, even for
    // variables that do not contribute to the constraint). It's possible a sparse representation
    // may be more efficient in certain cases.
    let mut jacobian = vec![0.; constraints.len() * free_variables.len()];

    // J^T * J square matrix.
    // let mut jtj = vec![0.; free_variables.len() * free_variables.len()];

    for _ in 0..100 {
        calculate_residuals_and_jacobian(
            &constraints,
            &free_variable_map,
            variables,
            &mut residuals,
            &mut jacobian,
        );

        let residuals_ = nalgebra::DVector::from_column_slice(&residuals);
        let residual = residuals_.norm();

        // TODO: revisit stopping conditions and the `lambda` schedule.
        if residual < 1e-4 {
            break;
        }

        if residual < prev_residual {
            lambda *= 0.5;
            if lambda < 1e-10 {
                lambda = 1e-10;
            }
        } else {
            lambda *= 2.;
        }
        prev_residual = residual;

        // Clone the Jacobian to a nalgebra matrix for now.
        // TODO: remove
        let mut jacobian_ = nalgebra::DMatrix::zeros(constraints.len(), free_variables.len());
        for i in 0..constraints.len() {
            for j in 0..free_variables.len() {
                jacobian_[(i, j)] = jacobian[i * free_variables.len() + j];
            }
        }

        // calculate_jtj(&jacobian, &mut jtj, constraints.len(), free_variables.len());
        // Augment JTJ by the damping factor `lambda`
        // for i in 0..free_variables.len() {
        //     jtj[i * free_variables.len() + i] += lambda;
        // }

        // Calculate J^T J and augment by the damping factor `lambda`.
        let jtj_: nalgebra::DMatrix<f64> = jacobian_.transpose() * (&jacobian_)
            + lambda * nalgebra::DMatrix::identity(free_variables.len(), free_variables.len());

        let gradient = -jacobian_.transpose() * residuals_;

        if let Some(delta) = jtj_.clone().lu().solve(&gradient) {
            if delta.norm() < 1e-6 {
                break;
            }
            for (idx, variable) in free_variables.iter().enumerate() {
                variables[*variable as usize] += delta[idx];
            }
        } else {
            panic!("Levenberg-Marquardt: failed to solve linear system");
        }
    }
}

// /// Calculate the J^T * J square matrix.
// fn calculate_jtj(jacobian: &[f64], jtj: &mut [f64], constraints: usize, variables: usize) {
//     for i in 0..variables {
//         for j in 0..variables {
//             let idx = i * variables + j;
//             jtj[idx] = 0.;
//
//             for c in 0..constraints {
//                 jtj[idx] += jacobian[c * variables + i] * jacobian[c * variables + j];
//             }
//         }
//     }
// }
