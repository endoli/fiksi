// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::f64;

use alloc::vec;

#[cfg(not(feature = "std"))]
use crate::floatfuncs::FloatFuncs;

use crate::{
    Subsystem,
    utils::{calculate_residuals, calculate_residuals_and_jacobian},
};

/// The Levenberg-Marquardt solver.
///
/// Solve for the free variables in `variables` minimizing the residuals of the constraints in
/// `constraint_set`. The variables given by the elements in `element_set` are seen as free, other
/// variables are seen as fixed parameters.
pub(crate) fn levenberg_marquardt(variables: &mut [f64], subsystem: &Subsystem<'_>) {
    // TODO: this is allocation-happy.
    // TODO: we don't use enough of nalgebra here to justify including that dependency.

    let mut variables_scratch = variables.to_vec();
    let variables_scratch = variables_scratch.as_mut_slice();

    // The (non-squared) residuals of the constraints.
    let mut residuals = nalgebra::DVector::zeros(subsystem.constraints().len());
    let mut residuals_scratch = nalgebra::DVector::zeros(subsystem.constraints().len());

    // All first-order partial derivatives of the constraints. This is a dense representation (each
    // pair of constraint and variable has a partial derivative represented here, even for
    // variables that do not contribute to the constraint). It's possible a sparse representation
    // may be more efficient in certain cases.
    let mut jacobian = vec![0.; subsystem.constraints().len() * subsystem.free_variables().len()];

    calculate_residuals_and_jacobian(
        subsystem,
        variables,
        residuals.as_mut_slice(),
        &mut jacobian,
    );

    let mut residual = residuals.norm();

    let mut lambda = 0.5;
    'steps: for _ in 0..100 {
        if residual < 1e-4 {
            break;
        }

        // Clone the Jacobian to a nalgebra matrix for now.
        // TODO: remove
        let mut jacobian_ = nalgebra::DMatrix::zeros(
            subsystem.constraints().len(),
            subsystem.free_variables().len(),
        );
        for i in 0..subsystem.constraints().len() {
            for j in 0..subsystem.free_variables().len() {
                jacobian_[(i, j)] = jacobian[i * subsystem.free_variables().len() + j];
            }
        }

        // Inner loop to find a suitable damping factor allowing a step to be accepted.
        loop {
            // Levenberg-Marquardt requires solving (J^T J + λ I) δ = J^T r(x) for δ, where r is the
            // vector-valued residual function, J is the Jacobian of r, λ is the damping factor, I is
            // the identity matrix, and δ is the update step of the variables.
            //
            // Here, these calculations are performed with finite precision using floating points. The
            // numerical imprecision expected in δ (the "condition number" of (J^T J)) is the square of
            // the condition number of J. In other words, the effect of order of magnitude differences
            // is doubled.
            //
            // Instead of solving for the above expression, we can equivalently solve for the
            // augmented matrices:
            // [  J        ]     [ r ]
            // [ sqrt(λ) I ] δ = [ 0 ]
            // using QR-decomposition, see e.g, Equation 3.2 of "The Levenberg-Marquardt algorithm:
            // implementations and theory" by J. J. Moré (1997).
            //
            // This is slower per computation and therefore slower in the well-conditioned case, but it
            // is numerically more stable as we don't form the square of the Jacobian. In numerically
            // ill-conditioned cases, the lack of stability due to forming the square can lead to
            // having to perform many more iterations to converge, as well as failure to convergence at
            // all.
            let mut j_aug =
                nalgebra::DMatrix::zeros(jacobian_.nrows() + jacobian_.ncols(), jacobian_.ncols());
            j_aug.rows_mut(0, jacobian_.nrows()).copy_from(&jacobian_);
            // Augment by the damping factor `lambda`.
            for idx in 0..jacobian_.ncols() {
                j_aug[(jacobian_.nrows() + idx, idx)] += f64::sqrt(lambda);
            }
            // The augmented residual matrix:
            // [ r ]
            // [ 0 ]
            let mut r_aug = nalgebra::DMatrix::zeros(jacobian_.nrows() + jacobian_.ncols(), 1);
            r_aug.rows_mut(0, jacobian_.nrows()).copy_from(&residuals);

            let qr = j_aug.col_piv_qr();
            let Some(mut delta) = qr
                .r()
                .solve_upper_triangular(&(-qr.q().transpose() * r_aug))
            else {
                lambda *= 8.;
                continue;
            };

            qr.p().inv_permute_rows(&mut delta);
            if delta.norm() < 1e-6 {
                break 'steps;
            }

            for (idx, variable) in subsystem.free_variables().enumerate() {
                variables_scratch[variable as usize] = variables[variable as usize] + delta[idx];
            }

            calculate_residuals(
                subsystem,
                variables_scratch,
                residuals_scratch.as_mut_slice(),
            );
            let residual_scratch = residuals_scratch.norm();

            if residual_scratch < residual {
                // Accept step
                lambda *= 0.125;
                if lambda < 1e-50 {
                    lambda = 1e-50;
                }
                residual = residual_scratch;
                variables.copy_from_slice(variables_scratch);

                // It might be nice to have a calculate_jacobian function here, but the additional
                // residual calculation shouldn't be too bad.
                calculate_residuals_and_jacobian(
                    subsystem,
                    variables_scratch,
                    residuals.as_mut_slice(),
                    &mut jacobian,
                );
                break;
            } else {
                // Reject step.
                lambda *= 2.;
            }
        }
    }
}
