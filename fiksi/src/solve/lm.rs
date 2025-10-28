// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::f64;

use alloc::vec;

#[cfg(not(feature = "std"))]
use crate::floatfuncs::FloatFuncs;

use super::Problem;

/// The Levenberg-Marquardt solver.
///
/// Solve for the free variables in `variables` minimizing the residuals in the [problem](Problem).
pub(crate) fn levenberg_marquardt<P: Problem>(problem: &mut P, variables: &mut [f64]) {
    debug_assert_eq!(
        problem.num_variables() as usize,
        variables.len(),
        "The number of variables as given by the `Problem` and the slice of variables passed in must match."
    );

    // Levenberg-Marquardt requires solving (J^T J + λ I) δ = J^T r for δ, where r is the
    // vector-valued residual function, J is the Jacobian of r, λ is the damping factor, I is the
    // identity matrix, and δ is the update step of the variables.
    //
    // Here, these calculations are performed with finite precision using floating points. When
    // following these normal equations, the numerical imprecision expected in δ (the "condition
    // number" of (J^T J)) is the square of the condition number of J. In other words, the effect
    // of order of magnitude differences is doubled.
    //
    // Instead of solving for the normal equations, we can equivalently solve for the augmented
    // matrices:
    // [  J        ]     [ r ]
    // [ sqrt(λ) I ] δ = [ 0 ]
    // using QR-decomposition, see e.g, Equation 3.2 of "The Levenberg-Marquardt algorithm:
    // implementations and theory" by J. J. Moré (1997). (Given a matrix A and its QR-decomposition
    // QR = A, the system A x = b can be solved for x as R x = Q^T b, where R is an
    // upper-triangular matrix and thus x can be solved efficiently through back substitution.)
    //
    // This is slower per computation and therefore slower in the well-conditioned case, but it is
    // numerically more stable as we don't form the square of the Jacobian. In numerically
    // ill-conditioned cases, the lack of stability due to forming the square can lead to having to
    // perform many more iterations to converge, as well as failure to converge at all.
    //
    // Further, note the Jacobian J changes once per actual step taken by Levenberg-Marquardt,
    // whereas λ changes potentially many times per step inside the inner loop. To avoid
    // decomposing J multiple times, we can instead decompose QR = J first and apply the orthogonal
    // transformations to the residuals as well, i.e., z = Q^T r. Then, again using
    // QR-decomposition, we can find the solution for the equivalent problem
    // [  R        ]     [ z ]
    // [ sqrt(λ) I ] δ = [ 0 ].
    //
    // Effectively, this performs a partial decomposition of the top part of [J; sqrt(λ)], also
    // storing the effect of the partial orthogonal transformations on the right-hand side (i.e.,
    // the residuals).

    let mut variables_scratch = variables.to_vec();
    let variables_scratch = variables_scratch.as_mut_slice();

    // The (non-squared) residuals of the expressions.
    let mut residuals = nalgebra::DVector::zeros(problem.num_residuals() as usize);
    let mut residuals_scratch = nalgebra::DVector::zeros(problem.num_residuals() as usize);

    // All first-order partial derivatives of the expressions, in row-major order. This is a dense
    // representation (each pair of expression and variable has a partial derivative represented
    // here, even for variables that do not contribute to the expression). It's possible a sparse
    // representation may be more efficient in certain cases.
    let mut jacobian =
        vec![0.; problem.num_residuals() as usize * problem.num_variables() as usize];

    // The same Jacobian, but a separate allocation for the column-major representation, which
    // nalgebra expects. TODO: perhaps make everything use a column-major allocation, as that's
    // somewhat more common (making the per-expression gradients have a stride instead).
    let mut jacobian_ = nalgebra::DMatrix::zeros(
        problem.num_residuals() as usize,
        problem.num_variables() as usize,
    );

    // The (augmented) residual matrix.
    let mut z_aug = nalgebra::DMatrix::zeros(jacobian_.nrows() + jacobian_.ncols(), 1);

    problem.calculate_residuals_and_jacobian(
        &*variables_scratch,
        residuals.as_mut_slice(),
        &mut jacobian,
    );

    let mut residual = residuals.norm();

    let mut lambda = 0.5;
    'steps: for _ in 0..100 {
        if residual < 1e-4 {
            break;
        }

        // Copy from the row-major Jacobian to the column-major Jacobian.
        for i in 0..problem.num_residuals() as usize {
            for j in 0..problem.num_variables() as usize {
                jacobian_[(i, j)] = jacobian[i * problem.num_variables() as usize + j];
            }
        }

        // (It seems that with the current nalgebra API, it's not entirely straightforward to reuse
        // the allocations for `jacobian_` or the inner `qr_r_aug`. The matrix is moved into
        // `ColPivQR`, and moving it out again requires an expensive resizing operation that copies
        // elements around.)
        //
        // We do a column-pivoting QR specifically, for its increased numerical stability. See also
        // the note above about the use of QR decomposition for solving.
        let partial_qr = jacobian_.clone().col_piv_qr();
        // We negate the residuals here, as we want to solve for `(J^T J + λ I) δ = -J^T r`.
        // By negating `r` here, we don't have to negate it each time below.
        residuals.neg_mut();
        partial_qr.q_tr_mul(&mut residuals);
        // Note: we'd like to use qr.unpack_r() here, but there may be some UB in `nalgebra`:
        // <https://github.com/endoli/fiksi/pull/91>.
        let partial_r = partial_qr.r();

        // Inner loop to find a suitable damping factor allowing a step to be accepted.
        loop {
            // The stacked matrix `[R; sqrt(lambda) I]`, where `R` is the upper-triangular matrix
            // from the decomposition `J = QR` we performed above.
            let mut r_aug =
                nalgebra::DMatrix::zeros(jacobian_.nrows() + jacobian_.ncols(), jacobian_.ncols());
            r_aug
                .rows_mut(0, jacobian_.nrows().min(jacobian_.ncols()))
                .copy_from(&partial_r);
            for idx in 0..jacobian_.ncols() {
                r_aug[(jacobian_.nrows() + idx, idx)] += f64::sqrt(lambda);
            }
            let qr = r_aug.qr();
            z_aug.rows_mut(0, jacobian_.nrows()).copy_from(&residuals);
            z_aug
                .rows_mut(jacobian_.nrows(), jacobian_.ncols())
                .fill(0.);
            qr.q_tr_mul(&mut z_aug);

            let mut delta = z_aug.rows_mut(0, jacobian_.ncols());
            // Note: we'd like to use qr.unpack_r() here, but there may be some UB in `nalgebra`:
            if !qr.r().solve_upper_triangular_mut(&mut delta) {
                lambda *= 8.;
                continue;
            };
            partial_qr.p().inv_permute_rows(&mut delta);

            if delta.norm() < 1e-6 {
                break 'steps;
            }

            for idx in 0..problem.num_variables() as usize {
                variables_scratch[idx] = variables[idx] + delta[idx];
            }

            problem.calculate_residuals(variables_scratch, residuals_scratch.as_mut_slice());
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
                problem.calculate_residuals_and_jacobian(
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
