// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::f64;

use alloc::vec;

#[cfg(not(feature = "std"))]
use crate::floatfuncs::FloatFuncs;

use solvi::{
    SparseColMat, TripletMat,
    decomposition::sparse::qr::{QrOrdering, SymbolicQr},
};

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
    // decomposing J multiple times, we could instead decompose QR = J first and apply the orthogonal
    // transformations to the residuals as well, i.e., z = Q^T r. Then, again using
    // QR-decomposition, we can find the solution for the equivalent problem
    // [  R        ]     [ z ]
    // [ sqrt(λ) I ] δ = [ 0 ].
    //
    // Effectively, this performs a partial decomposition of the top part of [J; sqrt(λ)], also
    // storing the effect of the partial orthogonal transformations on the right-hand side (i.e.,
    // the residuals). The bottom diagonal matrix `sqrt(λ) I` can be folded in to `R` quickly.
    // However, as this optimization only becomes useful for larger problems and cannot be used for
    // the column-based sparse decomposition we apply, we do not use the optimization here.

    let mut variables_scratch = variables.to_vec();
    let variables_scratch = variables_scratch.as_mut_slice();

    let nrows = problem.num_residuals() as usize;
    let ncols = problem.num_variables() as usize;

    // The (non-squared) residuals of the expressions.
    let mut residuals = nalgebra::DVector::zeros(nrows);
    let mut residuals_scratch = nalgebra::DVector::zeros(nrows);
    let mut b_augmented = vec![0.; nrows + ncols];

    // All first-order partial derivatives of the expressions, in column-major order. This is a
    // sparse representation (exactly all pairs of expressions and variables that are structurally
    // non-zero are represented by an entry). A dense representation will be more efficient for
    // small problems.
    let mut sparse_jacobian = TripletMat::<f64>::new(nrows, ncols);
    problem.calculate_residuals_and_sparse_jacobian(
        &*variables_scratch,
        residuals.as_mut_slice(),
        &mut sparse_jacobian,
    );
    residuals.neg_mut();

    // Add entries for the stacked diagonal matrix `sqrt(λ) I` to augment the matrix for damped
    // least squares.
    for idx in 0..ncols {
        // Damping factors are initially zero. Actual values are set (over and over) in the inner
        // loop.
        sparse_jacobian.push_triplet(nrows + idx, idx, 0.);
    }

    let mut sparse_jacobian_csc = SparseColMat::from_triplet_mat(&sparse_jacobian);

    // Perform symbolic sparse QR-decomposition on our augmented Jacobian matrix. This only takes
    // the structure of the augmented Jacobian into account. It finds an efficient column ordering
    // and precomputes the structure of the numeric work that needs to be done in the inner loop.
    let sparse_sqr = SymbolicQr::build(sparse_jacobian_csc.structure(), QrOrdering::Colamd);
    let mut sparse_qr = sparse_sqr.numeric();

    let mut sum_squared_residuals = residuals.norm_squared();

    let mut lambda = 0.5;
    'steps: for _ in 0..100 {
        if sum_squared_residuals < 1e-8 {
            break;
        }

        // Inner loop to find a suitable damping factor allowing a step to be accepted.
        loop {
            // Set the current damping factors. Note each damping entry is the bottom-most non-zero
            // row of each column (as the diagonal damping matrix is stacked below the Jacobian
            // proper).
            for idx in 0..ncols {
                *sparse_jacobian_csc
                    .index_column_mut(idx)
                    .0
                    .last_mut()
                    .unwrap() = f64::sqrt(lambda);
            }

            // Perform the numerical QR-decomposition of the augmented matrix.
            sparse_qr.factorize(&sparse_jacobian_csc);

            b_augmented[..nrows].copy_from_slice(residuals.as_slice());
            b_augmented[nrows..].fill(0.);
            let solved = sparse_qr.solve_mut(&mut b_augmented);

            if !solved {
                lambda *= 8.;
                continue;
            };

            let delta = nalgebra::DVectorView::from_slice(&b_augmented[..ncols], ncols);
            if delta.norm_squared() < 1e-12 {
                break 'steps;
            }

            for idx in 0..problem.num_variables() as usize {
                variables_scratch[idx] = variables[idx] + delta[idx];
            }

            problem.calculate_residuals(variables_scratch, residuals_scratch.as_mut_slice());
            let sum_squared_residuals_scratch = residuals_scratch.norm_squared();

            if sum_squared_residuals_scratch < sum_squared_residuals {
                // Accept step
                lambda *= 0.125;
                if lambda < 1e-50 {
                    lambda = 1e-50;
                }
                variables.copy_from_slice(variables_scratch);

                // Exit based on the "function tolerance": if the relative change in the sum
                // squared residuals is low, we assume we have converged. Note we may not have
                // converged to an actual root (i.e. a residual of zero) because a root might not
                // exist, or we may have converged to a local optimum, or the solver may simply
                // have stalled.
                if (sum_squared_residuals - sum_squared_residuals_scratch) / sum_squared_residuals
                    <= 1e-6
                {
                    break 'steps;
                }
                sum_squared_residuals = sum_squared_residuals_scratch;

                // It might be nice to have a calculate_jacobian function here, but the additional
                // residual calculation shouldn't be too bad.
                sparse_jacobian.clear();
                problem.calculate_residuals_and_sparse_jacobian(
                    &*variables_scratch,
                    residuals.as_mut_slice(),
                    &mut sparse_jacobian,
                );
                residuals.neg_mut();
                for idx in 0..ncols {
                    sparse_jacobian.push_triplet(nrows + idx, idx, 0.);
                }
                sparse_jacobian_csc = SparseColMat::from_triplet_mat(&sparse_jacobian);
                break;
            } else {
                // Reject step.
                lambda *= 2.;
            }
        }
    }
}
