// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::f64;

use alloc::vec;

use crate::{
    Subsystem,
    utils::{calculate_residuals_and_jacobian, sum_squares},
};

/// The limited-memory BFGS solver by Liu and Nocedal (1989), approximating the
/// Broyden–Fletcher–Goldfarb–Shanno method.
///
/// Solve for the free variables in `variables` minimizing the sum of squared residuals of the constraints in
/// `constraint_set`. The variables given by the elements in `element_set` are seen as free, other
/// variables are seen as fixed parameters.
///
/// See:
/// Liu, Dong C., and Jorge Nocedal. "On the limited memory BFGS method for large scale
/// optimization." Mathematical programming 45.1 (1989): 503-528.
pub(crate) fn lbfgs(variables: &mut [f64], subsystem: &Subsystem<'_>) {
    /// The max length of the limited history.
    const MAX_HISTORY: u8 = 5;

    /// The max number of iterations. The algorithm stops after this number of iterations if no
    /// convergence has been achieved beforehand.
    const MAX_ITERATIONS: u8 = 100;

    /// When the change in objective function (here, the sum of squared residuals) is lower than
    /// this, we deem the problem to have converged.
    const CONVERGENCE_THRESHOLD: f64 = 1e-10;

    /// We know we're optimizing for the sum of squared residuals. If it's close enough to 0, we
    /// have converged.
    const RESIDUAL_THRESHOLD: f64 = 1e-6;

    // The (non-squared) residuals of the constraints.
    let mut residuals = vec![0.; subsystem.constraints().len()];
    // All first-order partial derivatives of the constraints, as constraints x free variables.
    // This is in row-major order.
    let mut jacobian = vec![0.; subsystem.constraints().len() * subsystem.free_variables().len()];

    // Calculate initial residuals and gradients
    calculate_residuals_and_jacobian(subsystem, &*variables, &mut residuals, &mut jacobian);

    let mut prev_sum_squared_residuals = sum_squares(&residuals);
    if prev_sum_squared_residuals < 1e-4 {
        return;
    }

    let num_variables = subsystem.free_variables().len();
    // Gradient (`g_k`) scratch buffer.
    let mut gradient = vec![0.; num_variables];
    // Calculate initial gradient
    compute_gradient(&jacobian, &residuals, &mut gradient);

    // `MAX_HISTORY`-sized history storage for `s_k`, `y_k` and `ρ_k`.
    //
    // These are handled as ringbuffers in the algorithm below.
    let mut s_history = vec![0.; num_variables * MAX_HISTORY as usize];
    let mut y_history = vec![0.; num_variables * MAX_HISTORY as usize];
    let mut rho_history = vec![0.; MAX_HISTORY as usize];

    // `MAX_HISTORY`-sized scratch buffer for `α` (not a ringbuffer, this is updated every iteration).
    let mut alpha = vec![0.; MAX_HISTORY as usize];

    // Reusable scratch buffers for update step direction and variables.
    let mut direction = vec![0.; num_variables];
    let mut variables_scratch = vec![0.; num_variables];

    for k in 0..MAX_ITERATIONS {
        let history_len = u8::min(k, MAX_HISTORY);

        // The two-loop recursion algorithm for implicitly calculating the step direction (inverse
        // Hessian times gradient), see Algorithm 7.4 in Nocedal and Wright (2006).
        direction.copy_from_slice(&gradient);
        for i in (0..history_len).rev() {
            let history_idx = usize::from((k + i) % MAX_HISTORY);

            let s_i = &s_history[history_idx * num_variables..(history_idx + 1) * num_variables];
            let y_i = &y_history[history_idx * num_variables..(history_idx + 1) * num_variables];
            let rho_i = rho_history[history_idx];

            let mut dot_product = 0.;
            for j in 0..num_variables {
                dot_product += s_i[j] * direction[j];
            }

            alpha[usize::from(i)] = rho_i * dot_product;
            for j in 0..num_variables {
                direction[j] -= alpha[usize::from(i)] * y_i[j];
            }
        }

        if k > 0 {
            let history_prev_idx = usize::from((k - 1) % MAX_HISTORY);
            let s_prev = &s_history
                [history_prev_idx * num_variables..(history_prev_idx + 1) * num_variables];
            let y_prev = &y_history
                [history_prev_idx * num_variables..(history_prev_idx + 1) * num_variables];

            let mut s_dot_y = 0.;
            let mut y_dot_y = 0.;
            for j in 0..num_variables {
                s_dot_y += s_prev[j] * y_prev[j];
                y_dot_y += y_prev[j] * y_prev[j];
            }

            if y_dot_y > 0. {
                let scale = s_dot_y / y_dot_y;
                for d in &mut direction {
                    *d *= scale;
                }
            }
        }

        for i in 0..history_len {
            let history_idx = usize::from((k + i) % MAX_HISTORY);

            let s_i = &s_history[history_idx * num_variables..(history_idx + 1) * num_variables];
            let y_i = &y_history[history_idx * num_variables..(history_idx + 1) * num_variables];
            let rho_i = rho_history[history_idx];

            let mut dot_product = 0.;
            for j in 0..num_variables {
                dot_product += y_i[j] * direction[j];
            }

            let beta = rho_i * dot_product;
            for j in 0..num_variables {
                direction[j] += s_i[j] * (alpha[usize::from(i)] - beta);
            }
        }

        for d in &mut direction {
            *d *= -1.;
        }

        let history_idx = usize::from(k % MAX_HISTORY);

        // Store the old gradient in y_k, we'll use it in the y_k calculation and will be
        // overwriting the gradient buffer below.
        y_history[history_idx * num_variables..(history_idx + 1) * num_variables]
            .copy_from_slice(&gradient);

        variables_scratch.copy_from_slice(variables);
        // When the line search returns with `step_size`, the buffers `jacobian`, `residuals`, and
        // `gradient`, are filled with the values at `f(x + step_size * direction)`.
        let hager_zhang::Param {
            p: step_size,
            phi: sum_squared_residuals,
            ..
        } = hager_zhang::line_search(
            subsystem,
            variables,
            &mut variables_scratch,
            &mut jacobian,
            &mut residuals,
            prev_sum_squared_residuals,
            &mut gradient,
            &direction,
        );

        // Update variables.
        variables.copy_from_slice(&variables_scratch);

        // Compute rho_k = 1 / (y_k^T * s_k)
        let mut s_dot_y = 0.;
        for i in 0..num_variables {
            let idx = history_idx * num_variables + i;

            s_history[idx] = step_size * direction[i];
            y_history[idx] = gradient[i] - y_history[idx];
            s_dot_y += s_history[idx] * y_history[idx];
        }
        let rho_k = 1.0 / s_dot_y;
        rho_history[history_idx] = rho_k;

        if (prev_sum_squared_residuals - sum_squared_residuals).abs() < CONVERGENCE_THRESHOLD {
            break;
        }
        if sum_squared_residuals < RESIDUAL_THRESHOLD {
            break;
        }
        prev_sum_squared_residuals = sum_squared_residuals;
    }
}

/// Compute the sum of squared residual gradient from the residual vector and Jacobian: `J^T * r`.
///
/// Specifically, the gradient is ∇f(x) = ∇½||r||^2.
#[inline]
fn compute_gradient(jacobian: &[f64], residuals: &[f64], gradient: &mut [f64]) {
    gradient.fill(0.0);

    let num_variables = gradient.len();
    let num_constraints = residuals.len();

    for i in 0..num_variables {
        for c in 0..num_constraints {
            gradient[i] += jacobian[c * num_variables + i] * residuals[c];
        }
    }
}

/// Calculate dot product of two vectors
#[inline]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

mod hager_zhang {
    use crate::{
        Subsystem,
        utils::{calculate_residuals_and_jacobian, sum_squares},
    };

    use super::{compute_gradient, dot_product};

    /// Parameter for the first Wolfe condition, aka Armijo or sufficient descent, sometimes called `c1`.
    const DELTA: f64 = 1e-4;
    /// Parameter for the second Wolfe condition, to uphold curvature invariants, sometimes known
    /// as `c2`.
    const SIGMA: f64 = 0.9;

    /// Approximate Wolfe termination.
    const EPSILON: f64 = 1e-6;

    /// ½ for bisection.
    const THETA: f64 = 0.5;

    /// If secant2 does not shrink the interval by this factor, bisection is performed.
    const GAMMA: f64 = 0.66;

    /// Initial bracket expansion, from the initial bracketing routine defined in
    /// Hager, William W., and Hongchao Zhang. "Algorithm 851: `CG_DESCENT`, a conjugate gradient
    /// method with guaranteed descent." ACM Transactions on Mathematical Software (TOMS) 32.1
    /// (2006): 113-137.
    const RHO: f64 = 5.;

    const MAX_ITERATIONS: u8 = 100;

    /// A parameter to the ϕ (phi) function along with its evaluated function value and first
    /// derivative.
    #[derive(Clone, Copy)]
    pub(super) struct Param {
        /// The parameter value itself.
        pub p: f64,
        /// ϕ(p), i.e., the objective function evaluated at [`Self::p`].
        pub phi: f64,
        /// ϕ'(p), i.e., the objective function derivative evaluated at [`Self::p`].
        pub dphi: f64,
    }

    /// Helper struct for evaluating the system's residual and gradient.
    struct Eval<'a> {
        subsystem: &'a Subsystem<'a>,
        variables: &'a [f64],
        variables_scratch: &'a mut [f64],
        jacobian: &'a mut [f64],
        residuals: &'a mut [f64],
        gradient: &'a mut [f64],
        direction: &'a [f64],
    }

    impl Eval<'_> {
        fn calculate_phi(&mut self, p: f64) -> Param {
            for (idx, d) in self.subsystem.free_variables().zip(self.direction) {
                self.variables_scratch[idx as usize] = self.variables[idx as usize] + p * d;
            }

            calculate_residuals_and_jacobian(
                self.subsystem,
                self.variables_scratch,
                self.residuals,
                self.jacobian,
            );
            compute_gradient(self.jacobian, self.residuals, self.gradient);
            let phi = sum_squares(&*self.residuals);
            let dphi = dot_product(self.gradient, self.direction);

            Param { p, phi, dphi }
        }
    }

    /// A secant step for stepping towards the phi' root, as used in Hager and Zhang (2005).
    #[inline(always)]
    fn secant(a: Param, b: Param) -> f64 {
        (a.p * b.dphi - b.p * a.dphi) / (b.dphi - a.dphi)
    }

    enum HzResult<T> {
        Satisfied(Param),
        Unsatisfied(T),
    }

    struct HagerZhangLineSearch {
        phi0: f64,
        dphi0: f64,
    }

    impl HagerZhangLineSearch {
        /// Whether `c` satisfies the strong or approximate Wolfe conditions.
        fn satisfies_wolfe(&self, c: Param) -> bool {
            // Strong Wolfe conditions.
            if (c.phi <= self.phi0 + c.p * (DELTA * self.dphi0)) && (c.dphi >= SIGMA * self.dphi0) {
                return true;
            }

            // Approximate Wolfe condition (Hager and Zhang, 2005).
            if c.phi <= self.phi0 + EPSILON
                && (2. * DELTA - 1.) * self.dphi0 >= c.dphi
                && c.dphi >= SIGMA * self.dphi0
            {
                return true;
            }

            false
        }

        /// Perform the `update` step as defined in Hager and Zhang (2005).
        fn update(&self, eval: &mut Eval<'_>, a: Param, b: Param, c: Param) -> (Param, Param) {
            if c.p < a.p || c.p > b.p {
                // U0 in the paper
                return (a, b);
            }

            if c.dphi >= 0. {
                // U1 in the paper
                (a, c)
            } else if c.phi <= self.phi0 + EPSILON {
                // U2 in the paper
                (c, b)
            } else {
                // U3 in the paper
                let mut a = a;
                let mut b = c;
                loop {
                    let d = {
                        let p = (1. - THETA) * a.p + THETA * b.p;
                        eval.calculate_phi(p)
                    };
                    if d.dphi >= 0. {
                        return (a, d);
                    } else if d.phi <= self.phi0 + EPSILON {
                        a = d;
                    } else {
                        b = d;
                    }
                }
            }
        }

        /// The "double Secant" step of Hager and Zhang (2005), calculating a new bracketing
        /// `[a, b]` for the step size.
        ///
        /// Returns `HzResult::Satisfied(param)` if a point was found satisfying the Wolfe
        /// conditions. Returns `HzResult::Unsatisfied((a,b))` bracketing the point otherwise.
        fn secant2(&self, eval: &mut Eval<'_>, a: Param, b: Param) -> HzResult<(Param, Param)> {
            let c = {
                let p = secant(a, b);
                eval.calculate_phi(p)
            };
            if self.satisfies_wolfe(c) {
                return HzResult::Satisfied(c);
            }

            let (a_, b_) = self.update(eval, a, b, c);
            // TODO: optimize branches away by inlining `update`?
            if c.p == b_.p {
                let c_ = {
                    let p = secant(b, b_);
                    eval.calculate_phi(p)
                };
                if self.satisfies_wolfe(c_) {
                    return HzResult::Satisfied(c_);
                }
                HzResult::Unsatisfied(self.update(eval, a_, b_, c_))
            } else if c.p == a_.p {
                let c_ = {
                    let p = secant(a, a_);
                    eval.calculate_phi(p)
                };
                if self.satisfies_wolfe(c_) {
                    return HzResult::Satisfied(c_);
                }
                HzResult::Unsatisfied(self.update(eval, a_, b_, c_))
            } else {
                HzResult::Unsatisfied((a_, b_))
            }
        }

        /// Find an initial bracket around `c` that satisfies the Wolfe conditions within its
        /// range.
        fn bracket(&self, eval: &mut Eval<'_>, _c: Param) -> (Param, Param) {
            // TODO: implement the initial bracketing. Set to [0., 5.] for now.
            let a = Param {
                p: 0.,
                phi: self.phi0,
                dphi: self.dphi0,
            };
            let b = eval.calculate_phi(5.);
            (a, b)
        }

        /// The main search loop, updating the bracket and bisecting until a step size is found
        /// satisfying the Wolfe conditions.
        fn search(&self, eval: &mut Eval<'_>, a: Param, b: Param, c: Param) -> Param {
            let (mut a, mut b, mut c) = (a, b, c);

            for _ in 0..MAX_ITERATIONS {
                let (a_, b_) = match self.secant2(eval, a, b) {
                    HzResult::Satisfied(c) => return c,
                    HzResult::Unsatisfied((a_, b_)) => (a_, b_),
                };

                // If the bracket did not shrink enough, bisect.
                if b_.p - a_.p > GAMMA * (b.p - a.p) {
                    c = eval.calculate_phi(0.5 * (a.p + b.p));
                    if self.satisfies_wolfe(c) {
                        return c;
                    }

                    (a, b) = self.update(eval, a, b, c);
                } else {
                    (a, b) = (a_, b_);
                }
            }

            // Evaluate again to satisfy the guarantee that [`Eval::calculate_phi`] was last called
            // with parameter [`Param::p`].
            eval.calculate_phi(c.p);
            c
        }

        /// This guarantees [`Eval::calculate_phi`] was last called with parameter [`Param::p`].
        fn run(&self, eval: &mut Eval<'_>) -> Param {
            // For the L-BFGS routine as above, a step size of `1` is often accepted. In that case,
            // we can exit early.
            let c = eval.calculate_phi(1.);
            if self.satisfies_wolfe(c) {
                return c;
            }

            let (a, b) = self.bracket(eval, c);
            self.search(eval, a, b, c)
        }
    }

    /// Perform line search using the Hager-Zhang method.
    ///
    /// This finds the step size `alpha` to perform the variable update step
    /// `x <- x + alpha * direction`, with `alpha` such that the [Wolfe] conditions are satisfied.
    ///
    /// When this search returns with the found step size, the buffers `jacobian`, `residuals`, and
    /// `gradient`, are filled with the values at `f(x + alpha * direction)`.
    ///
    /// See the main paper:
    /// Hager, William W., and Hongchao Zhang. "A new conjugate gradient method with guaranteed
    /// descent and an efficient line search." SIAM Journal on optimization 16.1 (2005): 170-192.
    ///
    /// And the bracket initialization in:
    /// Hager, William W., and Hongchao Zhang. "Algorithm 851: `CG_DESCENT`, a conjugate gradient
    /// method with guaranteed descent." ACM Transactions on Mathematical Software (TOMS) 32.1
    /// (2006): 113-137.
    ///
    /// [Wolfe]: https://en.wikipedia.org/wiki/Wolfe_conditions
    pub(super) fn line_search(
        subsystem: &Subsystem<'_>,
        variables: &[f64],
        variables_scratch: &mut [f64],
        jacobian: &mut [f64],
        residuals: &mut [f64],
        sum_squared_residuals: f64,
        gradient: &mut [f64],
        direction: &[f64],
    ) -> Param {
        let phi0 = sum_squared_residuals;
        let dphi0 = dot_product(&*gradient, direction);
        let hz = HagerZhangLineSearch { phi0, dphi0 };

        let eval = &mut Eval {
            subsystem,
            variables,
            variables_scratch,
            jacobian,
            residuals,
            gradient,
            direction,
        };
        hz.run(eval)
    }
}
