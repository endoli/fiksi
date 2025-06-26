// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::f64;

use alloc::{vec, vec::Vec};

use crate::{
    ConstraintId, Edge, ElementId, Vertex,
    constraints::{PointPointDistance_, PointPointPointAngle_},
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
pub(crate) fn lbfgs(
    variables: &mut [f64],
    // TODO: actually use `element_set`
    _element_set: &[ElementId],
    constraint_set: &[ConstraintId],
    element_vertices: &[Vertex],
    constraint_edges: &[Edge],
) {
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

    // Identify free variables. This is a map from the free variable index to the `variables`
    // index.
    let mut free_variables: Vec<u32> = vec![];
    for vertex in element_vertices {
        #[expect(clippy::single_match, reason = "more to follow")]
        match vertex {
            Vertex::Point { idx } => {
                free_variables.extend(&[*idx, idx + 1]);
            }
            // In the current setup, not all vertices in the set contribute free variables.
            _ => {}
        }
    }
    let num_variables = free_variables.len();
    free_variables.sort_unstable();

    // Map from variable index into free variable index within the Jacobian matrix, gradient
    // vector, etc.
    let mut index_map = alloc::collections::BTreeMap::new();
    for (idx, &free_variable) in free_variables.iter().enumerate() {
        index_map.insert(
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
    // All first-order partial derivatives of the constraints, as constraints x free variables.
    // This is in row-major order.
    let mut jacobian = vec![0.; constraints.len() * num_variables];

    // Calculate initial residuals and gradients
    compute_residuals_and_jacobian(
        &constraints,
        &index_map,
        variables,
        &mut residuals,
        &mut jacobian,
        num_variables,
    );

    let mut prev_residual_sum = sum_squared_residuals(&residuals);
    if prev_residual_sum < 1e-4 {
        return;
    }

    // Gradient (`g_k`) scratch buffer.
    let mut gradient = vec![0.; num_variables];
    // Calculate initial gradient
    compute_gradient(
        &jacobian,
        &residuals,
        &mut gradient,
        constraints.len(),
        num_variables,
    );

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
        let step_size_ = hager_zhang::line_search(
            &constraints,
            &free_variables,
            &index_map,
            variables,
            &mut variables_scratch,
            constraints.len(),
            num_variables,
            &mut jacobian,
            &mut residuals,
            &mut gradient,
            &direction,
        );
        let step_size = step_size_.p;

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

        let residual_sum = sum_squared_residuals(&residuals);
        if (prev_residual_sum - residual_sum).abs() < CONVERGENCE_THRESHOLD {
            break;
        }
        if residual_sum < RESIDUAL_THRESHOLD {
            break;
        }
        prev_residual_sum = residual_sum;
    }
}

/// Compute residuals and Jacobian for all constraints.
///
/// The Jacobian is relative to the free variables.
fn compute_residuals_and_jacobian(
    constraints: &[&Edge],
    index_map: &alloc::collections::BTreeMap<u32, u32>,
    variables: &[f64],
    residuals: &mut [f64],
    jacobian: &mut [f64],
    num_variables: usize,
) {
    jacobian.fill(0.);
    residuals.fill(0.);

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
                    &mut jacobian
                        [constraint_idx * num_variables..(constraint_idx + 1) * num_variables],
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
                    &mut jacobian
                        [constraint_idx * num_variables..(constraint_idx + 1) * num_variables],
                );
            }
            Edge::LineLineAngle { .. } => {
                unimplemented!()
            }
        }
    }
}

/// Compute the sum of squared residual gradient from the residual vector and Jacobian: `J^T * r`.
///
/// Specifically, the gradient is ∇f(x) = ∇½||r||^2.
#[inline]
fn compute_gradient(
    jacobian: &[f64],
    residuals: &[f64],
    gradient: &mut [f64],
    num_constraints: usize,
    num_variables: usize,
) {
    gradient.fill(0.0);

    for i in 0..num_variables {
        for c in 0..num_constraints {
            gradient[i] += jacobian[c * num_variables + i] * residuals[c];
        }
    }
}

#[inline]
fn sum_squared_residuals(residuals: &[f64]) -> f64 {
    residuals.iter().map(|r| r * r).sum()
}

/// Calculate dot product of two vectors
#[inline]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

mod hager_zhang {
    use crate::Edge;

    use super::{
        compute_gradient, compute_residuals_and_jacobian, dot_product, sum_squared_residuals,
    };

    /// Parameter for the first Wolfe condition, aka Armijo or sufficient descent, sometimes called `c1`.
    const DELTA: f64 = 1e-4;
    /// Parameter for the second Wolfe condition, to uphold curvature invariants, sometimes known
    /// as `c2`.
    const SIGMA: f64 = 0.9;

    /// Approximate Wolfe termination.
    const EPSILON: f64 = 1e-6;

    /// ½ for bisection.
    const THETA: f64 = 0.5;
    // const ETA: f64 = 1.;

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
        constraints: &'a [&'a Edge],
        free_variables: &'a [u32],
        index_map: &'a alloc::collections::BTreeMap<u32, u32>,
        variables: &'a [f64],
        variables_scratch: &'a mut [f64],
        num_constraints: usize,
        num_variables: usize,
        jacobian: &'a mut [f64],
        residuals: &'a mut [f64],
        gradient: &'a mut [f64],
        direction: &'a [f64],
    }

    impl Eval<'_> {
        fn calculate_phi(&mut self, p: f64) -> Param {
            for (idx, d) in self.free_variables.iter().zip(self.direction) {
                self.variables_scratch[*idx as usize] = self.variables[*idx as usize] + p * d;
            }

            compute_residuals_and_jacobian(
                self.constraints,
                self.index_map,
                self.variables_scratch,
                self.residuals,
                self.jacobian,
                self.num_variables,
            );
            compute_gradient(
                self.jacobian,
                self.residuals,
                self.gradient,
                self.num_constraints,
                self.num_variables,
            );
            let phi = sum_squared_residuals(self.residuals);
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
    /// `gradient`, are filled with the values at `f(x + step_size * direction)`.
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
        constraints: &[&Edge],
        free_variables: &[u32],
        index_map: &alloc::collections::BTreeMap<u32, u32>,
        variables: &[f64],
        variables_scratch: &mut [f64],
        num_constraints: usize,
        num_variables: usize,
        jacobian: &mut [f64],
        residuals: &mut [f64],
        gradient: &mut [f64],
        direction: &[f64],
    ) -> Param {
        let phi0 = sum_squared_residuals(&*residuals);
        let dphi0 = dot_product(&*gradient, direction);
        let hz = HagerZhangLineSearch { phi0, dphi0 };

        let eval = &mut Eval {
            constraints,
            free_variables,
            index_map,
            variables,
            variables_scratch,
            num_constraints,
            num_variables,
            jacobian,
            residuals,
            gradient,
            direction,
        };
        hz.run(eval)
    }
}
