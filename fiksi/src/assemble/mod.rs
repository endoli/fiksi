// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Using decompositions and solvers, assemble full system solutions.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We cast indices from usize -> u32 a lot here, this should be fine for all except unreasonably large systems (more than ~2^30 elements or constraints). We may want to revisit how indices are represented."
)]

use alloc::{collections::BTreeSet, vec, vec::Vec};

use hashbrown::HashMap;
use solvi::TripletMat;

use crate::{
    ClusterKey, Decomposer, ElementId, EncodedElement, Expression, Pose2D, RecombinationStep, Rng,
    SolvingOptions, Subsystem, System, analyze, collections::IndexMap, constraints::expressions,
    solve, utils,
};

/// Calculate the "scale" of the system: the rough typical order of magnitude of system elements
/// and constraints.
///
/// This is the root mean square of the magnitude of Euclidean coordinate- and length-like entities
/// in the system.
///
/// Note: currently all variables in the system have a possible direct "length" interpretation (a
/// point's x-coordinate can be interpreted as the offset length from the origin, a circle radius
/// is a length directly, etc.) If we ever have variables representing angles, we'd need to exclude
/// them here. Angles in radians would already be on the order of ~1.0.
fn calculate_system_scale(system: &System) -> f64 {
    utils::root_mean_squares(system.variables.iter().copied().chain(
        system.expressions.iter().filter_map(|e| match e {
            Expression::PointPointDistance(expressions::PointPointDistance {
                distance, ..
            })
            | Expression::PointLineDistance(expressions::PointLineDistance { distance, .. }) => {
                Some(*distance)
            }
            _ => None,
        }),
    ))
}

pub(crate) fn solve(system: &mut System, opts: SolvingOptions) {
    let mut rng = Rng::from_seed(42);

    // For numeric solving, it's nice if the problem is well-conditioned. For Levenberg-Marquardt,
    // this means the Jacobian should be well-conditioned. We bring back the total scale of the
    // problem to be on the order of ~1.0 by calculating the root mean square of all lengths, and
    // dividing all lengths (in both element variables and constraints) by the scale before going
    // into numeric solving.
    //
    // This removes the `O(system scale)` effect on length-like residuals (such as point-point
    // distance), making, e.g., length and angle residuals (which are `O(1)` in radians) more
    // comparable.
    let system_scale = calculate_system_scale(&*system);
    {
        let system_scale_recip = 1. / system_scale;

        system
            .variables_transformed
            .resize(system.variables.len(), 0.);
        for (variable, variable_scaled) in system
            .variables
            .iter()
            .zip(system.variables_transformed.iter_mut())
        {
            *variable_scaled = *variable * system_scale_recip;
        }
        system.expressions_transformed.clear();
        system.expressions_transformed.extend(
            system
                .expressions
                .iter()
                .map(|e| e.transform(system_scale_recip)),
        );
    }

    for connected_component in system.graph.connected_components() {
        let (elements, constraints) = (
            connected_component.elements.clone(),
            connected_component.constraints.clone(),
        );

        if elements.is_empty() {
            continue;
        }

        let mut free_variables = BTreeSet::<u32>::new();
        for element_id in &elements {
            let element = &system.elements[element_id.id as usize];
            let variable_indices: &[u32] = match element {
                EncodedElement::Length { idx } => &[*idx],
                EncodedElement::Point { idx } => &[*idx, *idx + 1],
                EncodedElement::Line {
                    point1_idx,
                    point2_idx,
                } => &[*point1_idx, *point1_idx + 1, *point2_idx, *point2_idx + 1],
                EncodedElement::Circle {
                    center_idx,
                    radius_idx,
                } => &[*center_idx, *center_idx + 1, *radius_idx],
            };
            for &variable in variable_indices {
                if !system.fixed_variables.contains(&variable) {
                    free_variables.insert(variable);
                }
            }
        }

        if opts.perturb {
            for free_variable in free_variables.iter().copied() {
                let variable = &mut system.variables_transformed[free_variable as usize];
                // Nudge the variable by a random small factor of itself and by a random small flat
                // amount.
                //
                // Note this acts on scaled variables, so the flat amount is relative to the total
                // system scale.
                *variable +=
                    *variable * (1. / 8196.) * rng.next_f64() + (1. / 65568.) * rng.next_f64();
            }
        }

        match opts.decomposer {
            Decomposer::None => {
                let mut free_variables_values =
                    Vec::from_iter(free_variables.iter().copied().map(|free_variable_idx| {
                        system.variables_transformed[free_variable_idx as usize]
                    }));
                let mut subsystem = Subsystem::new(
                    &system.variables_transformed,
                    &system.expressions_transformed,
                    free_variables.iter().copied(),
                    constraints
                        .iter()
                        .flat_map(|c| {
                            let constraint = &system.constraints[c.id as usize];
                            (0..constraint.tag.valency()).map(|offset| {
                                system.constraints[c.id as usize].expressions_idx
                                    + u32::from(offset)
                            })
                        })
                        .collect(),
                );

                match opts.optimizer {
                    solve::Optimizer::LevenbergMarquardt => {
                        crate::solve::levenberg_marquardt(
                            &mut subsystem,
                            &mut free_variables_values,
                        );
                    }
                    solve::Optimizer::LBfgs => {
                        crate::solve::lbfgs(&mut subsystem, &mut free_variables_values);
                    }
                }

                // Update system variables' values with the free variables' values.
                for (free_variable_idx, variable_idx) in
                    subsystem.free_variables.into_iter().enumerate()
                {
                    system.variables[variable_idx as usize] =
                        system_scale * free_variables_values[free_variable_idx];
                }
            }

            Decomposer::SinglePass => {
                let mut free_variables_values = Vec::new();
                for scc in system
                    .equation_graph
                    .find_strongly_connected_expressions(&free_variables)
                {
                    free_variables_values.clear();
                    free_variables_values.extend(scc.free_variables.iter().copied().map(
                        |free_variable_idx| {
                            system.variables_transformed[free_variable_idx as usize]
                        },
                    ));

                    let mut subsystem = Subsystem::new(
                        &system.variables_transformed,
                        &system.expressions_transformed,
                        scc.free_variables.iter().copied(),
                        scc.expressions,
                    );

                    match opts.optimizer {
                        solve::Optimizer::LevenbergMarquardt => {
                            crate::solve::levenberg_marquardt(
                                &mut subsystem,
                                &mut free_variables_values,
                            );
                        }
                        solve::Optimizer::LBfgs => {
                            crate::solve::lbfgs(&mut subsystem, &mut free_variables_values);
                        }
                    }

                    for (free_variable_idx, variable_idx) in
                        subsystem.free_variables.into_iter().enumerate()
                    {
                        system.variables_transformed[variable_idx as usize] =
                            free_variables_values[free_variable_idx];
                        system.variables[variable_idx as usize] =
                            system_scale * free_variables_values[free_variable_idx];
                    }
                }
            }

            Decomposer::RecursiveAssembly => {
                let decomp = analyze::graph::recursive_assembly::decompose::<3>(
                    system.graph.clone(),
                    elements.iter().copied(),
                    constraints.iter().copied(),
                );
                let mut clustered_system = ClusteredSystem::default();
                let mut pose_and_element_variables = vec![];
                for step in decomp.steps() {
                    pose_and_element_variables.clear();
                    clustered_system.build(system, step, &mut pose_and_element_variables);

                    crate::solve::levenberg_marquardt(
                        &mut (&mut clustered_system, &*system),
                        &mut pose_and_element_variables,
                    );

                    // Update the system's variables with the solved values.
                    for (var_idx, updated_var_idx) in
                        clustered_system.variable_mapping_pose_and_element.iter()
                    {
                        system.variables_transformed[*var_idx as usize] =
                            pose_and_element_variables[*updated_var_idx as usize];
                        system.variables[*var_idx as usize] =
                            system_scale * pose_and_element_variables[*updated_var_idx as usize];
                    }

                    for (cluster_idx, &cluster) in clustered_system.clusters.keys().enumerate() {
                        let pose_start_idx = 3 * cluster_idx;
                        let pose = Pose2D::from_array([
                            pose_and_element_variables[pose_start_idx],
                            pose_and_element_variables[pose_start_idx + 1],
                            pose_and_element_variables[pose_start_idx + 2],
                        ]);
                        for &element_id in step.owned_elements(cluster).unwrap_or(&[]) {
                            if !clustered_system
                                .step_plus_frontier_elements
                                .contains(&element_id)
                            {
                                let element = &system.elements[element_id.id as usize];
                                match element {
                                    &EncodedElement::Point { idx } => {
                                        let point = kurbo::Point::new(
                                            system.variables_transformed[idx as usize],
                                            system.variables_transformed[idx as usize + 1],
                                        );
                                        let transformed = pose.transform_point(point);
                                        system.variables_transformed[idx as usize] = transformed.x;
                                        system.variables_transformed[idx as usize + 1] =
                                            transformed.y;
                                        system.variables[idx as usize] =
                                            system_scale * transformed.x;
                                        system.variables[idx as usize + 1] =
                                            system_scale * transformed.y;
                                    }
                                    &EncodedElement::Length { .. }
                                    | &EncodedElement::Line { .. }
                                    | &EncodedElement::Circle { .. } => {
                                        // no-op
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct ClusteredSystem {
    /// The number of free variables in this clustered system.
    ///
    /// These are the poses of the clusters this level transforms, and the
    /// variables of elements that are free at this level.
    num_variables: u32,

    /// All the elements and elements on clusters' frontiers that we update at this step.
    step_plus_frontier_elements: Vec<ElementId>,

    /// The clusters this clustered system rigidly transforms.
    ///
    /// We enforce coincidence (equality) between point elements that occur at the
    /// frontier of this level as well as all the frontiers of child clusters;
    /// i.e., if the *same* point element is on multiple frontiers, the clusters
    /// have to be rigidly transformed such that all clusters agree on its global
    /// position.
    // clusters: IndexSet<ClusterKey>,
    clusters: IndexMap<ClusterKey, Vec<ElementId>>,

    /// The to-be-optimized expressions of this clustered system.
    expressions: Vec<u32>,

    /// The number of additional pose expressions (due to coincidence constraints
    /// we add to ensure transformed clusters agree on global positioning of
    /// elements).
    num_pose_expressions: u32,

    /// Mapping of global element variable indices to variable indices within
    /// `pose_and_element_variables`.
    variable_mapping_pose_and_element: HashMap<u32, u32>,
}

impl ClusteredSystem {
    pub(crate) fn build(
        &mut self,
        system: &System,
        step: &RecombinationStep,
        // The to-be-optimized variables of this clustered system. These are the
        // cluster poses at this level (initially all 0) followed by the element
        // variables.
        pose_and_element_variables: &mut Vec<f64>,
    ) {
        self.clear();
        pose_and_element_variables.clear();

        for constraint_id in step.constraints() {
            let constraint = &system.constraints[constraint_id.id as usize];
            for offset in 0..constraint.tag.valency() as u32 {
                self.expressions.push(constraint.expressions_idx + offset);
            }
        }

        self.step_plus_frontier_elements
            .extend_from_slice(step.elements());

        {
            // When solving using a recursive assembly plan, elements we're now solving for may
            // occur on one or more existing cluster's frontiers. We need to solve for those
            // cluster's poses such that all clusters agree on those elements' global positions.
            // Additionally, changing one of those clusters' poses may move an element that we're
            // not directly solving for, but which may itself occur on some other cluster's
            // frontier. That cluster will then also have to be moved, etc.
            //
            // Therefore, we need to solve for the transitive closure of all elements affected.
            // Elements on a cluster's frontier or interior, which are *only* part of that single
            // cluster, are not solved for directly. Instead, they are transformed with their
            // cluster's pose after this solve.
            let mut reachable_clusters = vec![];
            for &element_id in step.elements() {
                let element = &system.elements[element_id.id as usize];
                // Only point elements (for now) are affected by rigid transformation.
                if !matches!(element, &EncodedElement::Point { .. }) {
                    continue;
                }
                if let Some(clusters) = step.on_frontiers(element_id) {
                    for cluster in clusters {
                        if !reachable_clusters.contains(cluster) {
                            reachable_clusters.push(*cluster);
                        }
                    }
                }
            }
            let mut i = 0;
            while let Some(&cluster) = reachable_clusters.get(i) {
                i += 1;
                for &element_id in step.frontier_elements(cluster).unwrap() {
                    let element = &system.elements[element_id.id as usize];
                    // Only point elements (for now) are affected by rigid transformation.
                    if !matches!(element, &EncodedElement::Point { .. }) {
                        continue;
                    }
                    if let Some(clusters) = step.on_frontiers(element_id) {
                        for cluster in clusters {
                            if !reachable_clusters.contains(cluster) {
                                reachable_clusters.push(*cluster);
                            }
                        }
                    }

                    let num_frontiers = step
                        .on_frontiers(element_id)
                        .map(|clusters| clusters.len())
                        .unwrap_or(0);
                    if !self.step_plus_frontier_elements.contains(&element_id) && num_frontiers > 1
                    {
                        self.step_plus_frontier_elements.push(element_id);
                    }
                }
            }
        }

        for &element_id in &self.step_plus_frontier_elements {
            let element = &system.elements[element_id.id as usize];

            if let Some(clusters) = step.on_frontiers(element_id) {
                // This element is on one or more existing clusters' frontiers.
                // Those clusters should be rigidly transformed such that all
                // clusters agree on the positioning of the element.
                match element {
                    EncodedElement::Length { .. } => {
                        // Lengths are not affected by rigid transforms.
                    }
                    &EncodedElement::Point { .. } => {
                        for &cluster in clusters {
                            // One pose expression (variable-variable equality) for
                            // each coordinate of the point.
                            self.num_pose_expressions += 2;
                            self.clusters.entry(cluster).or_default().push(element_id);
                        }
                    }
                    EncodedElement::Line { .. } => {
                        // Lines are not primitives and do not contribute
                        // variables.
                    }
                    EncodedElement::Circle { .. } => {
                        // Circles are not primitives and do not contribute
                        // variables.
                    }
                }
            } else {
                debug_assert!(
                    step.free_elements().contains(&element_id),
                    "If the element isn't on any existing cluster's frontier, the element must be seen for the first time and is free."
                );
            }
        }

        // Add the variables for the poses.
        pose_and_element_variables.resize(self.clusters.len() * 3, 0.);

        // Add the variables for the elements that we will be solving for this step. This is the
        // transitive closure of all elements affected, either directly or by being on more than
        // one cluster's frontier.
        for &element_id in &self.step_plus_frontier_elements {
            let element = &system.elements[element_id.id as usize];

            match element {
                &EncodedElement::Length { idx } => {
                    // TODO: Lengths that are not free here for the first time,
                    // have already become rigid, and probably shouldn't be solved
                    // again.
                    debug_assert!(
                        !self.variable_mapping_pose_and_element.contains_key(&idx),
                        "Elements should not be present more than once.",
                    );

                    self.variable_mapping_pose_and_element
                        .insert(idx, pose_and_element_variables.len() as u32);
                    pose_and_element_variables.push(system.variables_transformed[idx as usize]);
                }
                &EncodedElement::Point { idx } => {
                    debug_assert!(
                        !self.variable_mapping_pose_and_element.contains_key(&idx)
                            && !self
                                .variable_mapping_pose_and_element
                                .contains_key(&(idx + 1)),
                        "Elements should not be present more than once.",
                    );

                    self.variable_mapping_pose_and_element
                        .insert(idx, pose_and_element_variables.len() as u32);
                    self.variable_mapping_pose_and_element
                        .insert(idx + 1, pose_and_element_variables.len() as u32 + 1);
                    pose_and_element_variables.extend_from_slice(&[
                        system.variables_transformed[idx as usize],
                        system.variables_transformed[idx as usize + 1],
                    ]);
                }
                EncodedElement::Line { .. } => {}
                EncodedElement::Circle { .. } => {}
            }
        }
        self.num_variables = pose_and_element_variables.len() as u32;
    }

    fn clear(&mut self) {
        self.num_variables = 0;
        self.step_plus_frontier_elements.clear();
        self.clusters.clear();
        self.expressions.clear();
        self.num_pose_expressions = 0;
        self.variable_mapping_pose_and_element.clear();
    }

    fn calculate_residuals_and_jacobian(
        &mut self,
        system: &System,
        pose_and_element_variables: &[f64],
        residuals: &mut [f64],
        jacobian: &mut impl PushTriplet,
    ) {
        // Calculate the residuals and gradients of the expressions (of each
        // constraint) in this cluster.
        // TODO: make `VariableMap` a trait, so that we can reuse it here.
        let mut variable_indices = [0; 8];
        let mut variables = [0.; 8];
        let mut gradient = [0.; 8];
        let offset = self.num_pose_expressions as usize;
        for (expression_idx_in_problem, expression_id) in self.expressions.iter().enumerate() {
            let expression = &system.expressions_transformed[*expression_id as usize];
            for (variable_idx_in_expression, variable_idx) in expression
                .variable_indices(&mut variable_indices)
                .iter()
                .enumerate()
            {
                let mapped_variable_idx = *self
                    .variable_mapping_pose_and_element
                    .get(variable_idx)
                    .unwrap();
                variables[variable_idx_in_expression] =
                    pose_and_element_variables[mapped_variable_idx as usize];
            }

            let (residual, gradient) =
                expression.compute_residual_and_gradient(&variables, &mut gradient);
            residuals[offset + expression_idx_in_problem] = residual;

            for (variable_idx_in_expression, variable_idx) in expression
                .variable_indices(&mut variable_indices)
                .iter()
                .enumerate()
            {
                let mapped_variable_idx = *self
                    .variable_mapping_pose_and_element
                    .get(variable_idx)
                    .unwrap();
                jacobian.push_triplet(
                    offset + expression_idx_in_problem,
                    mapped_variable_idx as usize,
                    gradient[variable_idx_in_expression],
                );
            }
        }

        // Calculate the residuals and gradients of the coincidence constraints of
        // each point that occurs on child clusters' frontiers.
        let mut r = 0;
        for (cluster_idx, (_, points)) in self.clusters.iter().enumerate() {
            let pose_start_idx = 3 * cluster_idx;
            let pose = Pose2D::from_array([
                pose_and_element_variables[pose_start_idx],
                pose_and_element_variables[pose_start_idx + 1],
                pose_and_element_variables[pose_start_idx + 2],
            ]);
            for &point_id in points {
                let EncodedElement::Point { idx } = &system.elements[point_id.id as usize] else {
                    unreachable!()
                };
                let point = kurbo::Point::new(
                    system.variables_transformed[*idx as usize],
                    system.variables_transformed[*idx as usize + 1],
                );

                let rigidly_transformed = pose.transform_point(point);
                let updated_idx = *self.variable_mapping_pose_and_element.get(idx).unwrap();
                let updated = kurbo::Point::new(
                    pose_and_element_variables[updated_idx as usize],
                    pose_and_element_variables[updated_idx as usize + 1],
                );

                residuals[r] = rigidly_transformed.x - updated.x;
                residuals[r + 1] = rigidly_transformed.y - updated.y;

                // Note: while we compute the gradient through the chain rule here,
                // with the inner derivatives we're giving, we're actually just
                // computing the gradient directly. We're keeping this method
                // around as we may be using it in the future. Inlining sohuld take
                // care of the dead code.
                let gradient_x = pose.gradient_chain_rule_point(point, [1., 0.]);
                let gradient_y = pose.gradient_chain_rule_point(point, [0., 1.]);

                for (x, g) in gradient_x.into_iter().enumerate() {
                    jacobian.push_triplet(r, pose_start_idx + x, g);
                }
                for (y, g) in gradient_y.into_iter().enumerate() {
                    jacobian.push_triplet(r + 1, pose_start_idx + y, g);
                }

                // Write the gradients on the updated point.
                jacobian.push_triplet(r, updated_idx as usize, -1.);
                jacobian.push_triplet(r + 1, updated_idx as usize + 1, -1.);

                r += 2;
            }
        }
    }
}

/// A trait to push triplets to an underlying matrix storage.
///
/// This allows abstracting Jacobian-calculating code over sparse versus dense matrices.
///
/// If multiple values are pushed at the same `(row, col)` entry, their values are added.
/// This is the same behavior as [`solvi::TripletMat`].
trait PushTriplet {
    fn push_triplet(&mut self, row: usize, col: usize, value: f64);
}

impl PushTriplet for TripletMat<f64> {
    fn push_triplet(&mut self, row: usize, col: usize, value: f64) {
        self.push_triplet(row, col, value);
    }
}

/// A dense matrix in column-major format.
struct DenseColMat<'a> {
    num_cols: usize,
    matrix: &'a mut [f64],
}

impl PushTriplet for DenseColMat<'_> {
    fn push_triplet(&mut self, row: usize, col: usize, value: f64) {
        self.matrix[row * self.num_cols + col] = value;
    }
}

impl solve::Problem for (&'_ mut ClusteredSystem, &'_ System) {
    fn num_variables(&self) -> u32 {
        let (this, _system) = self;
        this.num_variables
    }

    fn num_residuals(&self) -> u32 {
        let (this, _system) = self;
        this.expressions.len() as u32 + this.num_pose_expressions
    }

    fn calculate_residuals(&mut self, pose_and_element_variables: &[f64], residuals: &mut [f64]) {
        let (this, system) = self;

        residuals.fill(0.);

        // TODO: make `VariableMap` a trait, so that we can reuse it here.
        let mut variable_indices = [0; 8];
        let mut variables = [0.; 8];
        let mut gradient = [0.; 8];

        let offset = this.num_pose_expressions as usize;
        for (expression_idx_in_problem, expression_id) in this.expressions.iter().enumerate() {
            let expression = &system.expressions_transformed[*expression_id as usize];
            for (i, variable_idx) in expression
                .variable_indices(&mut variable_indices)
                .iter()
                .enumerate()
            {
                let mapped_variable_idx = *this
                    .variable_mapping_pose_and_element
                    .get(variable_idx)
                    .unwrap();
                variables[i] = pose_and_element_variables[mapped_variable_idx as usize];
            }

            let (residual, _) = expression.compute_residual_and_gradient(&variables, &mut gradient);
            residuals[offset + expression_idx_in_problem] = residual;
        }

        let mut r = 0;
        for (cluster_idx, (_, points)) in this.clusters.iter().enumerate() {
            let pose_start_idx = 3 * cluster_idx;
            let pose = Pose2D::from_array([
                pose_and_element_variables[pose_start_idx],
                pose_and_element_variables[pose_start_idx + 1],
                pose_and_element_variables[pose_start_idx + 2],
            ]);
            for &point in points {
                let EncodedElement::Point { idx } = &system.elements[point.id as usize] else {
                    unreachable!()
                };
                let point = kurbo::Point::new(
                    system.variables_transformed[*idx as usize],
                    system.variables_transformed[*idx as usize + 1],
                );

                let rigidly_transformed = pose.transform_point(point);
                let updated_idx = *this.variable_mapping_pose_and_element.get(idx).unwrap();
                let updated = kurbo::Point::new(
                    pose_and_element_variables[updated_idx as usize],
                    pose_and_element_variables[updated_idx as usize + 1],
                );

                residuals[r] = rigidly_transformed.x - updated.x;
                residuals[r + 1] = rigidly_transformed.y - updated.y;
                r += 2;
            }
        }
    }

    fn calculate_residuals_and_jacobian(
        &mut self,
        pose_and_element_variables: &[f64],
        residuals: &mut [f64],
        jacobian: &mut [f64],
    ) {
        let (this, system) = self;

        this.calculate_residuals_and_jacobian(
            system,
            pose_and_element_variables,
            residuals,
            &mut DenseColMat {
                num_cols: this.num_variables as usize,
                matrix: jacobian,
            },
        );
    }

    fn calculate_residuals_and_sparse_jacobian(
        &mut self,
        pose_and_element_variables: &[f64],
        residuals: &mut [f64],
        jacobian: &mut TripletMat<f64>,
    ) {
        let (this, system) = self;

        this.calculate_residuals_and_jacobian(
            system,
            pose_and_element_variables,
            residuals,
            jacobian,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::{System, constraints, elements};

    #[test]
    fn system_scale() {
        let mut s = System::new();

        // Add elements with coordinates in range 0-250.
        let p0 = elements::Point::create(&mut s, 0. * 1e2, 0. * 1e2);
        let p1 = elements::Point::create(&mut s, 1. * 1e2, 0.2 * 1e2);
        let _p2 = elements::Point::create(&mut s, 0.1 * 1e2, 2.5 * 1e2);

        let scale = super::calculate_system_scale(&s);
        assert!(
            scale > 1e1 && scale < 1e3,
            "System should be on the order O(10^2)"
        );

        // Add a distance constraint at a larger order of magnitude, this should bump the system
        // scale quite a bit.
        constraints::PointPointDistance::create(&mut s, p0, p1, 1e4);
        let scale = super::calculate_system_scale(&s);
        assert!(
            scale > 1e3 && scale < 1e5,
            "System should be on the order O(10^4)"
        );
    }
}
