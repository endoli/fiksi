// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Using decompositions and solvers, assemble full system solutions.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We cast indices from usize -> u32 a lot here, this should be fine for all except unreasonably large systems (more than ~2^30 elements or constraints). We may want to revisit how indices are represented."
)]

use alloc::{collections::BTreeSet, vec, vec::Vec};

use hashbrown::HashMap;

use crate::{
    ClusterKey, Decomposer, ElementId, EncodedElement, Pose2D, RecombinationStep, Rng,
    SolvingOptions, Subsystem, System, analyze, collections::IndexMap, solve,
};

pub(crate) fn solve(system: &mut System, opts: SolvingOptions) {
    let mut rng = Rng::from_seed(42);

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
            match element {
                EncodedElement::Length { idx } => {
                    free_variables.extend(&[*idx]);
                }
                EncodedElement::Point { idx } => {
                    free_variables.extend(&[*idx, *idx + 1]);
                }
                EncodedElement::Line {
                    point1_idx,
                    point2_idx,
                } => {
                    free_variables.extend(&[
                        *point1_idx,
                        *point1_idx + 1,
                        *point2_idx,
                        *point2_idx + 1,
                    ]);
                }
                EncodedElement::Circle {
                    center_idx,
                    radius_idx,
                } => {
                    free_variables.extend(&[*center_idx, *center_idx + 1, *radius_idx]);
                }
            }
        }

        if opts.perturb {
            for free_variable in free_variables.iter().copied() {
                let variable = &mut system.variables[free_variable as usize];
                // TODO: the scale-independent perturbation here should be revisited. See
                // also https://github.com/endoli/fiksi/pull/41#discussion_r2234008761.
                *variable +=
                    *variable * (1. / 8196.) * rng.next_f64() + (1. / 65568.) * rng.next_f64();
            }
        }

        match opts.decomposer {
            Decomposer::None => {
                let mut free_variables_values = Vec::from_iter(
                    free_variables
                        .iter()
                        .copied()
                        .map(|free_variable_idx| system.variables[free_variable_idx as usize]),
                );
                let mut subsystem = Subsystem::new(
                    &system.variables,
                    &system.expressions,
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
                        free_variables_values[free_variable_idx];
                }
            }

            Decomposer::SinglePass => {
                let mut free_variables_values = Vec::new();
                for scc in system
                    .equation_graph
                    .find_strongly_connected_expressions(&free_variables)
                {
                    free_variables_values.clear();
                    free_variables_values.extend(
                        scc.free_variables
                            .iter()
                            .copied()
                            .map(|free_variable_idx| system.variables[free_variable_idx as usize]),
                    );

                    let mut subsystem = Subsystem::new(
                        &system.variables,
                        &system.expressions,
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
                        system.variables[variable_idx as usize] =
                            free_variables_values[free_variable_idx];
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
                        system.variables[*var_idx as usize] =
                            pose_and_element_variables[*updated_var_idx as usize];
                    }

                    for (cluster_idx, &cluster) in clustered_system.clusters.keys().enumerate() {
                        let pose_start_idx = 3 * cluster_idx;
                        let pose = Pose2D::from_array([
                            pose_and_element_variables[pose_start_idx],
                            pose_and_element_variables[pose_start_idx + 1],
                            pose_and_element_variables[pose_start_idx + 2],
                        ]);
                        for &element_id in step.owned_elements(cluster).unwrap_or(&[]) {
                            if !step.elements().contains(&element_id) {
                                let element = &system.elements[element_id.id as usize];
                                match element {
                                    &EncodedElement::Point { idx } => {
                                        let point = kurbo::Point::new(
                                            system.variables[idx as usize],
                                            system.variables[idx as usize + 1],
                                        );
                                        let transformed = pose.transform_point(point);
                                        system.variables[idx as usize] = transformed.x;
                                        system.variables[idx as usize + 1] = transformed.y;
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

        for &element_id in step.elements() {
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

        // Add the variables for the elements that are in the core or frontier of
        // the current step.
        for &element_id in step.elements() {
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
                    pose_and_element_variables.push(system.variables[idx as usize]);
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
                        system.variables[idx as usize],
                        system.variables[idx as usize + 1],
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
        self.clusters.clear();
        self.expressions.clear();
        self.num_pose_expressions = 0;
        self.variable_mapping_pose_and_element.clear();
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
            let expression = &system.expressions[*expression_id as usize];
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
                    system.variables[*idx as usize],
                    system.variables[*idx as usize + 1],
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

        residuals.fill(0.);
        jacobian.fill(0.);

        // Calculate the residuals and gradients of the expressions (of each
        // constraint) in this cluster.
        // TODO: make `VariableMap` a trait, so that we can reuse it here.
        let mut variable_indices = [0; 8];
        let mut variables = [0.; 8];
        let mut gradient = [0.; 8];
        let offset = this.num_pose_expressions as usize;
        for (expression_idx_in_problem, expression_id) in this.expressions.iter().enumerate() {
            let expression = &system.expressions[*expression_id as usize];
            for (variable_idx_in_expression, variable_idx) in expression
                .variable_indices(&mut variable_indices)
                .iter()
                .enumerate()
            {
                let mapped_variable_idx = *this
                    .variable_mapping_pose_and_element
                    .get(variable_idx)
                    .unwrap();
                variables[variable_idx_in_expression] =
                    pose_and_element_variables[mapped_variable_idx as usize];
            }

            let (residual, gradient) =
                expression.compute_residual_and_gradient(&variables, &mut gradient);
            residuals[offset + expression_idx_in_problem] = residual;

            let jacobian_row_start =
                (offset + expression_idx_in_problem) * this.num_variables as usize;
            for (variable_idx_in_expression, variable_idx) in expression
                .variable_indices(&mut variable_indices)
                .iter()
                .enumerate()
            {
                let mapped_variable_idx = *this
                    .variable_mapping_pose_and_element
                    .get(variable_idx)
                    .unwrap();
                jacobian[jacobian_row_start + mapped_variable_idx as usize] =
                    gradient[variable_idx_in_expression];
            }
        }

        // Calculate the residuals and gradients of the coincidence constraints of
        // each point that occurs on child clusters' frontiers.
        let mut r = 0;
        for (cluster_idx, (_, points)) in this.clusters.iter().enumerate() {
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
                    system.variables[*idx as usize],
                    system.variables[*idx as usize + 1],
                );

                let rigidly_transformed = pose.transform_point(point);
                let updated_idx = *this.variable_mapping_pose_and_element.get(idx).unwrap();
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

                let jacobian_row_start_x = r * this.num_variables as usize;
                let jacobian_row_start_y = (r + 1) * this.num_variables as usize;

                // Write the gradients on the child cluster pose.
                jacobian[jacobian_row_start_x + pose_start_idx
                    ..jacobian_row_start_x + pose_start_idx + 3]
                    .copy_from_slice(&gradient_x);
                jacobian[jacobian_row_start_y + pose_start_idx
                    ..jacobian_row_start_y + pose_start_idx + 3]
                    .copy_from_slice(&gradient_y);

                // Write the gradients on the updated point.
                jacobian[jacobian_row_start_x + updated_idx as usize] = -1.;
                jacobian[jacobian_row_start_y + updated_idx as usize + 1] = -1.;

                r += 2;
            }
        }
    }
}
