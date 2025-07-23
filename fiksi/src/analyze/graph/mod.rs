// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{
    collections::BTreeSet,
    {vec, vec::Vec},
};

use crate::{
    ConstraintId,
    elements::element::ElementId,
    graph::{Graph, IncidentElements},
};

/// A recombination plan.
///
/// This consists of [steps](RecombinationStep) to be solved to get to a full solution. Each step
/// contains the constraints that are to be solved, and the elements that become fixed after
/// solving that step. Once an element is fixed, its value should not be changed by future steps,
/// though it may still be part of future constraints.
///
/// See [`decompose`] for more information.
#[derive(Debug)]
pub(crate) struct RecombinationPlan {
    /// The partially ordered steps to take to solve the system.
    steps: Vec<RecombinationStep>,
}

impl RecombinationPlan {
    pub(crate) fn single(
        elements: impl IntoIterator<Item = ElementId>,
        constraints: impl IntoIterator<Item = ConstraintId>,
    ) -> Self {
        Self {
            steps: vec![RecombinationStep {
                constraints: constraints.into_iter().collect(),
                fixes_elements: elements.into_iter().collect(),
            }],
        }
    }

    /// Returns an iterator returning all the steps to be solved in order.
    pub(crate) fn steps(&self) -> impl Iterator<Item = &RecombinationStep> {
        self.steps.iter()
    }
}

/// A single step within a [`RecombinationPlan`].
///
/// This consists of constraints to solve this step, and the elements that become fixed after it.
#[derive(Debug)]
pub(crate) struct RecombinationStep {
    /// The constraints to solve together in this step.
    constraints: Vec<ConstraintId>,

    /// The elements that become fixed after solving this step.
    fixes_elements: Vec<ElementId>,
}

impl RecombinationStep {
    /// The constraints to solve together in this step.
    pub(crate) fn constraints(&self) -> &[ConstraintId] {
        &self.constraints
    }

    /// The elements that become fixed after solving this step.
    pub(crate) fn fixes_elements(&self) -> &[ElementId] {
        &self.fixes_elements
    }
}

/// Decompose a graph into a series of subgraphs that can be solved separately.
///
/// This returns a recombination plan consisting of several steps. Each step contains a set of
/// constraints to solve together, and the elements that become fixed (i.e., are no longer free)
/// after that step.
///
/// The decomposition attempts to find a good sequence of steps consisting of rigid subgraphs,
/// based on the target degrees of freedom geometry parameter `D` (normally `3` for 2D and `6` for
/// 3D). In case of a system containing under- and overconstrained parts, the decomposition will
/// inevitably contain steps with under- and overconstrained subgraphs.
///
/// This is based on the Modified Frontier Algorithm described in "Decomposition Plans for
/// Geometric Constraint Problems, Part II: New Algorithms" (2001) by C. M. Hoffman, A. Lomonosov
/// and M. Sitharam .
///
/// TODO: evaluate whether this also always works when vertices/edges are such that the graph is no
/// longer a connected component.
pub(crate) fn decompose<const D: i16>(
    mut graph: Graph,
    vertices: impl Iterator<Item = ElementId>,
    edges: impl Iterator<Item = ConstraintId>,
) -> RecombinationPlan {
    // This is the number of real constraints in the system. More "merged" constraints will be
    // added as part of rigidity bookkeeping in this algorithm, but these don't need to be
    // solved again.
    let num_real_constraints = u32::try_from(graph.constraints.len()).unwrap();
    let num_real_elements = u32::try_from(graph.elements.len()).unwrap();

    let mut vertices = BTreeSet::from_iter(vertices);
    let mut available_edges = BTreeSet::from_iter(edges);
    let mut constraints_handled = BTreeSet::new();
    let mut vertices_handled = BTreeSet::new();

    let mut blocked_clusters: BTreeSet<BTreeSet<ElementId>> = BTreeSet::new();

    let mut recombination_plan = RecombinationPlan { steps: vec![] };

    let mut step_constraints = Vec::new();
    let mut step_fixes_elements = Vec::new();
    while vertices.len() > 1 {
        // Find a minimal dense subgraph.
        let subgraph = dense::<D>(&graph, &blocked_clusters, &available_edges, &vertices);

        if subgraph.is_empty() {
            // No minimal dense subgraph was found. This means the remaining subgraphs are all
            // underconstrained.
            //
            // TODO: For now just push the entire subgraph (though that's not very efficient).
            // Perhaps this should figure out which constraints to solve in which order, and which
            // elements get fixed by that.
            recombination_plan.steps.push(RecombinationStep {
                constraints: available_edges
                    .difference(&constraints_handled)
                    .copied()
                    .collect(),
                fixes_elements: vertices.difference(&vertices_handled).copied().collect(),
            });

            break;
        }

        let mut frontier = BTreeSet::new();

        // Find frontier vertices
        for &vertex in &subgraph {
            let element = &graph.elements[vertex.id as usize];

            if vertex.id < num_real_elements && !vertices_handled.contains(&vertex) {
                step_fixes_elements.push(vertex);
                vertices_handled.insert(vertex);
            }

            let mut frontier_vertex = false;
            for edge_id in element
                .incident_constraints
                .iter()
                .filter(|edge| available_edges.contains(edge))
            {
                let edge = &graph.constraints[edge_id.id as usize];
                if edge
                    .incident_elements
                    .as_slice()
                    .iter()
                    .all(|element| subgraph.contains(element))
                {
                    if edge_id.id < num_real_constraints && !constraints_handled.contains(edge_id) {
                        step_constraints.push(*edge_id);
                        constraints_handled.insert(*edge_id);
                    }
                } else {
                    frontier_vertex = true;
                }
            }

            if !frontier_vertex {
                // Remove all inner vertices.
                vertices.remove(&vertex);
            } else {
                frontier.insert(vertex);
            }
        }

        if !step_fixes_elements.is_empty() || !step_constraints.is_empty() {
            recombination_plan.steps.push(RecombinationStep {
                constraints: core::mem::take(&mut step_constraints),
                fixes_elements: core::mem::take(&mut step_fixes_elements),
            });
        }

        if subgraph.len() - frontier.len() <= 1 {
            // No simplification is possible, as the inner cluster is empty or consists of
            // exactly one element, i.e., no elements are merged. This is expected, but mark
            // this subgraph so we don't find it again.
            //
            // TODO: perhaps when a simplification *is* made, remove old blocked clusters
            // containing any of the removed elements.
            blocked_clusters.insert(subgraph.clone());
            continue;
        }

        let cluster = graph.add_element(0);
        vertices.insert(cluster);

        let mut total_frontier_vertex_dof = 0;
        let mut total_incoming_edge_valency = 0;
        // Create edges from frontier vertices to the inner vertex cluster
        for &vertex in &frontier {
            let element = &graph.elements[vertex.id as usize];

            total_frontier_vertex_dof += element.dof;

            // The sum of weights of binary edges from this frontier vertex to vertices in the
            // inner cluster.
            let mut binary_edge_cluster_valency = 0;

            for edge_id in &element.incident_constraints {
                if !available_edges.contains(edge_id) {
                    continue;
                }

                let edge = &mut graph.constraints[edge_id.id as usize];
                let incident_elements = edge.incident_elements.as_slice();

                if incident_elements
                    .iter()
                    .all(|element| subgraph.contains(element))
                {
                    let new = edge
                        .incident_elements
                        .merge_elements(|element| !frontier.contains(&element), cluster);
                    if new.len() == 2 {
                        binary_edge_cluster_valency += edge.valency;
                        available_edges.remove(edge_id);
                    } else {
                        edge.incident_elements = new;
                    }
                }
            }

            if binary_edge_cluster_valency > 0 {
                let cluster_edge = graph.add_constraint(
                    binary_edge_cluster_valency,
                    IncidentElements::from_array([vertex, cluster]),
                );
                available_edges.insert(cluster_edge);
                total_incoming_edge_valency += binary_edge_cluster_valency;
            }
        }

        if total_incoming_edge_valency > 0 {
            graph.elements[cluster.id as usize].dof =
                total_frontier_vertex_dof - total_incoming_edge_valency - D;
        } else {
            vertices.remove(&cluster);
        }
    }

    recombination_plan
}

/// Find an arbitrary, minimal dense subgraph.
///
/// This searches for a minimal dense subgraph such that
/// `(elements' degrees of freedom) - (constraints' valencies) < D + 1`. See also "Finding Solvable
/// Subsets of Constraints Graph" (1997) by C. M. Hoffmann, A. Lomonosov, and M. Sitharam.
///
/// The subgraph is minimal in the sense that it contains no proper subgraph upholding that
/// condition. Trivial subgraphs of single elements are not considered.
///
/// TODO: this currently does an exhaustive search, actually finding optimal *minimum* dense
/// subgraphs, which is very slow. This should be replaced with a smarter algorithm, as it's not
/// necessary to find optimal solutions.
fn dense<const D: i16>(
    graph: &Graph,
    blocked_subgraphs: &BTreeSet<BTreeSet<ElementId>>,
    available_edges: &BTreeSet<ConstraintId>,
    vertices: &BTreeSet<ElementId>,
) -> BTreeSet<ElementId> {
    let mut available_vertices = vertices.clone();
    let mut subgraph = BTreeSet::new();

    fn recurse<const D: i16>(
        graph: &Graph,
        blocked_subgraphs: &BTreeSet<BTreeSet<ElementId>>,
        available_edges: &BTreeSet<ConstraintId>,
        available_vertices: &mut BTreeSet<ElementId>,
        subgraph: &mut BTreeSet<ElementId>,
        dof_vertices: i16,
        valency_edges: i16,
    ) -> bool {
        let k = const { -(D + 1) };

        for vertex in Vec::from_iter(available_vertices.iter().copied()) {
            available_vertices.remove(&vertex);

            let element = &graph.elements[vertex.id as usize];
            subgraph.insert(vertex);

            let mut additional_valency = 0;

            for edge in element
                .incident_constraints
                .iter()
                .filter(|edge| available_edges.contains(edge))
            {
                let edge = &graph.constraints[edge.id as usize];

                if edge
                    .incident_elements
                    .as_slice()
                    .iter()
                    .all(|element| subgraph.contains(element))
                {
                    additional_valency += edge.valency;
                }
            }

            let dof_vertices = dof_vertices + element.dof;
            let valency_edges = valency_edges + additional_valency;

            if !blocked_subgraphs.contains(subgraph)
                && subgraph.len() > 1
                && valency_edges - dof_vertices > k
            {
                return true;
            }

            if recurse::<D>(
                graph,
                blocked_subgraphs,
                available_edges,
                available_vertices,
                subgraph,
                dof_vertices,
                valency_edges,
            ) {
                return true;
            }

            subgraph.remove(&vertex);
            available_vertices.insert(vertex);
        }
        false
    }

    if recurse::<D>(
        graph,
        blocked_subgraphs,
        available_edges,
        &mut available_vertices,
        &mut subgraph,
        0,
        0,
    ) {
        subgraph
    } else {
        // No subgraph found. This means all subgraphs are underconstrained.
        BTreeSet::new()
    }
}
