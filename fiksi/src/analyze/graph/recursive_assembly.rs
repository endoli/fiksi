// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{collections::VecDeque, vec, vec::Vec};
use hashbrown::{HashMap, HashSet};

use crate::{
    ConstraintId,
    elements::element::ElementId,
    graph::{Graph, IncidentElements},
};

struct Merge {
    elements: Vec<ElementId>,
    constraints: Vec<ConstraintId>,
    extensions: Vec<Extend>,
}

struct Extend {
    elements: ElementId,
    constraints: Vec<ConstraintId>,
}

/// An opaque cluster key.
///
/// This can be compared within the same step of the recombination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ClusterKey(u32);

/// A recombination plan.
///
/// This consists of [steps](RecombinationStep) to be solved to get to a full solution. Each step
/// represents one new cluster, containing the constraints that are to be solved, the geometric
/// elements that are introduced for the first time, and in which previously-seen clusters elements
/// in this step have already occurred.
///
/// Elements occurring in previously-seen clusters, must remain rigid with respect to those
/// clusters (e.g., any number of points occurring in a previously-seen cluster, must remain at the
/// same relative position, and only that cluster as a whole is allowed to be rigidly transformed).
///
/// See [`decompose`] for more information.
#[derive(Debug)]
pub(crate) struct RecombinationPlan {
    /// The partially ordered steps to take to solve the system.
    pub(crate) steps: Vec<RecombinationStep>,
}

impl RecombinationPlan {
    pub(crate) fn single(
        elements: impl IntoIterator<Item = ElementId>,
        constraints: impl IntoIterator<Item = ConstraintId>,
    ) -> Self {
        let free_elements: Vec<_> = elements.into_iter().collect();
        Self {
            steps: vec![RecombinationStep {
                constraints: constraints.into_iter().collect(),
                elements: free_elements.clone(),
                free_elements,
                on_frontiers: HashMap::new(),
                owned_elements: HashMap::new(),
                frontier_elements: HashMap::new(),
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
/// See [`RecombinationPlan`] for more information.
#[derive(Debug)]
pub(crate) struct RecombinationStep {
    /// The constraints to solve together in this step.
    ///
    /// These constraints can link to elements that are part of this cluster's core or this
    /// cluster's frontier. Elements on this cluster's frontier can also be part of other clusters'
    /// frontiers.
    ///
    /// This never contains constraints already solved in other clusters (regardless of whether the
    /// constraint is incident to their cores or their frontiers); those constraints were solved
    /// for previously in the partial order of steps. Those constraints will remain satisfied if
    /// the resulting configuration of geometry is only ever rigidly transformed afterward.
    constraints: Vec<ConstraintId>,

    /// The elements involved in this step.
    elements: Vec<ElementId>,

    /// The elements that are free at this step. They are contracted into the rigid cluster
    /// created by this step.
    ///
    /// This means this is the first time we see this element.
    free_elements: Vec<ElementId>,

    /// The clusters whose frontiers elements are part of.
    ///
    /// Elements can only be removed from frontiers in later steps when all those clusters get
    /// simplified.
    ///
    /// If the element is changed/moved as part of satisfying this step's constraints, all clusters
    /// will have to be rigidly moved along with it.
    on_frontiers: HashMap<ElementId, Vec<ClusterKey>>,

    // Which elements are owned by each cluster.
    //
    // The first cluster an element is part of in the partial order, is the cluster that owns the
    // element. Once the cluster is merged into a parent cluster, the parent cluster becomes the
    // owner.
    owned_elements: HashMap<ClusterKey, Vec<ElementId>>,

    // Which elements are on each clusters' frontiers.
    frontier_elements: HashMap<ClusterKey, Vec<ElementId>>,
}

impl RecombinationStep {
    /// The constraints to solve together in this step.
    pub(crate) fn constraints(&self) -> &[ConstraintId] {
        &self.constraints
    }

    pub(crate) fn elements(&self) -> &[ElementId] {
        &self.elements
    }

    /// The elements that become fixed after solving this step.
    pub(crate) fn free_elements(&self) -> &[ElementId] {
        &self.free_elements
    }

    pub(crate) fn on_frontiers(&self, element: ElementId) -> Option<&[ClusterKey]> {
        self.on_frontiers.get(&element).map(Vec::as_slice)
    }

    pub(crate) fn owned_elements(&self, cluster: ClusterKey) -> Option<&[ElementId]> {
        self.owned_elements.get(&cluster).map(Vec::as_slice)
    }

    pub(crate) fn frontier_elements(&self, cluster: ClusterKey) -> Option<&[ElementId]> {
        self.frontier_elements.get(&cluster).map(Vec::as_slice)
    }
}

/// Decompose a graph into a series of subgraphs that can be solved separately.
///
/// This returns a recombination plan consisting of several steps. Each step introduces a single
/// cluster containing a set of constraints to solve together, the elements that are seen for the
/// first time (i.e., are no longer free) after this cluster is solved, and previously-solved child
/// clusters that can be rigidly transformed and whose elements are to remain rigid.
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
    // These are the number of real constraints and elements in the system. More "merged"
    // constraints and elements will be added as part of rigidity bookkeeping in this algorithm,
    // but the entities they're made of don't need to be solved again.
    //
    // Currently, the constraint and elements IDs are just indices into the vectors, meaning that
    // the IDs can be compared against these lengths to determine whether the constraint or element
    // is "real," or is a merged entity used for bookkeeping.
    let num_real_constraints = u32::try_from(graph.constraints.len()).unwrap();
    let num_real_elements = u32::try_from(graph.elements.len()).unwrap();

    let mut vertices = HashSet::from_iter(vertices);
    let mut available_edges = HashSet::from_iter(edges);
    let mut constraints_handled = HashSet::new();
    let mut vertices_handled = HashSet::new();

    // Cluster bookkeeping.
    // Current clusters' frontiers elements are part of.
    let mut on_frontiers: HashMap<ElementId, Vec<ClusterKey>> = HashMap::new();
    // Which elements a cluster "owns." The first cluster that encounters an element "owns" that
    // element, until the cluster is merged into a parent cluster.
    let mut owned_elements: HashMap<ClusterKey, Vec<ElementId>> = HashMap::new();
    let mut owning_cluster: HashMap<ElementId, ClusterKey> = HashMap::new();

    // All elements occurring on a cluster's frontier. When the cluster is merged into another
    // cluster, these are to be removed.
    let mut frontier_elements: HashMap<ClusterKey, Vec<ElementId>> = HashMap::new();

    // Clusters we have "blocked." If a cluster is found, but that cluster does not have a core of
    // two or more elements, it cannot be simplified. It is then possible we find the same cluster
    // over and over. Blocking those clusters ensures we make progress.
    let mut blocked_clusters: Vec<HashSet<ElementId>> = Vec::new();

    let mut recombination_plan = RecombinationPlan { steps: vec![] };

    let mut step_constraints = Vec::new();
    let mut step_fixes_elements = Vec::new();
    for step in 0_u32.. {
        let cluster_key = ClusterKey(step);

        // Find a minimum dense subgraph.
        let subgraph = dense_bfs::<D>(&graph, &blocked_clusters, &available_edges, &vertices);

        if subgraph.is_none() {
            // No minimal dense subgraph was found. This means the remaining subgraphs are all
            // underconstrained.
            //
            // TODO: For now just push the entire subgraph, though that may be less efficient that
            // finding some optimal solving order. However, finding an optimal solving order in
            // underconstrained graphs is hard: you'd generally have to choose to fix/anchor some
            // elements to make the system well-constrained. That makes the system *structurally*
            // well-constrained, but those elements' values may still require updating for the
            // system to be satisfiable.
            let constraints: Vec<ConstraintId> = available_edges
                .iter()
                .copied()
                .filter(|edge| {
                    edge.id < num_real_constraints && !constraints_handled.contains(edge)
                })
                .collect();
            let fixes_elements: Vec<ElementId> = vertices
                .iter()
                .copied()
                .filter(|vertex| {
                    vertex.id < num_real_elements && !vertices_handled.contains(vertex)
                })
                .collect();

            if !constraints.is_empty() {
                recombination_plan.steps.push(RecombinationStep {
                    constraints,
                    elements: vertices
                        .iter()
                        .copied()
                        .filter(|vertex| vertex.id < num_real_elements)
                        .collect(),
                    free_elements: fixes_elements,
                    on_frontiers: on_frontiers.clone(),
                    owned_elements: owned_elements.clone(),
                    frontier_elements: frontier_elements.clone(),
                });
            }

            break;
        }

        let subgraph = subgraph.unwrap();
        debug_assert!(
            !subgraph.is_empty(),
            "if a subgraph is returned, it shouldn't be empty"
        );

        let mut core = Vec::new();
        let mut frontier = HashSet::new();

        let mut real_elements = Vec::new();

        // Find frontier vertices and the constraints we're solving for this step.
        for &vertex in &subgraph {
            let element = &graph.elements[vertex.id as usize];

            if vertex.id < num_real_elements {
                real_elements.push(vertex);
            }
            if vertex.id < num_real_elements && !vertices_handled.contains(&vertex) {
                step_fixes_elements.push(vertex);
                vertices_handled.insert(vertex);

                owning_cluster.insert(vertex, cluster_key);
            }

            let mut frontier_vertex = false;
            for edge_id in element
                .incident_constraints
                .iter()
                .filter(|&edge| available_edges.contains(edge))
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
                core.push(vertex);
            } else {
                frontier.insert(vertex);
            }
        }

        if !step_constraints.is_empty() {
            // Emit the step before we perform the cluster contraction.
            recombination_plan.steps.push(RecombinationStep {
                constraints: core::mem::take(&mut step_constraints),
                elements: real_elements,
                free_elements: step_fixes_elements.clone(),
                on_frontiers: on_frontiers.clone(),
                owned_elements: owned_elements.clone(),
                frontier_elements: frontier_elements.clone(),
            });
        }

        if !core.is_empty() || !step_fixes_elements.is_empty() {
            // This cluster owns the elements that are fixed for the first time, but we also know
            // it will end up owning some elements if its core is not empty (because in that case,
            // either it will fix some elements in the core for the first time, or it will end up
            // owning other clusters).
            //
            // By ensuring we create the cluster's entry in `owned_elements` here, we can safely
            // assume the entry exists once we do cluster merging below.
            //
            // So, an entry is created here if an only if this cluster ends up owning some
            // elements.
            owned_elements.insert(cluster_key, core::mem::take(&mut step_fixes_elements));
        }

        // Keep track of which frontiers elements are now on.
        for &vertex in &core {
            // TODO: remove constraints that are fully within the core
            if vertex.id < num_real_elements {
                let element = &graph.elements[vertex.id as usize];
                for edge_id in &element.incident_constraints {
                    let constraint = &graph.constraints[edge_id.id as usize];
                    if constraint
                        .incident_elements
                        .as_slice()
                        .iter()
                        .all(|element| core.contains(element))
                    {
                        available_edges.remove(edge_id);
                    }
                }
            }

            // If a vertex owned by a different cluster is in the core, that entire cluster is
            // merged into this. After this, all references to and from the old cluster are
            // removed.
            let old_cluster_key = owning_cluster.insert(vertex, cluster_key).unwrap();
            if old_cluster_key != cluster_key {
                // Reparent all elements owned by the old cluster under the new cluster.
                let old_cluster_owned_elements = owned_elements.remove(&old_cluster_key).unwrap();
                for &vertex in &old_cluster_owned_elements {
                    owning_cluster.insert(vertex, cluster_key);
                }
                owned_elements
                    .get_mut(&cluster_key)
                    .unwrap()
                    .extend(old_cluster_owned_elements.into_iter());

                // Remove the old cluster from all vertices' frontier bookkeeping.
                let frontier_elements = frontier_elements.remove(&old_cluster_key).unwrap();
                for element in frontier_elements {
                    if let Some(on_frontiers) = on_frontiers.get_mut(&element) {
                        let idx = on_frontiers
                            .iter()
                            .position(|&c| c == old_cluster_key)
                            .unwrap();
                        on_frontiers.swap_remove(idx);
                    }
                }
            }

            on_frontiers.remove(&vertex);
        }
        for &vertex in frontier.iter() {
            on_frontiers.entry(vertex).or_default().push(cluster_key);

            if vertex.id < num_real_elements {
                frontier_elements
                    .entry(cluster_key)
                    .or_default()
                    .push(vertex);
            }
        }

        // If there are two or more vertices in the core, we contract all those core vertices into
        // a single vertex to simplify the graph. We do that next. The following checks whether
        // there indeed are two or more vertices in the core.
        if subgraph.len() - frontier.len() <= 1 {
            // There are less than two vertices in the core. No contraction is possible, i.e., no
            // elements are merged. This is expected, but mark this subgraph so we don't find it
            // again.
            //
            // TODO: perhaps when a contraction *is* performed, remove old blocked clusters
            // containing any of the removed elements.
            blocked_clusters.push(subgraph.clone());
            continue;
        }

        let core = graph.add_element(0);
        owning_cluster.insert(core, cluster_key);
        vertices.insert(core);

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
                        .merge_elements(|element| !frontier.contains(&element), core);
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
                    IncidentElements::from_array([vertex, core]),
                );
                available_edges.insert(cluster_edge);
                total_incoming_edge_valency += binary_edge_cluster_valency;
            }
        }

        if total_incoming_edge_valency > 0 {
            graph.elements[core.id as usize].dof =
                total_frontier_vertex_dof - total_incoming_edge_valency - D;
        } else {
            vertices.remove(&core);
        }
    }

    recombination_plan
}

/// Find an arbitrary, minimum dense subgraph.
///
/// This searches for a minimum dense subgraph such that
/// `(elements' degrees of freedom) - (constraints' valencies) < D + 1`. See also "Finding Solvable
/// Subsets of Constraints Graph" (1997) by C. M. Hoffmann, A. Lomonosov, and M. Sitharam.
///
/// The subgraph is minimum in the sense that it is the smallest subgraph containing no proper
/// subgraph upholding that condition. Trivial subgraphs of single elements are not considered.
///
/// Returns `None` if no non-trivial dense subgraph exists.
///
/// This does an exhaustive search, which is very slow for every moderately-sized graphs. This
/// should be replaced with a smarter algorithm, as it's not necessary to find the optimum
/// solution.
fn dense_bfs<const D: i16>(
    graph: &Graph,
    blocked_subgraphs: &[HashSet<ElementId>],
    available_edges: &HashSet<ConstraintId>,
    vertices: &HashSet<ElementId>,
) -> Option<HashSet<ElementId>> {
    let k: i16 = const { -(D + 1) };

    // Calculate the additional edge valency added to `next_subgraph` by including `new_vertex`.
    fn additional_valency(
        graph: &Graph,
        available_edges: &HashSet<ConstraintId>,
        next_subgraph: &HashSet<ElementId>,
        new_vertex: ElementId,
    ) -> i16 {
        let element = &graph.elements[new_vertex.id as usize];
        let mut add = 0;
        for &edge_id in element
            .incident_constraints
            .iter()
            .filter(|&e| available_edges.contains(e))
        {
            let edge = &graph.constraints[edge_id.id as usize];
            if edge
                .incident_elements
                .as_slice()
                .iter()
                .all(|u| next_subgraph.contains(u))
            {
                add += edge.valency;
            }
        }
        add
    }

    // Add all vertices that are adjacent to `subgraph` through `from_vertex` into
    // `adjacent_vertices`.
    fn extend_adjacent_vertices(
        adjacent_vertices: &mut HashSet<ElementId>,
        graph: &Graph,
        available_edges: &HashSet<ConstraintId>,
        from_vertex: ElementId,
        subgraph: &HashSet<ElementId>,
        vertices: &HashSet<ElementId>,
    ) {
        let element = &graph.elements[from_vertex.id as usize];
        for &edge_id in element
            .incident_constraints
            .iter()
            .filter(|&e| available_edges.contains(e))
        {
            let edge = &graph.constraints[edge_id.id as usize];
            for &incident_element in edge.incident_elements.as_slice() {
                if vertices.contains(&incident_element) && !subgraph.contains(&incident_element) {
                    adjacent_vertices.insert(incident_element);
                }
            }
        }
    }

    #[derive(Clone)]
    struct SubgraphState {
        /// The vertices in this subgraph.
        subgraph: HashSet<ElementId>,

        /// The degrees of freedom of this subgraph.
        ///
        /// This is the sum of the degrees of freedom of all vertices in the subgraph minus the sum
        /// of the valency of its internal edges.
        dof: i16,

        /// The vertices that are adjacent to this subgraph (i.e., vertices that are connected by
        /// an edge to at least one of the vertices in this subgraph, but that are not inside this
        /// subgraph themselves).
        adjacent_vertices: HashSet<ElementId>,
    }

    let mut queue = VecDeque::new();

    // Seed BFS with all singleton (size 1) subgraphs.
    for &vertex in vertices.iter() {
        let mut subgraph = HashSet::with_capacity(4);
        subgraph.insert(vertex);

        let mut adjacent_vertices = HashSet::with_capacity(8);
        extend_adjacent_vertices(
            &mut adjacent_vertices,
            graph,
            available_edges,
            vertex,
            &subgraph,
            vertices,
        );

        queue.push_back(SubgraphState {
            subgraph,
            dof: graph.elements[vertex.id as usize].dof,
            adjacent_vertices,
        });
    }

    while let Some(SubgraphState {
        subgraph: sub,
        dof,
        adjacent_vertices,
    }) = queue.pop_front()
    {
        for &vertex in adjacent_vertices.iter() {
            // Build next state.
            let mut next_subgraph = sub.clone();
            next_subgraph.insert(vertex);
            debug_assert!(
                next_subgraph.len() >= 2,
                "We should never be considering trivial subgraphs of 1 element."
            );

            let valency = additional_valency(graph, available_edges, &next_subgraph, vertex);
            let next_dof = dof + graph.elements[vertex.id as usize].dof - valency;

            if !blocked_subgraphs.contains(&next_subgraph) && next_dof > k {
                return Some(next_subgraph);
            }

            // Find the adjacent vertices of `next_subgraph` by removing the vertex we just added,
            // and adding the vertices adjacent to the vertex we just added.
            let mut next_adjacent_vertices = adjacent_vertices.clone();
            next_adjacent_vertices.remove(&vertex);
            extend_adjacent_vertices(
                &mut next_adjacent_vertices,
                graph,
                available_edges,
                vertex,
                &next_subgraph,
                vertices,
            );

            queue.push_back(SubgraphState {
                subgraph: next_subgraph,
                dof: next_dof,
                adjacent_vertices: next_adjacent_vertices,
            });
        }
    }

    // No subgraph found. This means all subgraphs are underconstrained.
    None
}
