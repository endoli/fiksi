// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{vec, vec::Vec};
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
    let mut blocked_clusters: Vec<Vec<ElementId>> = Vec::new();

    let mut recombination_plan = RecombinationPlan { steps: vec![] };

    let mut step_constraints = Vec::new();
    let mut step_fixes_elements = Vec::new();
    for step in 0_u32.. {
        let cluster_key = ClusterKey(step);
        let owned_elements_before = owned_elements.clone();
        owned_elements.insert(cluster_key, Vec::new());

        // Find a minimal dense subgraph.
        let subgraph = dense::<D>(&graph, &blocked_clusters, &available_edges, &vertices);

        if subgraph.is_empty() {
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
                    owned_elements: owned_elements_before,
                    frontier_elements: frontier_elements.clone(),
                });
            }

            break;
        }

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
                owned_elements.get_mut(&cluster_key).unwrap().push(vertex);
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
                free_elements: core::mem::take(&mut step_fixes_elements),
                on_frontiers: on_frontiers.clone(),
                owned_elements: owned_elements_before,
                frontier_elements: frontier_elements.clone(),
            });
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
            // merged into this.
            let old_cluster_key = owning_cluster.insert(vertex, cluster_key).unwrap();
            if old_cluster_key != cluster_key {
                if let Some(old_cluster_owned_elements) = owned_elements.remove(&old_cluster_key) {
                    owned_elements
                        .get_mut(&cluster_key)
                        .unwrap()
                        .extend(old_cluster_owned_elements.into_iter());
                }
                if let Some(frontier_elements) = frontier_elements.remove(&old_cluster_key) {
                    for element in frontier_elements {
                        let idx = on_frontiers
                            .get(&element)
                            .unwrap()
                            .iter()
                            .position(|&c| c == old_cluster_key)
                            .unwrap();
                        on_frontiers.get_mut(&element).unwrap().swap_remove(idx);
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

        if subgraph.len() - frontier.len() <= 1 {
            // No simplification is possible, as the cluster's core is empty or consists of exactly
            // one element, i.e., no elements are merged. This is expected. We block this subgraph
            // so we don't find it again. (Note we also block subgraphs if a simplification *is*
            // made, but we block the simplified subgraph in that case.)
            //
            // TODO: perhaps when a simplification *is* made, remove old blocked clusters
            // containing any of the removed elements.
            blocked_clusters.push(subgraph.clone());
            continue;
        }

        let core = graph.add_element(0);
        owning_cluster.insert(core, cluster_key);
        vertices.insert(core);

        // Collect all the new (top-level) elements of this cluster, so we can block it to ensure
        // we don't find it as the next dense subgraph.
        let mut cluster_elements = Vec::with_capacity(1 + frontier.len());

        let mut total_frontier_vertex_dof = 0;
        let mut total_incoming_edge_valency = 0;
        // Create edges from frontier vertices to the inner vertex cluster
        for &vertex in &frontier {
            cluster_elements.push(vertex);
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

        cluster_elements.push(core);
        cluster_elements.sort_unstable();
        blocked_clusters.push(cluster_elements);

        if total_incoming_edge_valency > 0 {
            graph.elements[core.id as usize].dof =
                total_frontier_vertex_dof - total_incoming_edge_valency - D;
        } else {
            vertices.remove(&core);
        }
    }

    recombination_plan
}

/// Find an arbitrary, minimal dense subgraph.
///
/// The returned subgraph consists of all its elements' IDs, in ascending order.
///
/// This searches for a minimal dense subgraph such that
/// `(elements' degrees of freedom) - (constraints' valencies) < D + 1`. See also "Finding Solvable
/// Subsets of Constraints Graph" (1997) by C. M. Hoffmann, A. Lomonosov, and M. Sitharam. Here,
/// the parameter `D` is the degree-dependent parameter, and is the const-generic parameter to this
/// function. This generally is 3 for 2D and 6 for 3D.
///
/// The subgraph is minimal in the sense that it contains no proper subgraph upholding that
/// condition. Trivial subgraphs of single elements are not considered. It is not *minimum*: i.e.,
/// it is not necessarily the smallest such subgraph. Finding the smallest such subgraph is known
/// to be NP-hard.
///
/// This is based on the observation that a
fn dense<const D: i16>(
    graph: &Graph,
    blocked_subgraphs: &[Vec<ElementId>],
    available_edges: &HashSet<ConstraintId>,
    vertices: &HashSet<ElementId>,
) -> Vec<ElementId> {
    // Build the bipartite flow graph.
    let mut bmf = dinic::BipartiteMaxFlow::new();

    let mut element_mapping = HashMap::<ElementId, usize>::new();
    let mut constraint_mapping = HashMap::<ConstraintId, usize>::new();
    let mut element_mapping_rev = HashMap::<usize, ElementId>::new();
    let mut constraint_mapping_rev = HashMap::<usize, ConstraintId>::new();

    for &vertex in vertices.iter() {
        let element = &graph.elements[vertex.id as usize];
        let v_idx = bmf.add_v(element.dof);
        element_mapping.insert(vertex, v_idx);
        element_mapping_rev.insert(v_idx, vertex);
    }

    for &constraint_id in available_edges.iter() {
        let constraint = &graph.constraints[constraint_id.id as usize];
        if !constraint
            .incident_elements
            .as_slice()
            .iter()
            .all(|&element| element_mapping.contains_key(&element))
        {
            continue;
        }

        // Add the edge with 0 valency; its flow is pushed later.
        let c_idx = bmf.add_u(0);
        constraint_mapping.insert(constraint_id, c_idx);
        constraint_mapping_rev.insert(c_idx, constraint_id);

        for &element in constraint.incident_elements.as_slice() {
            let v_idx = *element_mapping.get(&element).unwrap();
            bmf.add_uv(c_idx, v_idx);
        }
    }

    // Build the subgraph G' one vertex at a time, pushing flows for all edges one-by-one that are
    // adjacent to the just-added vertex and vertices already in G'.
    let mut g = HashSet::<ElementId>::new();
    for &vertex in vertices.iter() {
        g.insert(vertex);
        let element = &graph.elements[vertex.id as usize];

        // Push flow for all edges that are incident to the element we just added and the subgraph
        // of elements already added.
        for &constraint_id in element.incident_constraints.iter() {
            if !available_edges.contains(&constraint_id) {
                continue;
            }
            let constraint = &graph.constraints[constraint_id.id as usize];
            if !constraint
                .incident_elements
                .as_slice()
                .iter()
                .all(|&element| g.contains(&element))
            {
                continue;
            }

            let c_idx = *constraint_mapping.get(&constraint_id).unwrap();
            let old_capacity = bmf.increase_su_capacity(c_idx, constraint.valency);

            // We first attempt to distribute the regular flow of all now-included constraints.
            // This will almost always succeed (i.e., will not find a dense subgraph), unless there
            // is an completely over-identified subgraph with respect to the global coordinate
            // system. However, this is still useful to perform, as it allows us to iteratively
            // build up the flow distribution that we can continue to build on in later iterations,
            // instead of having to start from scratch each iteration. (We cannot easily reuse the
            // flow distribution resulting from the next step, where we add flow to this individual
            // edge, as the flow distribution resulting from that is no longer valid once the
            // edge's flow is reset to its original value again.)
            let mf = bmf.max_flow();
            if mf < i32::from(constraint.valency - old_capacity) {
                let min_cut = bmf.min_cut_partition();
                let elements = min_cut
                    .s_side_v
                    .iter()
                    .map(|&v| *element_mapping_rev.get(&v).unwrap())
                    .collect();
                if !blocked_subgraphs.contains(&elements) {
                    return elements;
                }
            }

            // If we could distribute the previous flow, try again, but subtract the
            // degree-dependent constant K = -(D+1) from the constraint's capacity. In other words,
            // add D + 1. This conceptually fixes the constraint with respect to a local coordinate
            // system.
            let mut bmf_clone = bmf.clone();
            bmf_clone.increase_su_capacity(c_idx, D + 1);
            let mf = bmf_clone.max_flow();
            if mf < i32::from(D + 1) {
                // Unable to dstribute all flow. This means part of the graph is rigid or
                // over-rigid with respect to the global coordinate system.
                let min_cut = bmf_clone.min_cut_partition();
                let elements = min_cut
                    .s_side_v
                    .iter()
                    .map(|&v| *element_mapping_rev.get(&v).unwrap())
                    .collect();
                if !blocked_subgraphs.contains(&elements) {
                    return elements;
                }
            }
        }
    }

    vec![]
}

mod dinic {
    use alloc::{collections::VecDeque, vec, vec::Vec};

    #[derive(Clone, Copy)]
    struct Edge {
        to: usize,
        rev: usize, // index of reverse edge in g[to]
        cap: i32,   // residual capacity
    }

    /// A minimum cut of a [`BipartiteMaxFlow`] graph. This is equivalent to the max flow.
    #[derive(Debug)]
    pub(crate) struct MinCut {
        /// The value of the minimum cut. This is equal to the max flow.
        pub value: i32,
        /// All vertices in U on the S side of the cut.
        pub s_side_u: Vec<usize>,
        /// All vertices in U on the T side of the cut.
        pub t_side_u: Vec<usize>,
        /// All vertices in U on the S side of the cut.
        pub s_side_v: Vec<usize>,
        pub t_side_v: Vec<usize>,
        pub cut_s_to_u: Vec<usize>,
        pub cut_v_to_t: Vec<usize>,
    }

    /// An S-T bipartite max flow graph.
    ///
    /// This contains a source vertex S, a target (or "sink") vertex T, and vertex sets U and V.
    /// Vertices in U are only adjacent to S and vertices in V. Vertices in V are only adjacent to
    /// T and vertices in U.
    ///
    /// Edges from S to U as well as edges from V to T have a variable integer flow capacity. Edges
    /// from U to V have infinite capacity.
    #[derive(Clone)]
    pub(crate) struct BipartiteMaxFlow {
        /// Adjaceny list of the residual flow graph (i.e., for each edge, its capacity and reverse
        /// edge).
        g: Vec<Vec<Edge>>,
        /// The current Dinic level graph: for each node, its current distance from S.
        level_of: Vec<i32>,
        /// For each node, the next arc we'll be looking at. We only look at each edge once per
        /// BFS.
        next_arc_of: Vec<usize>,

        /// Mapping from u and v vertex indices to node indices.
        u_nodes: Vec<usize>,
        v_nodes: Vec<usize>,

        /// The index of the S->U edge for each vertex in U into the source vertex's list of edges, as in, `self.g[Self::S][idx]`.
        su_edges: Vec<usize>,
        /// The index of the V->T edge for each vertex in V into the source vertex's list of edges, as in, `self.g[v_node][idx]`.
        ///
        /// TODO: the first parameter is just `v_node = self.u_nodes[v]`, so could be dropped.
        vt_edges: Vec<(usize, usize)>,

        /// The original capacities of each S->U edge, for each vertex in U.
        cap_su: Vec<i16>,
        /// The original capacities of each V->T edge, for each vertex in V.
        cap_vt: Vec<i16>,
    }

    impl BipartiteMaxFlow {
        /// The node index of the source vertex.
        const S: usize = 0;
        /// The node index of the target (or "sink") vertex.
        const T: usize = 1;
        /// The "effectively infinite" value for the infinite-capacity edges.
        const INF: i32 = i32::MAX;

        pub(crate) fn new() -> Self {
            let n = 2; // Initially, the graph only contains the source and target vertices.
            Self {
                g: vec![Vec::new(); n],
                level_of: vec![0; n],
                next_arc_of: vec![0; n],
                u_nodes: Vec::new(),
                v_nodes: Vec::new(),
                su_edges: Vec::new(),
                vt_edges: Vec::new(),
                cap_su: Vec::new(),
                cap_vt: Vec::new(),
            }
        }

        fn push_node(&mut self) -> usize {
            let id = self.g.len();
            self.g.push(Vec::new());
            self.level_of.push(0);
            self.next_arc_of.push(0);
            id
        }

        fn add_edge(&mut self, u: usize, v: usize, c: i32) -> (usize, usize) {
            let rev_u = self.g[v].len();
            let fwd_u = self.g[u].len();
            self.g[u].push(Edge {
                to: v,
                rev: rev_u,
                cap: c,
            });
            self.g[v].push(Edge {
                to: u,
                rev: fwd_u,
                cap: 0,
            });
            (fwd_u, rev_u)
        }

        /// Add a new V-vertex with capacity to T. Returns its index in this [`BipartiteMaxFlow`].
        pub(crate) fn add_v(&mut self, cap_vt: i16) -> usize {
            let v_node = self.push_node();
            let (idx, _) = self.add_edge(v_node, Self::T, cap_vt as i32);
            self.v_nodes.push(v_node);
            self.vt_edges.push((v_node, idx));
            self.cap_vt.push(cap_vt);
            self.v_nodes.len() - 1
        }

        /// Add a new U-vertex with capacity from S. Returns its index in this [`BipartiteMaxFlow`].
        pub(crate) fn add_u(&mut self, cap_su: i16) -> usize {
            let u_node = self.push_node();
            let (idx, _) = self.add_edge(Self::S, u_node, cap_su as i32);
            self.u_nodes.push(u_node);
            self.su_edges.push(idx);
            self.cap_su.push(cap_su);
            self.u_nodes.len() - 1
        }

        /// Add a U -> V edge by their indices within this [`BipartiteMaxFlow`].
        pub(crate) fn add_uv(&mut self, u_idx: usize, v_idx: usize) {
            let u_node = self.u_nodes[u_idx];
            let v_node = self.v_nodes[v_idx];
            self.add_edge(u_node, v_node, Self::INF);
        }

        /// Increase the capacity of the source-to-u edge.
        ///
        /// `add` must be non-negative.
        pub(crate) fn increase_su_capacity(&mut self, u_idx: usize, add: i16) -> i16 {
            debug_assert!(add >= 0, "The added capacity must be non-negative.");
            debug_assert!(
                u_idx < self.u_nodes.len(),
                "The U vertex index must be in bounds."
            );

            // Update capacity on the Source-to-u edge.
            let eidx = self.su_edges[u_idx];
            self.g[Self::S][eidx].cap = self.g[Self::S][eidx].cap.saturating_add(i32::from(add));

            let old_cap = self.cap_su[u_idx];
            let new_cap = self.cap_su[u_idx] + add;
            self.cap_su[u_idx] = new_cap;
            old_cap
        }

        fn bfs(&mut self) -> bool {
            self.level_of.fill(-1);
            let mut q = VecDeque::new();
            self.level_of[Self::S] = 0;
            q.push_back(Self::S);
            while let Some(v) = q.pop_front() {
                for e in &self.g[v] {
                    if e.cap > 0 && self.level_of[e.to] < 0 {
                        self.level_of[e.to] = self.level_of[v] + 1;
                        q.push_back(e.to);
                    }
                }
            }
            self.level_of[Self::T] >= 0
        }

        fn dfs(&mut self, v: usize, f: i32) -> i32 {
            if v == Self::T {
                return f;
            }
            while self.next_arc_of[v] < self.g[v].len() {
                let arc = self.next_arc_of[v];
                let (to, rev, cap) = {
                    let e = &self.g[v][arc];
                    (e.to, e.rev, e.cap)
                };
                if cap > 0 && self.level_of[v] < self.level_of[to] {
                    let d = self.dfs(to, i32::min(f, cap));
                    if d > 0 {
                        self.g[v][arc].cap -= d;
                        let rev_idx = rev;
                        self.g[to][rev_idx].cap += d;
                        return d;
                    }
                }
                self.next_arc_of[v] += 1;
            }
            0
        }

        /// Incremental max-flow, reusing existing flow distribution and residual state.
        ///
        /// This calculates a maximum distribution of possible flow from S to T not exceeding any
        /// of the edge capacities. Note this only needs to account for capacities of edges from S
        /// to U and edges from V to T. Edges between U and V have infinity capacity.
        ///
        /// Returns how much additional flow was pushed.
        pub(crate) fn max_flow(&mut self) -> i32 {
            let mut flow_inc = 0;
            while self.bfs() {
                self.next_arc_of.fill(0);
                loop {
                    let pushed = self.dfs(Self::S, i32::MAX);
                    if pushed == 0 {
                        break;
                    }
                    flow_inc += pushed;
                }
            }
            flow_inc
        }

        fn reachable_from_s(&self) -> Vec<bool> {
            let mut seen = vec![false; self.g.len()];
            let mut q = VecDeque::new();
            seen[Self::S] = true;
            q.push_back(Self::S);
            while let Some(v) = q.pop_front() {
                for e in &self.g[v] {
                    if e.cap > 0 && !seen[e.to] {
                        seen[e.to] = true;
                        q.push_back(e.to);
                    }
                }
            }
            seen
        }

        /// After having performed a [`Self::max_flow`], find the equivalent minimum cut of this
        /// graph.
        pub(crate) fn min_cut_partition(&self) -> MinCut {
            let seen = self.reachable_from_s();

            let mut s_side_u = Vec::new();
            let mut t_side_u = Vec::new();
            for (ui, &node) in self.u_nodes.iter().enumerate() {
                if seen[node] {
                    s_side_u.push(ui);
                } else {
                    t_side_u.push(ui);
                }
            }

            let mut s_side_v = Vec::new();
            let mut t_side_v = Vec::new();
            for (vi, &node) in self.v_nodes.iter().enumerate() {
                if seen[node] {
                    s_side_v.push(vi);
                } else {
                    t_side_v.push(vi);
                }
            }

            let mut cut_s_to_u = Vec::new();
            for (ui, &idx) in self.su_edges.iter().enumerate() {
                let e = &self.g[Self::S][idx];
                let to = e.to;
                if seen[Self::S] && !seen[to] {
                    cut_s_to_u.push(ui);
                }
            }
            let mut cut_v_to_t = Vec::new();
            for (vi, &(v_node, idx)) in self.vt_edges.iter().enumerate() {
                let _e = &self.g[v_node][idx]; // to == T
                if seen[v_node] && !seen[Self::T] {
                    cut_v_to_t.push(vi);
                }
            }

            let mut value: i32 = 0;
            for &u in &cut_s_to_u {
                value += self.cap_su[u] as i32;
            }
            for &v in &cut_v_to_t {
                value += self.cap_vt[v] as i32;
            }

            MinCut {
                value,
                s_side_u,
                t_side_u,
                s_side_v,
                t_side_v,
                cut_s_to_u,
                cut_v_to_t,
            }
        }
    }
}
