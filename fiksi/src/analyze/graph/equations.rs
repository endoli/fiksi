// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{
    collections::{VecDeque, btree_set::BTreeSet},
    vec::Vec,
};
use core::{fmt::Debug, hash::Hash};
use hashbrown::{HashMap, HashSet};

use crate::collections::{CollectionExt, IndexMap};

type VariableId = u32;
type ExpressionId = u32;

/// A graph that allows discovery of its vertices and finding neighboring vertices.
trait VertexGraph {
    type VertexId: Copy + Eq + Hash;

    /// The exact number of vertices in this graph.
    fn len_vertices(&self) -> usize;

    /// Iterate all the vertices in this graph.
    fn vertices(&self) -> impl Iterator<Item = Self::VertexId>;

    /// Iterate over all neighboring vertices of `vertex`.
    fn neighbors(&self, vertex: Self::VertexId) -> Option<impl Iterator<Item = Self::VertexId>>;
}

/// A bipartite graph with partite set A and partite set B that only allows discovery from A to B.
///
/// This is only a collection of types. See [`BipartiteGraphAToB`] and [`BipartiteGraphBToA`] for
/// vertex discovery and edge traversal.
///
/// Vertices in set A can have edge to set vertices in set B and vice-versa. Vertices in set A
/// cannot have edges towards other vertices in set B, and the same holds for vertices in set B.
///
/// Such a graph could look like the following.
///
/// ```text
/// Vertex set A:  a1     a2     a3
///                 │     ╱│╲    ╱│
///                 │    ╱ │ ╲  ╱ │
///                 │   ╱  │  ╲╱  │
///                 │  ╱   │  ╱╲  │
///                 │ ╱    │ ╱  ╲ │
///                 │╱     │╱    ╲│
/// Vertex set B:  b1     b2     b3
/// ```
pub(crate) trait BipartiteGraph {
    type VertexIdA: Copy + Eq + Hash;
    type VertexIdB: Copy + Eq + Hash;
}

/// A [bipartite graph][`BipartiteGraph`] with partite set A and partite set B that allows
/// discovery from A to B.
///
/// See also [`BipartiteGraphAToB`].
pub(crate) trait BipartiteGraphAToB: BipartiteGraph {
    fn len_vertices_a(&self) -> usize;

    /// Iterate all the vertices in set A.
    fn vertices_a(&self) -> impl Iterator<Item = Self::VertexIdA>;

    /// Iterate over all neighboring vertices (in set B) of `vertex` in set A.
    fn neighbors_of_a(
        &self,
        vertex: Self::VertexIdA,
    ) -> Option<impl Iterator<Item = Self::VertexIdB>>;
}

/// A [bipartite graph][`BipartiteGraph`] with partite set A and partite set B that allows
/// discovery from B to A.
///
/// See also [`BipartiteGraphAToB`].
pub(crate) trait BipartiteGraphBToA: BipartiteGraph {
    fn len_vertices_b(&self) -> usize;

    /// Iterate all the vertices in set B.
    fn vertices_b(&self) -> impl Iterator<Item = Self::VertexIdB>;

    /// Iterate over all neighboring vertices (in set A) of `vertex` in set B.
    fn neighbors_of_b(
        &self,
        vertex: Self::VertexIdB,
    ) -> Option<impl Iterator<Item = Self::VertexIdA>>;
}

/// Represents a matching on a [`BipartiteGraph`].
///
/// A matching is a set of edges such that no vertex has more than one adjacent edge in the
/// matching.
///
/// As this is on a bipartite graph, this matches vertices in set A to vertices in set B.
#[derive(Debug, Clone)]
struct Matching<A, B> {
    /// Maps vertices in set A to their matched vertices in set B.
    a_to_b: IndexMap<A, B>,
    /// Maps vertices in set B to their matched vertices in set A.
    b_to_a: IndexMap<B, A>,
}

impl<U, V> Matching<U, V> {
    fn new() -> Self {
        Self {
            a_to_b: IndexMap::new(),
            b_to_a: IndexMap::new(),
        }
    }
}

impl<A, B> Matching<A, B>
where
    A: Copy + Eq + Hash,
    B: Copy + Eq + Hash,
{
    /// Get the cardinality (size) of the matching.
    pub(crate) fn cardinality(&self) -> usize {
        self.a_to_b.len()
    }

    /// Check if a vertex in set A is matched.
    pub(crate) fn is_a_matched(&self, a: A) -> bool {
        self.a_to_b.contains_key(&a)
    }

    /// Check if a vertex in set B is matched.
    pub(crate) fn is_b_matched(&self, b: B) -> bool {
        self.b_to_a.contains_key(&b)
    }

    /// Get the matched vertex in set B for a vertex in set A.
    pub(crate) fn get_matched_for_a(&self, a: A) -> Option<B> {
        self.a_to_b.get(&a).copied()
    }

    /// Get the matched vertex in set A for a vertex in set B.
    pub(crate) fn get_matched_for_b(&self, b: B) -> Option<A> {
        self.b_to_a.get(&b).copied()
    }
}

/// A [bipartite](BipartiteGraph) system of equations graph.
///
/// The two vertex sets are 1) the variables of the system and 2) the expressions (constraints) on
/// the variables.
pub(crate) struct ExpressionGraph {
    // pub(crate) inner: BipartiteGraph<VariableId, ExpressionId>,
    pub(crate) variables: Vec<Vec<ExpressionId>>,
    pub(crate) expressions: Vec<Vec<VariableId>>,
}

/// Expressions that are to be solved together, and the variables that can be updated to do so.
pub(crate) struct StronglyConnectedExpressions {
    pub(crate) free_variables: Vec<VariableId>,
    pub(crate) expressions: Vec<ExpressionId>,
}

impl ExpressionGraph {
    pub(crate) fn new() -> Self {
        Self {
            variables: Vec::new(),
            expressions: Vec::new(),
        }
    }

    pub(crate) fn insert_variables<const N: usize>(&mut self) {
        self.variables
            .extend(core::iter::repeat_with(Vec::new).take(N));
    }

    pub(crate) fn insert_expression(&mut self, variables: impl IntoIterator<Item = VariableId>) {
        let id = self.expressions.len().try_into().unwrap();
        self.expressions.push(variables.into_iter().collect());

        for &var in self.expressions.last().unwrap() {
            self.variables[var as usize].push(id);
        }
    }

    /// Find the strongly connected expressions in this expression graph, considering
    /// `free_variables` as the variables that are allowed to be updated.
    ///
    /// This finds a sequence of sets of expressions (in topological order) that must be solved
    /// together, as well as the variables that are considered free at each step.
    pub(crate) fn find_strongly_connected_expressions<'g>(
        &'g self,
        free_variables: &'g BTreeSet<VariableId>,
    ) -> impl Iterator<Item = StronglyConnectedExpressions> + 'g {
        let masked = MaskedExpressionGraph {
            graph: self,
            free_variables,
        };
        let matching = find_maximum_matching(&masked);
        let sccs = find_strongly_connected_components(&MatchedBipartiteGraph {
            graph: &masked,
            matching: &matching,
        });

        sccs.into_iter().rev().map(move |expressions| {
            let mut scc_free_variables = HashSet::new();
            for expression in &expressions {
                let matched_var = matching.get_matched_for_b(*expression).unwrap();
                scc_free_variables.extend(
                    self.expressions[*expression as usize]
                        .iter()
                        .copied()
                        .filter(|&var| {
                            var == matched_var
                                || !matching.is_a_matched(var) && free_variables.contains(&var)
                        }),
                );
            }

            StronglyConnectedExpressions {
                free_variables: scc_free_variables.into_iter().collect(),
                expressions,
            }
        })
    }
}

/// An [`ExpressionGraph`] masked by free variables.
///
/// This represents a bipartite graph between variables and expressions, where the variables have
/// vertices iff they are in `free_variables`.
struct MaskedExpressionGraph<'g> {
    graph: &'g ExpressionGraph,
    free_variables: &'g BTreeSet<VariableId>,
}

impl BipartiteGraph for MaskedExpressionGraph<'_> {
    type VertexIdA = VariableId;
    type VertexIdB = ExpressionId;
}

impl BipartiteGraphAToB for MaskedExpressionGraph<'_> {
    fn len_vertices_a(&self) -> usize {
        self.free_variables.len()
    }

    fn vertices_a(&self) -> impl Iterator<Item = Self::VertexIdA> {
        self.free_variables.iter().copied()
    }

    fn neighbors_of_a(
        &self,
        vertex: Self::VertexIdA,
    ) -> Option<impl Iterator<Item = Self::VertexIdB>> {
        self.graph
            .variables
            .get(vertex as usize)
            .map(|expressions| expressions.iter().copied())
    }
}

impl BipartiteGraphBToA for MaskedExpressionGraph<'_> {
    fn len_vertices_b(&self) -> usize {
        self.graph.expressions.len()
    }

    fn vertices_b(&self) -> impl Iterator<Item = Self::VertexIdB> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "We don't allow this many expressions."
        )]
        {
            0..self.graph.expressions.len() as u32
        }
    }

    fn neighbors_of_b(
        &self,
        vertex: Self::VertexIdB,
    ) -> Option<impl Iterator<Item = Self::VertexIdA>> {
        self.graph
            .expressions
            .get(vertex as usize)
            .map(|variables| {
                variables
                    .iter()
                    .copied()
                    .filter(|v| self.free_variables.contains(v))
            })
    }
}

/// Finds a maximum cardinality matching of `graph`.
///
/// If the given bipartite graph is interpreted as an equation graph with variables in set A and
/// equations in set B, then the maximum matching can be interpreted as an assignment of which
/// equations calculate (and fix) which variables.
fn find_maximum_matching<G: BipartiteGraphAToB>(graph: &G) -> Matching<G::VertexIdA, G::VertexIdB> {
    hopcroft_karp::find_maximum_matching(graph)
}

mod hopcroft_karp {
    use super::*;

    /// Finds a maximum cardinality matching of `graph` using the Hopcroft-Karp algorithm.
    pub(crate) fn find_maximum_matching<G: BipartiteGraphAToB>(
        graph: &G,
    ) -> Matching<G::VertexIdA, G::VertexIdB> {
        let mut matching = Matching::new();
        let mut distance = IndexMap::<G::VertexIdA, u32>::with_capacity(graph.len_vertices_a());
        for a in graph.vertices_a() {
            distance.insert(a, u32::MAX);
        }

        // This is the distance of the "dummy vertex in A" that is considered to be connected to
        // every unmatched vertex in B, meaning it is initially connected to every vertex in B.
        let mut dummy_a_distance: u32 = u32::MAX;

        while bfs(graph, &matching, &mut distance, &mut dummy_a_distance) {
            for a in graph.vertices_a() {
                if !matching.is_a_matched(a) {
                    dfs(graph, &mut matching, &mut distance, dummy_a_distance, a);
                }
            }
        }

        matching
    }

    fn bfs<G: BipartiteGraphAToB>(
        graph: &G,
        matching: &Matching<G::VertexIdA, G::VertexIdB>,
        distance: &mut IndexMap<G::VertexIdA, u32>,
        dummy_a_distance: &mut u32,
    ) -> bool {
        let mut queue = VecDeque::new();

        for (a, a_distance) in distance.iter_mut() {
            if matching.is_a_matched(*a) {
                *a_distance = u32::MAX;
            } else {
                *a_distance = 0;
                queue.push_back(*a);
            }
        }

        *dummy_a_distance = u32::MAX;
        while let Some(a) = queue.pop_front() {
            let a_distance = distance.get(&a).copied().unwrap();

            if a_distance >= *dummy_a_distance {
                continue;
            }

            let new_dist = a_distance.saturating_add(1);
            for b in graph.neighbors_of_a(a).unwrap() {
                match matching.get_matched_for_b(b) {
                    None => {
                        if *dummy_a_distance == u32::MAX {
                            *dummy_a_distance = new_dist;
                        }
                    }
                    Some(matched_a) => {
                        if distance.get(&matched_a).copied().unwrap() == u32::MAX {
                            distance.insert(matched_a, new_dist);
                            queue.push_back(matched_a);
                        }
                    }
                }
            }
        }

        *dummy_a_distance != u32::MAX
    }

    fn dfs<G: BipartiteGraphAToB>(
        graph: &G,
        matching: &mut Matching<G::VertexIdA, G::VertexIdB>,
        distance: &mut IndexMap<G::VertexIdA, u32>,
        dummy_a_distance: u32,
        a: G::VertexIdA,
    ) -> bool {
        let a_distance_plus_one = distance.get(&a).copied().unwrap().saturating_add(1);

        for b in graph.neighbors_of_a(a).unwrap() {
            match matching.get_matched_for_b(b) {
                None => {
                    if dummy_a_distance == a_distance_plus_one {
                        matching.a_to_b.insert(a, b);
                        matching.b_to_a.insert(b, a);
                        return true;
                    }
                }
                Some(matched_a) => {
                    if *distance.get(&matched_a).unwrap() == a_distance_plus_one
                        && dfs(graph, matching, distance, dummy_a_distance, matched_a)
                    {
                        matching.a_to_b.insert(a, b);
                        matching.b_to_a.insert(b, a);
                        return true;
                    }
                }
            }
        }

        distance.insert(a, u32::MAX);
        false
    }
}

/// Interpret a bipartite graph with a matching that directs edges in a specific way.
///
/// All edges are directed from set A to set B. Further, in two cases edges are bidirectional: if
/// an edge between A and B is matched, it is bidirectional. If a vertex in A is unsaturated (none
/// of its edges are matched), all its edges are bidirectional.
///
/// This is a useful interpretation for, e.g., systems of equations, where variables are in set A
/// and expressions in set B. Expressions are pointed to by the variables they require as inputs.
/// Expressions points to variables they calculate.
///
/// Unmatched edges in A can be interpreted as being "free variables," in that there is no equation
/// directly calculating them; however, these variables may still need to be updated to make the
/// full system satisfiable. We cannot keep them fixed. In particular, these variables need to be
/// updated such that all equations they are an input for can be satisfied. By making edges of
/// unmatched vertices in A bidirectional, this has the effect of merging all equations coupled
/// through overlap in their free variables into a single strongly connected component.
struct MatchedBipartiteGraph<'g, G: BipartiteGraph> {
    graph: &'g G,
    matching: &'g Matching<G::VertexIdA, G::VertexIdB>,
}

impl<G: BipartiteGraphAToB + BipartiteGraphBToA> VertexGraph for MatchedBipartiteGraph<'_, G> {
    type VertexId = G::VertexIdB;

    fn len_vertices(&self) -> usize {
        self.matching.b_to_a.len()
    }

    fn vertices(&self) -> impl Iterator<Item = Self::VertexId> {
        self.matching.b_to_a.keys().copied()
    }

    fn neighbors(&self, vertex: Self::VertexId) -> Option<impl Iterator<Item = Self::VertexId>> {
        let matched_a = *self.matching.b_to_a.get(&vertex).unwrap();

        self.graph.neighbors_of_b(vertex).map(|neighbors| {
            neighbors
                .filter(move |&a| a == matched_a || !self.matching.is_a_matched(a))
                .flat_map(|a| self.graph.neighbors_of_a(a).unwrap())
                .filter(move |&b| b != vertex && self.matching.b_to_a.contains_key(&b))
        })
    }
}

/// Find strongly connected components in a [`VertexGraph`].
///
/// The strongly connected components are returned as a vec of vecs, where each inner vec is a
/// strongly connected component. The strongly connected components are partially ordered according
/// to the reverse-topogical order of the DAG of strongly connected components. The vertices within
/// a component are not in any particular order.
///
/// It is a logic error to call this with a graph that contains 2^32 or more vertices.
fn find_strongly_connected_components<G: VertexGraph>(graph: &G) -> Vec<Vec<G::VertexId>> {
    tarjan::tarjan_pearce(graph)
}

mod tarjan {
    use super::*;

    /// Find strongly connected components in `graph` using David J. Pearce's version of Robert E.
    /// Tarjan's algorithm.
    ///
    /// This implements Algorithm 3 of "A space-efficient algorithm for finding strongly connected
    /// components" (2016).
    pub(crate) fn tarjan_pearce<G: VertexGraph>(graph: &G) -> Vec<Vec<G::VertexId>> {
        let mut sccs = Vec::new();
        let vertices = graph.vertices();

        // Create the index-map and stack with enough capacity for the lower-bound of the vertex iterator.
        let mut index = 1;
        let mut c = u32::try_from(graph.len_vertices())
            .expect("Graph should have fewer than 2^32 vertices")
            .wrapping_sub(1); // Wrapping is fine here, as if there are no vertices, we don't do any work below.
        let mut root_index = HashMap::with_capacity(graph.len_vertices());
        let mut stack = Vec::with_capacity(graph.len_vertices());

        for vertex in vertices {
            if root_index.get(&vertex).is_none() {
                visit(
                    graph,
                    vertex,
                    &mut index,
                    &mut c,
                    &mut root_index,
                    &mut stack,
                    &mut sccs,
                );
            }
        }

        debug_assert_eq!(
            u32::try_from(graph.len_vertices())
                .unwrap()
                .wrapping_sub(1)
                .wrapping_sub(c),
            sccs.len().try_into().unwrap(),
            "Bookkeeping test: when we stop, `c` should have been decremented as many times as we've seen SCCs."
        );

        sccs
    }

    fn visit<G: VertexGraph>(
        graph: &G,
        vertex: G::VertexId,
        index: &mut u32,
        c: &mut u32,
        root_index: &mut HashMap<G::VertexId, u32>,
        stack: &mut Vec<G::VertexId>,
        sccs: &mut Vec<Vec<G::VertexId>>,
    ) {
        let mut root = true;
        let mut vertex_index = *index;
        root_index.insert(vertex, vertex_index);
        *index += 1;

        for neighbor in graph.neighbors(vertex).unwrap() {
            if root_index.get(&neighbor).is_none() {
                visit(graph, neighbor, index, c, root_index, stack, sccs);
            }

            let neighbor_index = *root_index.get(&neighbor).unwrap();
            if neighbor_index < vertex_index {
                vertex_index = neighbor_index;
                root_index.insert(vertex, vertex_index);
                root = false;
            }
        }

        if root {
            let mut scc = Vec::new();
            scc.push(vertex);

            *index -= 1;
            while let Some(top) = stack.last() {
                if vertex_index > *root_index.get(top).unwrap() {
                    break;
                }

                // This means `w` is in the same SCC as `vertex`.
                let w = stack.pop().unwrap();
                scc.push(w);

                root_index.insert(w, *c);
                *index -= 1;
            }
            vertex_index = *c;
            root_index.insert(vertex, vertex_index);

            if *c == 0 {
                debug_assert!(
                    stack.is_empty(),
                    "`c` will wrap when the number of SCCs is equal to the number of vertices and we've seen the last vertex, which means that stack must be empty."
                );
            }
            *c = c.wrapping_sub(1);

            sccs.push(scc);
        } else {
            stack.push(vertex);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximum_matching() {
        let mut graph = ExpressionGraph::new();

        // Create an expression graph with three variables and three expressions, such that the
        // expressions operate on variables (0,1), (0,2) and (1) respectively.
        graph.insert_variables::<3>();
        graph.insert_expression([0, 1]);
        graph.insert_expression([0, 2]);
        graph.insert_expression([1]);

        // All variables are considered free.
        let masked_graph = MaskedExpressionGraph {
            graph: &graph,
            free_variables: &([0, 1, 2].into_iter().collect()),
        };
        let matching = find_maximum_matching(&masked_graph);

        // Should find a perfect matching of size 3.
        assert_eq!(matching.cardinality(), 3);

        // Verify it's a valid matching (no conflicts)
        let mut used_exprs = IndexMap::new();
        for (var, expr) in &matching.a_to_b {
            assert!(graph.variables[*var as usize].contains(expr));
            assert!(used_exprs.insert(expr, var).is_none());
        }
    }
}
