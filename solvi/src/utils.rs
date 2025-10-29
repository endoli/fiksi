// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utilities.

use alloc::{vec, vec::Vec};

/// Post-order traversal of a rooted forest specified through a parent slice.
///
/// Returns a permutation `post` of length `parents.len()` such that `post[k]` is the k-th node in
/// an arbitrary [post-order][wikipedia] tree traversal of `parents`.
///
/// - `parents[j] == usize::MAX` means `j` is a root, otherwise `parents[j]` must be in
///   `0..parents.len()`.
/// - Cycles are not allowed; i.e., traversing `parents`  from any `node` along the ancestor path
///   `node <- parents[node]` (recursively) must end up at a root node.
///
/// # Example
///
/// We have the following rooted tree.
///
/// ```text
///    6
///    |
///    5
///  / | \
/// 1  |  4
/// |  |  |
/// 0  2  3
/// ```
///
/// This yields a partial order.
///
/// - 0 < 1
/// - 1 < 5
/// - 2 < 5
/// - 3 < 4
/// - 4 < 5
/// - 5 < 6
///
/// Note that there is no specific ordering between, e.g., the nodes 1 and 2.
///
/// ```rust
/// let post = solvi::utils::post_order(&[1, 5, 5, 4, 5, 6, usize::MAX]);
/// std::println!("{:?}", post); // for example, [2, 0, 1, 3, 4, 5, 6]
/// ```
///
/// [wikipedia]: https://en.wikipedia.org/w/index.php?title=Tree_traversal&oldid=1290430553#Post-order,_LRN
pub fn post_order(parents: &[usize]) -> Vec<usize> {
    let n = parents.len();

    // Build child lists: `head[parent]` is the highest-numbered child of `parent`, `next[node]` is
    // the next highest-numbered sibling of `node`.
    let mut head = vec![usize::MAX; n];
    let mut next = vec![usize::MAX; n];
    for node in 0..n {
        let parent = parents[node];
        if parent != usize::MAX {
            next[node] = head[parent];
            head[parent] = node;
        }
    }

    // Depth-first traversal using `head` as an iterator pointer for each node’s child list.
    let mut post = Vec::with_capacity(n);
    let mut stack: Vec<usize> = Vec::new();

    // Traverse from a given start node `node`.
    #[inline(always)]
    fn dfs_from(
        mut node: usize,
        head: &mut [usize],
        next: &[usize],
        post: &mut Vec<usize>,
        stack: &mut Vec<usize>,
    ) {
        loop {
            // Recursively descend along first-unprocessed child.
            while head[node] != usize::MAX {
                // `node` has at least one child. We move to it, remembering to come back to `node`
                // by pushing it to the `stack`. If the child has any siblings, we set the child's
                // first (highest-numbered) sibling to be the first child of `node`, such that we
                // visit that sibling next when we come back to `node`.
                let child = head[node];
                head[node] = next[child];
                stack.push(node);
                node = child;
            }

            // No (more) children for `node`: emit `node`. This is in post-order.
            post.push(node);

            // Backtrack to the most recent ancestor. It may still have children, and should itself
            // still be emitted to the post-order.
            if let Some(parent) = stack.pop() {
                node = parent;
            } else {
                break; // finished this root
            }
        }
    }

    // Start traversal from all roots (i.e., `parents[node] == usize::MAX`).
    for (node, &parent) in parents.iter().enumerate() {
        if parent == usize::MAX {
            dfs_from(node, &mut head, &next, &mut post, &mut stack);
        }
    }

    debug_assert_eq!(
        post.len(),
        n,
        "The generated post order should have the same length as the number of columns"
    );

    post
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_post_order() {
        // Given the following column elimination tree.
        //
        //    11
        //    |
        //    10
        //    |
        //    9
        //    |
        //    8
        //    |
        //    7
        //    |
        //    6
        //    |
        //    5
        //  / | \
        // 1  |  4
        // |  |  |
        // 0  2  3
        //
        // We have the following partial order:
        // - 0 < 1
        // - 1 < 5
        // - 2 < 5
        // - 3 < 4
        // - 4 < 5
        // - 5 < 6
        // - 6 < 7... etc.
        //
        // Note that there is no specific ordering between, e.g., 1 and 2.
        let parents = &[1, 5, 5, 4, 5, 6, 7, 8, 9, 10, 11, usize::MAX];
        let post = post_order(parents);
        let mut places_in_post_order = [0; 12];
        for (place_in_post_order, &col) in post.iter().enumerate() {
            places_in_post_order[col] = place_in_post_order;
        }

        // The column elimination tree defines a partial ordering where leaf columns are visited
        // before their parents.
        //
        // Check whether the returned post-order adheres to the known partial ordering.
        let valid_post_ordering = places_in_post_order[0] < places_in_post_order[1]
            && places_in_post_order[1] < places_in_post_order[5]
            && places_in_post_order[2] < places_in_post_order[5]
            && places_in_post_order[3] < places_in_post_order[4]
            && places_in_post_order[4] < places_in_post_order[5]
            && places_in_post_order[5] < places_in_post_order[6]
            && places_in_post_order[6] < places_in_post_order[7]
            && places_in_post_order[7] < places_in_post_order[8]
            && places_in_post_order[8] < places_in_post_order[9]
            && places_in_post_order[9] < places_in_post_order[10]
            && places_in_post_order[10] < places_in_post_order[11];
        assert!(
            valid_post_ordering,
            "Post order should agree with the known partial column elimination order."
        );
    }
}
