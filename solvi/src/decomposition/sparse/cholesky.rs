// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sparse Cholesky decomposition implementations.

use alloc::{vec, vec::Vec};

use crate::{SparseColMatStructure, utils};

/// Returns the parent column each column depends on in the Cholesky factorization of `A` or
/// `A^T A`.
///
/// This forms the "elimination tree" of `A` or `A^T A`. When the const-generic parameter
/// `SYMMETRIC` is `true`, this finds the elimination tree for `A`, and finds the elimination tree
/// for `A^T A` otherwise. Note `A^T A` is never formed explicitly. When finding the elimination
/// tree for `A` itself, the matrix is assumed to be symmetric and only its upper-triangular part
/// is looked at.
///
/// This returns `parents`, where the parent column of `col` is given as `parents[col]`. A root
/// column `root` is encoded as `parents[root] == usize::MAX`. Note there may be multiple roots,
/// i.e., `parents` is a rooted forest of elimination trees.
///
/// This is based on the algorithm in Figure 2.2 of "Computing Row and Column Counts for Sparse QR
/// and LU factorization" (2001) by Gilbert et al., which finds the elimination tree for `A^T A`,
/// and is modified to also allow finding elimination trees for `A`.
pub fn elimination_tree<const SYMMETRIC: bool>(a: &SparseColMatStructure) -> Vec<usize> {
    let (m, n) = a.shape();

    // Keeps track of each column's parents.
    let mut parents: Vec<usize> = vec![usize::MAX; n];

    // Keeps track of column's ancestors towards their roots, doing some best-effort
    // path-compression, so we do not have to traverse every parent every time.
    let mut ancestors: Vec<usize> = vec![usize::MAX; n];

    // For the A^T A case we need `prev` (length `m`, i.e., one per row).
    let mut prev_col: Vec<usize> = if SYMMETRIC {
        // With `SYMMETRIC == true`, this is unused. With just a bit of luck the compiler notices
        // during monomorphization this is is entirely dead code and optimizes it out. In any case,
        // this doesn't actually allocate.
        Vec::new()
    } else {
        vec![usize::MAX; m]
    };

    for col in 0..n {
        // For each structural nonzero in column j:
        let rows = a.index_column(col);
        for &row in rows {
            // Determine the traversal start `i`:
            // - When finding the elimation tree for `A`: start at row index itself.
            // - For `A^T A`: start from the previous column seen in this row.
            let mut k = if SYMMETRIC { row } else { prev_col[row] };

            // Traverse path from k toward the root while k < col, doing path compression.
            while k != usize::MAX {
                if k >= col {
                    // TODO: can we additionally check !SYMMETRIC here, so this branch is
                    // compiled out for the asymmetric case?
                    break; // ignore lower part (col >= k) for triu(A) case
                }

                let col_next = ancestors[k]; // Next hop in the compressed path towards the root.
                ancestors[k] = col; // Set ancestor of `k` to `col`
                if col_next == usize::MAX {
                    parents[k] = col; // `k` had no ancestor yet, parent is `col`
                }
                k = col_next; // climb up
            }

            // When finding the tree for A^T A: remember that `row` most recently touched `col`.
            if !SYMMETRIC {
                prev_col[row] = col;
            }
        }
    }

    parents
}

/// Calculate the row and column counts of structural non-zero values in the `L^T`-factor of the
/// Cholesky-decomposition `LL^T = A^T A` given the sparse matrix `A`.
///
/// (Equivalently, this calculates the structure of the `L`-factor of the Cholesky-decomposition
/// `LL^T = AA^T`.)
///
/// This follows "Computing Row and Column Counts for Sparse QR and LU Factorization" (2001) by
/// Gilbert et al.
///
/// This performs the count calculation, without forming `A^T A`, in time near-linear in the amount
/// of non-zeros of `A`.
///
/// When `A` is of full rank, the structure of the `L^T`-factor is the same as the structure of the
/// `R` factor in the QR-decomposition `QR = A^T A`, and `R` is equal to `L^T` except for possible
/// sign differences: `LL^T = A^T A = (QR)^T (QR) = R^T Q^T Q R = R^T R`. (See "Predicting Fill for
/// Sparse Orthogonal Factorization" (1986) by Coleman et al.)
pub fn cholesky_l_factor_counts(
    a: &SparseColMatStructure,
    parents: &[usize],
    postorder: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let (m, n) = a.shape();
    assert_eq!(n, parents.len(), "parents length must be n");
    assert_eq!(n, postorder.len(), "postorder length must be n");

    // Compute each columns' levels (distance to root) in the elimination tree.
    let (levels, max_level) = utils::node_depth_levels(&parents);

    // Create inverse post-order mapping (from each place in the postorder to the column at that
    // place).
    let mut places_in_postorder = vec![0usize; n]; // place_in_postorder[orig] = postorder index
    for (place, &col) in postorder.iter().enumerate() {
        places_in_postorder[col] = place;
    }

    // Compute the size of each column's subtree (i.e., the count of itself plus its descendants).
    //
    // In the post-order, every descendant is placed before its ancestors, so we can do a single
    // forward pass adding the size of each column's subtree to it's parent's subtree size.
    let mut subtree_size = vec![1usize; n];
    for &j in postorder {
        let parent = parents[j];
        if parent != usize::MAX {
            subtree_size[parent] += subtree_size[j];
        }
    }

    // For each column j, the descendant column that is first in the post-order. For leaf nodes,
    // this is j itself.
    let mut first_descendants = vec![0usize; n];
    for (place_in_postorder, &j) in postorder.iter().enumerate() {
        let first_descendant_place_in_postorder = place_in_postorder + 1 - subtree_size[j];
        first_descendants[j] = postorder[first_descendant_place_in_postorder];

        #[cfg(debug_assertions)]
        if subtree_size[j] == 1 {
            assert_eq!(
                first_descendants[j], j,
                "For leaf nodes, its first descendant should be the node itself"
            );
        }
    }

    // For every row, the column index of that row's first nonzero entry (with columns in their
    // post ordering).
    let mut first_columns = vec![usize::MAX; m];
    for &j in postorder {
        let rows = a.index_column(j);
        for &i in rows {
            if first_columns[i] == usize::MAX {
                first_columns[i] = j;
            }
        }
    }

    // The higher-adjacency set `hadj_f[j]`: this is the set of cliques in A^T A whose
    // earliest-placed column in the postorder) is j.
    let mut hadj_f: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (place_in_postorder, &j) in postorder.iter().enumerate() {
        let rows = a.index_column(j);
        for &i in rows {
            let f = first_columns[i];
            debug_assert_ne!(
                f,
                usize::MAX,
                "As this row has a nonzero in `j` (and perhaps other columns as well), it should have a corresponding `first_column`"
            );
            if place_in_postorder > places_in_postorder[f] {
                hadj_f[f].push(j);
            }
        }
    }

    let mut w = vec![0isize; n];
    for j in 0..n {
        if subtree_size[j] == 1 {
            w[j] = 1; // columns that are leaves in the elimination tree start at 1
        } else {
            w[j] = 0; // non-leaves start at 0
        }
    }

    // Column counts start at 1 (diagonal entry of L^T)
    let mut col_counts = vec![1usize; n];

    // For the skeleton test and previous leaf per "u"
    let mut prev_nbr = vec![usize::MAX; n]; // 0 means "none yet"
    let mut prev_f = vec![usize::MAX; n]; // previous leaf j seen for "u"

    // Disjoint set union for Tarjan-like LCA over the elimination tree
    let mut dsu_parent: Vec<usize> = (0..n).collect();

    // A naive way to find the actual row indices is to collect them into a vec of vecs below.
    //
    // Below we collect the row indices into a flat vec. In debug builds, we also construct the
    // simpler vec-of-vecs and check whether the two agree.
    #[cfg(debug_assertions)]
    let mut row_indices_naive: Vec<Vec<usize>> = vec![Vec::new(); n];

    fn find(dsu: &mut [usize], x: usize) -> usize {
        if dsu[x] != x {
            dsu[x] = find(dsu, dsu[x]);
        }
        dsu[x]
    }

    // Loop over columns in elimination tree post-order (j = postorder[0], ..., postorder[n-1]),
    // i.e., process childs before their parents.
    for (j_place_in_postorder, &j) in postorder.iter().enumerate() {
        if parents[j] != usize::MAX {
            // j is not the root of a subtree
            w[parents[j]] -= 1
        }
        // Process neighbors u in hadj_f[j]
        let first_descendant_place_in_postorder = places_in_postorder[first_descendants[j]];
        for &u in &hadj_f[j] {
            // The following branch discards edges that do not affect the result, leaving only the
            // "skeleton graph." This is the optimization as used in "An efficient algorithm to
            // compute row and column counts for sparse Cholesky factorization" (1994) by Gilbert
            // et al. and "A compact row storage scheme for Cholesky factors using elimination
            // trees" (1986) by Liu.
            //
            // Add 1 to both sides (wrapping in the case of `prev_nbr`), as
            // `prev_nbr[u] == usize::MAX` encodes "u seen for the first time."
            if first_descendant_place_in_postorder + 1 > prev_nbr[u].wrapping_add(1) {
                // j is a leaf of the row-subtree for u
                w[j] += 1;
                let p_leaf = prev_f[u];
                if p_leaf != usize::MAX {
                    let q = find(&mut dsu_parent, p_leaf);
                    col_counts[u] += levels[j] - levels[q];
                    w[q] -= 1;

                    #[cfg(debug_assertions)]
                    {
                        let mut t = j;
                        while t != q {
                            row_indices_naive[u].push(t);
                            t = parents[t];
                        }
                    }
                } else {
                    col_counts[u] += levels[j] - levels[u];

                    #[cfg(debug_assertions)]
                    {
                        let mut t = j;
                        while t != u {
                            row_indices_naive[u].push(t);
                            t = parents[t];
                        }
                    }
                }
                prev_f[u] = j;
            }
            // Record that neighbor j of u has been seen.
            prev_nbr[u] = j_place_in_postorder;
        }

        // UNION(j, parent(j)) - link to parent after finishing j
        let parent = parents[j];
        if parent != usize::MAX {
            dsu_parent[j] = parent;
        }
    }

    // Accumulate tallies up the elimination tree.
    let mut row_counts_signed: Vec<isize> = w.clone();
    for j in 0..n {
        let parent = parents[j];
        if parent != usize::MAX {
            row_counts_signed[parent] += row_counts_signed[j];
        }
    }

    let row_counts = row_counts_signed
        .into_iter()
        .map(|row_count| {
            debug_assert!(
                row_count >= 0,
                "row count should be non-negative after accumulation",
            );
            row_count as usize
        })
        .collect();

    let row_indices = {
        let num_non_zero = col_counts.iter().sum();
        let mut row_indices = vec![0; num_non_zero];
        let mut stack = Vec::with_capacity(max_level);

        let mut marker = vec![0; m + n];
        let mut start = 0;
        for j in 0..n {
            marker[j] = j + 1;
            for &i in a.index_column(j) {
                let mut k = first_columns[i];
                while k != usize::MAX && k < j && marker[k] != j + 1 {
                    stack.push(k);
                    marker[k] = j + 1;
                    k = parents[k];
                }
            }

            debug_assert!(
                stack.len() <= max_level,
                "Stack should remain smaller than the maximum depth of the elimination tree"
            );

            let mut idx = start;
            while let Some(k) = stack.pop() {
                row_indices[idx] = k;
                idx += 1;
            }
            // The row indices as we've collected them above are not in general in order. Sort
            // them.
            row_indices[start..idx].sort_unstable();
            row_indices[idx] = j;

            start += col_counts[j];
        }

        row_indices
    };

    #[cfg(debug_assertions)]
    {
        let mut idx = 0;
        for j in 0..n {
            row_indices_naive[j].sort_unstable();
            for &row_idx in &row_indices_naive[j] {
                assert_eq!(row_idx, row_indices[idx]);
                idx += 1;
            }
            assert_eq!(j, row_indices[idx]);
            idx += 1;
        }
    }

    (row_counts, col_counts, row_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;

    #[test]
    fn known_matrix() {
        // Build a sparse matrix with the following non-zero structure.
        //
        // This is the same structure as the matrix in Fig. 1 of "Multifrontal Multithreaded
        // Rank-revealing Sparse QR Factorization" (2011) by Timothy A. Davis.
        //
        //      1  2  3  4  5  6  7  8  9  10 11 12
        // 1  | x  x                 x
        // 2  | x              x              x
        // 3  | x  x                 x        x
        // 4  |    x           x              x
        // 5  |    x           x              x
        // 6  |    x                 x
        // 7  |       x        x  x              x
        // 8  |       x                          x
        // 9  |       x        x     x
        // 10 |       x           x  x
        // 11 |       x           x
        // 12 |          x                    x
        // 13 |          x  x     x     x
        // 14 |          x  x           x     x
        // 15 |          x  x                 x
        // 16 |             x     x  x        x  x
        // 17 |             x  x                 x
        // 18 |             x  x     x  x     x
        // 19 |                x  x     x        x
        // 20 |                x     x           x
        // 21 |                   x           x  x
        // 22 |                   x  x  x     x  x
        // 23 |                      x  x  x  x  x
        let row_indices: [&'static [usize]; 12] = [
            &[0, 1, 2],
            &[0, 2, 3, 4, 5],
            &[6, 7, 8, 9, 10],
            &[11, 12, 13, 14],
            &[12, 13, 14, 15, 16, 17],
            &[1, 3, 4, 6, 8, 16, 17, 18, 19],
            &[6, 9, 10, 12, 15, 18, 20, 21],
            &[0, 2, 5, 8, 9, 15, 17, 19, 21, 22],
            &[12, 13, 17, 18, 21, 22],
            &[22],
            &[1, 2, 3, 4, 11, 13, 14, 15, 17, 20, 21, 22],
            &[6, 7, 15, 16, 18, 19, 20, 21, 22],
        ];
        let mut column_pointers = vec![0; 13];
        for col in 0..12 {
            column_pointers[col + 1] = column_pointers[col] + row_indices[col].len();
        }
        assert_eq!(
            *column_pointers.last().unwrap(),
            78,
            "The matrix has 78 structural non-zeros."
        );

        let a_structure = SparseColMatStructure {
            nrows: 23,
            ncols: 12,
            column_pointers,
            row_indices: row_indices
                .iter()
                .flat_map(|rows| rows.iter().copied())
                .collect(),
        };

        // The column elimination tree of this matrix is as follows.
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
        let parents = elimination_tree::<false>(&a_structure);
        assert_eq!(
            parents.as_slice(),
            &[1, 5, 5, 4, 5, 6, 7, 8, 9, 10, 11, usize::MAX],
            "The matrix should have the known tree structure as shown in Fig. 1 of the aforementioned paper by Timothy A. Davis.",
        );

        let post = &utils::post_order(&parents);
        cholesky_l_factor_counts(&a_structure, &parents, &post);
    }

    #[test]
    fn dense_known_matrix() {
        let a_structure = SparseColMatStructure {
            nrows: 3,
            ncols: 3,
            row_indices: vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            column_pointers: vec![0, 3, 6, 9],
        };

        let parents = elimination_tree::<false>(&a_structure);
        let post = &utils::post_order(&parents);
        let (row_counts, col_counts, row_indices) =
            cholesky_l_factor_counts(&a_structure, &parents, &post);

        assert_eq!(&row_counts, &[3, 2, 1]);
        assert_eq!(&col_counts, &[1, 2, 3]);
        assert_eq!(&row_indices, &[0, 0, 1, 0, 1, 2]);
    }
}
