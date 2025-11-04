// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sparse Cholesky decomposition implementations.

#![expect(
    clippy::needless_range_loop,
    reason = "We loop over named dimensions for clarity in multiple places."
)]

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

/// The row and column counts of structural non-zero values in the `L^T`-factor of the
/// Cholesky-decomposition `LL^T = A^T A` for a (sparse) matrix `A`.
///
/// Equivalently, these are the row and column counts for `L`-factor of the Cholesky-decomposition
/// `LL^T = AA^T`.
///
/// When `A` is of full rank, the structure of the `L^T`-factor is the same as the structure of the
/// `R` factor in the QR-decomposition `QR = A^T A`, and `R` is equal to `L^T` except for possible
/// sign differences: `LL^T = A^T A = (QR)^T (QR) = R^T Q^T Q R = R^T R`. (See "Predicting Fill for
/// Sparse Orthogonal Factorization" (1986) by Coleman et al.)
#[derive(Clone, Debug)]
pub struct CholeskyCounts {
    row_counts: Vec<usize>,
    col_counts: Vec<usize>,

    levels: Vec<usize>,
    first_columns: Vec<usize>,
}

/// The structural non-zero values in the `L^T`- and `H`-factors of the Cholesky-decomposition
/// `LL^T = A^T A` for a (sparse) matrix `A`, as well as a row permutation.
///
/// Equivalently, the "L"-structure is that of the `L`-factor of the transposed
/// Cholesky-decomposition `LL^T = AA^T`.
///
/// The `H`-factor is a matrix whose columns are the Householder vectors applied to find the
/// QR-decomposition of A. The row permutation is such that when applying the k-th Householder on
/// matrix A, the diagonal `A_kk` is structurally non-zero (unless A has deficient column rank).
///
/// When `A` is of full rank, the structure of the `L^T`-factor is the same as the structure of the
/// `R` factor in the QR-decomposition `QR = A^T A`, and `R` is equal to `L^T` except for possible
/// sign differences: `LL^T = A^T A = (QR)^T (QR) = R^T Q^T Q R = R^T R`. (See "Predicting Fill for
/// Sparse Orthogonal Factorization" (1986) by Coleman et al.)
#[derive(Clone, Debug)]
pub struct CholeskyStructure {
    /// The structure of the non-zero values in the sparse Householder matrix for the QR-decomposition
    /// `QR = A` for a (sparse) matrix `A`.
    ///
    /// These are the column counts and the row indices.
    pub l_structure: SparseColMatStructure,

    /// The row permutation, mapping from original row indices to their permuted index.
    pub row_permutation: Vec<usize>,

    /// The structure of the non-zero values in the sparse Householder matrix for the QR-decomposition
    /// `QR = A` for a (sparse) matrix `A`.
    ///
    /// These are the column counts and the row indices. The columns of the Householder matrix are
    /// the Householder vectors.
    ///
    /// The columns of the Householder matrix are the Householder vectors.
    pub h_structure: SparseColMatStructure,
}

impl CholeskyCounts {
    /// Calculate the row and column counts of the `L^T`-factor of the Cholesky-decomposition given
    /// the sparse matrix `A`.
    ///
    /// This follows "Computing Row and Column Counts for Sparse QR and LU Factorization" (2001) by
    /// Gilbert et al.
    ///
    /// This performs the count calculation, without forming `A^T A`, in time near-linear in the
    /// amount of non-zeros of `A`.
    pub fn build(a: &SparseColMatStructure, parents: &[usize], postorder: &[usize]) -> Self {
        let (m, n) = a.shape();
        assert_eq!(n, parents.len(), "parents length must be n");
        assert_eq!(n, postorder.len(), "postorder length must be n");

        // Compute each columns' levels (distance to root) in the elimination tree.
        let (levels, _) = utils::node_depth_levels(parents);

        // Create inverse post-order mapping (from each place in the postorder to the column at that
        // place).
        let mut places_in_postorder = vec![0_usize; n]; // place_in_postorder[orig] = postorder index
        for (place, &col) in postorder.iter().enumerate() {
            places_in_postorder[col] = place;
        }

        // Compute the size of each column's subtree (i.e., the count of itself plus its descendants).
        //
        // In the post-order, every descendant is placed before its ancestors, so we can do a single
        // forward pass adding the size of each column's subtree to it's parent's subtree size.
        let mut subtree_size = vec![1_usize; n];
        for &j in postorder {
            let parent = parents[j];
            if parent != usize::MAX {
                subtree_size[parent] += subtree_size[j];
            }
        }

        // For each column j, the descendant column that is first in the post-order. For leaf nodes,
        // this is j itself.
        let mut first_descendants = vec![0_usize; n];
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

        // Vertex weights: these are used to calculate the non-zero row counts in the L^T-factor.
        //
        // From Section 3.1 in "Computing Row and Column Counts for Sparse QR and LU Factorization", in
        // the Cholesky-factor L for decomposing LL^T = B, with B a symmetric positive definite matrix,
        // the non-zero count in column j is the number of "row subtrees" containing vertex j. Such a
        // row subtree exists for every row i in factor L, where the i-th row subtree is the connected
        // subgraph of the elimination tree of B containing the columns that have non-zeroes in row i,
        // and every leaf in the i-th row subtree corresponds to a non-zero column in the i-th row of
        // B).
        //
        // This is calculated efficiently following Section 3.1 in a single forward traversal of the
        // elimination tree, keeping "vertex weight" counts for each j simultaneously, for a B such
        // that L in LL^T = B has the same structure as the L in LL^T = A^T A.
        //
        // Note that we calculate it here for the L^T-factor, and not the L-factor. Hence the column
        // and row counts are flipped.
        let mut vertex_weights = vec![0_isize; n];
        for j in 0..n {
            if subtree_size[j] == 1 {
                vertex_weights[j] = 1; // columns that are leaves in the elimination tree start at 1
            } else {
                vertex_weights[j] = 0; // non-leaves start at 0
            }
        }

        // Column counts start at 1 (diagonal entry of L^T)
        let mut col_counts = vec![1_usize; n];

        // For the skeleton test and previous leaf per "u"
        let mut prev_nbr = vec![usize::MAX; n]; // 0 means "none yet"
        let mut prev_f = vec![usize::MAX; n]; // previous leaf j seen for "u"

        // Disjoint set union for Tarjan-like LCA over the elimination tree
        let mut dsu_parent: Vec<usize> = (0..n).collect();

        fn find(dsu: &mut [usize], x: usize) -> usize {
            if dsu[x] != x {
                dsu[x] = find(dsu, dsu[x]);
            }
            dsu[x]
        }

        // Loop over columns in elimination tree post-order (j = postorder[0], ..., postorder[n-1]),
        // i.e., process children before their parents.
        for (j_place_in_postorder, &j) in postorder.iter().enumerate() {
            if parents[j] != usize::MAX {
                // j is not the root of a subtree
                vertex_weights[parents[j]] -= 1;
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
                    vertex_weights[j] += 1;
                    let p_leaf = prev_f[u];
                    if p_leaf != usize::MAX {
                        let q = find(&mut dsu_parent, p_leaf);
                        col_counts[u] += levels[j] - levels[q];
                        vertex_weights[q] -= 1;
                    } else {
                        col_counts[u] += levels[j] - levels[u];
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
        for j in 0..n {
            let parent = parents[j];
            if parent != usize::MAX {
                vertex_weights[parent] += vertex_weights[j];
            }
        }

        let row_counts = vertex_weights
            .into_iter()
            .map(|row_count| {
                debug_assert!(
                    row_count >= 0,
                    "row count should be non-negative after accumulation",
                );
                row_count as usize
            })
            .collect();

        Self {
            row_counts,
            col_counts,

            levels,
            first_columns,
        }
    }

    /// The number of non-zero entries in each row in the `L^T` factor.
    pub fn row_counts(&self) -> &[usize] {
        &self.row_counts
    }

    /// The number of non-zero entries in each column in the `L^T` factor.
    pub fn col_counts(&self) -> &[usize] {
        &self.col_counts
    }

    /// Consume `self`, returning a tuple of the row and column counts.
    ///
    /// See also [`Self::row_counts`] and [`Self::col_counts`].
    pub fn into_parts(self) -> (Vec<usize>, Vec<usize>) {
        (self.row_counts, self.col_counts)
    }
}

impl CholeskyStructure {
    /// This find the counts and structure of the sparse Householder matrix as in "Computing Row and
    /// Column Counts for Sparse QR and LU factorization" (2001) by Gilbert et al.
    ///
    /// A row permutation is found ensuring the kth diagonal entry when applying the k-th
    /// Householder is non-zero using the method presented in Section 5.3 of "Direct Methods for
    /// Sparse Linear Systems" by Timothy A. Davis.
    pub fn build(
        a: &SparseColMatStructure,
        parents: &[usize],
        postorder: &[usize],
        cholesky: &CholeskyCounts,
    ) -> Self {
        let (m, n) = a.shape();

        let CholeskyCounts {
            col_counts,
            levels,
            first_columns,
            ..
        } = cholesky;

        // As per the method in "Direct Methods for Sparse Linear Systems", we ensure each column
        // can have its own row assigned to that column's diagonal entry, as this is required for
        // correct counting of the Householder matrix structure. We add fictitious, structurally
        // empty rows to A when it is rank deficient.
        let mut m_fictitious = m;
        let mut row_permutation = vec![usize::MAX; m + n];

        {
            let mut next = vec![0; m];
            let mut head = vec![usize::MAX; n];
            let mut tail = vec![usize::MAX; n];
            let mut nqueue = vec![0_isize; n];

            for i in (0..m).rev() {
                let k = first_columns[i];

                // This is an empty row (it has no structural non-zeroes).
                if k == usize::MAX {
                    continue;
                }

                if nqueue[k] == 0 {
                    tail[k] = i;
                }
                nqueue[k] += 1;

                next[i] = head[k];
                head[k] = i;
            }

            for k in 0..n {
                let i = if head[k] == usize::MAX {
                    // This column has no rows it can be assigned, meaning the matrix is structurally
                    // rank-deficient. The theorem we rely on to calculate the Householder structure
                    // requires a non-zero diagonal, which we can simulate by just adding a fictitious
                    // row.
                    let cur = m_fictitious;
                    m_fictitious += 1;
                    cur
                } else {
                    head[k]
                };

                row_permutation[i] = k;

                nqueue[k] -= 1;
                if nqueue[k] <= 0 {
                    continue;
                }

                let parent = parents[k];
                if parent != usize::MAX {
                    if nqueue[parent] == 0 {
                        tail[parent] = tail[k];
                    }
                    next[tail[k]] = head[parent];
                    head[parent] = next[i];
                    nqueue[parent] += nqueue[k];
                }
            }

            let mut k = n;
            for i in 0..m {
                if row_permutation[i] == usize::MAX {
                    row_permutation[i] = k;
                    k += 1;
                }
            }
        }

        let mut h_row_indices = vec![Vec::new(); n];
        let row_indices = {
            let num_non_zero = col_counts.iter().sum();
            let mut row_indices = vec![0; num_non_zero];
            let mut stack = Vec::with_capacity(n);

            let mut marker = vec![0; m + n];
            let mut start = 0;
            for j in 0..n {
                marker[j] = j + 1;
                h_row_indices[j].push(j);

                for &i in a.index_column(j) {
                    let mut k = first_columns[i];
                    while k != usize::MAX && k < j && marker[k] != j + 1 {
                        stack.push(k);
                        marker[k] = j + 1;
                        k = parents[k];
                    }

                    let i = row_permutation[i];
                    if i > j && marker[i] < j + 1 {
                        h_row_indices[j].push(i);
                        marker[i] = j + 1;
                    }
                }

                debug_assert!(
                    stack.len() <= n,
                    "Stack should remain smaller than the number of columns",
                );

                let mut idx = start;
                while let Some(k) = stack.pop() {
                    row_indices[idx] = k;
                    idx += 1;

                    // If the column `j` we're currently building factors for is the parent of a
                    // previous column that has an entry in this column's L-factor here, that
                    // previous column's rows that haven't yet been carried forward are carried
                    // forward.
                    if parents[k] == j {
                        let (h_row_indices0, h_row_indicesj) = h_row_indices.split_at_mut(j);
                        for &row in &*h_row_indices0[k] {
                            if marker[row] < j + 1 {
                                marker[row] = j + 1;
                                h_row_indicesj[0].push(row);
                            }
                        }
                    }
                }
                // The row indices as we've collected them above are not in general in order. Sort
                // them.
                row_indices[start..idx].sort_unstable();
                row_indices[idx] = j;

                start += col_counts[j];
            }

            row_indices
        };

        // For each column, the rows whose first nonzero is column j.
        let mut first_in_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..m {
            let j = first_columns[i];
            if j != usize::MAX {
                first_in_rows[j].push(i);
            }
        }

        let mut householder_row_counts = vec![0; m];
        let mut householder_vertex_weights = vec![0_isize; n];
        let row_roots = {
            // The root of the tree containing column j.
            //
            // We iterate in reverse post order, so we are guaranteed to visit parents before their
            // children.
            let mut root_of = vec![0; n];
            for &j in postorder.iter().rev() {
                let parent = parents[j];
                if parent == usize::MAX {
                    root_of[j] = j;
                } else {
                    root_of[j] = root_of[parent];
                }
            }

            let mut row_root = vec![0; m];
            for i in 0..m {
                if row_permutation[i] >= m {
                    // Fictitious row.
                    continue;
                }

                let first_column = first_columns[i];

                if first_column == usize::MAX {
                    // Empty row in `A`.
                    continue;
                }
                if row_permutation[i] < root_of[first_column] {
                    row_root[i] = row_permutation[i];
                } else {
                    row_root[i] = root_of[first_column];
                }
                householder_row_counts[i] = 1 + levels[first_column] - levels[row_root[i]];
            }

            row_root
        };

        for j in 0..n {
            for &i in &first_in_rows[j] {
                householder_vertex_weights[j] += 1;
                let row_root = row_roots[i];
                if row_root == usize::MAX {
                    continue;
                }
                if parents[row_roots[i]] != usize::MAX {
                    householder_vertex_weights[parents[row_roots[i]]] -= 1;
                }
            }
        }

        for h_row_indices in h_row_indices.iter_mut() {
            h_row_indices.sort_unstable();
        }

        let l_structure = SparseColMatStructure {
            nrows: n,
            ncols: n,
            column_pointers: core::iter::once(0)
                .chain(utils::prefix_sum(col_counts.iter().copied()))
                .collect(),
            row_indices,
        };
        let h_structure = SparseColMatStructure {
            nrows: m,
            ncols: n,
            column_pointers: core::iter::once(0)
                .chain(utils::prefix_sum(h_row_indices.iter().map(|col| col.len())))
                .collect(),
            row_indices: h_row_indices.into_iter().flatten().collect(),
        };
        Self {
            l_structure,
            row_permutation,
            h_structure,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // The column elimination tree of this matrix is as follows (from Fig. 1 of "Multifrontal
        // Multithreaded Rank-revealing Sparse QR Factorization" (2011) by Timothy A. Davis).
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

        let post = utils::post_order(&parents);
        let l_counts = CholeskyCounts::build(&a_structure, &parents, &post);
        let CholeskyStructure { l_structure, .. } =
            CholeskyStructure::build(&a_structure, &parents, &post, &l_counts);

        assert_eq!(&l_counts.row_counts, &[5, 4, 5, 5, 7, 6, 5, 5, 4, 3, 2, 1]);
        assert_eq!(&l_counts.col_counts, &[1, 2, 1, 1, 2, 5, 5, 7, 6, 3, 10, 9]);

        // The structure of the R-factor of the QR-decomposition is as follows (from Fig. 1 of
        // "Multifrontal Multithreaded Rank-revealing Sparse QR Factorization" (2011) by Timothy A.
        // Davis).
        //
        //      1  2  3  4  5  6  7  8  9  10 11 12
        // 1  | x  x           x     x        x
        // 2  |    x           x     x        x
        // 3  |       x        x  x  x           x
        // 4  |          x  x     x     x     x
        // 5  |             x  x  x  x  x     x  x
        // 6  |                x  x  x  x     x  x
        // 7  |                   x  x  x     x  x
        // 8  |                      x  x  x  x  x
        // 9  |                         x  x  x  x
        // 10 |                            x  x  x
        // 11 |                               x  x
        // 12 |                                  x
        #[rustfmt::skip]
        assert_eq!(
            &l_structure.row_indices,
            &[
                0,
                0, 1,
                2,
                3,
                3, 4,
                0, 1, 2, 4, 5,
                2, 3, 4, 5, 6,
                0, 1, 2, 4, 5, 6, 7,
                3, 4, 5, 6, 7, 8,
                7, 8, 9,
                0, 1, 3, 4, 5, 6, 7, 8, 9, 10,
                2, 4, 5, 6, 7, 8, 9, 10, 11,
            ]
        );
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
        let post = utils::post_order(&parents);
        let l_counts = CholeskyCounts::build(&a_structure, &parents, &post);
        let CholeskyStructure { l_structure, .. } =
            CholeskyStructure::build(&a_structure, &parents, &post, &l_counts);

        assert_eq!(&l_counts.row_counts, &[3, 2, 1]);
        assert_eq!(&l_counts.col_counts, &[1, 2, 3]);
        assert_eq!(&l_structure.row_indices, &[0, 0, 1, 0, 1, 2]);
    }

    #[test]
    fn sparse_known_matrix() {
        //       1   2   3   4   5   6
        // 1  |  x   x   x   x
        // 2  |          x   x   x   x
        // 3  |                  x
        // 4  |                      x
        // 5  |  x
        // 6  |      x
        // 7  |          x
        // 8  |              x
        // 9  |                  x
        // 10 |                      x
        let a = SparseColMatStructure {
            nrows: 10,
            ncols: 6,
            row_indices: vec![0, 4, 0, 5, 0, 1, 6, 0, 1, 7, 1, 2, 8, 1, 3, 9],
            column_pointers: vec![0, 2, 4, 7, 10, 13, 16],
        };
        let parents = elimination_tree::<false>(&a);
        let post = utils::post_order(&parents);

        let l_counts = CholeskyCounts::build(&a, &parents, &post);
        let CholeskyStructure { l_structure, .. } =
            CholeskyStructure::build(&a, &parents, &post, &l_counts);

        #[rustfmt::skip]
        assert_eq!(
            &l_structure.row_indices,
            &[
                0,
                0, 1,
                0, 1, 2,
                0, 1, 2, 3,
                2, 3, 4,
                2, 3, 4, 5
            ]
        );
    }
}
