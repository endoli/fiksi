// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sparse Cholesky decomposition implementations.

use alloc::{vec, vec::Vec};

use crate::SparseColMatStructure;

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
    }
}
