// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{vec, vec::Vec};

use crate::CooMat;

/// The structure of a sparse column matrix.
///
/// This encodes the shape of the matrix and the structural non-zero values. The structure is
/// encoded in the [sparse column][csc] format, also known as CSC or CCS format. In this format,
/// values are stored in column-major order in memory. Structural zeroes (sparsity) are not not
/// stored.
///
/// To know which cells values belong to, there are two additional arrays for bookkeeping: one
/// encodes the rows of the matrix each value belongs to. The other encodes each column's starting
/// index into the value and row arrays.
///
/// [csc]: <https://en.wikipedia.org/w/index.php?title=Sparse_matrix&oldid=1300835532#Compressed_sparse_column_(CSC_or_CCS)>
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseColMatStructure {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,

    /// For each structural non-zero, stores its row index.
    pub(crate) row_indices: Vec<usize>,

    /// For each column in `0..num_cols`, stores the starting index of that column's structural
    /// non-zeros into `row_indices` (and any corresponding flat `values` array).
    pub(crate) column_pointers: Vec<usize>,
}

impl SparseColMatStructure {
    /// Get a the rows in the given column `col`.
    ///
    /// Panics if out of bounds.
    #[inline]
    pub fn index_column(&self, col: usize) -> &[usize] {
        let range = self.column_pointers[col]..self.column_pointers[col + 1];
        &self.row_indices[range]
    }

    /// The shape of this matrix as `(rows, cols)`.
    ///
    /// Also see [`SparseColMatStructure::nrows`] and [`SparseColMatStructure::ncols`].
    #[inline(always)]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// The number of rows this matrix has.
    ///
    /// Note some (or all) rows may have no structural values at all.
    ///
    /// Also see [`SparseColMatStructure::shape`].
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// The number of columns this matrix has.
    ///
    /// Note some (or all) columns may have no structural values at all.
    ///
    /// Also see [`SparseColMatStructure::shape`].
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.ncols
    }
}

/// A sparse column matrix.
///
/// The sparsity pattern of the matrix is explicitly encoded through a [`SparseColMatStructure`].
/// Only matrix cells included in this structure have a non-zero value, i.e., the values of all
/// matrix cells not included are implicitly zero. This can be an efficient method for storing and
/// performing operations on large matrices with relatively few non-zero values.
///
/// Note that structural values can still have an explicit zero value.
///
/// In the special case where all of the explicitly stored values are non-zero, this is a
/// _compressed_ sparse column matrix.
///
/// See also [`SparseColMatStructure`].
#[derive(Clone, Debug)]
pub struct SparseColMat<T> {
    /// The sparsity structure of this matrix, encoding for all the explicitly stored values. Any
    /// values not stored explicitly are implicitly zero.
    pub(crate) structure: SparseColMatStructure,

    /// The values of this matrix. These can be any value, including zero.
    pub(crate) values: Vec<T>,
}

impl<T> SparseColMat<T> {
    /// Get a tuple of the values and the rows of those values in the given column `col`.
    ///
    /// Panics if out of bounds.
    #[inline]
    pub fn index_column(&self, col: usize) -> (&[T], &[usize]) {
        let range = self.structure.column_pointers[col]..self.structure.column_pointers[col + 1];
        (
            &self.values[range.clone()],
            &self.structure.row_indices[range],
        )
    }

    /// The shape of this matrix as `(rows, cols)`.
    ///
    /// Also see [`SparseColMat::nrows`] and [`SparseColMat::ncols`].
    #[inline(always)]
    pub fn shape(&self) -> (usize, usize) {
        self.structure.shape()
    }

    /// The number of rows this matrix has.
    ///
    /// Note some (or all) rows may have no structural values at all.
    ///
    /// Also see [`SparseColMat::shape`].
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.structure.nrows()
    }

    /// The number of columns this matrix has.
    ///
    /// Note some (or all) columns may have no structural values at all.
    ///
    /// Also see [`SparseColMat::shape`].
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.structure.ncols()
    }
}

impl<T: core::ops::AddAssign + Copy> SparseColMat<T> {
    /// Construct a [`SparseColMat`] from a [`CooMat`].
    ///
    /// Values repeated at the same coordinate are summed. The resulting [`SparseColMat`] is not
    /// compressed and may contain numeric zeros.
    pub fn from_coo_mat(a: &CooMat<T>) -> Self {
        let nnz = a.values.len();

        let mut structure = SparseColMatStructure {
            nrows: a.nrows,
            ncols: a.ncols,
            row_indices: Vec::with_capacity(nnz),
            column_pointers: vec![0; a.ncols + 1],
        };

        let mut values_dedup = Vec::with_capacity(nnz);

        let mut indices = Vec::from_iter(0..nnz);
        indices.sort_unstable_by_key(|&idx| (a.col_indices[idx], a.row_indices[idx]));

        let mut prev_row = usize::MAX;
        let mut prev_col = usize::MAX;
        for idx in indices {
            let row = a.row_indices[idx];
            let col = a.col_indices[idx];
            if row == prev_row && col == prev_col {
                *values_dedup.last_mut().unwrap() += a.values[idx];
            } else {
                if col != prev_col {
                    for col in prev_col.wrapping_add(1)..=col {
                        structure.column_pointers[col] = values_dedup.len();
                    }
                }
                values_dedup.push(a.values[idx]);
                structure.row_indices.push(row);
            };

            prev_row = row;
            prev_col = col;
        }

        values_dedup.shrink_to_fit();
        structure.row_indices.shrink_to_fit();

        for col in prev_col.wrapping_add(1)..=a.ncols {
            structure.column_pointers[col] = values_dedup.len();
        }

        Self {
            structure,
            values: values_dedup,
        }
    }
}

impl<T: num_traits::real::Real> SparseColMat<T> {
    /// Solve the system `self * x = b` where the matrix `self` is assumed to be upper-triangular.
    ///
    /// The vector `b` must be of length `self.nrows()` and is overwritten with the result `x`.
    /// Lower-triangular values in `self` are ignored.
    ///
    /// Returns `true` iff the system was successfully solved. It is possible for the system to be
    /// unsolvable in the case of zeros on the diagonal.
    ///
    /// # Panics
    ///
    /// Panics when `b` is not of length `self.nrows()`.
    ///
    /// # Example
    ///
    /// For example, solving the linear system
    ///
    /// ```text
    /// [ 1  -2     | 2
    ///       4   1 | 1
    ///           2 | 4 ]
    /// ```
    ///
    /// has the following expected solution.
    ///
    /// ```text
    /// [  3/2
    ///   -1/4
    ///    2   ]
    /// ```
    ///
    /// Solving using `solvi`:
    ///
    /// ```rust
    ///     use solvi::{CooMat, SparseColMat};
    ///
    ///     let a = {
    ///         let mut mat = CooMat::new(3, 3);
    ///         mat.push_triplet(0, 0, 1.);
    ///         mat.push_triplet(0, 1, -2.);
    ///         mat.push_triplet(1, 1, 4.);
    ///         mat.push_triplet(1, 2, 1.);
    ///         mat.push_triplet(2, 2, 2.);
    ///         SparseColMat::from_coo_mat(&mat)
    ///     };
    ///     let mut b = [2., 1., 4.];
    ///     a.solve_upper_triangular_mut(&mut b);
    /// ```
    pub fn solve_upper_triangular_mut(&self, b: &mut [T]) -> bool {
        assert_eq!(
            self.nrows(),
            b.len(),
            "`b` must of length {}, but is of length {}",
            self.nrows(),
            b.len()
        );

        for i in (0..self.structure.nrows).rev() {
            let (values, rows) = self.index_column(i);
            let diag = if rows.last().copied() == Some(i) {
                *values.last().unwrap()
            } else {
                T::zero()
            };

            // In the case of a zero diagonal but also zero coefficient after subtraction of later
            // variable's contribution, the system is still solvable, though it no longer has a
            // unique solution. Here we choose to report this as failure.
            if T::is_zero(&diag) {
                return false;
            }

            let coeff = b[i] / diag;
            b[i] = coeff;

            for (value, row) in values
                .iter()
                .copied()
                .zip(rows.iter().copied())
                .take_while(|(_, row)| *row < i)
            {
                b[row] = b[row] - coeff * value;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use crate::{CooMat, SparseColMat};

    #[test]
    fn solve_upper_triangular() {
        const EPSILON: f64 = 1e-10;

        // Linear system:
        //
        // ```
        // [ 1  -2   0 | 2
        //       4   1 | 1
        //           2 | 4 ]
        // ```
        //
        // Expected solution:
        // ```
        // [  3/2
        //   -1/4
        //    2   ]
        // ```
        const X: [f64; 3] = [3. / 2., -1. / 4., 2.];

        let mut mat = CooMat::new(3, 3);
        mat.push_triplet(0, 0, 1.);
        mat.push_triplet(0, 1, -2.);
        mat.push_triplet(1, 1, 4.);
        mat.push_triplet(1, 2, 1.);
        mat.push_triplet(2, 2, 2.);

        let csc = SparseColMat::from_coo_mat(&mat);
        let mut b = [2., 1., 4.];
        csc.solve_upper_triangular_mut(&mut b);

        for (calculated, expected) in b.into_iter().zip(X) {
            assert!((calculated - expected).abs() < EPSILON);
        }
    }
}
