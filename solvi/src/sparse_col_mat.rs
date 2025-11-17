// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::{vec, vec::Vec};

use crate::TripletMat;

/// A formatting error.
///
/// These are the errors that can occur when checking the format of sparse data, like in
/// [`SparseColMatStructure::try_from_data`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SparseFormatError {
    /// Got a bad amount of pointers for the given matrix dimension.
    ///
    /// The number of pointers should be equal to the number of major axes plus 1. For a sparse
    /// column matrix: the number of column pointers should be equal to the number of columns plus
    /// one.
    BadNumberOfPointers {
        /// The number of expected pointers.
        expected: usize,

        /// The number of pointers actually given.
        got: usize,
    },

    /// Got a bad starting pointer. It should be `0`.
    BadStartPointer {
        /// The actual value of the first pointer.
        got: usize,
    },

    /// Got a bad starting pointer. It should be equal to the number of non-zeroes.
    BadEndPointer {
        /// The expected value of the end pointer.
        expected: usize,

        /// The actual value of the end pointer.
        got: usize,
    },

    /// The pointers were not ordered.
    ///
    /// For all slices, each slice end pointers should be greater than or equal to the slice start pointer.
    UnorderedPointer {
        /// The slice in which the error occurred.
        ///
        /// For a sparse column matrix, this is the column.
        slice: usize,

        /// The start pointer of this slice.
        start: usize,

        /// The end pointer of this slice.
        end: usize,
    },

    /// An index within a slice was unordered.
    UndorderedIndex {
        /// The slice in which the error occurred.
        ///
        /// For a sparse column matrix, this is the column.
        slice: usize,

        /// The position within the indices slice where the error occurred.
        ///
        /// For a sparse column matrix, this is the index into the row indices slice.
        idx: usize,

        /// The previous index value.
        ///
        /// For a sparse column matrix, this is the previous row index within this slice.
        prev: usize,

        /// The index value at the indicated position. This should be strictly greater than `prev`.
        ///
        /// For a sparse column matrix, this is the row index at position `idx`.
        got: usize,
    },

    /// An index within a slice was out of bounds.
    IndexOutOfBounds {
        /// The slice in which the error occurred.
        ///
        /// For a sparse column matrix, this is the column.
        slice: usize,

        /// The index into the indices slice.
        ///
        /// For a sparse column matrix, this is the index into the row indices slice.
        idx: usize,

        /// The index value.
        ///
        /// For a sparse column matrix, this is the row index at position `idx`.
        value: usize,

        /// The sparse length of the slice. All index values must be strictly smaller than this.
        ///
        /// For a sparse column matrix, this is the number of matrix rows.
        bound: usize,
    },
}

impl core::fmt::Display for SparseFormatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadNumberOfPointers { expected, got } => {
                f.write_fmt(core::format_args!(
                    "Got a bad amount of pointers. Expected: `{expected}` (size of major dimension + 1), got: `{got}`."
                ))
            }
            Self::BadStartPointer { got } => {
                f.write_fmt(core::format_args!(
                    "The start pointer should always be `0`. Got: `{got}`."
                ))
            }
            Self::BadEndPointer { expected, got } => {
                f.write_fmt(core::format_args!(
                    "The end pointer should always be equal to the amount of structural non-zeroes. Expected: `{expected}`, got: `{got}`."
                ))
            }
            Self::UnorderedPointer { slice, start, end } => {
                f.write_fmt(core::format_args!(
                    "The pointers for slice `{slice}` of the major axis are unordered. Start: `{start}`, end: `{end}`. Pointers must increase monotonically."
                ))
            }
            Self::UndorderedIndex { slice, idx, prev, got } => {
                f.write_fmt(core::format_args!(
                    "Index `{idx}` (in slice `{slice}`) is unordered. The previous index had value `{prev}`, this has value `{got}`. Indices must be strictly monotonically increasing (duplicates are not allowed)."
                ))
            }
            Self::IndexOutOfBounds { slice, idx, value, bound } => {
                f.write_fmt(core::format_args!(
                    "Index `{idx}` (in slice `{slice}`) is out of bounds. Length: `{bound}`, got index: `{value}`."
                ))
            }
        }
    }
}

impl core::error::Error for SparseFormatError {}

/// The structure of a sparse column matrix.
///
/// For a sparse column matrix with values, see [`SparseColMat`].
///
/// # Structure
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
/// For example, a 4x3 matrix with the following sparsity pattern (`x` marks a stored value)
///
/// ```text
///     1  2  3
/// 0 | x
/// 1 |       x
/// 2 | x  x
/// 3 |       x
/// ```
///
/// would look as follows.
///
/// ```text
/// row_indices:     [0, 2, 2, 1, 3]
/// column_pointers: [0, 2, 3, 5]
/// values (if any): [m00, m20, m22, m13, m34]
///                   --------  ---  --------  slices per column
/// ```
///
/// Column `j` owns the range `column_pointers[j]..column_pointers[j + 1]` in both `row_indices`
/// (and any parallel values array, usually stored together in [`SparseColMat`]). The final pointer
/// equals the number of structural entries.
///
/// If your matrices are large and sparse, this may be a good format. Because column values are
/// contiguous, algorithms that walk columns (such as factorization) can stream through memory.
/// Modifying the sparsity pattern is slow, as it requires shifting later row indices and values.
/// Finding a specific entry `(i,j)` is slow, requiring to scan through the row indices for a
/// specific column.
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
    /// Construct a new [`SparseColMatStructure`] from the given data.
    ///
    /// The formatting is checked according to the structure described
    /// [here][SparseColMatStructure]. An error is returned if the data is malformed.
    pub fn try_from_data(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        column_pointers: Vec<usize>,
    ) -> Result<Self, SparseFormatError> {
        let num_nonzero = row_indices.len();

        if column_pointers.len() != ncols + 1 {
            return Err(SparseFormatError::BadNumberOfPointers {
                expected: ncols + 1,
                got: column_pointers.len(),
            });
        }

        if column_pointers[0] != 0 {
            return Err(SparseFormatError::BadStartPointer {
                got: column_pointers[0],
            });
        }

        if column_pointers[ncols] != num_nonzero {
            return Err(SparseFormatError::BadEndPointer {
                expected: num_nonzero,
                got: column_pointers[ncols],
            });
        }

        for col in 0..ncols {
            let start = column_pointers[col];
            let end = column_pointers[col + 1];

            if end < start {
                return Err(SparseFormatError::UnorderedPointer {
                    slice: col,
                    start,
                    end,
                });
            }

            debug_assert!(
                end <= num_nonzero,
                "We checked that the column pointers are sorted and that the last column pointer points to the end of `num_nozero`, so are within `row_indices` bounds"
            );

            if end > start {
                let mut prev_row = row_indices[start];
                if prev_row >= nrows {
                    return Err(SparseFormatError::IndexOutOfBounds {
                        slice: col,
                        idx: start,
                        value: prev_row,
                        bound: nrows,
                    });
                }
                for (idx, &row) in row_indices.iter().enumerate().take(end).skip(start + 1) {
                    if row >= nrows {
                        return Err(SparseFormatError::IndexOutOfBounds {
                            slice: col,
                            idx,
                            value: row,
                            bound: nrows,
                        });
                    }

                    if row <= prev_row {
                        return Err(SparseFormatError::UndorderedIndex {
                            slice: col,
                            idx,
                            prev: prev_row,
                            got: row,
                        });
                    }

                    prev_row = row;
                }
            }
        }

        Ok(Self {
            nrows,
            ncols,
            row_indices,
            column_pointers,
        })
    }

    /// Get the index range of a column.
    ///
    /// This is useful for, e.g., indexing into a sparse matrix value array stored in parallel to
    /// this sparsity structure.
    ///
    /// Also see [`SparseColMat`] for a sparse matrix including its values.
    #[inline]
    #[track_caller]
    pub fn index_column_range(&self, col: usize) -> core::ops::Range<usize> {
        if col + 1 >= self.column_pointers.len() {
            panic!(
                "column index {col} out of range for matrix with {} columns",
                self.ncols()
            );
        }
        self.column_pointers[col]..self.column_pointers[col + 1]
    }

    /// Get a the rows in the given column `col`.
    ///
    /// Panics if out of bounds.
    #[inline]
    #[track_caller]
    pub fn index_column(&self, col: usize) -> &[usize] {
        if col + 1 >= self.column_pointers.len() {
            panic!(
                "column index {col} out of range for matrix with {} columns",
                self.ncols()
            );
        }
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
/// The sparsity pattern of the matrix is explicitly encoded through a [`SparseColMatStructure`]
/// (see its documentation for an explanation of the structure). Only matrix cells included in this
/// structure have a non-zero value, i.e., the values of all matrix cells not included are
/// implicitly zero. This can be an efficient method for storing and performing operations on large
/// matrices with relatively few non-zero values.
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
    #[track_caller]
    pub fn index_column(&self, col: usize) -> (&[T], &[usize]) {
        if col + 1 >= self.structure.column_pointers.len() {
            panic!(
                "column index {col} out of range for matrix with {} columns",
                self.ncols()
            );
        }
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
    /// Construct a [`SparseColMat`] from a [`TripletMat`].
    ///
    /// Values repeated at the same coordinate are summed. The resulting [`SparseColMat`] is not
    /// compressed and may contain numeric zeros.
    pub fn from_triplet_mat(a: &TripletMat<T>) -> Self {
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
    ///     use solvi::{TripletMat, SparseColMat};
    ///
    ///     let a = {
    ///         let mut mat = TripletMat::new(3, 3);
    ///         mat.push_triplet(0, 0, 1.);
    ///         mat.push_triplet(0, 1, -2.);
    ///         mat.push_triplet(1, 1, 4.);
    ///         mat.push_triplet(1, 2, 1.);
    ///         mat.push_triplet(2, 2, 2.);
    ///         SparseColMat::from_triplet_mat(&mat)
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
    use alloc::vec;

    use crate::{SparseColMat, SparseColMatStructure, TripletMat};

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

        let mut mat = TripletMat::new(3, 3);
        mat.push_triplet(0, 0, 1.);
        mat.push_triplet(0, 1, -2.);
        mat.push_triplet(1, 1, 4.);
        mat.push_triplet(1, 2, 1.);
        mat.push_triplet(2, 2, 2.);

        let csc = SparseColMat::from_triplet_mat(&mat);
        let mut b = [2., 1., 4.];
        csc.solve_upper_triangular_mut(&mut b);

        for (calculated, expected) in b.into_iter().zip(X) {
            assert!((calculated - expected).abs() < EPSILON);
        }
    }

    #[test]
    fn try_from_data() {
        assert!(
            SparseColMatStructure::try_from_data(
                4,
                5,
                vec![0, 1, 1, 2, 1, 2, 3],
                vec![0, 1, 2, 4, 4, 7]
            )
            .is_ok()
        );

        let r = SparseColMatStructure::try_from_data(
            4,
            5,
            vec![0, 1, 1, 2, 1, 2, 3],
            vec![0, 1, 3, 4, 4, 7],
        );
        let err = r.err().expect("Error");
        assert_eq!(
            &alloc::format!("{err}"),
            "Index `2` (in slice `1`) is unordered. The previous index had value `1`, this has value `1`. Indices must be strictly monotonically increasing (duplicates are not allowed)."
        );

        let r = SparseColMatStructure::try_from_data(
            4,
            5,
            vec![0, 1, 1, 2, 1, 2, 3],
            vec![0, 1, 3, 4, 4, 5],
        );
        let err = r.err().expect("Error");
        assert_eq!(
            &alloc::format!("{err}"),
            "The end pointer should always be equal to the amount of structural non-zeroes. Expected: `7`, got: `5`."
        );

        let r = SparseColMatStructure::try_from_data(
            4,
            5,
            vec![0, 10, 1, 2, 1, 2, 3],
            vec![0, 1, 2, 4, 4, 7],
        );
        let err = r.err().expect("Error");
        assert_eq!(
            &alloc::format!("{err}"),
            "Index `1` (in slice `1`) is out of bounds. Length: `4`, got index: `10`."
        );
    }
}
