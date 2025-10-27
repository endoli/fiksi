// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;

use crate::Triplet;

/// A coordinate list matrix.
///
/// This is effectively a specification of matrix size and a collection of [triplets](Triplet)
/// representing the matrix entries. Matrix cells without a corresponding triplet are implicitly
/// zero. Matrix cells with multiple triplets have as value the sum of the triplet values.
///
/// This format is useful for constructing matrices. It can be converted to different
/// representations (like [`SparseColMat`][crate::SparseColMat]).
#[derive(Clone, Debug)]
pub struct CooMat<T> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) row_indices: Vec<usize>,
    pub(crate) col_indices: Vec<usize>,
    pub(crate) values: Vec<T>,
}

impl<T> CooMat<T> {
    /// Construct a new [`CooMat`] with the given number of rows and columns.
    ///
    /// Values can be added using [`CooMat::push_triplet`].
    #[inline(always)]
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self::with_capacity(nrows, ncols, 0)
    }

    /// Construct a new [`CooMat`] with the given number of rows and columns, and with at least
    /// the specified `capacity`.
    ///
    /// The resulting matrix will be able to hold at least `capacity` [triplets][`Triplet`] without
    /// reallocating.
    ///
    /// Values can be added using [`CooMat::push_triplet`].
    #[inline]
    pub fn with_capacity(nrows: usize, ncols: usize, capacity: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_indices: Vec::with_capacity(capacity),
            col_indices: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    /// The shape of this matrix as `(rows, cols)`.
    ///
    /// Also see [`CooMat::nrows`] and [`CooMat::ncols`].
    #[inline(always)]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// The number of rows this matrix has.
    ///
    /// Note some (or all) rows may have no structural values at all.
    ///
    /// Also see [`CooMat::shape`].
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// The number of columns this matrix has.
    ///
    /// Note some (or all) columns may have no structural values at all.
    ///
    /// Also see [`CooMat::shape`].
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Add a triplet to this matrix.
    ///
    /// If the triplet is outside of the current matrix (i.e., `row >= self.nrows` or
    /// `col >= self.ncols`), the matrix is expanded to contain it.
    #[inline]
    pub fn push_triplet(&mut self, row: usize, col: usize, value: T) {
        self.nrows = usize::max(self.nrows, row + 1);
        self.ncols = usize::max(self.ncols, col + 1);
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
    }

    /// Clear the matrix, removing all values and setting its shape to `(0, 0)`.
    ///
    /// Note that this method has no effect on the allocated capacity of the matrix.
    #[inline]
    pub fn clear(&mut self) {
        self.nrows = 0;
        self.ncols = 0;
        self.row_indices.clear();
        self.col_indices.clear();
        self.values.clear();
    }

    /// Iterate over all the values in this matrix as [triplets][Triplet].
    #[inline]
    pub fn triplet_iter<'a>(&'a self) -> impl 'a + Iterator<Item = Triplet<&'a T>> {
        self.row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
            .map(|((&row, &col), value)| Triplet { row, col, value })
    }

    /// Iterate over all the values in this matrix as [triplets][Triplet] with mutable values.
    #[inline]
    pub fn triplet_iter_mut<'a>(&'a mut self) -> impl 'a + Iterator<Item = Triplet<&'a mut T>> {
        self.row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter_mut())
            .map(|((&row, &col), value)| Triplet { row, col, value })
    }

    /// Transpose this matrix, reflecting values across the diagonal.
    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            nrows: self.ncols,
            ncols: self.nrows,
            values: self.values,
            row_indices: self.col_indices,
            col_indices: self.row_indices,
        }
    }

    /// Consume the matrix, getting the underlying storage.
    #[inline(always)]
    pub fn into_parts(self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        (self.row_indices, self.col_indices, self.values)
    }
}

impl<T> core::iter::FromIterator<Triplet<T>> for CooMat<T> {
    fn from_iter<I: IntoIterator<Item = Triplet<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (min_size, _) = iter.size_hint();
        let mut coo = Self::with_capacity(0, 0, min_size);

        for Triplet { row, col, value } in iter {
            coo.nrows = usize::max(coo.nrows, row + 1);
            coo.ncols = usize::max(coo.ncols, col + 1);
            coo.row_indices.push(row);
            coo.col_indices.push(col);
            coo.values.push(value);
        }

        coo
    }
}

#[cfg(test)]
mod tests {
    use crate::CooMat;

    #[test]
    fn matrix_shape_expands_to_fit_triplets() {
        let mut a = CooMat::new(0, 0);
        assert_eq!(a.shape(), (0, 0));

        a.push_triplet(3, 5, 1.);
        assert_eq!(a.shape(), (4, 6));

        a.push_triplet(7, 3, 1.);
        assert_eq!(a.shape(), (8, 6));

        a.push_triplet(1, 10, 1.);
        assert_eq!(a.shape(), (8, 11));
    }
}
