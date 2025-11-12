// COLAMD, Copyright (c) 1998-2024, Timothy A. Davis and Stefan Larimore,
// All Rights Reserved.
// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: BSD-3-Clause

/// Whether the input matrix was ordered or jumbled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The input matrix was well-formed and in canonical representation: i.e., each column of the
    /// input matrix contained row indices in increasing order, with no duplicates.
    Ok,

    /// The input matrix was well-formed, but at least one column was (there were unsorted rows or
    /// duplicate entries).
    ///
    /// [`colamd`][crate::colamd()] had to do some extra work to sort the matrix first and remove
    /// duplicate entries, but it still was able to return a valid permutation.
    OkButJumbled {
        /// Highest numbered column that is unsorted or has duplicate entries
        highest_jumbled_column: i32,

        /// Last seen duplicate or unsorted row index
        last_seen_jumbled_row: i32,

        /// Number of duplicate or unsorted row indices
        jumbled_rows: i32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Statistics {
    /// Whether the matrix was ordered or jumbled.
    status: Status,

    /// The number of dense or empty rows that were ignored
    ignored_rows: i32,

    /// The number of dense or empty columns that were ignored
    ignored_columns: i32,

    /// The number of garabage collections performed.
    ///
    /// This can be excessively high if the length of `a` passed to [`colamd`][`crate::colamd()`]
    /// is close to the minimum required value.
    garbage_collections: i32,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Error {
    /// The first column-pointer is non-zero
    BadStartPointer {
        // The actual value of the first column pointer.
        got: i32,
    },

    /// The `a` matrix given to [`colamd`][crate::colamd()] is too small.
    ///
    /// Use [`colamd_recommended`][crate::colamd_recommended] to get a recommended length for `a`.
    ATooSmall {
        required_size: i32,
        actual_size: i32,
    },

    /// A column has a negative number of entries
    ColumnWithNegativeEntries {
        /// The column
        column: i32,

        /// The number of entries in the column
        entries: i32,
    },

    /// A row index is out of bounds
    RowIndexOutOfBounds {
        /// The column
        column: i32,

        /// The bad row
        row: i32,

        /// The number of rows in the matrix
        num_rows: i32,
    },
}

pub(crate) fn stats_to_result(stats: &[i32; 20]) -> Result<Statistics, Error> {
    // Conversion of the original status codes, defined in:
    // https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/9759b8c7538ecc92f9aa76b19fbf3f266432d113/COLAMD/Source/colamd.c#L272-L353
    match stats[3] {
        0 => Ok(Statistics {
            status: Status::Ok,
            ignored_rows: stats[0],
            ignored_columns: stats[1],
            garbage_collections: stats[2],
        }),
        1 => Ok(Statistics {
            status: Status::OkButJumbled {
                highest_jumbled_column: stats[4],
                last_seen_jumbled_row: stats[5],
                jumbled_rows: stats[6],
            },
            ignored_rows: stats[0],
            ignored_columns: stats[1],
            garbage_collections: stats[2],
        }),
        // Error codes -1 through -5 are not possible in our port (those are errors for null
        // pointers and negative signed integer inputs)
        -6 => Err(Error::BadStartPointer { got: stats[4] }),
        -7 => Err(Error::ATooSmall {
            required_size: stats[4],
            actual_size: stats[5],
        }),
        -8 => Err(Error::ColumnWithNegativeEntries {
            column: stats[4],
            entries: stats[5],
        }),
        -9 => Err(Error::RowIndexOutOfBounds {
            column: stats[4],
            row: stats[5],
            num_rows: stats[6],
        }),
        _ => unreachable!("non-existent status code"),
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::BadStartPointer { got } => f.write_fmt(core::format_args!(
                "The first column pointer must always be `0`. Got: `{got}`.",
            )),
            Error::ATooSmall {
                required_size,
                actual_size,
            } => f.write_fmt(core::format_args!(
                "The row indices and workspace slice given was too small. It must have a length of at least `{required_size}`. Got a slice of length `{actual_size}`.",
            )),
            Error::ColumnWithNegativeEntries {
                column,
                entries,
            } => f.write_fmt(core::format_args!(
                "Column `{column}` has a negative amount of entries: `{entries}`.",
            )),
            Error::RowIndexOutOfBounds {
                column,
                row,
                num_rows,
            } => f.write_fmt(core::format_args!(
                "Column `{column}` has an out-of-bounds row index `{row}`. Number of rows: `{num_rows}`.",
            )),
        }
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::{colamd, colamd_recommended};

    use super::Status;

    #[test]
    fn errors() {
        let a_len = colamd_recommended(5, 4, 3).unwrap();
        let mut a = vec![0; a_len];

        a[..5].copy_from_slice(&[0, 1, 1, 10, 2]);

        let err = colamd(4, 3, &mut a[..0], &mut [0, 2, 3, 5], None).unwrap_err();
        assert_eq!(
            &alloc::format!("{err}"),
            "The row indices and workspace slice given was too small. It must have a length of at least `57`. Got a slice of length `0`."
        );

        let err = colamd(4, 3, &mut a, &mut [0, 2, 3, 5], None).unwrap_err();
        assert_eq!(
            &alloc::format!("{err}"),
            "Column `2` has an out-of-bounds row index `10`. Number of rows: `4`."
        );

        a.fill(0);
        a[..5].copy_from_slice(&[0, 1, 1, 0, 2]);
        let err = colamd(4, 3, &mut a, &mut [2, 2, 3, 5], None).unwrap_err();
        assert_eq!(
            &alloc::format!("{err}"),
            "The first column pointer must always be `0`. Got: `2`."
        );

        a.fill(0);
        a[..5].copy_from_slice(&[0, 1, 1, 0, 2]);
        let err = colamd(4, 3, &mut a, &mut [0, 2, 0, 5], None).unwrap_err();
        assert_eq!(
            &alloc::format!("{err}"),
            "Column `1` has a negative amount of entries: `-2`."
        );
    }

    #[test]
    fn status() {
        let a_len = colamd_recommended(7, 4, 3).unwrap();
        let mut a = vec![0; a_len];

        // The matrix is fully well-formed and canonical.
        a[..7].copy_from_slice(&[0, 1, 1, 0, 1, 2, 3]);
        let stats = colamd(4, 3, &mut a, &mut [0, 2, 3, 7], None).unwrap();
        assert_eq!(stats.status, Status::Ok);

        // The last column has unordered rows.
        a.fill(0);
        a[..7].copy_from_slice(&[0, 1, 1, 2, 1, 0, 3]);
        let stats = colamd(4, 3, &mut a, &mut [0, 2, 3, 7], None).unwrap();
        assert_eq!(
            stats.status,
            Status::OkButJumbled {
                highest_jumbled_column: 2,
                last_seen_jumbled_row: 0,
                jumbled_rows: 2
            }
        );

        // The last column has a duplicate row.
        a.fill(0);
        a[..7].copy_from_slice(&[0, 1, 1, 0, 1, 1, 2]);
        let stats = colamd(4, 3, &mut a, &mut [0, 2, 3, 7], None).unwrap();
        assert_eq!(
            stats.status,
            Status::OkButJumbled {
                highest_jumbled_column: 2,
                last_seen_jumbled_row: 1,
                jumbled_rows: 1
            }
        );
    }
}
