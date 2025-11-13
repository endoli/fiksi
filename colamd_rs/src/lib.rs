// COLAMD, Copyright (c) 1998-2024, Timothy A. Davis and Stefan Larimore,
// All Rights Reserved.
// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: BSD-3-Clause

//! Column approximate minimum degree ordering
//!
//! This is a Rust port of [COLAMD][COLAMD], providing the `colamd` and `symamd` functions for
//! computing fill-in reducing column permutations for sparse matrix operations.
//!
//! `colamd` computes a permutation `Q` such that the Cholesky factorization of `(AQ)'(AQ)` has
//! less fill-in and requires fewer floating point operations than `A'A`. This also provides a good
//! ordering for sparse partial pivoting methods, `P(AQ) = LU`, where `Q` is computed prior to
//! numerical factorization, and `P` is computed during numerical factorization via conventional
//! partial pivoting with row interchanges.
//!
//! `symamd` computes a permutation `P` of a symmetric matrix `A` such that the Cholesky
//! factorization of `PAP'` has less fill-in and requires fewer floating point operations than `A`.
//! `symamd` constructs a matrix `M` such that `M'M` has the same nonzero pattern of `A`, and then
//! orders the columns of `M` using `colamd`. The column ordering of M is then returned as the row
//! and column ordering P of A.
//!
//! ## Features
//!
//! - `std` (enabled by default): Get floating point functions from the standard library
//!   (likely using your target's libc).
//! - `libm`: Use floating point implementations from [libm][].
//!
//! At least one of `std` and `libm` is required; `std` overrides `libm`.
//!
//! [COLAMD]: https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/9759b8c7538ecc92f9aa76b19fbf3f266432d113/COLAMD
#![cfg_attr(feature = "libm", doc = "[libm]: libm")]
#![cfg_attr(not(feature = "libm"), doc = "[libm]: https://crates.io/crates/libm")]
// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bot.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

// Keep clippy from complaining about unused libm in nostd test case.
#[cfg(feature = "libm")]
#[expect(unused, reason = "keep clippy happy")]
fn ensure_libm_dependency_used() -> f32 {
    libm::sqrtf(4_f32)
}

mod colamd;

pub use colamd::colamd_recommended;

/// Options controlling [`colamd`][colamd()] and `symamd` behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Rows with more than `max(16, dense_row_control * sqrt(n_col))` entries are removed prior to
    /// ordering.
    pub dense_row_control: f64,

    /// Columns with more than `max(16, dense_column_control * sqrt(min(n_row,n_col)))` entries are
    /// removed prior to ordering, and placed last in the output column ordering.
    pub dense_column_control: f64,

    /// Whether to do "aggressive absorption" during the elimination phase.
    ///
    /// For more information, see "Algorithm 836: COLAMD, a column approximate minimum degree
    /// ordering algorithm." (2004) by Timothy Davis et al.
    pub aggressive_row_absorption: bool,
}

impl Options {
    const DEFAULT: Self = Self {
        dense_row_control: 10.,
        dense_column_control: 10.,
        aggressive_row_absorption: true,
    };
}

impl Options {
    const fn as_knobs_array(self) -> [f64; 20] {
        let mut array = [0.; 20];
        array[0] = self.dense_row_control;
        array[1] = self.dense_column_control;
        array[2] = f64::from_bits(self.aggressive_row_absorption as u64);
        array
    }
}

impl core::default::Default for Options {
    fn default() -> Self {
        Options::DEFAULT
    }
}

/// Computes a column ordering for `A` such that the Cholesky decomposition of `A^T A` has less
/// fill-in.
///
/// This computes a column ordering (Q) of A such that P(AQ)=LU or (AQ)'AQ=LL' have less fill-in
/// and require fewer floating point operations than factorizing the unpermuted matrix A or A'A,
/// respectively.
///
/// # Example
///
/// We can calculate a column approximate minimum degree ordering for the following sparse matrix
///
/// ```text
///     1  2  3  4
/// 1 | x     x
/// 2 | x     x  x
/// 3 |    x  x
/// 4 |       x  x
/// 5 | x  x
/// ```
///
/// as follows.
///
/// ```rust
/// use colamd_rs::{colamd, colamd_recommended};
///
/// let num_nonzero = 11;
/// let nrows = 5;
/// let ncols = 4;
///
/// let a_len = colamd_recommended(num_nonzero, nrows, ncols).unwrap();
/// let mut row_indices = vec![0; a_len];
/// row_indices[..11].copy_from_slice(&[0, 1, 4, 2, 4, 0, 1, 2, 3, 1, 3]);
/// let column_pointers = &mut [0, 3, 5, 9, 11];
/// let stats = &mut [0; 20];
/// colamd(nrows, ncols, &mut row_indices, column_pointers, None, stats);
///
/// assert_eq!(column_pointers, &[1, 0, 2, 3, -1]);
/// ```
pub fn colamd(
    nrows: i32,
    ncols: i32,
    row_indices: &mut [i32],
    column_pointers: &mut [i32],
    options: Option<Options>,
    stats: &mut [i32; 20],
) -> i32 {
    assert_eq!(
        column_pointers.len(),
        ncols
            .checked_add(1)
            .and_then(|n| usize::try_from(n).ok())
            .expect("overflowed"),
        "`p` must be of length `n_col+1` (containing one column pointer to the start of each column, plus a pointer at the end",
    );

    let mut knobs = options.map(Options::as_knobs_array);
    let knobs = knobs
        .as_mut()
        .map(|k| k.as_mut_ptr())
        .unwrap_or(core::ptr::null_mut());

    unsafe { colamd::colamd(nrows, ncols, row_indices, column_pointers, knobs, stats) }
}

#[doc(hidden)]
#[expect(unused, reason = "check signatures anyway")]
pub fn symamd(
    n: i32,
    row_indices: &mut [i32],
    column_pointers: &mut [i32],
    permutation: &mut [i32],
    options: Option<Options>,
    stats: &mut [i32; 20],
) -> i32 {
    unimplemented!(
        "the implementation requires ANSI C `calloc` and `free`, rather than providing that, it should just be rewritten"
    );

    assert_eq!(
        column_pointers.len(),
        n.checked_add(1)
            .and_then(|n| usize::try_from(n).ok())
            .expect("overflowed"),
        "`p` must be of length `n_col+1` (containing one column pointer to the start of each column, plus a pointer at the end",
    );
    assert_eq!(
        column_pointers.len(),
        permutation.len(),
        "The column pointers and column permutation slice must have the same length"
    );

    let a_i = row_indices.as_mut_ptr();
    let p = column_pointers.as_mut_ptr();
    let permutation = permutation.as_mut_ptr();

    let mut knobs = options.map(Options::as_knobs_array);
    let knobs = knobs
        .as_mut()
        .map(|k| k.as_mut_ptr())
        .unwrap_or(core::ptr::null_mut());

    unsafe {
        colamd::symamd(
            n,
            a_i,
            p,
            permutation,
            knobs,
            stats,
            None, // wants `calloc`
            None, // wants `free`
        )
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{Options, colamd, colamd_recommended};

    #[test]
    fn colamd_known_value() {
        // Note: this example is from
        // https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/9759b8c7538ecc92f9aa76b19fbf3f266432d113/COLAMD/Source/colamd.c#L355-L377
        //
        // (though no expected result permutation is given).
        const A_LEN: usize = 100;
        let mut row_indices = vec![0; A_LEN];
        row_indices[..11].copy_from_slice(&[0, 1, 4, 2, 4, 0, 1, 2, 3, 1, 3]);
        let column_pointers = &mut [0, 3, 5, 9, 11];
        let stats = &mut [0; 20];
        colamd(5, 4, &mut row_indices, column_pointers, None, stats);

        // Running this through the original C version results in the permutation
        // `[1, 0, 2, 3, -1]`.
        assert_eq!(column_pointers, &[1, 0, 2, 3, -1]);
    }

    #[test]
    fn colamd_known_value_no_aggressive_absorption() {
        let a_len = colamd_recommended(7, 4, 3).unwrap();
        let column_pointers = &mut [0, 3, 4, 7];
        let mut row_indices = vec![0; a_len];
        row_indices[..7].copy_from_slice(&[0, 1, 2, 1, 0, 1, 3]);
        let mut knobs = Options::DEFAULT;
        knobs.aggressive_row_absorption = false;
        colamd(
            4,
            3,
            &mut row_indices,
            column_pointers,
            Some(knobs),
            &mut [0; 20],
        );

        // Running this through the original C version results in the permutation `[1, 2, 0, -1]`.
        assert_eq!(column_pointers, &[1, 2, 0, -1]);
    }
}
