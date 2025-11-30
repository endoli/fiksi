// COLAMD, Copyright (c) 1998-2024, Timothy A. Davis and Stefan Larimore,
// All Rights Reserved.
// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: BSD-3-Clause

/// Options controlling [`colamd`][crate::colamd()] and `symamd` behavior.
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
    pub const DEFAULT: Self = Self {
        dense_row_control: 10.,
        dense_column_control: 10.,
        aggressive_row_absorption: true,
    };
}

impl core::default::Default for Options {
    fn default() -> Self {
        Options::DEFAULT
    }
}
