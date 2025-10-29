// Copyright 2025 the Solvi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// A three-tuple representing a value at a specified matrix coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct Triplet<T> {
    /// The matrix row this value is in.
    pub row: usize,

    /// The matrix column this value is in.
    pub col: usize,

    /// The value.
    pub value: T,
}
