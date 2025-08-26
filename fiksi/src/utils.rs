// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use core::borrow::Borrow;

#[inline]
pub(crate) fn sum_squares(values: impl IntoIterator<Item = impl Borrow<f64>>) -> f64 {
    values
        .into_iter()
        .map(|v| {
            let v = *v.borrow();
            v * v
        })
        .sum()
}
