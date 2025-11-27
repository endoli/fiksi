// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use core::borrow::Borrow;

#[cfg(not(feature = "std"))]
use crate::floatfuncs::FloatFuncs;

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

#[inline]
pub(crate) fn mean_squares(values: impl IntoIterator<Item = impl Borrow<f64>>) -> f64 {
    let mut n: usize = 0;
    sum_squares(values.into_iter().inspect(|_| {
        n += 1;
    })) / n as f64
}

#[inline]
pub(crate) fn root_mean_squares(values: impl IntoIterator<Item = impl Borrow<f64>>) -> f64 {
    mean_squares(values).sqrt()
}
