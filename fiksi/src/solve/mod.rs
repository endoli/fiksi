// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Optimization and solving routines.

mod lbfgs;
mod lm;

pub(crate) use lbfgs::*;
pub(crate) use lm::*;

/// Numerical optimization algorithms.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum Optimizer {
    /// The [Levenberg-Marquardt][lm] algorithm.
    ///
    /// [lm]: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    LevenbergMarquardt,

    /// The [limited-memory BFGS][wikipedia] algorithm.
    ///
    /// [wikipedia]: https://en.wikipedia.org/wiki/Limited-memory_BFGS
    LBfgs,
}
