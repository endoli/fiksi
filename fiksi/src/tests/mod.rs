// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod basic;
mod fixed;
mod magnitude;
mod singular;
mod triangles;

/// A "good enough" residual value that is considered to have solved the constraint.
///
/// This would normally be a relative value, especially for things like distance constraints.
const RESIDUAL_THRESHOLD: f64 = 1e-4;
