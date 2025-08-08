// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod basic;
mod magnitude;
mod sets;
mod singular;
mod triangles;

/// A "good enough" sum of squared residuals that is considered to have solved the system.
///
/// This would normally depend on the domain, especially for things like distance constraints.
const RESIDUAL_THRESHOLD: f64 = 1e-5;
