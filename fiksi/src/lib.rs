// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fiksi is a geometric and parametric constraint solver.
//!
//! ## Features
//!
//! - `std` (enabled by default): Get floating point functions from the standard library
//!   (likely using your target's libc).
//! - `libm`: Use floating point implementations from [libm][].
//!
//! At least one of `std` and `libm` is required; `std` overrides `libm`.
//!
//! # Example
//!
//! ```rust
//! let mut gcs = fiksi::System::new();
//!
//! // Add three points, and constrain them into a triangle, such that
//! // - one corner has an angle of 10 degrees;
//! // - one corner has an angle of 60 degrees; and
//! // - the side between those corners is of length 5.
//! let p1 = gcs.add_element(fiksi::elements::Point::new(1., 0.));
//! let p2 = gcs.add_element(fiksi::elements::Point::new(0.8, 1.));
//! let p3 = gcs.add_element(fiksi::elements::Point::new(1.1, 2.));
//!
//! gcs.add_constraint(fiksi::constraints::PointPointDistance::new(p2, p3, 5.));
//! gcs.add_constraint(fiksi::constraints::PointPointPointAngle::new(
//!     p1,
//!     p2,
//!     p3,
//!     10f64.to_radians(),
//! ));
//! gcs.add_constraint(fiksi::constraints::PointPointPointAngle::new(
//!     p2,
//!     p3,
//!     p1,
//!     60f64.to_radians(),
//! ));
//! gcs.solve(None, fiksi::SolvingOptions::DEFAULT);
//! ```
#![cfg_attr(feature = "libm", doc = "[libm]: libm")]
#![cfg_attr(not(feature = "libm"), doc = "[libm]: https://crates.io/crates/libm")]
// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![no_std]
// TODO: remove these two
#![expect(dead_code, reason = "clean up later")]
#![expect(missing_debug_implementations, reason = "clean up later")]

#[cfg(all(not(feature = "std"), not(test)))]
mod floatfuncs;

// Keep clippy from complaining about unused libm in nostd test case.
#[cfg(feature = "libm")]
#[expect(unused, reason = "keep clippy happy")]
fn ensure_libm_dependency_used() -> f32 {
    libm::sqrtf(4_f32)
}

extern crate alloc;
use alloc::{collections::btree_set::BTreeSet, vec, vec::Vec};

pub use kurbo;

pub mod constraints;
pub mod elements;
pub mod solve;
pub(crate) mod utils;

#[cfg(test)]
mod tests;

pub(crate) use constraints::constraint::ConstraintId;
pub use constraints::{Constraint, constraint::ConstraintHandle};
use elements::element::ElementId;
pub use elements::{Element, element::ElementHandle};

use crate::constraints::{
    LineCircleTangency_, LineLineAngle_, PointLineIncidence_, PointPointDistance_,
    PointPointPointAngle_,
};

/// Vertices are the geometric elements of the constraint system.
///
/// The indices point into the start of the vertex's variables in [`System::variables`].
pub(crate) enum Vertex {
    Point { idx: u32 },
    Line { point1_idx: u32, point2_idx: u32 },
    Circle { center_idx: u32, radius_idx: u32 },
}

/// Edges are the constraints between geometric elements (i.e., edges between the vertices).
pub(crate) enum Edge {
    PointPointDistance(PointPointDistance_),
    PointPointPointAngle(PointPointPointAngle_),
    PointLineIncidence(PointLineIncidence_),
    LineLineAngle(LineLineAngle_),
    LineCircleTangency(LineCircleTangency_),
}

/// A handle to a set of constraints to solve for and variables that are considered free.
///
/// See [`System::create_solve_set`].
pub struct SolveSetHandle {
    /// The ID of the system the element set belongs to.
    system_id: u32,
    /// The ID of the element set within the system.
    id: u32,
}

/// Options used by [`System::solve`].
#[derive(PartialEq, Debug)]
pub struct SolvingOptions {
    /// The numerical optimization algorithm to use for solving constraint systems.
    pub optimizer: solve::Optimizer,
}

impl SolvingOptions {
    /// Construct the default [`SolvingOptions`].
    ///
    /// The defaults are as follows.
    ///
    /// ```rust
    /// assert_eq!(fiksi::SolvingOptions::DEFAULT, fiksi::SolvingOptions {
    ///     optimizer: fiksi::solve::Optimizer::LevenbergMarquardt,
    /// });
    /// ```
    pub const DEFAULT: Self = Self {
        optimizer: solve::Optimizer::LevenbergMarquardt,
    };
}

impl Default for SolvingOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Contains constraints that should be solved and elements whose variables are free.
pub(crate) struct SolveSet {
    pub(crate) elements: BTreeSet<ElementId>,
    pub(crate) constraints: BTreeSet<ConstraintId>,
}

impl SolveSet {
    pub(crate) fn new() -> Self {
        Self {
            elements: BTreeSet::new(),
            constraints: BTreeSet::new(),
        }
    }
}

/// A geometric constraint system.
///
/// Build the system by [adding elements](System::add_element) and
/// [constraints](System::add_constraint). Then solve (sub)systems using [`System::solve`].
pub struct System {
    id: u32,
    /// Geometric elements.
    element_vertices: Vec<Vertex>,
    /// Constraints between geometric elements.
    constraint_edges: Vec<Edge>,
    /// The variables of the geometric elements, such as point coordinates.
    variables: Vec<f64>,
    /// The sets of elements and constraints that can be solved for.
    solve_sets: Vec<SolveSet>,
}

impl System {
    /// Construct an empty geometric constraint system.
    pub fn new() -> Self {
        static COUNTER: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
        let id = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        Self {
            id,
            variables: vec![],
            element_vertices: vec![],
            constraint_edges: vec![],
            solve_sets: vec![],
        }
    }

    /// Add a solve set to the geometric constraint system.
    ///
    /// Solve sets can be used to create separate sets of constraints that should be solved, and
    /// elements whose variables are free and can be used for solving.
    ///
    /// This can be used to isolate distinct parts of constraint systems that should be solved
    /// separately, or to choose which variables are free and which are kept fixed.
    ///
    /// You always solve for at most one solve set at once. If you don't use a solve set, all
    /// constraints are solved for, and all variables are considered free.
    pub fn create_solve_set(&mut self) -> SolveSetHandle {
        let id = self.solve_sets.len();
        self.solve_sets.push(SolveSet::new());

        SolveSetHandle {
            system_id: self.id,
            id: id.try_into().expect("less than 2^32 solve sets"),
        }
    }

    /// Add an element.
    ///
    /// Give the element sets the element belongs to in `sets`.
    pub fn add_element<T: Element>(&mut self, element: T) -> ElementHandle<T> {
        let id = self
            .element_vertices
            .len()
            .try_into()
            .expect("less than 2^32 elements");

        element.add_into(&mut self.element_vertices, &mut self.variables);

        ElementHandle::from_ids(self.id, id)
    }

    /// Get the value of an element.
    pub fn get_element<T: Element>(&self, element: ElementHandle<T>) -> <T as Element>::Output {
        // TODO: return `Result` instead of panicking?
        assert_eq!(
            self.id, element.system_id,
            "Tried to get an element that is not part of this `System`"
        );

        T::from_vertex(
            &self.element_vertices[element.drop_system_id().id as usize],
            &self.variables,
        )
        .into()
    }

    /// Calculate the residual of a constraint.
    pub fn calculate_constraint_residual<T>(&self, constraint: ConstraintHandle<T>) -> f64 {
        let edge = &self.constraint_edges[constraint.drop_system_id().id as usize];
        let residual = &mut [0.];
        utils::calculate_residuals_and_jacobian(
            &[edge],
            &alloc::collections::BTreeMap::new(),
            &self.variables,
            residual,
            &mut [],
        );
        residual[0]
    }

    /// Add a constraint.
    ///
    /// Give the constraint sets the constraint belongs to in `sets`.
    pub fn add_constraint<T: Constraint>(&mut self, constraint: T) -> ConstraintHandle<T> {
        let id = self
            .constraint_edges
            .len()
            .try_into()
            .expect("less than 2^32 constraints");
        self.constraint_edges
            .push(constraint.as_edge(&self.element_vertices));

        ConstraintHandle::from_ids(self.id, id)
    }

    /// Add an element to the solve set.
    ///
    /// See [`System::create_solve_set`].
    pub fn add_element_to_solve_set<T>(
        &mut self,
        solve_set: &SolveSetHandle,
        element: &ElementHandle<T>,
    ) {
        // TODO: return `Result` instead of panicking?
        assert_eq!(
            self.id, solve_set.system_id,
            "Tried to use a solve set that is not part of this `System`"
        );
        assert_eq!(
            self.id, element.system_id,
            "Tried to use an element that is not part of this `System`"
        );
        let solve_set = &mut self.solve_sets[solve_set.id as usize];
        solve_set.elements.insert(element.drop_system_id());
    }

    /// Add a constraint to the solve set.
    ///
    /// See [`System::create_solve_set`].
    pub fn add_constraint_to_solve_set<T>(
        &mut self,
        solve_set: &SolveSetHandle,
        constraint: &ConstraintHandle<T>,
    ) {
        // TODO: return `Result` instead of panicking?
        assert_eq!(
            self.id, solve_set.system_id,
            "Tried to use a solve set that is not part of this `System`"
        );
        assert_eq!(
            self.id, constraint.system_id,
            "Tried to use an element that is not part of this `System`"
        );
        let solve_set = &mut self.solve_sets[solve_set.id as usize];
        solve_set.constraints.insert(constraint.drop_system_id());
    }

    /// Solve the system.
    ///
    /// The system is solved for constraints and free variables in `solve_set`. Variables not in
    /// `solve_set` are taken as fixed parameters. If no `solve_set` is given, all constraints are
    /// solved for, and all variables are considered free.
    ///
    /// Note handle values (such as the center point of a circle) are free if and only if the
    /// element the handle corresponds to is considered free, regardless of whether the circle
    /// itself is in `solve_set`.
    pub fn solve(&mut self, solve_set: Option<&SolveSetHandle>, opts: SolvingOptions) {
        match opts.optimizer {
            solve::Optimizer::LevenbergMarquardt => crate::solve::levenberg_marquardt(
                &mut self.variables,
                solve_set.map(|solve_set| &self.solve_sets[solve_set.id as usize]),
                &self.element_vertices,
                &self.constraint_edges,
            ),
            solve::Optimizer::LBfgs => crate::solve::lbfgs(
                &mut self.variables,
                solve_set.map(|solve_set| &self.solve_sets[solve_set.id as usize]),
                &self.element_vertices,
                &self.constraint_edges,
            ),
        }
    }
}

impl Default for System {
    fn default() -> Self {
        Self::new()
    }
}
