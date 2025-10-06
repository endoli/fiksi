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
//! use fiksi::{System, constraints, elements};
//!
//! let mut gcs = fiksi::System::new();
//!
//! // Add three points, and constrain them into a triangle, such that
//! // - one corner has an angle of 10 degrees;
//! // - one corner has an angle of 60 degrees; and
//! // - the side between those corners is of length 5.
//! let p1 = elements::Point::create(&mut gcs, 1., 0.);
//! let p2 = elements::Point::create(&mut gcs, 0.8, 1.);
//! let p3 = elements::Point::create(&mut gcs, 1.1, 2.);
//!
//! constraints::PointPointDistance::create(&mut gcs, p2, p3, 5.);
//! constraints::PointPointPointAngle::create(&mut gcs, p1, p2, p3, 10f64.to_radians());
//! constraints::PointPointPointAngle::create(&mut gcs, p2, p3, p1, 60f64.to_radians());
//!
//! gcs.solve(fiksi::SolvingOptions::DEFAULT);
//! ```
//!
//! # Manual
//!
//! The [Fiksi manual](crate::manual) explains more about usage and the design.
//!

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
#![cfg_attr(docsrs, feature(doc_cfg))]
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
use alloc::{vec, vec::Vec};

pub use kurbo;

// Only enable the `manual` module when generating documentation or when testing. Though
// `cargo test` also enables `doc`, a development environment probably only enables `test`.
#[cfg(any(test, doc))]
pub mod manual;

mod analyze;
mod assemble;
pub(crate) mod collections;
pub mod constraints;
pub mod elements;
pub(crate) mod graph;
mod rand;
pub mod solve;
mod subsystem;
pub(crate) mod utils;
mod variable_map;

#[cfg(test)]
mod tests;

pub(crate) use constraints::constraint::ConstraintId;
pub(crate) use constraints::expressions::Expression;
pub use constraints::{
    Constraint,
    constraint::{AnyConstraintHandle, ConstraintHandle, TaggedConstraintHandle},
};
use elements::element::ElementId;
pub use elements::{
    Element,
    element::{AnyElementHandle, ElementHandle, TaggedElementHandle},
};
pub(crate) use rand::Rng;
pub(crate) use subsystem::Subsystem;
pub(crate) use variable_map::{Variable, VariableMap};

use crate::{
    analyze::graph::{
        equations::ExpressionGraph,
        recursive_assembly::{ClusterKey, RecombinationStep},
    },
    constraints::{ConstraintTag, expressions::Pose2D},
    graph::Graph,
};

/// These are the geometric elements of the constraint system.
///
/// These elements have been flattened, in that elements referencing other elements (like a `Line`
/// referencing two `Point`s) now just point directly to the start of the underlying variables in
/// [`System::variables`].
pub(crate) enum EncodedElement {
    Length { idx: u32 },
    Point { idx: u32 },
    Line { point1_idx: u32, point2_idx: u32 },
    Circle { center_idx: u32, radius_idx: u32 },
}

/// These are the constraints between geometric elements.
///
/// These constraints have been flattened, in that they directly point to the variables in
/// [`System::variables`] referenced by their original element arguments.
pub(crate) struct EncodedConstraint {
    tag: ConstraintTag,
    expressions_idx: u32,
}

/// An element value.
pub enum ElementValue {
    /// An [`elements::Length`] value.
    Length(f64),
    /// An [`elements::Point`] value.
    Point(kurbo::Point),
    /// An [`elements::Line`] value.
    Line(kurbo::Line),
    /// An [`elements::Circle`] value.
    Circle(kurbo::Circle),
}

/// The decomposer used for splitting the full geometric constraint system into smaller, solvable
/// subsystems.
#[derive(Clone, Debug, PartialEq)]
pub enum Decomposer {
    /// No decomposer. Solve the entire system at once using a numeric optimizer.
    ///
    /// This is generally stable for small systems, and is the fastest method for very small
    /// systems on the order of tens of elements. Solving without decomposition becomes intractable
    /// as systems grow beyond hundreds of elements.
    None,

    /// Decompose the system of non-linear equations by matching constraints' expressions to
    /// elements' variables.
    ///
    /// This decomposition is relatively cheap to compute, and results in an ordering of groups of
    /// constraints' residual expressions and the elements' variables they calculate. Each
    /// expression calculates one variable. A single pass through these partially ordered groups,
    /// solving for all constraints within each group together, solves the system.
    ///
    /// This works well for fully-determined systems, especially those fully anchored to the global
    /// coordinate system; this works less well for systems with under-constrained parts. Severely
    /// under-constrained systems tend to generate big expression groups, which requires many
    /// expressions to be solved for at once.
    SinglePass,

    /// Warning: experimental. Perform a geometric recursive assembly of the system.
    ///
    /// This currently does not correctly handle constraints relative to the global coordinate
    /// system, nor fixing elements to the global coordinate system.
    ///
    /// This searches for minimal, rigid clusters of elements and constraints within the geometric
    /// system that can be solved separately. These rigid clusters can be transformed through
    /// displacement relative to each other, but internal geometry remains unchanged. A partial
    /// order for solving is induced by the hierarchy of such clusters.
    ///
    /// This works well for rigidly well-constrained systems, and is relatively robust against
    /// under-constrained geometry. All rigidly well-constrained geometry will be found and solved
    /// individually. Any left-over under-constrained part still requires solving for all its
    /// expressions at once.
    RecursiveAssembly,
}

/// Options used by [`System::solve`].
#[derive(PartialEq, Debug)]
pub struct SolvingOptions {
    /// The numerical optimization algorithm to use for solving constraint systems.
    pub optimizer: solve::Optimizer,

    /// Which decomposer to use, if any.
    pub decomposer: Decomposer,

    /// Whether to slightly perturb the values of elements before solving.
    ///
    /// This helps finding solutions when the initial state of elements is likely to be singular.
    /// For example, three collinear points with pairwise distance constraints is a singular
    /// system, and numerical methods may not find a solution.
    pub perturb: bool,
}

impl SolvingOptions {
    /// Construct the default [`SolvingOptions`].
    ///
    /// The defaults are as follows.
    ///
    /// ```rust
    /// assert_eq!(fiksi::SolvingOptions::DEFAULT, fiksi::SolvingOptions {
    ///     optimizer: fiksi::solve::Optimizer::LevenbergMarquardt,
    ///     decomposer: fiksi::Decomposer::None,
    ///     perturb: true,
    /// });
    /// ```
    pub const DEFAULT: Self = Self {
        optimizer: solve::Optimizer::LevenbergMarquardt,
        decomposer: Decomposer::None,
        perturb: true,
    };
}

impl Default for SolvingOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Returned by [`System::analyze`].
#[derive(Debug)]
pub struct Analysis {
    /// Constraints causing parts of the system to be overconstrained.
    pub overconstrained: Vec<AnyConstraintHandle>,
}

/// A geometric constraint system.
///
/// Build the system by [adding elements](Element) and [constraints](Constraint). Then solve
/// (sub)systems using [`System::solve`].
pub struct System {
    id: u32,

    /// Geometric element graph.
    graph: Graph,
    /// Bipartite equation graph.
    equation_graph: ExpressionGraph,

    /// Geometric elements.
    elements: Vec<EncodedElement>,
    /// The variables of the geometric elements, such as point coordinates.
    ///
    /// Note that elements have one or more variables.
    variables: Vec<f64>,
    /// Mapping from variables back to the primitive geometric element that it belongs to.
    variable_to_primitive: Vec<ElementId>,

    /// Constraints between geometric elements.
    constraints: Vec<EncodedConstraint>,
    /// The expressions of the constraints, such as point-point distance.
    ///
    /// Note that each constraint can encode one or more expressions (see also
    /// [`Constraint::VALENCY`]).
    expressions: Vec<Expression>,
    /// Mapping from expressions back to the constraint it belongs to.
    expression_to_constraint: Vec<ConstraintId>,
}

impl System {
    /// Construct an empty geometric constraint system.
    pub fn new() -> Self {
        static COUNTER: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
        let id = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        Self {
            id,
            graph: Graph::new(),
            equation_graph: ExpressionGraph::new(),
            elements: vec![],
            variables: vec![],
            variable_to_primitive: vec![],
            constraints: vec![],
            expressions: vec![],
            expression_to_constraint: vec![],
        }
    }

    /// Iterate over the handles of all elements in the system.
    ///
    /// You can use [`AnyElementHandle::get_value`] to get an element-tagged value or
    /// [`AnyElementHandle::as_tagged_element`] to get a typed handle.
    pub fn get_element_handles(&self) -> impl Iterator<Item = AnyElementHandle> {
        self.elements
            .iter()
            .enumerate()
            .map(|(id, encoded_element)| {
                AnyElementHandle::from_ids_and_tag(
                    self.id,
                    id.try_into().expect("less than 2^32 elements"),
                    encoded_element.into(),
                )
            })
    }

    /// Iterate over the handles of all constraints in the system.
    ///
    /// You can use [`AnyConstraintHandle::as_tagged_constraint`] to get a typed handle.
    pub fn get_constraint_handles(&self) -> impl Iterator<Item = AnyConstraintHandle> {
        self.constraints
            .iter()
            .enumerate()
            .map(|(id, encoded_constraint)| {
                AnyConstraintHandle::from_ids_and_tag(
                    self.id,
                    id.try_into().expect("less than 2^32 constraints"),
                    encoded_constraint.tag,
                )
            })
    }

    /// Add an element.
    ///
    /// Give the element sets the element belongs to in `sets`.
    pub(crate) fn add_element<T: Element, const N: usize>(
        &mut self,
        variables: [f64; N],
        encoded_element: impl FnOnce(u32) -> EncodedElement,
    ) -> ElementHandle<T> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the const panic ensures this never truncates"
        )]
        let dof = const {
            // the `16` here is arbitrary, but practically should be more than large enough
            if N > 16 {
                panic!("Element adds too many variables.")
            }
            N as i16
        };

        let element_handle = {
            let id = self
                .elements
                .len()
                .try_into()
                .expect("less than 2^32 elements");
            ElementHandle::from_ids(self.id, id)
        };
        let variables_idx = self
            .variables
            .len()
            .try_into()
            .expect("less than 2^32 variables");
        if N > 0 {
            self.variables.extend_from_slice(&variables);
            self.equation_graph.insert_variables::<N>();
            self.variable_to_primitive
                .extend((0..N).map(|_| element_handle.drop_system_id()));
        }

        // Currently 0 degree-of-freedom elements are also added to the graph (even though they
        // don't represent a primitive), as the `ElementId`s are shared between the system and the
        // graph.
        self.graph.add_element(dof);

        self.elements.push(encoded_element(variables_idx));
        element_handle
    }

    /// Add a constraint.
    ///
    /// Give the constraint sets the constraint belongs to in `sets`.
    pub(crate) fn add_constraint<T: Constraint>(
        &mut self,
        tag: ConstraintTag,
        expressions: impl IntoIterator<Item = Expression>,
    ) -> ConstraintHandle<T> {
        let id = self
            .constraints
            .len()
            .try_into()
            .expect("less than 2^32 constraints");
        let expressions_idx = self
            .expressions
            .len()
            .try_into()
            .expect("less than 2^32 expressions");
        self.constraints.push(EncodedConstraint {
            tag,
            expressions_idx,
        });

        let mut variable_indices = [0; 8];
        for expression in expressions.into_iter() {
            self.equation_graph.insert_expression(
                expression
                    .variable_indices(&mut variable_indices)
                    .iter()
                    .copied(),
            );
            self.expressions.push(expression);
            self.expression_to_constraint.push(ConstraintId { id });
        }

        ConstraintHandle::from_ids(self.id, id)
    }

    /// Analyze the system, without performing a full solve.
    ///
    /// This may change elements' positions in order to satisfy numeric requirements.
    pub fn analyze(&mut self) -> Analysis {
        let overconstrained = analyze::numerical::find_overconstraints(self);

        Analysis { overconstrained }
    }

    /// Solve the system.
    ///
    /// All constraints are solved for, and will be satisfied if the solving is successful.
    /// Currently all elements are considered free.
    pub fn solve(&mut self, opts: SolvingOptions) {
        assemble::solve(self, opts);
    }
}

impl Default for System {
    fn default() -> Self {
        Self::new()
    }
}
