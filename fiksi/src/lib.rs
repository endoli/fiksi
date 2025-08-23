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
//! gcs.solve(None, fiksi::SolvingOptions::DEFAULT);
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

// Only enable the `manual` module when generating documentation or when testing. Though
// `cargo test` also enables `doc`, a development environment probably only enables `test`.
#[cfg(any(test, doc))]
pub mod manual;

mod analyze;
pub(crate) mod collections;
pub mod constraints;
pub mod elements;
pub(crate) mod graph;
mod rand;
pub mod solve;
mod subsystem;
pub(crate) mod utils;

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

use crate::{analyze::graph::RecombinationPlan, constraints::ConstraintTag, graph::Graph};

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

/// A handle to a set of constraints to solve for and variables that are considered free.
///
/// See [`System::create_solve_set`].
pub struct SolveSetHandle {
    /// The ID of the system the element set belongs to.
    system_id: u32,
    /// The ID of the element set within the system.
    id: u32,
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

/// Options used by [`System::solve`].
#[derive(PartialEq, Debug)]
pub struct SolvingOptions {
    /// The numerical optimization algorithm to use for solving constraint systems.
    pub optimizer: solve::Optimizer,

    /// Whether to perform decomposition.
    pub decompose: bool,

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
    ///     decompose: false,
    ///     perturb: true,
    /// });
    /// ```
    pub const DEFAULT: Self = Self {
        optimizer: solve::Optimizer::LevenbergMarquardt,
        decompose: false,
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
/// Build the system by [adding elements](Element) and [constraints](Constraint). Then solve
/// (sub)systems using [`System::solve`].
pub struct System {
    id: u32,
    graph: Graph,
    /// Geometric elements.
    elements: Vec<EncodedElement>,
    /// Constraints between geometric elements.
    constraints: Vec<EncodedConstraint>,
    expressions: Vec<Expression>,
    expression_to_constraint: Vec<ConstraintId>,
    /// The variables of the geometric elements, such as point coordinates.
    variables: Vec<f64>,
    variable_to_primitive: Vec<ElementId>,
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
            graph: Graph::new(),
            variables: vec![],
            variable_to_primitive: vec![],
            elements: vec![],
            constraints: vec![],
            expressions: vec![],
            expression_to_constraint: vec![],
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
        self.expressions.extend(expressions);
        for _ in 0..self.expressions.len() - expressions_idx as usize {
            self.expression_to_constraint.push(ConstraintId { id });
        }

        ConstraintHandle::from_ids(self.id, id)
    }

    /// Add an element to the solve set.
    ///
    /// See [`System::create_solve_set`].
    pub fn add_element_to_solve_set<T: Element>(
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
    pub fn add_constraint_to_solve_set<T: Constraint>(
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

    /// Analyze the system, without performing a full solve.
    ///
    /// This may change elements' positions in order to satisfy numeric requirements.
    pub fn analyze(&mut self, solve_set: Option<&SolveSetHandle>) -> Analysis {
        let (elements, constraints) = if let Some(solve_set) = solve_set {
            let solve_set = &self.solve_sets[solve_set.id as usize];
            let elements = solve_set.elements.clone();
            let constraints = solve_set.constraints.clone();
            (elements, constraints)
        } else {
            (
                (0..self.elements.len().try_into().unwrap())
                    .map(|id| ElementId { id })
                    .collect(),
                (0..self.constraints.len().try_into().unwrap())
                    .map(|id| ConstraintId { id })
                    .collect(),
            )
        };

        let mut free_variables: Vec<u32> = vec![];
        for element_id in &elements {
            let element = &self.elements[element_id.id as usize];
            match element {
                EncodedElement::Point { idx } => {
                    free_variables.extend(&[*idx, *idx + 1]);
                }
                EncodedElement::Length { idx } => {
                    free_variables.extend(&[*idx]);
                }
                // In the current setup, not all vertices in the set contribute free variables.
                // E.g. `EncodedElement::Line` only refers to existing points, meaning it does not
                // contribute its own free variables. `EncodedElement::Circle` refers to a point,
                // but contributes its radius as free variable.
                EncodedElement::Circle { radius_idx, .. } => {
                    free_variables.extend(&[*radius_idx]);
                }
                _ => {}
            }
        }

        let subsystem = Subsystem::new(
            &self.expressions,
            free_variables,
            constraints
                .iter()
                .flat_map(|c| {
                    let constraint = &self.constraints[c.id as usize];
                    (0..constraint.tag.valency()).map(|offset| {
                        self.constraints[c.id as usize].expressions_idx + u32::from(offset)
                    })
                })
                .collect(),
        );

        let overconstrained = analyze::numerical::find_overconstraints(self, &subsystem);

        Analysis { overconstrained }
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
        let mut rng = Rng::from_seed(42);

        for connected_component in self.graph.connected_components() {
            let (elements, constraints) = if let Some(solve_set) = solve_set {
                let solve_set = &self.solve_sets[solve_set.id as usize];
                let elements = connected_component
                    .elements
                    .intersection(&solve_set.elements)
                    .copied()
                    .collect();
                let constraints = connected_component
                    .constraints
                    .intersection(&solve_set.constraints)
                    .copied()
                    .collect();
                (elements, constraints)
            } else {
                (
                    connected_component.elements.clone(),
                    connected_component.constraints.clone(),
                )
            };

            let recombination_plan = if opts.decompose {
                analyze::graph::decompose::<3>(
                    self.graph.clone(),
                    elements.iter().copied(),
                    constraints.iter().copied(),
                )
            } else {
                RecombinationPlan::single(elements, constraints)
            };

            for step in recombination_plan.steps() {
                let mut free_variables: Vec<u32> = vec![];
                for element_id in step.fixes_elements() {
                    let element = &self.elements[element_id.id as usize];
                    match element {
                        EncodedElement::Point { idx } => {
                            free_variables.extend(&[*idx, *idx + 1]);
                        }
                        EncodedElement::Length { idx } => {
                            free_variables.extend(&[*idx]);
                        }
                        // In the current setup, not all vertices in the set contribute free
                        // variables. E.g. `EncodedElement::Line` only refers to existing points,
                        // meaning it does not contribute its own free variables.
                        // `EncodedElement::Circle` refers to a point, but contributes its radius
                        // as free variable.
                        EncodedElement::Circle { radius_idx, .. } => {
                            free_variables.extend(&[*radius_idx]);
                        }
                        _ => {}
                    }
                }

                if opts.perturb {
                    for free_variable in free_variables.iter().copied() {
                        let variable = &mut self.variables[free_variable as usize];
                        // TODO: the scale-independent perturbation here should be revisited. See
                        // also https://github.com/endoli/fiksi/pull/41#discussion_r2234008761.
                        *variable += *variable * (1. / 8196.) * rng.next_f64()
                            + (1. / 65568.) * rng.next_f64();
                    }
                }

                let subsystem = Subsystem::new(
                    &self.expressions,
                    free_variables,
                    step.constraints()
                        .iter()
                        .flat_map(|c| {
                            let constraint = &self.constraints[c.id as usize];
                            (0..constraint.tag.valency()).map(|offset| {
                                self.constraints[c.id as usize].expressions_idx + u32::from(offset)
                            })
                        })
                        .collect(),
                );

                match opts.optimizer {
                    solve::Optimizer::LevenbergMarquardt => {
                        crate::solve::levenberg_marquardt(&mut self.variables, &subsystem);
                    }
                    solve::Optimizer::LBfgs => {
                        crate::solve::lbfgs(&mut self.variables, &subsystem);
                    }
                }
            }
        }
    }
}

impl Default for System {
    fn default() -> Self {
        Self::new()
    }
}
