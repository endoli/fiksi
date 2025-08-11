// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This is the Fiksi manual.
//!
//! It explains usage, the current design, and some possible future extensions. The design may
//! change in future revisions of Fiksi.
//!
//! # Geometric constraint solving
//!
//! A geometric constraint system is a specification of geometric elements, such as points and
//! circles; and constraints between those elements, such as a distance between a pair of points or
//! a point being incident on a line. The goal of a geometric constraint solver is to find an
//! arrangement of the system's elements that satisfies the constraints.
//!
//! # Geometry in Fiksi
//!
//! Fiksi includes 2D geometric elements and constraints between them.
//!
//! It has a notion of [rigidity][rigid-transformation], and specifically rigid motions. This
//! notion is useful for solving, as it allows a system to be decomposed into simpler parts that
//! can be solved separately. Determining whether a geometric constraint system is rigid is not in
//! general tractable, but it can be approximated well enough to be useful through analysis of
//! elements' and constraints' degrees of freedom as well as through numeric analysis.
//!
//! [rigid-transformation]: <https://en.wikipedia.org/w/index.php?title=Rigid_transformation&oldid=1291700504>
//!
//! ## Geometric elements
//!
//! The following elements are currently implemented.
//!
// TODO: should we mention internal representation?
//! | Element                          | Degrees of freedom |
//! | -- | -- |
//! | [Point][elements::Point]         | 2 |
//! | [Infinite line][elements::Line]  | 2 |
//! | [Circle][elements::Circle]       | 3 |
//!
//! The following is a non-exhaustive list of future elements considered for implementation.
//!
//! | Element            | Degrees of freedom |
//! | -- | -- |
//! | Line segment       | 4 |
//! | Circular arc       | 5 |
//! | Cubic Bezier curve | 8 |
//!
//! ## Geometric constraints
//!
//! Geometric constraints are represented as a system of equations providing residuals. The
//! equations are non-linear in general. A solution would be an arrangement such that all are 0. If
//! a solution does not exist, one may want to find an arrangement that minimizes an error metric.
//! The error metric used in Fiksi is the sum of squared residuals.
//!
//! The following constraints between geometric elements are currently implemented. The valency of
//! a constraint is the number of degrees of freedom taken away by a constraint from its arguments.
//!
//! | Constraint                       | Arity and arguments     | Parameters | Valency |
//! | -- | -- | -- | -- |
//! | [Point-point distance][ppd]      | 2 (point, point)        | Distance   | 1 |
//! | [Point-point-point angle][pppa]  | 3 (point, point, point) | Angle      | 1 |
//! | [Point-line incidence][pli]      | 2 (point, line)         |            | 1 |
//! | [Line-line angle][lla]           | 2 (line, line)          | Angle      | 1 |
//! | [Line-line parallelism][llp]     | 2 (line, line)          |            | 1 |
//! | [Line-circle tangency][lct]      | 2 (line, circle)        |            | 1 |
//!
//! [ppd]: `constraints::PointPointDistance`
//! [pppa]: `constraints::PointPointPointAngle`
//! [pli]: `constraints::PointLineIncidence`
//! [lla]: `constraints::LineLineAngle`
//! [llp]: `constraints::LineLineParallelism`
//! [lct]: `constraints::LineCircleTangency`
//!
//! The following is a non-exhaustive list of future constraints considered for implementation.
//!
//! | Constraint | Arity and arguments | Parameters | Valency |
//! | -- | -- | -- | -- |
//! | Point-point coincidence | 2 (point, point) | | 2 |
//! | Point-line distance | 2 (point, line) | Distance | 1 |
//! | Point-circle centrality | 3 (point, circle) | | 2 |
//!
//! ## Three dimensions
//!
//! It is possible to extend Fiksi to 3D, as the theory is mostly analogous. This would require
//! some notion of hierarchies.
//!
//! ## Hierarchies
//!
//! ...

#![expect(unused_imports, reason = "Importing items to link to them in docs")]

use crate::{constraints, elements};
