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
//! | Element                           | Degrees of freedom |
//! | -- | -- |
//! | [Point][elements::Point]          | 2 |
//! | [Boundless line][elements::Line]  | 2 |
//! | [Circle][elements::Circle]        | 3 |
//!
//! The internal representation of geometric elements and their degrees of freedom may not directly
//! correspond. For example, a boundless line may be represented by two points incident on the line
//! (for a total of four degrees of freedom), but a boundless line intrinsically only has two
//! degrees of freedom (an angle and a displacement from the origin).
//!
//! The following is a non-exhaustive list of future elements considered for implementation.
//!
//! | Element            | Degrees of freedom |
//! | -- | -- |
//! | Line segment       | 4 |
//! | Circular arc       | 5 |
//! | Ellipse            | 5 |
//! | Elliptic arc       | 7 |
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
//! | [Point-point coincidence][ppc]   | 2 (point, point)        |            | 2 |
//! | [Point-point distance][ppd]      | 2 (point, point)        | Distance   | 1 |
//! | [Point-point-point angle][pppa]  | 3 (point, point, point) | Angle      | 1 |
//! | [Point-line incidence][pli]      | 2 (point, line)         |            | 1 |
//! | [Point-line distance][pld]       | 2 (point, line)         | Distance   | 1 |
//! | [Point-circle incidence][pci]    | 2 (point, circle)       |            | 1 |
//! | [Line-line angle][lla]           | 2 (line, line)          | Angle      | 1 |
//! | [Line-line parallelism][llp]     | 2 (line, line)          |            | 1 |
//! | [Line-circle tangency][lct]      | 2 (line, circle)        |            | 1 |
//!
//! [ppc]: `constraints::PointPointCoincidence`
//! [ppd]: `constraints::PointPointDistance`
//! [pppa]: `constraints::PointPointPointAngle`
//! [pli]: `constraints::PointLineIncidence`
//! [pld]: `constraints::PointLineDistance`
//! [pci]: `constraints::PointCircleIncidence`
//! [lla]: `constraints::LineLineAngle`
//! [llp]: `constraints::LineLineParallelism`
//! [lct]: `constraints::LineCircleTangency`
//!
//! ## Decomposition
//!
//! General geometric constraint systems cannot be solved algebraically, and need to be solved
//! iteratively through numeric optimization. Generally, larger systems are harder to optimize, and
//! more often fail to converge. It is therefore useful to decompose systems into smaller systems
//! that can be solved separately.
//!
//! Geometric constraint systems can be represented as systems of equations. The equations in
//! general are non-linear. These systems of equations can be seen as a bipartite graph with the
//! free variables being one set of vertices and the equations the other set. Edges between
//! variables and equations encode which variables are inputs to which equations. For more
//! information, see, e.g., Chapter 4, Graph Representation of Constraint Networks in "Constraint
//! Management in Conceptual Design" by David Serrano (1982). Fiksi currently implements a
//! decomposition based on systems of equations.
//!
//! By finding a [maximum cardinality matching][matching] on this bipartite graph, we find a
//! possible order in which equations fix variables. We can use this maximum matching to define a
//! directionality on edges: all edges point from variables to equations; further, if a variable
//! and equation are matched in the maximum matching, their edge is bidirectional; and if a
//! variable is unsaturated (non of its edges are matched), all its edges are bidirectional.
//!
//! By finding strongly connected components (e.g., using [Tarjan's algorithm][tarjan]), we can
//! find sets of equations and free variables that must be solved together, as well as a partial
//! ordering between those components.
//!
//! While this is a general approach to decomposition, its application to geometric constraint
//! solving poses a challenge: as we consider unmatched free variables' edges to be bidirectional,
//! under-constrained systems are not easily decomposed. Note that unanchored but otherwise rigid
//! systems will have three unmatched free variables.
//!
//! [matching]: <https://en.wikipedia.org/wiki/Maximum_cardinality_matching>
//! [tarjan]: <https://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm>
//!
//! ## Three dimensions
//!
//! It is possible to extend Fiksi to 3D, as the theory is mostly analogous. 2D-in-3D geometry
//! would require some notion of hierarchies: 2D planes would be positioned in 3D space.
//! Constraints between elements in different 2D planes (and perhaps different 3D subspaces), would
//! work at the lowest level including all constrained elements, keeping all lower levels rigid,
//! updating this level's geometry and the transforms of the lower levels introduced at this level.

#![expect(unused_imports, reason = "Importing items to link to them in docs")]

use crate::{constraints, elements};
