// Copyright 2025 the Fiksi Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use fiksi::{SolvingOptions, System, constraints, elements};

/// Add "hinged triangles" to the system. These are triangles of three points with pairwise
/// distance constraints, where all triangles share one of their corners.
///
/// This returns a function to reset the positions of the added elements to their initial values.
fn add_hinged_triangles(
    s: &mut System,
    number_of_triangles: u32,
) -> impl Fn(&mut System) + 'static {
    let mut handles_and_values = vec![];

    let hinge = elements::Point::create(s, 0., 0.);
    handles_and_values.push((hinge, hinge.get_value(s)));

    for n in 0..number_of_triangles {
        let p1 = elements::Point::create(s, -1., f64::from(n));
        let p2 = elements::Point::create(s, 1., f64::from(n));
        handles_and_values.push((p1, p1.get_value(s)));
        handles_and_values.push((p2, p2.get_value(s)));

        constraints::PointPointDistance::create(s, hinge, p1, 2.);
        constraints::PointPointDistance::create(s, hinge, p2, 2.);
        constraints::PointPointDistance::create(s, p1, p2, 3.);
    }

    move |s| {
        for (handle, value) in &handles_and_values {
            handle.update_value(s, value.x, value.y);
        }
    }
}

/// Benchmarks solving geometric constraint systems of various numbers of rigid triangles hinged at
/// one shared point.
///
/// See [`add_hinged_triangles`].
fn bench_hinged_triangles(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve/hinged_triangles");

    for &size in &[1, 4, 16, 64] {
        group.throughput(Throughput::Elements(size as u64));

        let mut s = System::new();
        let reset = add_hinged_triangles(&mut s, size);

        group.bench_function(format!("size={size}"), |b| {
            let s = &mut s;
            b.iter(|| {
                // This also times the time required for resetting the system's elements to their
                // initial values, but that shouldn't be too bad.
                reset(s);
                black_box(s.solve(None, SolvingOptions::DEFAULT));
            })
        });

        // As a spot-check, test whether the system actually ends up being solved.
        let sse: f64 = s
            .get_constraint_handles()
            .map(|c| c.calculate_residual(&s))
            .map(|r| r * r)
            .sum();
        assert!(sse < 1e-4, "System was not solved.");
    }
}

criterion_group!(benches, bench_hinged_triangles);
criterion_main!(benches);
