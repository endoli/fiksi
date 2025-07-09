<div align="center">

# Fiksi

A geometric and parametric constraint solver.

[![Linebender Zulip, #kurbo channel](https://img.shields.io/badge/Linebender-%23kurbo-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/260979-kurbo)
[![dependency status](https://deps.rs/repo/github/endoli/fiksi/status.svg)](https://deps.rs/repo/github/endoli/fiksi)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
[![Build status](https://github.com/endoli/fiksi/workflows/CI/badge.svg)](https://github.com/endoli/fiksi/actions)
[![Crates.io](https://img.shields.io/crates/v/fiksi.svg)](https://crates.io/crates/fiksi)
[![Docs](https://docs.rs/fiksi/badge.svg)](https://docs.rs/fiksi)

</div>

<!-- We use cargo-rdme to update the README with the contents of lib.rs.
To edit the following section, update it in lib.rs, then run:
cargo rdme --workspace-project=fiksi --heading-base-level=0
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here. 
See https://linebender.org/blog/doc-include/ for related discussion. -->
[libm]: https://crates.io/crates/libm
<!-- cargo-rdme start -->

Fiksi is a geometric and parametric constraint solver.

## Features

- `std` (enabled by default): Get floating point functions from the standard library
  (likely using your target's libc).
- `libm`: Use floating point implementations from [libm][].

At least one of `std` and `libm` is required; `std` overrides `libm`.

# Example

```rust
use fiksi::{System, constraints, elements};

let mut gcs = fiksi::System::new();

// Add three points, and constrain them into a triangle, such that
// - one corner has an angle of 10 degrees;
// - one corner has an angle of 60 degrees; and
// - the side between those corners is of length 5.
let p1 = elements::Point::create(&mut gcs, 1., 0.);
let p2 = elements::Point::create(&mut gcs, 0.8, 1.);
let p3 = elements::Point::create(&mut gcs, 1.1, 2.);

constraints::PointPointDistance::create(&mut gcs, p2, p3, 5.);
constraints::PointPointPointAngle::create(&mut gcs, p1, p2, p3, 10f64.to_radians());
constraints::PointPointPointAngle::create(&mut gcs, p2, p3, p1, 60f64.to_radians());

gcs.solve(None, fiksi::SolvingOptions::DEFAULT);
```

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Fiksi has been verified to compile with **Rust 1.85** and later.

Future versions of Fiksi might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

<details>
<summary>Click here if compiling fails.</summary>

As time has passed, some of Fiksi's dependencies could have released versions with a higher Rust requirement.
If you encounter a compilation issue due to a dependency and don't want to upgrade your Rust toolchain, then you could downgrade the dependency.

```sh
# Use the problematic dependency's name and version
cargo update -p package_name --precise 0.1.1
```

</details>

## Community

[![Linebender Zulip](https://img.shields.io/badge/Linebender%20Zulip-%23kurbo-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/260979-kurbo)

Discussion of Fiksi development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#kurbo channel](https://xi.zulipchat.com/#narrow/channel/260979-kurbo).
All public content can be read without logging in.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Contributions are welcome by pull request. The [Rust code of conduct] applies.
Please feel free to add your name to the [AUTHORS] file in any substantive pull request.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.

[Rust Code of Conduct]: https://www.rust-lang.org/policies/code-of-conduct
[AUTHORS]: ./AUTHORS
