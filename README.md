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

Fiksi is a geometric and parametric constraint solver.

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

[![Linebender Zulip](https://img.shields.io/badge/Xi%20Zulip-%23kurbo-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/260979-kurbo)

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
