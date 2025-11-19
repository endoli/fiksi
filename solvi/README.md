<div align="center">

# Solvi

Sparse linear algebra routines.

[![Linebender Zulip, #kurbo channel](https://img.shields.io/badge/Linebender-%23kurbo-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/260979-kurbo)
[![dependency status](https://deps.rs/repo/github/endoli/fiksi/status.svg)](https://deps.rs/repo/github/endoli/fiksi)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
[![Build status](https://github.com/endoli/fiksi/workflows/CI/badge.svg)](https://github.com/endoli/fiksi/actions)
[![Crates.io](https://img.shields.io/crates/v/solvi.svg)](https://crates.io/crates/solvi)
[![Docs](https://docs.rs/solvi/badge.svg)](https://docs.rs/solvi)

</div>

Solvi provides sparse linear algebra routines.

## Minimum supported Rust Version (MSRV)

This version of Solvi has been verified to compile with **Rust 1.85** and later.

Future versions of Solvi might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

<details>
<summary>Click here if compiling fails.</summary>

As time has passed, some of Solvi's dependencies could have released versions with a higher Rust requirement.
If you encounter a compilation issue due to a dependency and don't want to upgrade your Rust toolchain, then you could downgrade the dependency.

```sh
# Use the problematic dependency's name and version
cargo update -p package_name --precise 0.1.1
```
</details>

## Community

[![Linebender Zulip](https://img.shields.io/badge/Linebender%20Zulip-%23kurbo-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/260979-kurbo)

Discussion of Solvi development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically the [#kurbo channel](https://xi.zulipchat.com/#narrow/channel/260979-kurbo).
All public content can be read without logging in.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Contributions are welcome by pull request. The [Rust code of conduct] applies.
Please feel free to add your name to the [AUTHORS] file in any substantive pull request.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.

[Rust Code of Conduct]: https://www.rust-lang.org/policies/code-of-conduct
[AUTHORS]: ./AUTHORS
