# Stability

The SDK has not been released. Until `0.1.0`, any public item can change; this file says what is
planned and how change will be handled afterwards.

## Levels

| Level | Meaning |
| --- | --- |
| **Unstable** | May change or disappear in any release. Behind a Cargo feature or documented as experimental. |
| **Stabilising** | The shape is settled and being reviewed. A change needs a changelog entry and a migration note. |
| **Stable** | Changes only in a major version (or a minor one while the version is `0.x`), with at least one release of deprecation first. |

## Crates

| Crate | Level now | Planned at `0.1.0` |
| --- | --- | --- |
| `opai-agents-core` | Unstable | Stabilising. Public structs become `#[non_exhaustive]` before release so fields can be added later. |
| `opai-agents-openai` | Unstable | Stabilising |
| `opai-agents` (facade) | Unstable | Stabilising |
| `opai-agents-macros` | Unstable | Stabilising |
| `opai-agents-mcp` | Unstable | Unstable |
| `opai-agents-sqlite` | Unstable | Unstable |
| `opai-agents-testing` | Unstable | Unstable |

## Behaviour versus API

Run semantics follow the pinned Python SDK (`vendor/PINNED_VERSION`) and are checked by the
conformance scenarios. The Rust API shape is allowed to differ from Python's; every difference is
listed in `docs/DEVIATIONS.md`.

## Serialized data

`RunState` carries a schema id (`openai-agents-rs/4` today). Older ids keep loading; a new id is
introduced only with a changelog entry. A paused run is resumed by the SDK version that wrote it or a
newer one, never an older one.

## Deprecation

Once a crate is Stabilising or Stable, an item is marked `#[deprecated(note = "use X")]` for at
least one release before removal, and the changelog lists it under **Deprecated** and later
**Removed**.

## Minimum Rust version

`rust-version` is `1.89`, checked by the `msrv` job in CI on every push: the published crates must
compile with that compiler, tests included. The floor is set by the dependencies (`uuid` needs 1.89;
1.88 fails to build it), not by the SDK's own code. One exception: the `postgres` feature needs
**1.94**, because `sqlx` 0.9 declares it and Cargo refuses it on anything older; every other
feature builds on 1.89. The project itself is developed and linted with
`1.99` (`rust-toolchain.toml`), so a newer standard-library API can slip in; the `msrv` job is what
catches it. Raising `rust-version` is a breaking change and goes in the changelog.
