# opai-agents (facade)

The crate users depend on. It keeps the `opai-agents` package name and re-exports
`opai_agents_core::*` plus whatever optional crates the enabled features pull in, so every public
path stays `opai_agents::…`.

## Hard rules

- **Re-exports must not drift.** `Runner` is re-implemented here to add the default model provider;
  `scripts/check_runner_api.py` fails the build if its methods stop matching core's. Any other type
  that gets wrapped rather than re-exported needs the same treatment.
- **Features are the only way to add an optional dependency.**

  | feature | default | pull in |
  | --- | --- | --- |
  | `openai` | yes | `opai-agents-openai` |
  | `mcp` | **no** | `opai-agents-mcp` (D6) |
  | `sqlite` | no | `opai-agents-sqlite` |

- **`mcp` is off by default, which has a sharp edge.** `cargo test --test mcp_interop` without
  `--features mcp` compiles to zero tests and **still passes**. CI, docs and any new command must
  pass the flag explicitly.

## Examples are not in this crate

They live at the repository root in `examples/`, as their own workspace member, so they depend on
`opai-agents` the way a user's project would. Keep it that way: if an example needs something the
facade does not re-export, that is a gap in the public API — do not let it reach into `core`.

The one thing this crate still owns is `benches/`: they are a performance guard on the public API,
not user-facing code.

## Why the default provider is installed here (D1 / C10)

Core does not know about OpenAI. The facade's `default_model_provider()` returns the OpenAI
provider when the `openai` feature is on, so a program that only sets `OPENAI_API_KEY` still works
with zero configuration. Without the feature, the error message must stay actionable.

## Verification

```
cargo test --workspace --all-features          # 262
cargo test --workspace --no-default-features   # 200 (197 pre-P3 + 3, see 06-progress.md)
cargo test --features mcp --test mcp_interop   # needs the Python venv; skips without it
```
