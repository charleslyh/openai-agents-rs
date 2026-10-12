# openai-agents-openai

The OpenAI provider and the two wire adapters: Responses and Chat Completions. The only crate that
depends on `async-openai` and on `reqwest` for model calls.

## Hard rules

- **Depends on `openai-agents-core`, never the reverse.** Core holds the `Model` /
  `ModelProvider` contract (`crates/core/src/model/`); this crate implements it.
- `retry_advice.rs` holds `openai_retry_advice`, moved here from core's retry module (D7). The
  generic retry machinery stays in core.

## Verification

- Request bodies are covered by insta snapshots: `cargo test -p openai-agents-openai`.
- `cargo test --test chat_convert_parity` compares conversions against the Python oracle in
  `conformance/` and needs its venv; it skips without one.

## Note

`api_defaults_to_chat_completions` and a `CompatibleProvider` doctest live here and run
unconditionally, which is why `--no-default-features` for the workspace counts 200 rather than the
pre-split 197.
