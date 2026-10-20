# {{name}}

An agent built with the [OpenAI Agents Rust SDK](https://github.com/charleslyh/openai-agents-rs).

## Run it

```bash
# Nothing configured yet? The offline model answers locally, so the loop is visible first.
AGENTS_OFFLINE=1 cargo run -- "Hello"

# Then set one variable and run the same command again.
OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-4o-mini cargo run -- "Hello"
```

The `providers` feature of the SDK is on in `Cargo.toml`. It picks the provider from the
environment: a preset whose key is set (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`,
`GROQ_API_KEY`, `OPENROUTER_API_KEY`), an endpoint declared with `AGENTS_PROVIDER_1_ENDPOINT`, or the
offline model with `AGENTS_OFFLINE=1`. `AGENTS_MODEL` names the model.

## What to read next

- `src/main.rs` is the whole agent: instructions, a model, and one run.
- The SDK's examples, one per feature, are in the repository under `examples/src/`.
- `docs/DEVIATIONS.md` lists where this SDK goes beyond the Python one, and
  `docs/MIGRATION-FROM-PYTHON.md` maps the API names you already know.
