# opai-agents-scaffold

`cargo opai new`: create a project that uses the Rust port of the OpenAI Agents SDK
([`opai-agents`](https://crates.io/crates/opai-agents)).

```bash
cargo install --git https://github.com/charleslyh/openai-agents-rs opai-agents-scaffold   # installs `cargo-opai`
cargo opai new my-agent              # templates: chat (default), tools, hitl
cd my-agent && AGENTS_OFFLINE=1 cargo run -- "Hello"   # no key needed
```

The generated project depends on `opai-agents` from crates.io; pass `--path <dir>` to depend on a
checkout instead. Documentation and examples are in the
[repository](https://github.com/charleslyh/openai-agents-rs).

Licensed under MIT.
