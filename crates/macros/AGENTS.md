# opai-agents-macros

The `#[function_tool]` procedural macro (Python's `@function_tool` decorator) and `#[derive(ToolMeta)]` (a tool declared on its argument struct, X-024).

## The re-export problem (C11)

The macro expands to `opai_agents::…` paths, and the derive attributes it emits carry
`#[serde(crate = "opai_agents::serde")]` / `#[schemars(crate = "opai_agents::schemars")]`.
Those paths resolve for a user who depends on the `opai-agents` facade, but not inside this
workspace, where the crates are named `opai-agents-core` and friends.

So the macro takes a `crate = "..."` override: tests inside the workspace pass the crate they are
in, everything else uses the facade default. If you add a crate whose tests use `#[function_tool]`,
pass the override there too.
