# openai-agents-macros

The `#[function_tool]` procedural macro (Python's `@function_tool` decorator).

## The re-export problem (C11)

The macro expands to `openai_agents::…` paths, and the derive attributes it emits carry
`#[serde(crate = "openai_agents::serde")]` / `#[schemars(crate = "openai_agents::schemars")]`.
Those paths resolve for a user who depends on the `openai-agents` facade, but not inside this
workspace, where the crates are named `openai-agents-core` and friends.

So the macro takes a `crate = "..."` override: tests inside the workspace pass the crate they are
in, everything else uses the facade default. If you add a crate whose tests use `#[function_tool]`,
pass the override there too.
