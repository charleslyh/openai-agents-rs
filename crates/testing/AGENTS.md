# opai-agents-testing

`ScriptedModel` and the item builders — the test doubles used across the other crates' tests.
Re-exported as `opai_agents::testing` (D2).

## Hard rules

- **Depends on core, and nothing in core depends on it.** Core's own unit tests may not use
  `ScriptedModel`; that was the condition for making this a crate at all.
- Core exposes a small public `replay_response` so test doubles do not need private wire helpers.
  Prefer extending that to reaching into core internals.
