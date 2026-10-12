# openai-agents-mcp

The MCP client (stdio and streamable HTTP transports), tool filters, approval policies and the
server manager. Optional: reached through the facade's `mcp` feature, which is **off by default**
(D6).

## Hard rules

- **The contract stays in core.** `McpServer`, `McpTool`, `McpCallToolResult`, `McpConfig` and
  `mcp_function_tools` live in `crates/core/src/mcp/`. Anything that implements `McpServer` works,
  including an in-process source that needs no transport. Do not move these here.
- **Do not put `reqwest` or `tokio/process` in core.** That dependency is the reason this is a
  separate, optional crate.

## Testing: a known trap

Tests that create a socket or spawn a server must take the `one_at_a_time` lock for the whole test.
A leaked socket can keep a port bound and make the next test flake — this cost 12 failures in 100
runs before the lock. The cost is serialisation, which stays within a few seconds.

## Verification

```
cargo test -p openai-agents-mcp                     # unit tests
cargo test --features mcp --test mcp_interop        # Python oracle, skips without the venv
```

Remember the facade's `mcp` feature is off: the second command needs the flag or it runs nothing.
