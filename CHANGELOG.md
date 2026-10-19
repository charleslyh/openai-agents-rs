# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Nothing has been released yet; the first
entry will be `0.1.0`.

## [Unreleased]

### Fixed

- Chat Completions and MCP streams no longer corrupt multi-byte characters (CJK text, emoji) that
  a network chunk boundary splits in two.
- Chat Completions streams that announce a message before a tool call no longer give both the same
  `output_index`.
- A Chat Completions request whose output JSON schema cannot be generated now fails instead of
  silently dropping `response_format`.
- A stdio MCP server that stops reading its input can no longer block `close()` forever.
- `CompactingSession` gives up on a summarizer that does not answer (`summarize_timeout`, default
  60 s) and pauses automatic compaction after a failure (`failure_cooldown`, default 60 s) instead
  of retrying at the end of every run.
- `InMemoryProcessor`, which is installed by default, is bounded (10 000 entries per collection,
  oldest dropped first) so a long-running process no longer grows without limit. Configure it with
  `InMemoryProcessor::with_capacity`.
- A poisoned lock no longer cascades into panics across the SDK.

### Changed

- **Security:** the MCP streamable-HTTP transport follows redirects only within the original origin
  (at most 5), so configured headers and request bodies never reach another host. It also sets a
  10 s connect timeout and a timeout on notifications and session close.
- `#[function_tool]` rejects unknown options at compile time. Previously a misspelt option compiled
  and was ignored.
- Streams that end without `response.completed` (Responses) or a `finish_reason` (Chat Completions)
  now log a warning, since the reply may be truncated.
- The agent loop is split into `run/turn_loop.rs`; behaviour is unchanged.

### Added

- `#[function_tool(timeout = <seconds>)]`.
- `openai_agents_core::utf8_stream::Utf8Stream` for decoding chunked byte streams.
- A `twenty_tools_ten_turns` benchmark.

### Removed

- Python-name aliases that duplicated the resolution API: `NeedsApproval`, `NeedsApprovalFn`,
  `with_needs_approval`, `with_needs_approval_fn`, `ToolApprovalItem`, `ApprovalDecision`,
  `ApprovalStore`. Use `NeedsResolution`, `with_needs_resolution`, `with_needs_resolution_fn`,
  `ToolInterruptionItem`, `ToolResolution` and `ResolutionStore`. The `needs_approval` macro option
  is still accepted.
