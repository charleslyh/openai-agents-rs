# Terminology

Four words describe how work is divided in this SDK. Code, docs, examples and comments use each for
one thing only.

| Term | Means | In code |
| --- | --- | --- |
| **Run** | One call to `Runner` (`run`, `run_streamed`, `run_blocking`, `run_state`): from the input to the final output, an interruption, or an error. | `Runner`, `RunOptions`, `RunResult`, `RunState`, `run_id` |
| **Turn** | One iteration of the loop inside a run: a model call plus the tool calls, handoffs and approvals that follow it. A run has one or more turns. | `max_turns`, `MaxTurnsExceeded`, `prepare_turn` |
| **Session** | Storage that keeps history between runs. A session is identified by a `session_id` and outlives any run, agent or process. | `Session`, `session_id`, `RunOptions::session`, `SessionSettings` |
| **Exchange** | A user message and everything up to the next user message (replies, tool calls and their outputs). It is the unit that history trimming and compaction keep or drop whole. | `recent_exchanges`, `max_exchanges`, `keep_recent_exchanges` |

## Rules

* "Turn" never means a user message or a question and answer. Use *exchange* for that, and *run*
  for one call to the runner. "Multi-turn" is not used to describe memory across runs; say "across
  runs" or "with a session".
* "Session" is the storage handle. Do not call it a conversation in prose or comments; call its
  content *history* or *stored items*.
* "Conversation" is a word of the OpenAI API, not of this SDK. It appears only as
  `RunOptions::conversation_id`, the id of a server-side Conversations API object. It is a
  different thing from a `Session`: it lives on the provider, a session lives where you store it.
* What was said in a run (prompts, replies, tool arguments) is a *transcript*; see
  `opai_agents_observability::transcript`. It belongs to one run and is not a session.
* A run's identity for tracing is the trace (`trace_id`); `group_id` links the traces of related
  runs, for example the runs of one chat thread.

## Where it came from

`Run`, `Turn` and `Session` follow the Python SDK (`max_turns`, `Session`, `Runner.run`).
`Exchange` is Rust-only: Python's `ToolOutputTrimmer(recent_turns=…)` counts the same unit under
the name "turn", which clashes with `max_turns`. The behaviour is the same; only the Rust builder
names differ (see D-039 in [`DEVIATIONS.md`](./DEVIATIONS.md)).
