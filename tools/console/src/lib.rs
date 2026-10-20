//! A live console for agent runs.
//!
//! The console takes newline-delimited JSON on `POST /ingest` — spans exported by the SDK's trace
//! processors, and conversation events recorded through `RunHooks` — keeps the last few runs in
//! memory, and draws them in a browser as they arrive: a waterfall of spans next to the prompts,
//! replies and tool calls.
//!
//! It knows nothing of the SDK. The input is JSON, so a process in any language can feed it.
//!
//! It listens on loopback only. What it holds is conversation text, which is as sensitive as the
//! conversations themselves; to look at it from another machine, tunnel to it.

pub mod server;
pub mod store;
