//! Tools that do one common thing each, ready to hand to an agent (Rust-only, X-019).
//!
//! Every project writes the same four tools first — arithmetic, the time, a web page, a file — and
//! every project gets the dangerous parts of the last two wrong the first time. These are those
//! four, with the sharp edges already filed off:
//!
//! * [`calculator`](mod@calculator): arithmetic from a parser, not from `eval`.
//! * [`clock`](mod@clock): what time it is, in one unambiguous shape.
//! * [`file_read`](mod@file_read) and [`file_list`](fn@file_list): a directory the agent may
//!   read, and nothing outside it.
//! * [`web_fetch`](mod@web_fetch) (`webfetch` feature): a URL, with the private network off
//!   limits.
//!
//! The two that can reach something are the ones with limits, and the limits are the tool, not a
//! setting beside it:
//!
//! | Tool | Refuses by default |
//! |------|--------------------|
//! | [`file_read`](mod@file_read) | paths outside its directory, `..`, symlinks that leave it, files over 1 MiB |
//! | [`web_fetch`](mod@web_fetch) | anything but `http`/`https`, loopback, link-local, private and reserved ranges (in `9.`, `10.`, `11.`, `21.`, `30.` too), cloud metadata addresses, redirects that land in those, responses over 1 MiB |
//!
//! A tool that needs to reach one of those has to be asked for it explicitly, so "the agent
//! fetched `169.254.169.254`" is not something that happens by accident.

pub mod calculator;
pub mod clock;
pub mod file_read;
#[cfg(feature = "webfetch")]
pub mod web_fetch;

pub use calculator::{calculator, evaluate, CalculatorArgs, MAX_EXPRESSION};
pub use clock::{clock, format_rfc3339, parse_offset, ClockArgs};
pub use file_read::{file_list, file_read, FileArgs, MAX_FILE_BYTES};
#[cfg(feature = "webfetch")]
pub use web_fetch::{web_fetch, FetchArgs, MAX_RESPONSE_BYTES};
