//! Ready-made guardrails: length, personal data, prompt injection, keywords, shape (Rust-only,
//! X-020).
//!
//! Every guardrail here is a [`Guard`]: a check over text that says pass, fail or rewrite, and
//! that becomes an input or an output guardrail with one call. They are built to be used without
//! reading their source, so what they do and what they cannot do is stated in each one's docs.
//!
//! | Guard | Does | Cannot |
//! |-------|------|--------|
//! | [`MaxLength`] | refuses text over a limit | tell long from abusive |
//! | [`PiiRedactor`] | rewrites emails, card numbers (Luhn), SSNs and phone numbers — or refuses | find names, addresses or a bare 10-digit number |
//! | [`PromptInjection`] | refuses the usual instruction-override phrasings | stop an injection phrased differently; it is a heuristic, not a boundary |
//! | [`BlockedKeywords`] | refuses a list of words or phrases | understand meaning |
//! | [`SchemaValidator`] | refuses output that does not match a (subset of a) JSON Schema | validate what it does not support: it refuses to be built with it |
//!
//! Three rules hold for all of them:
//!
//! * **A bad configuration fails at construction.** An empty keyword list, a pattern that is not a
//!   regular expression, a limit of zero and a schema keyword that is not supported all return a
//!   [`BuildError`]. The alternative is a filter that was never doing anything.
//! * **Rewriting is a first-class answer.** [`PiiRedactor`] carries on with the redacted text by
//!   default, and only refuses when asked.
//! * **A failure never repeats the text.** A reason says what rule was broken and, where it helps,
//!   where; it does not quote what was checked, and the trace gets only counts.
//!
//! Matching is by the `regex` crate, which runs in time linear in the text, so a pattern cannot be
//! turned into a denial of service by what a user types.
//!
//! ```rust
//! use opai_agents_guardrails::{Guard, MaxLength, PiiRedactor, PromptInjection};
//!
//! # fn build() -> Result<(), opai_agents_guardrails::BuildError> {
//! let input_guardrails = vec![
//!     MaxLength::chars(8_000)?.input(),
//!     PromptInjection::new()?.input(),
//!     PiiRedactor::new()?.input(),
//! ];
//! let output_guardrails = vec![PiiRedactor::new()?.output()];
//! # let _ = (input_guardrails, output_guardrails);
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod guard;
pub mod injection;
pub mod keywords;
pub mod length;
pub mod pii;
pub mod schema;
mod text;

pub use error::BuildError;
pub use guard::{check_input, check_output, Decision, Guard, ValueDecision};
pub use injection::PromptInjection;
pub use keywords::BlockedKeywords;
pub use length::MaxLength;
pub use pii::{PiiKind, PiiRedactor};
pub use schema::SchemaValidator;
