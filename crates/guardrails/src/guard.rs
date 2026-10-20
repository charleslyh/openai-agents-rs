//! The contract every guard here implements, and how a guard becomes a guardrail.
//!
//! A [`Guard`] looks at one piece of text and says pass, fail or rewrite. That is all the
//! built-in checks need, and keeping it that small is what lets the same guard protect the input
//! of a run and its output:
//!
//! ```rust
//! use openai_agents_guardrails::{Guard, MaxLength, PiiRedactor};
//!
//! # fn build() -> Result<(), openai_agents_guardrails::BuildError> {
//! let input = MaxLength::chars(4_000)?.input();
//! let redact = PiiRedactor::new()?.input();
//! let output = PiiRedactor::new()?.output();
//! # let _ = (input, redact, output);
//! # Ok(())
//! # }
//! ```
//!
//! How the text reaches the guard depends on where it sits:
//!
//! * **Input**: the text of the user's message, or of every user message in a list of input items.
//!   An input guard is built with `run_in_parallel(false)`: it must finish before the model sees
//!   anything, because a rewrite is only worth something if the model never saw the original.
//! * **Output**: the final output. A string is checked as it is; a structured output is walked and
//!   every string in it is checked, so a redaction reaches the field that held the personal data.
//!
//! What a guard puts into the run's trace (`output_info`) is the guard's name and counts — how many
//! pieces of text were looked at and how many were rewritten. Never the text, and never what
//! matched.

use std::sync::Arc;

use serde_json::{json, Value};

use openai_agents_core::guardrail::{
    GuardrailFunctionOutput, InputGuardrail, OutputGuardrail, Severity,
};
use openai_agents_core::items::InputLike;

/// What a guard decided about one piece of text.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Decision {
    /// Leave it as it is.
    Pass,
    /// Stop the run.
    Fail {
        /// Why. Never contains the text that was checked.
        reason: String,
        /// How serious it is: whether the guardrails still running are cancelled.
        severity: Severity,
    },
    /// Carry on with this text instead.
    Rewrite(String),
}

impl Decision {
    /// A failure.
    pub fn fail(reason: impl Into<String>, severity: Severity) -> Self {
        Self::Fail {
            reason: reason.into(),
            severity,
        }
    }
}

/// What a guard decided about a whole JSON value (a structured output).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ValueDecision {
    /// Leave it as it is.
    Pass,
    /// Stop the run.
    Fail {
        /// Why. Never contains the value that was checked.
        reason: String,
        /// How serious it is.
        severity: Severity,
    },
    /// Carry on with this value instead.
    Rewrite(Value),
}

/// A check over text. See the [module docs](self) for where the text comes from.
pub trait Guard: Send + Sync + 'static {
    /// Name used in the trace and in error messages.
    fn name(&self) -> &str;

    /// Decide about one piece of text.
    fn check(&self, text: &str) -> Decision;

    /// Decide about a whole value. The default walks it and checks every string, which is right
    /// for anything that looks at text; a guard that checks the *shape* (see
    /// [`SchemaValidator`](crate::SchemaValidator)) overrides it.
    fn check_value(&self, value: &Value) -> ValueDecision {
        let mut walk = Walk::default();
        let rewritten = walk.value(self, value);
        walk.into_decision(rewritten)
    }

    /// This guard as an input guardrail: runs before the model, so it may rewrite.
    fn input(self) -> InputGuardrail
    where
        Self: Sized,
    {
        let guard = Arc::new(self);
        let name = guard.name().to_string();
        InputGuardrail::new(name, move |_ctx, _agent, input| {
            let guard = Arc::clone(&guard);
            async move { check_input(guard.as_ref(), &input) }
        })
        .run_in_parallel(false)
    }

    /// This guard as an output guardrail.
    fn output(self) -> OutputGuardrail
    where
        Self: Sized,
    {
        let guard = Arc::new(self);
        let name = guard.name().to_string();
        OutputGuardrail::new(name, move |_ctx, _agent, output| {
            let guard = Arc::clone(&guard);
            async move { check_output(guard.as_ref(), &output) }
        })
    }
}

/// Run `guard` over the run's input, as the guardrail does. Public so a guard can be tested, or
/// used outside a run, with the same text handling.
pub fn check_input(guard: &dyn Guard, input: &InputLike) -> GuardrailFunctionOutput {
    match input {
        InputLike::Text(text) => {
            let mut walk = Walk::default();
            let rewritten = walk.text(guard, text);
            conclude(
                guard.name(),
                &walk,
                walk.clone().into_decision(Value::String(rewritten)),
            )
        }
        InputLike::Items(items) => {
            let mut walk = Walk::default();
            let rewritten: Vec<Value> = items.iter().map(|i| walk.input_item(guard, i)).collect();
            conclude(
                guard.name(),
                &walk,
                walk.clone().into_decision(Value::Array(rewritten)),
            )
        }
    }
}

/// Run `guard` over a run's final output, as the guardrail does.
pub fn check_output(guard: &dyn Guard, output: &Value) -> GuardrailFunctionOutput {
    // A guard that checks the shape of the whole value gets the whole value.
    let decision = guard.check_value(output);
    let walk = Walk {
        checked: count_strings(output),
        rewritten: usize::from(matches!(decision, ValueDecision::Rewrite(_))),
        fail: None,
    };
    conclude(guard.name(), &walk, decision)
}

/// How many strings a value holds: what an output guard looked at.
fn count_strings(value: &Value) -> usize {
    match value {
        Value::String(_) => 1,
        Value::Array(items) => items.iter().map(count_strings).sum(),
        Value::Object(map) => map.values().map(count_strings).sum(),
        _ => 0,
    }
}

fn conclude(name: &str, walk: &Walk, decision: ValueDecision) -> GuardrailFunctionOutput {
    let info = json!({
        "guard": name,
        "checked": walk.checked,
        "rewritten": walk.rewritten,
    });
    match decision {
        ValueDecision::Pass => GuardrailFunctionOutput::pass(info),
        ValueDecision::Fail { reason, severity } => {
            GuardrailFunctionOutput::fail(reason, severity).with_info(info)
        }
        ValueDecision::Rewrite(value) => GuardrailFunctionOutput::transform(value).with_info(info),
    }
}

/// Collects what happened while a guard walks text: how much it looked at, what it rewrote and
/// the most serious failure.
#[derive(Debug, Clone, Default)]
struct Walk {
    checked: usize,
    rewritten: usize,
    fail: Option<(String, Severity)>,
}

impl Walk {
    fn into_decision(self, rewritten: Value) -> ValueDecision {
        match self.fail {
            Some((reason, severity)) => ValueDecision::Fail { reason, severity },
            None if self.rewritten > 0 => ValueDecision::Rewrite(rewritten),
            None => ValueDecision::Pass,
        }
    }

    /// Check one string and return what should stand in its place.
    fn text<G: Guard + ?Sized>(&mut self, guard: &G, text: &str) -> String {
        self.checked += 1;
        match guard.check(text) {
            Decision::Pass => text.to_string(),
            Decision::Rewrite(new) => {
                self.rewritten += 1;
                new
            }
            Decision::Fail { reason, severity } => {
                // The most serious failure wins; the first of equals.
                if self.fail.as_ref().is_none_or(|(_, s)| severity > *s) {
                    self.fail = Some((reason, severity));
                }
                text.to_string()
            }
        }
    }

    /// Check every string in `value`, keys excepted.
    fn value<G: Guard + ?Sized>(&mut self, guard: &G, value: &Value) -> Value {
        match value {
            Value::String(text) => Value::String(self.text(guard, text)),
            Value::Array(items) => {
                Value::Array(items.iter().map(|i| self.value(guard, i)).collect())
            }
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), self.value(guard, v)))
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    /// Check what the user wrote in one input item, and nothing else: the roles, types and ids
    /// around it are the SDK's, not the user's.
    fn input_item<G: Guard + ?Sized>(&mut self, guard: &G, item: &Value) -> Value {
        if item.get("role").and_then(Value::as_str) != Some("user") {
            return item.clone();
        }
        let mut item = item.clone();
        match item.get("content").cloned() {
            Some(Value::String(text)) => {
                item["content"] = Value::String(self.text(guard, &text));
            }
            Some(Value::Array(parts)) => {
                let parts: Vec<Value> = parts
                    .into_iter()
                    .map(|mut part| {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            let new = self.text(guard, text);
                            part["text"] = Value::String(new);
                        }
                        part
                    })
                    .collect();
                item["content"] = Value::Array(parts);
            }
            _ => {}
        }
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Upper;
    impl Guard for Upper {
        fn name(&self) -> &str {
            "upper"
        }
        fn check(&self, text: &str) -> Decision {
            if text.contains("BAD") {
                Decision::fail("bad", Severity::Medium)
            } else if text.chars().any(char::is_lowercase) {
                Decision::Rewrite(text.to_uppercase())
            } else {
                Decision::Pass
            }
        }
    }

    #[test]
    fn plain_text_is_checked_and_rewritten() {
        let out = check_input(&Upper, &InputLike::Text("hello".into()));
        assert_eq!(out.transformed(), Some(&json!("HELLO")));
        let out = check_input(&Upper, &InputLike::Text("HELLO".into()));
        assert!(out.transformed().is_none() && !out.tripwire_triggered());
        let out = check_input(&Upper, &InputLike::Text("BAD".into()));
        assert!(out.tripwire_triggered());
    }

    /// Only what the user wrote is looked at: roles and types are the SDK's.
    #[test]
    fn only_user_text_in_input_items_is_checked() {
        let items = vec![
            json!({"role": "assistant", "content": "kept as is"}),
            json!({"role": "user", "content": [
                {"type": "input_text", "text": "hi"},
                {"type": "input_image", "image_url": "x"},
            ]}),
            json!({"role": "user", "content": "yo"}),
        ];
        let out = check_input(&Upper, &InputLike::Items(items));
        let new = out.transformed().expect("rewritten").clone();
        assert_eq!(new[0]["content"], "kept as is");
        assert_eq!(new[1]["content"][0]["text"], "HI");
        assert_eq!(new[1]["content"][0]["type"], "input_text");
        assert_eq!(new[1]["content"][1]["image_url"], "x");
        assert_eq!(new[2]["content"], "YO");
    }

    #[test]
    fn a_structured_output_is_walked_and_keys_are_left_alone() {
        let out = check_output(&Upper, &json!({"name": "ada", "tags": ["x", "Y"], "n": 3}));
        assert_eq!(
            out.transformed(),
            Some(&json!({"name": "ADA", "tags": ["X", "Y"], "n": 3}))
        );
    }

    #[test]
    fn the_most_serious_failure_wins() {
        struct Two;
        impl Guard for Two {
            fn name(&self) -> &str {
                "two"
            }
            fn check(&self, text: &str) -> Decision {
                match text {
                    "a" => Decision::fail("low one", Severity::Low),
                    "b" => Decision::fail("critical one", Severity::Critical),
                    _ => Decision::Pass,
                }
            }
        }
        let out = check_output(&Two, &json!(["a", "b"]));
        match out.verdict {
            openai_agents_core::guardrail::Verdict::Fail { reason, severity } => {
                assert_eq!(reason, "critical one");
                assert_eq!(severity, Severity::Critical);
            }
            other => panic!("expected a failure, got {other:?}"),
        }
    }

    /// The trace gets the guard's name and counts, never the text.
    #[test]
    fn output_info_carries_counts_and_no_text() {
        let out = check_input(&Upper, &InputLike::Text("secret words".into()));
        assert_eq!(out.output_info["guard"], "upper");
        assert_eq!(out.output_info["checked"], 1);
        assert_eq!(out.output_info["rewritten"], 1);
        assert!(!out.output_info.to_string().contains("secret"));
        let out = check_output(&Upper, &json!({"a": "x", "b": ["y", 1]}));
        assert_eq!(out.output_info["checked"], 2);
    }
}
