//! Test helpers for the team tests: a small, call-order-friendly face over the shared
//! `opai_agents_testing::ScriptedModel` (a dev dependency; see `DEV_ONLY_DEPS` in
//! `scripts/check_module_deps.py` for why that is allowed here).

#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use opai_agents_core::items::ItemHelpers;
use opai_agents_core::model::Model;
use opai_agents_core::usage::Usage;
use opai_agents_core::Agent;
use opai_agents_testing::{ModelStep, ScriptedModel};
use serde_json::Value;

/// One scripted model answer.
pub enum Step {
    /// A final text answer (15 tokens).
    Say(String),
    /// A tool call, then the script continues with the next step.
    Call { name: String, args: String },
    /// The model call fails.
    Fail(String),
    /// The model call panics.
    Panic,
    /// The model call never returns.
    Hang,
}

impl Step {
    pub fn say(text: &str) -> Self {
        Step::Say(text.to_string())
    }

    pub fn call(name: &str, args: serde_json::Value) -> Self {
        Step::Call {
            name: name.to_string(),
            args: args.to_string(),
        }
    }
}

/// A scripted model that records what it was asked.
pub struct Script {
    model: Arc<ScriptedModel>,
    call_ids: AtomicUsize,
}

impl Script {
    pub fn new(steps: Vec<Step>) -> Arc<Self> {
        let script = Arc::new(Self {
            model: Arc::new(ScriptedModel::new([])),
            call_ids: AtomicUsize::new(0),
        });
        for step in steps {
            script.push(step);
        }
        script
    }

    pub fn push(&self, step: Step) {
        self.model.enqueue(self.convert(step));
    }

    fn convert(&self, step: Step) -> ModelStep {
        let mut converted = match step {
            Step::Say(text) => ModelStep::from(ItemHelpers::text_message(text)),
            Step::Call { name, args } => {
                let n = self.call_ids.fetch_add(1, Ordering::SeqCst);
                ModelStep::from(ItemHelpers::function_tool_call(
                    name,
                    args,
                    format!("call-{n}"),
                ))
            }
            Step::Fail(message) => return ModelStep::raise_error(message),
            Step::Panic => return ModelStep::panic("scripted panic"),
            Step::Hang => return ModelStep::hang(),
        };
        converted.usage = Usage::from_tokens(10, 5);
        converted
    }

    /// Number of model calls so far.
    pub fn calls(&self) -> usize {
        self.model.calls().len()
    }

    /// The input of call `index`, as JSON text (always an item list).
    pub fn input(&self, index: usize) -> String {
        input_json(&self.model.calls()[index].input)
    }

    pub fn last_input(&self) -> String {
        self.model
            .last_call()
            .map(|c| input_json(&c.input))
            .unwrap_or_default()
    }

    /// Names of the tools offered on the last call.
    pub fn last_tools(&self) -> Vec<String> {
        self.model
            .last_call()
            .map(|c| c.tool_names)
            .unwrap_or_default()
    }

    pub fn last_instructions(&self) -> String {
        self.model
            .last_call()
            .and_then(|c| c.system_instructions)
            .unwrap_or_default()
    }
}

fn input_json(input: &Value) -> String {
    match input {
        Value::String(text) => serde_json::json!([{"role": "user", "content": text}]).to_string(),
        other => other.to_string(),
    }
}

/// An agent that answers with `script`.
pub fn agent(name: &str, script: &Arc<Script>) -> Agent {
    Agent::new(name)
        .instructions("Do the job you are given.")
        .model(Arc::clone(&script.model) as Arc<dyn Model>)
}

/// Poll `condition` until it holds, up to five seconds.
pub async fn wait_until(mut condition: impl FnMut() -> bool) {
    for _ in 0..500 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the condition did not become true within 5 seconds");
}
