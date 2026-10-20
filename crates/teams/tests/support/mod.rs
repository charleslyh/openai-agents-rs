//! Test doubles. `opai-agents-testing` cannot be used here: extension crates may not depend on one
//! another (see `scripts/check_module_deps.py`), so this is a minimal scripted model.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use opai_agents_core::error::ModelError;
use opai_agents_core::items::{ItemHelpers, ModelResponse};
use opai_agents_core::model::{Model, ModelRequest};
use opai_agents_core::usage::Usage;
use opai_agents_core::Agent;

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

/// A model that plays back `Step`s and records what it was asked.
pub struct Script {
    steps: Mutex<VecDeque<Step>>,
    inputs: Mutex<Vec<String>>,
    instructions: Mutex<Vec<String>>,
    calls: AtomicUsize,
    call_ids: AtomicUsize,
}

impl Script {
    pub fn new(steps: Vec<Step>) -> Arc<Self> {
        Arc::new(Self {
            steps: Mutex::new(steps.into()),
            inputs: Mutex::new(Vec::new()),
            instructions: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
            call_ids: AtomicUsize::new(0),
        })
    }

    pub fn push(&self, step: Step) {
        self.steps.lock().unwrap().push_back(step);
    }

    /// Number of model calls so far.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// The input of call `index`, as JSON text.
    pub fn input(&self, index: usize) -> String {
        self.inputs.lock().unwrap()[index].clone()
    }

    pub fn last_input(&self) -> String {
        self.inputs
            .lock()
            .unwrap()
            .last()
            .cloned()
            .unwrap_or_default()
    }

    pub fn last_instructions(&self) -> String {
        self.instructions
            .lock()
            .unwrap()
            .last()
            .cloned()
            .unwrap_or_default()
    }
}

#[async_trait]
impl Model for Script {
    async fn get_response(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inputs
            .lock()
            .unwrap()
            .push(serde_json::to_string(&request.input.to_owned_items()).unwrap());
        self.instructions
            .lock()
            .unwrap()
            .push(request.system_instructions.unwrap_or_default().to_string());

        let step = self.steps.lock().unwrap().pop_front();
        let output = match step {
            Some(Step::Say(text)) => vec![ItemHelpers::text_message(text)],
            Some(Step::Call { name, args }) => {
                let n = self.call_ids.fetch_add(1, Ordering::SeqCst);
                vec![ItemHelpers::function_tool_call(
                    name,
                    args,
                    format!("call-{n}"),
                )]
            }
            Some(Step::Fail(message)) => return Err(ModelError::Script(message)),
            Some(Step::Panic) => panic!("scripted panic"),
            Some(Step::Hang) => {
                std::future::pending::<()>().await;
                unreachable!()
            }
            None => return Err(ModelError::Script("the script ran out of steps".into())),
        };
        Ok(ModelResponse {
            output,
            usage: Usage::from_tokens(10, 5),
            response_id: Some("scripted".into()),
            request_id: None,
            raw_usage: None,
        })
    }
}

/// An agent that answers with `script`.
pub fn agent(name: &str, script: &Arc<Script>) -> Agent {
    Agent::new(name)
        .instructions("Do the job you are given.")
        .model(Arc::clone(script) as Arc<dyn Model>)
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
