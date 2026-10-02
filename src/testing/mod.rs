//! Testing helpers (Python: `agents.testing`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

use crate::tool::{FunctionTool, ToolContext};

pub use crate::items::ItemHelpers;
pub use crate::model::scripted::{ModelCall, ModelStep, ScriptedModel};

#[cfg(feature = "testing")]
mod openai_http;
#[cfg(feature = "testing")]
pub use openai_http::{MockCompletions, MockResponses, MockToolCall};

/// Records overlapping tool invocations so tests can assert parallel execution.
#[derive(Debug, Default)]
pub struct ConcurrentProbe {
    inflight: AtomicUsize,
    max_inflight: AtomicUsize,
    /// Tool names in completion order.
    pub completed: Mutex<Vec<String>>,
}

impl ConcurrentProbe {
    /// Create an empty probe.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Highest observed in-flight tool count.
    pub fn max_inflight(&self) -> usize {
        self.max_inflight.load(Ordering::SeqCst)
    }

    /// Completed tool names in finish order.
    pub fn completed_names(&self) -> Vec<String> {
        self.completed.lock().expect("probe").clone()
    }

    /// A mock tool that sleeps, then returns `output`.
    pub fn delayed_tool(
        self: &Arc<Self>,
        name: impl Into<String>,
        delay: Duration,
        output: impl Into<String>,
    ) -> FunctionTool {
        let probe = Arc::clone(self);
        let name = name.into();
        let output = output.into();
        let tool_name = name.clone();
        FunctionTool::new(
            name,
            format!("mock tool {tool_name}"),
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            move |_ctx: ToolContext, _args: String| {
                let probe = Arc::clone(&probe);
                let tool_name = tool_name.clone();
                let output = output.clone();
                async move {
                    let now = probe.inflight.fetch_add(1, Ordering::SeqCst) + 1;
                    probe.max_inflight.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(delay).await;
                    probe.inflight.fetch_sub(1, Ordering::SeqCst);
                    probe
                        .completed
                        .lock()
                        .expect("probe")
                        .push(tool_name);
                    Ok(Value::String(output))
                }
            },
        )
    }
}
