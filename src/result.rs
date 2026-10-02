//! Run results (Python: `agents.result.RunResult` Phase-1 subset).

use crate::agent::Agent;
use crate::items::{ItemHelpers, InputLike, ModelResponse, ResponseInputItem, RunItem};
use crate::usage::Usage;

/// Result of a completed `Runner::run` (Python: `RunResult`).
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Original input.
    pub input: InputLike,
    /// New items generated during the run.
    pub new_items: Vec<RunItem>,
    /// Raw model responses.
    pub raw_responses: Vec<ModelResponse>,
    /// Final output (string or JSON value).
    pub final_output: serde_json::Value,
    /// Last agent that ran.
    pub last_agent_name: String,
    /// Max turns configured for the run.
    pub max_turns: Option<usize>,
    /// Aggregated usage across raw responses.
    pub usage: Usage,
}

impl RunResult {
    /// Name of the last agent (Python: `last_agent.name`).
    pub fn last_agent<'a>(&self, starting: &'a Agent) -> &'a Agent {
        // Phase-1: single-agent runs; identity is the starting agent.
        let _ = &self.last_agent_name;
        starting
    }

    /// Convenience: last response id.
    pub fn last_response_id(&self) -> Option<&str> {
        self.raw_responses.last().and_then(|r| r.response_id.as_deref())
    }

    /// Build a continuation input list (Python: `to_input_list`).
    pub fn to_input_list(&self) -> Vec<ResponseInputItem> {
        let mut items = ItemHelpers::input_to_new_input_list(&self.input);
        for item in &self.new_items {
            items.push(item.raw_item().clone());
        }
        items
    }

    /// Final output as a string when possible.
    pub fn final_output_as_str(&self) -> Option<&str> {
        self.final_output.as_str()
    }
}
