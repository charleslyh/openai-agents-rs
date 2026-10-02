//! Token usage aggregation (Python: `agents.usage.Usage`).

use serde::{Deserialize, Serialize};

/// Aggregated model usage for a response or run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of model requests.
    pub requests: u64,
    /// Input tokens.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
}

impl Usage {
    /// Create usage with request count 1 and the given token counts.
    pub fn from_tokens(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            requests: 1,
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        }
    }

    /// Add another usage snapshot into this one.
    pub fn add(&mut self, other: &Usage) {
        self.requests += other.requests;
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}
