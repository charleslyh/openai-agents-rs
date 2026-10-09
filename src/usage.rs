//! Token usage aggregation (Python: `agents.usage.Usage`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Breakdown of input tokens (Python: `InputTokensDetails`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Input tokens served from the prompt cache.
    #[serde(default)]
    pub cached_tokens: u64,
    /// Input tokens written to the prompt cache.
    #[serde(default)]
    pub cache_write_tokens: u64,
}

/// Breakdown of output tokens (Python: `OutputTokensDetails`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Output tokens spent on reasoning.
    #[serde(default)]
    pub reasoning_tokens: u64,
}

/// Usage of one provider request (Python: `RequestUsage`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestUsage {
    /// Input tokens of this request.
    #[serde(default)]
    pub input_tokens: u64,
    /// Output tokens of this request.
    #[serde(default)]
    pub output_tokens: u64,
    /// Total tokens of this request.
    #[serde(default)]
    pub total_tokens: u64,
    /// Input token breakdown.
    #[serde(default)]
    pub input_tokens_details: InputTokensDetails,
    /// Output token breakdown.
    #[serde(default)]
    pub output_tokens_details: OutputTokensDetails,
}

/// Aggregated model usage for a response or run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of model requests.
    #[serde(default)]
    pub requests: u64,
    /// Input tokens.
    #[serde(default)]
    pub input_tokens: u64,
    /// Input token breakdown.
    #[serde(default)]
    pub input_tokens_details: InputTokensDetails,
    /// Output tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Output token breakdown.
    #[serde(default)]
    pub output_tokens_details: OutputTokensDetails,
    /// Total tokens.
    #[serde(default)]
    pub total_tokens: u64,
    /// Per-request breakdown, kept so cost can be computed per request
    /// (Python: `request_usage_entries`).
    #[serde(default)]
    pub request_usage_entries: Vec<RequestUsage>,
}

impl Usage {
    /// Create usage with request count 1 and the given token counts.
    pub fn from_tokens(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            requests: 1,
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            ..Self::default()
        }
    }

    /// Usage from a Responses API `usage` object (`input_tokens`, `output_tokens`,
    /// `total_tokens`, `input_tokens_details`, `output_tokens_details`).
    ///
    /// Missing fields count as zero, like Python's null guards; `total_tokens` is the
    /// provider's value, not recomputed.
    pub fn from_responses_usage(usage: &Value) -> Self {
        Self::from_provider_usage(
            usage,
            "input_tokens",
            "output_tokens",
            "input_tokens_details",
            "output_tokens_details",
        )
    }

    /// Usage from a Chat Completions `usage` object (`prompt_tokens`, `completion_tokens`,
    /// `prompt_tokens_details`, `completion_tokens_details`).
    pub fn from_chat_usage(usage: &Value) -> Self {
        Self::from_provider_usage(
            usage,
            "prompt_tokens",
            "completion_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        )
    }

    fn from_provider_usage(
        usage: &Value,
        input_key: &str,
        output_key: &str,
        input_details_key: &str,
        output_details_key: &str,
    ) -> Self {
        let count = |v: &Value, key: &str| v.get(key).and_then(Value::as_u64).unwrap_or(0);
        let input_details = usage.get(input_details_key).unwrap_or(&Value::Null);
        let output_details = usage.get(output_details_key).unwrap_or(&Value::Null);
        Self {
            requests: 1,
            input_tokens: count(usage, input_key),
            input_tokens_details: InputTokensDetails {
                cached_tokens: count(input_details, "cached_tokens"),
                cache_write_tokens: count(input_details, "cache_write_tokens"),
            },
            output_tokens: count(usage, output_key),
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: count(output_details, "reasoning_tokens"),
            },
            total_tokens: count(usage, "total_tokens"),
            request_usage_entries: Vec::new(),
        }
    }

    /// The Responses `usage` object for this usage, as emitted in `response.completed`.
    pub fn to_responses_usage(&self) -> Value {
        serde_json::json!({
            "input_tokens": self.input_tokens,
            "input_tokens_details": {"cached_tokens": self.input_tokens_details.cached_tokens},
            "output_tokens": self.output_tokens,
            "output_tokens_details": {"reasoning_tokens": self.output_tokens_details.reasoning_tokens},
            "total_tokens": self.total_tokens,
        })
    }

    /// Add another usage snapshot into this one (Python: `Usage.add`).
    ///
    /// A single-request snapshot with tokens also becomes a `request_usage_entries` item; a
    /// snapshot that already carries entries has them copied across instead.
    pub fn add(&mut self, other: &Usage) {
        self.requests += other.requests;
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.input_tokens_details.cached_tokens += other.input_tokens_details.cached_tokens;
        self.input_tokens_details.cache_write_tokens +=
            other.input_tokens_details.cache_write_tokens;
        self.output_tokens_details.reasoning_tokens += other.output_tokens_details.reasoning_tokens;

        if !other.request_usage_entries.is_empty() {
            self.request_usage_entries
                .extend(other.request_usage_entries.iter().copied());
        } else if other.requests == 1
            && (other.input_tokens > 0 || other.output_tokens > 0 || other.total_tokens > 0)
        {
            self.request_usage_entries.push(RequestUsage {
                input_tokens: other.input_tokens,
                output_tokens: other.output_tokens,
                total_tokens: other.total_tokens,
                input_tokens_details: other.input_tokens_details,
                output_tokens_details: other.output_tokens_details,
            });
        }
    }
}
