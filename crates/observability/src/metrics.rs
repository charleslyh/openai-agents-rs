//! What a run cost, counted from the span stream (Rust-only, X-018).
//!
//! A log of spans answers "what happened"; the numbers an operator watches answer "is it well":
//! how many turns, how often a tool fails, how many tokens, what it costs. [`Metrics`] counts those
//! from the same events an exporter writes, so there is one stream and one place to look.
//!
//! Cost needs a price, and prices change, so they are not baked in: [`PriceTable`] maps a model
//! name to a price per million tokens and defaults to free, which makes every number honest about
//! what it does not know.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde_json::Value;

use super::batched::EventSink;
use super::export::ExportEvent;

/// Price of a model, in USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    /// Input (prompt) tokens.
    pub input_per_million: f64,
    /// Output (completion) tokens.
    pub output_per_million: f64,
}

impl Price {
    /// A price in USD per million tokens.
    pub fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self {
            input_per_million,
            output_per_million,
        }
    }
}

/// What each model costs. Unknown models are free, so the totals never claim more than they know.
///
/// ```rust
/// use openai_agents_observability::{Price, PriceTable};
///
/// let mut prices = PriceTable::new();
/// prices.set("gpt-4o-mini", Price::new(0.15, 0.60));
/// assert_eq!(prices.get("gpt-4o-mini").unwrap().input_per_million, 0.15);
/// assert!(prices.get("an-unpriced-model").is_none());
/// ```
#[derive(Debug, Clone, Default)]
pub struct PriceTable {
    prices: HashMap<String, Price>,
}

impl PriceTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Price for `model`, replacing any price it had.
    pub fn set(&mut self, model: impl Into<String>, price: Price) {
        self.prices.insert(model.into(), price);
    }

    /// Price for `model`; `None` when the table does not know it.
    pub fn get(&self, model: &str) -> Option<Price> {
        self.prices.get(model).copied()
    }

    /// What `tokens` of this model cost, in USD; `None` for a model the table does not know.
    pub fn cost(&self, model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        let price = self.get(model)?;
        Some(
            price.input_per_million * input_tokens as f64 / 1_000_000.0
                + price.output_per_million * output_tokens as f64 / 1_000_000.0,
        )
    }
}

/// Counters for one process, updated as spans end.
#[derive(Debug, Default)]
pub struct Metrics {
    turns: AtomicU64,
    tool_calls: AtomicU64,
    tool_failures: AtomicU64,
    llm_requests: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    /// Cost in micro-dollars, so integer counters can hold it.
    cost_micros: AtomicU64,
}

impl Metrics {
    /// Empty counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Count turns, tool calls and usage from one event.
    pub fn record(&self, event: &ExportEvent, prices: &PriceTable) {
        let body = match event {
            ExportEvent::SpanEnd(body) | ExportEvent::SpanStart(body) => body,
            _ => return,
        };
        let is_end = matches!(event, ExportEvent::SpanEnd(_));
        let kind = body.get("span_kind").and_then(Value::as_str).unwrap_or("");
        match kind {
            "turn" => {
                self.turns.fetch_add(1, Ordering::Relaxed);
            }
            "function" if is_end => {
                self.tool_calls.fetch_add(1, Ordering::Relaxed);
                if body.get("error").is_some() {
                    self.tool_failures.fetch_add(1, Ordering::Relaxed);
                }
            }
            "generation" | "response" => {
                self.llm_requests.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        if let Some(usage) = body.get("usage").and_then(Value::as_object) {
            let input = usage
                .get("input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let output = usage
                .get("output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            self.input_tokens.fetch_add(input, Ordering::Relaxed);
            self.output_tokens.fetch_add(output, Ordering::Relaxed);
            if let Some(model) = body.get("name").and_then(Value::as_str) {
                if let Some(cost) = prices.cost(model, input, output) {
                    self.cost_micros
                        .fetch_add((cost * 1_000_000.0) as u64, Ordering::Relaxed);
                }
            }
        }
    }

    /// Turns started.
    pub fn turns(&self) -> u64 {
        self.turns.load(Ordering::Relaxed)
    }

    /// Tool calls that ended.
    pub fn tool_calls(&self) -> u64 {
        self.tool_calls.load(Ordering::Relaxed)
    }

    /// Tool calls that ended with an error.
    pub fn tool_failures(&self) -> u64 {
        self.tool_failures.load(Ordering::Relaxed)
    }

    /// Share of tool calls that failed, 0.0 when there were none.
    pub fn tool_failure_rate(&self) -> f64 {
        let calls = self.tool_calls();
        if calls == 0 {
            return 0.0;
        }
        self.tool_failures() as f64 / calls as f64
    }

    /// Input tokens seen in usage.
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens.load(Ordering::Relaxed)
    }

    /// Output tokens seen in usage.
    pub fn output_tokens(&self) -> u64 {
        self.output_tokens.load(Ordering::Relaxed)
    }

    /// Model requests seen.
    pub fn llm_requests(&self) -> u64 {
        self.llm_requests.load(Ordering::Relaxed)
    }

    /// Cost in USD of the usage seen so far, counting only models [`PriceTable`] knows.
    pub fn cost_usd(&self) -> f64 {
        self.cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }
}

/// A sink that counts instead of writing, so the numbers can be read from [`Metrics`].
pub struct MetricsSink {
    metrics: Arc<Metrics>,
    prices: Arc<PriceTable>,
}

impl MetricsSink {
    /// Count into `metrics`, pricing usage with `prices`.
    pub fn new(metrics: Arc<Metrics>, prices: Arc<PriceTable>) -> Self {
        Self { metrics, prices }
    }
}

impl EventSink for MetricsSink {
    fn write(&mut self, events: &[ExportEvent]) -> Result<(), String> {
        for event in events {
            self.metrics.record(event, &self.prices);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "metrics"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn span_end(kind: &str, name: &str, usage: Option<Value>, error: bool) -> ExportEvent {
        let mut body = json!({"span_kind": kind, "name": name, "span_id": "s"});
        if let Some(usage) = usage {
            body["usage"] = usage;
        }
        if error {
            body["error"] = json!({"message": "boom"});
        }
        ExportEvent::SpanEnd(body)
    }

    #[test]
    fn turns_tools_and_usage_are_counted() {
        let mut prices = PriceTable::new();
        prices.set("m", Price::new(1.0, 2.0));
        let metrics = Metrics::new();
        metrics.record(
            &span_end(
                "turn",
                "m",
                Some(json!({"input_tokens": 10, "output_tokens": 5})),
                false,
            ),
            &prices,
        );
        metrics.record(&span_end("function", "t", None, false), &prices);
        metrics.record(&span_end("function", "t", None, true), &prices);

        assert_eq!(metrics.turns(), 1);
        assert_eq!(metrics.input_tokens(), 10);
        assert_eq!(metrics.output_tokens(), 5);
        // A span that only starts is not a completed call.
        assert_eq!(metrics.tool_calls(), 2);
        assert_eq!(metrics.tool_failures(), 1);
        assert_eq!(metrics.tool_failure_rate(), 0.5);
        // 10 input at $1/M and 5 output at $2/M.
        assert!(
            (metrics.cost_usd() - 0.000_02).abs() < 1e-9,
            "{}",
            metrics.cost_usd()
        );
    }

    #[test]
    fn an_unpriced_model_costs_nothing_rather_than_a_guess() {
        let metrics = Metrics::new();
        metrics.record(
            &span_end(
                "turn",
                "unknown",
                Some(json!({"input_tokens": 1000})),
                false,
            ),
            &PriceTable::new(),
        );
        assert_eq!(metrics.cost_usd(), 0.0);
        assert_eq!(metrics.input_tokens(), 1000);
    }
}
