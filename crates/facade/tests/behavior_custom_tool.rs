//! A hand-written `impl Tool` goes through the same machinery as a built-in one.
//!
//! This is the contract `examples/src/12_structured_tool.rs` leans on, checked offline: the schema
//! comes from the argument struct, the policy pauses per call from the arguments, a declared
//! dependency is checked before the first turn, a failure reads as the tool's own text, and
//! middleware wraps the call.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use opai_agents::serde_json::{self, json, Value};
use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    async_trait, Agent, AgentsError, FunctionTool, NeedsResolution, NextTool, RunOptions, Runner,
    Tool, ToolContext, ToolFailureHandling, ToolMiddleware, ToolOutput, ToolPolicy, ToolSpec,
    TypeRequirement,
};

#[derive(opai_agents::serde::Deserialize, opai_agents::schemars::JsonSchema)]
#[serde(crate = "opai_agents::serde")]
#[schemars(crate = "opai_agents::schemars")]
struct RefundArgs {
    /// Order to refund.
    order_id: String,
    /// Amount in cents. Omit to refund the whole order.
    #[serde(default)]
    amount_cents: Option<u64>,
}

#[derive(Default)]
struct Ledger(Mutex<Vec<(String, u64)>>);

struct RefundTool {
    threshold: u64,
    requires: [TypeRequirement; 1],
}

impl RefundTool {
    fn new(threshold: u64) -> Self {
        Self {
            threshold,
            requires: [TypeRequirement::of::<Ledger>()],
        }
    }
}

#[async_trait]
impl Tool for RefundTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::typed::<RefundArgs>("refund_order", "Refund an order.")
    }

    fn policy(&self) -> ToolPolicy {
        let threshold = self.threshold;
        ToolPolicy::new()
            .needs_resolution(NeedsResolution::dynamic(move |params, _id| async move {
                params
                    .get("amount_cents")
                    .and_then(Value::as_u64)
                    .unwrap_or(2_500)
                    > threshold
            }))
            .on_failure(ToolFailureHandling::Custom(Arc::new(|_ctx, error| {
                format!("The refund was not made: {error}")
            })))
    }

    fn requires(&self) -> &[TypeRequirement] {
        &self.requires
    }

    async fn call(&self, ctx: &ToolContext) -> Result<ToolOutput, AgentsError> {
        let args: RefundArgs = serde_json::from_str(&ctx.tool_arguments)
            .map_err(|e| AgentsError::tool_with_source("invalid arguments for refund_order", e))?;
        let ledger = ctx.require::<Ledger>()?;
        let cents = args.amount_cents.unwrap_or(2_500);
        ledger
            .0
            .lock()
            .unwrap()
            .push((args.order_id.clone(), cents));
        Ok(ToolOutput::output(
            json!({"order_id": args.order_id, "refunded_cents": cents}),
        ))
    }
}

struct CountCalls(Arc<AtomicUsize>);

#[async_trait]
impl ToolMiddleware for CountCalls {
    async fn handle(&self, ctx: ToolContext, next: NextTool) -> Result<ToolOutput, AgentsError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        next(ctx).await
    }
}

fn call(args: Value) -> ModelStep {
    ModelStep::from(ItemHelpers::function_tool_call(
        "refund_order",
        args.to_string(),
        "c1",
    ))
}

fn text(s: &str) -> ModelStep {
    ModelStep::from(ItemHelpers::text_message(s))
}

fn agent(model: Arc<ScriptedModel>, calls: Arc<AtomicUsize>) -> Agent {
    let tool = FunctionTool::from_tool(RefundTool::new(10_000)).with_middleware(CountCalls(calls));
    Agent::new("support").model(model).tools(vec![tool])
}

#[tokio::test]
async fn the_schema_comes_from_the_argument_struct() {
    let tool = FunctionTool::from_tool(RefundTool::new(10_000));
    let schema = &tool.params_json_schema;
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], false);
    let props = schema["properties"].as_object().expect("properties");
    assert!(props.contains_key("order_id") && props.contains_key("amount_cents"));
    // The doc comment is the description the model reads.
    assert_eq!(props["order_id"]["description"], "Order to refund.");
    // The optional field is nullable and, under a strict schema, still listed as required.
    assert!(tool.strict_json_schema);
}

#[tokio::test]
async fn a_small_refund_runs_and_the_middleware_sees_it() {
    let model = Arc::new(ScriptedModel::new([
        call(json!({"order_id": "A-1", "amount_cents": 500})),
        text("done"),
    ]));
    let calls = Arc::new(AtomicUsize::new(0));
    let result = Runner::run(
        &agent(model.clone(), calls.clone()),
        "refund A-1",
        RunOptions::new().with(Ledger::default()),
    )
    .await
    .expect("run");

    assert!(
        result.interruptions.is_empty(),
        "under the threshold: no pause"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let sent = model.calls()[1].input.to_string();
    assert!(
        sent.contains("refunded_cents") && sent.contains("500"),
        "{sent}"
    );
}

#[tokio::test]
async fn a_large_refund_pauses_before_the_tool_runs() {
    let model = Arc::new(ScriptedModel::new([call(json!({
        "order_id": "A-2",
        "amount_cents": 50_000
    }))]));
    let calls = Arc::new(AtomicUsize::new(0));
    let result = Runner::run(
        &agent(model, calls.clone()),
        "refund A-2",
        RunOptions::new().with(Ledger::default()),
    )
    .await
    .expect("run");

    assert_eq!(result.interruptions.len(), 1, "over the threshold: pause");
    assert_eq!(result.interruptions[0].tool_name, "refund_order");
    assert_eq!(calls.load(Ordering::SeqCst), 0, "the tool did not run");
}

#[tokio::test]
async fn omitting_the_amount_means_the_whole_order_and_is_judged_as_such() {
    // No `amount_cents`: the policy treats it as 2_500, under the 10_000 threshold.
    let model = Arc::new(ScriptedModel::new([
        call(json!({"order_id": "A-3"})),
        text("done"),
    ]));
    let calls = Arc::new(AtomicUsize::new(0));
    let result = Runner::run(
        &agent(model, calls.clone()),
        "refund A-3",
        RunOptions::new().with(Ledger::default()),
    )
    .await
    .expect("run");
    assert!(result.interruptions.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Under a strict schema a model sends an omitted optional as `null` (seen from a real
/// endpoint: `"amount_cents": null`). That must parse as "whole order" and be judged like it.
#[tokio::test]
async fn a_null_optional_is_the_same_as_an_omitted_one() {
    let model = Arc::new(ScriptedModel::new([
        call(json!({"order_id": "A-5", "amount_cents": null})),
        text("done"),
    ]));
    let calls = Arc::new(AtomicUsize::new(0));
    let result = Runner::run(
        &agent(model.clone(), calls.clone()),
        "refund A-5",
        RunOptions::new().with(Ledger::default()),
    )
    .await
    .expect("run");

    assert!(result.interruptions.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let sent = model.calls()[1].input.to_string();
    assert!(sent.contains("2500"), "whole order, not a failure: {sent}");
}

#[tokio::test]
async fn a_missing_dependency_fails_before_the_first_model_call() {
    let model = Arc::new(ScriptedModel::new([text("never")]));
    let err = Runner::run(
        &agent(model.clone(), Arc::new(AtomicUsize::new(0))),
        "refund",
        RunOptions::new(),
    )
    .await
    .expect_err("Ledger is missing");

    let message = err.to_string();
    assert!(message.contains("refund_order"), "{message}");
    assert!(message.contains("Ledger"), "{message}");
    assert!(model.calls().is_empty(), "the model was never asked");
}

#[tokio::test]
async fn a_failing_call_reads_as_the_tools_own_text() {
    // `amount_cents` of the wrong type: the arguments do not parse, so `call` returns an error.
    let model = Arc::new(ScriptedModel::new([
        call(json!({"order_id": "A-4", "amount_cents": "lots"})),
        text("sorry"),
    ]));
    let result = Runner::run(
        &agent(model.clone(), Arc::new(AtomicUsize::new(0))),
        "refund A-4",
        RunOptions::new().with(Ledger::default()),
    )
    .await
    .expect("a tool failure is a result the model reads, not a failed run");

    assert_eq!(result.final_output_as_str(), Some("sorry"));
    let sent = model.calls()[1].input.to_string();
    assert!(sent.contains("The refund was not made"), "{sent}");
}
