//! A tool declared by annotations on its argument struct (`#[derive(ToolMeta)]` + `TypedTool`).
//!
//! The same contract as `behavior_custom_tool.rs` (which covers the hand-written `impl Tool`),
//! reached from the other side: everything below is read from the struct's attributes.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use opai_agents::serde_json::{json, Value};
use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    async_trait, Agent, AgentsError, FunctionTool, NextTool, RunOptions, Runner, ToolContext,
    ToolMeta, ToolMiddleware, ToolOutput, TypedTool,
};

#[derive(Default)]
struct Ledger(Mutex<Vec<(String, u64)>>);

/// Refund an order. Large refunds wait for a person.
#[derive(opai_agents::serde::Deserialize, opai_agents::schemars::JsonSchema, ToolMeta)]
#[serde(crate = "opai_agents::serde")]
#[schemars(crate = "opai_agents::schemars")]
#[tool(
    name = "refund_order",
    timeout = 5,
    requires(Ledger),
    needs_approval_if = is_large,
    on_failure = "The refund was not made: {error}"
)]
struct RefundArgs {
    /// Order to refund.
    order_id: String,
    /// Amount in cents. Omit to refund the whole order.
    #[serde(default)]
    amount_cents: Option<u64>,
}

fn is_large(args: &RefundArgs) -> bool {
    args.amount_cents.unwrap_or(2_500) > 10_000
}

#[async_trait]
impl TypedTool for RefundArgs {
    type Output = Value;

    async fn run(self, ctx: &ToolContext) -> Result<Value, AgentsError> {
        let cents = self.amount_cents.unwrap_or(2_500);
        ctx.require::<Ledger>()?
            .0
            .lock()
            .unwrap()
            .push((self.order_id.clone(), cents));
        Ok(json!({"order_id": self.order_id, "refunded_cents": cents}))
    }
}

/// Answer pong. Only `name` and `needs_approval` are set; the rest takes the defaults.
#[derive(opai_agents::serde::Deserialize, opai_agents::schemars::JsonSchema, ToolMeta)]
#[serde(crate = "opai_agents::serde")]
#[schemars(crate = "opai_agents::schemars")]
#[tool(name = "ping", needs_approval)]
struct Ping {}

#[async_trait]
impl TypedTool for Ping {
    type Output = &'static str;
    async fn run(self, _ctx: &ToolContext) -> Result<&'static str, AgentsError> {
        Ok("pong")
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
    let tool = FunctionTool::from_typed::<RefundArgs>().with_middleware(CountCalls(calls));
    Agent::new("support").model(model).tools(vec![tool])
}

#[test]
fn the_annotations_become_the_tool_definition() {
    let tool = FunctionTool::from_typed::<RefundArgs>();
    assert_eq!(tool.name, "refund_order");
    // The struct's doc comment, joined into one line.
    assert_eq!(
        tool.description,
        "Refund an order. Large refunds wait for a person."
    );
    assert_eq!(tool.timeout_seconds, Some(5.0));
    assert_eq!(tool.requires.len(), 1);
    assert!(tool.strict_json_schema);
    // A field's doc comment is the description the model reads.
    assert_eq!(
        tool.params_json_schema["properties"]["order_id"]["description"],
        "Order to refund."
    );
}

#[test]
fn a_bare_attribute_gets_the_defaults() {
    let tool = FunctionTool::from_typed::<Ping>();
    assert_eq!(tool.name, "ping");
    assert!(tool.timeout_seconds.is_none());
    assert!(tool.requires.is_empty());
    assert!(matches!(
        tool.needs_resolution,
        opai_agents::NeedsResolution::Fixed(true)
    ));
}

#[tokio::test]
async fn a_small_refund_runs_and_returns_the_typed_result() {
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

    assert!(result.interruptions.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let sent = model.calls()[1].input.to_string();
    assert!(
        sent.contains("refunded_cents") && sent.contains("500"),
        "{sent}"
    );
}

#[tokio::test]
async fn needs_approval_if_judges_each_call_from_its_parsed_arguments() {
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

    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 0, "the tool did not run");
}

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
    assert!(model.calls()[1].input.to_string().contains("2500"));
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
    assert!(
        message.contains("refund_order") && message.contains("Ledger"),
        "{message}"
    );
    assert!(model.calls().is_empty());
}

#[tokio::test]
async fn unparsable_arguments_read_as_the_on_failure_text() {
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
    .expect("a tool failure is a result the model reads");

    assert_eq!(result.final_output_as_str(), Some("sorry"));
    let sent = model.calls()[1].input.to_string();
    assert!(sent.contains("The refund was not made"), "{sent}");
}

// ---------------------------------------------------------------------------------------------
// Constraints on a field, written as annotations: the model sees them in the schema, and the
// `Deserialize` that follows is the same struct, so the two cannot disagree on the names.
// ---------------------------------------------------------------------------------------------

/// Why a refund is made.
#[derive(opai_agents::serde::Deserialize, opai_agents::schemars::JsonSchema)]
#[serde(crate = "opai_agents::serde", rename_all = "snake_case")]
#[schemars(crate = "opai_agents::schemars")]
enum Reason {
    /// The goods arrived broken.
    Damaged,
    /// The wrong goods were sent.
    WrongItem,
    /// The customer changed their mind.
    ChangedMind,
}

/// A tool whose fields carry constraints.
#[derive(opai_agents::serde::Deserialize, opai_agents::schemars::JsonSchema, ToolMeta)]
#[serde(crate = "opai_agents::serde")]
#[schemars(crate = "opai_agents::schemars")]
#[tool(name = "constrained")]
#[allow(dead_code)]
struct Constrained {
    /// Why the customer is being refunded.
    reason: Reason,
    /// Amount in cents, between 1 and 100000.
    #[schemars(range(min = 1, max = 100_000))]
    amount_cents: u64,
    /// Note for the audit trail, 1 to 200 characters.
    #[schemars(length(min = 1, max = 200))]
    note: String,
}

#[async_trait]
impl TypedTool for Constrained {
    type Output = u8;
    async fn run(self, _ctx: &ToolContext) -> Result<u8, AgentsError> {
        Ok(1)
    }
}

#[test]
fn field_constraints_reach_the_schema_the_model_reads() {
    let schema = FunctionTool::from_typed::<Constrained>().params_json_schema;
    let props = &schema["properties"];

    // An enum is a closed set of values, each with its own description from the variant's doc.
    let reasons = props["reason"]["anyOf"]
        .as_array()
        .expect("one entry per variant");
    let values: Vec<_> = reasons
        .iter()
        .map(|r| r["const"].as_str().unwrap())
        .collect();
    assert_eq!(values, ["damaged", "wrong_item", "changed_mind"]);
    assert_eq!(reasons[1]["description"], "The wrong goods were sent.");

    assert_eq!(props["amount_cents"]["minimum"], 1);
    assert_eq!(props["amount_cents"]["maximum"], 100_000);
    assert_eq!(props["note"]["minLength"], 1);
    assert_eq!(props["note"]["maxLength"], 200);
}

/// What the schema promises and what `Deserialize` enforces are not the same set: an enum is both,
/// a range or a length is only advice to the model. A tool that must hold the line checks it in
/// `run` (the example does, for the amount).
#[test]
fn an_enum_is_enforced_on_parse_but_a_range_is_only_advertised() {
    let parse = |v: Value| opai_agents::serde_json::from_value::<Constrained>(v);

    assert!(parse(json!({"reason": "refund_me", "amount_cents": 5, "note": "x"})).is_err());
    assert!(parse(json!({"reason": "damaged", "amount_cents": 5, "note": "x"})).is_ok());
    assert!(
        parse(json!({"reason": "damaged", "amount_cents": 0, "note": ""})).is_ok(),
        "serde does not check `range` or `length`"
    );
}
