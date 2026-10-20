//! A structured tool: the argument struct *is* the tool (Rust-only)
//!
//! `#[function_tool]` is the shortest path for a function. This example is the shape for a tool that
//! is a real component: typed arguments *and* a typed result, a declared dependency, and a
//! scheduling policy. All of it is written next to the fields, as annotations:
//!
//! - field doc comments become the parameter descriptions the model reads;
//! - the struct's doc comment becomes the tool description;
//! - `#[tool(..)]` carries the name, the timeout, the dependency (`requires`), the per-call pause
//!   for a human (`needs_approval_if`) and the text the model sees when the call fails;
//! - `impl TypedTool` is the body: it receives the parsed arguments and returns a typed result.
//!
//! A tool with state of its own (a client, a counter) implements `Tool` instead; see
//! `docs/DEVIATIONS.md` X-023. `ToolMiddleware` wraps either kind: audit logging around the call,
//! not inside it.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example structured_tool
//! ```

mod common;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use openai_agents::prelude::*;
use openai_agents::{async_trait, NextTool, ToolMeta, ToolMiddleware, TypedTool};

/// Refunds above this many cents pause for a person.
const APPROVAL_THRESHOLD_CENTS: u64 = 10_000;

/// What a customer's order is worth when `amount_cents` is omitted. A real tool would look the
/// order up; the example keeps one number so the policy has something to compare.
const WHOLE_ORDER_CENTS: u64 = 2_500;

// ---------------------------------------------------------------------------------------------
// The tool: arguments, annotations, body.
// ---------------------------------------------------------------------------------------------

/// Refund an order, whole or in part. Large refunds wait for human approval.
#[derive(openai_agents::serde::Deserialize, openai_agents::schemars::JsonSchema, ToolMeta)]
#[serde(crate = "openai_agents::serde")]
#[schemars(crate = "openai_agents::schemars")]
#[tool(
    name = "refund_order",
    timeout = 5,
    requires(Payments),
    needs_approval_if = is_large,
    on_failure = "The refund was not made: {error}"
)]
struct RefundArgs {
    /// Order to refund, as shown to the customer ("A-1001").
    order_id: String,
    /// Why the customer is being refunded; kept with the refund.
    reason: String,
    /// Amount in cents. Omit to refund the whole order.
    #[serde(default)]
    amount_cents: Option<u64>,
}

/// Decided per call from the parsed arguments: a small refund runs, a large one waits for a person.
fn is_large(args: &RefundArgs) -> bool {
    args.amount_cents.unwrap_or(WHOLE_ORDER_CENTS) > APPROVAL_THRESHOLD_CENTS
}

/// What the tool answers with. A struct instead of a loose `Value` keeps both halves of the
/// contract, arguments and result, in types the compiler checks.
#[derive(openai_agents::serde::Serialize)]
#[serde(crate = "openai_agents::serde")]
struct Refund {
    order_id: String,
    refunded_cents: u64,
    /// False when the order had already been refunded; the model is told so.
    executed: bool,
    reference: String,
    /// The reason the model gave, kept so the audit line shows why money moved.
    reason: String,
}

#[async_trait]
impl TypedTool for RefundArgs {
    type Output = Refund;

    async fn run(self, ctx: &ToolContext) -> Result<Refund, AgentsError> {
        // `requires(Payments)` above means the run would not have started without it.
        let payments = ctx.require::<Payments>()?;
        let cents = self.amount_cents.unwrap_or(WHOLE_ORDER_CENTS);
        Ok(payments.refund(&self.order_id, cents, self.reason))
    }
}

// ---------------------------------------------------------------------------------------------
// The dependency: something the tool needs and the run supplies.
// ---------------------------------------------------------------------------------------------

/// The payment system. A real tool would hold an HTTP client; this keeps an in-memory ledger so
/// the example has no moving parts besides the model.
#[derive(Default)]
struct Payments {
    ledger: Mutex<HashMap<String, u64>>,
}

impl Payments {
    /// Refund `cents` for `order`. An order is refunded once.
    fn refund(&self, order: &str, cents: u64, reason: String) -> Refund {
        let mut ledger = self.ledger.lock().expect("ledger");
        let executed = !ledger.contains_key(order);
        if executed {
            ledger.insert(order.to_string(), cents);
        }
        Refund {
            order_id: order.to_string(),
            refunded_cents: if executed { cents } else { 0 },
            executed,
            reference: if executed {
                format!("re_{}", ledger.len())
            } else {
                "already-refunded".to_string()
            },
            reason,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Middleware: work around the call.
// ---------------------------------------------------------------------------------------------

/// Writes one audit line per call, before and after. It cannot change what the tool does, but it
/// sees the arguments going in and the result coming out, which hooks (read-only observers of the
/// run) do not.
struct Audit {
    calls: AtomicUsize,
}

#[async_trait]
impl ToolMiddleware for Audit {
    async fn handle(&self, ctx: ToolContext, next: NextTool) -> Result<ToolOutput, AgentsError> {
        let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        println!("[audit #{n}] {} {}", ctx.tool_name, ctx.tool_arguments);
        let result = next(ctx).await;
        match &result {
            Ok(out) => println!("[audit #{n}] -> {:?}", out.value()),
            Err(error) => println!("[audit #{n}] -> failed: {error}"),
        }
        result
    }
}

// ---------------------------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The struct becomes the SDK's own tool type once; from here on it is indistinguishable from
    // one built with `FunctionTool::new` or `#[function_tool]`.
    let refund = FunctionTool::from_typed::<RefundArgs>().with_middleware(Audit {
        calls: AtomicUsize::new(0),
    });

    let agent = Agent::new("Support")
        .instructions(
            "You handle refund requests. Use refund_order, then tell the customer what happened \
             in one sentence.",
        )
        .model(common::model()?)
        .tools(vec![refund]);

    // The dependency the tool declared. Leave this out and the run fails before the first model
    // call, with a message naming `refund_order` and `Payments`.
    let options = RunOptions::new().with(Payments::default());

    let input = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Please refund order A-1001, the box arrived damaged.".to_string());
    let result = Runner::run(&agent, input, options).await?;
    println!("{}", result.final_output_as_str().unwrap_or_default());
    Ok(())
}
