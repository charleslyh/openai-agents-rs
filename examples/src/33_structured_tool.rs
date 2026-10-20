//! A structured tool: `impl Tool` instead of a closure (Rust-only)
//!
//! `#[function_tool]` is the shortest path and `FunctionTool::typed` adds typed arguments. This
//! example is the shape for a tool that is a real component: it has state, typed arguments *and* a
//! typed result, a declared dependency, and a scheduling policy.
//!
//! In the order the tool declares them:
//!
//! - `ToolSpec::typed::<Args>(..)`: the schema comes from the argument struct, so the schema the
//!   model sees and the `Deserialize` inside `call` cannot drift apart.
//! - `ToolPolicy`: whether a call pauses for a human (decided per call, from the arguments), how
//!   long it may take, and what the model is told when it fails.
//! - `Tool::requires`: the run refuses to start, naming the tool and the missing type, instead of
//!   failing inside the first call.
//! - `ToolMiddleware`: audit logging around the call, not inside it.
//!
//! `FunctionTool::from_tool` is the adapter. The custom tool then runs through the same machinery
//! as a built-in one; "implement the trait" and "build a `FunctionTool`" are one code path.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example structured_tool
//! ```

mod common;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents::prelude::*;
use openai_agents::serde_json::{self, Value};
use openai_agents::{
    async_trait, NextTool, ToolFailureHandling, ToolMiddleware, ToolPolicy, ToolSpec,
    ToolTimeoutBehavior, TypeRequirement,
};

// ---------------------------------------------------------------------------------------------
// The typed surface: arguments in, result out.
// ---------------------------------------------------------------------------------------------

/// Arguments, as the model must send them. The doc comments become the schema descriptions.
#[derive(openai_agents::serde::Deserialize, openai_agents::schemars::JsonSchema)]
#[serde(crate = "openai_agents::serde")]
#[schemars(crate = "openai_agents::schemars")]
struct RefundArgs {
    /// Order to refund, as shown to the customer ("A-1001").
    order_id: String,
    /// Why the customer is being refunded; kept with the refund.
    reason: String,
    /// Amount in cents. Omit to refund the whole order.
    #[serde(default)]
    amount_cents: Option<u64>,
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
// The tool.
// ---------------------------------------------------------------------------------------------

/// Refunds an order. A struct with behaviour of its own, not a closure.
struct RefundTool {
    /// Refunds above this many cents pause for a person.
    approval_threshold_cents: u64,
    /// The types `call` reads from the run. Kept as a field because `Tool::requires` returns a
    /// slice.
    requires: [TypeRequirement; 1],
}

impl RefundTool {
    fn new(approval_threshold_cents: u64) -> Self {
        Self {
            approval_threshold_cents,
            requires: [TypeRequirement::of::<Payments>()],
        }
    }
}

/// The part of a refund a customer's order is worth when `amount_cents` is omitted. A real tool
/// would look the order up; the example keeps one number so the policy has something to compare.
const WHOLE_ORDER_CENTS: u64 = 2_500;

#[async_trait]
impl Tool for RefundTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::typed::<RefundArgs>(
            "refund_order",
            "Refund an order, whole or in part. Large refunds wait for human approval.",
        )
    }

    fn policy(&self) -> ToolPolicy {
        let threshold = self.approval_threshold_cents;
        ToolPolicy::new()
            // Per call, not per tool: a small refund runs, a large one waits for a person.
            .needs_resolution(NeedsResolution::dynamic(
                move |params, _call_id| async move {
                    let cents = params
                        .get("amount_cents")
                        .and_then(Value::as_u64)
                        .unwrap_or(WHOLE_ORDER_CENTS);
                    cents > threshold
                },
            ))
            .timeout(5.0)
            .timeout_behavior(ToolTimeoutBehavior::ErrorAsResult)
            // Show the model why it failed so it can tell the customer; the default hides it.
            .on_failure(ToolFailureHandling::Custom(Arc::new(|_ctx, error| {
                format!("The refund was not made: {error}")
            })))
    }

    fn requires(&self) -> &[TypeRequirement] {
        &self.requires
    }

    async fn call(&self, ctx: &ToolContext) -> Result<ToolOutput, AgentsError> {
        // Typed in, typed out. A malformed call is an error the model reads and can correct.
        let args: RefundArgs = serde_json::from_str(&ctx.tool_arguments)
            .map_err(|e| AgentsError::tool_with_source("invalid arguments for refund_order", e))?;
        // `require` names this tool if the dependency is missing; `requires` above means the run
        // would not have started without it.
        let payments = ctx.require::<Payments>()?;

        let cents = args.amount_cents.unwrap_or(WHOLE_ORDER_CENTS);
        let refund = payments.refund(&args.order_id, cents, args.reason);
        let value = serde_json::to_value(refund)
            .map_err(|e| AgentsError::tool_with_source("could not encode the refund", e))?;
        Ok(ToolOutput::output(value))
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
    // The trait object becomes the SDK's own tool type once; from here on it is indistinguishable
    // from one built with `FunctionTool::new` or `#[function_tool]`.
    let refund = FunctionTool::from_tool(RefundTool::new(10_000)).with_middleware(Audit {
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
