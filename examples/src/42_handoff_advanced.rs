//! Handoffs under control: arguments, history and availability
//!
//! `handoff` shows the plain form. A handoff is also a tool the model calls, and four things
//! about it can be set:
//!
//! * **Its arguments.** `with_on_handoff_input(schema, ..)` makes the model say *why* it is
//!   handing off, and runs your callback with the parsed arguments (log it, open a ticket).
//! * **Its name.** `with_tool_name` replaces the generated `transfer_to_<agent>`.
//! * **What the next agent sees.** `with_input_filter` rewrites the conversation before the next
//!   agent gets it: drop tool noise, trim to the last few turns, remove what it must not know.
//! * **Whether it exists this turn.** `with_is_enabled(ToolEnabled::dynamic(..))` decides from the
//!   run context. A disabled handoff is not in the model's tool list at all, so it cannot be
//!   chosen by mistake or by prompt injection.
//!
//! The models are scripted, so the example needs no key and prints exactly what each side
//! received:
//!
//! 1. A customer on the `free` plan asks for a refund. Triage hands off to Refunds with a reason;
//!    the filter keeps only the customer's own messages for Refunds.
//! 2. The same Triage on the `enterprise` plan is offered one more handoff (to Legal).
//!
//! ```bash
//! cargo run -p opai-agents-examples --example handoff_advanced
//! ```

use std::sync::{Arc, Mutex};

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{handoff, handoff_input_filter, Agent, Model, RunOptions, Runner, ToolEnabled};
use serde_json::{json, Value};

/// Carried in the run context; the model never sees it.
struct Customer {
    plan: &'static str,
}

struct Scene {
    triage: Agent,
    triage_model: Arc<ScriptedModel>,
    refunds_model: Arc<ScriptedModel>,
    reasons: Arc<Mutex<Vec<Value>>>,
}

fn scene(triage_steps: Vec<ModelStep>) -> Scene {
    let refunds_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("Refund issued."),
    )]));
    let refunds = Agent::new("Refunds")
        .handoff_description("Issues refunds.")
        .model(Arc::clone(&refunds_model) as Arc<dyn Model>);
    let legal = Agent::new("Legal")
        .handoff_description("Handles contracts and disputes.")
        .model(Arc::new(ScriptedModel::new([])));

    let reasons = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&reasons);
    let to_refunds = handoff(refunds)
        .with_tool_name("escalate_to_refunds")
        .with_on_handoff_input(
            json!({
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "Why this is a refund case."}},
                "required": ["reason"]
            }),
            move |_ctx, input| {
                let sink = Arc::clone(&sink);
                async move {
                    sink.lock().unwrap().push(input);
                    Ok(())
                }
            },
        )
        .expect("a valid schema")
        // Refunds sees the customer's messages only: not Triage's tool calls, not the handoff turn.
        .with_input_filter(handoff_input_filter(|mut data| async move {
            data.pre_handoff_items.clear();
            data.new_items.clear();
            Ok(data)
        }));

    let to_legal = handoff(legal).with_is_enabled(ToolEnabled::dynamic(|ctx, _agent| async move {
        ctx.context::<Customer>()
            .is_some_and(|c| c.plan == "enterprise")
    }));

    let triage_model = Arc::new(ScriptedModel::new(triage_steps));
    let triage = Agent::new("Triage")
        .instructions("Route the customer to the right team.")
        .model(Arc::clone(&triage_model) as Arc<dyn Model>)
        .handoffs(vec![to_refunds, to_legal]);
    Scene {
        triage,
        triage_model,
        refunds_model,
        reasons,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 1. Free plan: one handoff offered, a reason captured, history filtered. ----
    let free = scene(vec![ModelStep::from(ItemHelpers::function_tool_call(
        "escalate_to_refunds",
        r#"{"reason":"charged twice for the same order"}"#,
        "h1",
    ))]);
    let result = Runner::run(
        &free.triage,
        "I was charged twice for order A-1001.",
        RunOptions::new().context(Customer { plan: "free" }),
    )
    .await?;

    println!("== free plan");
    println!(
        "tools offered to the model : {:?}",
        free.triage_model.first_call().unwrap().tool_names
    );
    println!(
        "reason given for handoff   : {:?}",
        free.reasons
            .lock()
            .unwrap()
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
    );
    println!("final agent                : {}", result.last_agent_name);
    println!(
        "answer                     : {}",
        result.final_output_as_str().unwrap_or_default()
    );
    println!(
        "what Refunds received      : {}",
        free.refunds_model.first_call().unwrap().input
    );

    // ---- 2. Enterprise plan: the same agent, one more handoff available. ----
    let enterprise = scene(vec![ModelStep::from(ItemHelpers::text_message("Noted."))]);
    Runner::run(
        &enterprise.triage,
        "We need to review our contract.",
        RunOptions::new().context(Customer { plan: "enterprise" }),
    )
    .await?;
    println!("\n== enterprise plan");
    println!(
        "tools offered to the model : {:?}",
        enterprise.triage_model.first_call().unwrap().tool_names
    );
    Ok(())
}
