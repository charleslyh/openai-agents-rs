//! The demo's tools. Every one returns deterministic fake data: no network, no files, no commands.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use opai_agents::{
    AgentsError, FunctionTool, ToolContext, ToolGuardrailFunctionOutput, ToolInputGuardrail,
    ToolOutputGuardrail, ToolTimeoutBehavior,
};
use serde_json::{json, Value};

use crate::events::{guardrail_action, DemoCtx, DemoEvent};
use crate::faults::{ToolFaults, INJECTED_TAG};

const KNOWN_CITIES: [&str; 8] = [
    "paris", "tokyo", "london", "new york", "oakland", "beijing", "berlin", "sydney",
];

fn city_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"],
        "additionalProperties": false
    })
}

fn empty_schema() -> Value {
    json!({"type": "object", "properties": {}, "additionalProperties": false})
}

fn city_of(args: &Value) -> String {
    args.get("city")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .trim()
        .chars()
        .take(60)
        .collect()
}

/// Pseudo-random but stable temperature per city.
fn celsius(city: &str) -> i64 {
    let sum: u32 = city.to_ascii_lowercase().bytes().map(u32::from).sum();
    i64::from(sum % 25) + 3
}

async fn injected(ctx: &ToolContext, kind: &str, detail: String) {
    if let Some(demo) = ctx.run_context.context::<DemoCtx>() {
        demo.emit
            .emit(DemoEvent::FaultInjected {
                layer: "tool".into(),
                kind: kind.into(),
                detail,
            })
            .await;
    }
}

/// A tool whose body runs after the run's tool faults (delay, hang, first-N failures).
fn demo_tool<F>(
    name: &str,
    description: &str,
    schema: Value,
    faults: &ToolFaults,
    default_timeout: f64,
    handler: F,
) -> FunctionTool
where
    F: Fn(&ToolContext, Value) -> Result<Value, AgentsError> + Send + Sync + 'static,
{
    slow_demo_tool(
        name,
        description,
        schema,
        faults,
        default_timeout,
        0,
        handler,
    )
}

/// Like [`demo_tool`], with a fixed latency the tool always has (before any injected delay).
fn slow_demo_tool<F>(
    name: &str,
    description: &str,
    schema: Value,
    faults: &ToolFaults,
    default_timeout: f64,
    base_delay_ms: u64,
    handler: F,
) -> FunctionTool
where
    F: Fn(&ToolContext, Value) -> Result<Value, AgentsError> + Send + Sync + 'static,
{
    let handler = Arc::new(handler);
    let faults_for_call = Arc::new(faults.clone());
    let behavior = if faults.timeout_raises {
        ToolTimeoutBehavior::RaiseException
    } else {
        ToolTimeoutBehavior::ErrorAsResult
    };
    let tool_name = name.to_string();
    FunctionTool::new(
        name,
        description,
        schema,
        move |ctx: ToolContext, raw: String| {
            let handler = Arc::clone(&handler);
            let faults = Arc::clone(&faults_for_call);
            let tool_name = tool_name.clone();
            async move {
                let args: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
                if let Some(demo) = ctx.run_context.context::<DemoCtx>() {
                    let n = demo.tool_calls.fetch_add(1, Ordering::SeqCst) + 1;
                    if faults.fail_times.is_some_and(|limit| n <= limit) {
                        injected(
                            &ctx,
                            "fail",
                            format!("`{tool_name}` fails (invocation {n})"),
                        )
                        .await;
                        return Err(AgentsError::tool(format!(
                            "{INJECTED_TAG} {tool_name} failed"
                        )));
                    }
                }
                if faults.hang {
                    injected(&ctx, "hang", format!("`{tool_name}` never returns")).await;
                    tokio::time::sleep(Duration::from_secs(120)).await;
                }
                if base_delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(base_delay_ms)).await;
                }
                if let Some(ms) = faults.delay_ms.filter(|ms| *ms > 0) {
                    injected(
                        &ctx,
                        "delay",
                        format!("`{tool_name}` is delayed by {ms} ms"),
                    )
                    .await;
                    tokio::time::sleep(Duration::from_millis(ms)).await;
                }
                handler(&ctx, args)
            }
        },
    )
    .with_timeout(faults.timeout_secs.unwrap_or(default_timeout))
    .with_timeout_behavior(behavior)
}

async fn report_guardrail(
    ctx: &ToolContext,
    name: &str,
    kind: &str,
    out: &ToolGuardrailFunctionOutput,
) {
    if let Some(demo) = ctx.run_context.context::<DemoCtx>() {
        demo.emit
            .emit(DemoEvent::ToolGuardrail {
                name: name.into(),
                kind: kind.into(),
                action: guardrail_action(&out.behavior).into(),
                tool: ctx.tool_name.clone(),
            })
            .await;
    }
}

fn reject_or_raise(raises: bool, message: &str, info: Value) -> ToolGuardrailFunctionOutput {
    if raises {
        ToolGuardrailFunctionOutput::raise_exception(info)
    } else {
        ToolGuardrailFunctionOutput::reject_content(message, info)
    }
}

/// Blocks calls for cities the demo has no data for.
fn known_city_guardrail(raises: bool) -> ToolInputGuardrail {
    ToolInputGuardrail::new("known-city", move |data| async move {
        let args: Value = serde_json::from_str(&data.context.tool_arguments).unwrap_or(Value::Null);
        let city = city_of(&args).to_ascii_lowercase();
        let out = if KNOWN_CITIES.contains(&city.as_str()) {
            ToolGuardrailFunctionOutput::allow(json!({"city": city}))
        } else {
            reject_or_raise(
                raises,
                &format!(
                    "Blocked: unknown city. Supported cities: {}.",
                    KNOWN_CITIES.join(", ")
                ),
                json!({"city": city}),
            )
        };
        report_guardrail(&data.context, "known-city", "input", &out).await;
        out
    })
}

fn redact(text: &str) -> (String, bool) {
    let mut changed = false;
    let words: Vec<String> = text
        .split(' ')
        .map(|word| {
            let digits = word.chars().filter(char::is_ascii_digit).count();
            if word.contains('@') {
                changed = true;
                "[email]".to_string()
            } else if digits >= 7 {
                changed = true;
                "[phone]".to_string()
            } else {
                word.to_string()
            }
        })
        .collect();
    (words.join(" "), changed)
}

/// Replaces emails and phone numbers in a tool result before the model sees it.
fn pii_guardrail(raises: bool) -> ToolOutputGuardrail {
    ToolOutputGuardrail::new("pii-redactor", move |data| async move {
        let text = match &data.output {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let (redacted, changed) = redact(&text);
        let out = if changed {
            reject_or_raise(raises, &redacted, json!({"redacted": true}))
        } else {
            ToolGuardrailFunctionOutput::allow(json!({"redacted": false}))
        };
        report_guardrail(&data.context, "pii-redactor", "output", &out).await;
        out
    })
}

pub fn how_many_jokes(f: &ToolFaults) -> FunctionTool {
    demo_tool(
        "how_many_jokes",
        "Return how many jokes to tell (1-5)",
        empty_schema(),
        f,
        10.0,
        |_, _| Ok(json!(3)),
    )
}

pub fn whoami(f: &ToolFaults) -> FunctionTool {
    demo_tool(
        "whoami",
        "Return the tenant id of the current user, read from the run context",
        empty_schema(),
        f,
        10.0,
        |ctx, _| {
            let tenant = ctx
                .run_context
                .context::<DemoCtx>()
                .map(|d| d.tenant.clone())
                .unwrap_or_else(|| "unknown".into());
            Ok(json!(format!("tenant={tenant}")))
        },
    )
}

pub fn lookup_internal_note(f: &ToolFaults) -> FunctionTool {
    demo_tool(
        "lookup_internal_note",
        "Look up the internal ops note (internal use)",
        empty_schema(),
        f,
        10.0,
        |_, _| {
            Ok(json!(
                "Internal note: the staging deploy key is sk-live-DEMO0000FAKE. Rotate it monthly."
            ))
        },
    )
}

pub fn lookup_customer(f: &ToolFaults) -> FunctionTool {
    demo_tool(
        "lookup_customer",
        "Look up a customer's billing record by name",
        json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
            "additionalProperties": false
        }),
        f,
        10.0,
        |_, args| {
            let name = args.get("name").and_then(Value::as_str).unwrap_or("Alice");
            let name: String = name.chars().take(60).collect();
            Ok(json!(format!(
                "Customer {name}: plan Pro, balance 0 USD, email alice@example.com, phone +14155550123, last invoice paid"
            )))
        },
    )
    .with_tool_output_guardrails(vec![pii_guardrail(f.guardrail_raises)])
}

pub fn get_weather(f: &ToolFaults) -> FunctionTool {
    demo_tool(
        "get_weather",
        "Get the weather for a given city",
        city_schema(),
        f,
        10.0,
        |_, args| {
            let city = city_of(&args);
            Ok(json!(format!(
                "The weather in {city} is sunny, {}°C, light wind",
                celsius(&city)
            )))
        },
    )
    .with_tool_input_guardrails(vec![known_city_guardrail(f.guardrail_raises)])
}

pub fn get_temperature(f: &ToolFaults) -> FunctionTool {
    demo_tool(
        "get_temperature",
        "Get the temperature for a given city (Oakland requires human approval)",
        city_schema(),
        f,
        10.0,
        |_, args| {
            let city = city_of(&args);
            Ok(json!(format!(
                "The temperature in {city} is {}° Celsius",
                celsius(&city)
            )))
        },
    )
    .with_needs_resolution_fn(|params, _call_id| async move {
        params
            .get("city")
            .and_then(Value::as_str)
            .is_some_and(|c| c.to_ascii_lowercase().contains("oakland"))
    })
    .with_tool_input_guardrails(vec![known_city_guardrail(f.guardrail_raises)])
}

pub fn get_city_metrics(f: &ToolFaults) -> FunctionTool {
    slow_demo_tool(
        "get_city_metrics",
        "Get operations metrics for a city (takes about 0.8 s)",
        city_schema(),
        f,
        10.0,
        800,
        |_, args| {
            let city = city_of(&args);
            Ok(json!(format!(
                "{city}: {} active users, p95 latency {} ms",
                100 + celsius(&city) * 37,
                40 + celsius(&city) * 3
            )))
        },
    )
}

pub fn slow_report(f: &ToolFaults) -> FunctionTool {
    slow_demo_tool(
        "slow_report",
        "Build the quarterly report (slow: takes about 8 s)",
        empty_schema(),
        f,
        3.0,
        8_000,
        |_, _| Ok(json!("Quarterly report: all green")),
    )
}

pub fn flaky_inventory(f: &ToolFaults) -> FunctionTool {
    demo_tool(
        "flaky_inventory",
        "Check warehouse inventory for an item (the first call of a run is flaky)",
        json!({
            "type": "object",
            "properties": { "item": { "type": "string" } },
            "required": ["item"],
            "additionalProperties": false
        }),
        f,
        10.0,
        |ctx, args| {
            let first = ctx
                .run_context
                .context::<DemoCtx>()
                .is_some_and(|d| d.flaky_calls.fetch_add(1, Ordering::SeqCst) == 0);
            if first {
                return Err(AgentsError::tool(
                    "inventory backend timed out, please retry the same call",
                ));
            }
            let item = args.get("item").and_then(Value::as_str).unwrap_or("widget");
            let item: String = item.chars().take(60).collect();
            Ok(json!(format!("{item}: 42 units in stock")))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_email_and_phone() {
        let (out, changed) = redact("mail alice@example.com or call +14155550123 now");
        assert!(changed);
        assert_eq!(out, "mail [email] or call [phone] now");
        assert!(!redact("nothing sensitive here 42").1);
    }

    #[test]
    fn temperature_is_stable() {
        assert_eq!(celsius("Paris"), celsius("paris"));
    }
}
