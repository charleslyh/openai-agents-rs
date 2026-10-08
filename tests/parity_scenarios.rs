//! Rust side of parity scenarios (compares against Python golden files).
//!
//! Each `tests/parity/scenarios/<name>.json` is run here and by `scripts/run_parity.py`; the
//! Python run writes `<name>.golden.json`, which this test then compares against.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use openai_agents::testing::{ModelStep, ScriptedModel};
use openai_agents::{
    handoff, Agent, AgentsError, CustomOutputSchema, FunctionTool, ModelError, RunItem,
    RunErrorHandlerResult, RunErrorHandlers, RunOptions, RunResult, Runner, ToolUseBehavior,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct Scenario {
    name: String,
    input: String,
    agent: AgentSpec,
    steps: Vec<StepSpec>,
    expect: Expect,
    /// `Runner.run(max_turns=...)`.
    #[serde(default)]
    max_turns: Option<usize>,
    /// When set, a `max_turns` error handler returns this as the final output.
    #[serde(default)]
    max_turns_handler_output: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AgentSpec {
    name: String,
    #[serde(default)]
    tools: Vec<ToolSpec>,
    #[serde(default = "default_behavior")]
    tool_use_behavior: String,
    /// Optional structured-output JSON Schema (Python: `Agent.output_type`).
    #[serde(default)]
    output_type: Option<Value>,
    /// Handoff targets, each with its own scripted steps.
    #[serde(default)]
    handoffs: Vec<HandoffSpec>,
}

#[derive(Debug, Deserialize)]
struct HandoffSpec {
    name: String,
    #[serde(default)]
    steps: Vec<StepSpec>,
}

fn default_behavior() -> String {
    "run_llm_again".into()
}

#[derive(Debug, Deserialize)]
struct ToolSpec {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    return_value: Option<String>,
    /// When set the tool fails with this message (Python: the function raises).
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StepSpec {
    output: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct Expect {
    #[serde(default)]
    final_output: Option<String>,
    #[serde(default)]
    raw_response_count: Option<usize>,
    /// Number of `ToolCallOutputItem`s the run must have produced.
    #[serde(default)]
    tool_output_count: Option<usize>,
    #[serde(default)]
    last_agent: Option<String>,
    /// Python exception class the run must raise (`ModelBehaviorError`, ...).
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Golden {
    #[serde(default)]
    final_output: Option<String>,
    #[serde(default)]
    raw_response_count: Option<usize>,
    #[serde(default)]
    last_agent: Option<String>,
    #[serde(default)]
    new_item_count: Option<usize>,
    #[serde(default)]
    tool_output_count: Option<usize>,
    #[serde(default)]
    error: Option<String>,
}

fn scenario_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parity/scenarios")
}

fn load_scenarios() -> Vec<(PathBuf, Scenario)> {
    let mut out = Vec::new();
    for entry in fs::read_dir(scenario_dir()).expect("scenarios dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".golden.json"))
        {
            continue;
        }
        let text = fs::read_to_string(&path).expect("read scenario");
        let scenario: Scenario = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        out.push((path, scenario));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn behavior(s: &str) -> ToolUseBehavior {
    match s {
        "stop_on_first_tool" => ToolUseBehavior::StopOnFirstTool,
        "run_llm_again" => ToolUseBehavior::RunLlmAgain,
        other => panic!("unsupported tool_use_behavior in parity: {other}"),
    }
}

fn scripted(steps: Vec<StepSpec>) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new(
        steps.into_iter().map(|s| ModelStep::output(s.output)),
    ))
}

fn build_tools(specs: Vec<ToolSpec>) -> Vec<FunctionTool> {
    specs
        .into_iter()
        .map(|t| {
            if let Some(message) = t.error {
                return FunctionTool::new(
                    t.name,
                    t.description.unwrap_or_else(|| "tool".into()),
                    serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }),
                    move |_ctx, _args| {
                        let message = message.clone();
                        async move { Err::<Value, _>(AgentsError::tool(message)) }
                    },
                );
            }
            FunctionTool::constant(
                t.name,
                t.description.unwrap_or_else(|| "tool".into()),
                t.return_value.unwrap_or_else(|| "ok".into()),
            )
        })
        .collect()
}

/// Name of the Python exception class that corresponds to a Rust error.
fn python_error_name(err: &AgentsError) -> &'static str {
    match err {
        AgentsError::Model(ModelError::Behavior(_)) => "ModelBehaviorError",
        AgentsError::User(_) => "UserError",
        AgentsError::MaxTurns(_) => "MaxTurnsExceeded",
        AgentsError::InputGuardrailTripwire(_) => "InputGuardrailTripwireTriggered",
        AgentsError::OutputGuardrailTripwire(_) => "OutputGuardrailTripwireTriggered",
        _ => "Other",
    }
}

fn tool_output_count(result: &RunResult) -> usize {
    result
        .new_items
        .iter()
        .filter(|i| matches!(i, RunItem::ToolCallOutput(_)))
        .count()
}

#[tokio::test]
async fn parity_scenarios_match_expect_and_golden() {
    for (path, scenario) in load_scenarios() {
        let name = scenario.name.clone();
        let model = scripted(scenario.steps);
        let mut handoff_models = Vec::new();
        let mut handoffs = Vec::new();
        for target in scenario.agent.handoffs {
            let target_model = scripted(target.steps);
            handoff_models.push(target_model.clone());
            handoffs.push(handoff(Agent::new(target.name).model(target_model)));
        }
        let mut agent = Agent::new(scenario.agent.name)
            .model(model.clone())
            .tools(build_tools(scenario.agent.tools))
            .handoffs(handoffs)
            .tool_use_behavior(behavior(&scenario.agent.tool_use_behavior));
        if let Some(schema) = scenario.agent.output_type {
            agent = agent.output_type(Arc::new(CustomOutputSchema::new("output", schema, true)));
        }

        let mut options = RunOptions::default();
        if let Some(max_turns) = scenario.max_turns {
            options.max_turns = Some(max_turns);
        }
        if let Some(output) = scenario.max_turns_handler_output {
            options.error_handlers = RunErrorHandlers::default().on_max_turns(move |_| {
                let output = output.clone();
                async move { Ok(Some(RunErrorHandlerResult::new(output))) }
            });
        }
        let outcome = Runner::run(&agent, scenario.input, options).await;
        let expect = scenario.expect;

        let golden_path = path.with_extension("golden.json");
        let golden: Option<Golden> = golden_path
            .exists()
            .then(|| serde_json::from_str(&fs::read_to_string(&golden_path).unwrap()).unwrap());

        if let Some(expected_error) = &expect.error {
            let err = outcome.expect_err(&format!("{name}: expected an error"));
            assert_eq!(python_error_name(&err), expected_error, "{name}: {err}");
            if let Some(golden) = golden {
                assert_eq!(golden.error.as_deref(), Some(expected_error.as_str()), "{name}");
            }
            continue;
        }

        let result = outcome.unwrap_or_else(|e| panic!("{name} failed: {e}"));

        if let Some(final_output) = &expect.final_output {
            // Structured outputs are JSON values, not strings, so compare as JSON.
            let expected: Value = serde_json::from_str(final_output)
                .unwrap_or_else(|_| Value::String(final_output.clone()));
            assert_eq!(result.final_output, expected, "{name}");
        }
        if let Some(count) = expect.raw_response_count {
            assert_eq!(result.raw_responses.len(), count, "{name}");
        }
        if let Some(count) = expect.tool_output_count {
            assert_eq!(tool_output_count(&result), count, "tool outputs {name}");
        }
        if let Some(last_agent) = &expect.last_agent {
            assert_eq!(&result.last_agent_name, last_agent, "{name}");
        }
        model.assert_complete();
        for m in &handoff_models {
            m.assert_complete();
        }

        if let Some(golden) = golden {
            // Python's golden stores the final output as text; structured outputs are compared
            // as parsed JSON so key order does not matter.
            if let Some(text) = &golden.final_output {
                let golden_output: Value = serde_json::from_str(text)
                    .unwrap_or_else(|_| Value::String(text.clone()));
                assert_eq!(result.final_output, golden_output, "golden mismatch {name}");
            }
            if let Some(count) = golden.raw_response_count {
                assert_eq!(result.raw_responses.len(), count, "golden raw responses {name}");
            }
            if let Some(last_agent) = &golden.last_agent {
                assert_eq!(&result.last_agent_name, last_agent, "golden last agent {name}");
            }
            if let Some(count) = golden.new_item_count {
                assert_eq!(result.new_items.len(), count, "golden new_item_count {name}");
            }
            if let Some(count) = golden.tool_output_count {
                assert_eq!(tool_output_count(&result), count, "golden tool outputs {name}");
            }
        }
    }
}
