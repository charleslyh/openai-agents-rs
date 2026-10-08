//! Rust side of parity scenarios (compares against Python golden files).

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use openai_agents::testing::{ModelStep, ScriptedModel};
use openai_agents::{
    Agent, CustomOutputSchema, FunctionTool, RunItem, RunOptions, Runner, ToolUseBehavior,
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
    final_output: String,
    raw_response_count: usize,
    /// Number of `ToolCallOutputItem`s the run must have produced.
    #[serde(default)]
    tool_output_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Golden {
    final_output: String,
    raw_response_count: usize,
    last_agent: String,
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
        let scenario: Scenario = serde_json::from_str(&text).expect("parse scenario");
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

#[tokio::test]
async fn parity_scenarios_match_expect_and_golden() {
    for (path, scenario) in load_scenarios() {
        let steps: Vec<ModelStep> = scenario
            .steps
            .into_iter()
            .map(|s| ModelStep::output(s.output))
            .collect();
        let model = Arc::new(ScriptedModel::new(steps));
        let tools = scenario
            .agent
            .tools
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
                            async move { Err::<Value, _>(openai_agents::AgentsError::tool(message)) }
                        },
                    );
                }
                FunctionTool::constant(
                    t.name,
                    t.description.unwrap_or_else(|| "tool".into()),
                    t.return_value.unwrap_or_else(|| "ok".into()),
                )
            })
            .collect();
        let mut agent = Agent::new(scenario.agent.name)
            .model(model.clone())
            .tools(tools)
            .tool_use_behavior(behavior(&scenario.agent.tool_use_behavior));
        if let Some(schema) = scenario.agent.output_type {
            agent = agent.output_type(Arc::new(CustomOutputSchema::new("output", schema, true)));
        }

        let result = Runner::run(&agent, scenario.input, RunOptions::default())
            .await
            .unwrap_or_else(|e| panic!("{} failed: {e}", scenario.name));

        // Structured outputs are JSON values, not strings, so compare as JSON.
        let expected: Value = serde_json::from_str(&scenario.expect.final_output)
            .unwrap_or_else(|_| Value::String(scenario.expect.final_output.clone()));
        assert_eq!(result.final_output, expected, "{}", scenario.name);
        assert_eq!(
            result.raw_responses.len(),
            scenario.expect.raw_response_count,
            "{}",
            scenario.name
        );
        if let Some(count) = scenario.expect.tool_output_count {
            let actual = result
                .new_items
                .iter()
                .filter(|i| matches!(i, RunItem::ToolCallOutput(_)))
                .count();
            assert_eq!(actual, count, "tool outputs {}", scenario.name);
        }
        model.assert_complete();

        let golden_path = path.with_extension("golden.json");
        if golden_path.exists() {
            let golden: Golden =
                serde_json::from_str(&fs::read_to_string(&golden_path).unwrap()).unwrap();
            assert_eq!(
                result.final_output_as_str(),
                Some(golden.final_output.as_str()),
                "golden mismatch {}",
                scenario.name
            );
            assert_eq!(
                result.raw_responses.len(),
                golden.raw_response_count,
                "golden mismatch {}",
                scenario.name
            );
            assert_eq!(result.last_agent_name, golden.last_agent);
        }
    }
}
