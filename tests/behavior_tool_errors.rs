//! B7 / B8 / B9 / B10 regression tests: tool failures, handoff + sibling calls, blocking input
//! guardrails and unknown tools.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    handoff, input_guardrail, Agent, AgentsError, FunctionTool, GuardrailFunctionOutput, Handoff,
    ModelError, RunItem, RunOptions, Runner, DEFAULT_TOOL_ERROR_MESSAGE,
};
use serde_json::{json, Value};

fn failing_tool(name: &str) -> FunctionTool {
    FunctionTool::new(
        name,
        "always fails",
        json!({"type": "object", "properties": {}, "additionalProperties": false}),
        |_ctx, _args| async { Err::<Value, _>(AgentsError::tool("secret internal detail")) },
    )
}

fn tool_output_texts(items: &[RunItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|i| match i {
            RunItem::ToolCallOutput(o) => o.raw_item["output"].as_str().map(str::to_string),
            _ => None,
        })
        .collect()
}

/// B7: a failing tool is reported to the model instead of aborting the run.
#[tokio::test]
async fn tool_error_is_returned_to_model() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("boom", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("recovered")),
    ]));
    let agent = Agent::new("a")
        .model(model.clone())
        .tools(vec![failing_tool("boom")]);
    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run must survive a tool error");
    assert_eq!(result.final_output_as_str(), Some("recovered"));
    // The exception text is never exposed to the model.
    assert_eq!(
        tool_output_texts(&result.new_items),
        vec![DEFAULT_TOOL_ERROR_MESSAGE]
    );
}

#[tokio::test]
async fn custom_failure_error_function_formats_message() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("boom", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("ok")),
    ]));
    let tool = failing_tool("boom").with_failure_error_function(|_ctx, e| format!("custom: {e}"));
    let agent = Agent::new("a").model(model).tools(vec![tool]);
    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(
        tool_output_texts(&result.new_items),
        vec!["custom: tool error: secret internal detail"]
    );
}

#[tokio::test]
async fn raise_on_error_aborts_the_run() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("boom", "{}", "c1"),
    )]));
    let agent = Agent::new("a")
        .model(model)
        .tools(vec![failing_tool("boom").raise_on_error()]);
    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, AgentsError::Tool { .. }), "{err}");
}

/// B8: sibling tools run next to a handoff and extra handoffs are answered, so no call id
/// is left without an output.
#[tokio::test]
async fn handoff_turn_runs_sibling_tools_and_answers_extra_handoffs() {
    let b = Agent::new("B").model(Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("b done"),
    )])));
    let c = Agent::new("C").model(Arc::new(ScriptedModel::new([])));
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let side = FunctionTool::new(
        "side",
        "side effect",
        json!({"type": "object", "properties": {}, "additionalProperties": false}),
        move |_ctx, _args| {
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(json!("side-ok"))
            }
        },
    );
    let output = vec![
        ItemHelpers::function_tool_call(Handoff::default_tool_name("B"), "{}", "h1"),
        ItemHelpers::function_tool_call("side", "{}", "t1"),
        ItemHelpers::function_tool_call(Handoff::default_tool_name("C"), "{}", "h2"),
    ];
    let a_model = Arc::new(ScriptedModel::new([ModelStep::from(output)]));
    let a = Agent::new("A")
        .model(a_model)
        .tools(vec![side])
        .handoffs(vec![handoff(b), handoff(c)]);

    let result = Runner::run(&a, "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.final_output_as_str(), Some("b done"));
    assert_eq!(result.last_agent_name, "B");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "sibling tool must run");
    let outputs = tool_output_texts(&result.new_items);
    assert!(outputs.contains(&"side-ok".to_string()), "{outputs:?}");
    let input = result.to_input_list();
    let answered = |id: &str| {
        input
            .iter()
            .any(|i| i["type"] == "function_call_output" && i["call_id"] == id)
    };
    assert!(
        answered("h1") && answered("t1") && answered("h2"),
        "{input:?}"
    );
    let ignored = input
        .iter()
        .find(|i| i["type"] == "function_call_output" && i["call_id"] == "h2")
        .expect("h2 output");
    assert_eq!(
        ignored["output"],
        "Multiple handoffs detected, ignoring this one."
    );
}

/// B9: a blocking guardrail runs before the model and prevents the model call when it trips.
#[tokio::test]
async fn blocking_input_guardrail_runs_before_model() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("hi"),
    )]));
    let g = input_guardrail("block", |_c, _a, _i| async {
        GuardrailFunctionOutput::trip(json!({"why": "blocked"}))
    })
    .run_in_parallel(false);
    let agent = Agent::new("a")
        .model(model.clone())
        .input_guardrails(vec![g]);
    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .unwrap_err();
    assert!(
        matches!(err, AgentsError::InputGuardrailTripwire(_)),
        "{err}"
    );
    assert!(model.calls().is_empty(), "model must not be called");
}

/// B9: a non-blocking guardrail that passes does not delay the model.
#[tokio::test]
async fn blocking_guardrail_pass_then_model_runs() {
    let ran = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&ran);
    let g = input_guardrail("ok", move |_c, _a, _i| {
        let flag = Arc::clone(&flag);
        async move {
            flag.store(true, Ordering::SeqCst);
            GuardrailFunctionOutput::pass(json!({}))
        }
    })
    .run_in_parallel(false);
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("hi"),
    )]));
    let agent = Agent::new("a")
        .model(model.clone())
        .input_guardrails(vec![g]);
    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert!(ran.load(Ordering::SeqCst));
    assert_eq!(result.input_guardrail_results.len(), 1);
}

/// B10: an unknown tool is a model behavior error, not a user error.
#[tokio::test]
async fn unknown_tool_is_model_behavior_error() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("nope", "{}", "c1"),
    )]));
    let agent = Agent::new("a").model(model);
    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, AgentsError::Model(ModelError::Behavior(m)) if m.contains("nope")),
        "{err}"
    );
}

/// B10: `tool_not_found_behavior = ReturnErrorToModel` lets the model recover.
#[tokio::test]
async fn unknown_tool_can_return_error_to_model() {
    use openai_agents::ToolNotFoundBehavior;
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("nope", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("fine")),
    ]));
    let agent = Agent::new("a").model(model);
    let mut options = RunOptions::default();
    options.run_config.tool_not_found_behavior = ToolNotFoundBehavior::ReturnErrorToModel;
    let result = Runner::run(&agent, "go", options).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("fine"));
    let outputs = tool_output_texts(&result.new_items);
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].contains("nope"), "{outputs:?}");
}

/// D-028: a custom `ToolUseBehavior` can end the run from the tool results.
#[tokio::test]
async fn custom_tool_use_behavior_can_finalize() {
    use openai_agents::{ToolUseBehavior, ToolsToFinalOutputResult};
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call("echo", "{}", "c1"),
    )]));
    let agent = Agent::new("a")
        .model(model)
        .tools(vec![FunctionTool::constant("echo", "e", "payload")])
        .tool_use_behavior(ToolUseBehavior::Custom(Arc::new(|_ctx, results| {
            ToolsToFinalOutputResult::final_output(json!(format!("got {}", results[0].tool_name)))
        })));
    let result = Runner::run(&agent, "go", RunOptions::default()).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("got echo"));
}

/// D-026: `call_model_input_filter` rewrites one call without touching the run history.
#[tokio::test]
async fn call_model_input_filter_edits_input_and_instructions() {
    use openai_agents::{ModelInputData, RunConfig};
    let model = Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("ok"))]));
    let agent = Agent::new("a").model(model.clone()).instructions("original");
    let mut options = RunOptions::default();
    options.run_config = RunConfig::default().with_call_model_input_filter(|data| async move {
        let mut input = data.model_data.input;
        input.push(json!({"role": "user", "content": "extra"}));
        Ok(ModelInputData { input, instructions: Some("filtered".into()) })
    });
    let result = Runner::run(&agent, "hi", options).await.expect("run");
    let call = &model.calls()[0];
    assert_eq!(call.system_instructions.as_deref(), Some("filtered"));
    assert_eq!(call.input.as_array().map(Vec::len), Some(2));
    assert_eq!(result.to_input_list().len(), 2, "history keeps only input + output");
}

/// D-003: `Handoff.input_filter` rewrites the history the next agent sees.
#[tokio::test]
async fn handoff_input_filter_rewrites_next_agent_input() {
    use openai_agents::handoff_input_filter;
    let b_model = Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("b"))]));
    let b = Agent::new("B").model(b_model.clone());
    let filter = handoff_input_filter(|mut data| async move {
        // Drop the handoff turn entirely and keep only the original input.
        data.new_items.clear();
        data.pre_handoff_items.clear();
        Ok(data)
    });
    let a_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call(Handoff::default_tool_name("B"), "{}", "h1"),
    )]));
    let a = Agent::new("A")
        .model(a_model)
        .handoffs(vec![handoff(b).with_input_filter(filter)]);
    Runner::run(&a, "hello", RunOptions::default()).await.expect("run");
    let input = b_model.calls()[0].input.clone();
    assert_eq!(input.as_array().map(Vec::len), Some(1), "{input}");
}

/// D-003: a run-level filter applies when the handoff has none; server-managed
/// conversations reject filters like Python.
#[tokio::test]
async fn run_level_handoff_filter_and_server_managed_conversation() {
    use openai_agents::{handoff_input_filter, RunConfig};
    let build = || {
        let b = Agent::new("B")
            .model(Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("b"))])));
        Agent::new("A")
            .model(Arc::new(ScriptedModel::new([ModelStep::from(
                ItemHelpers::function_tool_call(Handoff::default_tool_name("B"), "{}", "h1"),
            )])))
            .handoffs(vec![handoff(b)])
    };
    let mut options = RunOptions::default();
    options.run_config = RunConfig {
        handoff_input_filter: Some(handoff_input_filter(|data| async move { Ok(data) })),
        ..RunConfig::default()
    };
    Runner::run(&build(), "hi", options.clone()).await.expect("run");

    options.conversation_id = Some("conv".into());
    let err = Runner::run(&build(), "hi", options).await.unwrap_err();
    assert!(matches!(err, AgentsError::User(_)), "{err}");
}

/// D-003: `on_handoff` runs with the parsed arguments and the schema reaches the model.
#[tokio::test]
async fn on_handoff_receives_parsed_input_and_schema_is_advertised() {
    let seen = Arc::new(std::sync::Mutex::new(None::<Value>));
    let sink = Arc::clone(&seen);
    let b = Agent::new("B").model(Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("b"),
    )])));
    let schema = json!({
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"]
    });
    let h = handoff(b)
        .with_on_handoff_input(schema, move |_ctx, input| {
            let sink = Arc::clone(&sink);
            async move {
                *sink.lock().unwrap() = Some(input);
                Ok(())
            }
        })
        .expect("schema");
    let tool_name = h.tool_name.clone();
    let a_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::function_tool_call(tool_name, r#"{"reason":"billing"}"#, "h1"),
    )]));
    let a = Agent::new("A").model(a_model.clone()).handoffs(vec![h]);
    Runner::run(&a, "hi", RunOptions::default()).await.expect("run");
    assert_eq!(*seen.lock().unwrap(), Some(json!({"reason": "billing"})));
}

#[tokio::test]
async fn on_handoff_without_input_runs_and_bad_json_is_behavior_error() {
    let called = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&called);
    let b = Agent::new("B").model(Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("b"),
    )])));
    let h = handoff(b).with_on_handoff(move |_ctx| {
        let c = Arc::clone(&c);
        async move {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    });
    let name = h.tool_name.clone();
    let a = Agent::new("A")
        .model(Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::function_tool_call(
            name, "{}", "h1",
        ))])))
        .handoffs(vec![h]);
    Runner::run(&a, "hi", RunOptions::default()).await.expect("run");
    assert_eq!(called.load(Ordering::SeqCst), 1);

    let b2 = Agent::new("B2").model(Arc::new(ScriptedModel::new([])));
    let h2 = handoff(b2)
        .with_on_handoff_input(json!({"type": "object", "properties": {}}), |_c, _i| async { Ok(()) })
        .unwrap();
    let name2 = h2.tool_name.clone();
    let a2 = Agent::new("A2")
        .model(Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::function_tool_call(
            name2, "not json", "h1",
        ))])))
        .handoffs(vec![h2]);
    let err = Runner::run(&a2, "hi", RunOptions::default()).await.unwrap_err();
    assert!(matches!(err, AgentsError::Model(ModelError::Behavior(_))), "{err}");
}

/// D-029: `is_enabled` can be decided per turn from the run context; disabled tools and
/// handoffs are hidden from the model.
#[tokio::test]
async fn dynamic_is_enabled_hides_tools_and_handoffs() {
    use openai_agents::{RunContextWrapper, ToolEnabled};
    let b = Agent::new("B").model(Arc::new(ScriptedModel::new([])));
    let hidden_handoff = handoff(b).with_is_enabled(false);
    let flagged = FunctionTool::constant("flagged", "f", "x").with_is_enabled(ToolEnabled::dynamic(
        |ctx: RunContextWrapper, _agent| async move { ctx.context::<bool>().copied().unwrap_or(false) },
    ));
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::text_message("one")),
        ModelStep::from(ItemHelpers::text_message("two")),
    ]));
    let agent = Agent::new("A")
        .model(model.clone())
        .tools(vec![flagged, FunctionTool::constant("always", "a", "y")])
        .handoffs(vec![hidden_handoff]);

    Runner::run(&agent, "hi", RunOptions::default()).await.expect("run");
    let mut options = RunOptions::default();
    options.context = Some(Arc::new(true));
    Runner::run(&agent, "hi", options).await.expect("run");

    let calls = model.calls();
    assert_eq!(calls[0].tool_names, vec!["always"]);
    assert!(calls[1].tool_names.contains(&"flagged".to_string()));
    assert!(!calls[1].tool_names.iter().any(|n| n.starts_with("transfer_to")));
}

/// D-031: `cancel(AfterTurn)` finishes the turn and stops; `cancel(Immediate)` closes the stream.
#[tokio::test]
async fn streaming_cancel_after_turn_stops_before_next_turn() {
    use openai_agents::CancelMode;
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("never reached")),
    ]));
    let agent = Agent::new("a")
        .model(model.clone())
        .tools(vec![FunctionTool::constant("echo", "e", "x")]);
    let mut streamed = Runner::run_streamed(agent, "go", RunOptions::default());
    streamed.cancel(CancelMode::AfterTurn);
    let _ = streamed.collect_events().await;
    assert!(streamed.is_complete());
    assert!(streamed.is_cancelled());
    assert_eq!(streamed.final_output(), None);
    assert_eq!(model.calls().len(), 1, "second turn must not start");
}

#[tokio::test]
async fn streaming_cancel_immediate_closes_stream() {
    use openai_agents::CancelMode;
    let model = Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("hi"))]));
    let agent = Agent::new("a").model(model);
    let mut streamed = Runner::run_streamed(agent, "go", RunOptions::default());
    streamed.cancel(CancelMode::Immediate);
    assert!(streamed.next_event().await.is_none());
    assert!(streamed.is_cancelled() && streamed.is_complete());
    assert_eq!(streamed.final_output(), None);
}
