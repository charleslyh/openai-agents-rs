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

/// D-027: a session carries history across runs and stores each run's input and output.
#[tokio::test]
async fn session_carries_history_between_runs() {
    use openai_agents::{InMemorySession, Session};
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::text_message("first answer")),
        ModelStep::from(ItemHelpers::text_message("second answer")),
    ]));
    let agent = Agent::new("a").model(model.clone());
    let session = InMemorySession::shared("conv-1");
    let options = || {
        let mut o = RunOptions::default();
        o.session = Some(session.clone());
        o
    };

    Runner::run(&agent, "one", options()).await.expect("run 1");
    assert_eq!(session.get_items(None).await.unwrap().len(), 2);

    let result = Runner::run(&agent, "two", options()).await.expect("run 2");
    assert_eq!(result.final_output_as_str(), Some("second answer"));
    let second_input = model.calls()[1].input.clone();
    assert_eq!(second_input.as_array().map(Vec::len), Some(3), "{second_input}");
    assert_eq!(session.get_items(None).await.unwrap().len(), 4);
    assert_eq!(session.get_items(Some(1)).await.unwrap().len(), 1);
    assert!(session.pop_item().await.unwrap().is_some());
    session.clear_session().await.unwrap();
    assert!(session.get_items(None).await.unwrap().is_empty());
}

/// D-027: a session is written as the run goes, like Python: the input before the first model
/// call, each turn's items when the turn ends. A failed run keeps what it completed
/// (Python, checked on the same three scenarios: `[user]`, `[user, call, output]`, two turns).
#[tokio::test]
async fn session_keeps_the_turns_a_failed_run_completed() {
    use openai_agents::{InMemorySession, Session};
    let kinds = |items: Vec<Value>| -> Vec<String> {
        items
            .iter()
            .map(|i| i["type"].as_str().or(i["role"].as_str()).unwrap().to_string())
            .collect()
    };
    let echo = || FunctionTool::constant("echo", "e", "x");
    let call = |name: &str, id: &str| ModelStep::from(ItemHelpers::function_tool_call(name, "{}", id));
    let run = |steps: Vec<ModelStep>, max_turns: Option<usize>| async move {
        let agent = Agent::new("a").model(Arc::new(ScriptedModel::new(steps))).tools(vec![echo()]);
        let session = InMemorySession::shared("conv");
        let mut options = RunOptions::default();
        options.session = Some(session.clone());
        options.max_turns = max_turns;
        assert!(Runner::run(&agent, "go", options).await.is_err());
        session.get_items(None).await.unwrap()
    };

    assert_eq!(kinds(run(vec![call("nope", "c1")], None).await), ["user"]);
    assert_eq!(
        kinds(run(vec![call("echo", "c1"), call("nope", "c2")], None).await),
        ["user", "function_call", "function_call_output"]
    );
    assert_eq!(
        kinds(run(vec![call("echo", "c1"), call("echo", "c2")], Some(2)).await),
        ["user", "function_call", "function_call_output", "function_call", "function_call_output"]
    );
}

/// D-016: `Usage::add` follows Python — entries only for single requests with tokens, and
/// an older RunState without details still deserializes.
#[test]
fn usage_add_tracks_details_and_entries() {
    use openai_agents::Usage;
    let mut total = Usage::default();
    let mut one = Usage::from_responses_usage(&json!({
        "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
        "input_tokens_details": {"cached_tokens": 4},
        "output_tokens_details": {"reasoning_tokens": 1}}));
    one.request_usage_entries.clear();
    total.add(&one);
    total.add(&Usage { requests: 1, ..Usage::default() });
    assert_eq!(total.requests, 2);
    assert_eq!(total.input_tokens_details.cached_tokens, 4);
    assert_eq!(total.request_usage_entries.len(), 1, "empty request adds no entry");

    let legacy: Usage = serde_json::from_value(json!({
        "requests": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2})).unwrap();
    assert_eq!(legacy.input_tokens_details.cached_tokens, 0);
}

/// D-009: `tool_error_formatter` rewrites the default rejection and tool-not-found messages,
/// but never an explicit rejection message.
#[tokio::test]
async fn tool_error_formatter_rewrites_default_messages() {
    use openai_agents::{RunConfig, ToolErrorKind, ToolNotFoundBehavior};
    let formatter_config = || {
        let mut config = RunConfig::default().with_tool_error_formatter(|args| async move {
            match args.kind {
                ToolErrorKind::ApprovalRejected => Some(format!("denied {}", args.tool_name)),
                ToolErrorKind::ToolNotFound => None,
            }
        });
        config.tool_not_found_behavior = ToolNotFoundBehavior::ReturnErrorToModel;
        config
    };
    let dangerous = || FunctionTool::constant("danger", "d", "ran").with_needs_approval(true);
    let run_rejected = |message: Option<&'static str>| async move {
        let model = Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call("danger", "{}", "c1")),
            ModelStep::from(ItemHelpers::text_message("done")),
        ]));
        let agent = Agent::new("a").model(model).tools(vec![dangerous()]);
        let mut options = RunOptions::default();
        options.run_config = formatter_config();
        let first = Runner::run(&agent, "go", options.clone()).await.expect("run");
        let mut state = first.to_state().expect("state");
        state.reject(&first.interruptions[0], false, message);
        let result = Runner::run_state(&agent, state, options).await.expect("resume");
        tool_output_texts(&result.new_items)
    };
    assert_eq!(run_rejected(None).await, vec!["denied danger"]);
    assert_eq!(run_rejected(Some("not today")).await, vec!["not today"]);

    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("nope", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("fine")),
    ]));
    let agent = Agent::new("a").model(model);
    let mut options = RunOptions::default();
    options.run_config = formatter_config();
    let result = Runner::run(&agent, "go", options).await.expect("run");
    assert_eq!(tool_output_texts(&result.new_items), vec!["Tool 'nope' not found."]);
}

/// D-032: a handoff beats a same-named tool; `Error` rejects the configuration up front.
#[tokio::test]
async fn tool_name_collision_policy() {
    use openai_agents::{RunConfig, ToolNameCollisionPolicy};
    let build = |model: Arc<ScriptedModel>| {
        let b = Agent::new("B").model(Arc::new(ScriptedModel::new([])));
        let h = handoff(b);
        let clash = FunctionTool::constant(h.tool_name.clone(), "clashes with the handoff", "x");
        Agent::new("A").model(model).tools(vec![clash]).handoffs(vec![h])
    };

    let model = Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("ok"))]));
    Runner::run(&build(model.clone()), "hi", RunOptions::default()).await.expect("warn");
    let names = &model.calls()[0].tool_names;
    assert_eq!(names.iter().filter(|n| n.starts_with("transfer_to_b")).count(), 1, "{names:?}");

    let model = Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("ok"))]));
    let mut options = RunOptions::default();
    options.run_config = RunConfig {
        tool_name_collision_policy: ToolNameCollisionPolicy::Error,
        ..RunConfig::default()
    };
    let err = Runner::run(&build(model.clone()), "hi", options).await.unwrap_err();
    assert!(matches!(err, AgentsError::User(_)), "{err}");
    assert!(model.calls().is_empty(), "the model must not be called");
}

/// D-033: a `max_turns` error handler turns the error into a recorded final output.
#[tokio::test]
async fn max_turns_error_handler_produces_final_output() {
    use openai_agents::{RunErrorHandlerResult, RunErrorHandlers};
    let looping = || {
        let model = Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
            ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c2")),
        ]));
        Agent::new("a")
            .model(model)
            .tools(vec![FunctionTool::constant("echo", "e", "x")])
    };
    let mut options = RunOptions::default();
    options.max_turns = Some(2);
    options.error_handlers = RunErrorHandlers::default().on_max_turns(|input| async move {
        assert!(matches!(input.error, openai_agents::RunHandledError::MaxTurns(ref e) if e.max_turns == 2));
        assert_eq!(input.run_data.output.len(), 4, "two calls and two outputs");
        Ok(Some(RunErrorHandlerResult::new("gave up")))
    });
    let result = Runner::run(&looping(), "go", options.clone()).await.expect("handled");
    assert_eq!(result.final_output_as_str(), Some("gave up"));
    let last = result.new_items.last().expect("items");
    assert!(matches!(last, RunItem::Message(_)), "synthesized message is recorded");

    options.error_handlers = RunErrorHandlers::default().on_max_turns(|_| async { Ok(None) });
    let err = Runner::run(&looping(), "go", options).await.unwrap_err();
    assert!(matches!(err, AgentsError::MaxTurns(_)), "{err}");
}

/// D-033: a handler output that does not fit the structured schema is a user error.
#[tokio::test]
async fn max_turns_handler_output_is_validated_against_the_schema() {
    use openai_agents::{AgentOutputSchema, RunErrorHandlerResult, RunErrorHandlers};
    #[derive(serde::Deserialize, openai_agents::schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Count {
        n: i64,
    }
    let build = || {
        Agent::new("a")
            .model(Arc::new(ScriptedModel::new([ModelStep::from(
                ItemHelpers::function_tool_call("echo", "{}", "c1"),
            )])))
            .tools(vec![FunctionTool::constant("echo", "e", "x")])
            .output_type(Arc::new(AgentOutputSchema::of::<Count>().expect("schema")))
    };
    let run_with = |value: Value| async move {
        let mut options = RunOptions::default();
        options.max_turns = Some(1);
        options.error_handlers = RunErrorHandlers::default().on_max_turns(move |_| {
            let value = value.clone();
            async move { Ok(Some(RunErrorHandlerResult::new(value))) }
        });
        Runner::run(&build(), "go", options).await
    };
    assert_eq!(run_with(json!({"n": 3})).await.expect("valid").final_output, json!({"n": 3}));
    let err = run_with(json!({"n": "x"})).await.unwrap_err();
    assert!(matches!(err, AgentsError::User(_)), "{err}");
}

fn nesting_agents(a_steps: Vec<ModelStep>) -> (Agent, Arc<ScriptedModel>) {
    let b_model = Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("ok"))]));
    let b = Agent::new("B").model(b_model.clone());
    let a = Agent::new("A")
        .model(Arc::new(ScriptedModel::new(a_steps)))
        .tools(vec![FunctionTool::constant("echo", "e", "x")])
        .handoffs(vec![handoff(b)]);
    (a, b_model)
}

/// D-003: `nest_handoff_history` folds the earlier transcript into one summary message whose
/// text matches what the Python SDK produces (see the `handoff_nested_history` parity scenario).
#[tokio::test]
async fn nest_handoff_history_summarizes_the_transcript() {
    use openai_agents::RunConfig;
    let (a, b_model) = nesting_agents(vec![
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::function_tool_call("transfer_to_b", "{}", "h1")),
    ]);
    let mut options = RunOptions::default();
    options.run_config = RunConfig { nest_handoff_history: true, ..RunConfig::default() };
    Runner::run(&a, "hello", options).await.expect("run");
    let input = b_model.calls()[0].input.clone();
    let items = input.as_array().expect("items");
    assert_eq!(items.len(), 1, "{input}");
    assert_eq!(items[0]["role"], "assistant");
    assert_eq!(
        items[0]["content"].as_str().unwrap(),
        "For context, here is the conversation so far between the user and the previous agent:\n\
         <CONVERSATION HISTORY>\n\
         1. user: hello\n\
         2. {\"arguments\": \"{}\", \"call_id\": \"c1\", \"name\": \"echo\", \"type\": \"function_call\", \"id\": \"1\"}\n\
         3. {\"call_id\": \"c1\", \"output\": \"x\", \"type\": \"function_call_output\"}\n\
         4. {\"arguments\": \"{}\", \"call_id\": \"h1\", \"name\": \"transfer_to_b\", \"type\": \"function_call\", \"id\": \"1\"}\n\
         5. {\"call_id\": \"h1\", \"output\": \"{\\\"assistant\\\": \\\"B\\\"}\", \"type\": \"function_call_output\"}\n\
         </CONVERSATION HISTORY>"
    );
}

/// Nesting is off by default, a per-handoff flag overrides it, and an explicit filter replaces it.
#[tokio::test]
async fn nest_handoff_history_defaults_and_overrides() {
    use openai_agents::RunConfig;
    let steps = || vec![ModelStep::from(ItemHelpers::function_tool_call("transfer_to_b", "{}", "h1"))];

    let (a, b_model) = nesting_agents(steps());
    Runner::run(&a, "hi", RunOptions::default()).await.expect("run");
    assert_eq!(b_model.calls()[0].input.as_array().map(Vec::len), Some(3), "raw history by default");

    let (mut a, b_model) = nesting_agents(steps());
    a.handoffs = a.handoffs.into_iter().map(|h| h.with_nest_handoff_history(true)).collect();
    Runner::run(&a, "hi", RunOptions::default()).await.expect("run");
    assert_eq!(b_model.calls()[0].input.as_array().map(Vec::len), Some(1), "per-handoff opt-in");

    let (mut a, b_model) = nesting_agents(steps());
    a.handoffs = a
        .handoffs
        .into_iter()
        .map(|h| h.with_input_filter(openai_agents::handoff_input_filter(|d| async move { Ok(d) })))
        .collect();
    let mut options = RunOptions::default();
    options.run_config = RunConfig { nest_handoff_history: true, ..RunConfig::default() };
    Runner::run(&a, "hi", options).await.expect("run");
    assert_eq!(b_model.calls()[0].input.as_array().map(Vec::len), Some(3), "filter replaces nesting");
}

/// A mapper receives the flattened transcript and returns the exact history; a second handoff
/// flattens the first summary instead of nesting summaries inside summaries.
#[tokio::test]
async fn nest_handoff_history_mapper_and_flattening() {
    use openai_agents::{nest_handoff_history, HandoffInputData, RunConfig, RunContextWrapper};
    let (a, b_model) = nesting_agents(vec![ModelStep::from(
        ItemHelpers::function_tool_call("transfer_to_b", "{}", "h1"),
    )]);
    let mut options = RunOptions::default();
    options.run_config = RunConfig {
        nest_handoff_history: true,
        handoff_history_mapper: Some(Arc::new(|transcript| {
            vec![json!({"role": "user", "content": format!("{} items", transcript.len())})]
        })),
        ..RunConfig::default()
    };
    Runner::run(&a, "hi", options).await.expect("run");
    assert_eq!(b_model.calls()[0].input, json!([{"role": "user", "content": "3 items"}]));

    let data = |history: Vec<Value>| HandoffInputData {
        input_history: history,
        pre_handoff_items: vec![],
        new_items: vec![],
        run_context: RunContextWrapper::default(),
    };
    let first = nest_handoff_history(data(vec![json!({"role": "user", "content": "hi"})]), None);
    let second = nest_handoff_history(data(first.input_history), None);
    let text = second.input_history[0]["content"].as_str().unwrap().to_string();
    assert_eq!(text.matches("<CONVERSATION HISTORY>").count(), 1, "{text}");
    assert!(text.contains("1. user: hi"), "{text}");
}

/// D-029: `max_function_tool_concurrency` caps how many tools of one turn run at once, results
/// stay in call order, and 0 is rejected.
#[tokio::test]
async fn max_function_tool_concurrency_limits_parallelism() {
    use openai_agents::{RunConfig, ToolExecutionConfig};
    let running = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let slow = |name: &'static str| {
        let (running, peak) = (Arc::clone(&running), Arc::clone(&peak));
        FunctionTool::new(
            name,
            "slow",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
            move |_ctx, _args| {
                let (running, peak) = (Arc::clone(&running), Arc::clone(&peak));
                async move {
                    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                    running.fetch_sub(1, Ordering::SeqCst);
                    Ok(json!(name))
                }
            },
        )
    };
    let run = |limit: Option<usize>| {
        let model = Arc::new(ScriptedModel::new([
            ModelStep::from(vec![
                ItemHelpers::function_tool_call("a", "{}", "c1"),
                ItemHelpers::function_tool_call("b", "{}", "c2"),
                ItemHelpers::function_tool_call("c", "{}", "c3"),
            ]),
            ModelStep::from(ItemHelpers::text_message("done")),
        ]));
        let agent = Agent::new("x")
            .model(model)
            .tools(vec![slow("a"), slow("b"), slow("c")]);
        let mut options = RunOptions::default();
        options.run_config = RunConfig {
            tool_execution: Some(ToolExecutionConfig { max_function_tool_concurrency: limit }),
            ..RunConfig::default()
        };
        async move { Runner::run(&agent, "go", options).await }
    };

    peak.store(0, Ordering::SeqCst);
    run(None).await.expect("unlimited");
    assert_eq!(peak.load(Ordering::SeqCst), 3);

    peak.store(0, Ordering::SeqCst);
    let result = run(Some(2)).await.expect("limited");
    assert_eq!(peak.load(Ordering::SeqCst), 2);
    assert_eq!(tool_output_texts(&result.new_items), vec!["a", "b", "c"], "call order kept");

    assert!(matches!(run(Some(0)).await.unwrap_err(), AgentsError::User(_)));
}

/// D-029: per-tool timeouts. The default reports a model-visible message, a formatter can
/// replace it, and `RaiseException` fails the run without going through `failure_error_function`.
#[tokio::test]
async fn tool_timeout_behaviors() {
    use openai_agents::ToolTimeoutBehavior;
    let sleepy = || {
        FunctionTool::new(
            "sleepy",
            "sleeps",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
            |_ctx, _args| async {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                Ok(json!("late"))
            },
        )
        .with_timeout(0.05)
    };
    let run = |tool: FunctionTool| async move {
        let model = Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call("sleepy", "{}", "c1")),
            ModelStep::from(ItemHelpers::text_message("after")),
        ]));
        Runner::run(&Agent::new("a").model(model).tools(vec![tool]), "go", RunOptions::default()).await
    };

    let result = run(sleepy()).await.expect("error as result");
    assert_eq!(tool_output_texts(&result.new_items), vec!["Tool 'sleepy' timed out after 0.05 seconds."]);

    let formatted = sleepy().with_timeout_error_function(|_ctx, e| format!("custom: {e}"));
    let result = run(formatted).await.expect("formatted");
    assert_eq!(
        tool_output_texts(&result.new_items),
        vec!["custom: Tool 'sleepy' timed out after 0.05 seconds."]
    );

    let raising = sleepy()
        .with_timeout_behavior(ToolTimeoutBehavior::RaiseException)
        .with_failure_error_function(|_c, _e| "must not be used".into());
    let err = run(raising).await.unwrap_err();
    assert!(
        matches!(&err, AgentsError::ToolTimeout(t) if t.tool_name == "sleepy" && t.timeout_seconds == 0.05),
        "{err}"
    );

    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let err = run(sleepy().with_timeout(bad)).await.unwrap_err();
        assert!(matches!(err, AgentsError::User(_)), "{bad}: {err}");
    }
}

/// D-027: `session_settings.limit` loads only the latest stored items, and
/// `session_input_callback` decides the model input while only new items are saved.
#[tokio::test]
async fn session_settings_limit_and_input_callback() {
    use openai_agents::{InMemorySession, RunConfig, Session, SessionSettings};
    let seeded = || async {
        let session = InMemorySession::shared("s");
        session
            .add_items(vec![
                json!({"role": "user", "content": "old-1"}),
                json!({"role": "assistant", "content": "old-2"}),
                json!({"role": "user", "content": "old-3"}),
            ])
            .await
            .unwrap();
        session
    };

    // limit: only the newest 2 stored items reach the model.
    let session = seeded().await;
    let model = Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("ok"))]));
    let mut options = RunOptions::default();
    options.session = Some(session.clone());
    options.run_config = RunConfig {
        session_settings: Some(SessionSettings { limit: Some(2) }),
        ..RunConfig::default()
    };
    Runner::run(&Agent::new("a").model(model.clone()), "new", options).await.expect("run");
    let input = model.calls()[0].input.clone();
    assert_eq!(
        input.as_array().unwrap().iter().map(|i| i["content"].clone()).take(3).collect::<Vec<_>>(),
        vec![json!("old-2"), json!("old-3"), json!("new")]
    );
    assert_eq!(session.get_items(None).await.unwrap().len(), 3 + 2, "saves the new input and output");

    // callback: keep only the last history item and add a note; the note is saved, history is not
    // duplicated.
    let session = seeded().await;
    let model = Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("ok"))]));
    let mut options = RunOptions::default();
    options.session = Some(session.clone());
    options.run_config = RunConfig {
        session_input_callback: Some(Arc::new(|history, new_items| {
            Box::pin(async move {
                let mut out = vec![history.last().cloned().unwrap()];
                out.push(json!({"role": "user", "content": "note"}));
                out.extend(new_items);
                Ok(out)
            })
        })),
        ..RunConfig::default()
    };
    Runner::run(&Agent::new("a").model(model.clone()), "new", options).await.expect("run");
    let contents: Vec<Value> = model.calls()[0]
        .input
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["content"].clone())
        .collect();
    assert_eq!(contents, vec![json!("old-3"), json!("note"), json!("new")]);
    let saved: Vec<Value> = session
        .get_items(None)
        .await
        .unwrap()
        .iter()
        .skip(3)
        .map(|i| i["content"].clone())
        .collect();
    assert_eq!(saved[..2], [json!("note"), json!("new")], "{saved:?}");
}

/// D-029: tool input guardrails can allow, reject (the body and hooks never run) or stop the run.
#[tokio::test]
async fn tool_input_guardrails_allow_reject_and_raise() {
    use openai_agents::{ToolGuardrailBehavior, ToolGuardrailFunctionOutput, ToolInputGuardrail};
    let body_runs = Arc::new(AtomicUsize::new(0));
    let build = |guardrail: ToolInputGuardrail| {
        let runs = Arc::clone(&body_runs);
        let tool = FunctionTool::new(
            "t",
            "t",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
            move |_ctx, _args| {
                let runs = Arc::clone(&runs);
                async move {
                    runs.fetch_add(1, Ordering::SeqCst);
                    Ok(json!("body"))
                }
            },
        )
        .with_tool_input_guardrails(vec![guardrail]);
        let model = Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call("t", "{}", "c1")),
            ModelStep::from(ItemHelpers::text_message("done")),
        ]));
        Agent::new("a").model(model).tools(vec![tool])
    };
    let guard = |out: fn() -> ToolGuardrailFunctionOutput| {
        ToolInputGuardrail::new("g", move |_data| async move { out() })
    };

    let result = Runner::run(
        &build(guard(|| ToolGuardrailFunctionOutput::allow(json!({"ok": true})))),
        "go",
        RunOptions::default(),
    )
    .await
    .expect("allow");
    assert_eq!(body_runs.load(Ordering::SeqCst), 1);
    assert_eq!(tool_output_texts(&result.new_items), vec!["body"]);
    assert_eq!(result.tool_input_guardrail_results.len(), 1);
    assert_eq!(result.tool_input_guardrail_results[0].guardrail_name, "g");
    assert_eq!(result.tool_input_guardrail_results[0].output.behavior, ToolGuardrailBehavior::Allow);

    body_runs.store(0, Ordering::SeqCst);
    let result = Runner::run(
        &build(guard(|| ToolGuardrailFunctionOutput::reject_content("blocked", json!(null)))),
        "go",
        RunOptions::default(),
    )
    .await
    .expect("reject");
    assert_eq!(body_runs.load(Ordering::SeqCst), 0, "the tool body must not run");
    assert_eq!(tool_output_texts(&result.new_items), vec!["blocked"]);
    assert!(result.tool_output_guardrail_results.is_empty(), "output guardrails are skipped");

    let err = Runner::run(
        &build(guard(|| ToolGuardrailFunctionOutput::raise_exception(json!(null)))),
        "go",
        RunOptions::default(),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, AgentsError::ToolInputGuardrailTripwire(ref t) if t.guardrail_name == "g"), "{err}");
    assert_eq!(body_runs.load(Ordering::SeqCst), 0);
}

/// D-029: tool output guardrails see the result and can replace it or stop the run.
#[tokio::test]
async fn tool_output_guardrails_replace_or_raise() {
    use openai_agents::{ToolGuardrailFunctionOutput, ToolOutputGuardrail};
    let build = |guardrail: ToolOutputGuardrail| {
        let model = Arc::new(ScriptedModel::new([
            ModelStep::from(ItemHelpers::function_tool_call("t", "{}", "c1")),
            ModelStep::from(ItemHelpers::text_message("done")),
        ]));
        Agent::new("a").model(model).tools(vec![
            FunctionTool::constant("t", "t", "secret-123").with_tool_output_guardrails(vec![guardrail]),
        ])
    };

    let redact = ToolOutputGuardrail::new("redact", |data| async move {
        if data.output.as_str().is_some_and(|s| s.contains("secret")) {
            ToolGuardrailFunctionOutput::reject_content("[redacted]", json!({"matched": true}))
        } else {
            ToolGuardrailFunctionOutput::allow(json!(null))
        }
    });
    let result = Runner::run(&build(redact), "go", RunOptions::default()).await.expect("run");
    assert_eq!(tool_output_texts(&result.new_items), vec!["[redacted]"]);
    assert_eq!(result.tool_output_guardrail_results[0].output.output_info, json!({"matched": true}));

    let stop = ToolOutputGuardrail::new("stop", |_| async {
        ToolGuardrailFunctionOutput::raise_exception(json!(null))
    });
    let err = Runner::run(&build(stop), "go", RunOptions::default()).await.unwrap_err();
    assert!(matches!(err, AgentsError::ToolOutputGuardrailTripwire(_)), "{err}");
}

/// D-026: `reasoning_item_id_policy = Omit` strips reasoning ids from the next turn's input and
/// from `to_input_list`, and only from reasoning items.
#[tokio::test]
async fn reasoning_item_id_policy_omit_strips_reasoning_ids() {
    use openai_agents::{ReasoningItemIdPolicy, RunConfig};
    let reasoning = json!({"id": "rs_1", "type": "reasoning", "summary": []});
    let run = |policy: Option<ReasoningItemIdPolicy>| {
        let model = Arc::new(ScriptedModel::new([
            ModelStep::output([reasoning.clone(), ItemHelpers::function_tool_call("echo", "{}", "c1")]),
            ModelStep::from(ItemHelpers::text_message("done")),
        ]));
        let agent = Agent::new("a")
            .model(model.clone())
            .tools(vec![FunctionTool::constant("echo", "e", "x")]);
        let mut options = RunOptions::default();
        options.run_config = RunConfig { reasoning_item_id_policy: policy, ..RunConfig::default() };
        async move {
            let result = Runner::run(&agent, "go", options).await.expect("run");
            (model.calls()[1].input.clone(), result.to_input_list())
        }
    };

    let (input, list) = run(Some(ReasoningItemIdPolicy::Omit)).await;
    let reasoning_of = |items: &Value| {
        items.as_array().unwrap().iter().find(|i| i["type"] == "reasoning").unwrap().clone()
    };
    assert!(reasoning_of(&input).get("id").is_none(), "{input}");
    assert!(reasoning_of(&json!(list)).get("id").is_none());
    assert!(input.as_array().unwrap().iter().any(|i| i["call_id"] == "c1" && i["type"] == "function_call"));
    assert!(input.as_array().unwrap().iter().filter(|i| i["type"] != "reasoning").all(|i| i.get("id").is_some() || i["type"] == "function_call_output" || i.get("role").is_some()));

    let (input, list) = run(None).await;
    assert_eq!(reasoning_of(&input)["id"], "rs_1");
    assert_eq!(reasoning_of(&json!(list))["id"], "rs_1");
}

/// D-026: an output guardrail that rejects a *tool-derived* final output withholds it; the error
/// carries a placeholder (default, fixed text or formatter) instead. Model text is untouched.
#[tokio::test]
async fn output_guardrail_blocked_message_replaces_tool_output() {
    use openai_agents::{
        GuardrailFunctionOutput, OutputGuardrail, OutputGuardrailBlockedMessage, RunConfig,
        ToolUseBehavior, OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT,
    };
    let trip = || {
        OutputGuardrail::new("g", |_ctx, _agent, _out| async {
            GuardrailFunctionOutput::trip(json!("details"))
        })
    };
    let tripped = |config: RunConfig, from_tool: bool| async move {
        let step = if from_tool {
            ModelStep::from(ItemHelpers::function_tool_call("secret", "{}", "c1"))
        } else {
            ModelStep::from(ItemHelpers::text_message("PLAIN-MODEL-TEXT"))
        };
        let agent = Agent::new("a")
            .model(Arc::new(ScriptedModel::new([step])))
            .tools(vec![FunctionTool::constant("secret", "s", "TOP-SECRET-DATA")])
            .tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
            .output_guardrails(vec![trip()]);
        let mut options = RunOptions::default();
        options.run_config = config;
        match Runner::run(&agent, "go", options).await {
            Err(AgentsError::OutputGuardrailTripwire(t)) => t.result,
            other => panic!("expected a tripwire, got {other:?}"),
        }
    };
    let with = |message| RunConfig { output_guardrail_blocked_message: message, ..RunConfig::default() };

    let r = tripped(RunConfig::default(), true).await;
    assert_eq!(r.agent_output, json!(OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT));
    assert_eq!(r.output.output_info, Value::Null, "the guardrail's own details are dropped too");

    let r = tripped(with(Some(OutputGuardrailBlockedMessage::Text("[blocked]".into()))), true).await;
    assert_eq!(r.agent_output, json!("[blocked]"));

    let r = tripped(
        with(Some(OutputGuardrailBlockedMessage::formatter(|args| {
            Some(format!("{} by {} on {}", args.default_message.len(), args.guardrail_name, args.agent.name))
        }))),
        true,
    )
    .await;
    assert_eq!(r.agent_output, json!("39 by g on a"));

    for bad in [
        OutputGuardrailBlockedMessage::formatter(|_| None),
        OutputGuardrailBlockedMessage::formatter(|_| Some(String::new())),
        OutputGuardrailBlockedMessage::formatter(|_| panic!("formatter bug")),
    ] {
        let r = tripped(with(Some(bad)), true).await;
        assert_eq!(r.agent_output, json!(OUTPUT_GUARDRAIL_BLOCKED_TOOL_OUTPUT));
    }

    let r = tripped(with(Some(OutputGuardrailBlockedMessage::Text("[blocked]".into()))), false).await;
    assert_eq!(r.agent_output, json!("PLAIN-MODEL-TEXT"), "model text is not withheld");
    assert_eq!(r.output.output_info, json!("details"));

    let mut options = RunOptions::default();
    options.run_config = with(Some(OutputGuardrailBlockedMessage::Text(String::new())));
    let agent = Agent::new("a").model(Arc::new(ScriptedModel::new([])));
    assert!(matches!(
        Runner::run(&agent, "go", options).await.unwrap_err(),
        AgentsError::User(_)
    ));
}

fn refusal_message(text: &str) -> Value {
    json!({"id": "1", "type": "message", "role": "assistant", "status": "completed",
           "content": [{"type": "refusal", "refusal": text}]})
}

/// D-033: a refusal raises `ModelRefusalError` (even for plain-text agents) unless a
/// `model_refusal` handler supplies the output; Python parity scenarios cover the same cases.
#[tokio::test]
async fn model_refusal_raises_or_is_handled() {
    use openai_agents::{RunErrorHandlerResult, RunErrorHandlers, RunHandledError};
    let build = || {
        Agent::new("a").model(Arc::new(ScriptedModel::new([ModelStep::from(refusal_message("no can do"))])))
    };
    let err = Runner::run(&build(), "hi", RunOptions::default()).await.unwrap_err();
    match err {
        AgentsError::ModelRefusal(e) => assert_eq!(e.refusal, "no can do"),
        other => panic!("expected a refusal error, got {other}"),
    }

    let mut options = RunOptions::default();
    options.error_handlers = RunErrorHandlers::default().on_model_refusal(|input| async move {
        assert!(matches!(input.error, RunHandledError::ModelRefusal(ref e) if e.refusal == "no can do"));
        assert_eq!(input.run_data.raw_responses.len(), 1);
        assert_eq!(input.run_data.new_items.len(), 1);
        Ok(Some(RunErrorHandlerResult::new("REFUSED")))
    });
    let result = Runner::run(&build(), "hi", options.clone()).await.expect("handled");
    assert_eq!(result.final_output_as_str(), Some("REFUSED"));
    assert_eq!(result.new_items.len(), 2, "refusal message plus the synthesized one");

    options.error_handlers = RunErrorHandlers::default().on_model_refusal(|_| async {
        Ok(Some(RunErrorHandlerResult { final_output: json!("R"), include_in_history: false }))
    });
    let result = Runner::run(&build(), "hi", options.clone()).await.expect("handled");
    assert_eq!(result.new_items.len(), 1);

    options.error_handlers = RunErrorHandlers::default().on_model_refusal(|_| async { Ok(None) });
    assert!(matches!(
        Runner::run(&build(), "hi", options).await.unwrap_err(),
        AgentsError::ModelRefusal(_)
    ));
}

/// D-033: `invalid_final_output` handles unparsable and empty structured answers. Without a
/// handler an unparsable answer raises and an empty one asks the model again, as in Python.
#[tokio::test]
async fn invalid_final_output_is_handled_or_retried() {
    use openai_agents::{AgentOutputSchema, RunErrorHandlerResult, RunErrorHandlers, RunHandledError};
    #[derive(serde::Deserialize, openai_agents::schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Out {
        a: i64,
    }
    let build = |steps: Vec<&str>| {
        Agent::new("a")
            .model(Arc::new(ScriptedModel::new(
                steps.into_iter().map(|t| ModelStep::from(ItemHelpers::text_message(t))),
            )))
            .output_type(Arc::new(AgentOutputSchema::of::<Out>().expect("schema")))
    };
    let handled = |value: Value| {
        let mut options = RunOptions::default();
        options.error_handlers = RunErrorHandlers::default().on_invalid_final_output(move |input| {
            let value = value.clone();
            async move {
                assert!(matches!(input.error, RunHandledError::InvalidFinalOutput(_)));
                assert_eq!(input.run_data.raw_responses.len(), 1);
                Ok(Some(RunErrorHandlerResult::new(value)))
            }
        });
        options
    };

    let err = Runner::run(&build(vec!["nope"]), "hi", RunOptions::default()).await.unwrap_err();
    assert!(matches!(err, AgentsError::Model(ModelError::Behavior(_))), "{err}");

    let result = Runner::run(&build(vec!["nope"]), "hi", handled(json!({"a": 7}))).await.expect("handled");
    assert_eq!(result.final_output, json!({"a": 7}));
    assert_eq!(result.new_items.len(), 2);

    let err = Runner::run(&build(vec!["nope"]), "hi", handled(json!({"a": "x"}))).await.unwrap_err();
    assert!(matches!(err, AgentsError::User(_)), "{err}");

    let result = Runner::run(&build(vec![""]), "hi", handled(json!({"a": 7}))).await.expect("handled");
    assert_eq!(result.final_output, json!({"a": 7}));

    let result = Runner::run(&build(vec!["", r#"{"a":1}"#]), "hi", RunOptions::default())
        .await
        .expect("second attempt");
    assert_eq!(result.final_output, json!({"a": 1}));
    assert_eq!(result.raw_responses.len(), 2);
    assert_eq!(result.new_items.len(), 2);
}

/// `Session::replace_items`: the in-memory store swaps its history in one step.
#[tokio::test]
async fn in_memory_session_replace_items() {
    use openai_agents::{InMemorySession, Session};
    let session = InMemorySession::shared("s");
    session.add_items(vec![json!({"role": "user", "content": "a"})]).await.unwrap();
    session.replace_items(vec![json!({"role": "user", "content": "b"})]).await.unwrap();
    assert_eq!(session.get_items(None).await.unwrap(), vec![json!({"role": "user", "content": "b"})]);
}

/// Sets a flag when dropped, to observe that a spawned future was torn down.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// A streamed run that is stuck waiting on a slow parallel input guardrail, and the flag that
/// shows whether the guardrail future was dropped.
async fn streamed_run_with_slow_guardrail() -> (openai_agents::RunResultStreaming, Arc<AtomicBool>) {
    use openai_agents::{GuardrailFunctionOutput, InputGuardrail};
    let dropped = Arc::new(AtomicBool::new(false));
    let flag = dropped.clone();
    let agent = Agent::new("a")
        .model(Arc::new(ScriptedModel::new([ModelStep::from(ItemHelpers::text_message("hi"))])))
        .input_guardrails(vec![InputGuardrail::new("slow", move |_c, _a, _i| {
            let guard = DropFlag(flag.clone());
            async move {
                let _guard = guard;
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                GuardrailFunctionOutput::pass(Value::Null)
            }
        })]);
    let streamed = Runner::run_streamed(agent, "go", RunOptions::default());
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    (streamed, dropped)
}

/// D-031: an immediate cancel also stops the run's child tasks, not just the run task.
#[tokio::test]
async fn immediate_cancel_aborts_parallel_input_guardrails() {
    use openai_agents::CancelMode;
    let (mut streamed, dropped) = streamed_run_with_slow_guardrail().await;
    assert!(!dropped.load(Ordering::SeqCst), "the guardrail is still running");
    streamed.cancel(CancelMode::Immediate);
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    assert!(dropped.load(Ordering::SeqCst), "the guardrail task must be aborted with the run");
}

/// D-031: `cancel_on_drop` turns a dropped stream into an immediate cancel; the default keeps
/// the run going like Python.
#[tokio::test]
async fn cancel_on_drop_is_opt_in() {
    let (streamed, dropped) = streamed_run_with_slow_guardrail().await;
    drop(streamed);
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    assert!(!dropped.load(Ordering::SeqCst), "default: the run outlives its handle");

    let (streamed, dropped) = streamed_run_with_slow_guardrail().await;
    drop(streamed.cancel_on_drop(true));
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    assert!(dropped.load(Ordering::SeqCst), "opt-in: dropping the handle stops the run");
}
