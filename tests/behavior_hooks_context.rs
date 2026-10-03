//! Context, dynamic instructions, hooks and guardrails.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    input_guardrail, output_guardrail, Agent, AgentHooks, AgentsError, FunctionTool,
    GuardrailFunctionOutput, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered,
    RunContextWrapper, RunHooks, RunOptions, Runner,
};

#[derive(Debug, PartialEq)]
struct Tenant {
    id: String,
}

/// Context reaches tools through `ToolContext::run_context`.
#[tokio::test]
async fn tool_receives_user_context() {
    let seen = Arc::new(Mutex::new(None));
    let seen_cb = Arc::clone(&seen);
    let tool = FunctionTool::new(
        "who",
        "report the tenant",
        serde_json::json!({"type": "object", "properties": {}, "additionalProperties": false}),
        move |ctx, _args| {
            let seen_cb = Arc::clone(&seen_cb);
            async move {
                *seen_cb.lock().unwrap() = ctx.context::<Tenant>().map(|t| t.id.clone());
                Ok(serde_json::json!("ok"))
            }
        },
    );

    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("who", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("ctx").model(model).tools(vec![tool]);

    let mut opts = RunOptions::default();
    opts.context = Some(Arc::new(Tenant { id: "acme".into() }));
    let _ = Runner::run(&agent, "go", opts).await.expect("run");
    assert_eq!(*seen.lock().unwrap(), Some("acme".to_string()));
}

/// `try_context` reports a precise error instead of silently returning `None`.
#[tokio::test]
async fn missing_context_reports_a_typed_error() {
    let ctx = RunContextWrapper::new(None);
    assert!(ctx.context::<Tenant>().is_none());
    let err = ctx.try_context::<Tenant>().expect_err("must fail");
    assert!(err.to_string().contains("no run context"), "{err}");

    let ctx = RunContextWrapper::with(7u32);
    let err = ctx.try_context::<Tenant>().expect_err("must fail");
    assert!(err.to_string().contains("not of type"), "{err}");
}

/// Dynamic instructions receive the context and the agent (Python: callable `instructions`).
#[tokio::test]
async fn dynamic_instructions_see_context() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("dynamic")
        .model(model.clone())
        .dynamic_instructions(|ctx, agent| {
            let tenant = ctx
                .context::<Tenant>()
                .map(|t| t.id.clone())
                .unwrap_or_else(|| "anon".into());
            async move { format!("tenant={tenant} agent={}", agent.name) }
        });

    let mut opts = RunOptions::default();
    opts.context = Some(Arc::new(Tenant { id: "acme".into() }));
    let _ = Runner::run(&agent, "go", opts).await.expect("run");

    let calls = model.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].system_instructions.as_deref(),
        Some("tenant=acme agent=dynamic")
    );
}

#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl RunHooks for Recorder {
    async fn on_agent_start(&self, _context: RunContextWrapper, agent: &Agent) {
        self.events
            .lock()
            .unwrap()
            .push(format!("agent_start:{}", agent.name));
    }
    async fn on_agent_end(&self, _context: RunContextWrapper, agent: &Agent, output: &serde_json::Value) {
        self.events
            .lock()
            .unwrap()
            .push(format!("agent_end:{}:{output}", agent.name));
    }
    async fn on_llm_start(&self, _context: RunContextWrapper, agent: &Agent, _p: Option<&str>, _i: &[serde_json::Value]) {
        self.events
            .lock()
            .unwrap()
            .push(format!("llm_start:{}", agent.name));
    }
    async fn on_llm_end(&self, _context: RunContextWrapper, agent: &Agent, _r: &openai_agents::ModelResponse) {
        self.events
            .lock()
            .unwrap()
            .push(format!("llm_end:{}", agent.name));
    }
    async fn on_tool_start(&self, _context: openai_agents::ToolContext, _agent: &Agent, tool: &FunctionTool) {
        self.events
            .lock()
            .unwrap()
            .push(format!("tool_start:{}", tool.name));
    }
    async fn on_tool_end(
        &self,
        _context: openai_agents::ToolContext,
        _agent: &Agent,
        tool: &FunctionTool,
        _result: &serde_json::Value,
    ) {
        self.events
            .lock()
            .unwrap()
            .push(format!("tool_end:{}", tool.name));
    }
    async fn on_handoff(&self, _context: RunContextWrapper, from: &Agent, to: &Agent) {
        self.events
            .lock()
            .unwrap()
            .push(format!("handoff:{}->{}", from.name, to.name));
    }
}

/// Hooks fire in the documented order across a tool run.
#[tokio::test]
async fn run_hooks_fire_in_order() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent = Agent::new("hooked")
        .model(model)
        .tools(vec![FunctionTool::constant("echo", "echo", "v")]);

    let recorder: Arc<Recorder> = Arc::new(Recorder::default());
    let mut opts = RunOptions::default();
    opts.hooks = Some(Arc::clone(&recorder) as Arc<dyn RunHooks>);
    let _ = Runner::run(&agent, "go", opts).await.expect("run");

    let events = recorder.events.lock().unwrap().clone();
    assert_eq!(
        events,
        vec![
            "agent_start:hooked".to_string(),
            "llm_start:hooked".to_string(),
            "llm_end:hooked".to_string(),
            "tool_start:echo".to_string(),
            "tool_end:echo".to_string(),
            "llm_start:hooked".to_string(),
            "llm_end:hooked".to_string(),
            "agent_end:hooked:\"done\"".to_string(),
        ]
    );
}

/// Agent-scoped hooks fire alongside run-scoped ones.
#[tokio::test]
async fn agent_hooks_fire() {
    struct AgentRecorder(Arc<AtomicUsize>);

    #[async_trait::async_trait]
    impl AgentHooks for AgentRecorder {
        async fn on_start(&self, _context: RunContextWrapper, _agent: &Agent) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        async fn on_end(&self, _context: RunContextWrapper, _agent: &Agent, _output: &serde_json::Value) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let counter = Arc::new(AtomicUsize::new(0));
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("agent-hooks")
        .model(model)
        .hooks(Arc::new(AgentRecorder(Arc::clone(&counter))));
    let _ = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(counter.load(Ordering::SeqCst), 2);
}

/// A tripping input guardrail aborts before the run completes.
#[tokio::test]
async fn input_guardrail_tripwire_aborts() {
    let guardrail = input_guardrail("block", |_ctx, _agent, _input| async {
        GuardrailFunctionOutput::trip(serde_json::json!({"reason": "off topic"}))
    });
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("guarded")
        .model(model)
        .input_guardrails(vec![guardrail]);
    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect_err("tripwire must abort");
    match err {
        AgentsError::InputGuardrailTripwire(InputGuardrailTripwireTriggered { result }) => {
            assert_eq!(result.guardrail_name, "block");
            assert!(result.output.tripwire_triggered);
        }
        other => panic!("unexpected error: {other}"),
    }
}

/// A passing guardrail is recorded on the result.
#[tokio::test]
async fn passing_guardrails_are_recorded() {
    let input = input_guardrail("ok-input", |_ctx, _agent, _input| async {
        GuardrailFunctionOutput::pass(serde_json::json!({}))
    });
    let output = output_guardrail("ok-output", |_ctx, _agent, _out| async {
        GuardrailFunctionOutput::pass(serde_json::json!({}))
    });
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("guarded")
        .model(model)
        .input_guardrails(vec![input])
        .output_guardrails(vec![output]);
    let result = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect("run");
    assert_eq!(result.input_guardrail_results.len(), 1);
    assert_eq!(result.output_guardrail_results.len(), 1);
    assert_eq!(result.output_guardrail_results[0].agent_name, "guarded");
}

/// A tripping output guardrail aborts after the final output is produced.
#[tokio::test]
async fn output_guardrail_tripwire_aborts() {
    let guardrail = output_guardrail("no-secrets", |_ctx, _agent, _out| async {
        GuardrailFunctionOutput::trip(serde_json::json!({"reason": "leak"}))
    });
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("guarded")
        .model(model)
        .output_guardrails(vec![guardrail]);
    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect_err("tripwire must abort");
    match err {
        AgentsError::OutputGuardrailTripwire(OutputGuardrailTripwireTriggered { result }) => {
            assert_eq!(result.guardrail_name, "no-secrets");
        }
        other => panic!("unexpected error: {other}"),
    }
}

/// Guardrails declared on `RunConfig` apply even when the agent has none.
#[tokio::test]
async fn run_config_guardrails_apply() {
    let guardrail = input_guardrail("cfg-block", |_ctx, _agent, _input| async {
        GuardrailFunctionOutput::trip(serde_json::json!({}))
    });
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("ok"),
    )]));
    let agent = Agent::new("plain").model(model);
    let mut opts = RunOptions::default();
    opts.run_config.input_guardrails = vec![guardrail];
    let err = Runner::run(&agent, "go", opts)
        .await
        .expect_err("tripwire must abort");
    assert!(matches!(
        err,
        AgentsError::InputGuardrailTripwire(InputGuardrailTripwireTriggered {
            result: openai_agents::InputGuardrailResult {
                guardrail_name,
                ..
            }
        }) if guardrail_name == "cfg-block"
    ));
}
