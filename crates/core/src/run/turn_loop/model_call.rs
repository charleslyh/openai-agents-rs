//! One model call of a turn: building the input, asking the model (with retries and streaming),
//! and recording the response.

use super::input_guardrails::{reject_parallel_transform, tripwire};
use super::*;

impl TurnLoop<'_> {
    /// Ask the model for this turn's response and record it.
    pub(super) async fn call_model(
        &mut self,
        setup: &TurnSetup,
        model: &dyn Model,
        turn_guard: &mut Option<SpanGuard>,
    ) -> Result<ModelResponse, AgentsError> {
        let _gen_span = generation_span(
            self.current_agent
                .model_name
                .as_deref()
                .unwrap_or("scripted"),
        );
        let (instructions, input_items) = self.model_input().await?;
        self.notify_llm_start(instructions.as_deref(), &input_items)
            .await;
        let (response, early) = self
            .request_response(model, setup, instructions.as_deref(), &input_items)
            .await?;
        if let Some(results) = early {
            self.guardrails.pending = None;
            self.input_guardrail_results.extend(results);
        }
        self.record_response(&response, turn_guard).await;
        Ok(response)
    }

    /// The instructions and input items for this call, after `call_model_input_filter` and any
    /// `ModelMiddleware`.
    pub(super) async fn model_input(
        &self,
    ) -> Result<(Option<String>, Vec<ResponseInputItem>), AgentsError> {
        // Python: `instructions` may be a callable of `(context, agent)`.
        let instructions = self.current_agent.resolve_instructions(&self.context).await;
        let data = ModelInputData {
            input: self.current_input_items.clone(),
            instructions,
        };
        // Python (`maybe_filter_model_input`): the filter runs before the LLM hooks and only
        // affects this call.
        let mut data = match &self.options.run_config.model.input_filter {
            Some(filter) => {
                filter(CallModelData {
                    model_data: data,
                    agent: Arc::clone(&self.current_agent),
                    context: self.context.clone(),
                })
                .await?
            }
            None => data,
        };
        // Rust-only extension on top of the Python-parity filter above (see
        // `crate::model_middleware`): each step runs in the order it was added.
        for middleware in &self.options.run_config.model.middleware {
            data = middleware
                .prepare(data, &self.current_agent, &self.context)
                .await?;
        }
        Ok((data.instructions, data.input))
    }

    pub(super) async fn notify_llm_start(
        &self,
        instructions: Option<&str>,
        input_items: &[ResponseInputItem],
    ) {
        if let Some(h) = &self.options.hooks {
            h.on_llm_start(
                self.context.clone(),
                &self.current_agent,
                instructions,
                input_items,
            )
            .await;
        }
        if let Some(ah) = &self.current_agent.hooks {
            ah.on_llm_start(
                self.context.clone(),
                &self.current_agent,
                instructions,
                input_items,
            )
            .await;
        }
    }

    /// Call the model with retries. Python cancels the in-flight call when a concurrent input
    /// guardrail trips, so a tripwire never waits for (or pays for) the full response; results of
    /// a guardrail that finished first are returned alongside the response.
    pub(super) async fn request_response(
        &mut self,
        model: &dyn Model,
        setup: &TurnSetup,
        instructions: Option<&str>,
        input_items: &[ResponseInputItem],
    ) -> Result<(ModelResponse, Option<Vec<InputGuardrailResult>>), AgentsError> {
        // Everything an attempt reads is a plain reference, so each attempt rebuilds its request
        // and a retry can start it again from scratch.
        let emitted_unsafe = Arc::new(AtomicBool::new(false));
        // Python maps `trace_include_sensitive_data=False` to
        // `ModelTracing.ENABLED_WITHOUT_DATA`, so adapters drop payloads.
        let tracing_mode = if crate::tracing::tracing_disabled() {
            ModelTracing::Disabled
        } else if self.options.run_config.tracing.include_sensitive_data {
            ModelTracing::Enabled
        } else {
            ModelTracing::EnabledWithoutData
        };
        let previous_ref = self.previous_response_id.as_deref();
        let conversation_ref = self.options.conversation_id.as_deref();
        let output_schema_ref = self.current_agent.output_type.as_deref();
        let events_ref = &self.events;
        let settings_ref = &setup.model_settings;
        let tools_ref = &setup.tools;
        let attempt_flag = Arc::clone(&emitted_unsafe);
        let model_fut = crate::retry::call_with_retry(
            crate::retry::RetryCall {
                settings: settings_ref.retry.as_ref(),
                previous_response_id: previous_ref,
                conversation_id: conversation_ref,
                timeout: settings_ref.timeout,
                stream: events_ref.is_some(),
                emitted_unsafe_event: events_ref.is_some().then_some(&*emitted_unsafe),
            },
            |request| model.get_retry_advice(request),
            move || {
                let attempt_flag = Arc::clone(&attempt_flag);
                async move {
                    let req = ModelRequest {
                        system_instructions: instructions,
                        input: ModelInput::Items(input_items),
                        model_settings: settings_ref,
                        tools: tools_ref,
                        tracing: tracing_mode,
                        previous_response_id: previous_ref,
                        conversation_id: conversation_ref,
                        output_schema: output_schema_ref,
                    };
                    attempt_model_call(model, req, events_ref, attempt_flag).await
                }
            },
        );
        let model_fut = async move { model_fut.await.map_err(AgentsError::from) };
        tokio::pin!(model_fut);
        let mut early_guardrails = None;
        let response = match self.guardrails.pending.as_mut() {
            Some(task) => tokio::select! {
                joined = &mut *task => {
                    let results = joined.map_err(|e| {
                        AgentsError::internal(format!("input guardrail task failed: {e}"))
                    })?;
                    tripwire(&results)?;
                    reject_parallel_transform(&results)?;
                    early_guardrails = Some(results);
                    (&mut model_fut).await?
                }
                response = &mut model_fut => response?,
            },
            None => (&mut model_fut).await?,
        };
        Ok((response, early_guardrails))
    }

    /// Account for a response: usage, the snapshot streaming consumers read, and `on_llm_end`.
    pub(super) async fn record_response(
        &mut self,
        response: &ModelResponse,
        turn_guard: &mut Option<SpanGuard>,
    ) {
        if let Some(id) = &response.response_id {
            self.previous_response_id = Some(id.clone());
        }
        self.usage.add(&response.usage);
        for guard in [self.task_guard.as_mut(), turn_guard.as_mut()]
            .into_iter()
            .flatten()
        {
            guard.add_usage(&response.usage);
        }
        self.context.add_usage(&response.usage);
        self.raw_responses.push(response.clone());
        if let Some(snap) = &self.snapshot {
            let mut s = snap
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            s.new_items = self.generated_items.clone();
            s.raw_responses = self.raw_responses.clone();
            s.usage = self.usage.clone();
        }
        if let Some(h) = &self.options.hooks {
            h.on_llm_end(self.context.clone(), &self.current_agent, response)
                .await;
        }
        if let Some(ah) = &self.current_agent.hooks {
            ah.on_llm_end(self.context.clone(), &self.current_agent, response)
                .await;
        }
    }
}

/// One attempt at the model call, streaming raw events to the consumer when there is one.
async fn attempt_model_call(
    model: &dyn Model,
    req: ModelRequest<'_>,
    events: &Option<EventTx>,
    attempt_flag: Arc<AtomicBool>,
) -> Result<ModelResponse, ModelError> {
    if events.is_none() {
        return model.get_response(req).await;
    }
    let (raw_tx, mut raw_rx) = mpsc::channel::<Value>(64);
    let forward = {
        let events = events.clone();
        AbortOnDrop(tokio::spawn(async move {
            while let Some(data) = raw_rx.recv().await {
                // Python: once an event other than `response.created` / `response.in_progress`
                // reached the consumer, a replay would show it output twice.
                let kind = data.get("type").and_then(Value::as_str);
                if !matches!(kind, Some("response.created" | "response.in_progress")) {
                    attempt_flag.store(true, Ordering::SeqCst);
                }
                emit(&events, StreamEvent::RawResponse { data }).await;
            }
        }))
    };
    let outcome = model.stream_response(req, raw_tx).await;
    let _ = forward.await;
    outcome
}
