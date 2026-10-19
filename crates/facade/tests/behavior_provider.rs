//! Model providers and string model-name resolution.

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    Agent, AgentsError, ModelProvider, ModelRef, MultiProvider, RunOptions, Runner, UserError,
};

/// A test provider that always returns a scripted model.
struct ScriptedProvider;

impl ModelProvider for ScriptedProvider {
    fn get_model(
        &self,
        model_name: Option<&str>,
    ) -> Result<Arc<dyn openai_agents::Model>, UserError> {
        let _ = model_name;
        Ok(Arc::new(ScriptedModel::new([ModelStep::from(
            ItemHelpers::text_message("from provider"),
        )])))
    }
}

/// `Agent.model_name` resolves through `RunConfig.model_provider`.
#[tokio::test]
async fn agent_model_name_resolves_via_provider() {
    let agent = Agent::new("named").model_name("scripted-model");
    let mut opts = RunOptions::default();
    opts.run_config.model.provider = Some(Arc::new(ScriptedProvider));

    let result = Runner::run(&agent, "go", opts).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("from provider"));
}

/// `RunConfig.model` may be a name as well as an instance.
#[tokio::test]
async fn run_config_model_name_resolves_via_provider() {
    let agent = Agent::new("plain");
    let mut opts = RunOptions::default();
    opts.run_config.model.provider = Some(Arc::new(ScriptedProvider));
    opts.run_config.model.model_ref = Some(ModelRef::Name("scripted-model".into()));

    let result = Runner::run(&agent, "go", opts).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("from provider"));
}

/// `RunConfig.model` may also be a ready instance.
#[tokio::test]
async fn run_config_model_instance_wins() {
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("instance"),
    )]));
    let agent = Agent::new("plain");
    let mut opts = RunOptions::default();
    opts.run_config.model.provider = Some(Arc::new(ScriptedProvider));
    opts.run_config.model.model_ref =
        Some(ModelRef::Instance(model as Arc<dyn openai_agents::Model>));

    let result = Runner::run(&agent, "go", opts).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("instance"));
}

/// Without any model the error names the ways to fix it.
#[tokio::test]
async fn unresolved_model_reports_actionable_error() {
    let agent = Agent::new("no-model");
    let err = Runner::run(&agent, "go", RunOptions::default())
        .await
        .expect_err("must fail");
    assert!(matches!(err, AgentsError::User(_)), "{err}");
    assert!(err.to_string().contains("model_name"), "{err}");
}

/// MultiProvider routes `prefix/model` and defaults unprefixed names to `openai`.
#[test]
fn multi_provider_splits_names() {
    let router = MultiProvider::new().register("test", Arc::new(ScriptedProvider));
    assert_eq!(router.split("test/thing"), ("test", "thing"));
    assert_eq!(router.split("thing"), ("openai", "thing"));
    assert_eq!(router.split("a/b/c"), ("a", "b/c"));
}

#[tokio::test]
async fn multi_provider_routes_registered_prefix() {
    let router = MultiProvider::new().register("test", Arc::new(ScriptedProvider));
    assert!(router.get_model(Some("test/x")).is_ok());
    let err = router
        .get_model(Some("unknown/x"))
        .err()
        .expect("unregistered prefix must fail");
    assert!(err.to_string().contains("Unknown model provider"), "{err}");
}

/// A provider with no configured model name must say so.
#[test]
fn provider_without_name_is_an_error() {
    let router = MultiProvider::new().register("test", Arc::new(ScriptedProvider));
    let err = router.get_model(None).err().expect("must fail");
    assert!(err.to_string().contains("requires a model name"), "{err}");
}

/// The default provider reports a clear message when it cannot resolve names.
#[test]
fn missing_provider_is_actionable() {
    let provider = openai_agents::MissingProvider;
    let err = provider
        .get_model(Some("openai/gpt-x"))
        .err()
        .expect("must fail");
    assert!(err.to_string().contains("No ModelProvider"), "{err}");
}

/// Two `Runtime` instances keep their default providers independent (Rust-only, D-047 follow-up:
/// the process-wide `Runtime::global()` still backs `set_default_model_provider_factory`).
#[test]
fn separate_runtimes_do_not_share_a_default_provider() {
    use openai_agents::Runtime;

    let a = Runtime::new();
    let b = Runtime::new();
    assert!(a.set_model_provider(Arc::new(ScriptedProvider)));
    assert!(a.provider().get_model(Some("anything")).is_ok());
    let err = b
        .provider()
        .get_model(Some("anything"))
        .err()
        .expect("b has no provider of its own");
    assert!(err.to_string().contains("No ModelProvider"), "{err}");
}

/// Remembers the names it was asked for and hands out one shared scripted model.
struct RecordingProvider {
    names: std::sync::Mutex<Vec<String>>,
    model: Arc<ScriptedModel>,
}

impl ModelProvider for RecordingProvider {
    fn get_model(
        &self,
        model_name: Option<&str>,
    ) -> Result<Arc<dyn openai_agents::Model>, UserError> {
        self.names
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(model_name.unwrap_or_default().to_string());
        Ok(self.model.clone())
    }
}

fn recording_provider() -> Arc<RecordingProvider> {
    Arc::new(RecordingProvider {
        names: std::sync::Mutex::new(Vec::new()),
        model: Arc::new(ScriptedModel::new([ModelStep::from(
            ItemHelpers::text_message("ok"),
        )])),
    })
}

/// Rust-only (X-006): `name:<effort>` resolves `name` and sets the reasoning effort.
#[tokio::test]
async fn model_name_effort_suffix_is_stripped_and_applied() {
    use openai_agents::ReasoningEffort;

    let provider = recording_provider();
    let agent = Agent::new("thinker").model_name("o4-mini:high");
    let mut opts = RunOptions::default();
    opts.run_config.model.provider = Some(provider.clone());

    Runner::run(&agent, "go", opts).await.expect("run");
    assert_eq!(*provider.names.lock().unwrap(), vec!["o4-mini".to_string()]);
    let settings = provider.model.last_call().expect("called").model_settings;
    assert_eq!(
        settings.reasoning.and_then(|r| r.effort),
        Some(ReasoningEffort::High)
    );
}

/// An explicit effort wins over the suffix; a tag that is not an effort is left alone.
#[tokio::test]
async fn explicit_effort_wins_and_other_tags_are_untouched() {
    use openai_agents::ReasoningEffort;

    let provider = recording_provider();
    let agent = Agent::new("thinker").model_name("o4-mini:high");
    let mut opts = RunOptions::default().reasoning(ReasoningEffort::Low);
    opts.run_config.model.provider = Some(provider.clone());
    Runner::run(&agent, "go", opts).await.expect("run");
    let settings = provider.model.last_call().expect("called").model_settings;
    assert_eq!(
        settings.reasoning.and_then(|r| r.effort),
        Some(ReasoningEffort::Low)
    );

    let provider = recording_provider();
    let agent = Agent::new("local").model_name("llama3:8b");
    let mut opts = RunOptions::default();
    opts.run_config.model.provider = Some(provider.clone());
    Runner::run(&agent, "go", opts).await.expect("run");
    assert_eq!(
        *provider.names.lock().unwrap(),
        vec!["llama3:8b".to_string()]
    );
    let settings = provider.model.last_call().expect("called").model_settings;
    assert!(settings.reasoning.is_none());
}
