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
    opts.run_config.model_provider = Some(Arc::new(ScriptedProvider));

    let result = Runner::run(&agent, "go", opts).await.expect("run");
    assert_eq!(result.final_output_as_str(), Some("from provider"));
}

/// `RunConfig.model` may be a name as well as an instance.
#[tokio::test]
async fn run_config_model_name_resolves_via_provider() {
    let agent = Agent::new("plain");
    let mut opts = RunOptions::default();
    opts.run_config.model_provider = Some(Arc::new(ScriptedProvider));
    opts.run_config.model = Some(ModelRef::Name("scripted-model".into()));

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
    opts.run_config.model_provider = Some(Arc::new(ScriptedProvider));
    opts.run_config.model = Some(ModelRef::Instance(model as Arc<dyn openai_agents::Model>));

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
