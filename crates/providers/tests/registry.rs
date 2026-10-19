//! Routing (`prefix/model`, inference, default) and discovery from the environment.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_core::error::UserError;
use openai_agents_core::model::provider::ModelProvider;
use openai_agents_core::model::Model;
use openai_agents_providers::{Capabilities, Capability, ProviderRegistry, UnsupportedCapability};

/// Records the names it is asked for and answers with its own name, so a test can see which
/// provider a name was routed to.
struct RecordingProvider {
    name: String,
    asked: Mutex<Vec<String>>,
}

#[async_trait]
impl ModelProvider for RecordingProvider {
    async fn get_model(&self, model_name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        self.asked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(model_name.unwrap_or("<none>").to_string());
        Err(UserError::new(format!("routed to {}", self.name)))
    }
}

/// The recording providers answer with their own name, so a test can see where a name landed
/// without needing a `Model` (and without `Arc<dyn Model>` being `Debug`).
fn routed_to(result: Result<Arc<dyn Model>, UserError>) -> String {
    match result {
        Err(error) => error.to_string(),
        Ok(_) => panic!("a recording provider always answers with an error"),
    }
}

fn recording(name: &str) -> (Arc<RecordingProvider>, Capabilities) {
    (
        Arc::new(RecordingProvider {
            name: name.to_string(),
            asked: Mutex::new(Vec::new()),
        }),
        Capabilities::chat(),
    )
}

fn registry_with(prefixes: &[&str]) -> (ProviderRegistry, Vec<Arc<RecordingProvider>>) {
    let mut registry = ProviderRegistry::new();
    let mut providers = Vec::new();
    for prefix in prefixes {
        let (provider, caps) = recording(prefix);
        providers.push(Arc::clone(&provider));
        registry = registry.register(*prefix, provider, caps);
    }
    (registry, providers)
}

#[tokio::test]
async fn a_prefixed_name_goes_to_its_provider_without_the_prefix() {
    let (registry, providers) = registry_with(&["deepseek", "anthropic", "openai"]);
    let provider = registry
        .with_default("openai")
        .unwrap()
        .into_provider()
        .unwrap();
    let err = routed_to(provider.get_model(Some("deepseek/deepseek-chat")).await);
    assert!(err.to_string().contains("routed to deepseek"), "{err}");
    assert_eq!(asked(&providers[0]), vec!["deepseek-chat"]);
    assert!(providers[1].asked.lock().unwrap().is_empty());
}

#[tokio::test]
async fn an_unprefixed_brand_name_is_inferred() {
    let (registry, providers) = registry_with(&["anthropic", "openai"]);
    let provider = registry
        .with_default("openai")
        .unwrap()
        .into_provider()
        .unwrap();
    let err = routed_to(provider.get_model(Some("claude-sonnet-x")).await);
    assert!(err.to_string().contains("routed to anthropic"), "{err}");
    // The name reaches the provider unchanged: it is Anthropic's own model id.
    assert_eq!(asked(&providers[0]), vec!["claude-sonnet-x"]);
}

#[tokio::test]
async fn an_unprefixed_name_otherwise_goes_to_the_default() {
    let (registry, providers) = registry_with(&["openai", "anthropic"]);
    let provider = registry
        .with_default("openai")
        .unwrap()
        .into_provider()
        .unwrap();
    let err = routed_to(provider.get_model(Some("gpt-4o")).await);
    assert!(err.to_string().contains("routed to openai"), "{err}");
    assert_eq!(asked(&providers[0]), vec!["gpt-4o"]);
}

/// A local runtime is never inferred: `llama3` may be served by Ollama here and OpenRouter there.
#[tokio::test]
async fn a_name_no_brand_owns_is_not_inferred() {
    let (registry, providers) = registry_with(&["ollama", "openai"]);
    let provider = registry
        .with_default("openai")
        .unwrap()
        .into_provider()
        .unwrap();
    routed_to(provider.get_model(Some("llama3.1")).await);
    assert!(providers[0].asked.lock().unwrap().is_empty());
    assert_eq!(asked(&providers[1]), vec!["llama3.1"]);
}

#[tokio::test]
async fn an_unknown_prefix_says_what_is_registered() {
    let (registry, _) = registry_with(&["openai"]);
    let provider = registry.into_provider().unwrap();
    let err = routed_to(provider.get_model(Some("nope/x")).await);
    assert!(err.to_string().contains("openai"), "{err}");
    assert!(err.to_string().contains("nope"), "{err}");
}

#[tokio::test]
async fn with_a_name_of_its_own_the_default_provider_gets_none() {
    let (registry, providers) = registry_with(&["ollama"]);
    let provider = registry.into_provider().unwrap();
    routed_to(provider.get_model(None).await);
    assert_eq!(asked(&providers[0]), vec!["<none>"]);
}

#[test]
fn an_empty_registry_says_how_to_configure_one() {
    let Err(err) = ProviderRegistry::new().into_provider() else {
        panic!("an empty registry cannot build a provider")
    };
    let message = err.to_string();
    assert!(message.contains("DEEPSEEK_API_KEY"), "{message}");
    assert!(message.contains("AGENTS_PROVIDER_1_ENDPOINT"), "{message}");
}

#[test]
fn the_default_must_be_registered() {
    let (registry, _) = registry_with(&["openai"]);
    let Err(err) = registry.with_default("deepseek") else {
        panic!("deepseek is not registered")
    };
    assert!(err.to_string().contains("openai"), "{err}");
}

/// Everything that touches the environment runs in this one test: `cargo test` runs tests of a
/// binary in parallel threads, and the environment is process-wide.
#[tokio::test]
async fn discovery_from_the_environment() {
    // A preset whose key is set is registered; one whose key is not, is not.
    std::env::set_var("DEEPSEEK_API_KEY", "sk-test");
    std::env::remove_var("GROQ_API_KEY");
    let registry = ProviderRegistry::from_env().expect("from_env");
    assert!(registry.names().any(|n| n == "deepseek"));
    assert!(!registry.names().any(|n| n == "groq"));
    // A local runtime is only registered when you say where it is.
    assert!(!registry.names().any(|n| n == "ollama"));
    std::env::set_var("OLLAMA_HOST", "http://localhost:11434");
    let registry = ProviderRegistry::from_env().expect("from_env");
    assert!(registry.names().any(|n| n == "ollama"));
    std::env::remove_var("OLLAMA_HOST");
    assert!(
        registry.names().any(|n| n == "ollama"),
        "Ollama needs no key"
    );

    let entry = registry.get("deepseek").expect("registered");
    assert!(entry.capabilities.supports(Capability::Reasoning));
    assert!(!entry.capabilities.supports(Capability::Vision));

    // A declared endpoint, with its own name and default model.
    std::env::set_var("AGENTS_PROVIDER_1_ENDPOINT", "https://acme.example.com/v1");
    std::env::set_var("AGENTS_PROVIDER_1_NAME", "acme");
    std::env::set_var("AGENTS_PROVIDER_1_API_KEY", "sk-acme");
    std::env::set_var("AGENTS_PROVIDER_1_DEFAULT_MODEL", "acme-large");
    let registry = ProviderRegistry::from_env().expect("from_env");
    assert!(registry.names().any(|n| n == "acme"));

    // An endpoint that is neither https nor loopback is refused unless explicitly allowed.
    std::env::set_var("AGENTS_PROVIDER_1_ENDPOINT", "http://acme.example.com/v1");
    let Err(err) = ProviderRegistry::from_env() else {
        panic!("plain http to a remote host must be refused")
    };
    assert!(err.to_string().contains("https"), "{err}");
    std::env::set_var("AGENTS_PROVIDER_INSECURE_HTTP", "1");
    let registry = ProviderRegistry::from_env().expect("allowed explicitly");
    assert!(registry.names().any(|n| n == "acme"));

    // Offline wins over a key that is set: nothing may reach the network.
    std::env::set_var("AGENTS_OFFLINE", "1");
    let registry = ProviderRegistry::from_env().expect("from_env");
    assert_eq!(registry.names().collect::<Vec<_>>(), vec!["offline"]);
    assert_eq!(registry.default(), Some("offline"));
    let provider = registry.into_provider().expect("a provider");
    let model = provider
        .get_model(Some("gpt-4o"))
        .await
        .expect("an offline model");
    let answer = openai_agents_providers::OfflineModel::answer();
    assert!(answer.contains("offline"), "{answer}");
    std::env::remove_var("AGENTS_OFFLINE");
    drop(model);

    for var in [
        "DEEPSEEK_API_KEY",
        "AGENTS_PROVIDER_1_ENDPOINT",
        "AGENTS_PROVIDER_1_NAME",
        "AGENTS_PROVIDER_1_API_KEY",
        "AGENTS_PROVIDER_1_DEFAULT_MODEL",
        "AGENTS_PROVIDER_INSECURE_HTTP",
    ] {
        std::env::remove_var(var);
    }
}

#[test]
fn an_unsupported_capability_names_the_provider_and_the_capability() {
    let error = UnsupportedCapability::new("groq", Capability::Vision)
        .with_hint("use a provider with vision, or send the image as text");
    let message = error.to_string();
    assert!(message.contains("groq"), "{message}");
    assert!(message.contains("vision"), "{message}");
    assert!(
        message.contains("hint") || message.contains("use a provider"),
        "{message}"
    );
    let user: UserError = error.into();
    assert!(user.to_string().contains("does not support"));
}

fn asked(provider: &RecordingProvider) -> Vec<String> {
    provider
        .asked
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}
