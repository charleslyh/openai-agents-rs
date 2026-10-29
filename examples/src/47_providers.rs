//! Several model providers behind one router (Rust-only)
//!
//! `opai_agents::providers` answers "which server serves this model name?". It holds a table of
//! presets (base URL, key variable, protocol, capabilities, the model-name prefixes each one
//! owns), and a `ProviderRegistry` that turns the ones you registered into a single
//! `ModelProvider` which routes by name:
//!
//! * `fast/anything` goes to the provider registered as `fast` (an explicit prefix);
//! * `deepseek-chat` carries no prefix, but the table knows it belongs to `deepseek`;
//! * a name that matches nothing falls to an error that lists what is registered.
//!
//! `ProviderRegistry::from_env()` registers every preset whose key variable is set, which is what
//! `opai_agents::run` does (see `zero_config`). This example builds the registry by hand so that
//! it needs no key: three scripted providers stand in for real servers, and one more entry is a
//! real HTTP provider for a local server that is built but never called.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example providers
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use opai_agents::providers::{infer_preset, Capabilities, Provider, ProviderRegistry, PRESETS};
use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{Agent, Model, ModelProvider, RunOptions, Runner, UserError};

/// A provider that always answers with one scripted reply, standing in for a real server.
struct Canned(&'static str);

#[async_trait]
impl ModelProvider for Canned {
    async fn get_model(&self, _name: Option<&str>) -> Result<Arc<dyn Model>, UserError> {
        Ok(Arc::new(ScriptedModel::new([ModelStep::from(
            ItemHelpers::text_message(self.0),
        )])))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- The table: what the SDK knows about third-party servers. ----
    println!("presets (name, protocol, key variable, capabilities):");
    for spec in PRESETS {
        println!(
            "  {:<11} {:?}  {}  {} capabilities",
            spec.name,
            spec.api,
            spec.key_env.unwrap_or("(no key)"),
            spec.capabilities.supported().count()
        );
    }
    for name in ["deepseek-chat", "gemini-2.0-flash", "my-private-model"] {
        println!(
            "  `{name}` belongs to: {}",
            infer_preset(name).map_or("nobody the table knows", |spec| spec.name)
        );
    }

    // ---- A registry built by hand. ----
    let registry = ProviderRegistry::new()
        .register(
            "fast",
            Arc::new(Canned("fast: quick answer")),
            Capabilities::full(),
        )
        .register(
            "deepseek",
            Arc::new(Canned("deepseek: found by the model name")),
            Capabilities::full(),
        )
        .register(
            "smart",
            Arc::new(Canned("smart: careful answer")),
            Capabilities::full(),
        )
        // A real provider for a local OpenAI-compatible server (Ollama, vLLM, ...). Building it
        // opens no connection; the first run that routes here does.
        .register(
            "local",
            Provider::custom("local", "http://127.0.0.1:11434/v1")?
                .api_key("not-needed")
                .build()?,
            Capabilities::full(),
        )
        .with_default("smart")?;
    println!(
        "\nregistered: {}",
        registry.names().collect::<Vec<_>>().join(", ")
    );
    let router = registry.into_provider()?;

    // ---- One agent, three routes. ----
    let agent = Agent::new("Assistant").instructions("Answer briefly.");
    for model in ["fast/any-model", "smart/any-model", "deepseek-chat"] {
        let options = RunOptions::new()
            .provider(Some(Arc::clone(&router)))
            .model(model);
        let result = Runner::run(&agent, "Hello!", options).await?;
        println!(
            "model {:<16} -> {}",
            model,
            result.final_output_as_str().unwrap_or_default()
        );
    }

    // An unknown prefix is an error that says what is registered, not a silent fallback.
    match router.get_model(Some("nope/some-model")).await {
        Ok(_) => println!("\nnope/some-model: resolved (unexpected)"),
        Err(error) => println!("\nnope/some-model: {error}"),
    }
    Ok(())
}
