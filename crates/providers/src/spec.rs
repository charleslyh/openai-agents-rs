//! The preset table and the capability matrix.
//!
//! Every preset is static data: where the server is, which environment variable holds its key,
//! which protocol it speaks, and what it supports. Nothing here reads the environment or opens a
//! socket; [`super::Provider`] turns a spec into a provider and [`super::ProviderRegistry`] reads
//! the environment.
//!
//! The table goes stale the way any list of third-party servers does, so each entry carries the
//! date it was last checked (`LAST_CHECKED`) rather than pretending to be authoritative.

/// When the preset table was last checked against each provider's documentation.
pub const LAST_CHECKED: &str = "2026-10";

/// A capability a provider may or may not have (Rust-only, X-013).
///
/// Named so a request that needs one can say which, in [`super::UnsupportedCapability`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Capability {
    /// Stream the response token by token.
    Streaming,
    /// Emit `stream_options: {include_usage: true}` and get usage in a stream.
    StreamUsage,
    /// Call function tools.
    ToolCalls,
    /// Run several tool calls of one turn at once.
    ParallelToolCalls,
    /// Constrain the output to a JSON schema.
    StructuredOutput,
    /// Accept images.
    Vision,
    /// Accept PDF documents.
    PdfInput,
    /// Speak the Responses protocol (as opposed to Chat Completions only).
    ResponsesApi,
    /// Honour a reasoning effort / summary.
    Reasoning,
    /// Store the response server-side (`store: true` on Responses).
    Store,
}

impl Capability {
    /// Lower-case name, as it appears in an error.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Streaming => "streaming",
            Self::StreamUsage => "stream_usage",
            Self::ToolCalls => "tool_calls",
            Self::ParallelToolCalls => "parallel_tool_calls",
            Self::StructuredOutput => "structured_output",
            Self::Vision => "vision",
            Self::PdfInput => "pdf_input",
            Self::ResponsesApi => "responses_api",
            Self::Reasoning => "reasoning",
            Self::Store => "store",
        }
    }

    /// Every capability, for listing a matrix.
    pub fn all() -> &'static [Capability] {
        &[
            Self::Streaming,
            Self::StreamUsage,
            Self::ToolCalls,
            Self::ParallelToolCalls,
            Self::StructuredOutput,
            Self::Vision,
            Self::PdfInput,
            Self::ResponsesApi,
            Self::Reasoning,
            Self::Store,
        ]
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a provider supports (Rust-only, X-013).
///
/// Built with [`Self::chat`] (a plain Chat Completions server) or [`Self::full`], then narrowed.
/// It tells the adapters which request fields to send, so an unsupported combination fails with a
/// typed error instead of a field silently dropped by the server.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Capabilities {
    /// Bit set over [`Capability`], so adding one does not change the struct's size.
    bits: u32,
}

const fn bit(capability: Capability) -> u32 {
    1 << capability as u32
}

impl Capabilities {
    /// Nothing supported.
    pub const fn none() -> Self {
        Self { bits: 0 }
    }

    /// What nearly every Chat Completions server does: stream, call tools and answer in JSON.
    pub const fn chat() -> Self {
        Self::none()
            .with(Capability::Streaming)
            .with(Capability::ToolCalls)
            .with(Capability::ParallelToolCalls)
            .with(Capability::StructuredOutput)
    }

    /// Everything, for a first-party server that implements the whole protocol.
    pub const fn full() -> Self {
        Self::chat()
            .with(Capability::StreamUsage)
            .with(Capability::Vision)
            .with(Capability::PdfInput)
            .with(Capability::ResponsesApi)
            .with(Capability::Reasoning)
            .with(Capability::Store)
    }

    /// Add one capability.
    #[must_use]
    pub const fn with(mut self, capability: Capability) -> Self {
        self.bits |= bit(capability);
        self
    }

    /// Remove one capability.
    #[must_use]
    pub const fn without(mut self, capability: Capability) -> Self {
        self.bits &= !bit(capability);
        self
    }

    /// Whether `capability` is supported.
    pub const fn supports(&self, capability: Capability) -> bool {
        self.bits & bit(capability) != 0
    }

    /// The capabilities this set has, for a matrix listing.
    pub fn supported(&self) -> impl Iterator<Item = Capability> + '_ {
        Capability::all()
            .iter()
            .copied()
            .filter(|c| self.supports(*c))
    }

    /// The capabilities this set lacks, for a matrix listing.
    pub fn unsupported(&self) -> impl Iterator<Item = Capability> + '_ {
        Capability::all()
            .iter()
            .copied()
            .filter(|c| !self.supports(*c))
    }
}

/// One entry of the preset table.
///
/// Static data only: the protocol is always OpenAI-compatible, so a preset is a base URL, the
/// environment variable holding its key, the protocol flavour, what it supports, and which model
/// names belong to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProviderSpec {
    /// The name the provider is registered under (`deepseek/model`).
    pub name: &'static str,
    /// Base URL of the OpenAI-compatible endpoint.
    pub base_url: &'static str,
    /// Environment variable holding the API key; `None` for servers that need none.
    pub key_env: Option<&'static str>,
    /// The protocol to speak.
    pub api: ProviderApi,
    /// What the provider supports.
    pub capabilities: Capabilities,
    /// Lower-case prefixes that identify its models, for name inference.
    pub model_prefixes: &'static [&'static str],
    /// The model to use when none is named, if the provider has an obvious one.
    pub default_model: Option<&'static str>,
}

/// Which OpenAI protocol flavour a preset speaks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ProviderApi {
    /// Chat Completions, which every compatible server implements.
    #[default]
    ChatCompletions,
    /// The Responses protocol.
    Responses,
}

/// Every preset, in the order they are listed in documentation and errors.
pub const PRESETS: &[ProviderSpec] = &[
    ProviderSpec {
        name: "openai",
        base_url: "https://api.openai.com/v1",
        key_env: Some("OPENAI_API_KEY"),
        api: ProviderApi::Responses,
        capabilities: Capabilities::full(),
        model_prefixes: &["gpt-", "o1", "o3", "o4", "computer-use"],
        default_model: None,
    },
    ProviderSpec {
        name: "deepseek",
        base_url: "https://api.deepseek.com/v1",
        key_env: Some("DEEPSEEK_API_KEY"),
        api: ProviderApi::ChatCompletions,
        // `deepseek-reasoner` reports reasoning; no vision, no Responses endpoint.
        capabilities: Capabilities::chat().with(Capability::Reasoning),
        model_prefixes: &["deepseek"],
        default_model: Some("deepseek-chat"),
    },
    ProviderSpec {
        name: "gemini",
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
        key_env: Some("GEMINI_API_KEY"),
        api: ProviderApi::ChatCompletions,
        capabilities: Capabilities::chat()
            .with(Capability::Vision)
            .with(Capability::PdfInput)
            .with(Capability::Reasoning),
        model_prefixes: &["gemini", "gemma"],
        default_model: Some("gemini-2.5-flash"),
    },
    ProviderSpec {
        name: "groq",
        base_url: "https://api.groq.com/openai/v1",
        key_env: Some("GROQ_API_KEY"),
        api: ProviderApi::ChatCompletions,
        capabilities: Capabilities::chat().with(Capability::Reasoning),
        model_prefixes: &["groq/"],
        default_model: None,
    },
    ProviderSpec {
        name: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: Some("OPENROUTER_API_KEY"),
        api: ProviderApi::ChatCompletions,
        // Whatever the routed-to model supports; OpenRouter passes the fields through.
        capabilities: Capabilities::full(),
        model_prefixes: &[],
        default_model: None,
    },
    ProviderSpec {
        name: "anthropic",
        base_url: "https://api.anthropic.com/v1",
        key_env: Some("ANTHROPIC_API_KEY"),
        api: ProviderApi::ChatCompletions,
        capabilities: Capabilities::chat()
            .with(Capability::Vision)
            .with(Capability::PdfInput)
            .with(Capability::Reasoning),
        model_prefixes: &["claude"],
        default_model: None,
    },
    ProviderSpec {
        name: "ollama",
        base_url: "http://localhost:11434/v1",
        key_env: None,
        api: ProviderApi::ChatCompletions,
        // Depends on the model pulled locally, so only the protocol basics are claimed.
        capabilities: Capabilities::chat(),
        model_prefixes: &[],
        default_model: None,
    },
];

/// The preset registered under `name`, if there is one.
pub fn preset(name: &str) -> Option<&'static ProviderSpec> {
    PRESETS.iter().find(|p| p.name == name)
}

/// Which preset owns `model_name`, by its model prefixes (Rust-only, X-013).
///
/// Only brands that name their own models are inferred: `claude-sonnet-x` is Anthropic's,
/// `deepseek-chat` is DeepSeek's. Names like `llama3` or `qwen2.5` are not, because they are
/// served by local runtimes, OpenRouter and gateways alike — those need the `prefix/model` form.
pub fn infer_preset(model_name: &str) -> Option<&'static ProviderSpec> {
    let lower = model_name.to_ascii_lowercase();
    PRESETS.iter().find(|p| {
        p.model_prefixes
            .iter()
            .any(|prefix| lower.starts_with(prefix))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_name_is_unique() {
        let mut names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "duplicate preset name");
    }

    #[test]
    fn chat_and_full_differ_in_the_extras() {
        let chat = Capabilities::chat();
        assert!(chat.supports(Capability::ToolCalls));
        assert!(chat.supports(Capability::StructuredOutput));
        assert!(!chat.supports(Capability::Vision));
        assert!(!chat.supports(Capability::ResponsesApi));

        let full = Capabilities::full();
        for capability in Capability::all() {
            assert!(full.supports(*capability), "full must support {capability}");
        }
        assert!(full.unsupported().next().is_none());
        assert_eq!(
            chat.unsupported().collect::<Vec<_>>(),
            vec![
                Capability::StreamUsage,
                Capability::Vision,
                Capability::PdfInput,
                Capability::ResponsesApi,
                Capability::Reasoning,
                Capability::Store,
            ]
        );
    }

    #[test]
    fn capabilities_round_trip_through_bits() {
        let caps = Capabilities::none()
            .with(Capability::Vision)
            .with(Capability::PdfInput)
            .without(Capability::Vision);
        assert!(!caps.supports(Capability::Vision));
        assert!(caps.supports(Capability::PdfInput));
        assert_eq!(
            caps.supported().collect::<Vec<_>>(),
            vec![Capability::PdfInput]
        );
    }

    #[test]
    fn only_brands_that_name_their_models_are_inferred() {
        assert_eq!(
            infer_preset("claude-sonnet-x").map(|p| p.name),
            Some("anthropic")
        );
        assert_eq!(
            infer_preset("deepseek-chat").map(|p| p.name),
            Some("deepseek")
        );
        assert_eq!(
            infer_preset("gemini-2.5-pro").map(|p| p.name),
            Some("gemini")
        );
        assert_eq!(
            infer_preset("DEEPSEEK-chat").map(|p| p.name),
            Some("deepseek")
        );
        // Served by local runtimes, OpenRouter and gateways alike: not ours to guess.
        assert!(infer_preset("llama3.1").is_none());
        assert!(infer_preset("qwen2.5-7b").is_none());
        assert!(infer_preset("mistral-large").is_none());
    }

    #[test]
    fn every_preset_prefix_is_used_by_its_own_models() {
        for spec in PRESETS {
            for prefix in spec.model_prefixes {
                // A prefix must not belong to two presets, or inference picks by table order.
                let owners: Vec<&str> = PRESETS
                    .iter()
                    .filter(|p| p.model_prefixes.contains(prefix))
                    .map(|p| p.name)
                    .collect();
                assert_eq!(owners, vec![spec.name], "prefix `{prefix}` is ambiguous");
            }
        }
    }

    #[test]
    fn endpoints_must_be_https_or_loopback() {
        use super::super::provider::validate_endpoint;

        assert!(validate_endpoint("https://api.example.com/v1").is_ok());
        assert!(validate_endpoint("http://localhost:11434/v1").is_ok());
        assert!(validate_endpoint("http://127.0.0.1:8000/v1").is_ok());
        assert!(validate_endpoint("http://[::1]:8000/v1").is_ok());
        assert!(validate_endpoint("http://example.com/v1").is_err());
        assert!(validate_endpoint("ftp://example.com").is_err());
        assert!(validate_endpoint("api.example.com/v1").is_err());
        assert!(validate_endpoint("http://user:pw@example.com/v1").is_err());
        assert!(validate_endpoint("").is_err());
    }
}
