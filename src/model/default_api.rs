//! Which OpenAI-protocol HTTP API a provider talks by default
//! (Python: `set_default_openai_api`).

use std::sync::Mutex;

/// Which HTTP API the OpenAI-protocol models use by default (Python: `set_default_openai_api`).
///
/// Unlike Python, whose default is Responses, the default here is Chat Completions: this SDK
/// targets any server that speaks the OpenAI protocol, and Chat Completions is the one nearly
/// all of them implement (D-I).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DefaultOpenAiApi {
    /// Responses API.
    Responses,
    /// Chat Completions API (default).
    #[default]
    ChatCompletions,
}

static DEFAULT_OPENAI_API: Mutex<Option<DefaultOpenAiApi>> = Mutex::new(None);

/// Set the default OpenAI API (Python: `set_default_openai_api`).
///
/// An explicit choice applies to every provider that has not been given its own (see D-I).
pub fn set_default_openai_api(api: DefaultOpenAiApi) {
    *DEFAULT_OPENAI_API.lock().expect("api lock") = Some(api);
}

/// Get the default API: the explicit choice, else Chat Completions.
pub fn get_default_openai_api() -> DefaultOpenAiApi {
    explicit_default_openai_api().unwrap_or_default()
}

/// The API chosen with [`set_default_openai_api`], if any.
pub(crate) fn explicit_default_openai_api() -> Option<DefaultOpenAiApi> {
    *DEFAULT_OPENAI_API.lock().expect("api lock")
}
