//! A small model catalogue, for the two answers worth having locally (Rust-only, X-014).
//!
//! Asking a provider is the only way to know for sure, and this SDK does not do that. What a
//! catalogue can do without a round trip is catch the two mistakes that cost the most time: naming
//! a model that was retired, and typo-ing one.
//!
//! So this is deliberately a hint, never a gate: an unknown id is always allowed through (private
//! deployments and gateways serve models no catalogue knows), and only `Retired` is meant to stop
//! you. [`advise`] returns the text when there is something to say.

/// Whether a model is still served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    /// Current at [`super::LAST_CHECKED`].
    Current,
    /// Still served, but superseded.
    Legacy {
        /// What to use instead.
        replaced_by: &'static str,
    },
    /// No longer served: requests to it fail or 404.
    Retired {
        /// What to use instead, when there is an obvious one.
        replaced_by: Option<&'static str>,
    },
}

/// One entry of the catalogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelEntry {
    /// The provider it belongs to.
    pub provider: &'static str,
    /// The model id as it goes in a request.
    pub id: &'static str,
    /// Whether it is still served.
    pub status: ModelStatus,
    /// Why, or what to know about it.
    pub note: &'static str,
}

impl ModelEntry {
    /// What the status means for a request, if anything.
    pub fn advice(&self) -> Option<String> {
        match self.status {
            ModelStatus::Current => None,
            ModelStatus::Legacy { replaced_by } => Some(format!(
                "model `{}` is superseded; `{}` is the current one",
                self.id, replaced_by
            )),
            ModelStatus::Retired { replaced_by } => Some(match replaced_by {
                Some(replacement) => format!(
                    "model `{}` was retired; use `{}` instead ({})",
                    self.id, replacement, self.note
                ),
                None => format!("model `{}` was retired ({})", self.id, self.note),
            }),
        }
    }
}

/// The catalogue. Only models whose history is settled are listed: a wrong entry is worse than
/// none, and anything missing is simply unknown rather than wrong.
pub const MODELS: &[ModelEntry] = &[
    // OpenAI
    ModelEntry {
        provider: "openai",
        id: "gpt-4o",
        status: ModelStatus::Current,
        note: "multimodal flagship of its generation",
    },
    ModelEntry {
        provider: "openai",
        id: "gpt-4o-mini",
        status: ModelStatus::Current,
        note: "cheaper and faster than gpt-4o",
    },
    ModelEntry {
        provider: "openai",
        id: "gpt-4",
        status: ModelStatus::Legacy {
            replaced_by: "gpt-4o",
        },
        note: "superseded by gpt-4o",
    },
    ModelEntry {
        provider: "openai",
        id: "gpt-3.5-turbo",
        status: ModelStatus::Legacy {
            replaced_by: "gpt-4o-mini",
        },
        note: "cheaper and better at the same price point",
    },
    ModelEntry {
        provider: "openai",
        id: "gpt-4-0314",
        status: ModelStatus::Retired {
            replaced_by: Some("gpt-4o"),
        },
        note: "snapshot models were shut down",
    },
    ModelEntry {
        provider: "openai",
        id: "gpt-4-32k",
        status: ModelStatus::Retired {
            replaced_by: Some("gpt-4o"),
        },
        note: "32k context is no longer a separate model",
    },
    ModelEntry {
        provider: "openai",
        id: "text-davinci-003",
        status: ModelStatus::Retired { replaced_by: None },
        note: "the completions API was shut down",
    },
    // DeepSeek
    ModelEntry {
        provider: "deepseek",
        id: "deepseek-chat",
        status: ModelStatus::Current,
        note: "general chat",
    },
    ModelEntry {
        provider: "deepseek",
        id: "deepseek-reasoner",
        status: ModelStatus::Current,
        note: "reports reasoning",
    },
    ModelEntry {
        provider: "deepseek",
        id: "deepseek-coder",
        status: ModelStatus::Legacy {
            replaced_by: "deepseek-chat",
        },
        note: "folded into deepseek-chat",
    },
    // Gemini
    ModelEntry {
        provider: "gemini",
        id: "gemini-2.5-flash",
        status: ModelStatus::Current,
        note: "fast and cheap",
    },
    ModelEntry {
        provider: "gemini",
        id: "gemini-2.5-pro",
        status: ModelStatus::Current,
        note: "most capable",
    },
    ModelEntry {
        provider: "gemini",
        id: "gemini-1.5-pro",
        status: ModelStatus::Retired {
            replaced_by: Some("gemini-2.5-pro"),
        },
        note: "the 1.5 family was retired",
    },
    ModelEntry {
        provider: "gemini",
        id: "gemini-pro",
        status: ModelStatus::Retired {
            replaced_by: Some("gemini-2.5-pro"),
        },
        note: "renamed, then retired",
    },
    // Anthropic
    ModelEntry {
        provider: "anthropic",
        id: "claude-2",
        status: ModelStatus::Retired { replaced_by: None },
        note: "the Claude 2 family was retired",
    },
    ModelEntry {
        provider: "anthropic",
        id: "claude-instant-1",
        status: ModelStatus::Retired { replaced_by: None },
        note: "retired with Claude 2",
    },
];

/// The catalogue entry for `id`, if there is one.
pub fn model(id: &str) -> Option<&'static ModelEntry> {
    MODELS.iter().find(|m| m.id == id)
}

/// What is worth saying about `id`, if anything.
///
/// - known and retired: switch, and the reason.
/// - known and superseded: what is current.
/// - unknown but close to a known id: a suggestion. An unknown id is never an error: private
///   deployments and gateways serve models no catalogue knows, so this is a hint, not a gate.
pub fn advise(id: &str) -> Option<String> {
    if let Some(entry) = model(id) {
        return entry.advice();
    }
    suggest(id).map(|entry| format!("unknown model `{id}`; did you mean `{}`?", entry.id))
}

/// The known id closest to `id`, when one is close enough to be a typo.
fn suggest(id: &str) -> Option<&'static ModelEntry> {
    let lower = id.to_ascii_lowercase();
    MODELS
        .iter()
        .filter(|m| distance(m.id, &lower) <= 3 || m.id.starts_with(&lower) && !lower.is_empty())
        .min_by_key(|m| distance(m.id, &lower))
}

/// Levenshtein distance, for typo suggestions; ids are short.
fn distance(left: &str, right: &str) -> usize {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0usize; right.len() + 1];
    for (i, a) in left.iter().enumerate() {
        current[0] = i + 1;
        for (j, b) in right.iter().enumerate() {
            let cost = usize::from(a != b);
            current[j + 1] = (previous[j] + cost)
                .min(previous[j + 1] + 1)
                .min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The status of a known model.
    fn status_of(id: &str) -> String {
        model(id)
            .and_then(ModelEntry::advice)
            .unwrap_or_else(|| panic!("`{id}` is known but has no advice"))
    }

    #[test]
    fn retired_models_say_what_to_use_instead() {
        let advice = status_of("gpt-4-0314");
        assert!(advice.contains("retired"), "{advice}");
        assert!(advice.contains("gpt-4o"), "{advice}");
        let advice = status_of("gemini-1.5-pro");
        assert!(advice.contains("gemini-2.5-pro"), "{advice}");
    }

    #[test]
    fn legacy_models_name_the_current_one() {
        let advice = status_of("gpt-4");
        assert!(advice.contains("superseded"), "{advice}");
        assert!(advice.contains("gpt-4o"), "{advice}");
    }

    #[test]
    fn current_models_are_quiet() {
        assert!(model("gpt-4o").and_then(ModelEntry::advice).is_none());
        assert!(model("deepseek-chat")
            .and_then(ModelEntry::advice)
            .is_none());
    }

    #[test]
    fn a_typo_gets_a_suggestion() {
        let advice = advise("gpt-4oo").expect("close to gpt-4o");
        assert!(advice.contains("did you mean"), "{advice}");
        assert!(advice.contains("gpt-4o"), "{advice}");
    }

    /// A model the catalogue does not know is fine: it says nothing.
    #[test]
    fn an_unfamiliar_model_is_not_second_guessed() {
        assert!(advise("acme-large").is_none());
        assert!(advise("llama3.1:8b").is_none());
    }

    #[test]
    fn ids_are_unique() {
        let mut ids: Vec<&str> = MODELS.iter().map(|m| m.id).collect();
        let total = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "duplicate model id");
    }

    #[test]
    fn every_entry_belongs_to_a_preset() {
        for entry in MODELS {
            assert!(
                super::super::spec::preset(entry.provider).is_some(),
                "`{}` belongs to no preset",
                entry.provider
            );
        }
    }
}
