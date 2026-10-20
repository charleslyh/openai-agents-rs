//! Refuse a list of words or phrases.

use regex::{Regex, RegexBuilder};

use opai_agents_core::guardrail::Severity;

use crate::error::BuildError;
use crate::guard::{Decision, Guard};
use crate::text::normalise;

/// The longest keyword accepted.
pub const MAX_KEYWORD_CHARS: usize = 200;

/// The compiled size limit: a list too large to match cheaply is an error, not a slow filter.
const SIZE_LIMIT: usize = 4 * 1024 * 1024;

/// Refuses text that contains any of a list of words or phrases.
///
/// * **Case-insensitive**, and blind to invisible characters and to runs of whitespace: a phrase
///   written with a zero-width space in it, or with two spaces, is still found.
/// * **Whole words by default**: `kill` does not match `skill`. [`substrings`](Self::substrings)
///   matches anywhere, for a term that is rude as a fragment.
/// * Keywords are matched literally; none of them is a pattern.
///
/// The failure says that a blocked keyword was found, not which one: a refusal that names the word
/// tells the next attempt what to rephrase.
///
/// ```rust
/// use opai_agents_guardrails::{BlockedKeywords, Guard};
///
/// # fn build() -> Result<(), opai_agents_guardrails::BuildError> {
/// let guardrail = BlockedKeywords::new(["project atlas", "internal only"])?.input();
/// # let _ = guardrail;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BlockedKeywords {
    pattern: Regex,
    severity: Severity,
}

impl BlockedKeywords {
    /// Block `keywords` as whole words or phrases.
    ///
    /// An empty list, an empty keyword and one over [`MAX_KEYWORD_CHARS`] characters are errors.
    pub fn new<I, S>(keywords: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::build(keywords, true)
    }

    /// Block `keywords` wherever they appear, inside other words too.
    pub fn substrings<I, S>(keywords: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::build(keywords, false)
    }

    fn build<I, S>(keywords: I, whole_words: bool) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut alternatives = Vec::new();
        for keyword in keywords {
            let keyword = normalise(keyword.as_ref());
            if keyword.is_empty() {
                return Err(BuildError::new("a blocked keyword is empty"));
            }
            if keyword.chars().count() > MAX_KEYWORD_CHARS {
                return Err(BuildError::new(format!(
                    "a blocked keyword is longer than {MAX_KEYWORD_CHARS} characters"
                )));
            }
            alternatives.push(regex::escape(&keyword));
        }
        if alternatives.is_empty() {
            return Err(BuildError::new(
                "no keywords were given, so nothing would ever be blocked",
            ));
        }
        let any = alternatives.join("|");
        // A word boundary that also works next to punctuation (`c++`): the neighbour must not be a
        // letter, a digit or an underscore.
        let source = if whole_words {
            format!(r"(?:^|[^\p{{L}}\p{{N}}_])(?:{any})(?:$|[^\p{{L}}\p{{N}}_])")
        } else {
            format!("(?:{any})")
        };
        let pattern = RegexBuilder::new(&source)
            .case_insensitive(true)
            .size_limit(SIZE_LIMIT)
            .build()
            .map_err(|e| BuildError::new(format!("the keyword list cannot be matched: {e}")))?;
        Ok(Self {
            pattern,
            severity: Severity::High,
        })
    }

    /// How serious a failure is (default: [`Severity::High`]).
    #[must_use]
    pub fn severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }
}

impl Guard for BlockedKeywords {
    fn name(&self) -> &str {
        "blocked_keywords"
    }

    fn check(&self, text: &str) -> Decision {
        if self.pattern.is_match(&normalise(text)) {
            Decision::fail("the text contains a blocked keyword", self.severity)
        } else {
            Decision::Pass
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocked(guard: &BlockedKeywords, text: &str) -> bool {
        matches!(guard.check(text), Decision::Fail { .. })
    }

    #[test]
    fn whole_words_are_matched_without_regard_to_case() {
        let guard = BlockedKeywords::new(["kill", "project atlas"]).unwrap();
        assert!(blocked(&guard, "I will KILL it"));
        assert!(blocked(&guard, "kill"));
        assert!(blocked(&guard, "what is Project Atlas?"));
        assert!(blocked(&guard, "(kill)"));
        // The Scunthorpe problem: a word inside another word is not the word.
        assert!(!blocked(&guard, "a skill and a killer app"));
        assert!(!blocked(&guard, "atlas"));
    }

    #[test]
    fn substrings_match_inside_words() {
        let guard = BlockedKeywords::substrings(["kill"]).unwrap();
        assert!(blocked(&guard, "a skill"));
    }

    #[test]
    fn keywords_are_literal_not_patterns() {
        let guard = BlockedKeywords::new(["c++", "a.b", "(secret)"]).unwrap();
        assert!(blocked(&guard, "I like c++ a lot"));
        assert!(blocked(&guard, "the a.b file"));
        assert!(!blocked(&guard, "the axb file"));
        assert!(blocked(&guard, "x (secret) y"));
    }

    #[test]
    fn invisible_characters_and_spacing_do_not_hide_a_phrase() {
        let guard = BlockedKeywords::new(["project atlas"]).unwrap();
        assert!(blocked(&guard, "project\u{200b} atlas"));
        assert!(blocked(&guard, "project    \n atlas"));
    }

    #[test]
    fn a_bad_list_is_refused_at_construction() {
        assert!(BlockedKeywords::new(Vec::<&str>::new()).is_err());
        assert!(BlockedKeywords::new([""]).is_err());
        assert!(BlockedKeywords::new(["   "]).is_err());
        assert!(BlockedKeywords::new(["x".repeat(MAX_KEYWORD_CHARS + 1)]).is_err());
    }

    #[test]
    fn a_list_too_large_to_match_is_refused_not_slow() {
        let keywords: Vec<String> = (0..200_000)
            .map(|i| format!("keyword{i}-padding"))
            .collect();
        assert!(BlockedKeywords::new(keywords).is_err());
    }

    #[test]
    fn the_reason_names_no_keyword() {
        let Decision::Fail { reason, .. } = BlockedKeywords::new(["hunter2"])
            .unwrap()
            .check("my password is hunter2")
        else {
            panic!("expected a failure");
        };
        assert!(!reason.contains("hunter2"));
    }
}
