//! Refuse text over a limit.

use opai_agents_core::guardrail::Severity;

use crate::error::BuildError;
use crate::guard::{Decision, Guard};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    Chars,
    Bytes,
}

/// Refuses a piece of text longer than a limit.
///
/// The limit applies to each piece of text on its own — each user message, each string of a
/// structured output — not to their total. Counting is by characters or by UTF-8 bytes: bytes are
/// what a body size limit means, characters are what a person means by "length".
///
/// ```rust
/// use opai_agents_guardrails::{Guard, MaxLength};
///
/// # fn build() -> Result<(), opai_agents_guardrails::BuildError> {
/// let guardrail = MaxLength::chars(4_000)?.input();
/// # let _ = guardrail;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MaxLength {
    limit: usize,
    unit: Unit,
    severity: Severity,
}

impl MaxLength {
    /// At most `limit` characters. A limit of zero is an error: it refuses everything.
    pub fn chars(limit: usize) -> Result<Self, BuildError> {
        Self::new(limit, Unit::Chars)
    }

    /// At most `limit` bytes of UTF-8. A limit of zero is an error: it refuses everything.
    pub fn bytes(limit: usize) -> Result<Self, BuildError> {
        Self::new(limit, Unit::Bytes)
    }

    fn new(limit: usize, unit: Unit) -> Result<Self, BuildError> {
        if limit == 0 {
            return Err(BuildError::new(
                "a length limit of zero refuses everything; use at least 1",
            ));
        }
        Ok(Self {
            limit,
            unit,
            severity: Severity::High,
        })
    }

    /// How serious a failure is (default: [`Severity::High`]).
    #[must_use]
    pub fn severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    fn too_long(&self, text: &str) -> bool {
        match self.unit {
            Unit::Bytes => text.len() > self.limit,
            // Stop counting one past the limit: the answer is known, and the text may be huge.
            Unit::Chars => text.chars().take(self.limit.saturating_add(1)).count() > self.limit,
        }
    }
}

impl Guard for MaxLength {
    fn name(&self) -> &str {
        "max_length"
    }

    fn check(&self, text: &str) -> Decision {
        if self.too_long(text) {
            let unit = match self.unit {
                Unit::Chars => "characters",
                Unit::Bytes => "bytes",
            };
            Decision::fail(
                format!("a piece of text is longer than {} {unit}", self.limit),
                self.severity,
            )
        } else {
            Decision::Pass
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_limit_is_inclusive() {
        let guard = MaxLength::chars(5).unwrap();
        assert_eq!(guard.check("hello"), Decision::Pass);
        assert!(matches!(guard.check("hello!"), Decision::Fail { .. }));
        assert_eq!(guard.check(""), Decision::Pass);
    }

    #[test]
    fn characters_and_bytes_differ_for_multibyte_text() {
        // Seven characters, fourteen bytes.
        let text = "ééééééé";
        assert_eq!(text.chars().count(), 7);
        assert_eq!(text.len(), 14);
        assert_eq!(MaxLength::chars(7).unwrap().check(text), Decision::Pass);
        assert!(matches!(
            MaxLength::bytes(7).unwrap().check(text),
            Decision::Fail { .. }
        ));
    }

    #[test]
    fn a_limit_of_zero_is_refused_at_construction() {
        assert!(MaxLength::chars(0).is_err());
        assert!(MaxLength::bytes(0).is_err());
    }

    #[test]
    fn the_reason_does_not_repeat_the_text() {
        let Decision::Fail { reason, severity } = MaxLength::chars(3)
            .unwrap()
            .severity(Severity::Low)
            .check("secret text")
        else {
            panic!("expected a failure");
        };
        assert!(!reason.contains("secret"));
        assert_eq!(severity, Severity::Low);
    }

    #[test]
    fn a_huge_text_is_cheap_to_refuse() {
        let huge = "x".repeat(10_000_000);
        assert!(matches!(
            MaxLength::chars(10).unwrap().check(&huge),
            Decision::Fail { .. }
        ));
    }
}
