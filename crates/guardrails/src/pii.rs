//! Rewrite personal data out of text, or refuse it.

use std::borrow::Cow;
use std::collections::BTreeSet;

use regex::Regex;

use opai_agents_core::guardrail::Severity;

use crate::error::BuildError;
use crate::guard::{Decision, Guard};

/// A kind of personal data [`PiiRedactor`] knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum PiiKind {
    /// An email address.
    Email,
    /// A payment card number: 13 to 19 digits, spaces or dashes between them, and a valid Luhn
    /// check digit. A long number that fails the check is an order number, not a card.
    Card,
    /// A US social security number, `123-45-6789`, in an assignable range.
    Ssn,
    /// A phone number: international (`+` and 8 to 15 digits) or US style (`(555) 123-4567`,
    /// `555-123-4567`). A bare run of ten digits is not matched: it is as likely an id.
    Phone,
}

impl PiiKind {
    /// Every kind, in the order they are looked for. Cards go first so a 16-digit number is a
    /// card before it is anything else.
    pub const ALL: [PiiKind; 4] = [Self::Card, Self::Ssn, Self::Email, Self::Phone];

    /// The text a match is replaced with.
    pub fn placeholder(self) -> &'static str {
        match self {
            Self::Email => "[EMAIL]",
            Self::Card => "[CARD]",
            Self::Ssn => "[SSN]",
            Self::Phone => "[PHONE]",
        }
    }

    /// Lower-case name, as used in a failure reason.
    pub fn name(self) -> &'static str {
        match self {
            Self::Email => "email",
            Self::Card => "card",
            Self::Ssn => "ssn",
            Self::Phone => "phone",
        }
    }

    fn pattern(self) -> &'static str {
        match self {
            Self::Email => r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}",
            Self::Card => r"\b\d(?:[ \-]?\d){12,18}\b",
            Self::Ssn => r"\b\d{3}-\d{2}-\d{4}\b",
            Self::Phone => {
                r"\+\d(?:[ .\-]?\d){7,14}\b|(?:\(\d{3}\)|\b\d{3})[ .\-]?\d{3}[ .\-]\d{4}\b"
            }
        }
    }

    /// Whether a candidate the pattern found is really this kind.
    fn confirm(self, candidate: &str) -> bool {
        let digits: Vec<u32> = candidate.chars().filter_map(|c| c.to_digit(10)).collect();
        match self {
            Self::Card => (13..=19).contains(&digits.len()) && !all_same(&digits) && luhn(&digits),
            Self::Ssn => {
                let area = digits[0] * 100 + digits[1] * 10 + digits[2];
                let group = digits[3] * 10 + digits[4];
                let serial = digits[5] * 1000 + digits[6] * 100 + digits[7] * 10 + digits[8];
                area != 0 && area != 666 && area < 900 && group != 0 && serial != 0
            }
            Self::Phone => {
                if candidate.starts_with('+') {
                    (8..=15).contains(&digits.len())
                } else {
                    true
                }
            }
            Self::Email => true,
        }
    }
}

fn all_same(digits: &[u32]) -> bool {
    digits.windows(2).all(|w| w[0] == w[1])
}

/// The Luhn check: double every second digit from the right, subtract 9 from any over 9, and the
/// sum must be a multiple of ten.
fn luhn(digits: &[u32]) -> bool {
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == 1 {
                let doubled = d * 2;
                if doubled > 9 {
                    doubled - 9
                } else {
                    doubled
                }
            } else {
                d
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

/// What to do when personal data is found.
#[derive(Debug, Clone, Copy)]
enum Mode {
    /// Carry on with it replaced by a placeholder.
    Redact,
    /// Refuse.
    Refuse(Severity),
}

/// Rewrites personal data to a placeholder (`[EMAIL]`, `[CARD]`, `[SSN]`, `[PHONE]`) and carries
/// on, or refuses when asked to.
///
/// Redaction is the default because it is what most uses want: the model still gets a question it
/// can answer, and the session stores the redacted text, not the original. As an input guardrail it
/// runs before the model, so the model never sees the data; as an output guardrail it rewrites the
/// final output, including every string inside a structured one.
///
/// This is pattern matching, with the limits that implies: it finds the kinds in [`PiiKind`] and
/// no other — not names, not street addresses, not a number written out in words — and a phone
/// number is only recognised in the shapes [`PiiKind::Phone`] lists. A guard that says it found
/// nothing has not proved there is nothing. For data that must not leave, refuse upstream of the
/// model instead.
///
/// ```rust
/// use opai_agents_guardrails::{Decision, Guard, PiiRedactor};
///
/// # fn build() -> Result<(), opai_agents_guardrails::BuildError> {
/// let redactor = PiiRedactor::new()?;
/// assert_eq!(
///     redactor.check("write to ada@example.com"),
///     Decision::Rewrite("write to [EMAIL]".into())
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PiiRedactor {
    kinds: Vec<(PiiKind, Regex)>,
    mode: Mode,
}

impl PiiRedactor {
    /// Look for every [`PiiKind`].
    pub fn new() -> Result<Self, BuildError> {
        Self::only(PiiKind::ALL)
    }

    /// Look for these kinds only. An empty list is an error: it would never find anything.
    pub fn only(kinds: impl IntoIterator<Item = PiiKind>) -> Result<Self, BuildError> {
        let wanted: BTreeSet<PiiKind> = kinds.into_iter().collect();
        if wanted.is_empty() {
            return Err(BuildError::new(
                "no kinds of personal data were given, so nothing would ever be found",
            ));
        }
        let mut compiled = Vec::new();
        // In the order of `ALL`, not the caller's: a card is a card before it is a phone number.
        for kind in PiiKind::ALL.into_iter().filter(|k| wanted.contains(k)) {
            let regex = Regex::new(kind.pattern())
                .map_err(|e| BuildError::new(format!("the {} pattern: {e}", kind.name())))?;
            compiled.push((kind, regex));
        }
        Ok(Self {
            kinds: compiled,
            mode: Mode::Redact,
        })
    }

    /// Refuse the text instead of rewriting it.
    #[must_use]
    pub fn refusing(mut self, severity: Severity) -> Self {
        self.mode = Mode::Refuse(severity);
        self
    }

    /// `text` with what was found replaced, and the kinds that were found.
    pub fn redact(&self, text: &str) -> (String, Vec<PiiKind>) {
        let mut current = Cow::Borrowed(text);
        let mut found = Vec::new();
        for (kind, regex) in &self.kinds {
            let mut hit = false;
            let replaced = regex.replace_all(&current, |caps: &regex::Captures<'_>| {
                let candidate = &caps[0];
                if kind.confirm(candidate) {
                    hit = true;
                    kind.placeholder().to_string()
                } else {
                    candidate.to_string()
                }
            });
            if hit {
                found.push(*kind);
                current = Cow::Owned(replaced.into_owned());
            }
        }
        (current.into_owned(), found)
    }
}

impl Guard for PiiRedactor {
    fn name(&self) -> &str {
        "pii_redactor"
    }

    fn check(&self, text: &str) -> Decision {
        let (redacted, found) = self.redact(text);
        if found.is_empty() {
            return Decision::Pass;
        }
        match self.mode {
            Mode::Redact => Decision::Rewrite(redacted),
            Mode::Refuse(severity) => {
                let kinds: Vec<&str> = found.iter().map(|k| k.name()).collect();
                Decision::fail(
                    format!("the text contains personal data ({})", kinds.join(", ")),
                    severity,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn redact(text: &str) -> String {
        PiiRedactor::new().unwrap().redact(text).0
    }

    #[test]
    fn emails_are_redacted() {
        assert_eq!(redact("mail ada@example.com now"), "mail [EMAIL] now");
        assert_eq!(redact("a.b+tag@sub.example.co.uk"), "[EMAIL]");
        assert_eq!(
            redact("two: a@x.io and b@y.org"),
            "two: [EMAIL] and [EMAIL]"
        );
        // Not addresses.
        assert_eq!(redact("meet @ noon"), "meet @ noon");
        assert_eq!(redact("user@localhost"), "user@localhost");
    }

    #[test]
    fn cards_need_a_valid_luhn_check_digit() {
        // The well-known test numbers.
        assert_eq!(redact("4111 1111 1111 1111"), "[CARD]");
        assert_eq!(redact("card 4111-1111-1111-1111."), "card [CARD].");
        assert_eq!(redact("5500005555555559"), "[CARD]");
        assert_eq!(redact("378282246310005"), "[CARD]");
        // Sixteen digits, wrong check digit: an order number, left alone.
        assert_eq!(
            redact("order 4111 1111 1111 1112"),
            "order 4111 1111 1111 1112"
        );
        // A run of one digit passes Luhn and is still not a card.
        assert_eq!(redact("0000000000000000"), "0000000000000000");
        // Too long to be a card.
        assert_eq!(redact("41111111111111111111"), "41111111111111111111");
    }

    #[test]
    fn ssns_in_an_assignable_range_are_redacted() {
        assert_eq!(redact("ssn 123-45-6789"), "ssn [SSN]");
        assert_eq!(redact("000-12-3456"), "000-12-3456");
        assert_eq!(redact("666-12-3456"), "666-12-3456");
        assert_eq!(redact("900-12-3456"), "900-12-3456");
        assert_eq!(redact("123-00-6789"), "123-00-6789");
        assert_eq!(redact("123-45-0000"), "123-45-0000");
    }

    #[test]
    fn phones_in_the_listed_shapes_are_redacted() {
        assert_eq!(redact("call (555) 123-4567 now"), "call [PHONE] now");
        assert_eq!(redact("555-123-4567"), "[PHONE]");
        assert_eq!(redact("555.123.4567"), "[PHONE]");
        assert_eq!(redact("+44 20 7946 0958"), "[PHONE]");
        assert_eq!(redact("+1 555 123 4567"), "[PHONE]");
    }

    /// What it must not touch: the shapes that look like data and are not.
    #[test]
    fn near_misses_are_left_alone() {
        for text in [
            "2024-10-20",
            "version 1.2.3",
            "room 101, floor 3",
            "page 12 of 345",
            "1234567890",
            "the year 2026",
            "+12345",
            "ratio 3:1",
            "ip 192.168.1.1",
        ] {
            assert_eq!(redact(text), text, "`{text}` is not personal data");
        }
    }

    /// The honest limits, written as tests so they cannot drift into claims.
    #[test]
    fn what_it_cannot_find_is_documented_not_hidden() {
        for text in [
            "call me at five five five one two three",
            "Ada Lovelace, 10 Downing St",
        ] {
            assert_eq!(redact(text), text);
        }
    }

    #[test]
    fn a_card_is_a_card_before_it_is_a_phone_number() {
        assert_eq!(
            redact("4111111111111111 and 555-123-4567"),
            "[CARD] and [PHONE]"
        );
    }

    #[test]
    fn a_kind_can_be_chosen() {
        let only_email = PiiRedactor::only([PiiKind::Email]).unwrap();
        assert_eq!(
            only_email.redact("a@b.io 555-123-4567").0,
            "[EMAIL] 555-123-4567"
        );
        assert!(PiiRedactor::only([]).is_err());
    }

    #[test]
    fn redaction_is_the_default_and_refusing_is_a_choice() {
        let guard = PiiRedactor::new().unwrap();
        assert_eq!(guard.check("nothing here"), Decision::Pass);
        assert_eq!(
            guard.check("ada@example.com"),
            Decision::Rewrite("[EMAIL]".into())
        );
        let refusing = PiiRedactor::new().unwrap().refusing(Severity::Critical);
        let Decision::Fail { reason, severity } =
            refusing.check("ada@example.com 4111111111111111")
        else {
            panic!("expected a failure");
        };
        assert_eq!(severity, Severity::Critical);
        assert!(reason.contains("email") && reason.contains("card"));
        assert!(!reason.contains("ada@"));
    }

    /// Patterns run in time linear in the text, whatever shape the text is.
    #[test]
    fn adversarial_text_is_not_a_denial_of_service() {
        let guard = PiiRedactor::new().unwrap();
        let inputs = [
            "1 ".repeat(100_000),
            "1-".repeat(100_000),
            "a".repeat(200_000) + "@",
            format!("{}@", "a.".repeat(100_000)),
            "+1".to_string() + &" 1".repeat(100_000),
        ];
        let start = std::time::Instant::now();
        for input in &inputs {
            let _ = guard.check(input);
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(20),
            "took {:?}",
            start.elapsed()
        );
    }
}
