//! Text handling shared by the guards that match phrases.

/// Characters that take no space and change nothing a reader sees: the usual way to write
/// `ig​nore` so a phrase filter does not find it.
fn is_invisible(c: char) -> bool {
    matches!(
        c,
        '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2060}'..='\u{2064}' | '\u{feff}' | '\u{00ad}'
    )
}

/// `text` without invisible characters and with every run of whitespace as one space.
pub(crate) fn normalise(text: &str) -> String {
    let visible: String = text.chars().filter(|c| !is_invisible(*c)).collect();
    visible.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invisible_characters_and_whitespace_runs_go() {
        assert_eq!(normalise("ig\u{200b}nore \t  this\n"), "ignore this");
        assert_eq!(normalise("a\u{feff}b"), "ab");
    }
}
