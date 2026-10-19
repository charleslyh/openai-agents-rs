//! Incremental UTF-8 decoding for chunked byte streams.
//!
//! Network bodies arrive in arbitrary chunks, so a multi-byte character (CJK text, emoji) can be
//! split across two chunks. Decoding each chunk on its own turns both halves into U+FFFD and
//! silently corrupts the text; [`Utf8Stream`] holds back the incomplete tail until the next chunk.

/// Decodes a stream of byte chunks into text without splitting characters.
#[derive(Debug, Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// An empty decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `chunk` and return the text that is now complete.
    ///
    /// Bytes that can never form a character are replaced with U+FFFD; an incomplete character at
    /// the end of the chunk is kept for the next call.
    pub fn push(&mut self, chunk: &[u8]) -> String {
        self.pending.extend_from_slice(chunk);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    out.push_str(valid);
                    self.pending.clear();
                    return out;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    out.push_str(&String::from_utf8_lossy(&self.pending[..valid_up_to]));
                    match err.error_len() {
                        // The tail is the start of a character: wait for more bytes.
                        None => {
                            self.pending.drain(..valid_up_to);
                            return out;
                        }
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid_up_to + bad);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Utf8Stream;

    #[test]
    fn a_character_split_across_chunks_is_decoded_intact() {
        let text = "你好, 🙂!";
        let bytes = text.as_bytes();
        for split in 0..=bytes.len() {
            let mut decoder = Utf8Stream::new();
            let mut got = decoder.push(&bytes[..split]);
            got.push_str(&decoder.push(&bytes[split..]));
            assert_eq!(got, text, "split at {split}");
        }
    }

    #[test]
    fn byte_at_a_time_matches_the_original() {
        let text = "héllo 世界 🙂";
        let mut decoder = Utf8Stream::new();
        let got: String = text.bytes().map(|b| decoder.push(&[b])).collect();
        assert_eq!(got, text);
    }

    #[test]
    fn invalid_bytes_become_replacement_characters_and_decoding_continues() {
        let mut decoder = Utf8Stream::new();
        assert_eq!(decoder.push(b"a\xffb"), "a\u{FFFD}b");
        assert_eq!(decoder.push(b"c"), "c");
    }
}
