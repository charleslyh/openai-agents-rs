//! HTTP transport helpers: error mapping and the SSE reader.

use openai_agents_core::error::ModelError;

pub(crate) fn map_transport(err: impl std::fmt::Display) -> ModelError {
    ModelError::Transport(err.to_string())
}

/// Classify a `reqwest` failure: connection, timeout and body-read errors are the ones a retry
/// policy may replay; an undecodable body is not (Python: `APIConnectionError` / `APITimeoutError`).
pub(crate) fn map_reqwest(err: reqwest::Error) -> ModelError {
    if err.is_timeout() {
        return ModelError::Connection(openai_agents_core::error::ModelConnectionError {
            message: err.to_string(),
            is_timeout: true,
        });
    }
    if err.is_connect() || err.is_request() || err.is_body() {
        return ModelError::Connection(openai_agents_core::error::ModelConnectionError {
            message: err.to_string(),
            is_timeout: false,
        });
    }
    ModelError::Transport(err.to_string())
}

/// Turn a non-success HTTP response into a [`ModelError::Status`] that keeps the status, headers
/// and body the retry layer reads (`Retry-After`, `x-should-retry`, the error code).
pub(crate) async fn status_error(label: &str, resp: reqwest::Response) -> ModelError {
    let status = resp.status();
    let headers = resp
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|v| (name.as_str().to_ascii_lowercase(), v.to_string()))
        })
        .collect();
    let text = resp.text().await.unwrap_or_default();
    let body = serde_json::from_str::<serde_json::Value>(&text).unwrap_or_else(|_| {
        if text.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(text)
        }
    });
    ModelError::Status(openai_agents_core::error::ModelStatusError {
        status_code: status.as_u16(),
        message: format!("{label} status={status} body={body}"),
        body,
        headers,
    })
}

/// Whether a response carries a plain JSON body instead of an SSE stream.
///
/// Some gateways (and wiremock) ignore `stream: true` and answer with the full object, so
/// adapters must fall back to the non-streaming parse.
pub(crate) fn is_json_response(resp: &reqwest::Response) -> bool {
    resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("application/json")
}

/// Incremental SSE parser shared by the Chat Completions and Responses adapters.
///
/// OpenAI streams `text/event-stream` bodies as `data: {...}` lines separated by blank lines.
/// `[DONE]` terminators and non-`data:` lines (comments, `event:`) are skipped.
pub(crate) struct SseReader {
    buffer: String,
    decoder: openai_agents_core::utf8_stream::Utf8Stream,
}

impl SseReader {
    /// Start with an empty buffer.
    pub(crate) fn new() -> Self {
        Self {
            buffer: String::new(),
            decoder: Default::default(),
        }
    }

    /// Append raw bytes from the HTTP byte stream.
    pub(crate) fn feed(&mut self, chunk: &[u8]) {
        self.buffer.push_str(&self.decoder.push(chunk));
    }

    /// Drain the next complete `data:` payload, if one is buffered.
    pub(crate) fn next_event(&mut self) -> Option<Result<serde_json::Value, ModelError>> {
        while let Some(pos) = self.buffer.find('\n') {
            let mut line: String = self.buffer[..pos].to_string();
            self.buffer.drain(..=pos);
            if line.ends_with('\r') {
                line.pop();
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                continue;
            }
            return Some(
                serde_json::from_str(data).map_err(|e| {
                    ModelError::Behavior(format!("invalid SSE JSON: {e}; data={data}"))
                }),
            );
        }
        None
    }
}

#[cfg(test)]
mod sse_tests {
    use super::SseReader;

    #[test]
    fn multibyte_text_split_across_chunks_is_not_corrupted() {
        let line = "data: {\"delta\":\"你好🙂\"}\n\n".as_bytes().to_vec();
        for split in 1..line.len() {
            let mut reader = SseReader::new();
            reader.feed(&line[..split]);
            let early = reader.next_event();
            reader.feed(&line[split..]);
            let event = early.or_else(|| reader.next_event()).expect("event");
            assert_eq!(event.expect("json")["delta"], "你好🙂", "split at {split}");
        }
    }
}
