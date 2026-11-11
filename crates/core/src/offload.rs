//! Moving bulky tool outputs out of the context, and reading them back on demand.
//!
//! [`ToolOutputTrimmer`](crate::ToolOutputTrimmer) shortens an old tool output for good: what it
//! cut is gone. [`OffloadToolOutputs`] keeps it. The full text goes to a [`BlobStore`], the
//! history keeps a preview and a reference, and the model can fetch any part of the original with
//! the tool from [`offload_read_tool`] (or all of it wired up at once by [`offload_capability`]).
//!
//! The pieces fit together by a *namespace*, a string that names whose outputs these are
//! (usually the session id). The reducer writes into a namespace and the tool reads only from
//! the one it was built with, so a conversation cannot reach the outputs of another one as long
//! as each gets its own namespace.
//!
//! Mount the reducer on a [`CompactingSession`](crate::CompactingSession) rather than as a model
//! input filter: on a session the output is written once, while a filter would run (and check the
//! store) before every model call. References are derived from the content, so running again
//! over the same history writes nothing new.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::agent::{CapabilityParts, NamedCapability};
use crate::context::{recent_boundary, tool_names};
use crate::error::AgentsError;
use crate::reducer::{ContextReducer, ReduceInput, ReduceOutcome};
use crate::tool::FunctionTool;

/// Name of the tool that reads an offloaded output back.
pub const READ_OFFLOADED_TOOL: &str = "read_offloaded";

/// Start of the note that replaces an offloaded output.
const OFFLOADED_PREFIX: &str = "[Offloaded:";

/// Length of a reference, in hex characters (128 bits of a SHA-256 digest).
const REF_LEN: usize = 32;

/// Most characters one call of the read tool returns, whatever the model asks for.
pub const MAX_READ_CHARS: usize = 20_000;

/// Characters one call of the read tool returns when the model does not say.
const DEFAULT_READ_CHARS: usize = 4_000;

/// Where offloaded text lives. Keys are the references made by [`reference_for`].
///
/// An implementation must keep namespaces apart (a key written in one is invisible in another)
/// and treat `put` as idempotent. Namespaces are free text, so a backend that uses one in a path
/// or a query must not trust it.
#[async_trait]
pub trait BlobStore: Send + Sync {
    /// Store `text` under `key` in `namespace`. Writing the same key again is not an error.
    async fn put(&self, namespace: &str, key: &str, text: &str) -> Result<(), AgentsError>;

    /// The text stored under `key` in `namespace`, or `None` when there is none.
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<String>, AgentsError>;

    /// Remove everything in `namespace`, for example when its session is cleared.
    async fn delete_namespace(&self, namespace: &str) -> Result<(), AgentsError>;
}

/// The reference of `text`: the first 32 hex characters of its SHA-256 digest.
pub fn reference_for(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    let mut out = String::with_capacity(REF_LEN);
    for byte in digest.iter().take(REF_LEN / 2) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Whether `key` has the shape of a reference. A store can use it to refuse anything else before
/// it reaches a path or a query.
pub fn is_valid_reference(key: &str) -> bool {
    key.len() == REF_LEN && key.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// A [`BlobStore`] that keeps everything in memory, for tests and short-lived processes.
#[derive(Default)]
pub struct InMemoryBlobStore {
    blobs: Mutex<HashMap<(String, String), String>>,
}

impl InMemoryBlobStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty store, shared.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Number of blobs held, over all namespaces.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<(String, String), String>> {
        self.blobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[async_trait]
impl BlobStore for InMemoryBlobStore {
    async fn put(&self, namespace: &str, key: &str, text: &str) -> Result<(), AgentsError> {
        self.lock()
            .insert((namespace.to_string(), key.to_string()), text.to_string());
        Ok(())
    }

    async fn get(&self, namespace: &str, key: &str) -> Result<Option<String>, AgentsError> {
        Ok(self
            .lock()
            .get(&(namespace.to_string(), key.to_string()))
            .cloned())
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), AgentsError> {
        self.lock().retain(|(ns, _), _| ns != namespace);
        Ok(())
    }
}

/// Moves the full text of old, bulky tool outputs to a [`BlobStore`] and leaves a preview with a
/// reference in the history.
///
/// The latest `recent_exchanges` user messages (with what follows them) are never touched. In
/// older exchanges a string output longer than `max_output_chars` is stored and replaced by its
/// first `preview_chars` characters plus the reference. Outputs that were already offloaded, and
/// replacements that would not be shorter than the original, are left alone, so running it twice
/// changes nothing.
///
/// When the store fails, the output stays as it was and a warning is logged: the history is
/// never worse than before. Give the model [`offload_read_tool`] (or [`offload_capability`]) with
/// the same store and namespace, or it cannot read what was moved.
pub struct OffloadToolOutputs {
    store: Arc<dyn BlobStore>,
    namespace: String,
    recent_exchanges: usize,
    max_output_chars: usize,
    preview_chars: usize,
}

impl OffloadToolOutputs {
    /// Offload into `namespace` of `store`. Defaults: the 2 latest exchanges stay as they are,
    /// outputs over 2000 characters move, 300 characters stay as a preview.
    pub fn new(store: Arc<dyn BlobStore>, namespace: impl Into<String>) -> Self {
        Self {
            store,
            namespace: namespace.into(),
            recent_exchanges: 2,
            max_output_chars: 2_000,
            preview_chars: 300,
        }
    }

    /// Number of latest user messages (with what follows them) kept intact. At least 1.
    pub fn recent_exchanges(mut self, exchanges: usize) -> Self {
        self.recent_exchanges = exchanges.max(1);
        self
    }

    /// Outputs longer than this many characters are moved. At least 1.
    pub fn max_output_chars(mut self, chars: usize) -> Self {
        self.max_output_chars = chars.max(1);
        self
    }

    /// Characters of the original kept as a preview.
    pub fn preview_chars(mut self, chars: usize) -> Self {
        self.preview_chars = chars;
        self
    }

    async fn offload_one(&self, item: &Value, tool: &str) -> Option<Value> {
        let output = item.get("output")?.as_str()?;
        if output.starts_with(OFFLOADED_PREFIX) {
            return None;
        }
        let length = output.chars().count();
        if length <= self.max_output_chars {
            return None;
        }
        let preview: String = output.chars().take(self.preview_chars).collect();
        let key = reference_for(output);
        let name = if tool.is_empty() {
            "unknown_tool"
        } else {
            tool
        };
        let note = format!(
            "{OFFLOADED_PREFIX} ref={key} tool={name} chars={length}]\n{preview}...\n\
             (the rest is stored; read it with {READ_OFFLOADED_TOOL})"
        );
        // Never swap in something longer than what it replaces.
        if note.chars().count() >= length {
            return None;
        }
        if let Err(error) = self.store.put(&self.namespace, &key, output).await {
            ::tracing::warn!(%error, "could not offload a tool output; it stays in the history");
            return None;
        }
        let mut replaced = item.clone();
        replaced["output"] = Value::String(note);
        Some(replaced)
    }
}

#[async_trait]
impl ContextReducer for OffloadToolOutputs {
    fn name(&self) -> &str {
        "offload_tool_outputs"
    }

    async fn reduce(&self, input: ReduceInput<'_>) -> Result<ReduceOutcome, AgentsError> {
        let boundary = recent_boundary(input.items, self.recent_exchanges);
        if boundary == 0 {
            return Ok(ReduceOutcome::unchanged(&input));
        }
        let names = tool_names(input.items);
        let mut items = Vec::with_capacity(input.items.len());
        for (index, item) in input.items.iter().enumerate() {
            let is_output =
                item.get("type").and_then(Value::as_str) == Some("function_call_output");
            let replaced = if index < boundary && is_output {
                let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let tool = names.get(call_id).copied().unwrap_or("");
                self.offload_one(item, tool).await
            } else {
                None
            };
            items.push(replaced.unwrap_or_else(|| item.clone()));
        }
        Ok(ReduceOutcome::new(&input, items))
    }
}

#[derive(Deserialize)]
struct ReadArgs {
    #[serde(rename = "ref")]
    reference: String,
    #[serde(default)]
    offset: usize,
    #[serde(default)]
    length: Option<usize>,
}

/// The tool that reads back what [`OffloadToolOutputs`] moved, from `namespace` of `store` only.
///
/// The model passes the `ref` from the note and, optionally, an `offset` and a `length` in
/// characters. One call returns at most [`MAX_READ_CHARS`] characters, and says how much of the
/// original there is, so a long output is read in pieces.
pub fn offload_read_tool(store: Arc<dyn BlobStore>, namespace: impl Into<String>) -> FunctionTool {
    let namespace: Arc<str> = namespace.into().into();
    FunctionTool::new(
        READ_OFFLOADED_TOOL,
        "Read part of a tool output that was moved out of the conversation. Pass the `ref` \
         from the [Offloaded: ...] note, and optionally an `offset` and a `length` in \
         characters to read it in pieces.",
        json!({
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "The reference from the note."},
                "offset": {"type": "integer", "minimum": 0,
                           "description": "First character to read. Default 0."},
                "length": {"type": "integer", "minimum": 1, "maximum": MAX_READ_CHARS,
                           "description": "Characters to read. Default 4000."}
            },
            "required": ["ref"],
            "additionalProperties": false
        }),
        move |_context, arguments| {
            let store = Arc::clone(&store);
            let namespace = Arc::clone(&namespace);
            async move {
                let args: ReadArgs = serde_json::from_str(&arguments).map_err(|error| {
                    crate::error::UserError::new(format!("bad arguments: {error}"))
                })?;
                Ok(Value::String(
                    read_slice(store.as_ref(), &namespace, &args).await?,
                ))
            }
        },
    )
}

async fn read_slice(
    store: &dyn BlobStore,
    namespace: &str,
    args: &ReadArgs,
) -> Result<String, AgentsError> {
    if !is_valid_reference(&args.reference) {
        return Ok("That is not a valid reference.".into());
    }
    let Some(text) = store.get(namespace, &args.reference).await? else {
        return Ok("No stored output has that reference.".into());
    };
    let total = text.chars().count();
    let length = args
        .length
        .unwrap_or(DEFAULT_READ_CHARS)
        .clamp(1, MAX_READ_CHARS);
    if args.offset >= total {
        return Ok(format!(
            "The output has {total} characters; offset is past the end."
        ));
    }
    let slice: String = text.chars().skip(args.offset).take(length).collect();
    let end = args.offset + slice.chars().count();
    Ok(format!(
        "[characters {}-{} of {total}]\n{slice}",
        args.offset, end
    ))
}

/// A [`Capability`](crate::Capability) with the read tool and a line of instructions explaining the notes.
pub fn offload_capability(
    store: Arc<dyn BlobStore>,
    namespace: impl Into<String>,
) -> NamedCapability {
    CapabilityParts::new()
        .tool(offload_read_tool(store, namespace))
        .instructions(format!(
            "Some earlier tool outputs were moved out of the conversation and replaced by a \
             note that starts with \"{OFFLOADED_PREFIX}\". Each note has a ref; call \
             {READ_OFFLOADED_TOOL} with it when you need more than the preview."
        ))
        .named("offloaded_outputs")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::default_token_counter;

    fn user(text: &str) -> Value {
        json!({"role": "user", "content": text})
    }

    fn call(id: &str) -> Value {
        json!({"type": "function_call", "call_id": id, "name": "search", "arguments": "{}"})
    }

    fn output(id: &str, text: &str) -> Value {
        json!({"type": "function_call_output", "call_id": id, "output": text})
    }

    fn history() -> Vec<Value> {
        vec![
            user("one"),
            call("c1"),
            output("c1", &"a".repeat(5_000)),
            user("two"),
            call("c2"),
            output("c2", &"b".repeat(5_000)),
            user("three"),
            call("c3"),
            output("c3", &"c".repeat(5_000)),
        ]
    }

    async fn reduce(reducer: &OffloadToolOutputs, items: &[Value]) -> ReduceOutcome {
        let counter = default_token_counter();
        reducer
            .reduce(ReduceInput {
                items,
                reserved_tokens: 0,
                counter: &counter,
            })
            .await
            .unwrap()
    }

    #[test]
    fn references_have_a_fixed_shape_and_follow_the_content() {
        let a = reference_for("hello");
        assert!(is_valid_reference(&a));
        assert_eq!(a, reference_for("hello"));
        assert_ne!(a, reference_for("hello!"));
        for bad in [
            "",
            "../etc/passwd",
            &"A".repeat(32),
            &"a".repeat(31),
            &"g".repeat(32),
        ] {
            assert!(!is_valid_reference(bad), "{bad}");
        }
    }

    #[tokio::test]
    async fn old_outputs_move_to_the_store_and_recent_ones_stay() {
        let store = InMemoryBlobStore::shared();
        let reducer = OffloadToolOutputs::new(store.clone(), "s1").recent_exchanges(2);
        let items = history();
        let outcome = reduce(&reducer, &items).await;

        let first = outcome.items[2]["output"].as_str().unwrap();
        assert!(first.starts_with(OFFLOADED_PREFIX));
        assert!(first.contains(&reference_for(&"a".repeat(5_000))));
        assert!(first.contains("tool=search"));
        assert_eq!(outcome.items[5], items[5], "recent exchanges stay");
        assert_eq!(outcome.items[8], items[8]);
        assert_eq!(store.len(), 1);
        assert!(outcome.report.tokens_after < outcome.report.tokens_before);
    }

    #[tokio::test]
    async fn offloading_twice_changes_nothing_and_writes_nothing_new() {
        let store = InMemoryBlobStore::shared();
        let reducer = OffloadToolOutputs::new(store.clone(), "s1").recent_exchanges(1);
        let once = reduce(&reducer, &history()).await;
        let twice = reduce(&reducer, &once.items).await;
        assert_eq!(once.items, twice.items);
        assert!(!twice.report.changed);
        assert_eq!(store.len(), 2);
    }

    #[tokio::test]
    async fn the_read_tool_returns_pieces_and_stays_in_its_namespace() {
        let store = InMemoryBlobStore::shared();
        let text: String = (0..100)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let key = reference_for(&text);
        store.put("mine", &key, &text).await.unwrap();

        let args = |offset, length| ReadArgs {
            reference: key.clone(),
            offset,
            length,
        };
        let piece = read_slice(store.as_ref(), "mine", &args(10, Some(5)))
            .await
            .unwrap();
        assert_eq!(
            piece,
            format!("[characters 10-15 of 100]\n{}", &text[10..15])
        );

        let whole = read_slice(store.as_ref(), "mine", &args(0, None))
            .await
            .unwrap();
        assert!(whole.starts_with("[characters 0-100 of 100]"));

        let past = read_slice(store.as_ref(), "mine", &args(100, None))
            .await
            .unwrap();
        assert!(past.contains("past the end"));

        let other = read_slice(store.as_ref(), "theirs", &args(0, None))
            .await
            .unwrap();
        assert!(
            other.contains("No stored output"),
            "another namespace sees nothing"
        );

        let bad = ReadArgs {
            reference: "../x".into(),
            offset: 0,
            length: None,
        };
        let reply = read_slice(store.as_ref(), "mine", &bad).await.unwrap();
        assert!(reply.contains("not a valid reference"));
    }

    #[tokio::test]
    async fn read_length_is_capped() {
        let store = InMemoryBlobStore::shared();
        let text = "x".repeat(MAX_READ_CHARS + 500);
        let key = reference_for(&text);
        store.put("n", &key, &text).await.unwrap();
        let args = ReadArgs {
            reference: key,
            offset: 0,
            length: Some(usize::MAX),
        };
        let reply = read_slice(store.as_ref(), "n", &args).await.unwrap();
        let body = reply.split_once('\n').unwrap().1;
        assert_eq!(body.chars().count(), MAX_READ_CHARS);
    }

    struct FailingStore;

    #[async_trait]
    impl BlobStore for FailingStore {
        async fn put(&self, _: &str, _: &str, _: &str) -> Result<(), AgentsError> {
            Err(AgentsError::internal("disk full"))
        }
        async fn get(&self, _: &str, _: &str) -> Result<Option<String>, AgentsError> {
            Ok(None)
        }
        async fn delete_namespace(&self, _: &str) -> Result<(), AgentsError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_failing_store_leaves_the_history_as_it_was() {
        let reducer = OffloadToolOutputs::new(Arc::new(FailingStore), "s");
        let items = history();
        let outcome = reduce(&reducer, &items).await;
        assert_eq!(outcome.items, items);
        assert!(!outcome.report.changed);
    }

    #[tokio::test]
    async fn deleting_a_namespace_removes_only_its_blobs() {
        let store = InMemoryBlobStore::new();
        store.put("a", "k", "1").await.unwrap();
        store.put("b", "k", "2").await.unwrap();
        store.delete_namespace("a").await.unwrap();
        assert_eq!(store.get("a", "k").await.unwrap(), None);
        assert_eq!(store.get("b", "k").await.unwrap().as_deref(), Some("2"));
    }
}
