//! Testing helpers (Python: `agents.testing`).
//!
//! The surface mirrors `agents/testing/__init__.py`: a deterministic `ScriptedModel` plus item
//! builders. There is no HTTP mock layer here — HTTP-level assertions live in the crate's own
//! tests, which drive `wiremock` directly.

use serde_json::{json, Value};

pub use crate::items::ItemHelpers;
pub use crate::model::scripted::{
    ModelCall, ModelScriptError, ModelStep, ModelStepReason, ScriptedModel,
};

/// Build one normalized assistant text output item (Python: `assistant_message`).
pub fn assistant_message(text: impl Into<String>) -> Value {
    assistant_message_with_id(text, "scripted-message")
}

/// Build one normalized assistant text output item with an explicit item id.
///
/// Python: `assistant_message(text, item_id=...)`.
pub fn assistant_message_with_id(text: impl Into<String>, item_id: impl Into<String>) -> Value {
    json!({
        "id": item_id.into(),
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{
            "type": "output_text",
            "text": text.into(),
            "annotations": [],
            "logprobs": []
        }]
    })
}

/// Build one normalized function-tool call output item (Python: `function_call`).
///
/// `arguments` may be a JSON string or any serializable value.
pub fn function_call(
    name: impl Into<String>,
    arguments: Value,
    call_id: impl Into<String>,
) -> Value {
    let call_id = call_id.into();
    function_call_with_id(name, arguments, call_id.clone(), call_id)
}

/// Build one normalized function-tool call output item with an explicit item id.
///
/// Python: `function_call(name, arguments, call_id=..., item_id=..., namespace=...)`.
pub fn function_call_with_id(
    name: impl Into<String>,
    arguments: Value,
    call_id: impl Into<String>,
    item_id: impl Into<String>,
) -> Value {
    let arguments = match arguments {
        Value::String(s) => s,
        other => other.to_string(),
    };
    json!({
        "id": item_id.into(),
        "call_id": call_id.into(),
        "type": "function_call",
        "name": name.into(),
        "arguments": arguments
    })
}
