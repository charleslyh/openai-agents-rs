//! The Chat Completions converters give the same results as Python's `chatcmpl_converter`.
//!
//! `tests/parity/chat_convert_cases.json` holds shared inputs; `scripts/chat_convert_oracle.py`
//! runs them through the Python SDK. Skipped when the oracle venv is missing
//! (`scripts/setup_venv.sh`).
#![cfg(feature = "openai")]

use std::path::PathBuf;
use std::process::Command;

use openai_agents::model::openai::chat_convert::{
    chat_message_to_output_items_with, items_to_chat_messages, ChatConvertOptions,
};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn cases() -> Value {
    let text =
        std::fs::read_to_string(root().join("tests/parity/chat_convert_cases.json")).unwrap();
    serde_json::from_str(&text).expect("cases json")
}

fn python_results() -> Option<Value> {
    let python = root().join(".venv/bin/python");
    if !python.exists() {
        eprintln!("skipping: no .venv (run scripts/setup_venv.sh)");
        return None;
    }
    let out = Command::new(python)
        .arg(root().join("scripts/chat_convert_oracle.py"))
        .arg(root().join("tests/parity/chat_convert_cases.json"))
        .output()
        .expect("python");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The SDK may print warnings before the JSON line; the JSON is the last line.
    let stdout = String::from_utf8_lossy(&out.stdout);
    let last = stdout.lines().last().expect("oracle output");
    Some(serde_json::from_str(last).expect("oracle json"))
}

fn is_error(value: &Value) -> bool {
    value.get("error").is_some()
}

#[test]
fn items_to_messages_match_python() {
    let Some(python) = python_results() else {
        return;
    };
    let mut failures = Vec::new();
    for case in cases()["items_to_messages"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let items = case["items"].as_array().unwrap();
        let options = ChatConvertOptions {
            preserve_thinking_blocks: case["preserve_thinking_blocks"].as_bool().unwrap_or(false),
            model: case["model"].as_str().unwrap_or("").to_string(),
            replay_reasoning: None,
        };
        let expected = &python["items_to_messages"][name];
        match items_to_chat_messages(items, &options) {
            Ok(messages) if is_error(expected) => failures.push(format!(
                "{name}: python raised {expected}, rust returned {messages:?}"
            )),
            Ok(messages) => {
                if Value::Array(messages.clone()) != *expected {
                    failures.push(format!(
                        "{name}:\n  python {expected}\n  rust   {}",
                        Value::Array(messages)
                    ));
                }
            }
            Err(error) if !is_error(expected) => failures.push(format!(
                "{name}: rust failed ({error}), python gave {expected}"
            )),
            Err(_) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn reply_messages_match_python() {
    let Some(python) = python_results() else {
        return;
    };
    let mut failures = Vec::new();
    for case in cases()["message_to_output_items"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let expected = &python["message_to_output_items"][name];
        let actual = Value::Array(chat_message_to_output_items_with(
            &case["message"],
            case.get("provider_data"),
        ));
        if actual != *expected {
            failures.push(format!("{name}:\n  python {expected}\n  rust   {actual}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
