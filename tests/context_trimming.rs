//! Context trimming filters: `ToolOutputTrimmer` (checked against Python's) and
//! `ContextWindowTrimmer` (new, provider-neutral).

use std::sync::Arc;

use openai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use openai_agents::{
    chain_input_filters, Agent, ContextWindowTrimmer, RunOptions, Runner, ToolOutputTrimmer,
};
use serde_json::{json, Value};

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn call(name: &str, id: &str) -> Value {
    json!({"type": "function_call", "name": name, "arguments": "{}", "call_id": id})
}

fn output(id: &str, text: &str) -> Value {
    json!({"type": "function_call_output", "call_id": id, "output": text})
}

fn assistant(text: &str) -> Value {
    json!({"type": "message", "role": "assistant",
           "content": [{"type": "output_text", "text": text, "annotations": []}]})
}

/// Items covering every branch: big and small outputs, two tools, an unknown call id, non-ASCII
/// text, and enough user messages for a recent boundary.
fn sample_conversation() -> Vec<Value> {
    vec![
        user("first"),
        call("search", "c1"),
        output("c1", &"x".repeat(300)),
        call("calc", "c2"),
        output("c2", &"y".repeat(300)),
        call("search", "c3"),
        output("c3", "short"),
        output("orphan", &"z".repeat(300)),
        call("search", "c4"),
        output("c4", &"\u{4f60}\u{597d}".repeat(150)),
        assistant("done"),
        user("second"),
        call("search", "c5"),
        output("c5", &"a".repeat(300)),
        user("third"),
        call("search", "c6"),
        output("c6", &"b".repeat(300)),
    ]
}

#[test]
fn tool_output_trimmer_keeps_recent_turns_and_shrinks_old_outputs() {
    let items = sample_conversation();
    let trimmed = ToolOutputTrimmer::new()
        .recent_turns(2)
        .max_output_chars(100)
        .preview_chars(40)
        .trim(&items);
    assert_eq!(trimmed.len(), items.len());
    // Old outputs shrink; the originals are untouched.
    assert!(trimmed[2]["output"]
        .as_str()
        .unwrap()
        .starts_with("[Trimmed: search output"));
    assert_eq!(items[2]["output"].as_str().unwrap().len(), 300);
    assert_eq!(trimmed[6]["output"], "short", "below the limit");
    // The last two user messages and everything after stay intact.
    assert_eq!(&trimmed[11..], &items[11..]);
}

#[test]
fn tool_output_trimmer_respects_trimmable_tools_and_validates() {
    let items = sample_conversation();
    let trimmed = ToolOutputTrimmer::new()
        .max_output_chars(100)
        .trimmable_tools(["calc"])
        .trim(&items);
    assert_eq!(trimmed[2], items[2], "search is not trimmable");
    assert!(trimmed[4]["output"]
        .as_str()
        .unwrap()
        .starts_with("[Trimmed: calc output"));

    assert!(ToolOutputTrimmer::new().recent_turns(0).validate().is_err());
    assert!(ToolOutputTrimmer::new()
        .max_output_chars(0)
        .validate()
        .is_err());
    // Fewer user messages than `recent_turns`: nothing is old.
    assert_eq!(
        ToolOutputTrimmer::new()
            .recent_turns(9)
            .max_output_chars(1)
            .trim(&items),
        items
    );
}

const PYTHON_TRIMMER: &str = r#"
import json, sys
from agents.extensions import ToolOutputTrimmer
from agents.run_config import CallModelData, ModelInputData
cfg = json.loads(sys.argv[1])
items = json.loads(sys.stdin.read())
trimmer = ToolOutputTrimmer(**cfg)
data = CallModelData(model_data=ModelInputData(input=items, instructions=None), agent=None, context=None)
out = trimmer(data)
print(json.dumps(out.input, ensure_ascii=False))
"#;

/// Rust and Python produce identical trimmed lists for the same input and settings. Skipped
/// when the Python oracle venv is not installed (`scripts/setup_venv.sh`).
#[test]
fn tool_output_trimmer_matches_python() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".venv/bin/python");
    if !python.exists() {
        eprintln!("skipping: no .venv (run scripts/setup_venv.sh)");
        return;
    }
    let items = sample_conversation();
    let cases = [
        (
            json!({"recent_turns": 2, "max_output_chars": 100, "preview_chars": 40}),
            ToolOutputTrimmer::new()
                .recent_turns(2)
                .max_output_chars(100)
                .preview_chars(40),
        ),
        (
            json!({"recent_turns": 1, "max_output_chars": 50, "preview_chars": 10,
                   "trimmable_tools": ["search"]}),
            ToolOutputTrimmer::new()
                .recent_turns(1)
                .max_output_chars(50)
                .preview_chars(10)
                .trimmable_tools(["search"]),
        ),
        (
            json!({"recent_turns": 3, "max_output_chars": 100, "preview_chars": 200}),
            ToolOutputTrimmer::new()
                .recent_turns(3)
                .max_output_chars(100)
                .preview_chars(200),
        ),
    ];
    for (config, trimmer) in cases {
        let mut child = Command::new(&python)
            .args(["-c", PYTHON_TRIMMER, &config.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("python");
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(Value::Array(items.clone()).to_string().as_bytes())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let expected: Value = serde_json::from_slice(&out.stdout).expect("python json");
        assert_eq!(
            Value::Array(trimmer.trim(&items)),
            expected,
            "config {config}"
        );
    }
}

fn user_texts(items: &[Value]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i["role"] == "user")
        .map(|i| i["content"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn window_trimmer_drops_oldest_turns_whole() {
    let items = vec![
        json!({"role": "system", "content": "be brief"}),
        user("t1"),
        call("search", "c1"),
        output("c1", "r1"),
        assistant("a1"),
        user("t2"),
        assistant("a2"),
        user("t3"),
        call("search", "c3"),
        output("c3", "r3"),
        assistant("a3"),
    ];
    // Turn limit: the system message plus the latest two turns.
    let by_turns = ContextWindowTrimmer::new().max_turns(2).trim(&items, 0);
    assert_eq!(user_texts(&by_turns), ["t2", "t3"]);
    assert_eq!(by_turns[0]["role"], "system");
    assert_eq!(by_turns.len(), items.len() - 4, "turn 1 left as a unit");

    // One token per item: pinned (1) + latest turn (4) leaves room for the 2-item turn only.
    let by_tokens = ContextWindowTrimmer::new()
        .max_tokens(7)
        .token_counter(|_| 1)
        .trim(&items, 0);
    assert_eq!(user_texts(&by_tokens), ["t2", "t3"]);
    // Reserved tokens (instructions) shrink the room.
    let tight = ContextWindowTrimmer::new()
        .max_tokens(7)
        .token_counter(|_| 1)
        .trim(&items, 2);
    assert_eq!(user_texts(&tight), ["t3"]);
    // The latest turn is kept even when it alone exceeds the budget.
    let tiny = ContextWindowTrimmer::new()
        .max_tokens(1)
        .token_counter(|_| 1)
        .trim(&items, 0);
    assert_eq!(user_texts(&tiny), ["t3"]);
    assert!(tiny.iter().any(|i| i["call_id"] == "c3"));
    // No limits, no change.
    assert_eq!(ContextWindowTrimmer::new().trim(&items, 0), items);
    assert!(ContextWindowTrimmer::new()
        .max_turns(1)
        .trim(&[], 0)
        .is_empty());
}

#[tokio::test]
async fn filters_apply_to_the_model_call_only_and_chain() {
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call("echo", "{}", "c1")),
        ModelStep::from(ItemHelpers::text_message("done")),
    ]));
    let agent =
        Agent::new("a")
            .model(model.clone())
            .tools(vec![openai_agents::FunctionTool::constant(
                "echo",
                "e",
                "q".repeat(400),
            )]);
    let filter = chain_input_filters(vec![
        ToolOutputTrimmer::new()
            .recent_turns(1)
            .max_output_chars(10)
            .into_filter(),
        ContextWindowTrimmer::new().max_turns(1).into_filter(),
    ]);
    let mut options = RunOptions::default();
    options.run_config.call_model_input_filter = Some(filter);

    let input = vec![user("old question"), assistant("old answer"), user("new")];
    let result = Runner::run(&agent, input, options).await.expect("run");
    let second = model.calls()[1].input.clone();
    // The old turn is gone from the second call; the new turn and its tool exchange remain...
    assert_eq!(user_texts(second.as_array().unwrap()), ["new"]);
    assert!(second.to_string().contains("function_call_output"));
    // ...but the run's own history keeps everything.
    assert_eq!(
        result
            .to_input_list()
            .iter()
            .filter(|i| i["role"] == "user")
            .count(),
        2
    );
}
