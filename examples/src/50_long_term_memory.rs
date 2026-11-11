//! Moving context out of the conversation: offloaded tool outputs and long-term facts
//! (Rust-only, offline)
//!
//! Two ways to keep something without keeping it in the prompt:
//!
//! * **Offloaded tool outputs.** `OffloadToolOutputs` stores the full text of an old, bulky tool
//!   output in a `BlobStore` and leaves a preview with a reference. `offload_capability` gives the
//!   model a `read_offloaded` tool to read any part back. Unlike `ToolOutputTrimmer`, nothing is
//!   lost. The reducer and the tool share a *namespace* (here the session id), so one conversation
//!   cannot read another's outputs. `FileBlobStore` (feature `sessions`) keeps them on disk.
//! * **Long-term facts.** `FactMemorySession` extracts short facts from what the *user* says after
//!   each save, and `FactRecall` puts the relevant ones into the instructions of later model
//!   calls, in any conversation of the same scope (here a user id). Tool output is never a source.
//!   `FactMemory` lists, forgets and clears facts, so the user can see and delete what is kept.
//!
//! The models are scripted, so no key is needed. In a real program the extractor is
//! `ModelFactExtractor::new(model)`, which costs one model call per save.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example long_term_memory
//! ```

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{
    offload_capability, reference_for, Agent, CompactingSession, FactMemory, FactMemorySession,
    FactRecall, InMemoryBlobStore, InMemoryFactStore, InMemorySession, ModelFactExtractor,
    ModelOptions, OffloadToolOutputs, RunOptions, Runner, Session,
};
use serde_json::json;

async fn offloading() -> Result<(), Box<dyn std::error::Error>> {
    println!("== offloaded tool outputs ==");
    let blobs = InMemoryBlobStore::shared();
    let store = InMemorySession::shared("chat-1");

    // An earlier part of the conversation had a bulky tool output.
    let log = format!(
        "{}\nERROR code 42 in module billing\n{}",
        "ok\n".repeat(2_000),
        "ok\n".repeat(2_000)
    );
    store
        .add_items(vec![
            json!({"role": "user", "content": "check the logs"}),
            json!({"type": "function_call", "call_id": "c1", "name": "read_log", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "c1", "output": log}),
            json!({"role": "user", "content": "thanks"}),
            json!({"role": "user", "content": "one more thing"}),
        ])
        .await?;

    // Mount the reducer on the stored history: once past the trigger, old outputs move out.
    let session = CompactingSession::with_reducer(
        store.clone(),
        Arc::new(OffloadToolOutputs::new(blobs.clone(), "chat-1").recent_exchanges(2)),
    )
    .trigger_items(1)
    .shared();
    session
        .add_items(vec![json!({"role": "user", "content": "next question"})])
        .await?;

    let note = store.get_items(None).await?[2]["output"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    println!(
        "history now holds:\n{}\n",
        note.lines().take(2).collect::<Vec<_>>().join("\n")
    );

    // The model reads the part it needs. The offset points at the line with the error.
    let offset = log.find("ERROR").unwrap_or(0);
    let args = json!({"ref": reference_for(&log), "offset": offset, "length": 40}).to_string();
    let model = Arc::new(ScriptedModel::new([
        ModelStep::from(ItemHelpers::function_tool_call(
            "read_offloaded",
            &args,
            "r1",
        )),
        ModelStep::from(ItemHelpers::text_message(
            "The log says: error code 42 in billing.",
        )),
    ]));
    let agent = Agent::new("Assistant")
        .model(model.clone())
        .capability(offload_capability(blobs, "chat-1"));
    let result = Runner::run(
        &agent,
        "what was the error?",
        RunOptions::new().session(session),
    )
    .await?;
    println!("tool result the model saw:\n{}", tool_result(&model));
    println!(
        "agent> {}\n",
        result.final_output_as_str().unwrap_or_default()
    );
    Ok(())
}

fn tool_result(model: &ScriptedModel) -> String {
    model.calls()[1]
        .input
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .rev()
                .find(|i| i["type"] == "function_call_output")
        })
        .and_then(|item| item["output"].as_str())
        .unwrap_or_default()
        .to_string()
}

async fn long_term_facts() -> Result<(), Box<dyn std::error::Error>> {
    println!("== long-term facts ==");
    // The extractor model is scripted to answer what a real one would, as JSON.
    let extractor_model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message(
            r#"{"add": ["Prefers answers in Chinese", "Works on a Rust project"]}"#,
        ),
    )]));
    let memory = FactMemory::new(
        InMemoryFactStore::shared(),
        Arc::new(ModelFactExtractor::new(extractor_model)),
        "user-1", // the scope: one person's facts, kept apart from everyone else's
    )
    .shared();

    let with_recall = |session: Arc<dyn Session>| {
        let mut options = RunOptions::new().session(session);
        options.run_config.model = ModelOptions::new().middleware(FactRecall::new(memory.clone()));
        options
    };

    // Conversation one: the user says something worth keeping. The save triggers extraction.
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("Understood."),
    )]));
    let agent = Agent::new("Assistant")
        .instructions("Be helpful.")
        .model(model);
    let first: Arc<dyn Session> =
        FactMemorySession::new(InMemorySession::shared("chat-1"), memory.clone()).shared();
    Runner::run(
        &agent,
        "I work on a Rust project. Please answer in Chinese.",
        with_recall(first),
    )
    .await?;
    println!("remembered:");
    for fact in memory.list().await? {
        println!("  - {}", fact.text);
    }

    // Conversation two, a different session: the facts are in the instructions.
    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("好的"),
    )]));
    let agent = Agent::new("Assistant")
        .instructions("Be helpful.")
        .model(model.clone());
    let second: Arc<dyn Session> = InMemorySession::shared("chat-2");
    Runner::run(&agent, "How do I read a file?", with_recall(second)).await?;
    println!("\ninstructions of the model call in the new conversation:");
    println!(
        "{}",
        model.calls()[0]
            .system_instructions
            .clone()
            .unwrap_or_default()
    );

    // The user can delete what is kept.
    memory.forget_all().await?;
    println!("\nafter forget_all: {} facts", memory.list().await?.len());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    offloading().await?;
    long_term_facts().await
}
