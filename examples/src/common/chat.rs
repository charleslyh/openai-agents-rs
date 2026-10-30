//! One chat turn against a session, shared by the session examples.
//!
//! The examples differ in *which* session they use and *how many*; the turn itself (build an
//! agent, run it with the session, print the exchange) is the same, so it lives here.

use std::sync::Arc;

use opai_agents::{Agent, RunOptions, Runner, Session};
use serde_json::Value;

/// A fresh agent. Nothing about it is stored: the conversation lives in the session, so a new
/// agent given the same session continues the same conversation.
pub fn agent() -> Result<Agent, Box<dyn std::error::Error>> {
    Ok(Agent::new("Assistant")
        .instructions("Answer in one short sentence, using the conversation so far.")
        .model(super::model_or_offline()?))
}

/// Send `question` through a new agent that uses `session`, and print both sides.
///
/// The session supplies the earlier turns as model input and receives this turn when the run ends.
pub async fn ask(
    session: &Arc<dyn Session>,
    label: &str,
    question: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("[{label}] user> {question}");
    let result = Runner::run(
        &agent()?,
        question,
        RunOptions::new().session(Arc::clone(session)),
    )
    .await?;
    println!(
        "[{label}] agent> {}",
        result.final_output_as_str().unwrap_or_default()
    );
    Ok(())
}

/// Print what a session holds: one line per stored item.
///
/// Reading the history back is how to see that a conversation was restored even when the model
/// is the offline one and its replies say nothing about memory.
pub async fn show_history(
    session: &Arc<dyn Session>,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let items = session.get_items(None).await?;
    println!("[{label}] {} stored items:", items.len());
    for item in &items {
        let (role, text) = describe(item);
        let text: String = text.chars().take(60).collect();
        println!("    {role:<9} {text}");
    }
    Ok(())
}

/// The role (or item type) and the text of one stored item.
fn describe(item: &Value) -> (String, String) {
    let role = item["role"]
        .as_str()
        .or_else(|| item["type"].as_str())
        .unwrap_or("?")
        .to_string();
    let text = match &item["content"] {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    (role, text)
}
