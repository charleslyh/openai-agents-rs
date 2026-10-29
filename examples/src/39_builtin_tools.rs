//! Ready-made tools and the limits built into them (Rust-only)
//!
//! `opai_agents::tools` has the four tools every project writes first: `calculator`, `clock`,
//! `file_read` / `file_list` and `web_fetch`. The first two are harmless. The last two can reach
//! something, so their limits are part of the tool rather than a setting you may forget:
//!
//! * `file_read(dir)` only reads inside `dir`: `..`, absolute paths and symlinks that leave it are
//!   refused.
//! * `web_fetch()` only talks `https`, and refuses loopback, private, link-local and cloud
//!   metadata addresses, including after a redirect.
//!
//! The model here is scripted to ask for a mix of allowed and forbidden calls in one turn, so the
//! example shows both the answers and the refusals without a key or a network. A refusal comes
//! back to the model as a generic tool failure, which is how a real model would learn it asked
//! for something it may not have (the reason stays on your side; the example prints it below).
//!
//! With `OPENAI_API_KEY` set, `LIVE=1` hands the same tools to a real model instead and lets it
//! decide what to call:
//!
//! ```bash
//! cargo run -p opai-agents-examples --example builtin_tools
//! LIVE=1 EXAMPLE_INPUT="What is 12.5% of 480, and what time is it in +09:00?" \
//!   cargo run -p opai-agents-examples --example builtin_tools
//! ```

mod common;

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::tools::{calculator, clock, file_list, file_read, web_fetch};
use opai_agents::{Agent, Model, RunItem, RunOptions, Runner};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A directory the agent is allowed to read, with one file in it.
    let dir = std::env::temp_dir().join("agents-builtin-tools");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("notes.txt"), "Remember to water the plants.\n")?;

    let tools = vec![
        calculator(),
        clock(),
        file_list(&dir)?,
        file_read(&dir)?,
        web_fetch(),
    ];

    if std::env::var_os("LIVE").is_some() {
        let agent = Agent::new("Assistant")
            .instructions("Use the tools to answer. Be brief.")
            .model(common::model()?)
            .tools(tools);
        let input = common::example_input("What is 12.5% of 480, and what time is it in +09:00?");
        let result = Runner::run(&agent, input, RunOptions::new()).await?;
        println!("{}", result.final_output_as_str().unwrap_or_default());
        return Ok(());
    }

    // Each entry: the call the "model" makes, and why it is interesting.
    let calls = [
        (
            "calculator",
            json!({"expression": "(12 + 3) * 4"}),
            "allowed",
        ),
        ("clock", json!({"offset": "+09:00"}), "allowed"),
        (
            "file_read",
            json!({"path": "notes.txt"}),
            "inside its directory",
        ),
        (
            "file_read",
            json!({"path": "../../etc/passwd"}),
            "escapes with ..",
        ),
        ("file_read", json!({"path": "/etc/passwd"}), "absolute path"),
        (
            "web_fetch",
            json!({"url": "http://169.254.169.254/latest/meta-data/"}),
            "cloud metadata address",
        ),
        (
            "web_fetch",
            json!({"url": "https://localhost:8080/admin"}),
            "loopback",
        ),
    ];
    let items: Vec<_> = calls
        .iter()
        .enumerate()
        .map(|(i, (name, args, _))| {
            ItemHelpers::function_tool_call(*name, args.to_string(), format!("call-{i}"))
        })
        .collect();
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::new([
        ModelStep::from(items),
        ModelStep::from(ItemHelpers::text_message("Done.")),
    ]));
    let agent = Agent::new("Assistant").model(model).tools(tools);

    let result = Runner::run(&agent, "Try everything.", RunOptions::new()).await?;

    let outputs: Vec<_> = result
        .new_items
        .iter()
        .filter_map(|item| match item {
            RunItem::ToolCallOutput(o) => Some(&o.output),
            _ => None,
        })
        .collect();
    for ((name, args, why), output) in calls.iter().zip(outputs) {
        println!("{name}({args})  [{why}]\n  -> {output}\n");
    }

    // The model only sees a generic failure for a refused call, so the reason does not leak.
    // The reasons are available to you: the path check is a public function...
    println!("why the file reads were refused:");
    for path in ["../../etc/passwd", "/etc/passwd"] {
        if let Err(error) = file_read::resolve(&dir, path) {
            println!("  {path}: {error}");
        }
    }
    // ...and so is the address check behind `web_fetch`.
    println!("\nwhich addresses web_fetch will reach:");
    for ip in ["93.184.216.34", "169.254.169.254", "127.0.0.1", "10.0.0.5"] {
        let reachable = web_fetch::is_reachable(ip.parse()?);
        println!("  {ip}: {}", if reachable { "yes" } else { "refused" });
    }
    Ok(())
}
