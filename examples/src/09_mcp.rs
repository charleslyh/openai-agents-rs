//! Local MCP server over stdio
//!
//! Web demo checklist: "Local MCP server".
//!
//! The MCP server is this same executable re-run with `--mcp-server`: a tiny newline-delimited
//! JSON-RPC server with three fixed tools. The client blocks `mcp_admin_reset` with a
//! `ToolFilter`, so the model only sees `mcp_city_facts` and `mcp_convert_temperature`.
//!
//! Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`. Optional: `OPENAI_BASE_URL`, `EXAMPLE_INPUT`.
//!
//! ```bash
//! cargo run -p openai-agents-examples --example mcp
//! ```

mod common;

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use openai_agents::{Agent, McpClient, McpServer, RunOptions, Runner, StdioParams, ToolFilter};
use serde_json::{json, Value};

const SERVER_FLAG: &str = "--mcp-server";

fn tools() -> Value {
    json!([
        {
            "name": "mcp_city_facts",
            "description": "Fixed facts about a city (population, country)",
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        },
        {
            "name": "mcp_convert_temperature",
            "description": "Convert a temperature from Celsius to Fahrenheit",
            "inputSchema": {
                "type": "object",
                "properties": {"celsius": {"type": "number"}},
                "required": ["celsius"]
            }
        },
        {
            "name": "mcp_admin_reset",
            "description": "Reset the server (administrative; the client filters this tool out)",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ])
}

fn call(name: &str, args: &Value) -> Result<String, String> {
    match name {
        "mcp_city_facts" => Ok(
            match args
                .get("city")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str()
            {
                "paris" => "Paris, France: about 2.1 million people".to_string(),
                "tokyo" => "Tokyo, Japan: about 14 million people".to_string(),
                other => format!(
                    "No facts on file for {}",
                    other.chars().take(40).collect::<String>()
                ),
            },
        ),
        "mcp_convert_temperature" => {
            let c = args
                .get("celsius")
                .and_then(Value::as_f64)
                .ok_or("celsius must be a number")?;
            Ok(format!("{c} °C = {:.1} °F", c * 9.0 / 5.0 + 32.0))
        }
        "mcp_admin_reset" => Ok("reset done".to_string()),
        _ => Err(format!("unknown tool {name}")),
    }
}

fn handle(message: &Value) -> Option<Value> {
    let id = message.get("id")?.clone();
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => json!({
            "protocolVersion": params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or("2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "example-mcp", "version": "0.1.0"}
        }),
        "ping" => json!({}),
        "tools/list" => json!({"tools": tools()}),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(Value::Null);
            match call(name, &args) {
                Ok(text) => json!({"content": [{"type": "text", "text": text}], "isError": false}),
                Err(e) => json!({"content": [{"type": "text", "text": e}], "isError": true}),
            }
        }
        _ => {
            return Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": "Method not found"}
            }))
        }
    };
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

/// Serve until stdin closes. Nothing but protocol messages may reach stdout.
fn serve() -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let Ok(message) = serde_json::from_str::<Value>(&line?) else {
            continue;
        };
        if let Some(reply) = handle(&message) {
            writeln!(stdout, "{reply}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Server mode first: the child process has no API key and must not read the model env.
    if std::env::args().nth(1).as_deref() == Some(SERVER_FLAG) {
        serve()?;
        return Ok(());
    }

    let model = common::model()?;
    let exe = std::env::current_exe()?;
    let server: Arc<dyn McpServer> =
        McpClient::stdio(StdioParams::new(exe.to_string_lossy()).args([SERVER_FLAG]))
            .tool_filter(ToolFilter::block(["mcp_admin_reset"]))
            .shared();

    let agent = Agent::new("Assistant")
        .instructions("Use the MCP tools to answer questions about cities and temperatures.")
        .model(model)
        .mcp_servers(vec![Arc::clone(&server)]);

    let input = common::input_with_fallback(
        "question> ",
        "What do you know about Paris? Also convert 20 celsius to fahrenheit.",
    )?;
    let outcome = Runner::run(&agent, input, RunOptions::default()).await;
    let _ = server.cleanup().await;
    println!("{}", outcome?.final_output_as_str().unwrap_or_default());
    Ok(())
}
