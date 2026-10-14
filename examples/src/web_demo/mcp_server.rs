//! A tiny MCP server over stdio (newline-delimited JSON-RPC), run by this same executable when it
//! is started with `--mcp-server`. It exposes fixed, deterministic tools and does nothing else:
//! no files, no network, no commands.

use std::io::{self, BufRead, Write};

use serde_json::{json, Value};

/// The only argument that switches the executable into server mode.
pub const MCP_SERVER_FLAG: &str = "--mcp-server";

const SUPPORTED: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

fn tools() -> Value {
    json!([
        {
            "name": "mcp_city_facts",
            "description": "Fixed facts about a city (population, country), served by the demo MCP server",
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
            "description": "Reset the demo server (administrative; the client filters this tool out)",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ])
}

fn city_facts(city: &str) -> String {
    match city.to_ascii_lowercase().as_str() {
        "paris" => "Paris, France: about 2.1 million people".to_string(),
        "tokyo" => "Tokyo, Japan: about 14 million people".to_string(),
        "london" => "London, United Kingdom: about 8.9 million people".to_string(),
        "oakland" => "Oakland, United States: about 440 thousand people".to_string(),
        other => format!(
            "No facts on file for {}",
            other.chars().take(40).collect::<String>()
        ),
    }
}

fn call(name: &str, args: &Value) -> Result<String, String> {
    match name {
        "mcp_city_facts" => {
            let city = args.get("city").and_then(Value::as_str).unwrap_or("");
            Ok(city_facts(city))
        }
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
        "initialize" => {
            let wanted = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or("");
            let version = SUPPORTED
                .iter()
                .find(|v| **v == wanted)
                .unwrap_or(&SUPPORTED[0]);
            json!({
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "demo-mcp", "version": "0.1.0"}
            })
        }
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
pub fn run() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if let Some(reply) = handle(&message) {
            writeln!(stdout, "{reply}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_and_tools() {
        let init = handle(&json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}))
            .expect("reply");
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert!(handle(&json!({"jsonrpc":"2.0","method":"notifications/initialized"})).is_none());
        let list = handle(&json!({"jsonrpc":"2.0","id":2,"method":"tools/list"})).expect("reply");
        assert_eq!(list["result"]["tools"].as_array().map(Vec::len), Some(3));
        let call = handle(&json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"mcp_convert_temperature","arguments":{"celsius":100}}}))
            .expect("reply");
        assert_eq!(call["result"]["content"][0]["text"], "100 °C = 212.0 °F");
    }
}
