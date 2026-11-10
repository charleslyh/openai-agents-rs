//! Strict JSON schemas: what the API accepts, and what to do when it refuses
//!
//! Structured output (`output_type`) and function tools send a JSON Schema to the provider. In
//! strict mode, OpenAI only accepts schemas of one shape: every object lists *all* its properties
//! in `required` and sets `additionalProperties: false`. A schema derived from a Rust type with
//! `schemars` is not in that shape, so the SDK rewrites it with `ensure_strict_json_schema`.
//!
//! Most of the time this is invisible. It shows when a type cannot be made strict without
//! changing what it accepts: building its schema then fails. This example shows:
//!
//! 1. the rewrite, before and after, on a type with an optional field;
//! 2. a type the strict form cannot express (a free-form map), and the error it gives. It is raised
//!    when the schema is built, before any model call;
//! 3. the ways out: reshape the type, or opt out of strict mode, or hand-write the schema;
//! 4. what validation does with a model reply that does not match.
//!
//! It needs no key and makes no request.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example strict_schema
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

use opai_agents::strict_schema::ensure_strict_json_schema;
use opai_agents::{AgentOutputSchema, AgentOutputSchemaBase, CustomOutputSchema};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// A support ticket summary.
#[derive(Debug, Deserialize, JsonSchema)]
struct Ticket {
    /// One-line summary.
    title: String,
    /// Priority from 1 (low) to 5 (urgent).
    priority: u8,
    /// Who it is assigned to, if anyone.
    assignee: Option<String>,
}

/// Free-form keys: the model could invent any key, which strict mode cannot describe.
#[derive(Debug, Deserialize, JsonSchema)]
struct Labelled {
    labels: HashMap<String, String>,
}

/// The same data with named fields instead of free-form keys.
#[derive(Debug, Deserialize, JsonSchema)]
struct Labels {
    team: String,
    severity: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct LabelledFixed {
    labels: Labels,
}

fn show<T>(title: &str, strict: bool)
where
    T: std::any::Any + serde::de::DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    match AgentOutputSchema::of_with::<T>(strict) {
        Ok(schema) => println!("{title}: ok (strict = {})", schema.is_strict_json_schema()),
        Err(error) => println!("{title}: refused\n    {error}"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 1. The rewrite. ----
    let derived = serde_json::to_value(schemars::schema_for!(Ticket))?;
    let strict = ensure_strict_json_schema(&derived)?;
    println!(
        "-- derived by schemars:\n{}\n",
        serde_json::to_string_pretty(&derived)?
    );
    println!(
        "-- after ensure_strict_json_schema:\n{}\n",
        serde_json::to_string_pretty(&strict)?
    );
    println!(
        "`assignee` was optional; it is now required, and the model says null for \"nobody\".\n"
    );

    // ---- 2. What cannot be made strict. ----
    println!("-- a type the strict form cannot express");
    show::<Labelled>("HashMap<String, String> field", true);

    // ---- 3. The ways out. ----
    println!("\n-- ways out");
    show::<LabelledFixed>("reshape: named fields", true);
    show::<Labelled>("opt out of strict mode", false);

    // Or hand over a schema you wrote yourself, exactly as it should be sent.
    let custom = CustomOutputSchema::new(
        "verdict",
        json!({
            "type": "object",
            "properties": {"verdict": {"type": "string", "enum": ["approve", "reject"]}},
            "required": ["verdict"],
            "additionalProperties": false
        }),
        true,
    );
    println!(
        "hand-written schema: ok (strict = {})",
        custom.is_strict_json_schema()
    );

    // ---- 4. A reply that does not match is an error, not a guess. ----
    println!("\n-- validating model replies against the Ticket schema");
    let ticket = AgentOutputSchema::of::<Ticket>()?;
    for reply in [
        r#"{"title":"Printer on fire","priority":5,"assignee":null}"#,
        r#"{"title":"Printer on fire","priority":"high","assignee":null}"#,
        "not json at all",
    ] {
        match ticket.validate_json(reply) {
            Ok(value) => println!("  accepted: {value}"),
            Err(error) => println!("  rejected: {error}"),
        }
    }
    Ok(())
}
