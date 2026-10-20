//! Strict JSON Schema normalization (Python: `agents.strict_schema`).
//!
//! OpenAI's structured outputs require schemas where every object declares
//! `additionalProperties: false` and lists all of its properties in `required`.
//! [`ensure_strict_json_schema`] rewrites a schema produced by `schemars` into that
//! shape. The implementation mirrors `openai-agents-python` v0.23.1
//! (`src/agents/strict_schema.py`) including its resource guards.

mod refs;
#[cfg(test)]
mod tests;

use serde_json::{Map, Value};

use crate::error::{AgentsError, UserError};

use refs::{
    is_bare_ref, merge_ref_target, merge_single_all_of, resolve_ref, REF_NON_CONSTRAINING_SIBLINGS,
};

/// Upper bound on schema nodes expanded during conversion.
///
/// Real schemas are far smaller; the limit only trips on pathological input such as a
/// `$ref` fan-out that would otherwise expand exponentially (a denial-of-service vector
/// for schemas advertised by a third party).
const MAX_SCHEMA_NODES: u32 = 100_000;

/// Keep recursion comfortably below the stack limit. Counts nested maps and arrays.
const MAX_SCHEMA_DEPTH: u32 = 100;

const ADDITIONAL_PROPERTIES_ERROR: &str =
    "additionalProperties should not be set for object types. \
     Use a plain object schema, or build the tool without a strict schema.";

const SCHEMA_DEPTH_ERROR: &str =
    "JSON schema is too deeply nested to process safely. Simplify or flatten the schema.";

const SCHEMA_BUDGET_ERROR: &str =
    "JSON schema is too large to convert to a strict schema. This can\
     happen when a schema expands `$ref`s exponentially, which may indicate a malformed or\
     malicious schema.";

/// Convert `schema` into the `strict` form expected by the OpenAI API.
///
/// Returns a new schema; the input is not mutated. Fails with [`UserError`] when the
/// schema cannot be made strict without changing the values it accepts.
///
/// ```rust
/// use opai_agents_core::strict_schema::ensure_strict_json_schema;
/// use serde_json::json;
///
/// let strict = ensure_strict_json_schema(&json!({
///     "type": "object",
///     "properties": { "a": { "type": "string" } }
/// })).unwrap();
/// assert_eq!(strict["additionalProperties"], json!(false));
/// assert_eq!(strict["required"], json!(["a"]));
/// ```
pub fn ensure_strict_json_schema(schema: &Value) -> Result<Value, AgentsError> {
    validate_json_schema_depth(schema)?;

    let Some(object) = schema.as_object() else {
        return Err(UserError::new(format!("Expected a JSON object schema, got {schema}")).into());
    };

    if object.is_empty() {
        return Ok(empty_schema());
    }

    let mut budget = MAX_SCHEMA_NODES;
    let converted = ensure_strict_inner(schema.clone(), schema, &mut budget, 1, false)?;
    ensure_strict_root(converted)
}

/// The strict form of an empty schema (`{}`).
pub fn empty_schema() -> Value {
    let mut out = Map::new();
    out.insert("type".into(), Value::String("object".into()));
    out.insert("properties".into(), Value::Object(Map::new()));
    out.insert("required".into(), Value::Array(Vec::new()));
    out.insert("additionalProperties".into(), Value::Bool(false));
    Value::Object(out)
}

/// Reject container nesting that is unsafe for recursive schema processing.
fn validate_json_schema_depth(schema: &Value) -> Result<(), AgentsError> {
    let mut stack: Vec<(&Value, u32)> = vec![(schema, 1)];
    while let Some((value, depth)) = stack.pop() {
        if depth > MAX_SCHEMA_DEPTH {
            return Err(UserError::new(SCHEMA_DEPTH_ERROR).into());
        }
        match value {
            Value::Object(map) => {
                stack.extend(
                    map.values()
                        .filter(|v| matches!(v, Value::Object(_) | Value::Array(_)))
                        .map(|v| (v, depth + 1)),
                );
            }
            Value::Array(items) => {
                stack.extend(
                    items
                        .iter()
                        .filter(|v| matches!(v, Value::Object(_) | Value::Array(_)))
                        .map(|v| (v, depth + 1)),
                );
            }
            _ => {}
        }
    }
    Ok(())
}

fn spend(budget: &mut u32) -> Result<(), AgentsError> {
    *budget = budget.saturating_sub(1);
    if *budget == 0 {
        return Err(UserError::new(SCHEMA_BUDGET_ERROR).into());
    }
    Ok(())
}

fn ensure_strict_root(mut schema: Value) -> Result<Value, AgentsError> {
    if schema.get("anyOf").map(Value::is_array).unwrap_or(false) {
        return Err(
            UserError::new("The root of a strict JSON schema must not use `anyOf`.").into(),
        );
    }
    if let Some(Value::Array(types)) = schema.get("type") {
        if types.iter().any(|t| t == "object") {
            if types.len() == 1 {
                schema["type"] = Value::String("object".into());
            } else {
                return Err(UserError::new(format!(
                    "The root of a strict JSON schema must be a non-nullable object, but its type is \
                     {types:?}. Make the root a plain object, or build the tool without a strict schema."
                ))
                .into());
            }
        }
    }
    Ok(schema)
}

fn ensure_strict_inner(
    schema: Value,
    root: &Value,
    budget: &mut u32,
    depth: u32,
    inside_nested_resource: bool,
) -> Result<Value, AgentsError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(UserError::new(SCHEMA_DEPTH_ERROR).into());
    }
    let Value::Object(mut node) = schema else {
        return Err(UserError::new(format!("Expected a JSON object schema, got {schema}")).into());
    };

    // Remember an author-supplied closure before normalization adds its own.
    let explicitly_closed = node.get("additionalProperties") == Some(&Value::Bool(false));
    spend(budget)?;

    let inside_nested_resource = inside_nested_resource || declares_schema_resource(&node);

    if node.contains_key("$ref") {
        if inside_nested_resource {
            return Err(UserError::new(
                "JSON schema contains a reference owned by or crossing a nested `$id` resource that \
                 cannot be resolved against the document root.",
            )
            .into());
        }
        let allowed = REF_NON_CONSTRAINING_SIBLINGS;
        let mut incompatible: Vec<&String> = node
            .keys()
            .filter(|k| k.as_str() != "$ref" && !allowed.contains(&k.as_str()))
            .collect();
        incompatible.sort();
        if !incompatible.is_empty() {
            let joined = incompatible
                .iter()
                .map(|k| k.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(UserError::new(format!(
                "JSON schema contains a `$ref` with incompatible sibling keyword(s) ({joined}) that \
                 cannot be merged without changing its accepted values."
            ))
            .into());
        }
    }

    for defs_key in ["$defs", "definitions"] {
        if let Some(Value::Object(defs)) = node.get(defs_key).cloned() {
            let mut rewritten = Map::new();
            for (name, def) in defs {
                rewritten.insert(
                    name,
                    ensure_strict_inner(def, root, budget, depth + 1, inside_nested_resource)?,
                );
            }
            node.insert(defs_key.to_string(), Value::Object(rewritten));
        }
    }

    let typ = node.get("type").cloned();
    if typ.is_none()
        && node
            .get("properties")
            .map(Value::is_object)
            .unwrap_or(false)
    {
        node.insert("type".into(), Value::String("object".into()));
    } else if typ.is_none() {
        // Matches Python's `get("additionalProperties", False) is not False`: an absent key is
        // fine, but an explicit non-`false` value (including `{}`, which means "allow anything")
        // leaves a schema that cannot be made strict.
        match node.get("additionalProperties") {
            None | Some(Value::Bool(false)) => {}
            Some(_) => return Err(UserError::new(ADDITIONAL_PROPERTIES_ERROR).into()),
        }
    }

    let is_object = is_object_type(node.get("type"));
    if is_object && !node.contains_key("additionalProperties") {
        node.insert("additionalProperties".into(), Value::Bool(false));
    } else if is_object && node.get("additionalProperties") != Some(&Value::Bool(false)) {
        return Err(UserError::new(ADDITIONAL_PROPERTIES_ERROR).into());
    }

    if let Some(Value::Object(properties)) = node.get("properties").cloned() {
        let required: Vec<Value> = properties
            .keys()
            .map(|k| Value::String(k.clone()))
            .collect();
        let mut rewritten = Map::new();
        for (key, prop) in properties {
            rewritten.insert(
                key,
                ensure_strict_inner(prop, root, budget, depth + 1, inside_nested_resource)?,
            );
        }
        node.insert("properties".into(), Value::Object(rewritten));
        node.insert("required".into(), Value::Array(required));
    }

    if let Some(items) = node.get("items").cloned() {
        if items.is_object() {
            node.insert(
                "items".into(),
                ensure_strict_inner(items, root, budget, depth + 1, inside_nested_resource)?,
            );
        }
    }

    if let Some(Value::Array(any_of)) = node.get("anyOf").cloned() {
        let mut rewritten = Vec::with_capacity(any_of.len());
        for variant in any_of {
            rewritten.push(ensure_strict_inner(
                variant,
                root,
                budget,
                depth + 1,
                inside_nested_resource,
            )?);
        }
        node.insert("anyOf".into(), Value::Array(rewritten));
    }

    // `oneOf` is not supported by structured outputs in nested contexts, so convert it to
    // `anyOf`, which is equivalent for discriminated unions.
    if let Some(Value::Array(one_of)) = node.get("oneOf").cloned() {
        let mut existing = match node.get("anyOf") {
            Some(Value::Array(v)) => v.clone(),
            _ => Vec::new(),
        };
        for variant in one_of {
            existing.push(ensure_strict_inner(
                variant,
                root,
                budget,
                depth + 1,
                inside_nested_resource,
            )?);
        }
        node.insert("anyOf".into(), Value::Array(existing));
        node.remove("oneOf");
    }

    if let Some(Value::Array(all_of)) = node.get("allOf").cloned() {
        if all_of.len() == 1 {
            let entry = all_of.into_iter().next().unwrap_or(Value::Null);
            let strict_entry = if is_bare_ref(&entry) {
                if inside_nested_resource {
                    return Err(UserError::new(
                        "JSON schema contains a reference owned by or crossing a nested `$id` \
                         resource that cannot be resolved against the document root.",
                    )
                    .into());
                }
                spend(budget)?;
                let mut siblings = Map::new();
                if let Value::Object(map) = &entry {
                    for (k, v) in map {
                        if k != "$ref" {
                            siblings.insert(k.clone(), v.clone());
                        }
                    }
                }
                let target = resolve_ref(
                    root,
                    entry.get("$ref").and_then(Value::as_str).unwrap_or(""),
                )?;
                merge_ref_target(target, siblings)
            } else {
                ensure_strict_inner(entry, root, budget, depth + 1, inside_nested_resource)?
            };
            node.remove("allOf");
            if !explicitly_closed {
                node.remove("additionalProperties");
            }
            let merged = merge_single_all_of(strict_entry, &node)?;
            return ensure_strict_inner(
                Value::Object(merged),
                root,
                budget,
                depth + 1,
                inside_nested_resource,
            );
        }
        let mut rewritten = Vec::with_capacity(all_of.len());
        for entry in all_of {
            rewritten.push(ensure_strict_inner(
                entry,
                root,
                budget,
                depth + 1,
                inside_nested_resource,
            )?);
        }
        node.insert("allOf".into(), Value::Array(rewritten));
    }

    // `null` defaults are not meaningful for structured outputs.
    if node.get("default") == Some(&Value::Null) {
        node.remove("default");
    }

    // A `$ref` cannot be combined with sibling constraints, so unravel it into place.
    if let Some(Value::String(ref_str)) = node.get("$ref").cloned() {
        if node.len() > 1 {
            let resolved = resolve_ref(root, &ref_str)?;
            let Value::Object(resolved_map) = resolved else {
                return Err(UserError::new(format!(
                    "Expected `$ref: {ref_str}` to resolve to an object"
                ))
                .into());
            };
            node.remove("$ref");
            let mut merged = resolved_map;
            for (k, v) in node {
                merged.insert(k, v);
            }
            return ensure_strict_inner(
                Value::Object(merged),
                root,
                budget,
                depth + 1,
                inside_nested_resource,
            );
        }
    }

    Ok(Value::Object(node))
}

fn is_object_type(typ: Option<&Value>) -> bool {
    match typ {
        Some(Value::String(s)) => s == "object",
        Some(Value::Array(arr)) => arr.iter().any(|v| v == "object"),
        _ => false,
    }
}

pub(super) fn declares_schema_resource(node: &Map<String, Value>) -> bool {
    matches!(node.get("$id"), Some(Value::String(_)))
}
