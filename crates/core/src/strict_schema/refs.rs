//! `$ref` handling for strict schema conversion: resolving references and merging siblings.

use serde_json::{Map, Value};

use crate::error::{AgentsError, UserError};

use super::declares_schema_resource;

/// Keywords that may sit alongside a `$ref` without adding validation constraints.
pub(super) const REF_NON_CONSTRAINING_SIBLINGS: &[&str] = &[
    "$anchor",
    "$comment",
    "$defs",
    "$schema",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "default",
    "definitions",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
];

pub(super) fn is_bare_ref(entry: &Value) -> bool {
    let Value::Object(map) = entry else {
        return false;
    };
    map.contains_key("$ref")
        && map
            .keys()
            .all(|k| k == "$ref" || REF_NON_CONSTRAINING_SIBLINGS.contains(&k.as_str()))
}

pub(super) fn merge_ref_target(target: Value, siblings: Map<String, Value>) -> Value {
    let Value::Object(mut map) = target else {
        return target;
    };
    for (k, v) in siblings {
        map.entry(k).or_insert(v);
    }
    Value::Object(map)
}

/// Resolve a local `#/...` JSON pointer against the document root.
pub(super) fn resolve_ref(root: &Value, reference: &str) -> Result<Value, AgentsError> {
    let Some(pointer) = reference.strip_prefix("#/") else {
        return Err(UserError::new(format!(
            "Unexpected $ref format {reference:?}; it must start with `#/`"
        ))
        .into());
    };
    let mut current = root.clone();
    for raw_key in pointer.split('/') {
        if raw_key.is_empty() {
            continue;
        }
        let key = raw_key.replace("~1", "/").replace("~0", "~");
        let Value::Object(map) = current else {
            return Err(UserError::new(format!(
                "Encountered a non-object entry while resolving {reference}"
            ))
            .into());
        };
        let Some(next) = map.get(&key).cloned() else {
            return Err(UserError::new(format!(
                "Could not resolve {reference}: missing key `{key}`"
            ))
            .into());
        };
        if matches!(&next, Value::Object(m) if declares_schema_resource(m)) {
            return Err(UserError::new(
                "JSON schema contains a reference owned by or crossing a nested `$id` resource that \
                 cannot be resolved against the document root.",
            )
            .into());
        }
        current = next;
    }
    Ok(current)
}

/// Merge a singleton `allOf` entry into its parent, rejecting incompatible overlaps.
pub(super) fn merge_single_all_of(
    entry: Value,
    parent: &Map<String, Value>,
) -> Result<Map<String, Value>, AgentsError> {
    let Value::Object(mut merged) = entry else {
        return Err(UserError::new(
            "JSON schema contains a singleton `allOf` entry that is not an object.",
        )
        .into());
    };

    let mut incompatible: Vec<String> = Vec::new();
    if parent.get("additionalProperties") == Some(&Value::Bool(false))
        && !matches!(parent.get("properties"), Some(Value::Object(p)) if !p.is_empty())
        && matches!(merged.get("properties"), Some(Value::Object(p)) if !p.is_empty())
    {
        // `additionalProperties` only sees properties declared in its own schema, not allOf.
        incompatible.push("properties".into());
    }

    for (key, parent_value) in parent {
        match merged.get(key) {
            None => {
                merged.insert(key.clone(), parent_value.clone());
            }
            Some(_) if REF_NON_CONSTRAINING_SIBLINGS.contains(&key.as_str()) => {
                merged.insert(key.clone(), parent_value.clone());
            }
            Some(existing) if json_values_equal(existing, parent_value) => {}
            Some(_) if key == "properties" && parent_value == &Value::Object(Map::new()) => {}
            Some(_) if key == "required" && parent_value == &Value::Array(Vec::new()) => {}
            Some(_) => incompatible.push(key.clone()),
        }
    }

    if !incompatible.is_empty() {
        incompatible.sort();
        incompatible.dedup();
        return Err(UserError::new(format!(
            "JSON schema contains a singleton `allOf` entry with incompatible parent keyword(s) ({}) \
             that cannot be merged without changing its accepted values.",
            incompatible.join(", ")
        ))
        .into());
    }

    Ok(merged)
}

pub(super) fn json_values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Bool(_), _) | (_, Value::Bool(_)) => false,
        (Value::Number(a), Value::Number(b)) => a == b,
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| json_values_equal(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).map(|w| json_values_equal(v, w)).unwrap_or(false))
        }
        (Value::Null, Value::Null) => true,
        (Value::String(a), Value::String(b)) => a == b,
        _ => false,
    }
}
