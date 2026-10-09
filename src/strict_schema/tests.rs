//! Unit tests for strict schema conversion.

use super::*;
use serde_json::json;

#[test]
fn closes_objects_and_requires_all_properties() {
    let out = ensure_strict_json_schema(&json!({
        "type": "object",
        "properties": { "a": { "type": "string" } }
    }))
    .unwrap();
    assert_eq!(out["additionalProperties"], json!(false));
    assert_eq!(out["required"], json!(["a"]));
}

#[test]
fn nested_objects_are_closed_recursively() {
    let out = ensure_strict_json_schema(&json!({
        "type": "object",
        "properties": {
            "inner": { "type": "object", "properties": { "b": { "type": "integer" } } }
        }
    }))
    .unwrap();
    assert_eq!(
        out["properties"]["inner"]["additionalProperties"],
        json!(false)
    );
    assert_eq!(out["properties"]["inner"]["required"], json!(["b"]));
}

#[test]
fn array_items_are_closed() {
    let out = ensure_strict_json_schema(&json!({
        "type": "array",
        "items": { "type": "object", "properties": { "b": { "type": "boolean" } } }
    }))
    .unwrap();
    assert_eq!(out["items"]["additionalProperties"], json!(false));
    assert_eq!(out["items"]["required"], json!(["b"]));
}

#[test]
fn empty_schema_becomes_closed_object() {
    let out = ensure_strict_json_schema(&json!({})).unwrap();
    assert_eq!(out, empty_schema());
}

#[test]
fn one_of_becomes_any_of() {
    let out = ensure_strict_json_schema(&json!({
        "type": "object",
        "properties": {
            "v": { "oneOf": [{ "type": "string" }, { "type": "integer" }] }
        }
    }))
    .unwrap();
    assert!(out["properties"]["v"].get("oneOf").is_none());
    assert_eq!(out["properties"]["v"]["anyOf"].as_array().unwrap().len(), 2);
}

#[test]
fn null_default_is_stripped() {
    let out = ensure_strict_json_schema(&json!({
        "type": "string",
        "default": null
    }))
    .unwrap();
    assert!(out.get("default").is_none());
}

#[test]
fn rejects_open_additional_properties() {
    let err = ensure_strict_json_schema(&json!({
        "type": "object",
        "properties": {},
        "additionalProperties": true
    }))
    .unwrap_err();
    assert!(err.to_string().contains("additionalProperties"));
}

#[test]
fn rejects_any_of_at_root() {
    let err = ensure_strict_json_schema(&json!({
        "anyOf": [{ "type": "string" }]
    }))
    .unwrap_err();
    assert!(err.to_string().contains("anyOf"));
}
