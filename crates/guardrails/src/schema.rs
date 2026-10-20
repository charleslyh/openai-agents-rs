//! Refuse output whose shape is not the one expected.

use serde_json::{Map, Value};

use opai_agents_core::guardrail::Severity;

use crate::error::BuildError;
use crate::guard::{Decision, Guard, ValueDecision};

/// How deeply a schema may nest. A schema is configuration, so this is a sanity bound, and it is
/// also what bounds the validator's recursion: it follows the schema, never the value.
pub const MAX_SCHEMA_DEPTH: usize = 32;

/// Keywords that only describe, and change nothing about what is valid.
const ANNOTATIONS: &[&str] = &[
    "$schema",
    "$id",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "format",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// Keywords that are checked.
const SUPPORTED: &[&str] = &[
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
];

/// Refuses output that does not match a JSON Schema.
///
/// This validates a **subset** of JSON Schema: `type`, `enum`, `const`, `properties`, `required`,
/// `additionalProperties` (as `true` or `false`), `items` (one schema), `minItems`, `maxItems`,
/// `minLength`, `maxLength`, `minimum` and `maximum`. Descriptive keywords (`title`,
/// `description`, `default`, `format`, …) are accepted and ignored.
///
/// Anything else — `$ref`, `oneOf`, `anyOf`, `allOf`, `pattern`, `not`, `if` — is an error when the
/// guard is built. A validator that quietly skipped a keyword would accept what the schema was
/// written to refuse, so it refuses the schema instead. When that is in the way, validate with a
/// full JSON Schema implementation in a guardrail of your own.
///
/// Used as an output guardrail it checks the final output as a value. Used as an input guardrail
/// it checks that each piece of user text is itself JSON of the right shape.
///
/// A failure names the place and the rule (`/items/0/age: expected integer`), never the value.
///
/// ```rust
/// use opai_agents_guardrails::{Guard, SchemaValidator};
/// use serde_json::json;
///
/// # fn build() -> Result<(), opai_agents_guardrails::BuildError> {
/// let guardrail = SchemaValidator::new(json!({
///     "type": "object",
///     "required": ["answer"],
///     "properties": {"answer": {"type": "string", "maxLength": 500}},
///     "additionalProperties": false,
/// }))?
/// .output();
/// # let _ = guardrail;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SchemaValidator {
    schema: Value,
    severity: Severity,
}

impl SchemaValidator {
    /// Validate against `schema`, or return what in it is not supported.
    pub fn new(schema: Value) -> Result<Self, BuildError> {
        check_schema(&schema, "", 0)?;
        Ok(Self {
            schema,
            severity: Severity::High,
        })
    }

    /// How serious a failure is (default: [`Severity::High`]).
    #[must_use]
    pub fn severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// `Ok` if `value` matches, or the first problem found.
    pub fn validate(&self, value: &Value) -> Result<(), String> {
        validate(&self.schema, value, "")
    }
}

impl Guard for SchemaValidator {
    fn name(&self) -> &str {
        "schema_validator"
    }

    fn check(&self, text: &str) -> Decision {
        match serde_json::from_str::<Value>(text) {
            Err(_) => Decision::fail("the text is not valid JSON", self.severity),
            Ok(value) => match self.validate(&value) {
                Ok(()) => Decision::Pass,
                Err(reason) => Decision::fail(reason, self.severity),
            },
        }
    }

    fn check_value(&self, value: &Value) -> ValueDecision {
        match self.validate(value) {
            Ok(()) => ValueDecision::Pass,
            Err(reason) => ValueDecision::Fail {
                reason,
                severity: self.severity,
            },
        }
    }
}

fn pointer(path: &str) -> &str {
    if path.is_empty() {
        "/"
    } else {
        path
    }
}

/// Refuse a schema this validator would not enforce completely.
fn check_schema(schema: &Value, path: &str, depth: usize) -> Result<(), BuildError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(BuildError::new(format!(
            "the schema nests deeper than {MAX_SCHEMA_DEPTH} levels at {}",
            pointer(path)
        )));
    }
    let object = match schema {
        // `true` accepts anything, `false` nothing: both are fully understood.
        Value::Bool(_) => return Ok(()),
        Value::Object(object) => object,
        _ => {
            return Err(BuildError::new(format!(
                "the schema at {} must be an object or a boolean",
                pointer(path)
            )))
        }
    };
    for (keyword, value) in object {
        let here = format!("{path}/{keyword}");
        if ANNOTATIONS.contains(&keyword.as_str()) {
            continue;
        }
        if !SUPPORTED.contains(&keyword.as_str()) {
            return Err(BuildError::new(format!(
                "the schema keyword `{keyword}` at {} is not supported, and ignoring it would accept \
                 what the schema means to refuse",
                pointer(path)
            )));
        }
        match keyword.as_str() {
            "type" => check_types(value, &here)?,
            "enum" => {
                if !value.as_array().is_some_and(|a| !a.is_empty()) {
                    return Err(BuildError::new(format!(
                        "`enum` at {} must be a non-empty array",
                        pointer(path)
                    )));
                }
            }
            "required" => {
                if !value
                    .as_array()
                    .is_some_and(|a| a.iter().all(Value::is_string))
                {
                    return Err(BuildError::new(format!(
                        "`required` at {} must be an array of strings",
                        pointer(path)
                    )));
                }
            }
            "additionalProperties" => {
                if !value.is_boolean() {
                    return Err(BuildError::new(format!(
                        "`additionalProperties` at {} must be true or false; a schema there is not \
                         supported",
                        pointer(path)
                    )));
                }
            }
            "properties" => {
                let Some(properties) = value.as_object() else {
                    return Err(BuildError::new(format!(
                        "`properties` at {} must be an object",
                        pointer(path)
                    )));
                };
                for (name, sub) in properties {
                    check_schema(sub, &format!("{here}/{name}"), depth + 1)?;
                }
            }
            "items" => check_schema(value, &here, depth + 1)?,
            "minItems" | "maxItems" | "minLength" | "maxLength" => {
                if value.as_u64().is_none() {
                    return Err(BuildError::new(format!(
                        "`{keyword}` at {} must be a non-negative integer",
                        pointer(path)
                    )));
                }
            }
            "minimum" | "maximum" if !value.is_number() => {
                return Err(BuildError::new(format!(
                    "`{keyword}` at {} must be a number",
                    pointer(path)
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

const TYPES: &[&str] = &[
    "null", "boolean", "integer", "number", "string", "array", "object",
];

fn check_types(value: &Value, path: &str) -> Result<(), BuildError> {
    let names: Vec<&Value> = match value {
        Value::String(_) => vec![value],
        Value::Array(items) if !items.is_empty() => items.iter().collect(),
        _ => {
            return Err(BuildError::new(format!(
                "`type` at {path} must be a type name or a non-empty array of them"
            )))
        }
    };
    for name in names {
        if !name.as_str().is_some_and(|n| TYPES.contains(&n)) {
            return Err(BuildError::new(format!(
                "`type` at {path} holds a name that is not a JSON Schema type"
            )));
        }
    }
    Ok(())
}

fn has_type(value: &Value, name: &str) -> bool {
    match name {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "number" => value.is_number(),
        "integer" => {
            value.is_i64()
                || value.is_u64()
                || value
                    .as_f64()
                    .is_some_and(|f| f.is_finite() && f.fract() == 0.0)
        }
        _ => false,
    }
}

/// The first rule `value` breaks, said without quoting `value`.
fn validate(schema: &Value, value: &Value, path: &str) -> Result<(), String> {
    let object: &Map<String, Value> = match schema {
        Value::Bool(true) => return Ok(()),
        Value::Bool(false) => return Err(format!("{}: nothing is allowed here", pointer(path))),
        Value::Object(object) => object,
        _ => return Ok(()),
    };

    if let Some(types) = object.get("type") {
        let names: Vec<&str> = match types {
            Value::String(name) => vec![name.as_str()],
            Value::Array(items) => items.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        if !names.iter().any(|name| has_type(value, name)) {
            return Err(format!(
                "{}: expected {}",
                pointer(path),
                names.join(" or ")
            ));
        }
    }
    if let Some(Value::Array(allowed)) = object.get("enum") {
        if !allowed.contains(value) {
            return Err(format!("{}: not one of the allowed values", pointer(path)));
        }
    }
    if let Some(expected) = object.get("const") {
        if expected != value {
            return Err(format!("{}: not the required value", pointer(path)));
        }
    }

    match value {
        Value::Object(map) => validate_object(object, map, path)?,
        Value::Array(items) => validate_array(object, items, path)?,
        Value::String(text) => {
            let length = text.chars().count();
            if let Some(min) = object.get("minLength").and_then(Value::as_u64) {
                if (length as u64) < min {
                    return Err(format!("{}: shorter than {min} characters", pointer(path)));
                }
            }
            if let Some(max) = object.get("maxLength").and_then(Value::as_u64) {
                if length as u64 > max {
                    return Err(format!("{}: longer than {max} characters", pointer(path)));
                }
            }
        }
        Value::Number(number) => {
            if let Some(x) = number.as_f64() {
                if let Some(min) = object.get("minimum").and_then(Value::as_f64) {
                    if x < min {
                        return Err(format!("{}: below the minimum", pointer(path)));
                    }
                }
                if let Some(max) = object.get("maximum").and_then(Value::as_f64) {
                    if x > max {
                        return Err(format!("{}: above the maximum", pointer(path)));
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_object(
    schema: &Map<String, Value>,
    map: &Map<String, Value>,
    path: &str,
) -> Result<(), String> {
    if let Some(Value::Array(required)) = schema.get("required") {
        for name in required.iter().filter_map(Value::as_str) {
            if !map.contains_key(name) {
                return Err(format!(
                    "{}: missing the required property `{name}`",
                    pointer(path)
                ));
            }
        }
    }
    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties {
        for (name, sub) in properties {
            if let Some(child) = map.get(name) {
                validate(sub, child, &format!("{path}/{name}"))?;
            }
        }
    }
    if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
        for name in map.keys() {
            if !properties.is_some_and(|p| p.contains_key(name)) {
                return Err(format!(
                    "{}: has a property the schema does not allow",
                    pointer(path)
                ));
            }
        }
    }
    Ok(())
}

fn validate_array(schema: &Map<String, Value>, items: &[Value], path: &str) -> Result<(), String> {
    if let Some(min) = schema.get("minItems").and_then(Value::as_u64) {
        if (items.len() as u64) < min {
            return Err(format!("{}: fewer than {min} items", pointer(path)));
        }
    }
    if let Some(max) = schema.get("maxItems").and_then(Value::as_u64) {
        if items.len() as u64 > max {
            return Err(format!("{}: more than {max} items", pointer(path)));
        }
    }
    if let Some(sub) = schema.get("items") {
        for (index, item) in items.iter().enumerate() {
            validate(sub, item, &format!("{path}/{index}"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn validator(schema: Value) -> SchemaValidator {
        SchemaValidator::new(schema).expect("a supported schema")
    }

    fn reason(guard: &SchemaValidator, value: Value) -> String {
        guard.validate(&value).expect_err("must not validate")
    }

    #[test]
    fn types_are_checked() {
        let guard = validator(json!({"type": "string"}));
        assert!(guard.validate(&json!("x")).is_ok());
        assert!(guard.validate(&json!(1)).is_err());
        let guard = validator(json!({"type": ["string", "null"]}));
        assert!(guard.validate(&json!(null)).is_ok());
        assert!(guard.validate(&json!(true)).is_err());
        // Integers: a whole float counts, a fraction does not.
        let guard = validator(json!({"type": "integer"}));
        assert!(guard.validate(&json!(3)).is_ok());
        assert!(guard.validate(&json!(3.0)).is_ok());
        assert!(guard.validate(&json!(3.5)).is_err());
        assert!(validator(json!({"type": "number"}))
            .validate(&json!(3.5))
            .is_ok());
    }

    #[test]
    fn objects_need_their_required_properties_and_may_forbid_extras() {
        let guard = validator(json!({
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}, "score": {"type": "integer"}},
            "additionalProperties": false,
        }));
        assert!(guard.validate(&json!({"answer": "x", "score": 1})).is_ok());
        assert!(reason(&guard, json!({})).contains("missing the required property `answer`"));
        assert!(reason(&guard, json!({"answer": 1})).starts_with("/answer: expected string"));
        assert!(reason(&guard, json!({"answer": "x", "extra": 1})).contains("does not allow"));
        // Without additionalProperties: false, extras are fine.
        let open = validator(json!({"type": "object", "properties": {}}));
        assert!(open.validate(&json!({"anything": 1})).is_ok());
    }

    #[test]
    fn arrays_strings_and_numbers_have_bounds() {
        let guard = validator(json!({
            "type": "array", "minItems": 1, "maxItems": 2,
            "items": {"type": "integer", "minimum": 0, "maximum": 10},
        }));
        assert!(guard.validate(&json!([1, 2])).is_ok());
        assert!(reason(&guard, json!([])).contains("fewer than 1"));
        assert!(reason(&guard, json!([1, 2, 3])).contains("more than 2"));
        assert!(reason(&guard, json!([11])).starts_with("/0: above the maximum"));
        assert!(reason(&guard, json!([-1])).contains("below the minimum"));
        let text = validator(json!({"type": "string", "minLength": 2, "maxLength": 3}));
        assert!(text.validate(&json!("ab")).is_ok());
        assert!(text.validate(&json!("a")).is_err());
        assert!(text.validate(&json!("abcd")).is_err());
        // Characters, not bytes.
        assert!(text.validate(&json!("éé")).is_ok());
    }

    #[test]
    fn enum_and_const_are_checked() {
        let guard = validator(json!({"enum": ["a", "b"]}));
        assert!(guard.validate(&json!("a")).is_ok());
        assert!(guard.validate(&json!("c")).is_err());
        assert!(validator(json!({"const": 7})).validate(&json!(8)).is_err());
    }

    #[test]
    fn the_failure_names_the_place_and_never_the_value() {
        let guard = validator(json!({
            "type": "object",
            "properties": {"user": {"type": "object", "properties": {"age": {"type": "integer"}}}},
        }));
        let message = reason(&guard, json!({"user": {"age": "SECRET-VALUE"}}));
        assert_eq!(message, "/user/age: expected integer");
        assert!(!message.contains("SECRET"));
    }

    /// The point of failing at construction: a keyword that is skipped accepts what it should refuse.
    #[test]
    fn an_unsupported_keyword_is_refused_when_the_guard_is_built() {
        for schema in [
            json!({"$ref": "#/$defs/x"}),
            json!({"oneOf": [{"type": "string"}]}),
            json!({"anyOf": []}),
            json!({"allOf": []}),
            json!({"pattern": "^a+$"}),
            json!({"not": {"type": "string"}}),
            json!({"properties": {"x": {"pattern": "a"}}}),
            json!({"items": {"if": {}}}),
            json!({"additionalProperties": {"type": "string"}}),
        ] {
            assert!(
                SchemaValidator::new(schema.clone()).is_err(),
                "{schema} must be refused"
            );
        }
    }

    #[test]
    fn malformed_schemas_are_refused() {
        for schema in [
            json!("string"),
            json!(3),
            json!({"type": "str"}),
            json!({"type": []}),
            json!({"type": 5}),
            json!({"enum": []}),
            json!({"required": "a"}),
            json!({"required": [1]}),
            json!({"minLength": -1}),
            json!({"minimum": "x"}),
            json!({"properties": []}),
        ] {
            assert!(
                SchemaValidator::new(schema.clone()).is_err(),
                "{schema} must be refused"
            );
        }
    }

    #[test]
    fn descriptive_keywords_are_accepted_and_ignored() {
        let guard = validator(json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Reply", "description": "d", "type": "string", "format": "email", "default": "x",
        }));
        assert!(guard.validate(&json!("not an email")).is_ok());
    }

    #[test]
    fn booleans_are_schemas() {
        assert!(validator(json!(true)).validate(&json!(1)).is_ok());
        assert!(validator(json!(false)).validate(&json!(1)).is_err());
    }

    #[test]
    fn a_schema_nested_too_deeply_is_refused() {
        let mut schema = json!({"type": "string"});
        for _ in 0..(MAX_SCHEMA_DEPTH + 2) {
            schema = json!({"type": "array", "items": schema});
        }
        assert!(SchemaValidator::new(schema).is_err());
    }

    /// As a guard over text, the text must itself be JSON.
    #[test]
    fn text_is_parsed_first() {
        let guard = validator(json!({"type": "object", "required": ["a"]}));
        assert_eq!(guard.check(r#"{"a": 1}"#), Decision::Pass);
        assert!(matches!(guard.check("not json"), Decision::Fail { .. }));
        assert!(matches!(guard.check("{}"), Decision::Fail { .. }));
    }

    /// As an output guardrail the value is checked as it is, not as text.
    #[test]
    fn a_value_is_checked_as_a_value() {
        let guard = validator(json!({"type": "object", "required": ["a"]}));
        assert_eq!(guard.check_value(&json!({"a": 1})), ValueDecision::Pass);
        assert!(matches!(
            guard.check_value(&json!({"b": 1})),
            ValueDecision::Fail { .. }
        ));
    }
}
