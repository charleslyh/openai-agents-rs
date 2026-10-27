//! Turning a panic payload into text, so a panic inside a run becomes an error the caller sees.
//!
//! The payload of `panic!("...")` is a `&'static str` or a `String`; anything else (a custom
//! `panic_any` value) has no readable form. The text is the panic *message* only: it never
//! includes the arguments of the call that panicked, because those can hold user data.

use std::any::Any;

/// The readable message of a panic payload, or a fixed text when there is none.
pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "panic with a non-text payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_static_and_owned_messages() {
        let literal = std::panic::catch_unwind(|| panic!("boom")).unwrap_err();
        assert_eq!(panic_message(literal.as_ref()), "boom");

        let owned = std::panic::catch_unwind(|| panic!("{}", String::from("owned"))).unwrap_err();
        assert_eq!(panic_message(owned.as_ref()), "owned");
    }

    #[test]
    fn falls_back_for_other_payloads() {
        let other = std::panic::catch_unwind(|| std::panic::panic_any(7_u32)).unwrap_err();
        assert_eq!(
            panic_message(other.as_ref()),
            "panic with a non-text payload"
        );
    }
}
