//! Turning a panic payload into text, so a panic inside a run becomes an error the caller sees.
//!
//! The payload of `panic!("...")` is a `&'static str` or a `String`; anything else (a custom
//! `panic_any` value) has no readable form. The text is the panic *message* only: it never
//! includes the arguments of the call that panicked, because those can hold user data.

use std::any::Any;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;

use crate::error::AgentsError;

/// Run `fut`, turning a panic inside it into `Err(message)`.
///
/// `AssertUnwindSafe` is deliberate: the future is dropped right after a panic and nothing it
/// touched is observed again, so a broken invariant cannot be seen from outside.
pub(crate) async fn catch<F: Future>(fut: F) -> Result<F::Output, String> {
    AssertUnwindSafe(fut)
        .catch_unwind()
        .await
        .map_err(|payload| panic_message(payload.as_ref()))
}

/// Run application code (`what` names it) and report a panic as a run-ending `Internal` error.
///
/// For hooks and guardrails: they belong to the application, not to the model, so the message is
/// kept for the developer and logged. The caller gets an error it can handle instead of an unwind
/// through the whole run.
pub(crate) async fn guard<F: Future>(what: &str, fut: F) -> Result<F::Output, AgentsError> {
    catch(fut).await.map_err(|message| {
        tracing::error!(%what, %message, "application code panicked during a run");
        AgentsError::internal(format!("{what} panicked: {message}"))
    })
}

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
