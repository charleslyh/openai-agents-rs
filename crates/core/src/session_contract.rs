//! The behaviour every [`Session`] backend must have, as a test suite any backend can run.
//!
//! It lives here, in core, on purpose: an extension crate may not depend on another extension, so
//! a contract shared by the bundled backends cannot live in `opai-agents-testing`. A backend
//! outside this workspace can run it too, which is the point — the same eight behaviours, whether
//! the store is a file, Redis, Postgres or something else.
//!
//! ```rust,no_run
//! use opai_agents_core::{session_contract, Session};
//!
//! # async fn check(session: &dyn Session) {
//! session_contract::assert_session_contract(session).await;
//! # }
//! ```

use std::sync::Arc;

use serde_json::{json, Value};

use crate::memory::Session;

/// Run every contract check against `session`, panicking with the first failure.
///
/// The session must be empty when this is called.
pub async fn assert_session_contract(session: &dyn Session) {
    assert_empty(session).await;
    assert_order_and_limit(session).await;
    assert_pop_item(session).await;
    assert_replace_items(session).await;
    assert_replace_is_not_observed_half_done(session).await;
    assert_clear_session(session).await;
    assert_large_item(session).await;
    session
        .clear_session()
        .await
        .expect("the contract leaves the session empty");
}

fn item(text: &str) -> Value {
    json!({"type": "message", "role": "user", "content": text})
}

fn text_of(item: &Value) -> &str {
    item["content"].as_str().unwrap_or_default()
}

/// A new session holds nothing, and reading it is not an error.
async fn assert_empty(session: &dyn Session) {
    let items = session
        .get_items(None)
        .await
        .expect("read an empty session");
    assert!(items.is_empty(), "a new session must be empty: {items:?}");
    assert!(
        session.pop_item().await.expect("pop").is_none(),
        "nothing to pop"
    );
}

/// Items come back oldest first; `limit` takes the newest ones, not the oldest.
async fn assert_order_and_limit(session: &dyn Session) {
    session
        .clear_session()
        .await
        .expect("reset before this check");
    session
        .add_items(vec![item("one"), item("two"), item("three")])
        .await
        .expect("add");

    let all = session.get_items(None).await.expect("read all");
    assert_eq!(
        all.iter().map(text_of).collect::<Vec<_>>(),
        vec!["one", "two", "three"],
        "items come back in the order they were added"
    );

    let latest = session.get_items(Some(2)).await.expect("read with a limit");
    assert_eq!(
        latest.iter().map(text_of).collect::<Vec<_>>(),
        vec!["two", "three"],
        "a limit takes the newest items"
    );

    let over = session
        .get_items(Some(99))
        .await
        .expect("read with a big limit");
    assert_eq!(over.len(), 3, "a limit above the history changes nothing");
}

/// `pop_item` removes and returns the newest item.
async fn assert_pop_item(session: &dyn Session) {
    session
        .clear_session()
        .await
        .expect("reset before this check");
    session
        .add_items(vec![item("one"), item("two")])
        .await
        .expect("add");

    let popped = session.pop_item().await.expect("pop");
    assert_eq!(text_of(&popped.expect("an item")), "two");
    let left = session.get_items(None).await.expect("read");
    assert_eq!(left.iter().map(text_of).collect::<Vec<_>>(), vec!["one"]);

    assert!(session.pop_item().await.expect("pop").is_some());
    assert!(session.pop_item().await.expect("pop").is_none());
}

/// `replace_items` swaps the whole history, including to and from empty.
async fn assert_replace_items(session: &dyn Session) {
    session
        .clear_session()
        .await
        .expect("reset before this check");
    session
        .add_items(vec![item("one"), item("two")])
        .await
        .expect("add");
    session
        .replace_items(vec![item("summary")])
        .await
        .expect("replace");
    let items = session.get_items(None).await.expect("read");
    assert_eq!(
        items.iter().map(text_of).collect::<Vec<_>>(),
        vec!["summary"]
    );

    session
        .replace_items(vec![])
        .await
        .expect("replace with none");
    assert!(
        session.get_items(None).await.expect("read").is_empty(),
        "replacing with nothing empties the session"
    );
}

/// A reader never sees a half-written history: a concurrent read returns the old contents or the
/// new ones, never a mix.
async fn assert_replace_is_not_observed_half_done(session: &dyn Session) {
    session
        .clear_session()
        .await
        .expect("reset before this check");
    session
        .add_items((0..20).map(|i| item(&format!("old-{i}"))).collect())
        .await
        .expect("add");

    let new: Vec<Value> = (0..7).map(|i| item(&format!("new-{i}"))).collect();
    let ((), observed) = futures::join!(
        async { session.replace_items(new.clone()).await.expect("replace") },
        read_repeatedly(session)
    );
    for batch in observed {
        let texts: Vec<&str> = batch.iter().map(text_of).collect();
        let ok = texts.iter().all(|t| t.starts_with("old-")) && texts.len() == 20
            || texts.iter().all(|t| t.starts_with("new-")) && texts.len() == 7;
        assert!(ok, "a read saw a half-written history: {texts:?}");
    }
}

/// Read the history in a loop, so a concurrent write has something to interleave with.
async fn read_repeatedly(session: &dyn Session) -> Vec<Vec<Value>> {
    let mut seen = Vec::new();
    for _ in 0..50 {
        if let Ok(items) = session.get_items(None).await {
            seen.push(items);
        }
        tokio::task::yield_now().await;
    }
    seen
}

/// `clear_session` empties it, and clearing an empty one is fine.
async fn assert_clear_session(session: &dyn Session) {
    session.add_items(vec![item("one")]).await.expect("add");
    session.clear_session().await.expect("clear");
    assert!(session.get_items(None).await.expect("read").is_empty());
    session.clear_session().await.expect("clear again");
}

/// A large item survives the round trip unchanged.
async fn assert_large_item(session: &dyn Session) {
    session
        .clear_session()
        .await
        .expect("reset before this check");
    let big = "x".repeat(1 << 20);
    session.add_items(vec![item(&big)]).await.expect("add");
    let items = session.get_items(None).await.expect("read");
    assert_eq!(items.len(), 1, "one item came back");
    assert_eq!(
        text_of(&items[0]).len(),
        big.len(),
        "a 1 MiB item must round trip unchanged"
    );
}

/// The contract for a backend that can be shared with spawned tasks: concurrent writes from
/// several tasks all land.
pub async fn assert_concurrent_writes(session: Arc<dyn Session>, writers: usize) {
    session
        .clear_session()
        .await
        .expect("reset before this check");
    let mut tasks = Vec::new();
    for i in 0..writers {
        let session = Arc::clone(&session);
        tasks.push(tokio::spawn(async move {
            session
                .add_items(vec![item(&format!("from-{i}"))])
                .await
                .expect("add");
        }));
    }
    for task in tasks {
        task.await.expect("the writer finished");
    }
    let items = session.get_items(None).await.expect("read");
    assert_eq!(
        items.len(),
        writers,
        "every concurrent write must land: {items:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemorySession;
    use std::sync::Arc;

    #[tokio::test]
    async fn in_memory_passes_the_contract() {
        let session = InMemorySession::new("contract");
        assert_session_contract(&session).await;
        assert_concurrent_writes(Arc::new(InMemorySession::new("shared")), 16).await;
    }
}
