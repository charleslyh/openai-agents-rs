//! Long-term facts in a SQLite database.
//!
//! [`SqliteFactStore`] is the persistent [`FactStore`] for
//! [`FactMemory`](opai_agents_core::FactMemory). One table, `agent_facts`, holds
//! `(seq, scope, id, text)`:
//!
//! * `seq` grows with every write, so "oldest first" is `ORDER BY seq`, and storing a fact that
//!   exists already moves it to the newest place.
//! * `(scope, id)` is unique, so the same fact is kept once per scope, and scopes never mix: every
//!   query filters by scope, and the scope is always a bound parameter.
//! * Table names are spliced into SQL, so only plain identifiers are accepted.
//!
//! Like [`SqliteSession`](crate::SqliteSession), it shares one connection behind a mutex and runs
//! every call on the blocking pool. The facts are stored as plain text: put the file where only
//! the process can read it, and delete a person's facts with `FactMemory::forget_all`.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use opai_agents_core::error::AgentsError;
use opai_agents_core::{Fact, FactStore};
use rusqlite::{params, Connection};

use crate::sqlite::validate_table_name;

const DEFAULT_TABLE: &str = "agent_facts";

/// A [`FactStore`] in a SQLite database.
#[derive(Clone)]
pub struct SqliteFactStore {
    table: Arc<str>,
    connection: Arc<Mutex<Connection>>,
}

impl std::fmt::Debug for SqliteFactStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteFactStore")
            .field("table", &self.table)
            .finish()
    }
}

fn storage_error(error: impl std::fmt::Display) -> AgentsError {
    AgentsError::internal(format!("sqlite fact store: {error}"))
}

impl SqliteFactStore {
    /// A store in a process-local in-memory database.
    pub fn in_memory() -> Result<Self, AgentsError> {
        Self::from_connection(
            Connection::open_in_memory().map_err(storage_error)?,
            DEFAULT_TABLE,
            false,
        )
    }

    /// A store in the database file at `path`, created when missing. The file may be shared with
    /// [`SqliteSession`](crate::SqliteSession): the tables do not overlap.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AgentsError> {
        Self::open_with_table(path, DEFAULT_TABLE)
    }

    /// Like [`open`](Self::open) with a custom table name.
    pub fn open_with_table(path: impl AsRef<Path>, table: &str) -> Result<Self, AgentsError> {
        Self::from_connection(Connection::open(path).map_err(storage_error)?, table, true)
    }

    /// Wrap in an `Arc`, ready for `FactMemory::new`.
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    fn from_connection(
        connection: Connection,
        table: &str,
        use_wal: bool,
    ) -> Result<Self, AgentsError> {
        validate_table_name(table)?;
        connection
            .busy_timeout(std::time::Duration::from_secs(5))
            .map_err(storage_error)?;
        if use_wal {
            connection
                .pragma_update(None, "journal_mode", "WAL")
                .map_err(storage_error)?;
        }
        connection
            .execute_batch(&format!(
                "CREATE TABLE IF NOT EXISTS {table} (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    UNIQUE (scope, id)
                );
                CREATE INDEX IF NOT EXISTS idx_{table}_scope ON {table} (scope, seq);"
            ))
            .map_err(storage_error)?;
        Ok(Self {
            table: table.into(),
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    /// Run `work` on the blocking pool with exclusive use of the connection.
    async fn with_connection<T, F>(&self, work: F) -> Result<T, AgentsError>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection, &str) -> rusqlite::Result<T> + Send + 'static,
    {
        let this = self.clone();
        tokio::task::spawn_blocking(move || {
            let mut connection = this
                .connection
                .lock()
                .map_err(|_| storage_error("connection lock poisoned"))?;
            work(&mut connection, &this.table).map_err(storage_error)
        })
        .await
        .map_err(storage_error)?
    }
}

#[async_trait]
impl FactStore for SqliteFactStore {
    async fn list(&self, scope: &str) -> Result<Vec<Fact>, AgentsError> {
        let scope = scope.to_string();
        self.with_connection(move |conn, table| {
            let mut statement = conn.prepare_cached(&format!(
                "SELECT id, text FROM {table} WHERE scope = ?1 ORDER BY seq"
            ))?;
            let rows = statement.query_map(params![scope], |row| {
                Ok(Fact {
                    id: row.get(0)?,
                    text: row.get(1)?,
                })
            })?;
            rows.collect()
        })
        .await
    }

    async fn put(&self, scope: &str, fact: Fact) -> Result<(), AgentsError> {
        let scope = scope.to_string();
        self.with_connection(move |conn, table| {
            // Delete and insert in one transaction: a fact that exists becomes the newest.
            let tx = conn.transaction()?;
            tx.execute(
                &format!("DELETE FROM {table} WHERE scope = ?1 AND id = ?2"),
                params![scope, fact.id],
            )?;
            tx.execute(
                &format!("INSERT INTO {table} (scope, id, text) VALUES (?1, ?2, ?3)"),
                params![scope, fact.id, fact.text],
            )?;
            tx.commit()
        })
        .await
    }

    async fn remove(&self, scope: &str, id: &str) -> Result<bool, AgentsError> {
        let (scope, id) = (scope.to_string(), id.to_string());
        self.with_connection(move |conn, table| {
            let removed = conn.execute(
                &format!("DELETE FROM {table} WHERE scope = ?1 AND id = ?2"),
                params![scope, id],
            )?;
            Ok(removed > 0)
        })
        .await
    }

    async fn clear(&self, scope: &str) -> Result<(), AgentsError> {
        let scope = scope.to_string();
        self.with_connection(move |conn, table| {
            conn.execute(
                &format!("DELETE FROM {table} WHERE scope = ?1"),
                params![scope],
            )?;
            Ok(())
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use opai_agents_core::{AgentsError, FactChange, FactExtractor, FactMemory};

    use super::*;

    fn fact(text: &str) -> Fact {
        Fact::new(text).unwrap()
    }

    fn path(name: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        std::env::temp_dir().join(format!("agents-facts-{name}-{unique}.db"))
    }

    fn texts(facts: Vec<Fact>) -> Vec<String> {
        facts.into_iter().map(|f| f.text).collect()
    }

    #[tokio::test]
    async fn facts_keep_their_order_scopes_and_move_to_the_newest_when_stored_again() {
        let store = SqliteFactStore::in_memory().unwrap();
        store.put("a", fact("one")).await.unwrap();
        store.put("a", fact("two")).await.unwrap();
        store.put("a", fact("one")).await.unwrap();
        store.put("b", fact("other")).await.unwrap();
        assert_eq!(texts(store.list("a").await.unwrap()), ["two", "one"]);
        assert_eq!(texts(store.list("b").await.unwrap()), ["other"]);
        assert!(store.list("nobody").await.unwrap().is_empty());

        assert!(store.remove("a", &fact("two").id).await.unwrap());
        assert!(!store.remove("a", &fact("two").id).await.unwrap());
        assert!(
            !store.remove("b", &fact("one").id).await.unwrap(),
            "an id of another scope is not removed"
        );
        store.clear("a").await.unwrap();
        assert!(store.list("a").await.unwrap().is_empty());
        assert_eq!(store.list("b").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn facts_survive_reopening_the_file() {
        let file = path("reopen");
        {
            let store = SqliteFactStore::open(&file).unwrap();
            store.put("u", fact("prefers rust")).await.unwrap();
        }
        let store = SqliteFactStore::open(&file).unwrap();
        assert_eq!(texts(store.list("u").await.unwrap()), ["prefers rust"]);
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", file.display()));
        }
    }

    #[tokio::test]
    async fn a_hostile_scope_or_text_is_data_and_never_sql() {
        let store = SqliteFactStore::in_memory().unwrap();
        let scope = "x'; DROP TABLE agent_facts; --";
        store
            .put(scope, fact("it's \"quoted\"; DELETE FROM agent_facts"))
            .await
            .unwrap();
        store.put("safe", fact("still here")).await.unwrap();
        assert_eq!(store.list(scope).await.unwrap().len(), 1);
        assert_eq!(store.list("safe").await.unwrap().len(), 1);
        // A scope that looks like a wildcard matches nothing else.
        assert!(store.list("%").await.unwrap().is_empty());
    }

    #[test]
    fn table_names_are_validated() {
        let file = path("table");
        for bad in ["", "a b", "a;b", "1abc", "a-b", "x)--"] {
            assert!(
                SqliteFactStore::open_with_table(&file, bad).is_err(),
                "{bad}"
            );
        }
        assert!(SqliteFactStore::open_with_table(&file, "my_facts").is_ok());
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", file.display()));
        }
    }

    struct Adds(&'static str);

    #[async_trait]
    impl FactExtractor for Adds {
        async fn extract(&self, _: &[Fact], _: &[String]) -> Result<Vec<FactChange>, AgentsError> {
            Ok(vec![FactChange::Add(self.0.into())])
        }
    }

    #[tokio::test]
    async fn fact_memory_works_on_top_of_it_and_searches_by_keyword() {
        let store = SqliteFactStore::in_memory().unwrap().shared();
        let memory = FactMemory::new(store.clone(), Arc::new(Adds("uses postgres")), "u");
        let user = serde_json::json!({"role": "user", "content": "we use postgres"});
        assert_eq!(memory.remember(&[user]).await.unwrap(), 1);
        store.put("u", fact("likes tea")).await.unwrap();
        let found = memory.recall("which postgres version", 5).await.unwrap();
        assert_eq!(texts(found), ["uses postgres"]);
        memory.forget_all().await.unwrap();
        assert!(memory.list().await.unwrap().is_empty());
    }
}
