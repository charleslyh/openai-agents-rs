//! SQLite-backed [`Session`] (Python: `SQLiteSession`).
//!
//! Uses the Python SDK's schema, so a database file written by one SDK can be read by the other:
//! `agent_sessions(session_id, created_at, updated_at)` and
//! `agent_messages(id, session_id, message_data, created_at)` where `message_data` is the item as
//! JSON text. Enabled by the `sqlite` cargo feature.
//!
//! One connection is shared behind a mutex and every call runs on the blocking thread pool, so
//! the session never blocks the async runtime. Cloning the session shares the connection.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;

use super::Session;
use crate::error::{AgentsError, UserError};

const DEFAULT_SESSIONS_TABLE: &str = "agent_sessions";
const DEFAULT_MESSAGES_TABLE: &str = "agent_messages";

/// Conversation storage in a SQLite database (Python: `SQLiteSession`).
#[derive(Clone)]
pub struct SqliteSession {
    session_id: Arc<str>,
    sessions_table: Arc<str>,
    messages_table: Arc<str>,
    connection: Arc<Mutex<Connection>>,
}

impl std::fmt::Debug for SqliteSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteSession")
            .field("session_id", &self.session_id)
            .field("sessions_table", &self.sessions_table)
            .field("messages_table", &self.messages_table)
            .finish()
    }
}

fn storage_error(error: impl std::fmt::Display) -> AgentsError {
    AgentsError::internal(format!("sqlite session: {error}"))
}

/// Table names are spliced into SQL, so only plain identifiers are accepted.
fn validate_table_name(name: &str) -> Result<(), AgentsError> {
    let valid = !name.is_empty()
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if valid {
        Ok(())
    } else {
        Err(UserError::new(format!(
            "invalid SQLite table name `{name}`: use letters, digits and underscores only"
        ))
        .into())
    }
}

impl SqliteSession {
    /// A session stored in a process-local in-memory database (Python: `SQLiteSession(id)`).
    pub fn in_memory(session_id: impl Into<String>) -> Result<Self, AgentsError> {
        Self::from_connection(
            session_id.into(),
            Connection::open_in_memory().map_err(storage_error)?,
            DEFAULT_SESSIONS_TABLE,
            DEFAULT_MESSAGES_TABLE,
            false,
        )
    }

    /// A session stored in the database file at `path`, created when missing
    /// (Python: `SQLiteSession(id, db_path)`). Several sessions can share one file.
    pub fn open(
        session_id: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<Self, AgentsError> {
        Self::open_with_tables(
            session_id,
            path,
            DEFAULT_SESSIONS_TABLE,
            DEFAULT_MESSAGES_TABLE,
        )
    }

    /// Like [`open`](Self::open) with custom table names
    /// (Python: `sessions_table` / `messages_table`).
    pub fn open_with_tables(
        session_id: impl Into<String>,
        path: impl AsRef<Path>,
        sessions_table: &str,
        messages_table: &str,
    ) -> Result<Self, AgentsError> {
        Self::from_connection(
            session_id.into(),
            Connection::open(path).map_err(storage_error)?,
            sessions_table,
            messages_table,
            true,
        )
    }

    /// Wrap the session in an `Arc` ready for `RunOptions::session`.
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    fn from_connection(
        session_id: String,
        connection: Connection,
        sessions_table: &str,
        messages_table: &str,
        use_wal: bool,
    ) -> Result<Self, AgentsError> {
        validate_table_name(sessions_table)?;
        validate_table_name(messages_table)?;
        connection
            .busy_timeout(std::time::Duration::from_secs(5))
            .map_err(storage_error)?;
        connection
            .pragma_update(None, "foreign_keys", "ON")
            .map_err(storage_error)?;
        if use_wal {
            // Python enables WAL for file databases so readers do not block the writer.
            connection
                .pragma_update(None, "journal_mode", "WAL")
                .map_err(storage_error)?;
        }
        connection
            .execute_batch(&format!(
                "CREATE TABLE IF NOT EXISTS {sessions_table} (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS {messages_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES {sessions_table} (session_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_{messages_table}_session_id
                    ON {messages_table} (session_id, id);"
            ))
            .map_err(storage_error)?;
        Ok(Self {
            session_id: session_id.into(),
            sessions_table: sessions_table.into(),
            messages_table: messages_table.into(),
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    /// Run `work` on the blocking pool with exclusive use of the connection.
    async fn with_connection<T, F>(&self, work: F) -> Result<T, AgentsError>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection, &SqliteSession) -> rusqlite::Result<T> + Send + 'static,
    {
        let this = self.clone();
        tokio::task::spawn_blocking(move || {
            let mut connection = this
                .connection
                .lock()
                .map_err(|_| storage_error("connection lock poisoned"))?;
            work(&mut connection, &this).map_err(storage_error)
        })
        .await
        .map_err(storage_error)?
    }
}

/// Append `items` and bump the session row, creating it when missing.
fn insert_items(
    tx: &rusqlite::Transaction<'_>,
    this: &SqliteSession,
    items: &[Value],
) -> rusqlite::Result<()> {
    tx.execute(
        &format!(
            "INSERT OR IGNORE INTO {} (session_id) VALUES (?1)",
            this.sessions_table
        ),
        params![&*this.session_id],
    )?;
    {
        let mut insert = tx.prepare(&format!(
            "INSERT INTO {} (session_id, message_data) VALUES (?1, ?2)",
            this.messages_table
        ))?;
        for item in items {
            let text = serde_json::to_string(item)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
            insert.execute(params![&*this.session_id, text])?;
        }
    }
    tx.execute(
        &format!(
            "UPDATE {} SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?1",
            this.sessions_table
        ),
        params![&*this.session_id],
    )?;
    Ok(())
}

#[async_trait]
impl Session for SqliteSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<Value>, AgentsError> {
        self.with_connection(move |conn, this| {
            if limit == Some(0) {
                return Ok(Vec::new());
            }
            // Newest first, so a limit takes the latest items. Rows that are not valid JSON are
            // skipped and do not count toward the limit (Python widens its window the same way).
            let mut statement = conn.prepare_cached(&format!(
                "SELECT message_data FROM {} WHERE session_id = ?1 ORDER BY id DESC",
                this.messages_table
            ))?;
            let mut rows = statement.query(params![&*this.session_id])?;
            let mut items = Vec::new();
            while let Some(row) = rows.next()? {
                let text: String = row.get(0)?;
                if let Ok(item) = serde_json::from_str::<Value>(&text) {
                    items.push(item);
                    if limit.is_some_and(|n| items.len() >= n) {
                        break;
                    }
                }
            }
            items.reverse();
            Ok(items)
        })
        .await
    }

    async fn add_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        if items.is_empty() {
            return Ok(());
        }
        self.with_connection(move |conn, this| {
            let tx = conn.transaction()?;
            insert_items(&tx, this, &items)?;
            tx.commit()
        })
        .await
    }

    async fn replace_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        self.with_connection(move |conn, this| {
            // One transaction: readers see the old history or the new one, never a mix.
            let tx = conn.transaction()?;
            tx.execute(
                &format!("DELETE FROM {} WHERE session_id = ?1", this.messages_table),
                params![&*this.session_id],
            )?;
            insert_items(&tx, this, &items)?;
            tx.commit()
        })
        .await
    }

    async fn pop_item(&self) -> Result<Option<Value>, AgentsError> {
        self.with_connection(move |conn, this| {
            let tx = conn.transaction()?;
            let sql = format!(
                "DELETE FROM {table} WHERE id = (
                     SELECT id FROM {table} WHERE session_id = ?1 ORDER BY id DESC LIMIT 1
                 ) RETURNING message_data",
                table = this.messages_table
            );
            let popped = loop {
                let text: Option<String> = tx
                    .query_row(&sql, params![&*this.session_id], |row| row.get(0))
                    .optional()?;
                match text {
                    None => break None,
                    // An unreadable tail row is dropped, then the next one is tried (Python).
                    Some(text) => {
                        if let Ok(item) = serde_json::from_str::<Value>(&text) {
                            break Some(item);
                        }
                    }
                }
            };
            tx.commit()?;
            Ok(popped)
        })
        .await
    }

    async fn clear_session(&self) -> Result<(), AgentsError> {
        self.with_connection(move |conn, this| {
            let tx = conn.transaction()?;
            tx.execute(
                &format!("DELETE FROM {} WHERE session_id = ?1", this.messages_table),
                params![&*this.session_id],
            )?;
            tx.execute(
                &format!("DELETE FROM {} WHERE session_id = ?1", this.sessions_table),
                params![&*this.session_id],
            )?;
            tx.commit()
        })
        .await
    }
}
