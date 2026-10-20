//! A session and a paused run in Postgres (Rust-only, X-017 / X-016).
//!
//! The backend for a deployment that already has a database: the history is rows, and a paused run
//! is one row holding the same envelope the file store writes, so integrity, expiry and schema are
//! decided once (see the shared `state` module) and behave the same here.
//!
//! A table name reaches SQL as an identifier and cannot be bound as a parameter, so it is checked
//! before it is used: letters, digits and underscores, starting with a letter or an underscore, at
//! most 63 characters (Postgres' own limit). Anything else is refused, which is what keeps a name
//! from being an injection — every *value*, on the other hand, is bound, never formatted into the
//! statement. sqlx refuses a statement built at runtime unless it is marked safe, and one private
//! helper is the single place that mark is made.
//!
//! `replace_items` — what compaction calls — runs in one transaction, so a reader sees the old
//! history or the new one and never a mix.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use sqlx::{AssertSqlSafe, PgPool, Row};

use opai_agents_core::error::UserError;
use opai_agents_core::{
    AgentsError, RunState, RunStateMigration, RunStateStore, Session, SessionSettings,
};

use super::state::{decode, key_from_env, now_seconds, to_state, validate_id, wrap};

/// Where the signing key for paused runs is read from.
pub const STATE_SIGNING_KEY_ENV: &str = "AGENTS_STATE_KEY";

/// A conversation in Postgres rows.
pub struct PostgresSession {
    id: String,
    table: String,
    pool: PgPool,
}

impl PostgresSession {
    /// Connect to `url` and use the session `session_id` in `table`.
    ///
    /// [`Self::create_table`] must have run once; it is not called here, so opening a session does
    /// not need the rights to create one.
    pub async fn connect(
        url: &str,
        session_id: impl Into<String>,
        table: &str,
    ) -> Result<Self, AgentsError> {
        let pool = PgPool::connect(url).await.map_err(map_err)?;
        Self::open(pool, session_id, table)
    }

    /// Use an existing pool, so a deployment can share one with the rest of its code.
    pub fn open(
        pool: PgPool,
        session_id: impl Into<String>,
        table: &str,
    ) -> Result<Self, AgentsError> {
        let id = session_id.into();
        validate_id(&id, "session id")?;
        Ok(Self {
            id,
            table: validate_table(table)?,
            pool,
        })
    }

    /// Create the table and its index if they are not there yet.
    ///
    /// Safe to call from several processes at once: `IF NOT EXISTS` alone races with itself (two
    /// creators can both decide the table is missing), so the creation takes an advisory lock held
    /// for the transaction.
    pub async fn create_table(&self) -> Result<(), AgentsError> {
        let table = &self.table;
        let mut tx = self.pool.begin().await.map_err(map_err)?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)")
            .bind(advisory_lock_key("session", table))
            .execute(&mut *tx)
            .await
            .map_err(map_err)?;
        sqlx::query(statement(format!(
            "CREATE TABLE IF NOT EXISTS {table} (
                seq BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                item JSONB NOT NULL
            )"
        )))
        .execute(&mut *tx)
        .await
        .map_err(map_err)?;
        sqlx::query(statement(format!(
            "CREATE INDEX IF NOT EXISTS {table}_session_id_seq_idx ON {table} (session_id, seq)"
        )))
        .execute(&mut *tx)
        .await
        .map_err(map_err)?;
        tx.commit().await.map_err(map_err)
    }

    /// The table this session's rows are in.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// The pool, for a caller that needs to do something else with it.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

#[async_trait]
impl Session for PostgresSession {
    fn session_id(&self) -> &str {
        &self.id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        None
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<Value>, AgentsError> {
        let table = &self.table;
        // A limit takes the newest rows, so the newest are selected first and put back in order.
        let sql = match limit {
            Some(_) => format!(
                "SELECT item::text FROM (
                    SELECT item, seq FROM {table} WHERE session_id = $1 ORDER BY seq DESC LIMIT $2
                 ) AS recent ORDER BY seq ASC"
            ),
            None => {
                format!("SELECT item::text FROM {table} WHERE session_id = $1 ORDER BY seq ASC")
            }
        };
        let query = sqlx::query(statement(sql)).bind(&self.id);
        let query = match limit {
            Some(0) => return Ok(Vec::new()),
            Some(n) => query.bind(n as i64),
            None => query,
        };
        let rows = query.fetch_all(&self.pool).await.map_err(map_err)?;
        rows.iter()
            .map(|row| parse_item(&row.get::<String, _>(0)))
            .collect()
    }

    async fn add_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        if items.is_empty() {
            return Ok(());
        }
        let mut tx = self.pool.begin().await.map_err(map_err)?;
        insert_items(&mut tx, &self.table, &self.id, &items).await?;
        tx.commit().await.map_err(map_err)
    }

    async fn pop_item(&self) -> Result<Option<Value>, AgentsError> {
        let table = &self.table;
        let row = sqlx::query(statement(format!(
            "DELETE FROM {table} WHERE seq = (
                SELECT seq FROM {table} WHERE session_id = $1 ORDER BY seq DESC LIMIT 1
             ) RETURNING item::text"
        )))
        .bind(&self.id)
        .fetch_optional(&self.pool)
        .await
        .map_err(map_err)?;
        match row {
            Some(row) => Ok(Some(parse_item(&row.get::<String, _>(0))?)),
            None => Ok(None),
        }
    }

    async fn clear_session(&self) -> Result<(), AgentsError> {
        let table = &self.table;
        sqlx::query(statement(format!(
            "DELETE FROM {table} WHERE session_id = $1"
        )))
        .bind(&self.id)
        .execute(&self.pool)
        .await
        .map_err(map_err)?;
        Ok(())
    }

    /// One transaction, so a reader sees the old history or the new one and never a mix.
    async fn replace_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        let table = &self.table;
        let mut tx = self.pool.begin().await.map_err(map_err)?;
        sqlx::query(statement(format!(
            "DELETE FROM {table} WHERE session_id = $1"
        )))
        .bind(&self.id)
        .execute(&mut *tx)
        .await
        .map_err(map_err)?;
        if !items.is_empty() {
            insert_items(&mut tx, table, &self.id, &items).await?;
        }
        tx.commit().await.map_err(map_err)
    }
}

impl std::fmt::Debug for PostgresSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresSession")
            .field("id", &self.id)
            .field("table", &self.table)
            .finish()
    }
}

/// A paused run in a Postgres row.
pub struct PostgresRunStateStore {
    pool: PgPool,
    table: String,
    /// Signing key, when one is configured.
    signing_key: Option<Vec<u8>>,
    migrate: Option<RunStateMigration>,
}

impl PostgresRunStateStore {
    /// Connect to `url`, signing states when `AGENTS_STATE_KEY` is set.
    pub async fn from_env(url: &str, table: &str) -> Result<Self, AgentsError> {
        let signing_key = key_from_env(STATE_SIGNING_KEY_ENV);
        Self::connect(url, table, signing_key).await
    }

    /// Connect to `url` without signing. For tests, and for a database only this process writes.
    pub async fn unsigned(url: &str, table: &str) -> Result<Self, AgentsError> {
        Self::connect(url, table, None).await
    }

    /// Use an existing pool.
    pub fn open(
        pool: PgPool,
        table: &str,
        signing_key: Option<Vec<u8>>,
    ) -> Result<Self, AgentsError> {
        Ok(Self {
            pool,
            table: validate_table(table)?,
            signing_key,
            migrate: None,
        })
    }

    async fn connect(
        url: &str,
        table: &str,
        signing_key: Option<Vec<u8>>,
    ) -> Result<Self, AgentsError> {
        let pool = PgPool::connect(url).await.map_err(map_err)?;
        Self::open(pool, table, signing_key)
    }

    /// Update states written by an older build before they are resumed.
    #[must_use]
    pub fn with_migration(mut self, migrate: RunStateMigration) -> Self {
        self.migrate = Some(migrate);
        self
    }

    /// Whether states are signed.
    pub fn is_signed(&self) -> bool {
        self.signing_key.is_some()
    }

    /// Create the table if it is not there yet.
    ///
    /// Safe to call from several processes at once, for the same reason as
    /// [`PostgresSession::create_table`].
    pub async fn create_table(&self) -> Result<(), AgentsError> {
        let table = &self.table;
        let mut tx = self.pool.begin().await.map_err(map_err)?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)")
            .bind(advisory_lock_key("state", table))
            .execute(&mut *tx)
            .await
            .map_err(map_err)?;
        sqlx::query(statement(format!(
            "CREATE TABLE IF NOT EXISTS {table} (
                state_key TEXT PRIMARY KEY,
                envelope JSONB NOT NULL,
                expires_at BIGINT
            )"
        )))
        .execute(&mut *tx)
        .await
        .map_err(map_err)?;
        tx.commit().await.map_err(map_err)
    }
}

#[async_trait]
impl RunStateStore for PostgresRunStateStore {
    async fn save(
        &self,
        key: &str,
        state: &RunState,
        ttl: Option<Duration>,
    ) -> Result<(), AgentsError> {
        validate_id(key, "run state key")?;
        let table = &self.table;
        let envelope = wrap(state, ttl, self.signing_key.as_deref())?;
        let json = serde_json::to_string(&envelope)
            .map_err(|e| UserError::new(format!("cannot serialize a paused run: {e}")))?;
        let expires_at = ttl.map(|ttl| now_seconds().saturating_add(ttl.as_secs()) as i64);
        sqlx::query(statement(format!(
            "INSERT INTO {table} (state_key, envelope, expires_at) VALUES ($1, $2::jsonb, $3)
             ON CONFLICT (state_key) DO UPDATE
             SET envelope = EXCLUDED.envelope, expires_at = EXCLUDED.expires_at"
        )))
        .bind(key)
        .bind(json)
        .bind(expires_at)
        .execute(&self.pool)
        .await
        .map_err(map_err)?;
        Ok(())
    }

    async fn load(&self, key: &str) -> Result<Option<RunState>, AgentsError> {
        validate_id(key, "run state key")?;
        let table = &self.table;
        let row = sqlx::query(statement(format!(
            "SELECT envelope::text FROM {table} WHERE state_key = $1"
        )))
        .bind(key)
        .fetch_optional(&self.pool)
        .await
        .map_err(map_err)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let raw = row.get::<String, _>(0);
        let label = format!("{table}.{key}");
        let envelope: Value = serde_json::from_str(&raw)
            .map_err(|e| UserError::new(format!("{label} is not a stored run: {e}")))?;
        let Some(state_json) = decode(&envelope, self.signing_key.as_deref(), &label)? else {
            self.delete(key).await?;
            return Ok(None);
        };
        Ok(Some(to_state(state_json, self.migrate.as_ref())?))
    }

    async fn delete(&self, key: &str) -> Result<(), AgentsError> {
        validate_id(key, "run state key")?;
        let table = &self.table;
        sqlx::query(statement(format!(
            "DELETE FROM {table} WHERE state_key = $1"
        )))
        .bind(key)
        .execute(&self.pool)
        .await
        .map_err(map_err)?;
        Ok(())
    }
}

impl std::fmt::Debug for PostgresRunStateStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresRunStateStore")
            .field("table", &self.table)
            .field("signed", &self.signing_key.is_some())
            .field("has_migration", &self.migrate.is_some())
            .finish()
    }
}

/// Insert `items` on `tx`, one bound row each.
async fn insert_items(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table: &str,
    session_id: &str,
    items: &[Value],
) -> Result<(), AgentsError> {
    for item in items {
        let json = serde_json::to_string(item)
            .map_err(|e| UserError::new(format!("cannot serialize a session item: {e}")))?;
        sqlx::query(statement(format!(
            "INSERT INTO {table} (session_id, item) VALUES ($1, $2::jsonb)"
        )))
        .bind(session_id)
        .bind(json)
        .execute(&mut **tx)
        .await
        .map_err(map_err)?;
    }
    Ok(())
}

/// The key of the advisory lock that guards creating `table`: one lock per kind and table, so two
/// processes creating the same table wait for each other instead of racing.
fn advisory_lock_key(kind: &str, table: &str) -> i64 {
    // FNV-1a, 64 bit: no dependency for a hash that only has to be stable within one program.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in format!("agents:{kind}:{table}").bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash as i64
}

/// Mark a statement built at runtime as audited, which sqlx requires.
///
/// Everything handed to this was built from a literal, a table name that [`validate_table`] has
/// already checked, and values that are bound afterwards — never from anything a caller passed as
/// SQL. Keeping the mark in one function is what makes that claim auditable.
fn statement(sql: String) -> AssertSqlSafe<String> {
    AssertSqlSafe(sql)
}

fn parse_item(line: &str) -> Result<Value, AgentsError> {
    serde_json::from_str(line)
        .map_err(|e| AgentsError::from(UserError::new(format!("a stored item is not JSON: {e}"))))
}

fn map_err(e: sqlx::Error) -> AgentsError {
    AgentsError::from(UserError::new(format!("postgres: {e}")))
}

/// Refuse a table name that is not a plain identifier: it is the one part of a statement that
/// cannot be bound, so it is checked instead.
fn validate_table(name: &str) -> Result<String, AgentsError> {
    let ok = !name.is_empty()
        && name.len() <= 63
        && name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ok {
        return Err(AgentsError::from(UserError::new(format!(
            "{name:?} is not a table name: use letters, digits and underscores, starting with a \
             letter or an underscore, at most 63 characters"
        ))));
    }
    Ok(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_plain_identifier_is_a_table_name() {
        assert_eq!(validate_table("agent_sessions").unwrap(), "agent_sessions");
        assert!(validate_table("_private").is_ok());
        for bad in [
            "",
            "a.b",
            "sessions; DROP TABLE users",
            "sessions\"",
            "1sessions",
            "sessions-1",
            &"x".repeat(64),
        ] {
            assert!(validate_table(bad).is_err(), "`{bad}` must be refused");
        }
    }
}
