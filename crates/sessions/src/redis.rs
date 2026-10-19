//! A session and a paused run in Redis (Rust-only, X-017 / X-016).
//!
//! The backend for a history more than one process or machine shares. A session is one list,
//! oldest first; a paused run is one key holding the same envelope the file store writes, so
//! integrity, expiry and schema are decided once (see the shared `state` module) and behave the same here.
//!
//! Two things the commands are chosen for:
//!
//! * **A reader never sees a half history.** `replace_items` — what compaction calls — runs as one
//!   `MULTI` transaction: the list is deleted and refilled while no other command can interleave.
//! * **A history does not grow forever.** [`RedisSession::with_ttl`] sets a TTL on the list, and
//!   [`RunStateStore::save`] takes one per state.
//!
//! The connection is multiplexed, so one of them serves every task of the process.

use std::time::Duration;

use async_trait::async_trait;
use redis::aio::MultiplexedConnection;
use redis::{Client, RedisError};
use serde_json::Value;

use openai_agents_core::error::UserError;
use openai_agents_core::{
    AgentsError, RunState, RunStateMigration, RunStateStore, Session, SessionSettings,
};

use super::state::{decode, key_from_env, to_state, validate_id, wrap};

/// Where a session's list is kept, before the session id.
pub const SESSION_KEY_PREFIX: &str = "agents:session";

/// Where a paused run is kept, before its key.
pub const STATE_KEY_PREFIX: &str = "agents:state";

/// Where the signing key for paused runs is read from.
pub const STATE_SIGNING_KEY_ENV: &str = "AGENTS_STATE_KEY";

/// A conversation in a Redis list.
pub struct RedisSession {
    id: String,
    key: String,
    conn: MultiplexedConnection,
    ttl: Option<Duration>,
}

impl RedisSession {
    /// Connect to `url` and use the session `session_id` there.
    pub async fn open(url: &str, session_id: impl Into<String>) -> Result<Self, AgentsError> {
        let client = Client::open(url).map_err(|e| {
            AgentsError::from(UserError::new(format!(
                "cannot use the Redis URL {url}: {e}"
            )))
        })?;
        Self::with_client(&client, session_id, None).await
    }

    /// Use an existing client, so a deployment can share one connection with the rest of its code.
    pub async fn with_client(
        client: &Client,
        session_id: impl Into<String>,
        ttl: Option<Duration>,
    ) -> Result<Self, AgentsError> {
        let id = session_id.into();
        validate_id(&id, "session id")?;
        let conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(map_err)?;
        Ok(Self {
            key: format!("{SESSION_KEY_PREFIX}:{id}"),
            id,
            conn,
            ttl,
        })
    }

    /// Expire the history `ttl` after the last write, so a forgotten session does not stay forever.
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// The key this session's list is kept under.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The connection, for a caller that needs to do something else with it.
    pub fn connection(&self) -> MultiplexedConnection {
        self.conn.clone()
    }
}

#[async_trait]
impl Session for RedisSession {
    fn session_id(&self) -> &str {
        &self.id
    }

    fn session_settings(&self) -> Option<SessionSettings> {
        None
    }

    async fn get_items(&self, limit: Option<usize>) -> Result<Vec<Value>, AgentsError> {
        let raw: Vec<String> = match limit {
            Some(0) => Vec::new(),
            // A limit takes the newest items, which is the tail of the list.
            Some(n) => redis::cmd("LRANGE")
                .arg(&self.key)
                .arg(-(n as i64))
                .arg(-1)
                .query_async(&mut self.conn.clone())
                .await
                .map_err(map_err)?,
            None => redis::cmd("LRANGE")
                .arg(&self.key)
                .arg(0)
                .arg(-1)
                .query_async(&mut self.conn.clone())
                .await
                .map_err(map_err)?,
        };
        decode_items(raw)
    }

    async fn add_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        if items.is_empty() {
            return Ok(());
        }
        let encoded = encode_items(&items)?;
        let mut pipe = redis::pipe();
        pipe.atomic().rpush(&self.key, encoded);
        if let Some(secs) = ttl_seconds(self.ttl) {
            pipe.expire(&self.key, secs);
        }
        pipe.query_async::<()>(&mut self.conn.clone())
            .await
            .map_err(map_err)
    }

    async fn pop_item(&self) -> Result<Option<Value>, AgentsError> {
        let raw: Option<String> = redis::cmd("RPOP")
            .arg(&self.key)
            .query_async(&mut self.conn.clone())
            .await
            .map_err(map_err)?;
        match raw {
            Some(line) => Ok(Some(parse_item(&line)?)),
            None => Ok(None),
        }
    }

    async fn clear_session(&self) -> Result<(), AgentsError> {
        redis::cmd("DEL")
            .arg(&self.key)
            .query_async::<()>(&mut self.conn.clone())
            .await
            .map_err(map_err)
    }

    /// One transaction, so a reader sees the old history or the new one and never a mix.
    async fn replace_items(&self, items: Vec<Value>) -> Result<(), AgentsError> {
        let mut pipe = redis::pipe();
        pipe.atomic().del(&self.key);
        if !items.is_empty() {
            pipe.rpush(&self.key, encode_items(&items)?);
            if let Some(secs) = ttl_seconds(self.ttl) {
                pipe.expire(&self.key, secs);
            }
        }
        pipe.query_async::<()>(&mut self.conn.clone())
            .await
            .map_err(map_err)
    }
}

impl std::fmt::Debug for RedisSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisSession")
            .field("id", &self.id)
            .field("key", &self.key)
            .field("ttl", &self.ttl)
            .finish()
    }
}

/// A paused run in a Redis key.
pub struct RedisRunStateStore {
    conn: MultiplexedConnection,
    /// Signing key, when one is configured.
    signing_key: Option<Vec<u8>>,
    migrate: Option<RunStateMigration>,
}

impl RedisRunStateStore {
    /// Connect to `url`, signing states when `AGENTS_STATE_KEY` is set.
    pub async fn from_env(url: &str) -> Result<Self, AgentsError> {
        let signing_key = key_from_env(STATE_SIGNING_KEY_ENV);
        Self::open(url, signing_key).await
    }

    /// Connect to `url` without signing. For tests, and for a Redis only this process can write.
    pub async fn unsigned(url: &str) -> Result<Self, AgentsError> {
        Self::open(url, None).await
    }

    async fn open(url: &str, signing_key: Option<Vec<u8>>) -> Result<Self, AgentsError> {
        let client = Client::open(url).map_err(|e| {
            AgentsError::from(UserError::new(format!(
                "cannot use the Redis URL {url}: {e}"
            )))
        })?;
        let conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(map_err)?;
        Ok(Self {
            conn,
            signing_key,
            migrate: None,
        })
    }

    /// Adapt states written by an older build before they are resumed.
    #[must_use]
    pub fn with_migration(mut self, migrate: RunStateMigration) -> Self {
        self.migrate = Some(migrate);
        self
    }

    /// Whether states are signed.
    pub fn is_signed(&self) -> bool {
        self.signing_key.is_some()
    }
}

#[async_trait]
impl RunStateStore for RedisRunStateStore {
    async fn save(
        &self,
        key: &str,
        state: &RunState,
        ttl: Option<Duration>,
    ) -> Result<(), AgentsError> {
        validate_id(key, "run state key")?;
        let envelope = wrap(state, ttl, self.signing_key.as_deref())?;
        let json = serde_json::to_string(&envelope)
            .map_err(|e| UserError::new(format!("cannot serialize a paused run: {e}")))?;
        let redis_key = format!("{STATE_KEY_PREFIX}:{key}");
        let mut pipe = redis::pipe();
        pipe.atomic().set(&redis_key, json);
        if let Some(secs) = ttl_seconds(ttl) {
            pipe.expire(&redis_key, secs);
        }
        pipe.query_async::<()>(&mut self.conn.clone())
            .await
            .map_err(map_err)
    }

    async fn load(&self, key: &str) -> Result<Option<RunState>, AgentsError> {
        validate_id(key, "run state key")?;
        let redis_key = format!("{STATE_KEY_PREFIX}:{key}");
        let raw: Option<String> = redis::cmd("GET")
            .arg(&redis_key)
            .query_async(&mut self.conn.clone())
            .await
            .map_err(map_err)?;
        let Some(raw) = raw else {
            return Ok(None);
        };
        let envelope: Value = serde_json::from_str(&raw)
            .map_err(|e| UserError::new(format!("{redis_key} is not a stored run: {e}")))?;
        let Some(state_json) = decode(&envelope, self.signing_key.as_deref(), &redis_key)? else {
            // Expired: it reads as absent, and is removed.
            self.delete(key).await?;
            return Ok(None);
        };
        Ok(Some(to_state(state_json, self.migrate.as_ref())?))
    }

    async fn delete(&self, key: &str) -> Result<(), AgentsError> {
        validate_id(key, "run state key")?;
        let redis_key = format!("{STATE_KEY_PREFIX}:{key}");
        redis::cmd("DEL")
            .arg(&redis_key)
            .query_async::<()>(&mut self.conn.clone())
            .await
            .map_err(map_err)
    }
}

impl std::fmt::Debug for RedisRunStateStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisRunStateStore")
            .field("signed", &self.signing_key.is_some())
            .field("has_migration", &self.migrate.is_some())
            .finish()
    }
}

fn ttl_seconds(ttl: Option<Duration>) -> Option<i64> {
    ttl.map(|d| d.as_secs().max(1) as i64)
}

fn encode_items(items: &[Value]) -> Result<Vec<String>, AgentsError> {
    items
        .iter()
        .map(|item| {
            serde_json::to_string(item)
                .map_err(|e| UserError::new(format!("cannot serialize a session item: {e}")).into())
        })
        .collect()
}

fn decode_items(raw: Vec<String>) -> Result<Vec<Value>, AgentsError> {
    raw.iter().map(|line| parse_item(line)).collect()
}

fn parse_item(line: &str) -> Result<Value, AgentsError> {
    serde_json::from_str(line)
        .map_err(|e| UserError::new(format!("a stored item is not JSON: {e}")).into())
}

fn map_err(e: RedisError) -> AgentsError {
    UserError::new(format!("redis: {e}")).into()
}
