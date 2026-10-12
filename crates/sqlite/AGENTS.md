# openai-agents-sqlite

`SqliteSession`: file or in-memory conversation storage, with the same schema as the Python SDK so
a session can be read from either side. Reached through the facade's `sqlite` feature.

The point of this crate is `rusqlite`: it keeps a database driver out of `openai-agents-core`, so a
build that never persists a session does not compile one.

Verification: `cargo test --features sqlite --test sqlite_session` (needs the Python venv for the
cross-language schema check; skips without it).
