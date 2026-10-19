//! Tool middleware: wrap a tool's invocation to rewrite its call or its result
//! (Rust-only; Python has no equivalent — the closest are `tool_input_guardrails` /
//! `tool_output_guardrails`, which only ever *replace* a call with a message or *reject* a
//! result, never arbitrarily transform the call).
//!
//! [`RunHooks`](crate::lifecycle::RunHooks) and [`AgentHooks`](crate::lifecycle::AgentHooks) are
//! read-only observers: they cannot change what runs. A [`ToolMiddleware`] sits *around* the
//! call instead (auth, rate limiting, audit logging, caching), and can skip the inner tool
//! entirely (returning a cached [`ToolOutput`] without calling `next`).
//!
//! ```
//! use std::sync::Arc;
//! use async_trait::async_trait;
//! use openai_agents_core::error::AgentsError;
//! use openai_agents_core::run_context::ToolContext;
//! use openai_agents_core::tool::{FunctionTool, ToolOutput};
//! use openai_agents_core::tool_middleware::{NextTool, ToolMiddleware};
//!
//! struct CallCounter(std::sync::atomic::AtomicUsize);
//!
//! #[async_trait]
//! impl ToolMiddleware for CallCounter {
//!     async fn handle(&self, ctx: ToolContext, next: NextTool) -> Result<ToolOutput, AgentsError> {
//!         self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//!         next(ctx).await
//!     }
//! }
//!
//! let tool = FunctionTool::constant("ping", "replies pong", "pong")
//!     .with_middleware(CallCounter(std::sync::atomic::AtomicUsize::new(0)));
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::AgentsError;
use crate::run_context::ToolContext;
use crate::tool::{FunctionTool, ToolOutput};

/// The rest of the chain (the inner middleware, or the tool body itself), captured as a plain
/// callback so [`ToolMiddleware::handle`] can call it zero or more times — or not at all, to
/// short-circuit the call (e.g. serve a cached [`ToolOutput`]).
pub type NextTool = Arc<
    dyn Fn(ToolContext) -> Pin<Box<dyn Future<Output = Result<ToolOutput, AgentsError>> + Send>>
        + Send
        + Sync,
>;

/// Wraps a tool's invocation (Python has no equivalent).
///
/// Composed with [`FunctionTool::with_middleware`], onion-style: the last middleware added is
/// the outermost one, so it sees the call first and the result last.
#[async_trait]
pub trait ToolMiddleware: Send + Sync {
    /// Handle one call. Call `next(ctx)` to continue the chain (and read or rewrite what it
    /// returns), or return without calling it to skip the tool entirely.
    async fn handle(&self, ctx: ToolContext, next: NextTool) -> Result<ToolOutput, AgentsError>;
}

impl FunctionTool {
    /// Wrap this tool's invocation with `middleware`.
    ///
    /// The tool's `on_invoke_tool` becomes the innermost link of the chain; `middleware` runs
    /// around it (and around any middleware added earlier). `ctx.tool_arguments` carries the raw
    /// JSON arguments the inner call will parse.
    #[must_use]
    pub fn with_middleware<M>(mut self, middleware: M) -> Self
    where
        M: ToolMiddleware + 'static,
    {
        let middleware = Arc::new(middleware);
        let inner = Arc::clone(&self.on_invoke_tool);
        self.on_invoke_tool = Arc::new(move |ctx, _args| {
            let middleware = Arc::clone(&middleware);
            let inner = Arc::clone(&inner);
            Box::pin(async move {
                let next: NextTool = Arc::new(move |ctx: ToolContext| {
                    let inner = Arc::clone(&inner);
                    let args = ctx.tool_arguments.clone();
                    Box::pin(async move { inner(ctx, args).await })
                });
                middleware.handle(ctx, next).await
            })
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::{json, Value};

    use super::*;
    use crate::run_context::RunContextWrapper;

    struct CountingAuth {
        calls: AtomicUsize,
        allow: bool,
    }

    #[async_trait]
    impl ToolMiddleware for CountingAuth {
        async fn handle(
            &self,
            ctx: ToolContext,
            next: NextTool,
        ) -> Result<ToolOutput, AgentsError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if !self.allow {
                return Ok(ToolOutput::output(Value::String("denied".into())));
            }
            next(ctx).await
        }
    }

    fn ctx(args: &str) -> ToolContext {
        ToolContext::new("probe", "call-1", args, RunContextWrapper::default())
    }

    #[tokio::test]
    async fn middleware_runs_before_the_tool_body() {
        let mw = CountingAuth {
            calls: AtomicUsize::new(0),
            allow: true,
        };
        let tool = FunctionTool::constant("probe", "d", "ok").with_middleware(mw);
        let out = (tool.on_invoke_tool)(ctx("{}"), "{}".into()).await.unwrap();
        assert_eq!(out.value(), Some(&Value::String("ok".into())));
    }

    #[tokio::test]
    async fn middleware_can_short_circuit_without_calling_next() {
        let mw = CountingAuth {
            calls: AtomicUsize::new(0),
            allow: false,
        };
        let tool = FunctionTool::constant("probe", "d", "ok").with_middleware(mw);
        let out = (tool.on_invoke_tool)(ctx("{}"), "{}".into()).await.unwrap();
        assert_eq!(out.value(), Some(&Value::String("denied".into())));
    }

    #[tokio::test]
    async fn middleware_composes_outermost_last_added() {
        let order: Arc<std::sync::Mutex<Vec<&'static str>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        struct Tag(Arc<std::sync::Mutex<Vec<&'static str>>>, &'static str);
        #[async_trait]
        impl ToolMiddleware for Tag {
            async fn handle(
                &self,
                ctx: ToolContext,
                next: NextTool,
            ) -> Result<ToolOutput, AgentsError> {
                self.0.lock().unwrap().push(self.1);
                next(ctx).await
            }
        }

        let tool = FunctionTool::constant("probe", "d", "ok")
            .with_middleware(Tag(Arc::clone(&order), "inner"))
            .with_middleware(Tag(Arc::clone(&order), "outer"));
        let _ = (tool.on_invoke_tool)(ctx("{}"), "{}".into()).await.unwrap();
        assert_eq!(*order.lock().unwrap(), vec!["outer", "inner"]);
    }

    #[tokio::test]
    async fn middleware_sees_tool_arguments_from_context() {
        struct EchoArgsAsOutput;
        #[async_trait]
        impl ToolMiddleware for EchoArgsAsOutput {
            async fn handle(
                &self,
                ctx: ToolContext,
                _next: NextTool,
            ) -> Result<ToolOutput, AgentsError> {
                Ok(ToolOutput::output(json!({ "seen": ctx.tool_arguments })))
            }
        }
        let tool = FunctionTool::constant("probe", "d", "ok").with_middleware(EchoArgsAsOutput);
        let out = (tool.on_invoke_tool)(ctx(r#"{"a":1}"#), r#"{"a":1}"#.into())
            .await
            .unwrap();
        assert_eq!(out.value(), Some(&json!({ "seen": r#"{"a":1}"# })));
    }
}
