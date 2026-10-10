//! The payload handed to a model, and the filter that may rewrite it
//! (Python: `ModelInputData`, `CallModelData`).

use std::sync::Arc;

use serde_json::Value;

use crate::agent::Agent;
use crate::error::AgentsError;
use crate::run_context::RunContextWrapper;

/// The input and instructions about to be sent to the model
/// (Python: `ModelInputData`).
#[derive(Debug, Clone)]
pub struct ModelInputData {
    /// Responses input items.
    pub input: Vec<Value>,
    /// System instructions.
    pub instructions: Option<String>,
}

/// Payload given to [`RunConfig::call_model_input_filter`](crate::run::RunConfig::call_model_input_filter)
/// (Python: `CallModelData`).
#[derive(Clone)]
pub struct CallModelData {
    /// The model input as prepared by the runner.
    pub model_data: ModelInputData,
    /// The agent about to be called.
    pub agent: Arc<Agent>,
    /// The run context.
    pub context: RunContextWrapper,
}

/// Rewrites the model input right before each model call (Python: `CallModelInputFilter`).
///
/// The filtered data is used for this call only; the run history is unchanged.
pub type CallModelInputFilter = Arc<
    dyn Fn(
            CallModelData,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ModelInputData, AgentsError>> + Send>,
        > + Send
        + Sync,
>;
