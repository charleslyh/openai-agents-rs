//! The function tools of one turn: resolving name collisions, planning which calls to make,
//! and executing them (Python: `run_internal/tool_planning.py` and `tool_execution.py`).
//!
//! - [`plan`] decides which calls run, which are already answered and which need approval
//! - [`execute`] runs the planned calls, with guardrails, timeouts and the concurrency window
//! - [`output`] turns results into run items and finds them again

mod execute;
mod output;
mod plan;

pub(crate) use execute::execute_planned_tools;
pub(crate) use output::{
    finalize_tool_output, has_tool_output, push_tool_output, tool_output_item, value_to_tool_string,
};
pub(crate) use plan::{
    plan_tool_calls, resolve_tool_name_collisions, tools_for_agent, validate_tool_timeout,
    ToolPlan, MULTIPLE_HANDOFFS_MESSAGE,
};
