//! `use opai_agents::prelude::*` is enough for an agent with a macro tool, a run and a result.

use std::sync::Arc;

use opai_agents::prelude::*;
use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};

/// Greet someone.
#[function_tool]
fn greet(name: String) -> String {
    format!("hello {name}")
}

#[tokio::test]
async fn the_prelude_covers_a_basic_program() {
    // The prelude's `Runner` is the facade one (installs the default provider), not core's.
    let _: opai_agents::Runner = Runner;

    let model = Arc::new(ScriptedModel::new([ModelStep::from(
        ItemHelpers::text_message("done"),
    )]));
    let agent = Agent::new("a")
        .instructions("Be brief.")
        .model(model.clone())
        .tools(vec![greet()]);
    let result: RunResult = Runner::run(&agent, "hi", RunOptions::new().max_turns(2))
        .await
        .expect("run");

    assert_eq!(result.final_output_as_str(), Some("done"));
    assert_eq!(model.last_call().expect("called").tool_names, vec!["greet"]);
}
