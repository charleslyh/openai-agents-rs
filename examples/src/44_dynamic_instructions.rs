//! Instructions generated per run
//!
//! A fixed `instructions("...")` string is the same for everyone. `dynamic_instructions` takes a
//! function of the run context and the agent, called at the start of each model turn, so one
//! agent definition serves every user: their name, language, plan, or whatever your service knows
//! and the model should act on.
//!
//! The context is a value you pass with `RunOptions::context`; the model never sees it, only the
//! prompt text built from it. Keep secrets out of the prompt text for the same reason you would
//! keep them out of a log: whatever the function returns is sent to the provider.
//!
//! The model is scripted so the example needs no key and can print the exact system prompt each
//! run sent.
//!
//! ```bash
//! cargo run -p opai-agents-examples --example dynamic_instructions
//! ```

use std::sync::Arc;

use opai_agents::testing::{ItemHelpers, ModelStep, ScriptedModel};
use opai_agents::{Agent, Model, RunOptions, Runner};

/// What the service knows about the caller.
struct User {
    name: &'static str,
    language: &'static str,
    plan: &'static str,
}

fn support_agent(model: Arc<dyn Model>) -> Agent {
    Agent::new("Support")
        .model(model)
        .dynamic_instructions(|ctx, agent| {
            // Called with the run context and the agent; here it is the user passed to the run.
            let prompt = match ctx.context::<User>() {
                Some(user) => {
                    let mut prompt = format!(
                        "You are {}, a support agent. You are talking to {}. Reply in {}.",
                        agent.name, user.name, user.language
                    );
                    if user.plan == "enterprise" {
                        prompt.push_str(" This customer has priority support: offer a call.");
                    }
                    prompt
                }
                None => format!(
                    "You are {}, a support agent. Ask who you are talking to.",
                    agent.name
                ),
            };
            async move { prompt }
        })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let users = [
        Some(User {
            name: "Mina",
            language: "Korean",
            plan: "free",
        }),
        Some(User {
            name: "Jonas",
            language: "German",
            plan: "enterprise",
        }),
        None,
    ];

    for user in users {
        let model = Arc::new(ScriptedModel::new([ModelStep::from(
            ItemHelpers::text_message("(reply)"),
        )]));
        let agent = support_agent(Arc::clone(&model) as Arc<dyn Model>);
        let mut options = RunOptions::new();
        let label = user.as_ref().map_or("no user", |u| u.name);
        if let Some(user) = user {
            options = options.context(user);
        }
        Runner::run(&agent, "I need help.", options).await?;
        println!(
            "[{label}]\n  system prompt: {}\n",
            model
                .first_call()
                .unwrap()
                .system_instructions
                .unwrap_or_default()
        );
    }
    Ok(())
}
