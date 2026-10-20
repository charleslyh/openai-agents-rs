//! The zero-configuration entry point: `opai_agents::run` takes the provider from the
//! environment (Rust-only, X-015).

#![cfg(feature = "providers")]

/// Both halves live in one test on purpose: `cargo test` runs the tests of a binary in parallel
/// threads, and the environment is process-wide.
#[tokio::test]
async fn without_a_key_it_says_what_to_set_and_offline_it_answers() {
    for var in ["AGENTS_OFFLINE", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"] {
        std::env::remove_var(var);
    }
    std::env::remove_var("AGENTS_PROVIDER_1_ENDPOINT");

    // Nothing configured: the error names the variables that would fix it.
    let error = opai_agents::run("You answer.", "Hello")
        .await
        .expect_err("no provider is configured");
    let message = error.to_string();
    assert!(message.contains("DEEPSEEK_API_KEY"), "{message}");
    assert!(message.contains("AGENTS_PROVIDER_1_ENDPOINT"), "{message}");

    // Offline, one line is enough: the run completes without a key and without the network.
    std::env::set_var("AGENTS_OFFLINE", "1");
    let result = opai_agents::run("You answer in one sentence.", "What is Rust?").await;
    std::env::remove_var("AGENTS_OFFLINE");

    let result = result.expect("the offline model answers");
    let text = result.final_output_as_str().unwrap_or_default();
    assert!(text.contains("offline"), "{text}");
}
