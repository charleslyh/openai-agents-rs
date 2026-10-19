import re


def sub(p, old, new, count=1):
    s = open(p).read()
    assert old in s, (p, old[:80])
    s = s.replace(old, new, count)
    open(p, 'w').write(s)


# 22_tracing: no RunConfig literal
sub('examples/src/22_tracing.rs', """    let mut run_config = RunConfig::default();
    run_config.tracing.workflow_name = Some("tracing-example".into());
    let mut options = RunOptions::default();
    options.run_config = run_config;
""", """    let options = RunOptions::new().workflow_name("tracing-example");
""")
s = open('examples/src/22_tracing.rs').read()
s = re.sub(r"\nuse openai_agents::\{function_tool, Agent, RunConfig, RunOptions, Runner\};",
           "\nuse openai_agents::\{function_tool, Agent, RunOptions, Runner\};", s)
open('examples/src/22_tracing.rs', 'w').write(s)

# 06_parallel_tools
sub('examples/src/06_parallel_tools.rs', """    let mut options = RunOptions::new().tune(|s| s.parallel_tool_calls = Some(true));
    options.run_config.tools = ToolOptions::new().execution(ToolExecutionConfig {
        max_function_tool_concurrency: Some(4),
    });
""", """    let options = RunOptions::new()
        .tune(|s| s.parallel_tool_calls = Some(true))
        .tools(|t| t.execution(ToolExecutionConfig::new().concurrency(4)));
""")

# 19_run_context
sub('examples/src/19_run_context.rs', """    let mut options = RunOptions::default();
    options.context = Some(Arc::new(Tenant {
""", """    let options = RunOptions::new().context(Tenant {
""")

# 03_memory / 04_compaction
sub('examples/src/03_memory.rs', """    let mut options = RunOptions::default();
    options.session = Some(Arc::clone(&session));
""", """    let options = RunOptions::new().session(Arc::clone(&session));
""")
sub('examples/src/04_compaction.rs', """    let mut options = RunOptions::default();
    options.session = Some(Arc::clone(&session));
""", """    let options = RunOptions::new().session(Arc::clone(&session));
""")

# 20_hooks
sub('examples/src/20_hooks.rs', """    let mut options = RunOptions::default();
    options.hooks = Some(Arc::new(PrintRunHooks));
""", """    let options = RunOptions::new().hooks(Arc::new(PrintRunHooks));
""")
print("ok")
