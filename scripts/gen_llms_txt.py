#!/usr/bin/env python3
"""Write `llms.txt` and `llms-full.txt`: the docs as an index and as one flat file.

`llms.txt` is short and hand-written here, because what a coding assistant needs first is the
*shape* of the API, not a dump. `llms-full.txt` is generated: the index plus the documents it points
at, in the order a reader would work through them.

`--check` fails when either file is out of date, so a change to a linked document has to be
regenerated in the same commit (CI runs it).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (path, one-line description). The order is the order in `llms-full.txt`.
DOCS = [
    ("README.md", "quick start, crates and features, examples"),
    ("docs/COMPAT.md", "support matrix: what the Python SDK has and what happens here"),
    ("docs/DEVIATIONS.md", "every intentional difference, with an id (D-… / X-…)"),
    ("docs/MIGRATION-FROM-PYTHON.md", "Python API name → Rust API name"),
    ("docs/COMPARISON.md", "how this compares with the Python SDK, Rig and ADK-Rust"),
    ("STABILITY.md", "what is stable, and the deprecation policy"),
]

INDEX = """# OpenAI Agents SDK for Rust

> A Rust port of the OpenAI Agents Python SDK: agents, a runner with tools, handoffs, guardrails,
> sessions, tracing and human-in-the-loop, provider-neutral. Behaviours are pinned to the Python
> implementation v0.23.1 and checked against it by an oracle.

## Install

```toml
opai-agents = "0.2"                                                  # core + OpenAI
opai-agents = { version = "0.2", features = ["providers", "tools"] } # presets, offline model, tools
```

```rust
use opai_agents::prelude::*;

let agent = Agent::new("Assistant").instructions("Be brief.");
let result = Runner::run(&agent, "Hello", RunOptions::new()).await?;
println!("{}", result.final_output_as_str().unwrap_or_default());
```

`opai_agents::run("instructions", "input")` picks the provider from the environment; with
`AGENTS_OFFLINE=1` it answers locally and needs no key.

## The shape of the API

- `Agent`: `instructions`, `tools`, `handoffs`, `model` / `model_name`, `model_settings`,
  `output_type`, `input_guardrails` / `output_guardrails`, `hooks`, `capability`.
- `Runner::run(agent, input, RunOptions)`, `run_streamed`, `run_blocking`, `run_state`.
- `RunOptions`: builder. `.temperature(..)`, `.max_turns(..)`, `.reasoning(..)`, `.session(..)`,
  `.provider(..)`, `.resolve_with(closure)`, `.with(value)` for typed context.
- Tools: `#[function_tool]` over `async fn`, or `FunctionTool::typed` over a `schemars` struct, or
  the `Tool` trait for something custom.
- Pausing: `FunctionTool::with_needs_resolution(true)` + `RunResult::to_state()` + `RunState::approve`
  / `reject` / `provide_output` + `Runner::run_state`.
- Guardrails: `input_guardrail` / `output_guardrail` closures returning
  `GuardrailFunctionOutput`, whose verdict is `Pass`, `Fail { reason, severity }` or `Transform`.
- Bundles: `CapabilityParts::new().tool(..).instructions(..).named("x")`, then `.capability(..)`.
  `.deferred("description")` keeps it out of the prompt until the model calls `load_capability`.
- Contracts to implement: `Tool`, `Model`, `ModelProvider`, `Session`, `TracingProcessor`,
  `Capability`, `RunStateStore`.

## Crates

`opai-agents` re-exports everything. Under it: `core` (no HTTP), `macros`, `testing`, `openai`,
`providers`, `mcp`, `sqlite`, `sessions`, `observability`, `tools`, `guardrails`.

## Docs

"""


def index_text() -> str:
    out = [INDEX]
    for path, description in DOCS:
        out.append(f"- [{path}]({path}): {description}\n")
    out.append(
        "\n## Optional\n\n"
        "- `cargo opai new <name> --template <chat|tools|hitl>` scaffolds a project.\n"
        "- `conformance/scenarios/` holds the declarative parity scenarios.\n"
    )
    return "".join(out)


def full_text() -> str:
    parts = [index_text(), "\n---\n\n"]
    for path, _ in DOCS:
        body = (ROOT / path).read_text(encoding="utf-8").strip()
        parts.append(f"# Source: {path}\n\n{body}\n\n---\n\n")
    return "".join(parts)


def main() -> int:
    targets = {
        "llms.txt": index_text(),
        "llms-full.txt": full_text(),
    }
    if "--check" in sys.argv[1:]:
        stale = [
            name
            for name, text in targets.items()
            if not (ROOT / name).is_file() or (ROOT / name).read_text(encoding="utf-8") != text
        ]
        if stale:
            print("out of date: " + ", ".join(stale) + " (run scripts/gen_llms_txt.py)")
            return 1
        print("llms.txt ok")
        return 0
    for name, text in targets.items():
        (ROOT / name).write_text(text, encoding="utf-8")
        print(f"wrote {name} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
