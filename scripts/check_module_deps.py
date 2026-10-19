#!/usr/bin/env python3
"""Check the dependency structure of the workspace: between crates, and inside core.

The refactor in `docs/refactor/01-principles.md` puts every module in a tier and forbids a
lower tier from importing a higher one. Two rules are enforced:

1. no upward edges between tiers (`EXCEPTIONS` lists the known ones);
2. the only dependency cycles allowed are the ones in `ALLOWED_CYCLES`. A tier check cannot see
   same-tier cycles, so without this rule a single stray import can silently fold a module
   like `run` into the domain core.

Usage:
    python3 scripts/check_module_deps.py            # fail on violations
    python3 scripts/check_module_deps.py --explain  # print the current edges too

Exit code 1 means a new upward dependency or a new cycle appeared. Either move the type down,
or register it here *with* a `docs/refactor/03-cycle-breaking.md` id and a plan to remove it:
the exception list must only ever shrink.

The crate-level rules (`CRATE_DEPS`, `BANNED_IN_CORE`) are checked against `cargo metadata`:
core depends on no extension, every extension depends on core and nothing else, and core never
declares a network, database or child-process dependency. They are what keeps `core` buildable
without an HTTP client, SQLite or process spawning.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "crates" / "core" / "src"

# Which workspace crates each crate may depend on (normal, build *and* dev dependencies).
#
# Dev dependencies count on purpose: if core's own tests depended on an extension crate that itself
# depends on core, `cargo test` would link two copies of core and trait objects would stop lining
# up (the "double types" problem). Extension crates must never depend on one another either, so
# any of them can be dropped from a build without touching the rest.
CORE = "openai-agents-core"
EXTENSIONS = {
    "openai-agents-macros",
    "openai-agents-mcp",
    "openai-agents-openai",
    "openai-agents-sqlite",
    "openai-agents-testing",
}
FACADE = "openai-agents"
EXAMPLES = "openai-agents-examples"
WEB_DEMO = "openai-agents-web-demo"
CRATE_DEPS: dict[str, set[str]] = {
    CORE: set(),
    "openai-agents-macros": set(),
    "openai-agents-mcp": {CORE},
    "openai-agents-openai": {CORE},
    "openai-agents-sqlite": {CORE},
    "openai-agents-testing": {CORE},
    FACADE: {CORE} | EXTENSIONS,
    # The root `examples/` package is user-facing code: it may reach the facade, but must not reach
    # past it into core or an extension. If an example needs something the facade does not
    # re-export, that is a gap in the public API, not a reason to depend on core directly.
    EXAMPLES: {FACADE},
    # The web demo (`demo/web`) follows the same rule as the examples.
    WEB_DEMO: {FACADE},
}

# Third-party crates core must not depend on, and tokio features it must not enable. They are what
# a build without the OpenAI adapters, MCP or SQLite is meant to be free of.
BANNED_IN_CORE = {"reqwest", "async-openai", "rusqlite", "hyper"}
BANNED_TOKIO_FEATURES_IN_CORE = {"process", "net", "fs", "io-std"}

# Tier 0 has no crate-internal dependencies; each higher tier may depend on any lower tier.
TIERS: dict[str, int] = {
    # 0 — leaves: pure data and pure functions. `error` is here because `items` carries errors.
    "support": 0,
    "error": 0,
    "usage": 0,
    "items": 0,
    "stream_events": 0,
    "extensions": 0,
    # 1 — settings and derived data, plus the cross-cutting infra that only needs tier 0
    "strict_schema": 1,
    "retry": 1,
    "model_settings": 1,
    "tracing": 1,
    # 2 — domain contracts and their runtime data
    "agent_output": 2,
    "run_context": 2,
    "tool_guardrails": 2,
    "guardrail": 2,
    "tool": 2,
    "run_state": 2,
    "agent": 2,
    "handoffs": 2,
    "lifecycle": 2,
    "memory": 2,
    "mcp": 2,
    "model": 2,
    "context": 2,
    "runtime": 2,
    "result": 2,
    # 3 — orchestration
    "run": 3,
    "testing": 3,
}

# Known upward edges. Every entry needs a refactor id and a removal plan. Currently none.
EXCEPTIONS: dict[tuple[str, str], str] = {}

# The dependency cycles that are accepted, as exact module sets.
#
# This is the domain core. `Agent` holds tools, handoffs, guardrails, hooks, the model and MCP
# servers, and the callbacks of every one of those receive `&Agent` (`ToolContext`-style data is
# already out of the way). That mutual reference has the same shape as in the Python SDK; removing
# it would change the signature of every user callback and needs either generics or an
# `AgentView` mirror of `Agent` (see docs/refactor/07-acyclic-evaluation.md).
#
# Things that used to be in this cycle and must stay out of it: `run`, `result`, `context`,
# `memory`, `error`, `retry`, `tracing`, and `run_state` (it only needs plain data from
# `error::payload`). Any module outside this set that joins a cycle is a violation.
ALLOWED_CYCLES: list[frozenset[str]] = [
    frozenset(
        {
            "agent",
            "guardrail",
            "handoffs",
            "lifecycle",
            "mcp",
            "model",
            "tool",
            "tool_guardrails",
        }
    ),
]

# Third-party crates re-exported at the crate root for the `#[function_tool]` macro
# (see `src/lib.rs`), so `crate::schemars::..` is not a module of ours.
REEXPORTED_CRATES = {"schemars", "serde"}

USE_RE = re.compile(r"^\s*(?:pub(?:\(crate\))?\s+)?use\s+crate::([a-z_][a-z0-9_]*)")
PATH_RE = re.compile(r"crate::([a-z_][a-z0-9_]*)::")


def module_of(rel: Path) -> str:
    """`src/run/agent_loop.rs` -> `run`; `src/usage.rs` -> `usage`."""
    parts = rel.with_suffix("").parts
    return parts[0] if len(parts) > 1 else parts[0]


def edges() -> list[tuple[Path, str, str]]:
    out: list[tuple[Path, str, str]] = []
    for path in sorted(SRC.rglob("*.rs")):
        rel = path.relative_to(SRC)
        src_module = module_of(rel)
        # Doc comments mention paths in intra-doc links; those are not compile dependencies.
        code = "\n".join(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("//")
        )
        targets = set()
        for line in code.splitlines():
            m = USE_RE.match(line)
            if m:
                targets.add(m.group(1))
        targets.update(PATH_RE.findall(code))
        for target in sorted(targets):
            if target == src_module or target in REEXPORTED_CRATES:
                continue
            out.append((rel, src_module, target))
    return out


def check_crates() -> list[str]:
    """Violations of the crate-level rules, from `cargo metadata`."""
    try:
        out = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return [f"could not run `cargo metadata`: {error}"]

    packages = {p["name"]: p for p in json.loads(out)["packages"]}
    problems: list[str] = []

    unknown = sorted(set(packages) - set(CRATE_DEPS))
    if unknown:
        problems.append(
            "workspace crates missing from CRATE_DEPS: " + ", ".join(unknown)
            + " (say which workspace crates each may depend on)"
        )
    missing = sorted(set(CRATE_DEPS) - set(packages))
    if missing:
        problems.append("CRATE_DEPS names crates that no longer exist: " + ", ".join(missing))

    for name, package in sorted(packages.items()):
        if name not in CRATE_DEPS:
            continue
        for dep in package["dependencies"]:
            if dep["name"] in packages and dep["name"] not in CRATE_DEPS[name]:
                kind = dep["kind"] or "normal"
                problems.append(
                    f"{name} depends on {dep['name']} ({kind}); it may only depend on "
                    f"{sorted(CRATE_DEPS[name]) or 'no workspace crate'}"
                )

    core = packages.get(CORE)
    if core:
        for dep in core["dependencies"]:
            if dep["name"] in BANNED_IN_CORE:
                problems.append(f"{CORE} declares `{dep['name']}`; that belongs in an extension crate")
            if dep["name"] == "tokio":
                banned = sorted(BANNED_TOKIO_FEATURES_IN_CORE & set(dep["features"]))
                if banned:
                    problems.append(f"{CORE} enables tokio feature(s) {banned}; core must not spawn processes or open sockets")
    return problems


def cycles(all_edges: list[tuple[Path, str, str]]) -> list[frozenset[str]]:
    """Strongly connected components with more than one module (Tarjan)."""
    graph: dict[str, set[str]] = {}
    for _, src_module, target in all_edges:
        graph.setdefault(src_module, set()).add(target)
        graph.setdefault(target, set())

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    found: list[frozenset[str]] = []
    counter = [0]

    def visit(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph[v]):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                found.append(frozenset(comp))

    for v in sorted(graph):
        if v not in index:
            visit(v)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain", action="store_true", help="print every edge with its tier")
    args = parser.parse_args()

    violations: list[str] = []
    unknown: set[str] = set()
    all_edges = edges()
    if not SRC.is_dir() or not all_edges:
        # A gate that checks nothing must not report success: it means the source moved.
        print(f"no module dependencies found under {SRC}; did the sources move?", file=sys.stderr)
        return 1

    for rel, src_module, target in all_edges:
        if src_module not in TIERS:
            unknown.add(src_module)
            continue
        if target not in TIERS:
            unknown.add(target)
            continue
        if TIERS[target] <= TIERS[src_module]:
            if args.explain:
                print(f"ok   {rel.as_posix()}: {src_module} -> {target}")
            continue
        if (src_module, target) in EXCEPTIONS:
            if args.explain:
                print(
                    f"known {rel.as_posix()}: {src_module} -> {target} "
                    f"({EXCEPTIONS[(src_module, target)]})"
                )
            continue
        violations.append(f"{rel.as_posix()}: {src_module} (tier {TIERS[src_module]}) -> "
                          f"{target} (tier {TIERS[target]})")

    if unknown:
        print("unknown modules, add them to TIERS: " + ", ".join(sorted(unknown)), file=sys.stderr)
        return 1

    for comp in cycles(all_edges):
        if comp not in ALLOWED_CYCLES:
            extra = sorted(comp - set().union(*ALLOWED_CYCLES))
            violations.append(
                "unexpected dependency cycle: " + ", ".join(sorted(comp))
                + (f" (not part of an allowed cycle: {', '.join(extra)})" if extra else "")
            )

    if violations:
        print("module dependency violations:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nFix by moving the type down a tier, or register it in EXCEPTIONS / ALLOWED_CYCLES "
            "with a docs/refactor id.",
            file=sys.stderr,
        )
        return 1

    crate_problems = check_crates()
    if crate_problems:
        print("crate dependency violations:", file=sys.stderr)
        for problem in crate_problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if not args.explain:
        print("module deps ok (crates and modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
