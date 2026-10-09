#!/usr/bin/env python3
"""Check that `src/` modules only depend downward.

The refactor in `docs/refactor/01-principles.md` puts every module in a tier and forbids a
lower tier from importing a higher one. This is the mechanical gate for that rule.

Usage:
    python3 scripts/check_module_deps.py            # fail on violations
    python3 scripts/check_module_deps.py --explain  # print the current edges too

Exit code 1 means a new upward dependency appeared. Either move the type down a tier, or add an
exception here *with* a `docs/refactor/03-cycle-breaking.md` id and a plan to remove it: the
exception list must only ever shrink.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Tier 0 has no crate-internal dependencies; each higher tier may depend on any lower tier.
TIERS: dict[str, int] = {
    # 0 — leaves: pure data and pure functions. `error` is here because `items` carries errors.
    "support": 0,
    "error": 0,
    "usage": 0,
    "items": 0,
    "stream_events": 0,
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
    "result": 2,
    # 3 — orchestration
    "run": 3,
    "testing": 3,
}

# Known upward edges. Every entry needs a refactor id and a removal plan.
EXCEPTIONS: dict[tuple[str, str], str] = {
    ("agent", "mcp"): "C6: agent may only use the mcp contract, not client/transport/manager",
    ("agent", "run"): "C3: as_tool; the runner entry point has to move under core",
    ("tool", "run_state"): "C4: nested RunState in ToolResult",
}

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
        text = path.read_text(encoding="utf-8")
        targets = set()
        for line in text.splitlines():
            m = USE_RE.match(line)
            if m:
                targets.add(m.group(1))
        targets.update(PATH_RE.findall(text))
        for target in sorted(targets):
            if target == src_module or target in REEXPORTED_CRATES:
                continue
            out.append((rel, src_module, target))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain", action="store_true", help="print every edge with its tier")
    args = parser.parse_args()

    violations: list[str] = []
    unknown: set[str] = set()

    for rel, src_module, target in edges():
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

    if violations:
        print("upward module dependencies:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nFix by moving the type down a tier, or register it in EXCEPTIONS with a "
            "docs/refactor id.",
            file=sys.stderr,
        )
        return 1

    if not args.explain:
        print("module deps ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
