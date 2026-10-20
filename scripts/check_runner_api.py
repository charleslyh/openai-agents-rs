#!/usr/bin/env python3
"""Check that the facade `Runner` mirrors `opai_agents_core::Runner`.

Core knows no model provider, so `crates/facade/src/runner.rs` wraps every `Runner` method to
install the default OpenAI provider first. A method added to the core runner but not wrapped would
simply not exist for users of `opai-agents` (a compile error, not a silent change), and a changed
signature would drift. This script compares the two lists and fails on any difference, so the
wrapper cannot fall behind unnoticed.

Usage:
    python3 scripts/check_runner_api.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "crates" / "core" / "src" / "run" / "mod.rs"
FACADE = ROOT / "crates" / "facade" / "src" / "runner.rs"

# `pub fn` / `pub async fn` inside `impl Runner { ... }`, up to the opening brace of the body.
METHOD_RE = re.compile(r"^    (pub (?:async )?fn \w+\([^{]*?)\s*\{", re.MULTILINE | re.DOTALL)


def runner_impl(text: str) -> str:
    """The text of `impl Runner { ... }` (the first one), found by brace matching."""
    start = text.find("impl Runner {")
    if start < 0:
        raise SystemExit("no `impl Runner {` found")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise SystemExit("unbalanced braces in `impl Runner`")


def signatures(path: Path) -> dict[str, str]:
    body = runner_impl(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for match in METHOD_RE.finditer(body):
        signature = re.sub(r"\s+", " ", match.group(1)).replace("( ", "(").replace(", )", ")")
        signature = signature.replace(",)", ")")
        name = re.search(r"fn (\w+)", signature).group(1)
        out[name] = signature
    return out


def main() -> int:
    core, facade = signatures(CORE), signatures(FACADE)
    if not core:
        print(f"no public Runner methods found in {CORE}; did the file move?", file=sys.stderr)
        return 1

    problems: list[str] = []
    for name in sorted(core.keys() - facade.keys()):
        problems.append(f"core has `Runner::{name}`, the facade runner does not")
    for name in sorted(facade.keys() - core.keys()):
        problems.append(f"the facade runner has `Runner::{name}`, core does not")
    for name in sorted(core.keys() & facade.keys()):
        if core[name] != facade[name]:
            problems.append(
                f"`Runner::{name}` differs:\n    core:   {core[name]}\n    facade: {facade[name]}"
            )

    if problems:
        print("facade Runner is out of step with the core Runner:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"runner api ok ({len(core)} methods)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
