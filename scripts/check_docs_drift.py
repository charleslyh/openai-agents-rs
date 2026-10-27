#!/usr/bin/env python3
"""Fail when documentation states facts the code no longer has.

Checks (each one was wrong at some point):
  1. The run-state schema id in STABILITY.md and docs/COMPAT.md is the one in the code.
  2. Every crate under crates/ has a row in STABILITY.md and is named in docs/DEVIATIONS.md D-007.
  3. The crate count D-007 spells out ("twelve crates") equals the number of published crates.
  4. The workspace version, the version of every internal `opai-agents*` workspace dependency, and
     the `opai-agents = "X.Y"` install snippets in the docs agree.

Standard library only. Run from anywhere: python3 scripts/check_docs_drift.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUMBERS = {
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
}


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def package_name(manifest: Path) -> str:
    m = re.search(r'^name\s*=\s*"([^"]+)"', manifest.read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit(f"{manifest}: no package name")
    return m.group(1)


def check_versions() -> list[str]:
    root = read("Cargo.toml")
    m = re.search(r'\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"', root, re.M | re.S)
    if not m:
        return ["Cargo.toml: no [workspace.package] version"]
    version = m.group(1)
    series = ".".join(version.split(".")[:2])
    errors = []
    for name, dep in re.findall(r'^(opai-agents[\w-]*)\s*=\s*(\{[^}]*\})', root, re.M):
        v = re.search(r'version\s*=\s*"([^"]+)"', dep)
        if not v or v.group(1) != version:
            got = v.group(1) if v else "none"
            errors.append(f"Cargo.toml: workspace dependency {name} has version {got}, expected {version}")
    for doc in ("README.md", "crates/teams/README.md", "scripts/gen_llms_txt.py"):
        for found in re.findall(r'opai-agents\s*=\s*(?:\{\s*version\s*=\s*)?"(\d+\.\d+)"', read(doc)):
            if found != series:
                errors.append(f"{doc}: install snippet says {found}, the workspace is {series}")
    return errors


def main() -> int:
    errors: list[str] = []

    code = read("crates/core/src/run_state/mod.rs")
    m = re.search(r'RUN_STATE_SCHEMA_VERSION:\s*&str\s*=\s*"([^"]+)"', code)
    if not m:
        errors.append("cannot find RUN_STATE_SCHEMA_VERSION in crates/core/src/run_state/mod.rs")
    else:
        schema = m.group(1)
        for doc in ("STABILITY.md", "docs/COMPAT.md"):
            if f"`{schema}`" not in read(doc):
                errors.append(f"{doc}: does not mention the current schema `{schema}`")

    names = sorted(
        package_name(p)
        for p in (ROOT / "crates").glob("*/Cargo.toml")
        if not re.search(r"^publish\s*=\s*false", p.read_text(encoding="utf-8"), re.M)
    )
    stability = read("STABILITY.md")
    deviations = read("docs/DEVIATIONS.md")
    d007 = next((l for l in deviations.splitlines() if l.startswith("| D-007")), "")
    for name in names:
        if f"`{name}`" not in stability:
            errors.append(f"STABILITY.md: no row for {name}")
        if f"`{name}`" not in d007:
            errors.append(f"docs/DEVIATIONS.md D-007: {name} is not listed")

    m = re.search(r"workspace of (\w+) crates", d007)
    if not m or NUMBERS.get(m.group(1)) != len(names):
        got = m.group(1) if m else "nothing"
        errors.append(f"docs/DEVIATIONS.md D-007 says {got} crates, there are {len(names)}")

    errors.extend(check_versions())

    for e in errors:
        print(f"docs drift: {e}", file=sys.stderr)
    if not errors:
        print("docs drift ok")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
