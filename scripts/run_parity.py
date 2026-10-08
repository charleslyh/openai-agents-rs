#!/usr/bin/env python3
"""Python oracle for parity scenarios (verification layer 3).

Uses vendor/openai-agents-python @ v0.23.1 (or an installed openai-agents==0.23.1).
Writes golden JSON next to each scenario.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "openai-agents-python" / "src"
SCENARIO_DIR = ROOT / "tests" / "parity" / "scenarios"


def _ensure_import() -> None:
    if VENDOR.is_dir():
        sys.path.insert(0, str(VENDOR))


def _build_steps(raw_steps: list[dict]):
    from agents.testing import ModelStep

    steps = []
    for step in raw_steps:
        if "error" in step and step["error"]:
            steps.append(ModelStep.raise_error(RuntimeError(step["error"])))
        else:
            steps.append(ModelStep(output=step.get("output", [])))
    return steps


def _build_tools(tool_specs: list[dict]):
    from agents import function_tool

    tools = []
    for spec in tool_specs:
        name = spec["name"]
        return_value = spec.get("return_value", "ok")
        error = spec.get("error")

        def _make(rv: str, err: str | None):
            def _fn() -> str:
                if err is not None:
                    raise RuntimeError(err)
                return rv

            return _fn

        tools.append(
            function_tool(
                _make(return_value, error),
                name_override=name,
                description_override=spec.get("description", name),
            )
        )
    return tools


def _canonical_json(text):
    """Canonical form for JSON text so key order does not matter; other text is unchanged."""
    try:
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return text


async def run_scenario(path: Path) -> dict:
    from agents import Agent, Runner
    from agents.testing import ScriptedModel

    scenario = json.loads(path.read_text(encoding="utf-8"))
    model = ScriptedModel(steps=_build_steps(scenario["steps"]))
    agent_spec = scenario["agent"]
    agent = Agent(
        name=agent_spec["name"],
        instructions=agent_spec.get("instructions"),
        model=model,
        tools=_build_tools(agent_spec.get("tools", [])),
        tool_use_behavior=agent_spec.get("tool_use_behavior", "run_llm_again"),
    )
    result = await Runner.run(agent, scenario["input"])
    final = result.final_output
    if not isinstance(final, str):
        # Structured outputs are compared as canonical JSON, like the Rust test does.
        try:
            final = json.dumps(final, sort_keys=True, separators=(",", ":"))
        except TypeError:
            final = str(final)
    golden = {
        "name": scenario["name"],
        "final_output": final,
        "raw_response_count": len(result.raw_responses),
        "last_agent": result.last_agent.name,
        "new_item_count": len(result.new_items),
        "tool_output_count": sum(
            1 for item in result.new_items if item.type == "tool_call_output_item"
        ),
    }
    expect = scenario.get("expect", {})
    for key, value in expect.items():
        actual = golden.get(key)
        if key == "final_output":
            value, actual = _canonical_json(value), _canonical_json(actual)
        if actual != value:
            raise AssertionError(
                f"{scenario['name']}: expect {key}={value!r}, python got {actual!r}"
            )
    model.assert_complete()
    return golden


async def main() -> int:
    _ensure_import()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="Write tests/parity/scenarios/<name>.golden.json",
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario paths")
    args = parser.parse_args()

    paths = (
        [Path(p) for p in args.scenarios]
        if args.scenarios
        else sorted(SCENARIO_DIR.glob("*.json"))
    )
    paths = [p for p in paths if not p.name.endswith(".golden.json")]

    all_ok = True
    for path in paths:
        try:
            golden = await run_scenario(path)
            print(f"OK  {path.name}: final_output={golden['final_output']!r}")
            if args.write_golden:
                out = path.with_suffix(".golden.json")
                out.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
                print(f"    wrote {out.relative_to(ROOT)}")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            print(f"FAIL {path.name}: {exc}", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
