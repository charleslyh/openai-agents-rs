"""Run the shared Chat Completions conversion cases through the Python SDK.

Reads `tests/parity/chat_convert_cases.json` and prints, as JSON, what the Python converter
produces for each case so the Rust test (`tests/chat_convert_parity.rs`) can compare. A case that
makes Python raise is reported as `{"error": "<ExceptionName>"}`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from openai.types.chat import ChatCompletionMessage

from agents.models.chatcmpl_converter import Converter

logging.disable(logging.CRITICAL)


def convert_items(case: dict) -> object:
    try:
        return Converter.items_to_messages(
            case["items"],
            model=case.get("model"),
            preserve_thinking_blocks=case.get("preserve_thinking_blocks", False),
        )
    except Exception as exc:  # noqa: BLE001 - the exception class is the observable contract
        return {"error": type(exc).__name__}


def convert_message(case: dict) -> object:
    try:
        items = Converter.message_to_output_items(
            ChatCompletionMessage(**case["message"]), provider_data=case.get("provider_data")
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__}
    return [item.model_dump(exclude_none=True) for item in items]


def main() -> None:
    path = Path(sys.argv[1])
    cases = json.loads(path.read_text(encoding="utf-8"))
    out = {
        "items_to_messages": {c["name"]: convert_items(c) for c in cases["items_to_messages"]},
        "message_to_output_items": {
            c["name"]: convert_message(c) for c in cases["message_to_output_items"]
        },
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
