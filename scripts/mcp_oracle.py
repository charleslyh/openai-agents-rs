"""Describe what the Python SDK makes of the shared MCP test server.

    python scripts/mcp_oracle.py <python> <server.py>

Prints a JSON list of the function tools `MCPUtil` builds from the server (name, description,
parameters, strict flag), so `tests/mcp_interop.rs` can compare the Rust conversion with it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from agents import Agent, RunContextWrapper
from agents.mcp import MCPServerStdio, MCPUtil

logging.disable(logging.CRITICAL)


async def main(python: str, server_py: str) -> None:
    async with MCPServerStdio(
        params={"command": python, "args": [server_py, "stdio"]}, cache_tools_list=False
    ) as server:
        tools = await MCPUtil.get_all_function_tools(
            [server],
            convert_schemas_to_strict=False,
            run_context=RunContextWrapper(context=None),
            agent=Agent(name="oracle"),
        )
        print(
            json.dumps(
                [
                    {
                        "name": t.name,
                        "description": t.description,
                        "params_json_schema": t.params_json_schema,
                        "strict": t.strict_json_schema,
                    }
                    for t in tools
                ]
            )
        )


asyncio.run(main(sys.argv[1], sys.argv[2]))
