"""A real MCP server (the official Python `mcp` package) used by the Rust interoperability tests.

    python server.py stdio
    python server.py http <port> [json]   # streamable HTTP at http://127.0.0.1:<port>/mcp
                                          # (`json`: plain JSON replies instead of SSE streams)

Tools cover the interesting shapes: typed arguments, text results, structured results, a result the
server flags as an error, a raised exception, an image, and a slow call.
"""

from __future__ import annotations

import os
import sys
import time

from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent, TextContent


def build() -> MCPServer:
    server = MCPServer("rust-test-server")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def echo(text: str) -> str:
        """Echo the text back."""
        return f"echo: {text}"

    @server.tool()
    def greet(name: str, excited: bool = False) -> str:
        """Greet someone, optionally excitedly."""
        return f"Hello, {name}{'!' if excited else '.'}"

    @server.tool()
    def two_parts() -> list[TextContent]:
        """Return two text blocks."""
        return [TextContent(type="text", text="first"), TextContent(type="text", text="second")]

    @server.tool()
    def picture() -> list[ImageContent | TextContent]:
        """Return an image and a caption."""
        return [
            ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"),
            TextContent(type="text", text="a caption"),
        ]

    @server.tool()
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("the tool exploded")

    @server.tool()
    def env_var(name: str) -> str:
        """Report an environment variable of the server process."""
        return os.environ.get(name, "<unset>")

    @server.tool()
    def slow(seconds: float) -> str:
        """Sleep, then answer."""
        time.sleep(seconds)
        return "finally"

    return server


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "stdio":
        build().run(transport="stdio")
    else:
        build().run(
            transport="streamable-http",
            host="127.0.0.1",
            port=int(sys.argv[2]),
            json_response="json" in sys.argv[3:],
        )
