"""MCP prompts. One module per prompt, each exposing `register(mcp)`.

Same conventions as `kinship.server.tools`: read data only through
`repo_session()`. New modules are picked up automatically; names starting with
`_` are skipped.
"""

import sys

from mcp.server.mcpserver import MCPServer

from kinship.server._discovery import register_package


def register_all(mcp: MCPServer) -> list[str]:
    """Register every prompt module in this package."""
    return register_package(mcp, sys.modules[__name__])
