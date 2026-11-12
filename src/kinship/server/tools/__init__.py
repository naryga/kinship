"""MCP tools. One module per tool (or small group), each exposing `register(mcp)`.

A tool reads data only through `repo_session()` and returns `tool_result(...)`:

    from mcp.server.mcpserver import MCPServer
    from mcp_types import CallToolResult

    from kinship.server.context import repo_session
    from kinship.server.tools._result import tool_result


    def register(mcp: MCPServer) -> None:
        @mcp.tool(annotations=READ_ONLY)
        def my_tool(company: str) -> CallToolResult:
            \"\"\"Shown to the assistant as the tool description.\"\"\"
            with repo_session() as repo:
                rows = repo.something(company)
            return tool_result(f"{len(rows)} matches for {company}.", {"matches": rows})

New modules are picked up automatically; names starting with `_` are skipped.
"""

import sys

from mcp.server.mcpserver import MCPServer

from kinship.server._discovery import register_package


def register_all(mcp: MCPServer) -> list[str]:
    """Register every tool module in this package."""
    return register_package(mcp, sys.modules[__name__])
