"""`network_summary`: how much data the owner has loaded, and when."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult

from kinship.server.context import repo_session
from kinship.server.tools._result import READ_ONLY, tool_result


def register(mcp: MCPServer) -> None:
    @mcp.tool(title="Network summary", annotations=READ_ONLY)
    def network_summary() -> CallToolResult:
        """Summarize the LinkedIn data loaded for you: row counts per kind of record
        (connections, companies, messages, ...) and the date of the latest import.
        Use it to check that data is present before asking more specific questions."""
        with repo_session() as repo:
            counts = repo.counts()
            latest = repo.latest_import()

        if latest is None:
            summary = "No LinkedIn export has been imported yet."
        else:
            summary = (
                f"{counts['connections']} connections, {counts['companies']} companies and "
                f"{counts['messages']} messages; latest import on {latest.imported_at:%Y-%m-%d}."
            )
        return tool_result(
            summary,
            {"counts": counts, "latest_import": latest.as_dict() if latest else None},
        )
