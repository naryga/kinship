"""`search_connections`: find connections by name, title or company."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult
from pydantic import Field

from kinship.server.context import repo_session
from kinship.server.tools._result import READ_ONLY, tool_result


def register(mcp: MCPServer) -> None:
    @mcp.tool(title="Search connections", annotations=READ_ONLY)
    def search_connections(
        query: Annotated[
            str,
            Field(
                description=(
                    "Words to look for in a connection's name, job title or company, in any "
                    'order and any case, e.g. "ada", "staff engineer" or "designer acme". '
                    "Every word must match somewhere. Leave empty to use only the filters."
                )
            ),
        ] = "",
        title: Annotated[
            str | None,
            Field(description='Only people whose job title contains this text, e.g. "designer".'),
        ] = None,
        company: Annotated[
            str | None,
            Field(
                description=(
                    "Only people currently at this company. Spelling variants are matched "
                    'the same way as in find_connections_at_company ("gift health" finds '
                    '"Gifthealth").'
                )
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(description="Maximum number of people to return, 1 to 100. Default 20."),
        ] = 20,
    ) -> CallToolResult:
        """Search your current LinkedIn connections by name, job title and company.

        Use it for questions like "which product managers do I know?" (title) or
        "find Ada" (query). Returns `total` (all matches) and `connections`: up to
        `limit` people with name, title, company, profile_url and connected_on.
        People whose name matches come first, then alphabetically by last name.
        When `company` is given, `matched_companies` says what it was matched to.
        People who are no longer connections are not included. For one person's
        full details call `get_connection` with their profile_url."""
        with repo_session() as repo:
            found = repo.search_connections(query, title=title, company=company, limit=limit)

        shown = len(found.connections)
        if found.total == 0:
            summary = "No connections match."
        elif shown < found.total:
            summary = f"{found.total} connections match; showing the first {shown}."
        else:
            summary = f"{found.total} connection{' matches' if found.total == 1 else 's match'}."
        data: dict[str, Any] = {"total": found.total, "connections": found.connections}
        if found.company is not None:
            if found.company.matched:
                data["matched_companies"] = found.company.matched_companies()
            else:
                asked = " ".join(found.company.query.split())
                summary += f' No company matches "{asked}"; filtered on the name as written.'
                if found.company.suggestions:
                    data["company_suggestions"] = list(found.company.suggestions)
        return tool_result(summary, data)
