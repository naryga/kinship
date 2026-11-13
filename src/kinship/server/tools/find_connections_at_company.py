"""`find_connections_at_company`: who the owner knows at a company, now or in the past."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult
from pydantic import Field

from kinship.server.context import repo_session
from kinship.server.tools._result import READ_ONLY, tool_result

MAX_RESULTS = 200
"""Most people returned in one call; the counts always cover everyone."""


def register(mcp: MCPServer) -> None:
    @mcp.tool(title="Find connections at a company", annotations=READ_ONLY)
    def find_connections_at_company(
        company: Annotated[
            str,
            Field(
                description=(
                    "Company name as the user said it. Case, spacing, punctuation and legal "
                    'suffixes are ignored: "gift health" finds "Gifthealth" and '
                    '"Gift Health, Inc.". A partial name ("quill") works when nothing '
                    "matches exactly."
                )
            ),
        ],
        include_former: Annotated[
            bool,
            Field(
                description=(
                    "Also return people who used to work there and have since moved to "
                    "another company, according to the history of your LinkedIn exports. "
                    "Only job changes between two imported exports are known."
                )
            ),
        ] = False,
    ) -> CallToolResult:
        """List your LinkedIn connections who work at a company ("who do I know at X?").

        Returns `matched_companies` (what the name was matched to, with other
        spellings), `current_count`, and `connections`: name, title, company,
        profile_url, connected_on and `status`. With `include_former`, people with
        status "former" carry their current company and title plus `former_company`,
        `former_title` and `left_by` (date of the first export that showed them
        elsewhere); `"removed": true` marks someone who is no longer a connection.
        If nothing matches, `suggestions` lists similar company names to try.
        For one person's full details call `get_connection` with their profile_url."""
        with repo_session() as repo:
            resolution = repo.resolve_company(company)
            people = repo.connections_at_companies(resolution.company_ids, include_former)

        asked = " ".join(company.split())
        suggestions = list(resolution.suggestions)
        data: dict[str, Any] = {"matched_companies": resolution.matched_companies()}
        if not resolution.matched:
            summary = f'No company matching "{asked}" among your connections.'
            if suggestions:
                summary += f" Similar companies: {', '.join(suggestions)}."
                data["suggestions"] = suggestions
            return tool_result(summary, data | {"current_count": 0, "connections": []})

        current = sum(1 for person in people if person["status"] == "current")
        former = len(people) - current
        matched = ", ".join(match.display_name for match in resolution.matched)
        summary = f"{current} current connection{'' if current == 1 else 's'} at {matched}"
        data["current_count"] = current
        if include_former:
            summary += f", {former} former"
            data["former_count"] = former
        summary += "."
        if resolution.matched[0].match != "exact":
            summary += f' No company is named exactly "{asked}"; showing the closest.'
        data["connections"] = people[:MAX_RESULTS]
        if len(people) > MAX_RESULTS:
            data["truncated"] = True
            summary += f" Showing the first {MAX_RESULTS}."
        if suggestions:
            data["suggestions"] = suggestions
        return tool_result(summary, data)
