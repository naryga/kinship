"""`get_connection`: everything known about one connection."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult
from pydantic import Field

from kinship.server.context import repo_session
from kinship.server.tools._result import READ_ONLY, tool_result


def _describe(person: dict[str, Any]) -> str:
    """One line about a found connection."""
    text = person.get("name") or person.get("profile_url") or "Unnamed connection"
    title, company = person.get("title"), person.get("company")
    if title and company:
        text += f": {title} at {company}"
    elif title or company:
        text += f": {title}" if title else f": at {company}"
    facts = []
    if person.get("removed"):
        facts.append("no longer a connection")
    elif person.get("connected_on"):
        facts.append(f"connected {person['connected_on']}")
    count = person.get("message_count", 0)
    if count:
        messages = f"{count} message{'' if count == 1 else 's'}"
        if person.get("last_contact"):
            messages += f", last contact {person['last_contact']}"
        facts.append(messages)
    else:
        facts.append("no messages")
    return f"{text} ({'; '.join(facts)})."


def register(mcp: MCPServer) -> None:
    @mcp.tool(title="Get one connection", annotations=READ_ONLY)
    def get_connection(
        name_or_url: Annotated[
            str,
            Field(
                description=(
                    "The person's LinkedIn profile URL (the profile_url returned by the other "
                    'tools; always unambiguous) or their name, e.g. "Ada Lovelace". Part of '
                    "a name works when only one connection matches it."
                )
            ),
        ],
    ) -> CallToolResult:
        """Look up one LinkedIn connection in detail.

        When exactly one person matches, `connection` holds: name, title, company,
        profile_url, connected_on, email; `role_history` (company, title and `as_of`,
        the date of the export that first showed that role; oldest first);
        `message_count` and `last_contact` (date of the latest message with them);
        `endorsements` and `recommendations`, each split into `received` (from them
        to you) and `given` (from you to them); your private `notes` about them; and
        the `invitation` that connected you. Fields with no data are left out;
        `"removed": true` means they are no longer a connection.
        When several people match, `candidates` lists them: call again with the
        profile_url of the right one. When nobody matches, `suggestions` lists
        people with similar names."""
        with repo_session() as repo:
            found = repo.lookup_connection(name_or_url)

        asked = " ".join(name_or_url.split())
        if found.status == "found" and found.connection is not None:
            return tool_result(
                _describe(found.connection), {"match": "found", "connection": found.connection}
            )
        if found.status == "ambiguous":
            count = len(found.candidates)
            many = f"More than {count}" if found.more_candidates else str(count)
            data: dict[str, Any] = {"match": "ambiguous", "candidates": list(found.candidates)}
            if found.more_candidates:
                data["more_candidates"] = True
            return tool_result(
                f'{many} connections match "{asked}". Call get_connection again with the '
                "profile_url of the one you mean.",
                data,
            )
        summary = f'No connection matches "{asked}".'
        data = {"match": "none"}
        if found.suggestions:
            data["suggestions"] = list(found.suggestions)
            names = ", ".join(person.get("name", "?") for person in found.suggestions)
            summary += f" Similar names: {names}."
        return tool_result(summary, data)
