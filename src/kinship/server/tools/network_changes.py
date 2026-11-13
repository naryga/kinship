"""`network_changes`: job changes, new and removed connections between exports."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, TextContent
from pydantic import Field

from kinship.server.context import repo_session
from kinship.server.tools._result import READ_ONLY, tool_result

MAX_LIMIT = 200


def parse_since(value: str) -> datetime:
    """An ISO date or datetime as an aware datetime (a date is its first instant, UTC).

    Raises `ValueError` for anything else.
    """
    text = value.strip()
    try:
        parsed: date = date.fromisoformat(text)
    except ValueError:
        parsed = datetime.fromisoformat(text)
    if not isinstance(parsed, datetime):
        parsed = datetime(parsed.year, parsed.month, parsed.day)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _day(moment: datetime | None) -> str | None:
    return moment.date().isoformat() if moment else None


def _compact(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value not in (None, "", [], {})}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def register(mcp: MCPServer) -> None:
    @mcp.tool(title="Network changes", annotations=READ_ONLY)
    def network_changes(
        since: Annotated[
            str | None,
            Field(
                description=(
                    "Report changes after this ISO date or datetime, e.g. 2026-05-01. "
                    "Leave it out for what changed in your latest export compared with "
                    "the one before it."
                )
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=MAX_LIMIT,
                description="Maximum entries in each list; `counts` always has the full totals.",
            ),
        ] = 50,
    ) -> CallToolResult:
        """List what changed in your network: who changed jobs, who you newly connected
        with and who is no longer a connection.

        Changes are found by comparing your LinkedIn exports, so each is dated by the
        export that first showed it (`as_of`), not the day it happened, and only
        exports dated after `since` count (`window` says which). `job_changes` gives
        each move as `from` and `to` with `change` set to "company", "title" or
        "company_and_title"; a company that was merely respelled is not a move and is
        listed under `company_spelling_changes`. `new_connections` and
        `removed_connections` are people first seen in, or missing from, an export in
        the window. `applied_companies` lists jobs you applied to in the window at
        companies where you have connections (all your applications when there were
        none in the window; `applied_companies_scope` says which). With a single
        export there is nothing to compare, so the lists are empty."""
        requested: datetime | None = None
        if since is not None and since.strip():
            try:
                requested = parse_since(since)
            except ValueError:
                message = (
                    f'Invalid `since` value "{since}": use an ISO date such as 2026-05-01 '
                    "or a datetime such as 2026-05-01T12:00:00Z."
                )
                return CallToolResult(
                    content=[TextContent(type="text", text=message)], is_error=True
                )

        with repo_session() as repo:
            dated = [info for info in repo.imports_in_order() if info.exported_at is not None]
            if not dated:
                return tool_result(
                    "No LinkedIn export has been imported yet.",
                    {"window": {"imports_considered": 0}, "counts": {}},
                )
            latest = dated[-1].exported_at
            # Default: since the export before the latest one (or the only one).
            start = requested or dated[-2 if len(dated) > 1 else -1].exported_at
            assert latest is not None and start is not None

            job_changes = repo.job_changes(start)
            new = repo.new_connections(start)
            removed = repo.removed_connections(start)
            applied, applied_scope = repo.applied_companies(start), "since"
            if not applied:
                applied, applied_scope = repo.applied_companies(), "all"

        moves = [change for change in job_changes if not change.is_spelling_only]
        respelled = [change for change in job_changes if change.is_spelling_only]

        def change_kind(company_changed: bool, title_changed: bool) -> str:
            if company_changed and title_changed:
                return "company_and_title"
            return "company" if company_changed else "title"

        data: dict[str, Any] = {
            "window": {
                "since": start.isoformat(),
                "latest_export": latest.isoformat(),
                "imports_considered": sum(1 for info in dated if info.exported_at > start),
                "imports_total": len(dated),
            },
            "counts": {
                "job_changes": len(moves),
                "company_moves": sum(1 for change in moves if change.company_changed),
                "title_only_changes": sum(1 for change in moves if not change.company_changed),
                "company_spelling_changes": len(respelled),
                "new_connections": len(new),
                "removed_connections": len(removed),
                "applied_companies": len(applied),
            },
            "job_changes": [
                _compact(
                    {
                        "name": change.person.name,
                        "profile_url": change.person.profile_url,
                        "change": change_kind(change.company_changed, change.title_changed),
                        "from": _compact(
                            {"company": change.from_company, "title": change.from_title}
                        ),
                        "to": _compact({"company": change.to_company, "title": change.to_title}),
                        "as_of": _day(change.as_of),
                    }
                )
                for change in moves[:limit]
            ],
            "new_connections": [
                {**event.person.as_dict(), "first_seen": _day(event.at)} for event in new[:limit]
            ],
            "removed_connections": [
                {**event.person.as_dict(), "removed_as_of": _day(event.at)}
                for event in removed[:limit]
            ],
            "company_spelling_changes": [
                _compact(
                    {
                        "name": change.person.name,
                        "profile_url": change.person.profile_url,
                        "from": change.from_company,
                        "to": change.to_company,
                        "as_of": _day(change.as_of),
                    }
                )
                for change in respelled[:limit]
            ],
            "applied_companies": [
                _compact(
                    {
                        "company": item.company,
                        "job_title": item.job_title,
                        "applied_at": _day(item.applied_at),
                        "connection_count": item.connection_count,
                        "connections": list(item.connection_names),
                    }
                )
                for item in applied[:limit]
            ],
        }
        data = _compact(data)  # drop the lists that are empty; `counts` still says 0
        if applied:
            data["applied_companies_scope"] = applied_scope

        if len(dated) == 1:
            summary = "Only one export has been imported, so there is nothing to compare yet."
        else:
            summary = (
                f"Since {start.date().isoformat()}: {_plural(len(moves), 'job change')}, "
                f"{_plural(len(new), 'new connection')} and "
                f"{_plural(len(removed), 'removed connection')}."
            )
        return tool_result(summary, data)
