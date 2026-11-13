"""`warm_paths`: who could introduce the owner to a company, best path first."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult
from pydantic import Field

from kinship.companies import company_key
from kinship.repo import CompanyMatch, Repository
from kinship.server.context import repo_session
from kinship.server.tools._result import READ_ONLY, tool_result
from kinship.warmth import rank_key, score_warmth

MAX_LIMIT = 50
MAX_FORMER_EMPLOYEES = 5
MAX_OTHER_MATCHES = 5
MIN_SUGGESTION_WORD = 3


def _company(match: CompanyMatch) -> dict[str, Any]:
    return {
        "name": match.display_name,
        "match": match.match,
        "current_connections": match.connection_count,
    }


def _suggestions(repo: Repository, query: str) -> list[str]:
    """Company names sharing a word with a query that matched nothing."""
    names: dict[str, None] = {}
    for word in query.split():
        key = company_key(word)
        if key is None or len(key) < MIN_SUGGESTION_WORD:
            continue
        for match in repo.find_companies(word, limit=MAX_OTHER_MATCHES):
            names.setdefault(match.display_name)
    return list(names)[:MAX_OTHER_MATCHES]


def register(mcp: MCPServer) -> None:
    @mcp.tool(title="Warm introduction paths", annotations=READ_ONLY)
    def warm_paths(
        company: Annotated[
            str,
            Field(
                description=(
                    "Company to find an introduction to, as you would write it "
                    '(e.g. "Gift Health"). Case, punctuation and suffixes such as Inc. '
                    "are ignored; a partial name matches the closest company."
                )
            ),
        ],
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=MAX_LIMIT,
                description="Maximum number of current employees to return, best first.",
            ),
        ] = 10,
    ) -> CallToolResult:
        """Rank your connections at a company by how well you know them, to find the
        warmest person to ask for an introduction or referral.

        Each person has a `score` and the `reasons` behind it, strongest first:
        recommendations either of you wrote for the other, companies you both worked
        at, messages you exchanged (how many and how recently; skipped when your
        export has no messages), a note you wrote about them, skill endorsements in
        either direction, and who sent the connection invitation. A score of 0 means
        you are connected but nothing else is recorded. `paths` lists people who work
        at the company now; `former_employees` lists up to 5 connections who used to
        work there and have since moved on, who can often still introduce you.
        `matched_companies` says which company the name resolved to; if it is not the
        one you meant, try a name from `other_matches`."""
        with repo_session() as repo:
            matches = repo.find_companies(company, limit=MAX_OTHER_MATCHES + 1)
            if not matches:
                suggestions = _suggestions(repo, company)
                data: dict[str, Any] = {"company": company, "matched_companies": [], "paths": []}
                if suggestions:
                    data["suggestions"] = suggestions
                return tool_result(f'No company matching "{company}" in your network.', data)

            # An exact match is unique. Otherwise take the single best partial match
            # rather than mixing people from several companies into one ranking.
            target, others = matches[0], matches[1:]
            target_ids = [target.id]
            current = repo.connections_at_companies(target_ids)
            former = repo.former_connections_of_companies(target_ids)
            signals = repo.warmth_signals(
                [person.id for person in (*current, *former)], exclude_company_ids=target_ids
            )
            use_messages = repo.has_messages()
            latest = repo.latest_import()

        as_of = (latest.exported_at or latest.imported_at) if latest else None

        def ranked(people: list[Any]) -> list[dict[str, Any]]:
            scored = []
            for person in people:
                # `as_of` is only missing when there are no imports, hence no people.
                warmth = score_warmth(signals[person.id], as_of, use_messages=use_messages)
                key = rank_key(warmth, person.connected_on, person.name, person.id)
                record = {
                    **person.as_dict(),
                    "score": warmth.score,
                    "reasons": list(warmth.reasons),
                }
                scored.append((key, record))
            return [record for _, record in sorted(scored, key=lambda item: item[0])]

        paths = ranked(current)
        former_employees = ranked(former)[:MAX_FORMER_EMPLOYEES]

        data = {
            "company": company,
            "matched_companies": [_company(target)],
            "total_current_connections": len(paths),
            "paths": paths[:limit],
        }
        if others:
            data["other_matches"] = [match.display_name for match in others[:MAX_OTHER_MATCHES]]
        if former_employees:
            data["former_employees"] = former_employees
            data["total_former_employees"] = len(former)
        if not use_messages:
            data["messages_used"] = False

        name = target.display_name
        if paths:
            best = paths[0]
            summary = (
                f"{len(paths)} current connection{'' if len(paths) == 1 else 's'} at {name}; "
                f"warmest: {best['name']} (score {best['score']:g})."
            )
        elif former_employees:
            summary = (
                f"No current connections at {name}; "
                f"{len(former)} connection{'' if len(former) == 1 else 's'} used to work there."
            )
        else:
            summary = f"No connections at {name}."
        return tool_result(summary, data)
