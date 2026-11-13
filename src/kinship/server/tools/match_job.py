"""`match_job`: the owner's relevant history and nearby connections for a job description."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, TextContent
from pydantic import Field

from kinship.rag.match import (
    DEFAULT_GROUP_LIMIT,
    DEFAULT_K,
    MAX_DESCRIPTION_CHARS,
    MISSING_TERMS_NOTE,
    HistoryItem,
    JobMatch,
    PersonMatch,
)
from kinship.rag.match import match_job as run_match
from kinship.server.context import current_embedder, repo_session
from kinship.server.tools._result import READ_ONLY, tool_result

MAX_K = 20
MAX_TEXT_CHARS = 800
MAX_NAME_CHARS = 120


def _error(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\N{HORIZONTAL ELLIPSIS}"


def _plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def _history_item(item: HistoryItem) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source": item.source,
        "title": item.title,
        "text": _shorten(item.text, MAX_TEXT_CHARS),
        "score": round(item.score, 4),
        "why": item.why,
    }
    if item.matched_terms:
        data["matched_terms"] = list(item.matched_terms)
    if len(item.text) > MAX_TEXT_CHARS:
        data["truncated"] = True
    return data


def _people(matches: list[PersonMatch]) -> list[dict[str, Any]]:
    return [{**match.person.as_dict(), "why": match.why} for match in matches]


def _role(result: JobMatch) -> dict[str, Any]:
    role = result.role
    data: dict[str, Any] = {}
    if role.title:
        data["title"] = _shorten(role.title, MAX_NAME_CHARS)
    if role.company:
        data["company"] = _shorten(role.company, MAX_NAME_CHARS)
    inferred = [
        name
        for name, value, was_inferred in (
            ("title", role.title, role.title_inferred),
            ("company", role.company, role.company_inferred),
        )
        if value and was_inferred
    ]
    if inferred:
        data["inferred"] = True
        data["inferred_fields"] = inferred
    if role.company:
        data["company_in_network"] = role.company_known
    if role.other_companies:
        data["other_companies_mentioned"] = list(role.other_companies)
    if role.company_suggestions:
        data["company_suggestions"] = list(role.company_suggestions)
    return data


def _summary(result: JobMatch) -> str:
    role = result.role
    name = _shorten(role.title or "This role", MAX_NAME_CHARS)
    if role.company:
        name += f" at {_shorten(role.company, MAX_NAME_CHARS)}"
    parts = [_plural(len(result.history), "relevant part", "relevant parts") + " of your history"]
    totals = result.totals
    if role.company_known:
        parts.append(_plural(totals["at_company"], "connection", "connections") + " at the company")
        if totals["former_at_company"]:
            parts.append(f"{totals['former_at_company']} who used to work there")
    elif role.company:
        parts.append("no connections at the company")
    parts.append(f"{totals['similar_titles']} with similar titles")
    summary = f"{name}: {', '.join(parts)}"
    if result.mode_used == "fts":
        summary += " (full-text only)"
    return summary + "."


def register(mcp: MCPServer) -> None:
    @mcp.tool(title="Match a job description", annotations=READ_ONLY)
    def match_job(
        description_text: Annotated[
            str,
            Field(
                description=(
                    "The job description, pasted as plain text: ideally the whole posting, "
                    "starting with the job title. Only the first "
                    f"{MAX_DESCRIPTION_CHARS} characters are read."
                )
            ),
        ],
        company: Annotated[
            str | None,
            Field(
                description=(
                    "The hiring company, if you know it (e.g. 'Gift Health'). Omit to let "
                    "the tool look for one of the user's known companies in the text; pass "
                    "it when `role.company` in a previous result was wrong or missing."
                )
            ),
        ] = None,
        title: Annotated[
            str | None,
            Field(
                description=(
                    "The job title, if you know it (e.g. 'Senior Data Engineer'). Omit to "
                    "let the tool read it from the top of the description; pass it when "
                    "`role.title` in a previous result was wrong or missing."
                )
            ),
        ] = None,
        k: Annotated[
            int,
            Field(
                description=(
                    f"How many parts of the user's history to return, 1 to {MAX_K}. "
                    f"Default {DEFAULT_K}."
                )
            ),
        ] = DEFAULT_K,
    ) -> CallToolResult:
        """Match a pasted job description against the user's own LinkedIn data: which
        parts of their history fit the role, and who in their network is near it.

        Use it when the user is considering or applying for a specific job: to tailor a
        résumé or cover letter with evidence from their past positions and from what
        others wrote about them, and to find who to ask about the role or for a referral.

        The result has:
        - `role`: the title and company the tool worked with. `inferred` means they were
          read from the text; if they are wrong, call again with `title` / `company`.
        - `history`: the best-fitting positions, recommendations and profile text (direct
          messages are never searched), found with hybrid full-text + vector retrieval.
          Each has `why`: the description's terms that occur in that text and which
          retrievers found it. `mode` is `hybrid`, or `fts` with a `note` when vector
          search was unavailable.
        - `matched_skills`: the user's listed skills that the description names.
          `missing_terms` is a rough list of prominent words of the description found
          nowhere in the user's history or skills; treat it as a hint, not a verdict.
        - `connections`: `at_company` (people who work there now), `similar_titles`
          (people anywhere whose job title shares a meaningful word with the role) and
          `former_at_company` (people who used to work there). Each person has `why`;
          each group is capped at 5 and `connection_totals` gives the full counts.

        Follow up with `warm_paths(company)` to rank who could best introduce the user,
        and `get_connection` for detail on one person."""
        if not (description_text or "").strip():
            return _error(
                "description_text must not be empty: paste the text of the job description."
            )
        k = min(max(k, 1), MAX_K)

        with repo_session() as repo:
            result = run_match(
                repo,
                current_embedder(),
                description_text,
                company=company,
                title=title,
                k=k,
                group_limit=DEFAULT_GROUP_LIMIT,
            )

        data: dict[str, Any] = {"role": _role(result)}
        if result.mode_used:
            data["mode"] = result.mode_used
        if result.note:
            data["note"] = result.note
        if result.truncated:
            data["description_truncated"] = True
        data["history"] = [_history_item(item) for item in result.history]
        if result.matched_skills:
            data["matched_skills"] = result.matched_skills
        if result.missing_terms:
            data["missing_terms"] = result.missing_terms
            data["missing_terms_note"] = MISSING_TERMS_NOTE
        data["connections"] = {
            "at_company": _people(result.at_company),
            "similar_titles": _people(result.similar_titles),
            "former_at_company": _people(result.former_at_company),
        }
        data["connection_totals"] = result.totals
        return tool_result(_summary(result), data)
