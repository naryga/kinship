"""`semantic_search`: hybrid (full-text + vector) retrieval over the owner's own text."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, TextContent
from pydantic import Field

from kinship.rag.search import SearchHit, search, validate_sources
from kinship.server.context import current_embedder, repo_session
from kinship.server.tools._result import READ_ONLY, tool_result

DEFAULT_K = 5
MAX_K = 20
MAX_TEXT_CHARS = 1200
MAX_SUMMARY_QUERY_CHARS = 80

_RETRIEVERS = {"fts": "full_text", "vector": "vector"}


def _error(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\N{HORIZONTAL ELLIPSIS}"


def _hit(hit: SearchHit) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source": hit.source_table,
        "source_id": hit.source_id,
        "title": hit.title,
        "text": _shorten(hit.text, MAX_TEXT_CHARS),
        "score": round(hit.score, 4),
        "matched": [_RETRIEVERS[name] for name in hit.matched],
    }
    if hit.chunk_index:
        data["part"] = hit.chunk_index + 1
    if len(hit.text) > MAX_TEXT_CHARS:
        data["truncated"] = True
    return data


def register(mcp: MCPServer) -> None:
    @mcp.tool(title="Semantic search", annotations=READ_ONLY)
    def semantic_search(
        query: Annotated[
            str,
            Field(
                description=(
                    "What to look for, in natural language or keywords, e.g. "
                    "'praise for my data pipeline work' or 'Kubernetes migration'. "
                    'Put a phrase in "double quotes" to favour exact matches.'
                )
            ),
        ],
        sources: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Limit the search to some kinds of text: 'recommendations' "
                    "(recommendations received and given), 'my_positions' (the user's own "
                    "job history), 'profiles' (the user's headline and summary), 'messages' "
                    "(direct-message conversations, only if the user chose to index them). "
                    "Omit to search everything."
                )
            ),
        ] = None,
        k: Annotated[
            int,
            Field(description=f"How many results to return, 1 to {MAX_K}. Default {DEFAULT_K}."),
        ] = DEFAULT_K,
    ) -> CallToolResult:
        """Search the free text of the user's own LinkedIn data by meaning and keywords:
        recommendations they received or gave, the descriptions of their own positions,
        and their profile headline and summary. Message conversations are searchable only
        if the user chose to index them. Results are the best-matching passages, each with
        its source, a title saying where it comes from, the text and a relevance score.

        Use it for questions about what the user has done or what others said about them,
        for example to find evidence for a job application or the wording of a
        recommendation. It does not search the user's connections: to find people, companies
        or job changes, use the structured tools (connection and company lookups, warm
        paths, network changes) instead. Results combine full-text and vector ranking; if
        vector search is unavailable the result says so and comes from full-text only."""
        query = (query or "").strip()
        if not query:
            return _error("query must not be empty: say what to search for.")
        try:
            chosen = validate_sources(sources)
        except ValueError as error:
            return _error(str(error))
        k = min(max(k, 1), MAX_K)

        with repo_session() as repo:
            result = search(repo, current_embedder(), query, sources=chosen, k=k)

        count = len(result.hits)
        mode = "full-text only" if result.mode_used == "fts" else result.mode_used
        summary = (
            f"{count} result{'' if count == 1 else 's'} ({mode}) "
            f"for '{_shorten(query, MAX_SUMMARY_QUERY_CHARS)}'"
        )
        if chosen:
            summary += f" in {', '.join(chosen)}"
        summary += "."
        data: dict[str, Any] = {
            "query": _shorten(query, 500),
            "mode": result.mode_used,
            "count": count,
            "results": [_hit(hit) for hit in result.hits],
        }
        if chosen:
            data["sources"] = list(chosen)
        if result.note:
            data["note"] = result.note
        return tool_result(summary, data)
