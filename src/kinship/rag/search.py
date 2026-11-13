"""Hybrid retrieval: the one entry point for searching an owner's chunks.

    from kinship.rag.search import search

    result = search(repo, embedder, "who praised my data pipeline work?", k=5)
    for hit in result.hits:
        print(hit.title, hit.score)

`search` runs Postgres full-text search and pgvector cosine search through the
owner-bound `Repository` and merges the two rankings with reciprocal rank
fusion. Tools, `match_job` and the evaluation harness all call this function;
it imports nothing from the server.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

from kinship.rag.chunking import SOURCE_TABLES
from kinship.rag.embedder import Embedder, EmbedderError
from kinship.rag.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from kinship.repo import ChunkHit, Repository

logger = logging.getLogger(__name__)

Mode = Literal["hybrid", "fts", "vector"]
MODES: tuple[str, ...] = get_args(Mode)

DEFAULT_K = 5
DEFAULT_CANDIDATES = 50
MIN_SIMILARITY = 1e-6
"""Vector hits need a cosine similarity above this (i.e. above zero) to count."""


@dataclass(frozen=True)
class SearchHit:
    """One result.

    `source_table` and `source_id` identify the row the chunk was built from:
    `recommendations.id`, `my_positions.id`, `profiles.id`, or for `messages`
    the id of the first message of the conversation (`chunk_index` then counts
    the conversation's chunks). `title` is that row described in words.

    `score` depends on `SearchResult.mode_used`: the fused RRF score for
    `hybrid`, `ts_rank` for `fts`, cosine similarity for `vector`. `fts_rank`
    and `vector_rank` are the 1-based ranks in each retriever's candidate list,
    or None if that retriever did not return the chunk.
    """

    chunk_id: int
    source_table: str
    source_id: int
    chunk_index: int
    text: str
    score: float
    title: str
    fts_rank: int | None = None
    vector_rank: int | None = None

    @property
    def matched(self) -> tuple[str, ...]:
        """Which retrievers returned this chunk: `fts`, `vector` or both."""
        names = (("fts", self.fts_rank), ("vector", self.vector_rank))
        return tuple(name for name, rank in names if rank is not None)


@dataclass(frozen=True)
class SearchResult:
    """Hits, best first, and how they were found.

    `mode_used` differs from the requested mode when hybrid search fell back
    to full-text only; `note` then says why.
    """

    query: str
    mode: str
    mode_used: str
    hits: list[SearchHit] = field(default_factory=list)
    note: str | None = None


def validate_sources(sources: Sequence[str] | None) -> tuple[str, ...] | None:
    """`sources` as a tuple of known source tables; None or empty means all of them."""
    if sources is None:
        return None
    if isinstance(sources, str):
        raise ValueError("sources must be a list of source names, not a single string")
    chosen = tuple(dict.fromkeys(sources))
    unknown = [source for source in chosen if source not in SOURCE_TABLES]
    if unknown:
        raise ValueError(
            f"Unknown source(s): {', '.join(map(repr, unknown))}. "
            f"Valid sources: {', '.join(SOURCE_TABLES)}."
        )
    return chosen or None


def search(
    repo: Repository,
    embedder: Embedder | None,
    query: str,
    *,
    sources: Sequence[str] | None = None,
    k: int = DEFAULT_K,
    mode: Mode = "hybrid",
    candidates: int = DEFAULT_CANDIDATES,
) -> SearchResult:
    """Search the chunks of `repo`'s owner and return the best `k`.

    - `mode="hybrid"` (default): full-text and vector search, `candidates`
      results each, merged with reciprocal rank fusion. Without an `embedder`,
      when none of the owner's chunks (in `sources`) carries a vector from the
      embedder's model, or when embedding the query fails, it answers from
      full-text search alone: `mode_used` is then `"fts"` and `note` explains.
    - `mode="fts"`: full-text search only; `embedder` is not used.
    - `mode="vector"`: vector search only; raises `ValueError` without an
      embedder and lets an `EmbedderError` through. With no usable vectors it
      returns no hits and a `note`.

    `sources` limits the search to some of `kinship.rag.chunking.SOURCE_TABLES`.
    Raises `ValueError` for an empty query, an unknown source or mode, or `k < 1`.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Valid modes: {', '.join(MODES)}.")
    if k < 1:
        raise ValueError("k must be at least 1")
    chosen = validate_sources(sources)
    candidates = max(candidates, k)

    fts: list[ChunkHit] = []
    vector: list[ChunkHit] = []
    note: str | None = None
    mode_used: str = mode

    if mode in ("hybrid", "fts"):
        fts = repo.fts_search(query, sources=chosen, limit=candidates)

    if mode == "vector" and embedder is None:
        raise ValueError("mode='vector' needs an embedder")
    if mode in ("hybrid", "vector"):
        problem = _vector_unavailable(repo, embedder, chosen)
        if problem is None:
            assert embedder is not None
            try:
                query_vector = embedder.embed_query(query)
            except EmbedderError as error:
                if mode == "vector":
                    raise
                logger.warning("Embedding the query failed; using full-text search only: %s", error)
                problem = "Embedding the query failed"
            else:
                nearest = repo.vector_search(
                    query_vector, model=embedder.model, sources=chosen, limit=candidates
                )
                # Nearest-neighbour search always returns something. A chunk at a
                # right angle to the query (or pointing away) is not a match.
                vector = [hit for hit in nearest if hit.score > MIN_SIMILARITY]
        if problem is not None and mode == "hybrid":
            mode_used = "fts"
            note = f"{problem}; results are from full-text search only."
        elif problem is not None:
            note = f"{problem}."

    fts_by_id = {hit.chunk_id: hit for hit in fts}
    vector_by_id = {hit.chunk_id: hit for hit in vector}
    if mode_used == "hybrid":
        ranked = reciprocal_rank_fusion([list(fts_by_id), list(vector_by_id)])
    else:
        ranked = [(hit.chunk_id, hit.score) for hit in (fts if mode_used == "fts" else vector)]
    ranked = ranked[:k]

    chunks = {
        chunk_id: (fts_by_id.get(chunk_id) or vector_by_id[chunk_id]) for chunk_id, _ in ranked
    }
    info = repo.source_info((c.source_table, c.source_id) for c in chunks.values())
    hits = []
    for chunk_id, score in ranked:
        chunk = chunks[chunk_id]
        source = info.get((chunk.source_table, chunk.source_id))
        hits.append(
            SearchHit(
                chunk_id=chunk_id,
                source_table=chunk.source_table,
                source_id=chunk.source_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=score,
                title=source.title if source else _fallback_title(chunk),
                fts_rank=fts_by_id[chunk_id].rank if chunk_id in fts_by_id else None,
                vector_rank=vector_by_id[chunk_id].rank if chunk_id in vector_by_id else None,
            )
        )
    return SearchResult(query=query, mode=mode, mode_used=mode_used, hits=hits, note=note)


def _vector_unavailable(
    repo: Repository, embedder: Embedder | None, sources: tuple[str, ...] | None
) -> str | None:
    """Why vector search cannot contribute, or None if it can."""
    if embedder is None:
        return "No embedder is configured"
    if repo.embedded_count(embedder.model, sources=sources) == 0:
        return (
            f"None of the searched chunks has a vector from {embedder.model} (run `kinship embed`)"
        )
    return None


def _fallback_title(chunk: ChunkHit) -> str:
    """For a chunk whose source row is gone: the chunk's own first line."""
    first_line = chunk.text.strip().splitlines()[0] if chunk.text.strip() else ""
    return first_line[:120].rstrip(":") or f"{chunk.source_table} #{chunk.source_id}"
