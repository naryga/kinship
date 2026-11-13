"""Retrieval: chunking, embedders and keeping `chunks` up to date.

- `kinship.rag.chunking`: rows -> `ChunkDraft` (pure functions).
- `kinship.rag.embedder`: the `Embedder` protocol, `OpenAIEmbedder`, `FakeEmbedder`.
- `kinship.rag.store`: the owner-bound writer for `chunks`.
- `kinship.rag.indexer`: `sync_chunks`, the job behind `kinship embed`.
"""

from kinship.rag.chunking import ChunkDraft, build_drafts, content_hash
from kinship.rag.embedder import (
    Embedder,
    EmbedderConfigError,
    EmbedderError,
    FakeEmbedder,
    OpenAIEmbedder,
    get_embedder,
)
from kinship.rag.indexer import EmbedSummary, OwnerNotFoundError, resolve_owner_id, sync_chunks

__all__ = [
    "ChunkDraft",
    "EmbedSummary",
    "Embedder",
    "EmbedderConfigError",
    "EmbedderError",
    "FakeEmbedder",
    "OpenAIEmbedder",
    "OwnerNotFoundError",
    "build_drafts",
    "content_hash",
    "get_embedder",
    "resolve_owner_id",
    "sync_chunks",
]
