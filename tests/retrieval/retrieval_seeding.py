"""Data and helpers for retrieval tests: two owners with the same synthetic export.

`build_corpus` imports the synthetic v2 export for owners A and B and embeds
every chunk with `FakeEmbedder`, so their chunk texts are identical. B also has
one extra recommendation containing `B_ONLY_WORD`, which A must never see.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx2
import sqlalchemy as sa
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from sqlalchemy.orm import Session
from starlette.applications import Starlette

from kinship.config import Settings
from kinship.db.models import Chunk, Owner, Recommendation
from kinship.ingest import import_export
from kinship.parser import ParsedExport
from kinship.rag import FakeEmbedder, sync_chunks

BASE_URL = "http://127.0.0.1:8000"
TOKEN_A = "retrieval-token-a"
TOKEN_B = "retrieval-token-b"

B_ONLY_WORD = "zeppelin"
B_ONLY_TEXT = f"Sam piloted our {B_ONLY_WORD} logistics programme and nobody else knows about it."


@dataclass(frozen=True)
class Corpus:
    a: int
    b: int


def build_corpus(session: Session, export: ParsedExport) -> Corpus:
    """Two owners with the same export; B gets one extra recommendation. Does not commit."""
    owners = [Owner(display_name="Retrieval A"), Owner(display_name="Retrieval B")]
    session.add_all(owners)
    session.flush()
    for owner in owners:
        import_export(session, owner.id, export)
    session.add(
        Recommendation(
            owner_id=owners[1].id,
            direction="received",
            first_name="Zed",
            last_name="Onlybee",
            text=B_ONLY_TEXT,
        )
    )
    session.flush()
    for owner in owners:
        sync_chunks(session, owner.id, FakeEmbedder())
    session.flush()
    return Corpus(a=owners[0].id, b=owners[1].id)


def chunk_ids(session: Session, owner_id: int) -> set[int]:
    """Ids of this owner's chunks, straight from the table."""
    return set(session.scalars(sa.select(Chunk.id).where(Chunk.owner_id == owner_id)))


def vec(text: str) -> list[float]:
    return FakeEmbedder().embed_query(text)


def settings_for(corpus: Corpus, **overrides: str) -> Settings:
    """Settings with a token per owner and no embedder configuration, ignoring `.env`."""
    values = {
        "kinship_tokens": f"{TOKEN_A}:{corpus.a},{TOKEN_B}:{corpus.b}",
        "openai_api_key": "",
        "kinship_embedder": "",
        **overrides,
    }
    return Settings(_env_file=None, **values)


@asynccontextmanager
async def connect(app: Starlette, token: str) -> AsyncIterator[Client]:
    """A connected SDK `Client` for `app`, authenticated with `token`."""
    async with (
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {token}"},
        ) as http,
        Client(streamable_http_client(f"{BASE_URL}/mcp", http_client=http)) as client,
    ):
        yield client
