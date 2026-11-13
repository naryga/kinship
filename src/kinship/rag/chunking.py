"""Turning an owner's rows into chunk text. Pure functions, no database.

Per `docs/DESIGN.md`: one chunk per recommendation, one per position, one for
the profile, and (only when asked for) messages grouped by conversation and
packed into chunks of about 500 tokens.

`source_table` is the real table name, so retrieval can filter by source.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from kinship.repo.chunks import (
    ChunkSources,
    MessageSource,
    PositionSource,
    ProfileSource,
    RecommendationSource,
)

RECOMMENDATIONS = "recommendations"
POSITIONS = "my_positions"
PROFILES = "profiles"
MESSAGES = "messages"
SOURCE_TABLES = (RECOMMENDATIONS, POSITIONS, PROFILES, MESSAGES)

MAX_CHUNK_TOKENS = 500
CHARS_PER_TOKEN = 4

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk as it should exist, before it is compared with the database."""

    source_table: str
    source_id: int
    chunk_index: int
    text: str

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.source_table, self.source_id, self.chunk_index)

    @property
    def content_hash(self) -> str:
        return content_hash(self.text)


def content_hash(text: str) -> str:
    """SHA-256 of the chunk text, as stored in `chunks.content_hash`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> int:
    """A cheap token estimate: about four characters per token, rounded up."""
    return -(-len(text) // CHARS_PER_TOKEN)


def build_drafts(sources: ChunkSources, *, max_tokens: int = MAX_CHUNK_TOKENS) -> list[ChunkDraft]:
    """Every chunk for one owner's rows. Rows with no meaningful text give none."""
    drafts: list[ChunkDraft] = []
    drafts += filter(None, map(recommendation_chunk, sources.recommendations))
    drafts += filter(None, map(position_chunk, sources.positions))
    if sources.profile is not None:
        drafts += filter(None, [profile_chunk(sources.profile)])
    drafts += message_chunks(sources.messages, max_tokens=max_tokens)
    return drafts


# -- one chunk per row --------------------------------------------------------


def _clean(value: str | None) -> str:
    return (value or "").strip()


def recommendation_chunk(row: RecommendationSource) -> ChunkDraft | None:
    """`Recommendation received from <name>, <title> at <company>:` then the text."""
    text = _clean(row.text)
    if not text:
        return None
    header = "Recommendation given" if row.direction == "given" else "Recommendation received"
    name = " ".join(filter(None, [_clean(row.first_name), _clean(row.last_name)]))
    if name:
        header += (" to " if row.direction == "given" else " from ") + name
    role = " at ".join(filter(None, [_clean(row.job_title), _clean(row.company)]))
    if role:
        header += f", {role}" if name else f" ({role})"
    return ChunkDraft(RECOMMENDATIONS, row.id, 0, f"{header}:\n{text}")


def _month(value: date) -> str:
    return f"{_MONTHS[value.month - 1]} {value.year}"


def position_chunk(row: PositionSource) -> ChunkDraft | None:
    """`Position: <title> at <company> (<start> - <end>), <location>` then the description."""
    title, company = _clean(row.title), _clean(row.company)
    description = _clean(row.description)
    if not (title or company or description):
        return None
    header = "Position: " + (" at ".join(filter(None, [title, company])) or "(untitled)")
    if row.started_on or row.ended_on:
        start = _month(row.started_on) if row.started_on else "?"
        end = _month(row.ended_on) if row.ended_on else "present"
        header += f" ({start} \N{EN DASH} {end})"
    if _clean(row.location):
        header += f", {_clean(row.location)}"
    return ChunkDraft(POSITIONS, row.id, 0, "\n".join(filter(None, [header, description])))


def profile_chunk(row: ProfileSource) -> ChunkDraft | None:
    """`Profile: <headline>`, `Industry: <industry>`, then the summary."""
    headline, industry, summary = _clean(row.headline), _clean(row.industry), _clean(row.summary)
    if not (headline or industry or summary):
        return None
    lines = [
        f"Profile: {headline}" if headline else "Profile",
        f"Industry: {industry}" if industry else "",
        summary,
    ]
    return ChunkDraft(PROFILES, row.id, 0, "\n".join(filter(None, lines)))


# -- messages -----------------------------------------------------------------


def split_text(text: str, limit: int) -> list[str]:
    """Split `text` into pieces of at most `limit` characters.

    Prefers sentence ends, then whitespace; a single word longer than `limit`
    is cut. Text that already fits is returned unchanged.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return [text]
    units: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        if len(sentence) <= limit:
            units.append(sentence)
            continue
        for word in sentence.split():
            units += [word[i : i + limit] for i in range(0, len(word), limit)]
    pieces: list[str] = []
    current = ""
    for unit in units:
        if not unit:
            continue
        if current and len(current) + 1 + len(unit) > limit:
            pieces.append(current)
            current = unit
        else:
            current = f"{current} {unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


def _message_lines(message: MessageSource, limit: int) -> list[str]:
    """`<from_name> (<date>): <body>`, split into several lines if it is too long."""
    body = _clean(message.body)
    if not body:
        return []
    prefix = _clean(message.from_name) or "Unknown"
    if message.sent_at is not None:
        prefix += f" ({message.sent_at.date().isoformat()})"
    prefix += ": "
    room = max(limit - len(prefix), limit // 2)
    return [(prefix + piece)[:limit] for piece in split_text(body, room)]


def _pack(lines: Iterable[str], limit: int) -> list[str]:
    """Greedily join lines into texts of at most `limit` characters."""
    texts: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + 1 + len(line) > limit:
            texts.append("\n".join(current))
            current, size = [], 0
        size += len(line) + (1 if current else 0)
        current.append(line)
    if current:
        texts.append("\n".join(current))
    return texts


def message_chunks(
    messages: Sequence[MessageSource], *, max_tokens: int = MAX_CHUNK_TOKENS
) -> list[ChunkDraft]:
    """Messages grouped by conversation, in time order, about `max_tokens` per chunk.

    A chunk never splits a message unless that one message is over the cap.
    `source_id` is the smallest message id in the conversation, which does not
    change when later messages arrive; `chunk_index` counts chunks within the
    conversation. A message without a `conversation_id` is its own conversation.
    Packing is greedy from the start, so new messages only change the last chunk.
    """
    limit = max_tokens * CHARS_PER_TOKEN
    conversations: dict[tuple[str, object], list[MessageSource]] = {}
    for message in messages:
        key = ("c", message.conversation_id) if message.conversation_id else ("m", message.id)
        conversations.setdefault(key, []).append(message)

    drafts: list[ChunkDraft] = []
    for group in conversations.values():
        group.sort(
            key=lambda m: (m.sent_at is None, m.sent_at.timestamp() if m.sent_at else 0, m.id)
        )
        lines = [line for message in group for line in _message_lines(message, limit)]
        source_id = min(message.id for message in group)
        drafts += [
            ChunkDraft(MESSAGES, source_id, index, text)
            for index, text in enumerate(_pack(lines, limit))
        ]
    drafts.sort(key=lambda d: (d.source_id, d.chunk_index))
    return drafts
