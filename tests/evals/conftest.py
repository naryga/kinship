"""Fixtures for the retrieval evaluation tests.

- `golden`: the committed golden set.
- `eval_owners`: owners A and B with the synthetic v2 export, inside the
  rolled-back `session` fixture. B also has rows that reuse A's natural keys
  and the golden queries' words. Chunks are embedded with `FakeEmbedder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from ingest.conftest import V2_AT, V2_DIR
from sqlalchemy.orm import Session

from kinship.db.models import MyPosition, Owner, Recommendation
from kinship.ingest import import_export
from kinship.parser import ParsedExport, parse_export
from kinship.rag import FakeEmbedder, sync_chunks
from kinship.rag.evaluate import GoldenQuery, default_golden_path, load_golden

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_FLOOR = 0.65
"""The `--min-recall` that `.github/workflows/ci.yml` passes."""


@dataclass(frozen=True)
class EvalOwners:
    a: int
    b: int


@pytest.fixture(scope="session")
def golden() -> list[GoldenQuery]:
    return load_golden(default_golden_path())


@pytest.fixture(scope="session")
def v2_parsed() -> ParsedExport:
    return parse_export(V2_DIR, exported_at=V2_AT)


def add_owner(session: Session, name: str, export: ParsedExport) -> int:
    """A new owner with `export` imported and embedded. Does not commit."""
    owner = Owner(display_name=name)
    session.add(owner)
    session.flush()
    import_export(session, owner.id, export)
    sync_chunks(session, owner.id, FakeEmbedder())
    session.flush()
    return owner.id


def add_decoys(session: Session, owner_id: int, golden: list[GoldenQuery]) -> None:
    """Give `owner_id` rows that answer every golden query, and embed them.

    One recommendation repeats every query text under a name the golden set
    uses; one position reuses a golden company and title. If either leaked to
    another owner it would outrank that owner's own rows, or make a golden
    item ambiguous.
    """
    everything = " ".join(query.query for query in golden)
    session.add_all(
        [
            Recommendation(
                owner_id=owner_id,
                direction="received",
                first_name="Freya",
                last_name="Rookwood",
                text=everything,
            ),
            Recommendation(
                owner_id=owner_id,
                direction="given",
                first_name="Onlybee",
                last_name="Zeppelin",
                text=everything,
            ),
            MyPosition(
                owner_id=owner_id,
                company="Copperline Freight",
                title="Junior Developer",
                description=everything,
            ),
        ]
    )
    session.flush()
    sync_chunks(session, owner_id, FakeEmbedder())
    session.flush()


@pytest.fixture
def eval_owners(session: Session, v2_parsed: ParsedExport, golden: list[GoldenQuery]) -> EvalOwners:
    a = add_owner(session, "Eval A", v2_parsed)
    b = add_owner(session, "Eval B", v2_parsed)
    add_decoys(session, b, golden)
    return EvalOwners(a=a, b=b)
