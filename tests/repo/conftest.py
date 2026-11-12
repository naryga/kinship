"""Fixtures for repository tests: two owners with the same shape of data.

Everything lives in the rolled-back `session` fixture from `tests/conftest.py`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest
from seeding import SeededNetwork, seed_network
from sqlalchemy.orm import Session

from kinship.db.models import Owner
from kinship.repo import Repository


@dataclass(frozen=True)
class TwoOwners:
    a: SeededNetwork
    b: SeededNetwork
    repo_a: Repository
    repo_b: Repository


@pytest.fixture
def two_owners(session: Session, make_owner: Callable[..., Owner]) -> TwoOwners:
    """Owner A (3 connections) and owner B (5 connections), each with a `Repository`."""
    a = seed_network(session, make_owner("Owner A"), connections=3, tag="alpha")
    b = seed_network(session, make_owner("Owner B"), connections=5, tag="beta")
    return TwoOwners(
        a=a,
        b=b,
        repo_a=Repository(session, a.owner_id),
        repo_b=Repository(session, b.owner_id),
    )
