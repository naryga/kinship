"""The repository layer: the only code that queries owned tables.

    with open_repository(session_factory, owner_id) as repo:
        repo.counts()

`Repository` is the single facade. Each area of queries lives in its own module
as a `BaseRepository` subclass and is mixed in below. To add one, import it and
add it to the base list on its own line (see `kinship.repo.base` for the rules).
"""

from kinship.repo.base import BaseRepository, OwnerScope, owned_models
from kinship.repo.chunks import ChunksRepository
from kinship.repo.companies import CompaniesRepository, CompanyMatch
from kinship.repo.imports import ImportInfo, ImportsRepository
from kinship.repo.session import (
    SessionFactory,
    database_is_up,
    default_session_factory,
    open_repository,
)


# One mixin per line so that parallel branches merge cleanly.
class Repository(
    ChunksRepository,
    CompaniesRepository,
    ImportsRepository,
    BaseRepository,
):
    """All owner-scoped queries, bound to one session and one owner."""


__all__ = [
    "BaseRepository",
    "ChunksRepository",
    "CompaniesRepository",
    "CompanyMatch",
    "ImportInfo",
    "ImportsRepository",
    "OwnerScope",
    "Repository",
    "SessionFactory",
    "database_is_up",
    "default_session_factory",
    "open_repository",
    "owned_models",
]
