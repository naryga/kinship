"""Owner-scoped query building. Every read in Kinship goes through this module.

The rule (see `AGENTS.md` and `docs/DESIGN.md`):

- Tool, resource and prompt code (`kinship.server`) never imports
  `kinship.db`, a `Session` or `sqlalchemy`. It calls methods on a
  `Repository`. `tests/server/test_layering.py` enforces this.
- Repository code never calls `sqlalchemy.select()` or touches the session
  directly. It builds statements with the helpers below, which add the owner
  filter, and runs them with `scalars` / `scalar` / `rows`.
  `tests/repo/test_layering.py` enforces this.

Two guards keep one owner's rows away from another:

1. `select`, `get`, `count` and `exists` always add `Model.owner_id == owner_id`
   for the model they are given. For a join, add `self.owned(Other)` for the
   other side as well, so both sides are filtered explicitly.
2. As a backstop, every statement is executed with a `with_loader_criteria`
   option that applies the same filter to *every* owned entity in it, including
   joined ones and aliases. A forgotten `owned()` on a join therefore still
   cannot return another owner's rows.

Adding queries (later issues): create `repo/<area>.py` with a class that
subclasses `BaseRepository`, then add it to the `Repository` facade in
`repo/__init__.py` on its own line.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import sqlalchemy as sa
from sqlalchemy.orm import Session, with_loader_criteria

from kinship.db.models import Base, OwnedMixin

T = TypeVar("T")

LOCAL_SETTINGS = frozenset({"hnsw.ef_search"})
"""Settings a repository may change for the rest of its own transaction."""


def owned_models() -> list[type[Any]]:
    """Every mapped model that carries `owner_id`, ordered by table name."""
    models = [m.class_ for m in Base.registry.mappers if issubclass(m.class_, OwnedMixin)]
    return sorted(models, key=lambda model: model.__tablename__)


def _require_owned(model: Any) -> None:
    if getattr(model, "owner_id", None) is None:
        raise TypeError(f"{model!r} is not an owned model: it has no owner_id column")


class OwnerScope:
    """An owner id plus the helpers that turn it into SQL filters."""

    owner_id: int

    def __init__(self, owner_id: int) -> None:
        # bool is an int subclass; reject it along with None, str, etc.
        if isinstance(owner_id, bool) or not isinstance(owner_id, int):
            raise TypeError(f"owner_id must be an int, got {type(owner_id).__name__}")
        if owner_id <= 0:
            raise ValueError("owner_id must be a positive integer")
        self.owner_id = owner_id

    def owned(self, *models: Any) -> sa.ColumnElement[bool]:
        """The owner filter for one or more owned models (or aliases of them).

        Use it for the joined side of a join:

            self.select(Connection)
                .join(Company, Connection.company_id == Company.id)
                .where(self.owned(Company))
        """
        if not models:
            raise TypeError("owned() needs at least one model")
        for model in models:
            _require_owned(model)
        return sa.and_(*(model.owner_id == self.owner_id for model in models))

    def select(self, model: Any, *columns: Any) -> sa.Select[Any]:
        """`SELECT ... FROM model WHERE model.owner_id = :owner_id`.

        With no `columns` this selects the entity; otherwise it selects just
        those columns or expressions (e.g. `func.count()`), still from `model`.
        Chain `.where()`, `.join()`, `.order_by()`, `.limit()` as usual.
        """
        stmt = sa.select(*columns).select_from(model) if columns else sa.select(model)
        return stmt.where(self.owned(model))


class BaseRepository(OwnerScope):
    """Base class for every repository: a session bound to exactly one owner."""

    def __init__(self, session: Session, owner_id: int) -> None:
        super().__init__(owner_id)
        if not isinstance(session, Session):
            raise TypeError("session must be a SQLAlchemy Session")
        self._session = session

    # -- running statements -------------------------------------------------

    def _scoped(self, stmt: Any) -> Any:
        """Add the backstop owner filter for every owned entity in `stmt`."""
        owner_id = self.owner_id
        return stmt.options(
            with_loader_criteria(
                OwnedMixin,
                lambda cls: cls.owner_id == owner_id,
                include_aliases=True,
            )
        )

    def scalars(self, stmt: sa.Select[Any]) -> Sequence[Any]:
        """All results of a single-entity or single-column statement."""
        return self._session.scalars(self._scoped(stmt)).all()

    def scalar(self, stmt: sa.Select[Any]) -> Any:
        """The first column of the first row, or None if there is no row."""
        return self._session.scalars(self._scoped(stmt)).first()

    def rows(self, stmt: sa.Select[Any]) -> Sequence[sa.Row[Any]]:
        """All rows of a multi-column statement."""
        return self._session.execute(self._scoped(stmt)).all()

    def set_local(self, name: str, value: int | str) -> None:
        """Like `SET LOCAL name = value`: change a setting until the transaction ends.

        Only names in `LOCAL_SETTINGS` are accepted, and name and value are sent
        as bound parameters. It reads no table, so there is nothing to scope.
        """
        if name not in LOCAL_SETTINGS:
            raise ValueError(f"{name!r} is not a setting a repository may change")
        self._session.execute(sa.select(sa.func.set_config(name, str(value), True)))

    # -- common queries -----------------------------------------------------

    def get(self, model: type[T], row_id: int) -> T | None:
        """The row with this primary key, or None if it is missing or another owner's."""
        return self.scalar(self.select(model).where(model.id == row_id))  # type: ignore[attr-defined]

    def count(self, model: Any, *criteria: sa.ColumnElement[bool]) -> int:
        """Number of this owner's rows in `model` matching `criteria`."""
        return self.scalar(self.select(model, sa.func.count()).where(*criteria)) or 0

    def exists(self, model: Any, *criteria: sa.ColumnElement[bool]) -> bool:
        """Whether this owner has any row in `model` matching `criteria`."""
        return self.scalar(self.select(model, model.id).where(*criteria).limit(1)) is not None
