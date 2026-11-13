"""Owner-bound database access for the importer.

`ImportStore` is the only place the importer touches the database. It is built
for exactly one owner, and every statement it issues filters on that owner's
id (lookups by id included) or writes rows stamped with it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, NamedTuple

from sqlalchemy import bindparam, delete, func, insert, select, update
from sqlalchemy.orm import Session

from kinship.db.models import (
    ConnectionHistory,
    ConnectionSighting,
    Import,
    Message,
    OwnedMixin,
    Owner,
)


class SightingRow(NamedTuple):
    connection_id: int
    import_id: int
    company_raw: str | None
    position: str | None


class HistoryRow(NamedTuple):
    id: int
    connection_id: int
    import_id: int
    company_raw: str | None
    position: str | None


class MessageRow(NamedTuple):
    """A stored message without its body; `body_sha256` stands in for it."""

    id: int
    conversation_id: str | None
    sent_at: datetime | None
    from_name: str | None
    to_names: list[str]
    body_sha256: str
    connection_id: int | None


class ImportStore:
    """All reads and writes of one owner's rows during an import."""

    def __init__(self, session: Session, owner_id: int) -> None:
        if not isinstance(owner_id, int) or isinstance(owner_id, bool):
            raise TypeError("owner_id must be an int")
        self._session = session
        self.owner_id = owner_id

    # -- owner and imports -------------------------------------------------

    def lock_owner(self) -> bool:
        """Lock the owner row so imports for one owner run one at a time.

        Returns False if the owner does not exist.
        """
        found = self._session.scalar(
            select(Owner.id).where(Owner.id == self.owner_id).with_for_update()
        )
        return found is not None

    def find_import(self, source_sha256: str) -> Import | None:
        return self._session.scalar(
            select(Import).where(
                Import.owner_id == self.owner_id, Import.source_sha256 == source_sha256
            )
        )

    def imports(self) -> list[Import]:
        return self.rows(Import)

    # -- generic owned-table access ---------------------------------------

    def rows[T: OwnedMixin](self, model: type[T]) -> list[T]:
        """Every row of `model` that belongs to this owner, oldest id first."""
        statement = select(model).where(model.owner_id == self.owner_id).order_by(model.id)  # type: ignore[attr-defined]
        return list(self._session.scalars(statement))

    def add[T: OwnedMixin](self, row: T) -> T:
        """Stamp `row` with this owner and add it to the session."""
        row.owner_id = self.owner_id
        self._session.add(row)
        return row

    def insert_many(self, model: type[OwnedMixin], rows: Iterable[Mapping[str, Any]]) -> int:
        """Bulk-insert `rows` (column name -> value), each stamped with this owner."""
        values = [{**row, "owner_id": self.owner_id} for row in rows]
        if values:
            self._session.execute(insert(model.__table__), values)  # type: ignore[attr-defined]
        return len(values)

    def delete_ids(self, model: type[OwnedMixin], ids: Sequence[int]) -> int:
        """Delete this owner's rows of `model` with the given ids."""
        if not ids:
            return 0
        self._session.execute(
            delete(model)
            .where(model.owner_id == self.owner_id, model.id.in_(ids))  # type: ignore[attr-defined]
            .execution_options(synchronize_session=False)
        )
        return len(ids)

    def flush(self) -> None:
        self._session.flush()

    def values(self, model: type[OwnedMixin], *columns: str) -> list[tuple[Any, ...]]:
        """`(id, *columns)` for every row of `model` that belongs to this owner."""
        table = model.__table__  # type: ignore[attr-defined]
        result = self._session.execute(
            select(table.c.id, *(table.c[name] for name in columns))
            .where(table.c.owner_id == self.owner_id)
            .order_by(table.c.id)
        )
        return [tuple(row) for row in result]

    def set_values(self, model: type[OwnedMixin], column: str, values: Mapping[int, Any]) -> None:
        """Set one column on this owner's rows of `model` (row id -> new value)."""
        if not values:
            return
        table = model.__table__  # type: ignore[attr-defined]
        self._session.execute(
            update(table)
            .where(table.c.owner_id == self.owner_id, table.c.id == bindparam("row_id"))
            .values({column: bindparam("new_value")}),
            [{"row_id": row_id, "new_value": value} for row_id, value in values.items()],
        )
        # The update bypassed the ORM: make loaded rows re-read the column.
        for loaded in list(self._session.identity_map.values()):
            if isinstance(loaded, model) and loaded.id in values:  # type: ignore[attr-defined]
                self._session.expire(loaded, [column])

    # -- column-only reads for the larger tables ---------------------------

    def sightings(self) -> list[SightingRow]:
        table = ConnectionSighting
        result = self._session.execute(
            select(table.connection_id, table.import_id, table.company_raw, table.position).where(
                table.owner_id == self.owner_id
            )
        )
        return [SightingRow(*row) for row in result]

    def history(self) -> list[HistoryRow]:
        table = ConnectionHistory
        result = self._session.execute(
            select(
                table.id, table.connection_id, table.import_id, table.company_raw, table.position
            ).where(table.owner_id == self.owner_id)
        )
        return [HistoryRow(*row) for row in result]

    def messages(self) -> list[MessageRow]:
        """This owner's messages, with the body hashed in the database, not loaded."""
        body_sha256 = func.encode(
            func.sha256(func.convert_to(func.coalesce(Message.body, ""), "UTF8")), "hex"
        )
        result = self._session.execute(
            select(
                Message.id,
                Message.conversation_id,
                Message.sent_at,
                Message.from_name,
                Message.to_names,
                body_sha256,
                Message.connection_id,
            )
            .where(Message.owner_id == self.owner_id)
            .order_by(Message.id)
        )
        return [MessageRow(*row) for row in result]

    def set_message_connections(self, links: Mapping[int, int | None]) -> None:
        """Set `connection_id` on this owner's messages (message id -> connection id)."""
        if not links:
            return
        table = Message.__table__
        self._session.execute(
            update(table)
            .where(table.c.owner_id == self.owner_id, table.c.id == bindparam("message_id"))
            .values(connection_id=bindparam("linked_connection_id")),
            [
                {"message_id": message_id, "linked_connection_id": connection_id}
                for message_id, connection_id in links.items()
            ],
        )
