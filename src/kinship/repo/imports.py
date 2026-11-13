"""Import bookkeeping queries: the latest import and per-table row counts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from kinship.db.models import Import
from kinship.repo.base import BaseRepository, owned_models


@dataclass(frozen=True)
class ImportInfo:
    """One imported LinkedIn export, detached from the session."""

    id: int
    source_sha256: str
    exported_at: datetime | None
    imported_at: datetime
    file_counts: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form (timestamps as ISO 8601 strings)."""
        data = asdict(self)
        data["exported_at"] = self.exported_at.isoformat() if self.exported_at else None
        data["imported_at"] = self.imported_at.isoformat()
        return data


class ImportsRepository(BaseRepository):
    def latest_import(self) -> ImportInfo | None:
        """The owner's newest import, or None before the first one.

        "Newest" is the latest `exported_at` (the export's own date), not the
        one loaded last: an old export imported late is not the latest. Imports
        without an export date sort last; ties go to the higher id.
        """
        row = self.scalar(
            self.select(Import)
            .order_by(Import.exported_at.desc().nulls_last(), Import.id.desc())
            .limit(1)
        )
        if row is None:
            return None
        return ImportInfo(
            id=row.id,
            source_sha256=row.source_sha256,
            exported_at=row.exported_at,
            imported_at=row.imported_at,
            file_counts=dict(row.file_counts or {}),
        )

    def counts(self) -> dict[str, int]:
        """Row counts for this owner, keyed by table name, for every owned table."""
        counts: dict[str, int] = {}
        for model in owned_models():
            counts[model.__tablename__] = self.count(model)
        return counts
