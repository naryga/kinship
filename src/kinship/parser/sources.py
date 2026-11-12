"""Locate the allowlisted files inside a zip or an extracted directory.

Listing names and timestamps never opens a file. Only entries selected by
:func:`select_entries` are ever read.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

_SEPARATORS = re.compile(r"[ _]+")


@dataclass(frozen=True)
class Entry:
    """One file in the export: its relative posix path, mtime and a lazy reader."""

    name: str
    modified: datetime | None
    read: Callable[[], bytes]


def zip_entries(archive: zipfile.ZipFile) -> list[Entry]:
    entries = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        try:
            modified = datetime(*info.date_time, tzinfo=UTC)
        except ValueError:
            modified = None
        entries.append(
            Entry(info.filename, modified, lambda info=info: archive.read(info)),
        )
    return entries


def directory_entries(root: Path) -> list[Entry]:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        entries.append(Entry(path.relative_to(root).as_posix(), modified, path.read_bytes))
    return entries


def _stem_pattern(filename: str) -> re.Pattern[str]:
    stem = _SEPARATORS.sub("_", filename.lower().removesuffix(".csv"))
    # The base name, then only numeric suffixes: `_<memberid>` and split parts
    # such as `_1`. Anything else (`Profile Summary`, `Job Applicant Saved
    # Answers`) is a different file and must not match.
    return re.compile(re.escape(stem) + r"((?:_\d+)*)")


class Matcher:
    """Maps a path inside the export to a logical allowlisted file, or nothing."""

    def __init__(self, filenames: dict[str, str]) -> None:
        self._patterns = {key: _stem_pattern(name) for key, name in filenames.items()}

    def match(self, name: str) -> tuple[str, tuple[int, ...]] | None:
        """Return ``(logical key, numeric suffixes)`` for an allowlisted path."""
        basename = PurePosixPath(name.replace("\\", "/")).name.lower()
        if not basename.endswith(".csv"):
            return None
        stem = _SEPARATORS.sub("_", basename.removesuffix(".csv").strip())
        for key, pattern in self._patterns.items():
            found = pattern.fullmatch(stem)
            if found:
                suffixes = tuple(int(part) for part in found.group(1).split("_") if part)
                return key, suffixes
        return None


def select_entries(
    entries: list[Entry], matcher: Matcher, warnings: list[str]
) -> dict[str, list[Entry]]:
    """Group allowlisted entries per logical file, split parts in numeric order.

    If the same file name appears in more than one folder, the shallowest copy
    wins and the others are ignored (unopened) with a warning.
    """
    chosen: dict[str, dict[tuple[int, ...], Entry]] = {}
    for entry in sorted(entries, key=lambda e: (e.name.count("/"), e.name.lower(), e.name)):
        matched = matcher.match(entry.name)
        if matched is None:
            continue
        key, suffixes = matched
        parts = chosen.setdefault(key, {})
        if suffixes in parts:
            warnings.append(
                f"{entry.name}: duplicate of {parts[suffixes].name}; ignored",
            )
            continue
        parts[suffixes] = entry
    return {key: [parts[suffix] for suffix in sorted(parts)] for key, parts in chosen.items()}
