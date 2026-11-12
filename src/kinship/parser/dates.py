"""Liberal date parsing for the mixed formats LinkedIn exports use."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime

_ZONE = re.compile(r"\s*\b(?:UTC|GMT)\b", re.IGNORECASE)
_SPACES = re.compile(r"\s+")

# Tried in order. The "UTC"/"GMT" token is removed before matching.
_DATETIME_FORMATS = (
    "%Y/%m/%d %H:%M:%S",  # 2024/01/15 14:23:11 UTC
    "%Y-%m-%d %H:%M:%S",  # 2024-01-15 14:23:11 UTC
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%m/%d/%y, %I:%M %p",  # 01/15/24, 02:23 PM
    "%m/%d/%Y, %I:%M %p",
    "%m/%d/%y, %I:%M:%S %p",
    "%m/%d/%Y, %I:%M:%S %p",
    "%a %b %d %H:%M:%S %Y",  # Mon Jan 15 14:23:11 UTC 2024
)

# Partial dates resolve to the first day of the period.
_DATE_FORMATS = (
    "%d %b %Y",  # 15 Jan 2024
    "%d %B %Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",  # 1/15/2024
    "%m/%d/%y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %Y",  # Jan 2020 -> 2020-01-01
    "%B %Y",
    "%Y-%m",
    "%Y",  # 2020 -> 2020-01-01
)


def _clean(raw: str | None) -> str:
    if raw is None:
        return ""
    # Collapses narrow no-break spaces (seen before AM/PM) and friends.
    return _SPACES.sub(" ", raw).strip()


def _strptime(text: str, formats: tuple[str, ...]) -> datetime | None:
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_date_only(text: str) -> date | None:
    parsed = _strptime(text, _DATE_FORMATS)
    return parsed.date() if parsed else None


def _parse_datetime_only(text: str) -> datetime | None:
    parsed = _strptime(_ZONE.sub("", text).strip(), _DATETIME_FORMATS)
    if parsed is not None:
        return parsed.replace(tzinfo=UTC)
    # ISO 8601 with a time part; plain dates are left to the date formats.
    if len(text) > 10 and text[:4].isdigit():
        try:
            return _as_utc(datetime.fromisoformat(text))
        except ValueError:
            return None
    return None


def parse_datetime(raw: str | None) -> datetime | None:
    """Parse any observed timestamp format into an aware UTC datetime.

    Date-only values become midnight UTC. Blank or unparseable input returns
    ``None``; callers decide whether a non-blank ``None`` deserves a warning.
    """
    text = _clean(raw)
    if not text:
        return None
    parsed = _parse_datetime_only(text)
    if parsed is not None:
        return parsed
    day = _parse_date_only(text)
    if day is not None:
        return datetime(day.year, day.month, day.day, tzinfo=UTC)
    return None


def parse_date(raw: str | None) -> date | None:
    """Parse any observed date format into a ``date``.

    Month-only values become the first of the month and year-only values
    January 1st. Timestamps are reduced to their UTC date. Blank or unparseable
    input returns ``None``.
    """
    text = _clean(raw)
    if not text:
        return None
    day = _parse_date_only(text)
    if day is not None:
        return day
    parsed = _parse_datetime_only(text)
    return parsed.date() if parsed else None
