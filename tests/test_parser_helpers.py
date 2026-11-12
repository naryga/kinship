"""Unit tests for the parser's date and profile-URL helpers."""

from datetime import UTC, date, datetime

import pytest

from kinship.parser import canonical_profile_url, parse_date, parse_datetime

JAN_15 = datetime(2024, 1, 15, 14, 23, 11, tzinfo=UTC)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("15 Jan 2024", date(2024, 1, 15)),
        ("5 Jan 2024", date(2024, 1, 5)),
        ("Jan 2020", date(2020, 1, 1)),
        ("September 2019", date(2019, 9, 1)),
        ("2020", date(2020, 1, 1)),
        ("2024-01-15", date(2024, 1, 15)),
        ("1/15/2024", date(2024, 1, 15)),
        ("01/15/24", date(2024, 1, 15)),
        ("2024/01/15 14:23:11 UTC", date(2024, 1, 15)),
        ("2024-01-15 14:23:11 UTC", date(2024, 1, 15)),
        ("01/15/24, 02:23 PM", date(2024, 1, 15)),
        ("Mon Jan 15 14:23:11 UTC 2024", date(2024, 1, 15)),
        ("2024-01-15T23:30:00-05:00", date(2024, 1, 16)),
        ("  15 Jan 2024  ", date(2024, 1, 15)),
    ],
)
def test_parse_date_formats(raw: str, expected: date) -> None:
    assert parse_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024/01/15 14:23:11 UTC", JAN_15),
        ("2024-01-15 14:23:11 UTC", JAN_15),
        ("2024-01-15 14:23:11", JAN_15),
        ("Mon Jan 15 14:23:11 UTC 2024", JAN_15),
        ("2024-01-15T14:23:11Z", JAN_15),
        ("2024-01-15T14:23:11+00:00", JAN_15),
        ("2024-01-15T16:23:11+02:00", JAN_15),
        ("2024-01-15T14:23:11", JAN_15),
        ("01/15/24, 02:23 PM", datetime(2024, 1, 15, 14, 23, tzinfo=UTC)),
        ("1/15/24, 2:23 PM", datetime(2024, 1, 15, 14, 23, tzinfo=UTC)),
        ("1/15/24, 2:23\u202fAM", datetime(2024, 1, 15, 2, 23, tzinfo=UTC)),
        ("1/15/2024", datetime(2024, 1, 15, tzinfo=UTC)),
        ("15 Jan 2024", datetime(2024, 1, 15, tzinfo=UTC)),
        ("2024-01-15", datetime(2024, 1, 15, tzinfo=UTC)),
        ("Jan 2020", datetime(2020, 1, 1, tzinfo=UTC)),
        ("2020", datetime(2020, 1, 1, tzinfo=UTC)),
    ],
)
def test_parse_datetime_formats(raw: str, expected: datetime) -> None:
    parsed = parse_datetime(raw)
    assert parsed == expected
    assert parsed is not None
    assert parsed.utcoffset() == UTC.utcoffset(None)


@pytest.mark.parametrize("raw", [None, "", "   ", "not a date", "13/45/2024", "20"])
def test_blank_or_unparseable_dates_are_none(raw: str | None) -> None:
    assert parse_date(raw) is None
    assert parse_datetime(raw) is None


CANONICAL = "https://www.linkedin.com/in/ada-example-123"


@pytest.mark.parametrize(
    "raw",
    [
        "https://www.linkedin.com/in/ada-example-123",
        "https://www.linkedin.com/in/ada-example-123/",
        "http://www.linkedin.com/in/ada-example-123",
        "HTTPS://WWW.LinkedIn.com/in/Ada-Example-123",
        "https://linkedin.com/in/ada-example-123",
        "https://uk.linkedin.com/in/ada-example-123",
        "www.linkedin.com/in/ada-example-123",
        "linkedin.com/in/ada-example-123/",
        "//www.linkedin.com/in/ada-example-123",
        "https://www.linkedin.com:443/in/ada-example-123",
        "https://www.linkedin.com/in/ada-example-123?trk=abc&x=1",
        "https://www.linkedin.com/in/ada-example-123/#about",
        "https://www.linkedin.com//in/ada%2Dexample%2D123",
        "  https://www.linkedin.com/in/ada-example-123  ",
    ],
)
def test_profile_urls_collapse_to_one_form(raw: str) -> None:
    assert canonical_profile_url(raw) == CANONICAL


def test_profile_url_percent_encoding_is_normalized() -> None:
    expected = "https://www.linkedin.com/in/jos%C3%A9-example"
    assert canonical_profile_url("https://www.linkedin.com/in/jos%c3%a9-example") == expected
    assert canonical_profile_url("https://www.linkedin.com/in/JOS%C3%89-example") == expected
    assert canonical_profile_url("https://www.linkedin.com/in/josé-example/") == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "n/a"])
def test_blank_profile_url_is_none(raw: str | None) -> None:
    assert canonical_profile_url(raw) is None
