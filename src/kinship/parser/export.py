"""Parse a LinkedIn data export (zip or extracted directory) into typed records.

The header rows and quirks handled here are documented in
``docs/linkedin-export-format.md``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from kinship.parser.dates import parse_date, parse_datetime
from kinship.parser.records import (
    CompanyFollow,
    Connection,
    Education,
    EndorsementGiven,
    EndorsementReceived,
    Invitation,
    JobApplication,
    Message,
    Note,
    ParsedExport,
    Position,
    Profile,
    ProfileSummary,
    Recommendation,
    SavedJob,
    Skill,
)
from kinship.parser.sources import Entry, Matcher, directory_entries, select_entries, zip_entries
from kinship.parser.urls import canonical_profile_url

# How many leading rows may be preamble before the real header.
_HEADER_SEARCH_ROWS = 10
# Long messages and application answers exceed the csv module's 128 KiB default.
_FIELD_SIZE_LIMIT = 16 * 1024 * 1024
_URL_SEPARATORS = re.compile(r"[;,\s]+")


class ExportError(Exception):
    """The path is not a usable LinkedIn export (bad path or no Connections.csv)."""


class _Row:
    """Cells of one data row, addressed by the expected header's column names."""

    def __init__(self, index: dict[str, int], cells: list[str], line: int, issues: _Issues) -> None:
        self._index = index
        self._cells = cells
        self._line = line
        self._issues = issues

    def text(self, column: str) -> str | None:
        position = self._index.get(column.lower())
        if position is None or position >= len(self._cells):
            return None
        return self._cells[position].strip() or None

    def date(self, column: str) -> date | None:
        return self._convert(column, parse_date)

    def datetime(self, column: str) -> datetime | None:
        return self._convert(column, parse_datetime)

    def url(self, column: str) -> str | None:
        return self._convert(column, canonical_profile_url)

    def urls(self, column: str) -> tuple[str, ...]:
        raw = self.text(column)
        if raw is None:
            return ()
        found = (canonical_profile_url(part) for part in _URL_SEPARATORS.split(raw))
        return tuple(url for url in found if url)

    def _convert(self, column: str, convert: Callable[[str], Any]) -> Any:
        raw = self.text(column)
        if raw is None:
            return None
        value = convert(raw)
        if value is None:
            self._issues.unparseable(column, self._line)
        return value


class _Issues:
    """Counts unparseable values per column so warnings stay short."""

    def __init__(self) -> None:
        self._bad: dict[str, tuple[int, int]] = {}

    def unparseable(self, column: str, line: int) -> None:
        count, first = self._bad.get(column, (0, line))
        self._bad[column] = (count + 1, first)

    def warnings(self, name: str) -> list[str]:
        return [
            f"{name}: {count} unparseable value(s) in column '{column}' "
            f"(first at line {first}); stored as blank"
            for column, (count, first) in self._bad.items()
        ]


@dataclass(frozen=True)
class _FileSpec:
    key: str
    filename: str
    headers: tuple[tuple[str, ...], ...]
    build: Callable[[_Row], Any]


def _columns(header: str) -> tuple[str, ...]:
    return tuple(header.split(","))


def _connection(row: _Row) -> Connection:
    return Connection(
        first_name=row.text("First Name"),
        last_name=row.text("Last Name"),
        profile_url=row.url("URL"),
        email=row.text("Email Address"),
        company=row.text("Company"),
        position=row.text("Position"),
        connected_on=row.date("Connected On"),
    )


def _position(row: _Row) -> Position:
    return Position(
        company_name=row.text("Company Name"),
        title=row.text("Title"),
        description=row.text("Description"),
        location=row.text("Location"),
        started_on=row.date("Started On"),
        finished_on=row.date("Finished On"),
    )


def _profile(row: _Row) -> Profile:
    # Every other Profile.csv column (names, address, birth date, ...) is dropped.
    return Profile(
        headline=row.text("Headline"),
        summary=row.text("Summary"),
        industry=row.text("Industry"),
    )


def _endorsement_received(row: _Row) -> EndorsementReceived:
    return EndorsementReceived(
        endorsed_at=row.datetime("Endorsement Date"),
        skill_name=row.text("Skill Name"),
        endorser_first_name=row.text("Endorser First Name"),
        endorser_last_name=row.text("Endorser Last Name"),
        endorser_profile_url=row.url("Endorser Public Url"),
        status=row.text("Endorsement Status"),
    )


def _endorsement_given(row: _Row) -> EndorsementGiven:
    return EndorsementGiven(
        endorsed_at=row.datetime("Endorsement Date"),
        skill_name=row.text("Skill Name"),
        endorsee_first_name=row.text("Endorsee First Name"),
        endorsee_last_name=row.text("Endorsee Last Name"),
        endorsee_profile_url=row.url("Endorsee Public Url"),
        status=row.text("Endorsement Status"),
    )


def _recommendation(row: _Row) -> Recommendation:
    return Recommendation(
        first_name=row.text("First Name"),
        last_name=row.text("Last Name"),
        company=row.text("Company"),
        job_title=row.text("Job Title"),
        text=row.text("Text"),
        created_at=row.datetime("Creation Date"),
        status=row.text("Status"),
    )


def _invitation(row: _Row) -> Invitation:
    return Invitation(
        from_name=row.text("From"),
        to_name=row.text("To"),
        sent_at=row.datetime("Sent At"),
        message=row.text("Message"),
        direction=row.text("Direction"),
        inviter_profile_url=row.url("inviterProfileUrl"),
        invitee_profile_url=row.url("inviteeProfileUrl"),
    )


def _note(row: _Row) -> Note:
    return Note(
        first_name=row.text("Connection First Name"),
        last_name=row.text("Connection Last Name"),
        profile_url=row.url("Connection Profile URL"),
        note=row.text("Note"),
        created_at=row.datetime("Created On"),
        edited_at=row.datetime("Edited On"),
    )


def _company_follow(row: _Row) -> CompanyFollow:
    return CompanyFollow(
        organization=row.text("Organization"),
        followed_at=row.datetime("Followed On"),
    )


def _education(row: _Row) -> Education:
    return Education(
        school_name=row.text("School Name"),
        started_on=row.date("Start Date"),
        ended_on=row.date("End Date"),
        notes=row.text("Notes"),
        degree_name=row.text("Degree Name"),
        activities=row.text("Activities"),
    )


def _job_application(row: _Row) -> JobApplication:
    return JobApplication(
        applied_at=row.datetime("Application Date"),
        contact_email=row.text("Contact Email"),
        contact_phone_number=row.text("Contact Phone Number"),
        company_name=row.text("Company Name"),
        job_title=row.text("Job Title"),
        job_url=row.text("Job Url"),
        resume_name=row.text("Resume Name"),
        question_and_answers=row.text("Question And Answers"),
    )


def _saved_job(row: _Row) -> SavedJob:
    return SavedJob(
        saved_at=row.datetime("Saved Date"),
        job_url=row.text("Job Url"),
        job_title=row.text("Job Title"),
        company_name=row.text("Company Name"),
    )


def _message(row: _Row) -> Message:
    return Message(
        conversation_id=row.text("CONVERSATION ID"),
        conversation_title=row.text("CONVERSATION TITLE"),
        from_name=row.text("FROM"),
        sender_profile_url=row.url("SENDER PROFILE URL"),
        to_names=row.text("TO"),
        recipient_profile_urls=row.urls("RECIPIENT PROFILE URLS"),
        sent_at=row.datetime("DATE"),
        subject=row.text("SUBJECT"),
        content=row.text("CONTENT"),
        folder=row.text("FOLDER"),
    )


_RECOMMENDATION_HEADER = _columns(
    "First Name,Last Name,Company,Job Title,Text,Creation Date,Status"
)

# The ingest allowlist. Nothing outside this tuple is ever opened.
_SPECS: tuple[_FileSpec, ...] = (
    _FileSpec(
        "connections",
        "Connections.csv",
        (_columns("First Name,Last Name,URL,Email Address,Company,Position,Connected On"),),
        _connection,
    ),
    _FileSpec(
        "positions",
        "Positions.csv",
        (_columns("Company Name,Title,Description,Location,Started On,Finished On"),),
        _position,
    ),
    _FileSpec(
        "profile",
        "Profile.csv",
        (
            _columns(
                "First Name,Last Name,Maiden Name,Address,Birth Date,Headline,Summary,Industry,"
                "Zip Code,Geo Location,Twitter Handles,Websites,Instant Messengers"
            ),
        ),
        _profile,
    ),
    _FileSpec(
        "profile_summary",
        "Profile Summary.csv",
        (("Profile Summary",),),
        lambda row: ProfileSummary(text=row.text("Profile Summary")),
    ),
    _FileSpec("skills", "Skills.csv", (("Name",),), lambda row: Skill(name=row.text("Name"))),
    _FileSpec(
        "endorsements_received",
        "Endorsement_Received_Info.csv",
        (
            _columns(
                "Endorsement Date,Skill Name,Endorser First Name,Endorser Last Name,"
                "Endorser Public Url,Endorsement Status"
            ),
        ),
        _endorsement_received,
    ),
    _FileSpec(
        "endorsements_given",
        "Endorsement_Given_Info.csv",
        (
            _columns(
                "Endorsement Date,Skill Name,Endorsee First Name,Endorsee Last Name,"
                "Endorsee Public Url,Endorsement Status"
            ),
        ),
        _endorsement_given,
    ),
    _FileSpec(
        "recommendations_received",
        "Recommendations_Received.csv",
        (_RECOMMENDATION_HEADER,),
        _recommendation,
    ),
    _FileSpec(
        "recommendations_given",
        "Recommendations_Given.csv",
        (_RECOMMENDATION_HEADER,),
        _recommendation,
    ),
    _FileSpec(
        "invitations",
        "Invitations.csv",
        (_columns("From,To,Sent At,Message,Direction,inviterProfileUrl,inviteeProfileUrl"),),
        _invitation,
    ),
    _FileSpec(
        "notes",
        "Notes.csv",
        (
            _columns(
                "Connection First Name,Connection Last Name,Connection Profile URL,Note,"
                "Created On,Edited On"
            ),
        ),
        _note,
    ),
    _FileSpec(
        "company_follows",
        "Company Follows.csv",
        (("Organization", "Followed On"),),
        _company_follow,
    ),
    _FileSpec(
        "education",
        "Education.csv",
        (_columns("School Name,Start Date,End Date,Notes,Degree Name,Activities"),),
        _education,
    ),
    _FileSpec(
        "job_applications",
        "Job Applications.csv",
        (
            _columns(
                "Application Date,Contact Email,Contact Phone Number,Company Name,Job Title,"
                "Job Url,Resume Name,Question And Answers"
            ),
        ),
        _job_application,
    ),
    _FileSpec(
        "saved_jobs",
        "Saved Jobs.csv",
        (_columns("Saved Date,Job Url,Job Title,Company Name"),),
        _saved_job,
    ),
    # Not in the sampled export, so the header is unconfirmed: accept the
    # current layout and the older one without the two profile-URL columns.
    _FileSpec(
        "messages",
        "messages.csv",
        (
            _columns(
                "CONVERSATION ID,CONVERSATION TITLE,FROM,SENDER PROFILE URL,TO,"
                "RECIPIENT PROFILE URLS,DATE,SUBJECT,CONTENT,FOLDER"
            ),
            _columns("CONVERSATION ID,CONVERSATION TITLE,FROM,TO,DATE,SUBJECT,CONTENT,FOLDER"),
        ),
        _message,
    ),
)

_MATCHER = Matcher({spec.key: spec.filename for spec in _SPECS})


def _match_header(cells: list[str], spec: _FileSpec) -> tuple[dict[str, int], int] | None:
    """Return ``(column index, extra column count)`` if ``cells`` is the header."""
    found = [cell.strip().lower() for cell in cells]
    while found and not found[-1]:  # trailing-comma quirk: empty column names
        found.pop()
    variants = [tuple(name.lower() for name in header) for header in spec.headers]
    for exact in (True, False):
        for expected in sorted(variants, key=len, reverse=True):
            width = len(expected)
            if tuple(found[:width]) != expected or (exact and len(found) != width):
                continue
            index: dict[str, int] = {}
            for position, name in enumerate(expected):
                index.setdefault(name, position)  # repeated names: first one wins
            return index, len(found) - width
    return None


def _parse_file(spec: _FileSpec, name: str, data: bytes, warnings: list[str]) -> list[Any]:
    text = data.decode("utf-8-sig", errors="replace")
    if "�" in text:
        warnings.append(f"{name}: not valid UTF-8; undecodable bytes were replaced")
    csv.field_size_limit(max(csv.field_size_limit(), _FIELD_SIZE_LIMIT))
    reader = csv.reader(io.StringIO(text, newline=""))
    issues = _Issues()
    records: list[Any] = []
    index: dict[str, int] | None = None
    try:
        for number, cells in enumerate(reader):
            if index is None:
                matched = _match_header(cells, spec)
                if matched is not None:
                    index, extra = matched
                    if extra:
                        warnings.append(f"{name}: {extra} unexpected extra column(s) ignored")
                elif number + 1 >= _HEADER_SEARCH_ROWS:
                    break
                continue
            if not any(cell.strip() for cell in cells):
                continue
            records.append(spec.build(_Row(index, cells, reader.line_num, issues)))
    except csv.Error as error:
        warnings.append(f"{name}: malformed CSV at line {reader.line_num} ({error}); rest skipped")
    if index is None:
        warnings.append(f"{name}: expected header not found; treated as empty")
    warnings.extend(issues.warnings(name))
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _newest(entries: list[Entry]) -> datetime | None:
    stamps = [entry.modified for entry in entries if entry.modified is not None]
    return max(stamps) if stamps else None


def parse_export(path: Path, *, exported_at: datetime | None = None) -> ParsedExport:
    """Parse a LinkedIn export zip or extracted directory.

    Only allowlisted files are opened. ``exported_at`` overrides the derived
    export time (newest zip member timestamp, or newest allowlisted file mtime
    for a directory); a naive override is taken as UTC.

    Raises :class:`ExportError` if ``path`` is neither a zip nor a directory, or
    if it has no ``Connections.csv``.
    """
    path = Path(path)
    warnings: list[str] = []
    parsed: dict[str, list[Any]] = {spec.key: [] for spec in _SPECS}

    with ExitStack() as stack:
        is_zip = False
        if path.is_dir():
            entries = directory_entries(path)
        elif path.is_file() and zipfile.is_zipfile(path):
            is_zip = True
            try:
                entries = zip_entries(stack.enter_context(zipfile.ZipFile(path)))
            except (zipfile.BadZipFile, OSError) as error:
                raise ExportError(f"{path}: cannot read zip archive ({error})") from error
        else:
            raise ExportError(f"{path}: not a LinkedIn export zip or directory")

        selected = select_entries(entries, _MATCHER, warnings)
        if "connections" not in selected:
            raise ExportError(f"{path}: Connections.csv not found; not a LinkedIn export")

        hashed: list[tuple[str, str]] = []
        for spec in _SPECS:
            for entry in selected.get(spec.key, []):
                try:
                    data = entry.read()
                except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError) as error:
                    if spec.key == "connections":
                        raise ExportError(f"{path}: cannot read {entry.name} ({error})") from error
                    warnings.append(f"{entry.name}: cannot be read ({error}); treated as empty")
                    continue
                hashed.append((entry.name.lower(), hashlib.sha256(data).hexdigest()))
                parsed[spec.key].extend(_parse_file(spec, entry.name, data, warnings))

    if is_zip:
        source_sha256 = _sha256_file(path)
        derived = _newest(entries)
    else:
        digest = hashlib.sha256()
        for name, content_hash in sorted(hashed):
            digest.update(f"{name}\0{content_hash}\n".encode())
        source_sha256 = digest.hexdigest()
        derived = _newest([entry for group in selected.values() for entry in group])
    if exported_at is not None:
        derived = _to_utc(exported_at)
    elif derived is None:
        derived = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)

    profiles = parsed.pop("profile")
    if len(profiles) > 1:
        warnings.append(f"Profile.csv: {len(profiles)} rows found; only the first is kept")
    profile = profiles[0] if profiles else None
    file_counts = {key: len(records) for key, records in parsed.items()}
    file_counts["profile"] = 1 if profile else 0
    return ParsedExport(
        source_sha256=source_sha256,
        exported_at=derived,
        file_counts={spec.key: file_counts[spec.key] for spec in _SPECS},
        warnings=warnings,
        profile=profile,
        **parsed,
    )
