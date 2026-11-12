"""Render the synthetic world as a LinkedIn data export (directory or zip).

Headers match ``docs/linkedin-export-format.md`` exactly. The real export mixes
date formats between files, and so does this one, on purpose:

| File                              | Column             | Format                         |
|-----------------------------------|--------------------|--------------------------------|
| ``Connections.csv``               | Connected On       | ``15 Jan 2024``                |
| ``Positions.csv``                 | Started/Finished On| ``Jan 2020`` or just ``2020``; |
|                                   |                    | Finished On blank if current   |
| ``Education.csv``                 | Start/End Date     | ``2020`` or ``Jan 2020``       |
| ``Endorsement_*_Info.csv``        | Endorsement Date   | ``2024/01/15 14:23:11 UTC``    |
| ``Recommendations_*.csv``         | Creation Date      | ``01/15/24, 02:23 PM``         |
| ``Invitations.csv``               | Sent At            | ``1/15/24, 2:23 PM``           |
| ``Notes.csv``                     | Created/Edited On  | ``2024-01-15 14:23:11 UTC``    |
| ``Company Follows.csv``           | Followed On        | ``Mon Jan 15 14:23:11 UTC 2024``|
| ``Jobs/Job Applications.csv``     | Application Date   | ``1/15/24, 2:23 PM``           |
| ``Jobs/Saved Jobs*.csv``          | Saved Date         | ``1/15/24, 2:23 PM``           |
| ``messages.csv``                  | DATE               | ``2024-01-15 14:23:11 UTC``    |

Other quirks reproduced:

- ``Connections.csv`` in version 1 starts with a three-line ``Notes:`` preamble
  (older exports); version 2 has none (newer exports).
- Most connections have a blank email, a few a blank URL, a couple no company.
- ``Company Follows.csv`` has a header (and rows) ending in a trailing comma.
- ``Jobs/Saved Jobs.csv`` is split, continuing in ``Jobs/Saved Jobs_1.csv``.
- With ``include_messages=False``, ``messages.csv`` holds only the placeholder
  sentence ``No conversations found``.
- The same person's profile URL is spelled differently outside
  ``Connections.csv`` (trailing slash, ``http://``, upper-case host, ``?trk=``).
- Activity files in the complete variant carry a ``_<memberid>`` suffix.
- Files outside the ingest allowlist are present so the parser can prove it
  ignores them.

Month and weekday names are spelled from tables here, not ``strftime``, so the
output does not depend on the locale.
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Literal

from kinship.synth.world import DEFAULT_CONNECTIONS, World, build_world

Variant = Literal["basic", "complete"]

MESSAGES_PLACEHOLDER = "No conversations found"

# Headers of the files Kinship ingests, as in docs/linkedin-export-format.md.
HEADERS: dict[str, tuple[str, ...]] = {
    "Connections.csv": (
        "First Name",
        "Last Name",
        "URL",
        "Email Address",
        "Company",
        "Position",
        "Connected On",
    ),
    "Positions.csv": (
        "Company Name",
        "Title",
        "Description",
        "Location",
        "Started On",
        "Finished On",
    ),
    "Profile.csv": (
        "First Name",
        "Last Name",
        "Maiden Name",
        "Address",
        "Birth Date",
        "Headline",
        "Summary",
        "Industry",
        "Zip Code",
        "Geo Location",
        "Twitter Handles",
        "Websites",
        "Instant Messengers",
    ),
    "Profile Summary.csv": ("Profile Summary",),
    "Skills.csv": ("Name",),
    "Endorsement_Received_Info.csv": (
        "Endorsement Date",
        "Skill Name",
        "Endorser First Name",
        "Endorser Last Name",
        "Endorser Public Url",
        "Endorsement Status",
    ),
    "Endorsement_Given_Info.csv": (
        "Endorsement Date",
        "Skill Name",
        "Endorsee First Name",
        "Endorsee Last Name",
        "Endorsee Public Url",
        "Endorsement Status",
    ),
    "Recommendations_Received.csv": (
        "First Name",
        "Last Name",
        "Company",
        "Job Title",
        "Text",
        "Creation Date",
        "Status",
    ),
    "Recommendations_Given.csv": (
        "First Name",
        "Last Name",
        "Company",
        "Job Title",
        "Text",
        "Creation Date",
        "Status",
    ),
    "Invitations.csv": (
        "From",
        "To",
        "Sent At",
        "Message",
        "Direction",
        "inviterProfileUrl",
        "inviteeProfileUrl",
    ),
    "Notes.csv": (
        "Connection First Name",
        "Connection Last Name",
        "Connection Profile URL",
        "Note",
        "Created On",
        "Edited On",
    ),
    "Company Follows.csv": ("Organization", "Followed On"),
    "Education.csv": (
        "School Name",
        "Start Date",
        "End Date",
        "Notes",
        "Degree Name",
        "Activities",
    ),
    "Jobs/Job Applications.csv": (
        "Application Date",
        "Contact Email",
        "Contact Phone Number",
        "Company Name",
        "Job Title",
        "Job Url",
        "Resume Name",
        "Question And Answers",
    ),
    "Jobs/Saved Jobs.csv": ("Saved Date", "Job Url", "Job Title", "Company Name"),
    # Not in the real sample; header chosen by Kinship until an archive confirms it.
    "messages.csv": (
        "CONVERSATION ID",
        "CONVERSATION TITLE",
        "FROM",
        "SENDER PROFILE URL",
        "TO",
        "RECIPIENT PROFILE URLS",
        "DATE",
        "SUBJECT",
        "CONTENT",
        "FOLDER",
    ),
}
HEADERS["Jobs/Saved Jobs_1.csv"] = HEADERS["Jobs/Saved Jobs.csv"]

# The one file whose header (and rows) end with a trailing comma.
TRAILING_COMMA_FILE = "Company Follows.csv"

CONNECTIONS_PREAMBLE = (
    "Notes:\n"
    '"When exporting your connection data, you may notice that some of the email '
    "addresses are missing. You will only see email addresses for connections who "
    "have allowed their connections to see or download their email address using "
    "this setting https://www.linkedin.com/psettings/privacy/email. You can learn "
    'more here https://www.linkedin.com/help/linkedin/answer/261"\n'
    "\n"
)

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _day_month_year(d: date) -> str:  # 15 Jan 2024
    return f"{d.day:02d} {_MONTHS[d.month - 1]} {d.year}"


def _month_year(d: date | None, year_only: bool = False) -> str:  # Jan 2020 / 2020 / blank
    if d is None:
        return ""
    return str(d.year) if year_only else f"{_MONTHS[d.month - 1]} {d.year}"


def _slash_utc(t: datetime) -> str:  # 2024/01/15 14:23:11 UTC
    return f"{t:%Y/%m/%d %H:%M:%S} UTC"


def _iso_utc(t: datetime | None) -> str:  # 2024-01-15 14:23:11 UTC
    return "" if t is None else f"{t:%Y-%m-%d %H:%M:%S} UTC"


def _us_padded(t: datetime) -> str:  # 01/15/24, 02:23 PM
    return f"{t:%m/%d/%y}, {(t.hour - 1) % 12 + 1:02d}:{t:%M} {'AM' if t.hour < 12 else 'PM'}"


def _us_short(t: datetime) -> str:  # 1/15/24, 2:23 PM
    return (
        f"{t.month}/{t.day}/{t:%y}, {(t.hour - 1) % 12 + 1}:{t:%M} {'AM' if t.hour < 12 else 'PM'}"
    )


def _unix_style(t: datetime) -> str:  # Mon Jan 15 14:23:11 UTC 2024
    return f"{_DAYS[t.weekday()]} {_MONTHS[t.month - 1]} {t.day:02d} {t:%H:%M:%S} UTC {t.year}"


def _vary_url(url: str, i: int) -> str:
    """Spell a canonical profile URL the way other files sometimes do."""
    if not url:
        return url
    style = i % 5
    if style == 1:
        return url + "/"
    if style == 2:
        return url.replace("https://", "http://", 1)
    if style == 3:
        return url.replace("www.linkedin.com", "WWW.LINKEDIN.COM", 1)
    if style == 4:
        return url + "?trk=public_profile"
    return url


def _csv(
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    preamble: str = "",
    trailing_comma: bool = False,
) -> bytes:
    buffer = io.StringIO()
    buffer.write(preamble)
    writer = csv.writer(buffer, lineterminator="\n")
    extra = [""] if trailing_comma else []
    writer.writerow([*header, *extra])
    for row in rows:
        writer.writerow([*row, *extra])
    return buffer.getvalue().encode("utf-8")


def render_export(
    world: World,
    *,
    version: int = 1,
    variant: Variant = "basic",
    include_messages: bool = True,
) -> dict[str, bytes]:
    """Return every file of the export as ``{archive member name: bytes}``."""
    if variant not in ("basic", "complete"):
        raise ValueError(f"variant must be 'basic' or 'complete', got {variant!r}")
    owner = world.owner
    files: dict[str, bytes] = {}

    def add(name: str, rows: Iterable[Sequence[str]], **options: object) -> None:
        files[name] = _csv(HEADERS[name], rows, **options)  # type: ignore[arg-type]

    add(
        "Connections.csv",
        (
            (
                c.first_name,
                c.last_name,
                c.url,
                c.email,
                c.company,
                c.position,
                _day_month_year(c.connected_on),
            )
            for c in world.connections(version)
        ),
        preamble=CONNECTIONS_PREAMBLE if version == 1 else "",
    )
    add(
        "Positions.csv",
        (
            (
                p.company,
                p.title,
                p.description,
                p.location,
                _month_year(p.started_on, p.started_year_only),
                _month_year(p.finished_on, p.finished_year_only),
            )
            for p in owner.positions
        ),
    )
    add(
        "Profile.csv",
        [
            (
                owner.first_name,
                owner.last_name,
                "",
                "1 Example Street, Faketown",
                "Jan 1",
                owner.headline,
                owner.summary,
                owner.industry,
                "00000",
                "Faketown, Example State",
                "",
                "[COMPANY:https://example.com/sam]",
                "",
            )
        ],
    )
    add("Profile Summary.csv", [(owner.summary,)])
    add("Skills.csv", ((skill,) for skill in owner.skills))
    for name, endorsements in (
        ("Endorsement_Received_Info.csv", world.endorsements_received),
        ("Endorsement_Given_Info.csv", world.endorsements_given),
    ):
        add(
            name,
            (
                (
                    _slash_utc(e.at),
                    e.skill,
                    e.first_name,
                    e.last_name,
                    _vary_url(e.url, i),
                    e.status,
                )
                for i, e in enumerate(world.visible(endorsements, version))
            ),
        )
    for name, recommendations in (
        ("Recommendations_Received.csv", world.recommendations_received),
        ("Recommendations_Given.csv", world.recommendations_given),
    ):
        add(
            name,
            (
                (
                    r.first_name,
                    r.last_name,
                    r.company,
                    r.job_title,
                    r.text,
                    _us_padded(r.at),
                    r.status,
                )
                for r in world.visible(recommendations, version)
            ),
        )
    add(
        "Invitations.csv",
        (
            (
                inv.from_name,
                inv.to_name,
                _us_short(inv.at),
                inv.message,
                inv.direction,
                _vary_url(inv.inviter_url, i + 1),
                _vary_url(inv.invitee_url, i + 3),
            )
            for i, inv in enumerate(world.visible(world.invitations, version))
        ),
    )
    add(
        "Notes.csv",
        (
            (
                n.first_name,
                n.last_name,
                _vary_url(n.url, i + 2),
                n.text,
                _iso_utc(n.at),
                _iso_utc(n.edited_at),
            )
            for i, n in enumerate(world.visible(world.notes, version))
        ),
    )
    add(
        TRAILING_COMMA_FILE,
        (
            (f.organization, _unix_style(f.at))
            for f in world.visible(world.company_follows, version)
        ),
        trailing_comma=True,
    )
    add(
        "Education.csv",
        ((e.school, e.start, e.end, e.notes, e.degree, e.activities) for e in owner.education),
    )

    if include_messages:
        add(
            "messages.csv",
            (
                (
                    m.conversation_id,
                    m.title,
                    m.from_name,
                    _vary_url(m.sender_url, i),
                    m.to_name,
                    _vary_url(m.recipient_urls, i + 2),
                    _iso_utc(m.at),
                    m.subject,
                    m.content,
                    m.folder,
                )
                for i, m in enumerate(world.visible(world.messages, version))
            ),
        )
    else:
        files["messages.csv"] = (MESSAGES_PLACEHOLDER + "\n").encode()

    files.update(_never_ingested(world, version, variant))

    if variant == "basic":
        add(
            "Jobs/Job Applications.csv",
            (
                (
                    _us_short(a.at),
                    owner.email,
                    "",
                    a.company,
                    a.job_title,
                    a.job_url,
                    "Sam_Sampleton_Resume.pdf",
                    "Are you legally authorized to work here?:Yes | Years of experience:10",
                )
                for a in world.visible(world.job_applications, version)
            ),
        )
        saved = [
            (_us_short(s.at), s.job_url, s.job_title, s.company)
            for s in world.visible(world.saved_jobs, version)
        ]
        add("Jobs/Saved Jobs.csv", saved[:3])
        add("Jobs/Saved Jobs_1.csv", saved[3:])

    return dict(sorted(files.items()))


def _never_ingested(world: World, version: int, variant: Variant) -> dict[str, bytes]:
    """Files outside the allowlist, with fake content the parser must ignore."""
    owner = world.owner
    stamp = _iso_utc(datetime.combine(world.exported_at(version), time(9, 30)))
    files = {
        "Ad_Targeting.csv": _csv(
            ("Member Age", "Company Names", "Job Titles", "Member Interests"),
            [("35-54", "Example Co", "Data Engineer", "Synthetic Data; Test Fixtures")],
        ),
        "Email Addresses.csv": _csv(
            ("Email Address", "Confirmed", "Primary", "Updated On"),
            [(owner.email, "Yes", "Yes", "1/1/20, 9:00 AM")],
        ),
        "PhoneNumbers.csv": _csv(
            ("Extension", "Number", "Type"),
            [("", "+1 555 0100", "Mobile")],
        ),
        "Registration.csv": _csv(
            ("Registered At", "Registration Ip", "Subscription Types"),
            [("1/1/10, 9:00 AM", "192.0.2.10", "")],
        ),
    }
    if variant == "basic":
        files["Jobs/Job Seeker Preferences.csv"] = _csv(
            ("Locations", "Industries", "Open To Recruiters", "Job Titles"),
            [("Faketown", "Hospital & Health Care", "Yes", "Data Engineer")],
        )
        files["Verifications/Verifications.csv"] = _csv(
            ("First name", "Last name", "Verification type", "Verified date"),
            [(owner.first_name, owner.last_name, "Workplace", "2024-01-15")],
        )
    else:
        suffix = world.member_id
        files[f"Comments_{suffix}.csv"] = _csv(
            ("Date", "Link", "Message"),
            [(stamp, "https://www.linkedin.com/feed/update/urn:li:activity:9000001", "Agreed.")],
        )
        files[f"Reactions_{suffix}.csv"] = _csv(
            ("Date", "Type", "Link"),
            [(stamp, "LIKE", "https://www.linkedin.com/feed/update/urn:li:activity:9000002")],
        )
        files[f"Shares_{suffix}.csv"] = _csv(
            ("Date", "ShareLink", "ShareCommentary", "SharedUrl", "MediaUrl", "Visibility"),
            [
                (
                    stamp,
                    "https://www.linkedin.com/feed/update/urn:li:share:9000003",
                    "Synthetic post about idempotent pipelines.",
                    "",
                    "",
                    "MEMBER_NETWORK",
                )
            ],
        )
        files["Logins.csv"] = _csv(
            ("Login Date", "IP Address", "User Agent", "Login Type"),
            [(stamp, "192.0.2.10", "ExampleBrowser/1.0", "Login")],
        )
        files["Languages.csv"] = _csv(("Name", "Proficiency"), [("Examplish", "Native")])
    return files


def archive_name(variant: Variant, exported_at: date) -> str:
    """File name LinkedIn would give the archive, e.g. ``Basic_LinkedInDataExport_...``."""
    return f"{variant.capitalize()}_LinkedInDataExport_{exported_at:%m-%d-%Y}.zip"


def generate_export(
    out: Path,
    *,
    seed: int = 42,
    version: int = 1,
    variant: Variant = "basic",
    n_connections: int = DEFAULT_CONNECTIONS,
    as_zip: bool = False,
    include_messages: bool = True,
    exported_at: date | None = None,
) -> Path:
    """Write a synthetic LinkedIn export and return its path.

    ``version=2`` is a later export of the same seeded world as version 1 (see
    ``kinship.synth.world``). ``exported_at`` only changes the timestamps stamped
    on the output; the content is fixed by ``seed``, ``version`` and
    ``n_connections``.

    As a directory, files are written under ``out``, which must be missing or
    empty. As a zip, ``out`` is the archive path if it ends in ``.zip``, else a
    directory that receives a LinkedIn-style archive name. Zip output is
    byte-for-byte deterministic: members are sorted, stored uncompressed and
    stamped with the export date.
    """
    out = Path(out)
    world = build_world(seed, n_connections)
    files = render_export(
        world, version=version, variant=variant, include_messages=include_messages
    )
    stamp = exported_at or world.exported_at(version)

    if as_zip:
        if out.suffix.lower() != ".zip":
            out = out / archive_name(variant, stamp)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
            for name, content in files.items():
                info = zipfile.ZipInfo(
                    name, date_time=(stamp.year, stamp.month, stamp.day, 12, 0, 0)
                )
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o644 << 16
                archive.writestr(info, content)
        return out

    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"{out} exists and is not an empty directory")
    mtime = datetime.combine(stamp, time(12), tzinfo=UTC).timestamp()
    for name, content in files.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        os.utime(path, (mtime, mtime))
    return out
