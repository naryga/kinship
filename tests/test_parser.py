"""Tests for ``kinship.parser.parse_export`` against handwritten fake exports.

Every person, company and URL below is invented. Zips are built at runtime.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import os
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from kinship.parser import (
    CompanyFollow,
    Connection,
    Education,
    EndorsementGiven,
    EndorsementReceived,
    ExportError,
    Invitation,
    JobApplication,
    Message,
    Note,
    Position,
    Profile,
    ProfileSummary,
    Recommendation,
    SavedJob,
    Skill,
    parse_export,
)

ADA = "https://www.linkedin.com/in/ada-example"
BO = "https://www.linkedin.com/in/bo-sample"
ME = "https://www.linkedin.com/in/owner-fake"

CONNECTIONS = (
    "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"
    "Ada,Example,https://www.linkedin.com/in/Ada-Example/,ada@example.test,"
    '"Fakeco, Inc.",Engineer,15 Jan 2024\n'
    "Bo,Sample,http://linkedin.com/in/bo-sample?trk=x,,Sample Labs,  Designer  ,03 Feb 2023\n"
    ",,,,,,\n"
    "Cy,Nolink,,,,,\n"
)

FILES: dict[str, str] = {
    "Connections.csv": CONNECTIONS,
    "Positions.csv": (
        "Company Name,Title,Description,Location,Started On,Finished On\n"
        'Fakeco,Staff Engineer,"Built things,\nacross two lines",Springfield,Jan 2020,\n'
        "Oldco,Engineer,,,2015,Dec 2019\n"
    ),
    "Profile.csv": (
        "First Name,Last Name,Maiden Name,Address,Birth Date,Headline,Summary,Industry,"
        "Zip Code,Geo Location,Twitter Handles,Websites,Instant Messengers\n"
        "Owner,Fake,,1 Secret Street,Jan 1,Builder of fake things,I make test data.,"
        "Software,00000,Nowhere,[@secret],[secret.test],\n"
    ),
    "Profile Summary.csv": 'Profile Summary\n"A longer, fake summary."\n',
    "Skills.csv": "Name\nPython\nSQL\n",
    "Endorsement_Received_Info.csv": (
        "Endorsement Date,Skill Name,Endorser First Name,Endorser Last Name,"
        "Endorser Public Url,Endorsement Status\n"
        "2024/01/15 14:23:11 UTC,Python,Ada,Example,www.linkedin.com/in/ada-example,ACCEPTED\n"
    ),
    "Endorsement_Given_Info.csv": (
        "Endorsement Date,Skill Name,Endorsee First Name,Endorsee Last Name,"
        "Endorsee Public Url,Endorsement Status\n"
        "2023/06/01 08:00:00 UTC,Design,Bo,Sample,https://www.linkedin.com/in/bo-sample/,ACCEPTED\n"
    ),
    "Recommendations_Received.csv": (
        "First Name,Last Name,Company,Job Title,Text,Creation Date,Status\n"
        'Ada,Example,Fakeco,Engineer,"Great, truly.","01/15/24, 02:23 PM",VISIBLE\n'
    ),
    "Recommendations_Given.csv": (
        "First Name,Last Name,Company,Job Title,Text,Creation Date,Status\n"
        'Bo,Sample,Sample Labs,Designer,Wonderful.,"1/5/24, 9:05 AM",VISIBLE\n'
    ),
    "Invitations.csv": (
        "From,To,Sent At,Message,Direction,inviterProfileUrl,inviteeProfileUrl\n"
        'Ada Example,Owner Fake,"1/15/24, 2:23 PM",Hello!,INCOMING,'
        "https://www.linkedin.com/in/ADA-example,https://www.linkedin.com/in/owner-fake\n"
    ),
    "Notes.csv": (
        "Connection First Name,Connection Last Name,Connection Profile URL,Note,"
        "Created On,Edited On\n"
        "Ada,Example,https://www.linkedin.com/in/ada-example,Met at a fake conference,"
        "2024-01-15 14:23:11 UTC,\n"
    ),
    "Company Follows.csv": "Organization,Followed On\nFakeco,Mon Jan 15 14:23:11 UTC 2024\n",
    "Education.csv": (
        "School Name,Start Date,End Date,Notes,Degree Name,Activities\n"
        "Fake University,2008,2012,,BSc Pretend Studies,Chess club\n"
    ),
    "Jobs/Job Applications.csv": (
        "Application Date,Contact Email,Contact Phone Number,Company Name,Job Title,Job Url,"
        "Resume Name,Question And Answers\n"
        '"1/15/24, 2:23 PM",owner@example.test,,Fakeco,Engineer,'
        "http://www.linkedin.com/jobs/view/1,resume.pdf,Q: Why? | A: Because.\n"
    ),
    "Jobs/Saved Jobs.csv": (
        "Saved Date,Job Url,Job Title,Company Name\n"
        '"1/15/24, 2:23 PM",http://www.linkedin.com/jobs/view/2,First Job,Fakeco\n'
    ),
    "Jobs/Saved Jobs_1.csv": (
        "Saved Date,Job Url,Job Title,Company Name\n"
        '"1/16/24, 9:00 AM",http://www.linkedin.com/jobs/view/3,Second Job,Sample Labs\n'
    ),
    "messages.csv": (
        "CONVERSATION ID,CONVERSATION TITLE,FROM,SENDER PROFILE URL,TO,RECIPIENT PROFILE URLS,"
        "DATE,SUBJECT,CONTENT,FOLDER\n"
        "conv-1,,Ada Example,https://www.linkedin.com/in/ada-example,Owner Fake,"
        "https://www.linkedin.com/in/owner-fake;https://www.linkedin.com/in/Bo-Sample/,"
        '2024-01-15 14:23:11 UTC,Hi,"Hello, there",INBOX\n'
    ),
}

AT = datetime(2024, 1, 15, 14, 23, 11, tzinfo=UTC)
AT_MINUTE = datetime(2024, 1, 15, 14, 23, tzinfo=UTC)


def write_dir(root: Path, files: dict[str, str | bytes]) -> Path:
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode())
    return root


def write_zip(
    path: Path,
    files: dict[str, str | bytes],
    *,
    prefix: str = "",
    when: tuple[int, int, int, int, int, int] = (2024, 3, 1, 12, 0, 0),
) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(zipfile.ZipInfo(prefix + name, date_time=when), content)
    return path


def check_full_export(parsed) -> None:
    assert parsed.warnings == []
    assert parsed.connections == [
        Connection("Ada", "Example", ADA, "ada@example.test", "Fakeco, Inc.", "Engineer",
                   date(2024, 1, 15)),
        Connection("Bo", "Sample", BO, None, "Sample Labs", "Designer", date(2023, 2, 3)),
        Connection("Cy", "Nolink", None, None, None, None, None),
    ]  # fmt: skip
    assert parsed.positions == [
        Position("Fakeco", "Staff Engineer", "Built things,\nacross two lines", "Springfield",
                 date(2020, 1, 1), None),
        Position("Oldco", "Engineer", None, None, date(2015, 1, 1), date(2019, 12, 1)),
    ]  # fmt: skip
    assert parsed.profile == Profile("Builder of fake things", "I make test data.", "Software")
    assert parsed.profile_summary == [ProfileSummary("A longer, fake summary.")]
    assert parsed.skills == [Skill("Python"), Skill("SQL")]
    assert parsed.endorsements_received == [
        EndorsementReceived(AT, "Python", "Ada", "Example", ADA, "ACCEPTED")
    ]
    assert parsed.endorsements_given == [
        EndorsementGiven(
            datetime(2023, 6, 1, 8, tzinfo=UTC), "Design", "Bo", "Sample", BO, "ACCEPTED"
        )
    ]
    assert parsed.recommendations_received == [
        Recommendation("Ada", "Example", "Fakeco", "Engineer", "Great, truly.", AT_MINUTE,
                       "VISIBLE")
    ]  # fmt: skip
    assert parsed.recommendations_given == [
        Recommendation("Bo", "Sample", "Sample Labs", "Designer", "Wonderful.",
                       datetime(2024, 1, 5, 9, 5, tzinfo=UTC), "VISIBLE")
    ]  # fmt: skip
    assert parsed.invitations == [
        Invitation("Ada Example", "Owner Fake", AT_MINUTE, "Hello!", "INCOMING", ADA, ME)
    ]
    assert parsed.notes == [Note("Ada", "Example", ADA, "Met at a fake conference", AT, None)]
    assert parsed.company_follows == [CompanyFollow("Fakeco", AT)]
    assert parsed.education == [
        Education("Fake University", date(2008, 1, 1), date(2012, 1, 1), None,
                  "BSc Pretend Studies", "Chess club")
    ]  # fmt: skip
    assert parsed.job_applications == [
        JobApplication(AT_MINUTE, "owner@example.test", None, "Fakeco", "Engineer",
                       "http://www.linkedin.com/jobs/view/1", "resume.pdf",
                       "Q: Why? | A: Because.")
    ]  # fmt: skip
    assert parsed.saved_jobs == [
        SavedJob(AT_MINUTE, "http://www.linkedin.com/jobs/view/2", "First Job", "Fakeco"),
        SavedJob(datetime(2024, 1, 16, 9, tzinfo=UTC), "http://www.linkedin.com/jobs/view/3",
                 "Second Job", "Sample Labs"),
    ]  # fmt: skip
    assert parsed.messages == [
        Message("conv-1", None, "Ada Example", ADA, "Owner Fake", (ME, BO), AT, "Hi",
                "Hello, there", "INBOX")
    ]  # fmt: skip
    assert parsed.file_counts == {
        "connections": 3,
        "positions": 2,
        "profile": 1,
        "profile_summary": 1,
        "skills": 2,
        "endorsements_received": 1,
        "endorsements_given": 1,
        "recommendations_received": 1,
        "recommendations_given": 1,
        "invitations": 1,
        "notes": 1,
        "company_follows": 1,
        "education": 1,
        "job_applications": 1,
        "saved_jobs": 2,
        "messages": 1,
    }


def test_parses_every_allowlisted_file_from_a_directory(tmp_path: Path) -> None:
    check_full_export(parse_export(write_dir(tmp_path, FILES)))


def test_parses_every_allowlisted_file_from_a_zip(tmp_path: Path) -> None:
    check_full_export(parse_export(write_zip(tmp_path / "export.zip", FILES)))


def test_zip_with_a_top_level_wrapper_folder(tmp_path: Path) -> None:
    archive = write_zip(tmp_path / "export.zip", FILES, prefix="Basic_LinkedInDataExport/")
    check_full_export(parse_export(archive))


def test_records_are_frozen(tmp_path: Path) -> None:
    parsed = parse_export(write_dir(tmp_path, {"Connections.csv": CONNECTIONS}))
    with pytest.raises(AttributeError):
        parsed.connections[0].company = "Other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        parsed.exported_at = datetime.now(UTC)  # type: ignore[misc]


# --- file discovery ---------------------------------------------------------


def test_files_are_found_case_insensitively_with_member_id_suffixes(tmp_path: Path) -> None:
    files = {
        "CONNECTIONS.CSV": CONNECTIONS,
        "skills_123456789.csv": "Name\nPython\n",
        "sub/Folder/company follows.CSV": "Organization,Followed On\nFakeco,2024-01-15\n",
        "JOBS/saved jobs_123456789_2.csv": FILES["Jobs/Saved Jobs_1.csv"],
        "JOBS/Saved Jobs_123456789.csv": FILES["Jobs/Saved Jobs.csv"],
    }
    for build in (write_dir(tmp_path / "d", files), write_zip(tmp_path / "e.zip", files)):
        parsed = parse_export(build)
        assert parsed.warnings == []
        assert len(parsed.connections) == 3
        assert parsed.skills == [Skill("Python")]
        assert parsed.company_follows == [
            CompanyFollow("Fakeco", datetime(2024, 1, 15, tzinfo=UTC))
        ]
        assert [job.job_title for job in parsed.saved_jobs] == ["First Job", "Second Job"]


def test_split_files_are_concatenated_in_numeric_order(tmp_path: Path) -> None:
    header = "Saved Date,Job Url,Job Title,Company Name\n"
    files = {"Connections.csv": CONNECTIONS, "Jobs/Saved Jobs.csv": header + ",,job 0,\n"}
    for part in (10, 2, 1):
        files[f"Jobs/Saved Jobs_{part}.csv"] = header + f",,job {part},\n"
    parsed = parse_export(write_dir(tmp_path, files))
    assert [job.job_title for job in parsed.saved_jobs] == ["job 0", "job 1", "job 2", "job 10"]
    assert parsed.file_counts["saved_jobs"] == 4


def test_lookalike_names_do_not_match_allowlisted_files(tmp_path: Path) -> None:
    files = {
        "Connections.csv": CONNECTIONS,
        # Would be misread as Profile.csv / Saved Jobs.csv by naive prefix matching.
        "Profile Summary.csv": "Profile Summary\nOnly a summary\n",
        "Jobs/Job Applicant Saved Answers.csv": "Question,Answer\nsecret,secret\n",
        "Jobs/Job Applicant Saved Screening Question Responses.csv": "x\nsecret\n",
        "SavedJobAlerts.csv": "x\nsecret\n",
        "Skills_old.csv": "Name\nsecret\n",
        "Connections.csv.bak": "secret",
        "Connections.txt": "secret",
    }
    parsed = parse_export(write_dir(tmp_path, files))
    assert parsed.warnings == []
    assert parsed.profile is None
    assert parsed.profile_summary == [ProfileSummary("Only a summary")]
    assert parsed.saved_jobs == []
    assert parsed.skills == []
    assert parsed.file_counts["profile"] == 0


def test_duplicate_file_in_a_deeper_folder_is_ignored_with_a_warning(tmp_path: Path) -> None:
    files = {"Connections.csv": CONNECTIONS, "old/Connections.csv": CONNECTIONS}
    parsed = parse_export(write_dir(tmp_path, files))
    assert len(parsed.connections) == 3
    assert parsed.warnings == ["old/Connections.csv: duplicate of Connections.csv; ignored"]


NEVER_INGESTED = [
    "Ad_Targeting.csv",
    "Ads Clicked.csv",
    "Email Addresses.csv",
    "Inferences_about_you.csv",
    "Logins.csv",
    "PhoneNumbers.csv",
    "Private_identity_asset.csv",
    "Receipts_v2.csv",
    "Registration.csv",
    "Security Challenges.csv",
    "SearchQueries.csv",
    "Verifications/Verifications.csv",
    "Jobs/Job Seeker Preferences.csv",
    "Jobs/Job Applicant Saved Answers.csv",
    "Comments_123456789.csv",
    "Reactions_123456789.csv",
    "Shares_123456789.csv",
    "Member_Follows_123456789.csv",
    "Learning.csv",
    "Events.csv",
    "Rich_Media.csv",
    "Certifications.csv",
    "Languages.csv",
    "SavedJobAlerts.csv",
    "Articles/Articles/some-post.html",
]
UNDECODABLE = b"\xff\xfe\x00secret\x80\x81"


def _with_never_ingested() -> dict[str, str | bytes]:
    files: dict[str, str | bytes] = dict(FILES)
    files.update(dict.fromkeys(NEVER_INGESTED, UNDECODABLE))
    return files


def _is_allowlisted(name: str) -> bool:
    return name.replace(os.sep, "/") in FILES


def test_directory_never_opens_non_allowlisted_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_dir(tmp_path / "export", _with_never_ingested())
    opened: list[str] = []
    real_open = io.open

    def tracking_open(file, *args, **kwargs):
        if not isinstance(file, int):
            opened.append(os.fspath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", tracking_open)
    monkeypatch.setattr(builtins, "open", tracking_open)
    parsed = parse_export(root)
    monkeypatch.undo()

    check_full_export(parsed)
    inside = {os.path.relpath(name, root) for name in opened if str(root) in str(name)}
    assert inside == {name.replace("/", os.sep) for name in FILES}
    assert all(_is_allowlisted(name) for name in inside)


def test_zip_never_opens_non_allowlisted_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = write_zip(tmp_path / "export.zip", _with_never_ingested())
    opened: list[str] = []
    real_open = zipfile.ZipFile.open

    def tracking_open(self, name, *args, **kwargs):
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return real_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", tracking_open)
    parsed = parse_export(archive)
    monkeypatch.undo()

    check_full_export(parsed)
    assert set(opened) == set(FILES)


def test_directory_hash_ignores_non_allowlisted_files(tmp_path: Path) -> None:
    plain = parse_export(write_dir(tmp_path / "a", FILES))
    noisy = parse_export(write_dir(tmp_path / "b", _with_never_ingested()))
    assert plain.source_sha256 == noisy.source_sha256


# --- header handling --------------------------------------------------------

PREAMBLE = (
    "Notes:\n"
    '"When exporting your connection data, you may notice that some of the email '
    'addresses are missing. Learn more, if you like."\n'
    "\n"
)


def test_preamble_lines_before_the_header_are_skipped(tmp_path: Path) -> None:
    parsed = parse_export(write_dir(tmp_path, {"Connections.csv": PREAMBLE + CONNECTIONS}))
    assert parsed.warnings == []
    assert [c.first_name for c in parsed.connections] == ["Ada", "Bo", "Cy"]


def test_bom_crlf_and_trailing_comma_header(tmp_path: Path) -> None:
    content = (
        "﻿First Name, Last Name ,URL,Email Address,Company,Position,Connected On,,\r\n"
        "Ada,Example,https://www.linkedin.com/in/ada-example,,Fakeco,Engineer,15 Jan 2024,,\r\n"
    )
    parsed = parse_export(write_dir(tmp_path, {"Connections.csv": content.encode("utf-8")}))
    assert parsed.warnings == []
    assert parsed.connections == [
        Connection("Ada", "Example", ADA, None, "Fakeco", "Engineer", date(2024, 1, 15))
    ]


def test_extra_trailing_columns_are_tolerated_with_a_warning(tmp_path: Path) -> None:
    files = {
        "Connections.csv": CONNECTIONS,
        "Skills.csv": "Name,Proficiency\nPython,Expert\n",
    }
    parsed = parse_export(write_dir(tmp_path, files))
    assert parsed.skills == [Skill("Python")]
    assert parsed.warnings == ["Skills.csv: 1 unexpected extra column(s) ignored"]


def test_short_rows_are_padded_with_blanks(tmp_path: Path) -> None:
    content = "First Name,Last Name,URL,Email Address,Company,Position,Connected On\nAda,Example\n"
    parsed = parse_export(write_dir(tmp_path, {"Connections.csv": content}))
    assert parsed.connections == [Connection("Ada", "Example", None, None, None, None, None)]


@pytest.mark.parametrize(
    "content",
    [
        "No conversations found\n",
        "",
        "Some,Other,Header\n1,2,3\n",
        "filler\n" * 12
        + "CONVERSATION ID,CONVERSATION TITLE,FROM,TO,DATE,SUBJECT,CONTENT,FOLDER\nc,,a,b,,,,\n",
    ],
)
def test_file_without_its_expected_header_is_empty_with_a_warning(
    tmp_path: Path, content: str
) -> None:
    files = {"Connections.csv": CONNECTIONS, "messages.csv": content}
    parsed = parse_export(write_dir(tmp_path, files))
    assert parsed.messages == []
    assert parsed.file_counts["messages"] == 0
    assert parsed.warnings == ["messages.csv: expected header not found; treated as empty"]
    assert len(parsed.connections) == 3


def test_connections_with_a_wrong_header_is_empty_not_an_error(tmp_path: Path) -> None:
    parsed = parse_export(write_dir(tmp_path, {"Connections.csv": "Nothing to see here\n"}))
    assert parsed.connections == []
    assert parsed.warnings == ["Connections.csv: expected header not found; treated as empty"]


def test_older_messages_header_without_profile_urls(tmp_path: Path) -> None:
    files = {
        "Connections.csv": CONNECTIONS,
        "messages.csv": (
            "CONVERSATION ID,CONVERSATION TITLE,FROM,TO,DATE,SUBJECT,CONTENT,FOLDER\n"
            "conv-9,Chat,Bo Sample,Owner Fake,2023-02-03 10:00:00 UTC,,Old format,ARCHIVE\n"
        ),
    }
    parsed = parse_export(write_dir(tmp_path, files))
    assert parsed.warnings == []
    assert parsed.messages == [
        Message("conv-9", "Chat", "Bo Sample", None, "Owner Fake", (),
                datetime(2023, 2, 3, 10, tzinfo=UTC), None, "Old format", "ARCHIVE")
    ]  # fmt: skip


def test_profile_keeps_only_headline_summary_and_industry(tmp_path: Path) -> None:
    parsed = parse_export(write_dir(tmp_path, FILES))
    assert parsed.profile is not None
    assert parsed.profile.__slots__ == ("headline", "summary", "industry")
    assert "Secret" not in repr(parsed) and "secret" not in repr(parsed)


def test_unparseable_dates_become_none_with_one_warning_per_column(tmp_path: Path) -> None:
    content = (
        "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"
        "Ada,Example,,,,,someday\n"
        "Bo,Sample,,,,,15 Jan 2024\n"
        "Cy,Nolink,,,,,32 Jan 2024\n"
    )
    parsed = parse_export(write_dir(tmp_path, {"Connections.csv": content}))
    assert [c.connected_on for c in parsed.connections] == [None, date(2024, 1, 15), None]
    assert parsed.warnings == [
        "Connections.csv: 2 unparseable value(s) in column 'Connected On' "
        "(first at line 2); stored as blank"
    ]


# --- missing files and bad paths --------------------------------------------


def test_only_connections_is_required(tmp_path: Path) -> None:
    parsed = parse_export(write_dir(tmp_path, {"Connections.csv": CONNECTIONS}))
    assert parsed.warnings == []
    assert parsed.profile is None
    assert parsed.messages == []
    assert parsed.file_counts["connections"] == 3
    assert sum(parsed.file_counts.values()) == 3
    assert len(parsed.file_counts) == 16


def test_missing_connections_is_an_error(tmp_path: Path) -> None:
    files = {name: content for name, content in FILES.items() if name != "Connections.csv"}
    with pytest.raises(ExportError, match=r"Connections\.csv not found"):
        parse_export(write_dir(tmp_path / "d", files))
    with pytest.raises(ExportError, match=r"Connections\.csv not found"):
        parse_export(write_zip(tmp_path / "e.zip", files))


def test_path_that_is_neither_zip_nor_directory_is_an_error(tmp_path: Path) -> None:
    not_a_zip = tmp_path / "Connections.csv"
    not_a_zip.write_text(CONNECTIONS)
    with pytest.raises(ExportError):
        parse_export(not_a_zip)
    with pytest.raises(ExportError):
        parse_export(tmp_path / "does-not-exist.zip")


# --- source hash and exported_at --------------------------------------------


def test_zip_hash_is_the_sha256_of_the_zip_bytes(tmp_path: Path) -> None:
    archive = write_zip(tmp_path / "export.zip", FILES)
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert parse_export(archive).source_sha256 == expected


def test_directory_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    first = parse_export(write_dir(tmp_path / "a", FILES))
    again = parse_export(write_dir(tmp_path / "b", dict(reversed(FILES.items()))))
    changed = dict(FILES, **{"Skills.csv": "Name\nPython\nSQL\nRust\n"})
    other = parse_export(write_dir(tmp_path / "c", changed))
    assert len(first.source_sha256) == 64
    assert first.source_sha256 == again.source_sha256
    assert first.source_sha256 != other.source_sha256


def test_exported_at_from_newest_zip_member(tmp_path: Path) -> None:
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(zipfile.ZipInfo("Connections.csv", (2024, 3, 1, 12, 0, 0)), CONNECTIONS)
        archive.writestr(zipfile.ZipInfo("Skills.csv", (2024, 3, 2, 8, 30, 10)), "Name\n")
    assert parse_export(path).exported_at == datetime(2024, 3, 2, 8, 30, 10, tzinfo=UTC)


def test_exported_at_from_newest_allowlisted_file_mtime(tmp_path: Path) -> None:
    files = {"Connections.csv": CONNECTIONS, "Skills.csv": "Name\n", "Logins.csv": "secret"}
    root = write_dir(tmp_path, files)
    stamps = {"Connections.csv": 1_700_000_000, "Skills.csv": 1_700_086_400}
    for name, stamp in stamps.items():
        os.utime(root / name, (stamp, stamp))
    os.utime(root / "Logins.csv", (1_800_000_000, 1_800_000_000))  # newer, but not allowlisted
    parsed = parse_export(root)
    assert parsed.exported_at == datetime.fromtimestamp(1_700_086_400, tz=UTC)
    assert parsed.exported_at.tzinfo is UTC


def test_exported_at_override_wins_and_is_utc(tmp_path: Path) -> None:
    root = write_dir(tmp_path / "d", {"Connections.csv": CONNECTIONS})
    archive = write_zip(tmp_path / "e.zip", {"Connections.csv": CONNECTIONS})
    naive = datetime(2022, 5, 6, 7, 8, 9)
    for source in (root, archive):
        assert parse_export(source, exported_at=naive).exported_at == naive.replace(tzinfo=UTC)
    aware = datetime.fromisoformat("2022-05-06T09:08:09+02:00")
    parsed = parse_export(root, exported_at=aware)
    assert parsed.exported_at == aware
    assert parsed.exported_at.utcoffset() == UTC.utcoffset(None)
