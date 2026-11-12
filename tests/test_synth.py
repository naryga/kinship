"""Tests for the synthetic LinkedIn export generator. No database needed."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kinship.cli import app
from kinship.synth import (
    HEADERS,
    MESSAGES_PLACEHOLDER,
    build_world,
    diff_truth,
    generate_export,
    render_export,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAT_DOC = ROOT / "docs" / "linkedin-export-format.md"
FIXTURES = Path(__file__).parent / "fixtures" / "synthetic"
SEED = 42


def _doc_headers() -> dict[str, list[str]]:
    """File name -> header, read from the allowlist table in the format doc."""
    headers = {}
    for line in FORMAT_DOC.read_text(encoding="utf-8").splitlines():
        cells = re.findall(r"`([^`]+)`", line)
        if line.startswith("| `") and len(cells) >= 2:
            headers[cells[0]] = cells[-1].split(",")
    return headers


def _rows(content: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(content.decode("utf-8"))))


def _connections(files: dict[str, bytes]) -> dict[str, dict[str, str]]:
    """Connections keyed by URL (blank-URL rows keyed by name)."""
    rows = _rows(files["Connections.csv"])
    start = next(i for i, row in enumerate(rows) if row and row[0] == "First Name")
    header = rows[start]
    records = [dict(zip(header, row, strict=True)) for row in rows[start + 1 :]]
    return {r["URL"] or f"{r['First Name']} {r['Last Name']}": r for r in records}


def _canonical(url: str) -> str:
    url = url.split("?")[0].rstrip("/")
    return re.sub(r"^https?://www\.linkedin\.com", "https://www.linkedin.com", url, flags=re.I)


@pytest.fixture(scope="module")
def v1() -> dict[str, bytes]:
    return render_export(build_world(SEED), version=1)


@pytest.fixture(scope="module")
def v2() -> dict[str, bytes]:
    return render_export(build_world(SEED), version=2)


def _tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --- headers ---------------------------------------------------------------------


def test_headers_match_format_doc(v1: dict[str, bytes]) -> None:
    doc = _doc_headers()
    assert len(doc) == 15
    for name, expected in doc.items():
        targets = ["Jobs/Saved Jobs.csv", "Jobs/Saved Jobs_1.csv"] if "*" in name else [name]
        for target in targets:
            assert list(HEADERS[target]) == expected, target
            if target == "Connections.csv":
                continue  # preamble comes first; checked in the quirk tests
            first_line = v1[target].decode().split("\n", 1)[0]
            # Company Follows.csv deliberately ends its header with a comma.
            assert first_line.rstrip(",").split(",") == expected, target


def test_every_data_row_matches_its_header_width(
    v1: dict[str, bytes], v2: dict[str, bytes]
) -> None:
    for files in (v1, v2):
        for name, content in files.items():
            rows = _rows(content)
            if name == "Connections.csv" and rows[0] == ["Notes:"]:
                rows = rows[3:]
            assert len(rows) >= 2, f"{name} has no data rows"
            assert {len(row) for row in rows} == {len(rows[0])}, name


def test_issue_file_coverage(v1: dict[str, bytes]) -> None:
    wanted = {
        "Connections.csv",
        "Positions.csv",
        "Profile.csv",
        "Profile Summary.csv",
        "Skills.csv",
        "messages.csv",
        "Endorsement_Received_Info.csv",
        "Endorsement_Given_Info.csv",
        "Recommendations_Received.csv",
        "Recommendations_Given.csv",
        "Invitations.csv",
        "Notes.csv",
        "Company Follows.csv",
        "Education.csv",
        "Jobs/Job Applications.csv",
        "Jobs/Saved Jobs.csv",
        "Jobs/Saved Jobs_1.csv",
    }
    assert wanted <= set(v1)
    # Files outside the allowlist exist so the parser can prove it skips them.
    assert {"Ad_Targeting.csv", "PhoneNumbers.csv", "Verifications/Verifications.csv"} <= set(v1)


def test_variants() -> None:
    world = build_world(SEED)
    basic = set(render_export(world, variant="basic"))
    complete = set(render_export(world, variant="complete"))
    assert any(name.startswith("Jobs/") for name in basic)
    assert any(name.startswith("Verifications/") for name in basic)
    assert not any("/" in name for name in complete)
    assert not any(name.startswith(("Comments_", "Reactions_", "Shares_")) for name in basic)
    member = world.member_id
    assert member.isdigit()
    assert {
        f"Comments_{member}.csv",
        f"Reactions_{member}.csv",
        f"Shares_{member}.csv",
        "Logins.csv",
    } <= complete
    with pytest.raises(ValueError):
        render_export(world, variant="full")  # type: ignore[arg-type]


# --- determinism -----------------------------------------------------------------


def test_same_seed_same_bytes_and_other_seed_differs() -> None:
    build_world.cache_clear()
    first = render_export(build_world(7), version=2, variant="complete")
    build_world.cache_clear()
    again = render_export(build_world(7), version=2, variant="complete")
    assert first == again
    assert render_export(build_world(8), version=2, variant="complete") != first


@pytest.mark.parametrize("version", [1, 2])
def test_committed_fixture_is_reproducible(version: int, tmp_path: Path) -> None:
    """Drift guard: regenerate with `kinship synth` if this fails on purpose."""
    out = generate_export(tmp_path / "export", seed=SEED, version=version)
    committed = _tree(FIXTURES / f"export_v{version}")
    assert committed, "fixture missing"
    assert _tree(out) == committed


def test_zip_is_deterministic_and_matches_directory(tmp_path: Path) -> None:
    directory = generate_export(tmp_path / "dir", seed=SEED, version=2)
    one = generate_export(tmp_path / "a", seed=SEED, version=2, as_zip=True)
    two = generate_export(tmp_path / "b" / "custom.zip", seed=SEED, version=2, as_zip=True)
    assert one.name == "Basic_LinkedInDataExport_11-02-2026.zip"
    assert two.name == "custom.zip"
    assert one.read_bytes() == two.read_bytes()
    with zipfile.ZipFile(one) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
        stamps = {info.date_time for info in archive.infolist()}
    assert members == _tree(directory)
    assert stamps == {(2026, 11, 2, 12, 0, 0)}


def test_exported_at_only_changes_the_stamp(tmp_path: Path) -> None:
    path = generate_export(tmp_path, seed=SEED, as_zip=True, exported_at=date(2024, 2, 3))
    assert path.name == "Basic_LinkedInDataExport_02-03-2024.zip"
    expected = render_export(build_world(SEED))["Connections.csv"]
    with zipfile.ZipFile(path) as archive:
        assert {info.date_time[:3] for info in archive.infolist()} == {(2024, 2, 3)}
        assert archive.read("Connections.csv") == expected


def test_refuses_non_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        generate_export(tmp_path)


def test_connection_count_is_validated() -> None:
    with pytest.raises(ValueError):
        build_world(SEED, 5)
    assert len(build_world(SEED, 120).connections(1)) == 120


# --- version 2 vs version 1 --------------------------------------------------------


@pytest.mark.parametrize(("seed", "n"), [(SEED, 40), (3, 20), (11, 150)])
def test_v2_differs_from_v1_exactly_as_truth_says(seed: int, n: int) -> None:
    world = build_world(seed, n)
    truth = diff_truth(seed, n)
    old = _connections(render_export(world, version=1))
    new = _connections(render_export(world, version=2))

    assert truth.v1_count == len(old) == n
    assert truth.v2_count == len(new) == n - len(truth.removed) + len(truth.added)
    assert truth.added and truth.removed and truth.company_changed and truth.title_changed
    assert set(new) - set(old) == set(truth.added)
    assert set(old) - set(new) == set(truth.removed)

    def job(record: dict[str, str]) -> tuple[str, str]:
        return record["Company"], record["Position"]

    both = set(old) & set(new)
    assert {u for u in both if job(old[u]) != job(new[u])} == set(truth.job_changed)
    assert {u for u in both if old[u]["Position"] != new[u]["Position"]} == set(truth.title_changed)
    company_differs = {u for u in both if old[u]["Company"] != new[u]["Company"]}
    assert company_differs == set(truth.company_changed) | set(truth.company_renamed)
    assert not set(truth.company_changed) & set(truth.company_renamed)
    assert sum(old[u] == new[u] for u in both) == truth.unchanged_count

    by_url_v1 = {c.url: c for c in world.connections(1)}
    by_url_v2 = {c.url: c for c in world.connections(2)}
    for url in truth.company_changed:
        assert by_url_v1[url].company_key != by_url_v2[url].company_key
    for url in truth.company_renamed:
        assert by_url_v1[url].company_key == by_url_v2[url].company_key
    # New connections connected after the first export.
    assert all(by_url_v2[u].connected_on > truth.v1_exported_at for u in truth.added)
    assert truth.v2_exported_at > truth.v1_exported_at


def test_v2_activity_files_only_grow(v1: dict[str, bytes], v2: dict[str, bytes]) -> None:
    for name in (
        "Invitations.csv",
        "messages.csv",
        "Endorsement_Received_Info.csv",
        "Endorsement_Given_Info.csv",
        "Recommendations_Received.csv",
        "Recommendations_Given.csv",
        "Jobs/Job Applications.csv",
        "Company Follows.csv",
    ):
        # Compare on columns that carry no per-row URL spelling.
        before = {(r[0], r[1], r[2]) for r in _rows(v1[name])}
        after = {(r[0], r[1], r[2]) for r in _rows(v2[name])}
        assert len(_rows(v2[name])) > len(_rows(v1[name])), name
        assert before <= after, name


# --- quirks ----------------------------------------------------------------------


def test_notes_preamble_only_in_version_1(v1: dict[str, bytes], v2: dict[str, bytes]) -> None:
    header = ",".join(HEADERS["Connections.csv"])
    lines = v1["Connections.csv"].decode().split("\n")
    assert lines[0] == "Notes:"
    assert lines[1].startswith('"When exporting your connection data')
    assert lines[2] == ""
    assert lines[3] == header
    assert v2["Connections.csv"].decode().split("\n")[0] == header


def test_blank_emails_urls_and_companies(v1: dict[str, bytes]) -> None:
    records = list(_connections(v1).values())
    blank_email = sum(not r["Email Address"] for r in records)
    assert len(records) / 2 < blank_email < len(records)
    assert all(r["Email Address"].endswith("@example.com") for r in records if r["Email Address"])
    assert sum(not r["URL"] for r in records) == diff_truth(SEED).blank_url_count == 2
    assert sum(not r["Company"] for r in records) == 2
    for r in records:
        assert not r["URL"] or re.fullmatch(
            r"https://www\.linkedin\.com/in/[a-z0-9-]+-[0-9a-f]{8}", r["URL"]
        )


def test_company_name_variants(v1: dict[str, bytes], v2: dict[str, bytes]) -> None:
    acme = {"Acme", "Acme, Inc.", "ACME Corp"}
    for files in (v1, v2):
        companies = {r["Company"] for r in _connections(files).values()}
        assert acme <= companies
        assert {"Gifthealth", "Gift Health, Inc.", "gift health"} <= companies
    world = build_world(SEED)
    assert {c.company for c in world.connections_at("acme", 1)} == acme
    for c in world.connections(1):
        assert c.company in world.company_groups.get(c.company_key, ("",))


def test_mixed_date_formats(v1: dict[str, bytes]) -> None:
    def column(name: str, index: int) -> list[str]:
        rows = _rows(v1[name])
        if rows[0] == ["Notes:"]:
            rows = rows[3:]
        return [row[index] for row in rows[1:]]

    def check(name: str, index: int, pattern: str, allow_blank: bool = False) -> None:
        values = column(name, index)
        assert values, name
        for value in values:
            if not (allow_blank and value == ""):
                assert re.fullmatch(pattern, value), (name, value)

    mon = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    short = r"\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M"
    check("Connections.csv", 6, rf"\d{{2}} {mon} \d{{4}}")
    check("Positions.csv", 4, rf"({mon} )?\d{{4}}")
    check("Positions.csv", 5, rf"({mon} )?\d{{4}}", allow_blank=True)
    check("Education.csv", 1, rf"({mon} )?\d{{4}}")
    check("Endorsement_Received_Info.csv", 0, r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} UTC")
    check("Endorsement_Given_Info.csv", 0, r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} UTC")
    check("Recommendations_Received.csv", 5, r"\d{2}/\d{2}/\d{2}, \d{2}:\d{2} [AP]M")
    check("Recommendations_Given.csv", 5, r"\d{2}/\d{2}/\d{2}, \d{2}:\d{2} [AP]M")
    check("Invitations.csv", 2, short)
    check("Notes.csv", 4, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC")
    check(
        "Company Follows.csv",
        1,
        rf"(Mon|Tue|Wed|Thu|Fri|Sat|Sun) {mon} \d{{2}} \d{{2}}:\d{{2}}:\d{{2}} UTC \d{{4}}",
    )
    check("Jobs/Job Applications.csv", 0, short)
    check("Jobs/Saved Jobs.csv", 0, short)
    check("messages.csv", 6, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC")

    started = column("Positions.csv", 4)
    assert any(re.fullmatch(r"\d{4}", v) for v in started)  # year only
    assert any(" " in v for v in started)  # month and year
    assert column("Positions.csv", 5).count("") == 1  # the current role


def test_trailing_comma_header_and_split_file(v1: dict[str, bytes]) -> None:
    follows = v1["Company Follows.csv"].decode().split("\n")
    assert follows[0] == "Organization,Followed On,"
    assert all(line.endswith(",") for line in follows if line)
    header = ",".join(HEADERS["Jobs/Saved Jobs.csv"])
    for name in ("Jobs/Saved Jobs.csv", "Jobs/Saved Jobs_1.csv"):
        assert v1[name].decode().split("\n")[0] == header
        assert len(_rows(v1[name])) > 1


def test_messages_placeholder(v1: dict[str, bytes]) -> None:
    files = render_export(build_world(SEED), include_messages=False)
    assert files["messages.csv"].decode().strip() == MESSAGES_PLACEHOLDER
    files.pop("messages.csv")
    assert files == {k: v for k, v in v1.items() if k != "messages.csv"}


def test_profile_urls_vary_but_canonicalize_to_connections(v1: dict[str, bytes]) -> None:
    world = build_world(SEED)
    known = {c.url for c in world.connections(1) if c.url} | {world.owner.url}
    seen: set[str] = set()
    for name, columns in (
        ("Endorsement_Received_Info.csv", [4]),
        ("Endorsement_Given_Info.csv", [4]),
        ("Notes.csv", [2]),
        ("messages.csv", [3, 5]),
    ):
        for row in _rows(v1[name])[1:]:
            for index in columns:
                seen.add(row[index])
                assert _canonical(row[index]) in known, (name, row[index])
    assert any(u.endswith("/") for u in seen)
    assert any(u.startswith("http://") for u in seen)
    assert any("WWW.LINKEDIN.COM" in u for u in seen)
    assert any("?trk=" in u for u in seen)
    assert any(u in known for u in seen)
    # Invitations also include people who never became connections.
    invitations = _rows(v1["Invitations.csv"])[1:]
    invitees = {_canonical(r[6]) for r in invitations}
    assert invitees & known
    assert invitees - known
    assert {r[4] for r in invitations} == {"INCOMING", "OUTGOING"}


def test_multiline_text_and_distinct_topics(v1: dict[str, bytes], v2: dict[str, bytes]) -> None:
    texts = [
        r[4]
        for name in ("Recommendations_Received.csv", "Recommendations_Given.csv")
        for r in _rows(v2[name])[1:]
    ]
    assert len(texts) == len(set(texts)) == 14
    assert any("\n" in t for t in texts)
    assert any("\n" in r[2] for r in _rows(v1["Positions.csv"])[1:])
    assert any("\n" in r[8] for r in _rows(v1["messages.csv"])[1:])


def test_world_supports_later_issues() -> None:
    world = build_world(SEED)
    owner_past = {p.company_key for p in world.owner.positions if p.finished_on}
    assert any(world.connections_at(key, 1) for key in owner_past)  # shared past employers
    with_path = [a for a in world.job_applications if a.company_key]
    assert len(with_path) >= 3
    assert all(world.connections_at(a.company_key, 1) for a in with_path)
    assert any(not a.company_key for a in world.job_applications)
    assert world.notes
    assert world.endorsements_given
    assert len(world.visible(world.messages, 1)) < len(world.visible(world.messages, 2))


def test_nothing_looks_real(v1: dict[str, bytes], v2: dict[str, bytes]) -> None:
    for files in (v1, v2):
        text = "\n".join(content.decode() for content in files.values())
        emails = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", text))
        assert emails
        assert all(e.endswith("@example.com") for e in emails)
        hosts = {h.lower() for h in re.findall(r"https?://([^/\s,\"\]]+)", text)}
        assert hosts <= {"www.linkedin.com", "example.com"}


# --- CLI -------------------------------------------------------------------------


def test_cli_synth(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "export"
    result = runner.invoke(app, ["synth", str(out)])
    assert result.exit_code == 0, result.output
    assert _tree(out) == _tree(FIXTURES / "export_v1")

    again = runner.invoke(app, ["synth", str(out)])
    assert again.exit_code == 1  # refuses to write into a non-empty directory

    zipped = tmp_path / "zipped"
    args = ["synth", str(zipped), "--seed", "7", "--version", "2", "--variant", "complete"]
    result = runner.invoke(app, [*args, "--zip", "--connections", "25", "--no-messages"])
    assert result.exit_code == 0, result.output
    expected = render_export(
        build_world(7, 25), version=2, variant="complete", include_messages=False
    )
    with zipfile.ZipFile(zipped / "Complete_LinkedInDataExport_11-02-2026.zip") as archive:
        assert {n: archive.read(n) for n in archive.namelist()} == expected
        assert archive.read("messages.csv").decode().strip() == MESSAGES_PLACEHOLDER

    assert runner.invoke(app, ["synth", str(tmp_path / "x"), "--variant", "full"]).exit_code != 0
    assert runner.invoke(app, ["synth", str(tmp_path / "y"), "--version", "3"]).exit_code != 0
