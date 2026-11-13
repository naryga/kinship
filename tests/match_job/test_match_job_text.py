"""The pure text side of `match_job`: stems, title inference, terms, title overlap."""

from __future__ import annotations

import pytest
from match_job_seeding import SAMPLE_COMPANY, SAMPLE_JOB, SAMPLE_TITLE

from kinship.rag import match
from kinship.rag.match import analyse, evidence, explain, infer_title, stem, stems, title_overlap


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("Pipelines", "pipeline"),
        ("pipeline", "pipeline"),
        ("claims", "claim"),
        ("mentoring", "mentor"),
        ("Mentor", "mentor"),
        ("rejected", "reject"),
        ("batches", "batch"),
        ("policies", "policy"),
        ("engineering", "engineer"),
        ("analysis", "analysis"),
        ("SQL", "sql"),
        ("data", "data"),
        ("class", "class"),
    ],
)
def test_stem(word: str, expected: str) -> None:
    assert stem(word) == expected


@pytest.mark.parametrize(
    ("text", "title", "company"),
    [
        (SAMPLE_JOB, SAMPLE_TITLE, None),
        ("Job Title: Staff Platform Engineer\nCompany: Acme, Inc.\n\nWe build.", None, None),
        ("Senior Data Engineer at Gift Health\n\nAbout us ...", SAMPLE_TITLE, "Gift Health"),
        ("Acme | Product Manager (Remote)\nWe make anvils.", "Product Manager", None),
        ("# Hiring: Security Engineer\n", "Security Engineer", None),
        ("**Role:** UX Researcher @ Halcyon Grid", "UX Researcher", "Halcyon Grid"),
        (
            "About us\nWe are a small studio.\n\nWe are looking for a Technical Writer to "
            "own our docs.",
            "Technical Writer",
            None,
        ),
        ("Role-based access control is what we sell.\nJoin us.", None, None),
        ("Position - Firmware Engineer\nEmployer - Norwhal Labs", "Firmware Engineer", None),
        ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod.", None, None),
        ("", None, None),
    ],
    ids=range(11),
)
def test_infer_title(text: str, title: str | None, company: str | None) -> None:
    found_title, found_company = infer_title(text)
    if text.startswith("Job Title:"):
        assert (found_title, found_company) == ("Staff Platform Engineer", "Acme, Inc.")
    elif text.startswith("Position -"):
        assert (found_title, found_company) == ("Firmware Engineer", "Norwhal Labs")
    else:
        assert (found_title, found_company) == (title, company)


def test_analyse_builds_focused_and_section_queries() -> None:
    analysis = analyse(SAMPLE_JOB, company=SAMPLE_COMPANY)

    assert analysis.title == SAMPLE_TITLE
    assert not analysis.truncated
    focused = analysis.focused_query.split()
    assert focused[:2] == ["data", "engineer"]  # the title leads
    assert len(focused) <= match.MAX_FOCUSED_TERMS
    for expected in ("claims", "pipelines", "airflow", "billing", "warehouse", "dbt"):
        assert expected in focused
    # No stop words, boilerplate, seniority words or the company's own name.
    for unwanted in ("the", "and", "experience", "senior", "tessellate", "health", "looking"):
        assert unwanted not in focused
        assert all(unwanted not in query.split() for query in analysis.section_queries)

    assert 1 <= len(analysis.section_queries) <= match.MAX_SECTIONS
    everything = " ".join(analysis.section_queries)
    assert "kafka" in everything and "runbook" in everything
    # The benefits paragraph and the heading block get no query of their own.
    assert "holiday" not in everything and "faketown" not in everything
    for query in analysis.section_queries:
        assert len(query.split()) <= match.MAX_SECTION_TERMS

    by_key = {term.key: term for term in analysis.terms}
    assert by_key["claim"].count >= 4 and by_key["claim"].text == "claims"
    assert by_key["snowflake"].proper and by_key["kafka"].proper
    assert not by_key["pipeline"].proper
    assert {"data pipeline", "healthcare bill", "apache airflow"} <= {
        phrase.key for phrase in analysis.phrases
    }


def test_analyse_title_override_and_truncation() -> None:
    analysis = analyse("x" * 5 + " pipelines " * 5000, title="  Head   of Data ")
    assert analysis.title == "Head of Data"
    assert analysis.truncated
    assert len(analysis.text) == match.MAX_DESCRIPTION_CHARS
    assert analysis.focused_query.split()[0] == "data"

    empty = analyse("?!... --- \x00 ///")
    assert empty.focused_query == "" and empty.section_queries == () and empty.title is None


def test_evidence_lists_only_terms_in_both_texts() -> None:
    analysis = analyse(SAMPLE_JOB, company=SAMPLE_COMPANY)
    chunk = (
        "Position: Data Engineer at Acme\nBuilt Airflow batch pipelines and dbt models "
        "feeding the analytics warehouse. Mentored two engineers on data quality checks."
    )
    found = evidence(analysis, chunk)

    assert 1 <= len(found) <= match.MAX_EVIDENCE_TERMS
    assert found[0] == "data engineer"
    assert "data quality" in found or "quality checks" in found
    job_stems, chunk_stems = f" {' '.join(stems(SAMPLE_JOB))} ", f" {' '.join(stems(chunk))} "
    for term in found:
        key = f" {' '.join(stems(term))} "
        assert key in job_stems and key in chunk_stems, term
    # "data" and "engineer" are not repeated next to the phrase that holds them.
    assert "data" not in found and "engineer" not in found

    assert evidence(analysis, "Nothing in common whatsoever: kites, zeppelins.") == ()
    assert explain(found[:2], ("fts", "vector")).startswith("matched on: data engineer, ")
    assert explain(found[:2], ("fts", "vector")).endswith("; full-text + vector")
    assert explain(["dbt"], ("fts",)) == "matched on: dbt; full-text"
    assert "no keyword in common" in explain((), ("vector",))


@pytest.mark.parametrize(
    ("role", "other", "shared", "similar"),
    [
        ("Senior Data Engineer", "Data Engineer", ("data", "engineer"), True),
        ("Senior Data Engineer", "Staff Data Engineer II", ("data", "engineer"), True),
        ("Senior Data Engineer", "Data Scientist", ("data",), True),
        ("Senior Data Engineer", "Firmware Engineer", ("engineer",), False),
        ("Senior Data Engineer", "Senior Solutions Consultant", (), False),
        ("Senior Data Engineer", "Lead Engineer", ("engineer",), False),
        ("Data Engineers", "data engineering manager", ("data", "engineers"), True),
        ("Product Manager", "Group Product Manager", ("product", "manager"), True),
        ("Product Manager", "Engineering Manager", ("manager",), False),
        ("Engineer", "Firmware Engineer", ("engineer",), True),  # nothing more specific to ask
        ("Senior Data Engineer", None, (), False),
        (None, "Data Engineer", (), False),
        ("Head of Data", "Data Engineer", ("data",), True),
    ],
)
def test_title_overlap(
    role: str | None, other: str | None, shared: tuple[str, ...], similar: bool
) -> None:
    overlap = title_overlap(role, other)
    assert overlap.shared == shared
    assert bool(overlap) is similar
