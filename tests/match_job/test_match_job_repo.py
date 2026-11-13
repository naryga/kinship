"""`JobMatchRepository`: the owner-scoped queries behind `match_job`."""

from __future__ import annotations

from ingest.conftest import V1_AT, V2_AT
from match_job_seeding import (
    B_ONLY_AT,
    B_ONLY_COMPANY,
    B_ONLY_JOB,
    B_ONLY_NAMES,
    B_ONLY_SKILL,
    B_ONLY_WORD,
    SAMPLE_COMPANY,
    SAMPLE_JOB,
    JobWorld,
)
from sqlalchemy.orm import Session

from kinship.db.models import Company
from kinship.rag.match import HISTORY_SOURCES
from kinship.repo import Repository
from kinship.synth import build_world
from kinship.synth.data import OWNER_SKILLS

WORLD = build_world()


def names(companies) -> list[str]:
    return [company.display_name for company in companies]


# -- companies named in a text ------------------------------------------------------


def test_companies_named_in_finds_any_spelling_on_word_boundaries(repo_a: Repository) -> None:
    found = repo_a.companies_named_in(SAMPLE_JOB)
    assert names(found) == [SAMPLE_COMPANY]
    assert found[0].mentions == 3 and found[0].exact_spelling
    assert found[0].connection_count == len(WORLD.connections_at("tessellate", 2))

    # Case, punctuation, spacing and legal suffixes do not matter...
    gifthealth = repo_a.find_companies("Gifthealth")[0]
    for text in ("join GIFT HEALTH today", "at Gift Health, Inc.", "the gifthealth team"):
        assert [c.id for c in repo_a.companies_named_in(text)] == [gifthealth.id], text
    twice = repo_a.companies_named_in("Gifthealth (formerly Gift Health, Inc.)")
    assert [c.mentions for c in twice] == [2]
    assert names(repo_a.companies_named_in("Juniper and Wren Studio is hiring")) == [
        "Juniper & Wren Studio"
    ]
    # ...but a name inside another word is not a mention.
    for text in ("acmeology", "macme", "pacme corp", "giftedhealth", "Quillfeather"):
        assert repo_a.companies_named_in(text) == [], text
    assert repo_a.companies_named_in("") == []
    assert repo_a.companies_named_in("   \n ") == []
    assert repo_a.companies_named_in(SAMPLE_JOB, limit=0) == []


def test_companies_named_in_ranks_exact_spellings_and_mentions(repo_a: Repository) -> None:
    text = "Acme. acme? ACME! We compete with Halcyon Grid and with saltmarsh bank."
    found = repo_a.companies_named_in(text)
    # Halcyon Grid and Acme are spelled as stored; Acme is mentioned more often.
    assert names(found) == ["Acme", "Halcyon Grid", "Saltmarsh Bank"]
    assert [c.mentions for c in found] == [3, 1, 1]
    assert [c.exact_spelling for c in found] == [True, True, False]
    assert names(repo_a.companies_named_in(text, limit=1)) == ["Acme"]


def test_companies_named_in_skips_short_and_generic_names(
    session: Session, job_world: JobWorld, repo_a: Repository
) -> None:
    session.add_all(
        [
            Company(owner_id=job_world.a, name_normalized="xy", display_name="Xy", aliases=["Xy"]),
            Company(
                owner_id=job_world.a,
                name_normalized="remote",
                display_name="Remote",
                aliases=["Remote", "remote"],
            ),
            Company(
                owner_id=job_world.a,
                name_normalized="selfemployed",
                display_name="Self-employed",
                aliases=["Self-employed"],
            ),
        ]
    )
    session.flush()
    text = "Remote role. Xy marks the spot; self-employed contractors welcome. Call Norwhal Labs."
    assert names(repo_a.companies_named_in(text)) == ["Norwhal Labs"]


def test_companies_named_in_only_sees_the_owners_companies(
    repo_a: Repository, repo_b: Repository, job_world: JobWorld
) -> None:
    assert repo_a.companies_named_in(B_ONLY_JOB) == []
    assert repo_a.companies_named_in(B_ONLY_COMPANY) == []
    found = repo_b.companies_named_in(B_ONLY_JOB)
    assert names(found) == [B_ONLY_COMPANY]
    assert found[0].id == job_world.b_company_id and found[0].connection_count == 2

    both = f"{SAMPLE_JOB}\nWe partner with {B_ONLY_COMPANY}."
    assert names(repo_a.companies_named_in(both)) == [SAMPLE_COMPANY]
    assert names(repo_b.companies_named_in(both)) == [SAMPLE_COMPANY, B_ONLY_COMPANY]
    # Same name, different rows.
    assert repo_a.companies_named_in(both)[0].id != repo_b.companies_named_in(both)[0].id


# -- skills and terms -----------------------------------------------------------------


def test_skill_names_are_the_owners(repo_a: Repository, repo_b: Repository) -> None:
    assert repo_a.skill_names() == sorted(OWNER_SKILLS)
    assert set(repo_b.skill_names()) == {*OWNER_SKILLS, B_ONLY_SKILL}


def test_terms_in_chunks(repo_a: Repository, repo_b: Repository) -> None:
    terms = ["airflow", "pipeline", "mentoring", "snowflake", "kafka", B_ONLY_WORD, "the", ""]
    # Stemmed like full-text search: "pipeline" finds "pipelines", "mentoring" finds "Mentor".
    assert repo_a.terms_in_chunks(terms) == {"airflow", "pipeline", "mentoring"}
    assert repo_b.terms_in_chunks(terms) == {"airflow", "pipeline", "mentoring", B_ONLY_WORD}
    assert repo_a.terms_in_chunks([]) == set()

    # "kubernetes" is only in a recommendation, "freight" in a position and the profile.
    assert repo_a.terms_in_chunks(["kubernetes", "freight"], sources=["my_positions"]) == {
        "freight"
    }
    assert repo_a.terms_in_chunks(["kubernetes"], sources=HISTORY_SOURCES) == {"kubernetes"}
    assert repo_a.terms_in_chunks(["kubernetes"], sources=["messages"]) == set()
    # Odd input is data, never syntax.
    odd = ["'; drop table chunks; --", "a & | !", "%", "\\", "airflow"]
    assert repo_a.terms_in_chunks(odd) == {"airflow"}
    many = [f"word{i}" for i in range(200)] + ["airflow"]
    assert repo_a.terms_in_chunks(many) == set()  # only the first 40 are checked


# -- titles ---------------------------------------------------------------------------


def test_connections_with_title_words(repo_a: Repository) -> None:
    current = WORLD.connections(2)
    found = repo_a.connections_with_title_words(["data"])
    assert {p.name for p in found} == {c.full_name for c in current if "data" in c.position.lower()}
    assert len(found) == 2

    either = repo_a.connections_with_title_words(["DATA", "product"])
    assert {p.name for p in either} == {
        c.full_name for c in current if {"data", "product"} & set(c.position.lower().split())
    }
    assert [(p.name.split()[-1], p.name) for p in either] == sorted(
        (p.name.split()[-1], p.name) for p in either
    )
    assert all(p.title and p.profile_url for p in either)

    # Removed connections are gone: Jules Kettleby was a Payments Engineer in export 1.
    removed_titles = {c.position for c in WORLD.connections(1)} - {c.position for c in current}
    assert "Payments Engineer" in removed_titles
    assert repo_a.connections_with_title_words(["payments"]) == []
    assert repo_a.connections_with_title_words([]) == []
    assert repo_a.connections_with_title_words(["", "  "]) == []
    assert repo_a.connections_with_title_words(["data"], limit=1) == found[:1]
    # LIKE wildcards are escaped.
    assert repo_a.connections_with_title_words(["%", "_"]) == []
    assert repo_a.connections_with_title_words(["d%a"]) == []


def test_title_search_is_owner_scoped(repo_a: Repository, repo_b: Repository) -> None:
    for word in ("data", "kite", "engineer", "rigger"):
        mine = {p.name for p in repo_a.connections_with_title_words([word])}
        theirs = {p.name for p in repo_b.connections_with_title_words([word])}
        assert not mine & set(B_ONLY_NAMES)
        assert mine <= theirs
    assert {p.name for p in repo_b.connections_with_title_words(["kite"])} == {B_ONLY_NAMES[2]}
    a_ids = {p.id for p in repo_a.connections_with_title_words(["data"])}
    b_ids = {p.id for p in repo_b.connections_with_title_words(["data"])}
    assert a_ids.isdisjoint(b_ids)


# -- departures -----------------------------------------------------------------------


def test_company_departures(repo_a: Repository) -> None:
    acme = repo_a.find_companies("Acme")[0]
    gifthealth = repo_a.find_companies("Gifthealth")[0]
    former = repo_a.former_connections_of_companies([acme.id])
    assert [p.name for p in former] == ["Elio Emberly"]
    elio = former[0]

    left = repo_a.company_departures([acme.id], [elio.id])[elio.id]
    assert (left.company_id, left.title) == (acme.id, "UX Researcher")
    assert (left.last_seen_at, left.moved_by) == (V1_AT, V2_AT)

    # Still at Gifthealth: seen there, no later row elsewhere.
    still = repo_a.company_departures([gifthealth.id], [elio.id])[elio.id]
    assert (still.title, still.last_seen_at, still.moved_by) == (
        "Senior UX Researcher",
        V2_AT,
        None,
    )
    everyone = [p.id for p in repo_a.connections_at_companies([gifthealth.id])]
    assert repo_a.company_departures([acme.id], everyone).keys() == {elio.id}
    assert repo_a.company_departures([], [elio.id]) == {}
    assert repo_a.company_departures([acme.id], []) == {}


def test_company_departures_are_owner_scoped(
    repo_a: Repository, repo_b: Repository, job_world: JobWorld
) -> None:
    leaver = job_world.b_connection_ids[2]
    tessellate_a = repo_a.find_companies(SAMPLE_COMPANY)[0].id
    tessellate_b = repo_b.find_companies(SAMPLE_COMPANY)[0].id

    left = repo_b.company_departures([tessellate_b], [leaver])[leaver]
    assert (left.title, left.last_seen_at, left.moved_by) == (
        "Claims Data Analyst",
        V2_AT,
        B_ONLY_AT,
    )
    # A cannot read B's connection, with either owner's company id.
    assert repo_a.company_departures([tessellate_b], [leaver]) == {}
    assert repo_a.company_departures([tessellate_a], [leaver]) == {}
    assert repo_a.company_departures([tessellate_a, tessellate_b], job_world.b_connection_ids) == {}
    assert repo_a.former_connections_of_companies([tessellate_b]) == []
    assert repo_a.connections_at_companies([job_world.b_company_id]) == []
