"""`kinship.rag.match.match_job` against the synthetic world, with the fake embedder."""

from __future__ import annotations

import dataclasses
import json

import pytest
import sqlalchemy as sa
from match_job_seeding import (
    B_ONLY_COMPANY,
    B_ONLY_JOB,
    B_ONLY_NAMES,
    B_ONLY_SKILL,
    B_ONLY_WORD,
    LEAK_MARKERS,
    SAMPLE_COMPANY,
    SAMPLE_COMPANY_KEY,
    SAMPLE_JOB,
    SAMPLE_TITLE,
    JobWorld,
)
from sqlalchemy.orm import Session

from kinship.db.models import Chunk, Connection, MyPosition, Profile, Recommendation
from kinship.rag import FakeEmbedder
from kinship.rag import match as match_module
from kinship.rag.match import HISTORY_SOURCES, JobMatch, match_job, stems, title_overlap
from kinship.repo import Repository
from kinship.synth import build_world

WORLD = build_world()
MODELS = {"recommendations": Recommendation, "my_positions": MyPosition, "profiles": Profile}
EVERYONE = 1000


def run(repo: Repository, text: str = SAMPLE_JOB, **kwargs) -> JobMatch:
    kwargs.setdefault("embedder", FakeEmbedder())
    return match_job(repo, kwargs.pop("embedder"), text, **kwargs)


def people(matches) -> list[str]:
    return [match.person.name for match in matches]


def spaced(text: str) -> str:
    return f" {' '.join(stems(text))} "


def dump(result: JobMatch) -> str:
    return json.dumps(dataclasses.asdict(result), default=str)


# -- part 1: history ----------------------------------------------------------------


def test_sample_job_finds_the_relevant_history(repo_a: Repository) -> None:
    result = run(repo_a)

    assert result.mode_used == "hybrid" and result.note is None and not result.truncated
    titles = [item.title for item in result.history]
    assert len(titles) == 5
    assert titles[0] == "Position: Senior Data Engineer at Gift Health, Inc."
    assert "Position: Data Engineer at Acme, Inc." in titles[:3]
    # The two recommendations about pipelines and about claims / healthcare billing.
    assert "Recommendation received from Freya Rookwood (2022-12-09)" in titles
    assert "Recommendation received from Vesper Lindenmoor (2021-12-22)" in titles
    assert {item.source for item in result.history} <= set(HISTORY_SOURCES)
    scores = [item.score for item in result.history]
    assert scores == sorted(scores, reverse=True) and scores[-1] > 0


def test_why_lists_terms_that_are_in_the_description_and_in_the_text(repo_a: Repository) -> None:
    result = run(repo_a, k=20)
    job = spaced(SAMPLE_JOB)
    assert len(result.history) > 5

    for item in result.history:
        assert item.retrievers and set(item.retrievers) <= {"fts", "vector"}
        assert len(item.matched_terms) <= match_module.MAX_EVIDENCE_TERMS
        chunk = spaced(item.text)
        for term in item.matched_terms:
            assert spaced(term) in job, (item.title, term)
            assert spaced(term) in chunk, (item.title, term)
            # Never the hiring company's own name.
            assert term not in ("tessellate", "health", "tessellate health")
        if item.matched_terms:
            assert item.why.startswith(f"matched on: {', '.join(item.matched_terms)}; ")
        else:
            assert item.why.startswith("no keyword in common")
        assert item.why.endswith(("full-text", "vector", "full-text + vector", "vector)"))
        if "fts" in item.retrievers:
            assert item.matched_terms, item.title

    by_title = {item.title: item for item in result.history}
    gifthealth = by_title["Position: Senior Data Engineer at Gift Health, Inc."]
    assert gifthealth.matched_terms[0] == "data engineer"
    assert {"claims", "data pipelines", "healthcare billing"} <= set(gifthealth.matched_terms)
    assert gifthealth.why.endswith("; full-text + vector")
    # Literally there, not only by stem.
    for term in ("data pipelines", "healthcare billing", "claims"):
        assert term in SAMPLE_JOB.lower() and term in gifthealth.text.lower()
    acme = by_title["Position: Data Engineer at Acme, Inc."]
    assert {"airflow", "dbt", "warehouse"} <= set(acme.matched_terms)
    freya = by_title["Recommendation received from Freya Rookwood (2022-12-09)"]
    assert {"data pipelines", "airflow", "warehouse"} <= set(freya.matched_terms)
    vesper = by_title["Recommendation received from Vesper Lindenmoor (2021-12-22)"]
    assert {"healthcare billing", "pharmacy claims"} <= set(vesper.matched_terms)


def test_history_never_includes_messages(session: Session, job_world: JobWorld) -> None:
    from kinship.rag import sync_chunks

    sync_chunks(session, job_world.a, FakeEmbedder(), include_messages=True)
    session.flush()
    assert session.scalar(
        sa.select(sa.func.count())
        .select_from(Chunk)
        .where(Chunk.owner_id == job_world.a, Chunk.source_table == "messages")
    )
    # A description made of a message's words still gets no message back.
    text = "Data Engineer\nOur Airflow backfills keep timing out; batch pipelines, partitions."
    result = run(Repository(session, job_world.a), text, k=20)
    assert result.history
    assert {item.source for item in result.history} <= set(HISTORY_SOURCES)


def test_matched_skills_and_missing_terms(repo_a: Repository) -> None:
    result = run(repo_a)
    assert result.matched_skills == [
        "Apache Airflow",
        "Data Pipelines",
        "dbt",
        "Healthcare Billing",
        "Mentoring",  # the description says "Mentor other engineers"
        "Python",
        "SQL",
    ]
    # Named tools that Sam's history and skills never mention.
    assert result.missing_terms == ["snowflake", "kafka", "spark"]
    for term in result.missing_terms:
        assert term in SAMPLE_JOB.lower()

    no_gaps = run(repo_a, "Data Engineer\nWe use Airflow, dbt and SQL. Airflow, dbt, SQL!")
    assert no_gaps.missing_terms == []
    assert no_gaps.matched_skills == ["dbt", "SQL"]


# -- the role ---------------------------------------------------------------------------


def test_title_and_company_are_inferred_from_the_text(repo_a: Repository) -> None:
    role = run(repo_a).role
    assert (role.title, role.company) == (SAMPLE_TITLE, SAMPLE_COMPANY)
    assert role.title_inferred and role.company_inferred and role.company_known
    assert role.other_companies == () and role.company_suggestions == ()

    # A company the text labels wins over one it merely mentions.
    labelled = run(
        repo_a,
        "Title: Analytics Engineer\nCompany: Norwhal Labs\n\nYou will work with our partner "
        "Acme. Acme is great. Acme!",
    ).role
    assert (labelled.title, labelled.company) == ("Analytics Engineer", "Norwhal Labs")
    assert labelled.company_known and labelled.other_companies == ("Acme",)

    # Any spelling resolves to the owner's company.
    spelled = run(repo_a, "Data Engineer at ACME Corp.\n\nBuild pipelines.").role
    assert (spelled.title, spelled.company, spelled.company_known) == (
        "Data Engineer",
        "Acme",
        True,
    )


def test_parameters_override_what_the_text_says(repo_a: Repository) -> None:
    result = run(repo_a, company="acme inc", title="  Product   Manager ")
    role = result.role
    assert (role.title, role.company) == ("Product Manager", "Acme")
    assert not role.title_inferred and not role.company_inferred and role.company_known
    assert role.other_companies == (SAMPLE_COMPANY,)

    assert set(people(run(repo_a, company="Acme", group_limit=EVERYONE).at_company)) == {
        c.full_name for c in WORLD.connections_at("acme", 2)
    }
    similar = run(repo_a, title="Product Manager", group_limit=EVERYONE).similar_titles
    assert people(similar)[:3] == ["Reese Jesper", "Devon Marrowby", "Freya Underhill"]
    assert all("product" in match.person.title.lower() for match in similar)

    # Only one of the two given: the other is still inferred.
    only_title = run(repo_a, title="Claims Lead").role
    assert (only_title.title, only_title.company) == ("Claims Lead", SAMPLE_COMPANY)
    assert not only_title.title_inferred and only_title.company_inferred
    # The history part does not depend on the company.
    assert [i.title for i in result.history][:2] == [i.title for i in run(repo_a).history][:2]


def test_a_company_the_owner_does_not_know(repo_a: Repository) -> None:
    for kwargs in ({"company": "Zzyzx Dynamics"}, {}):
        text = SAMPLE_JOB if kwargs else SAMPLE_JOB.replace(SAMPLE_COMPANY, "Zzyzx Dynamics")
        result = run(repo_a, text, **kwargs)
        role = result.role
        assert role.title == SAMPLE_TITLE and not role.company_known
        assert role.company == kwargs.get("company")  # nothing is made up from the text
        assert result.at_company == [] and result.former_at_company == []
        assert result.totals["at_company"] == result.totals["former_at_company"] == 0
        assert people(result.similar_titles) == ["Sage Foxglove", "Yael Oakhollow"]
        assert len(result.history) == 5

    # A given name that is not exact matches nothing, but near misses are suggested.
    near = run(repo_a, company="Quillfeather").role
    assert (near.company, near.company_known) == ("Quillfeather", False)
    assert near.company_suggestions == ("Quillfeather Systems, LLC",)
    labelled = run(repo_a, "Role: Baker\nCompany: Sourdough & Sons\n\nBake bread.").role
    assert (labelled.company, labelled.company_inferred, labelled.company_known) == (
        "Sourdough & Sons",
        True,
        False,
    )


# -- part 2: connections ----------------------------------------------------------------


def test_at_company_is_the_truth(repo_a: Repository) -> None:
    result = run(repo_a, group_limit=EVERYONE)
    truth = WORLD.connections_at(SAMPLE_COMPANY_KEY, 2)
    assert len(truth) == 2
    assert sorted(people(result.at_company)) == sorted(c.full_name for c in truth)
    assert {m.person.profile_url for m in result.at_company} == {c.url for c in truth}
    assert result.totals["at_company"] == len(truth)
    # Jules Kettleby was there in export 1 and is no longer a connection.
    assert "Jules Kettleby" in {c.full_name for c in WORLD.connections_at(SAMPLE_COMPANY_KEY, 1)}
    assert "Jules Kettleby" not in dump(result)

    by_name = {m.person.name: m for m in result.at_company}
    assert people(result.at_company)[0] == "Sage Foxglove"  # the similar title comes first
    assert by_name["Sage Foxglove"].why == (
        "Works at Tessellate Health now; similar title (data, engineer)"
    )
    assert by_name["Ulric Quarrymoor"].why == "Works at Tessellate Health now"
    assert by_name["Ulric Quarrymoor"].person.as_dict() == {
        "name": "Ulric Quarrymoor",
        "title": "Claims Operations Lead",
        "company": "Tessellate Health",
        "profile_url": "https://www.linkedin.com/in/ulric-quarrymoor-7302676a",
        "connected_on": "2025-11-22",
    }


def test_similar_titles_share_a_substantive_word(repo_a: Repository) -> None:
    result = run(repo_a, group_limit=EVERYONE)
    expected = {c.full_name for c in WORLD.connections(2) if "data" in c.position.lower().split()}
    assert expected == {"Sage Foxglove", "Yael Oakhollow"}
    # Both share the same words; the one at the company comes first.
    assert people(result.similar_titles) == ["Sage Foxglove", "Yael Oakhollow"]
    assert result.totals["similar_titles"] == 2
    assert result.similar_titles[0].why == (
        "Similar title: Data Engineer \N{LEFT RIGHT ARROW} Senior Data Engineer "
        "(shared: data, engineer); works at Tessellate Health"
    )
    assert result.similar_titles[1].why.endswith("(shared: data, engineer)")

    # A broad title: everybody returned shares a word that is not just "engineer".
    broad = run(repo_a, title="Senior Machine Learning Platform Engineer", group_limit=EVERYONE)
    assert len(broad.similar_titles) >= 3
    for match in broad.similar_titles:
        words = set(match.person.title.lower().split())
        assert words & {"machine", "learning", "platform"}, match.person.title
        assert title_overlap(broad.role.title, match.person.title)
    best = broad.similar_titles[0].person.title
    assert "Machine Learning Engineer" in best
    engineers = [c for c in WORLD.connections(2) if c.position.endswith("Engineer")]
    assert len(engineers) > len(broad.similar_titles)

    assert run(repo_a, title="Zookeeper").similar_titles == []
    capped = run(repo_a, title="Engineer", group_limit=3)
    assert len(capped.similar_titles) == 3 and capped.totals["similar_titles"] > 3


def test_former_at_company(repo_a: Repository) -> None:
    result = run(repo_a, company="Acme")
    assert people(result.former_at_company) == ["Elio Emberly"]
    former = result.former_at_company[0]
    assert former.why == "Worked at Acme as UX Researcher; had moved on by 2026-11-02"
    assert former.person.company == "Gifthealth" and former.person.title == "Senior UX Researcher"
    assert result.totals["former_at_company"] == 1
    assert "Elio Emberly" not in people(result.at_company)
    assert run(repo_a).former_at_company == []


# -- degraded and odd input -------------------------------------------------------------


def test_without_an_embedder_it_is_full_text_only(repo_a: Repository) -> None:
    result = run(repo_a, embedder=None)
    assert result.mode_used == "fts"
    assert result.note == "No embedder is configured; results are from full-text search only."
    titles = [item.title for item in result.history]
    assert titles[0] == "Position: Senior Data Engineer at Gift Health, Inc."
    assert "Recommendation received from Vesper Lindenmoor (2021-12-22)" in titles
    for item in result.history:
        assert item.retrievers == ("fts",)
        assert item.why.endswith("; full-text") and item.matched_terms
    assert people(result.at_company) == people(run(repo_a).at_company)


def test_without_stored_vectors_it_is_full_text_only(session: Session, job_world: JobWorld):
    session.execute(
        sa.update(Chunk).where(Chunk.owner_id == job_world.a).values(embedding=None, model=None)
    )
    result = run(Repository(session, job_world.a))
    assert result.mode_used == "fts" and "kinship embed" in result.note
    assert len(result.history) == 5
    # B's vectors are untouched and unaffected.
    assert run(Repository(session, job_world.b)).mode_used == "hybrid"


@pytest.mark.parametrize(
    "text",
    [
        "?!... --- /// \x00",
        "'; DROP TABLE chunks; --",
        "a & b | !c <-> d:* (((",
        '"unbalanced quote and \\ backslash % _',
        "\U0001f600 \U0001f680",
        "the and of to with",
        "x",
        "data " * 30_000,
        "Senior Data Engineer\n" + "pipelines & claims | 'airflow' :* !! " * 2_000,
        "\n".join(f"line{i} unique{i} words{i}" for i in range(4_000)),
    ],
    ids=range(10),
)
def test_odd_input_does_not_error(repo_a: Repository, text: str) -> None:
    result = run(repo_a, text)
    assert len(result.history) <= 5
    assert result.truncated == (len(text.strip()) > match_module.MAX_DESCRIPTION_CHARS)
    if result.mode_used is None:
        assert result.history == [] and result.note == "The description has no searchable words."
    json.loads(dump(result))
    # The session is still usable afterwards.
    assert run(repo_a).history


def test_bad_arguments(repo_a: Repository) -> None:
    for text in ("", "   \n\t", None):
        with pytest.raises(ValueError, match="must not be empty"):
            run(repo_a, text)
    with pytest.raises(ValueError):
        run(repo_a, k=0)
    with pytest.raises(ValueError):
        run(repo_a, group_limit=0)
    assert len(run(repo_a, k=1).history) == 1
    assert len(run(repo_a, k=2, group_limit=1).at_company) == 1


def test_number_of_queries_is_bounded(repo_a: Repository, monkeypatch) -> None:
    calls: list[dict] = []
    real = match_module.search

    def recording(repo, embedder, query, **kwargs):
        calls.append({"query": query, **kwargs})
        return real(repo, embedder, query, **kwargs)

    monkeypatch.setattr(match_module, "search", recording)
    long_job = SAMPLE_JOB + "\n\n".join(
        f"Section {i}\nWe need kubernetes terraform observability dashboards number{i} "
        f"alpha{i} beta{i} gamma{i}"
        for i in range(30)
    )
    run(repo_a, long_job)
    assert 2 <= len(calls) <= 1 + match_module.MAX_SECTIONS
    for call in calls:
        assert call["mode"] == "hybrid" and tuple(call["sources"]) == HISTORY_SOURCES
        assert len(call["query"].split()) <= match_module.MAX_FOCUSED_TERMS
        assert "messages" not in call["sources"]


# -- owner isolation --------------------------------------------------------------------


def source_owners(session: Session, result: JobMatch) -> set[int]:
    owners = set()
    for item in result.history:
        model = MODELS[item.source]
        owners.add(session.scalar(sa.select(model.owner_id).where(model.id == item.source_id)))
    return owners


def person_owners(session: Session, result: JobMatch) -> set[int]:
    ids = [
        match.person.id
        for group in (result.at_company, result.similar_titles, result.former_at_company)
        for match in group
    ]
    return set(session.scalars(sa.select(Connection.owner_id).where(Connection.id.in_(ids))))


def test_a_never_sees_bs_rows(
    session: Session, job_world: JobWorld, repo_a: Repository, repo_b: Repository
) -> None:
    for kwargs in ({}, {"embedder": None}, {"company": B_ONLY_COMPANY}, {"title": "Kite Rigger"}):
        for text in (SAMPLE_JOB, B_ONLY_JOB, f"{SAMPLE_JOB}\n{B_ONLY_JOB}"):
            mine = run(repo_a, text, k=20, group_limit=EVERYONE, **kwargs)
            theirs = run(repo_b, text, k=20, group_limit=EVERYONE, **kwargs)

            shown = dump(dataclasses.replace(mine, role=match_module.Role()))
            for marker in LEAK_MARKERS:
                assert marker not in shown, (marker, kwargs)
            assert source_owners(session, mine) <= {job_world.a}
            assert person_owners(session, mine) <= {job_world.a}
            assert source_owners(session, theirs) == {job_world.b}
            assert person_owners(session, theirs) <= {job_world.b}
            assert not set(people(mine.similar_titles)) & set(B_ONLY_NAMES)


def test_b_sees_its_own_extra_rows(repo_a: Repository, repo_b: Repository) -> None:
    mine = run(repo_a, k=20, group_limit=EVERYONE)
    theirs = run(repo_b, k=20, group_limit=EVERYONE)

    # Same exports: B has everything A has, plus its own people and recommendation.
    assert set(people(theirs.at_company)) == {*people(mine.at_company), B_ONLY_NAMES[1]}
    assert set(people(theirs.similar_titles)) == {
        *people(mine.similar_titles),
        B_ONLY_NAMES[0],
        B_ONLY_NAMES[1],
    }
    # Same shared words: the two at the company come first.
    assert people(theirs.similar_titles)[:2] == ["Sage Foxglove", B_ONLY_NAMES[1]]
    assert people(theirs.former_at_company) == [B_ONLY_NAMES[2]]
    assert theirs.former_at_company[0].why == (
        "Worked at Tessellate Health as Claims Data Analyst; had moved on by 2026-12-01"
    )
    assert mine.former_at_company == []
    assert any(B_ONLY_WORD in item.text for item in theirs.history)
    assert not any(B_ONLY_WORD in item.text for item in mine.history)
    assert {i.title for i in mine.history[:5]} <= {i.title for i in theirs.history}


def test_company_inference_never_matches_another_owners_company(
    repo_a: Repository, repo_b: Repository
) -> None:
    mine, theirs = run(repo_a, B_ONLY_JOB), run(repo_b, B_ONLY_JOB)

    assert (mine.role.title, mine.role.company, mine.role.company_known) == (
        "Kite Engineer",
        None,
        False,
    )
    assert mine.at_company == [] and mine.former_at_company == []
    assert mine.matched_skills == ["Data Pipelines", "SQL"]

    assert (theirs.role.company, theirs.role.company_known) == (B_ONLY_COMPANY, True)
    assert set(people(theirs.at_company)) == {B_ONLY_NAMES[0], B_ONLY_NAMES[2]}
    assert theirs.matched_skills == ["Data Pipelines", "SQL", B_ONLY_SKILL]
    assert people(theirs.similar_titles) == [B_ONLY_NAMES[2]]  # Kite Rigger

    # Told the name outright, A still learns nothing: it is not one of A's companies.
    told = run(repo_a, B_ONLY_JOB, company=B_ONLY_COMPANY)
    assert (told.role.company, told.role.company_known) == (B_ONLY_COMPANY, False)
    assert told.role.company_suggestions == () and told.at_company == []
    both = run(repo_a, f"{SAMPLE_JOB}\nWe partner with {B_ONLY_COMPANY}.")
    assert both.role.company == SAMPLE_COMPANY and both.role.other_companies == ()
    assert run(repo_b, f"{SAMPLE_JOB}\nWe partner with {B_ONLY_COMPANY}.").role.other_companies == (
        B_ONLY_COMPANY,
    )
