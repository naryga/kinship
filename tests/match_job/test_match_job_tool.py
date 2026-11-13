"""The `match_job` MCP tool over HTTP, with the fake embedder injected."""

from __future__ import annotations

import json

import pytest
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
)
from mcp import Client
from mcp_testing import TOKEN_A, TOKEN_B

from kinship.rag import FakeEmbedder
from kinship.rag.match import MAX_DESCRIPTION_CHARS, MISSING_TERMS_NOTE, stems
from kinship.server.app import build_mcp
from kinship.server.tools import match_job as tool_module
from kinship.synth import build_world

pytestmark = pytest.mark.anyio

TOOL = "match_job"
WORLD = build_world()
PERSON_KEYS = {"name", "title", "company", "profile_url", "connected_on", "why"}
GROUPS = ("at_company", "similar_titles", "former_at_company")


def names(people: list[dict]) -> list[str]:
    return [person["name"] for person in people]


async def test_tool_is_listed_with_described_parameters(job_app, job_client) -> None:
    async with job_client(job_app, TOKEN_A) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    tool = tools[TOOL]
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.destructive_hint is False
    assert tool.annotations.open_world_hint is False
    properties = tool.input_schema["properties"]
    assert set(properties) == {"description_text", "company", "title", "k"}
    assert tool.input_schema["required"] == ["description_text"]
    for name, schema in properties.items():
        assert len(schema.get("description", "")) > 40, name
    description = tool.description
    for phrase in (
        "résumé",
        "cover letter",
        "who to ask",
        "warm_paths(company)",
        "get_connection",
        "inferred",
        "hybrid",
        "at_company",
        "similar_titles",
        "former_at_company",
        "missing_terms",
        "never searched",
    ):
        assert phrase in description, phrase


async def test_round_trip_with_the_sample_job(job_app, job_client) -> None:
    async with job_client(job_app, TOKEN_A) as client:
        result = await client.call_tool(TOOL, {"description_text": SAMPLE_JOB})

    assert not result.is_error
    data = result.structured_content
    assert list(data) == [
        "role",
        "mode",
        "history",
        "matched_skills",
        "missing_terms",
        "missing_terms_note",
        "connections",
        "connection_totals",
    ]
    assert data["role"] == {
        "title": SAMPLE_TITLE,
        "company": SAMPLE_COMPANY,
        "inferred": True,
        "inferred_fields": ["title", "company"],
        "company_in_network": True,
    }
    assert data["mode"] == "hybrid"

    history = data["history"]
    assert len(history) == 5
    titles = [item["title"] for item in history]
    assert titles[0] == "Position: Senior Data Engineer at Gift Health, Inc."
    assert "Position: Data Engineer at Acme, Inc." in titles
    assert "Recommendation received from Freya Rookwood (2022-12-09)" in titles
    assert "Recommendation received from Vesper Lindenmoor (2021-12-22)" in titles
    job = f" {' '.join(stems(SAMPLE_JOB))} "
    for item in history:
        assert set(item) == {"source", "title", "text", "score", "why", "matched_terms"}
        assert item["source"] in ("my_positions", "recommendations", "profiles")
        assert 0 < item["score"] < 1
        assert item["why"] == f"matched on: {', '.join(item['matched_terms'])}; full-text + vector"
        text = f" {' '.join(stems(item['text']))} "
        for term in item["matched_terms"]:
            key = f" {' '.join(stems(term))} "
            assert key in job and key in text, (item["title"], term)
    assert [item["score"] for item in history] == sorted(
        (item["score"] for item in history), reverse=True
    )
    assert history[0]["why"] == (
        "matched on: data engineer, claims, data pipelines, healthcare billing, pharmacy, "
        "payers; full-text + vector"
    )

    assert data["matched_skills"] == [
        "Apache Airflow",
        "Data Pipelines",
        "dbt",
        "Healthcare Billing",
        "Mentoring",
        "Python",
        "SQL",
    ]
    assert data["missing_terms"] == ["snowflake", "kafka", "spark"]
    assert data["missing_terms_note"] == MISSING_TERMS_NOTE
    assert "Heuristic" in data["missing_terms_note"]

    connections = data["connections"]
    assert list(connections) == list(GROUPS)
    truth = WORLD.connections_at(SAMPLE_COMPANY_KEY, 2)
    assert sorted(names(connections["at_company"])) == sorted(c.full_name for c in truth)
    assert names(connections["similar_titles"]) == ["Sage Foxglove", "Yael Oakhollow"]
    assert connections["former_at_company"] == []
    assert connections["at_company"][0] == {
        "name": "Sage Foxglove",
        "title": "Data Engineer",
        "company": "Tessellate Health",
        "profile_url": "https://www.linkedin.com/in/sage-foxglove-442aaf3b",
        "connected_on": "2016-04-06",
        "why": "Works at Tessellate Health now; similar title (data, engineer)",
    }
    for group in GROUPS:
        for person in connections[group]:
            assert set(person) <= PERSON_KEYS and person["why"]
    assert data["connection_totals"] == {
        "at_company": 2,
        "similar_titles": 2,
        "former_at_company": 0,
    }

    text = result.content[0].text
    summary, _, payload = text.partition("\n")
    assert summary == (
        "Senior Data Engineer at Tessellate Health: 5 relevant parts of your history, "
        "2 connections at the company, 2 with similar titles."
    )
    assert json.loads(payload) == data
    assert "owner_id" not in text and '"id"' not in text and "source_id" not in text


async def test_overrides_k_and_former_colleagues(job_app, job_client, monkeypatch) -> None:
    asked_for: list[int] = []
    real = tool_module.run_match

    def recording(*args, **kwargs):
        asked_for.append(kwargs["k"])
        return real(*args, **kwargs)

    monkeypatch.setattr(tool_module, "run_match", recording)
    async with job_client(job_app, TOKEN_A) as client:
        acme = await client.call_tool(
            TOOL,
            {"description_text": SAMPLE_JOB, "company": "acme, inc.", "title": "Product Manager"},
        )
        one = await client.call_tool(TOOL, {"description_text": SAMPLE_JOB, "k": 1})
        many = await client.call_tool(TOOL, {"description_text": SAMPLE_JOB, "k": 500})
        none = await client.call_tool(TOOL, {"description_text": SAMPLE_JOB, "k": -2})

    data = acme.structured_content
    assert data["role"] == {
        "title": "Product Manager",
        "company": "Acme",
        "company_in_network": True,
        "other_companies_mentioned": [SAMPLE_COMPANY],
    }
    connections = data["connections"]
    at_acme = {c.full_name for c in WORLD.connections_at("acme", 2)}
    assert len(connections["at_company"]) == 5 < len(at_acme)  # capped
    assert set(names(connections["at_company"])) <= at_acme
    assert names(connections["at_company"])[0] == "Reese Jesper"  # Group Product Manager
    assert data["connection_totals"]["at_company"] == len(at_acme)
    assert names(connections["similar_titles"])[0] == "Reese Jesper"
    assert connections["similar_titles"][0]["why"].endswith("; works at Acme")
    assert connections["former_at_company"] == [
        {
            "name": "Elio Emberly",
            "title": "Senior UX Researcher",
            "company": "Gifthealth",
            "profile_url": "https://www.linkedin.com/in/elio-emberly-d3525970",
            "connected_on": "2015-10-18",
            "why": "Worked at Acme as UX Researcher; had moved on by 2026-11-02",
        }
    ]
    assert acme.content[0].text.startswith(
        f"Product Manager at Acme: 5 relevant parts of your history, {len(at_acme)} "
        "connections at the company, 1 who used to work there, 4 with similar titles."
    )

    assert asked_for == [5, 1, 20, 1]  # k is clamped to 1..20
    assert len(one.structured_content["history"]) == 1
    assert one.content[0].text.startswith(
        "Senior Data Engineer at Tessellate Health: 1 relevant part of your history, "
    )
    assert 5 < len(many.structured_content["history"]) <= 20
    assert len(none.structured_content["history"]) == 1


async def test_without_an_embedder_the_tool_answers_from_full_text(
    make_job_app, job_client
) -> None:
    async with make_job_app(embedder=None) as app, job_client(app, TOKEN_A) as client:
        result = await client.call_tool(TOOL, {"description_text": SAMPLE_JOB})

    assert not result.is_error
    data = result.structured_content
    assert data["mode"] == "fts"
    assert data["note"] == "No embedder is configured; results are from full-text search only."
    assert data["history"][0]["title"] == "Position: Senior Data Engineer at Gift Health, Inc."
    assert all(item["why"].endswith("; full-text") for item in data["history"])
    assert names(data["connections"]["similar_titles"]) == ["Sage Foxglove", "Yael Oakhollow"]
    assert (
        result.content[0]
        .text.partition("\n")[0]
        .endswith("2 with similar titles (full-text only).")
    )


async def test_a_job_at_an_unknown_company(job_app, job_client) -> None:
    unknown = SAMPLE_JOB.replace(SAMPLE_COMPANY, "Zzyzx Dynamics")
    async with job_client(job_app, TOKEN_A) as client:
        inferred = await client.call_tool(TOOL, {"description_text": unknown})
        given = await client.call_tool(
            TOOL, {"description_text": unknown, "company": "Zzyzx Dynamics"}
        )
        near = await client.call_tool(TOOL, {"description_text": unknown, "company": "Quill"})

    data = inferred.structured_content
    assert not inferred.is_error
    assert data["role"] == {"title": SAMPLE_TITLE, "inferred": True, "inferred_fields": ["title"]}
    assert data["connections"]["at_company"] == []
    assert data["connections"]["former_at_company"] == []
    assert names(data["connections"]["similar_titles"]) == ["Sage Foxglove", "Yael Oakhollow"]
    assert len(data["history"]) == 5
    assert inferred.content[0].text.startswith(
        "Senior Data Engineer: 5 relevant parts of your history, 2 with similar titles."
    )

    assert given.structured_content["role"] == {
        "title": SAMPLE_TITLE,
        "company": "Zzyzx Dynamics",
        "inferred": True,
        "inferred_fields": ["title"],
        "company_in_network": False,
    }
    assert given.structured_content["connections"]["at_company"] == []
    assert given.content[0].text.startswith(
        "Senior Data Engineer at Zzyzx Dynamics: 5 relevant parts of your history, "
        "no connections at the company, 2 with similar titles."
    )
    assert near.structured_content["role"]["company_suggestions"] == ["Quillfeather Systems, LLC"]


async def test_empty_input_gives_an_error_result(job_app, job_client) -> None:
    async with job_client(job_app, TOKEN_A) as client:
        empty = await client.call_tool(TOOL, {"description_text": "  \n\t "})
        missing = await client.call_tool(TOOL, {})

    assert empty.is_error
    assert "description_text must not be empty" in empty.content[0].text
    assert missing.is_error


@pytest.mark.parametrize(
    "text",
    [
        "?!... --- ///",
        "'; DROP TABLE chunks; --",
        "a & b | !c <-> d:* (((",
        '"unbalanced \\ % _',
        "\U0001f600",
        "data pipelines " * 20_000,
        "Senior Data Engineer\n" + "x" * 300_000,
        "X" * 500,
    ],
    ids=range(8),
)
async def test_odd_input_does_not_error(job_app, job_client, text: str) -> None:
    async with job_client(job_app, TOKEN_A) as client:
        result = await client.call_tool(TOOL, {"description_text": text, "title": "Y" * 500})
        after = await client.call_tool(TOOL, {"description_text": SAMPLE_JOB})

    assert not result.is_error
    data = result.structured_content
    assert set(data["connections"]) == set(GROUPS)
    assert data.get("description_truncated", False) == (len(text) > MAX_DESCRIPTION_CHARS)
    assert len(data["role"]["title"]) <= tool_module.MAX_NAME_CHARS
    assert len(result.content[0].text.partition("\n")[0]) < 300
    assert len(result.content[0].text) < 30_000
    assert len(after.structured_content["history"]) == 5


def test_long_text_is_truncated_and_flagged() -> None:
    from kinship.rag.match import HistoryItem

    def item(text: str) -> HistoryItem:
        return HistoryItem(
            source="recommendations",
            source_id=7,
            title="Recommendation received from Freya Rookwood",
            text=text,
            score=0.123456,
            matched_terms=(),
            retrievers=("vector",),
            why="no keyword in common with the description; similar in meaning (vector)",
        )

    long = tool_module._history_item(item("word " * 1000))
    assert long["truncated"] is True and len(long["text"]) <= tool_module.MAX_TEXT_CHARS
    assert long["text"].endswith("\N{HORIZONTAL ELLIPSIS}")
    assert long["score"] == 0.1235 and "matched_terms" not in long and "source_id" not in long
    short = tool_module._history_item(item("x" * tool_module.MAX_TEXT_CHARS))
    assert "truncated" not in short and short["text"] == "x" * tool_module.MAX_TEXT_CHARS


async def test_no_data_outside_an_authenticated_request(job_session_factory) -> None:
    async with Client(build_mcp(job_session_factory, FakeEmbedder())) as client:
        result = await client.call_tool(TOOL, {"description_text": SAMPLE_JOB})
    assert result.is_error
    assert "history" not in (result.structured_content or {})


# -- owner isolation ------------------------------------------------------------------


async def test_each_token_only_sees_its_own_data(job_app, job_client) -> None:
    calls = [
        {"description_text": SAMPLE_JOB, "k": 20},
        {"description_text": B_ONLY_JOB, "k": 20},
        {"description_text": SAMPLE_JOB, "company": B_ONLY_COMPANY, "k": 20},
        {"description_text": SAMPLE_JOB, "title": "Kite Rigger"},
        {"description_text": f"{SAMPLE_JOB}\nWe partner with {B_ONLY_COMPANY}.", "k": 20},
        {"description_text": f"{B_ONLY_WORD} {B_ONLY_SKILL} telemetry", "title": "Engineer"},
    ]
    async with job_client(job_app, TOKEN_A) as a, job_client(job_app, TOKEN_B) as b:
        for args in calls:
            # Interleave the two owners on purpose.
            result_a = await a.call_tool(TOOL, args)
            result_b = await b.call_tool(TOOL, args)
            again_a = await a.call_tool(TOOL, args)

            data_a, data_b = result_a.structured_content, result_b.structured_content
            assert again_a.structured_content == data_a
            # Everything A is shown, except echoes of what A itself typed.
            echoes = ("role", "missing_terms")
            shown = json.dumps({key: value for key, value in data_a.items() if key not in echoes})
            shown += json.dumps(data_a["role"].get("other_companies_mentioned", []))
            shown += json.dumps(data_a["role"].get("company_suggestions", []))
            for marker in LEAK_MARKERS:
                assert marker not in shown, (marker, args)
            assert data_a["role"].get("company_in_network") is not True or (
                data_a["role"]["company"] != B_ONLY_COMPANY
            )
            for group in GROUPS:
                assert not set(names(data_a["connections"][group])) & set(B_ONLY_NAMES)
            assert (
                data_a["connection_totals"]["at_company"]
                <= (data_b["connection_totals"]["at_company"])
            )

        sample_a = (await a.call_tool(TOOL, calls[0])).structured_content
        sample_b = (await b.call_tool(TOOL, calls[0])).structured_content
        kite_a = (await a.call_tool(TOOL, calls[1])).structured_content
        kite_b = (await b.call_tool(TOOL, calls[1])).structured_content
        told_a = (await a.call_tool(TOOL, calls[2])).structured_content

    # B sees its own extra rows for the same question.
    assert names(sample_a["connections"]["similar_titles"]) == ["Sage Foxglove", "Yael Oakhollow"]
    assert set(names(sample_b["connections"]["similar_titles"])) == {
        "Sage Foxglove",
        "Yael Oakhollow",
        B_ONLY_NAMES[0],
        B_ONLY_NAMES[1],
    }
    assert B_ONLY_NAMES[1] in names(sample_b["connections"]["at_company"])
    assert names(sample_b["connections"]["former_at_company"]) == [B_ONLY_NAMES[2]]
    assert any(B_ONLY_WORD in item["text"] for item in sample_b["history"])
    assert sample_a["connection_totals"] == {
        "at_company": 2,
        "similar_titles": 2,
        "former_at_company": 0,
    }

    # A's company inference never matches a company only B has.
    assert kite_a["role"] == {
        "title": "Kite Engineer",
        "inferred": True,
        "inferred_fields": ["title"],
    }
    assert kite_a["connections"]["at_company"] == []
    assert kite_a["matched_skills"] == ["Data Pipelines", "SQL"]
    assert kite_b["role"]["company"] == B_ONLY_COMPANY and kite_b["role"]["company_in_network"]
    assert set(names(kite_b["connections"]["at_company"])) == {B_ONLY_NAMES[0], B_ONLY_NAMES[2]}
    assert B_ONLY_SKILL in kite_b["matched_skills"]
    assert told_a["role"]["company"] == B_ONLY_COMPANY
    assert told_a["role"]["company_in_network"] is False
    assert told_a["connections"]["at_company"] == []
