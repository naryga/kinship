"""The `semantic_search` MCP tool over HTTP, with the fake embedder injected."""

from __future__ import annotations

import json
import logging

import pytest
import sqlalchemy as sa
from mcp import Client
from retrieval_seeding import (
    B_ONLY_TEXT,
    B_ONLY_WORD,
    TOKEN_A,
    TOKEN_B,
    Corpus,
    connect,
    settings_for,
)

from kinship.db.models import MyPosition, Profile, Recommendation
from kinship.rag import FakeEmbedder
from kinship.rag.chunking import SOURCE_TABLES
from kinship.rag.search import SearchHit
from kinship.server.app import build_mcp, configured_embedder, create_app
from kinship.server.context import NoOwnerError, bind_owner, current_embedder
from kinship.server.tools import semantic_search as tool_module

pytestmark = pytest.mark.anyio

TOOL = "semantic_search"
MODELS = {"recommendations": Recommendation, "my_positions": MyPosition, "profiles": Profile}


def source_owner_ids(session_factory, results: list[dict]) -> set[int]:
    """The owners of the rows that `results` point at (`source`, `source_id`)."""
    owners: set[int] = set()
    with session_factory() as s:
        for result in results:
            model = MODELS[result["source"]]
            owners.add(s.scalar(sa.select(model.owner_id).where(model.id == result["source_id"])))
    return owners


async def test_tool_is_listed_with_described_parameters(search_app) -> None:
    async with connect(search_app, TOKEN_A) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    tool = tools[TOOL]
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.destructive_hint is False
    properties = tool.input_schema["properties"]
    assert set(properties) == {"query", "sources", "k"}
    assert tool.input_schema["required"] == ["query"]
    for name, schema in properties.items():
        assert len(schema.get("description", "")) > 20, name
    for source in SOURCE_TABLES:
        assert source in properties["sources"]["description"]
    description = tool.description
    for word in ("recommendations", "positions", "profile", "Message", "structured tools"):
        assert word in description


async def test_round_trip_returns_fused_results(search_app, committed_corpus, session_factory):
    async with connect(search_app, TOKEN_A) as client:
        result = await client.call_tool(TOOL, {"query": "SOC 2 security audit"})

    assert not result.is_error
    data = result.structured_content
    assert data["query"] == "SOC 2 security audit"
    assert data["mode"] == "hybrid"
    assert data["count"] == len(data["results"]) == 5
    assert "note" not in data and "sources" not in data

    top = data["results"][0]
    assert top["source"] == "recommendations"
    assert top["title"] == "Recommendation received from Casey Foxglove (2023-08-21)"
    assert "Sam led our SOC 2 security audit" in top["text"]
    assert top["matched"] == ["full_text", "vector"]
    assert 0 < top["score"] < 1
    assert "truncated" not in top
    scores = [r["score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)
    for item in data["results"]:
        assert set(item) <= {"source", "source_id", "title", "text", "score", "matched", "part"}
        assert item["matched"] and set(item["matched"]) <= {"full_text", "vector"}
    assert source_owner_ids(session_factory, data["results"]) == {committed_corpus.a}

    text = result.content[0].text
    summary, _, payload = text.partition("\n")
    assert summary == "5 results (hybrid) for 'SOC 2 security audit'."
    assert json.loads(payload) == data
    assert "owner_id" not in text


async def test_sources_and_k(search_app, monkeypatch) -> None:
    asked_for: list[int] = []
    real_search = tool_module.search

    def recording_search(*args, **kwargs):
        asked_for.append(kwargs["k"])
        return real_search(*args, **kwargs)

    monkeypatch.setattr(tool_module, "search", recording_search)
    async with connect(search_app, TOKEN_A) as client:
        positions = (
            await client.call_tool(
                TOOL, {"query": "data pipelines", "sources": ["my_positions"], "k": 2}
            )
        ).structured_content
        many = (await client.call_tool(TOOL, {"query": "Sam", "k": 500})).structured_content
        few = (await client.call_tool(TOOL, {"query": "Sam", "k": 0})).structured_content
        negative = (await client.call_tool(TOOL, {"query": "Sam", "k": -3})).structured_content
        one = await client.call_tool(TOOL, {"query": "Kubernetes Terraform", "k": 1})

    assert positions["count"] == 2
    assert positions["sources"] == ["my_positions"]
    assert {r["source"] for r in positions["results"]} == {"my_positions"}
    assert all(r["title"].startswith("Position: ") for r in positions["results"])
    assert asked_for == [2, 20, 1, 1, 1]  # k is clamped to 1..20
    assert 5 < many["count"] <= 20
    assert few["count"] == negative["count"] == 1
    assert one.content[0].text.startswith("1 result (hybrid) for 'Kubernetes Terraform'.")


async def test_bad_input_gives_an_error_result(search_app) -> None:
    async with connect(search_app, TOKEN_A) as client:
        unknown = await client.call_tool(TOOL, {"query": "data", "sources": ["connections"]})
        empty = await client.call_tool(TOOL, {"query": "   "})
        missing = await client.call_tool(TOOL, {})

    assert unknown.is_error
    message = unknown.content[0].text
    assert "'connections'" in message
    for source in SOURCE_TABLES:
        assert source in message
    assert empty.is_error
    assert "query must not be empty" in empty.content[0].text
    assert missing.is_error


@pytest.mark.parametrize(
    "query",
    ["'; drop table chunks; --", "a & | !", '"unbalanced', "\U0001f600", "pipelines " * 1100],
    ids=range(5),
)
async def test_odd_queries_do_not_error(search_app, query: str) -> None:
    async with connect(search_app, TOKEN_A) as client:
        result = await client.call_tool(TOOL, {"query": query})
        after = await client.call_tool(TOOL, {"query": "SOC 2 security audit"})

    assert not result.is_error
    assert len(result.structured_content["query"]) <= 500
    assert len(result.content[0].text.partition("\n")[0]) < 200
    assert after.structured_content["count"] == 5


async def test_without_an_embedder_the_tool_answers_from_full_text(make_search_app) -> None:
    async with make_search_app(embedder=None) as app, connect(app, TOKEN_A) as client:
        result = await client.call_tool(TOOL, {"query": "SOC 2 security audit"})

    assert not result.is_error
    data = result.structured_content
    assert data["mode"] == "fts"
    assert data["note"] == "No embedder is configured; results are from full-text search only."
    assert data["results"][0]["title"].startswith("Recommendation received from Casey Foxglove")
    assert all(r["matched"] == ["full_text"] for r in data["results"])
    assert result.content[0].text.startswith("3 results (full-text only) for ")


async def test_embedder_comes_from_settings_and_is_never_fake_by_default(
    committed_corpus: Corpus, session_factory, caplog
) -> None:
    no_key = settings_for(committed_corpus)
    with caplog.at_level(logging.WARNING, logger="kinship.server.app"):
        assert configured_embedder(no_key) is None
    assert "full-text only" in caplog.text
    assert isinstance(
        configured_embedder(settings_for(committed_corpus, kinship_embedder="fake")), FakeEmbedder
    )

    # The default app for settings without an API key starts and searches full-text.
    app = create_app(no_key, session_factory)
    async with app.router.lifespan_context(app), connect(app, TOKEN_A) as client:
        data = (await client.call_tool(TOOL, {"query": "security audit"})).structured_content
    assert data["mode"] == "fts"

    with pytest.raises(ValueError):
        create_app(no_key, session_factory, embedder="openai")


async def test_current_embedder_follows_the_request_binding(
    committed_corpus: Corpus, session_factory
) -> None:
    embedder = FakeEmbedder()
    with pytest.raises(NoOwnerError):
        current_embedder()
    with bind_owner(committed_corpus.a, session_factory, embedder):
        assert current_embedder() is embedder
    with bind_owner(committed_corpus.a, session_factory):
        assert current_embedder() is None

    # In-process, with no authenticated HTTP request, the tool reaches no data.
    async with Client(build_mcp(session_factory, embedder)) as client:
        result = await client.call_tool(TOOL, {"query": "security audit"})
    assert result.is_error
    assert "results" not in (result.structured_content or {})


def test_long_text_is_truncated_and_flagged() -> None:
    def hit(text: str, chunk_index: int = 0) -> SearchHit:
        return SearchHit(
            chunk_id=1,
            source_table="messages",
            source_id=7,
            chunk_index=chunk_index,
            text=text,
            score=0.123456,
            title="Conversation started by Freya Rookwood (2024-03-05)",
            fts_rank=2,
        )

    long = tool_module._hit(hit("word " * 1000, chunk_index=2))
    assert long["truncated"] is True
    assert len(long["text"]) <= tool_module.MAX_TEXT_CHARS
    assert long["text"].endswith("\N{HORIZONTAL ELLIPSIS}")
    assert (long["part"], long["score"], long["matched"]) == (3, 0.1235, ["full_text"])

    short = tool_module._hit(hit("x" * tool_module.MAX_TEXT_CHARS))
    assert "truncated" not in short and "part" not in short
    assert short["text"] == "x" * tool_module.MAX_TEXT_CHARS


# -- owner isolation ------------------------------------------------------------


async def test_each_token_searches_only_its_own_chunks(
    search_app, committed_corpus: Corpus, session_factory
) -> None:
    queries = ["SOC 2 security audit", "data pipelines", B_ONLY_WORD, B_ONLY_TEXT, "Onlybee"]
    async with connect(search_app, TOKEN_A) as a, connect(search_app, TOKEN_B) as b:
        for query in queries:
            # Interleave the two owners on purpose.
            args = {"query": query, "k": 20}
            result_a = await a.call_tool(TOOL, args)
            result_b = await b.call_tool(TOOL, args)
            again_a = await a.call_tool(TOOL, args)

            data_a, data_b = result_a.structured_content, result_b.structured_content
            assert again_a.structured_content == data_a
            shown_to_a = json.dumps(data_a["results"])
            assert B_ONLY_WORD not in shown_to_a and "Onlybee" not in shown_to_a
            if data_a["results"]:
                assert source_owner_ids(session_factory, data_a["results"]) == {committed_corpus.a}
            assert source_owner_ids(session_factory, data_b["results"]) == {committed_corpus.b}
            keys_a = {(r["source"], r["source_id"]) for r in data_a["results"]}
            assert keys_a.isdisjoint((r["source"], r["source_id"]) for r in data_b["results"])

        only_b = (await b.call_tool(TOOL, {"query": B_ONLY_WORD})).structured_content
        nothing = (await a.call_tool(TOOL, {"query": B_ONLY_WORD, "k": 20})).structured_content
        same_a = (
            await a.call_tool(TOOL, {"query": "SOC 2 security audit", "k": 3})
        ).structured_content
        same_b = (
            await b.call_tool(TOOL, {"query": "SOC 2 security audit", "k": 3})
        ).structured_content

    assert only_b["results"][0]["title"] == "Recommendation received from Zed Onlybee"
    assert B_ONLY_WORD in only_b["results"][0]["text"]
    assert all("full_text" not in r["matched"] for r in nothing["results"])
    # Identical data gives identical answers, from different rows.
    strip = lambda results: [{k: v for k, v in r.items() if k != "source_id"} for r in results]  # noqa: E731
    assert strip(same_a["results"]) == strip(same_b["results"])
    assert [r["source_id"] for r in same_a["results"]] != [
        r["source_id"] for r in same_b["results"]
    ]
