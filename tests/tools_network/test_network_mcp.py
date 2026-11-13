"""MCP round trips for `warm_paths`, `network_changes`, the imports resource and the prompt.

Owners A and B hold the same two synthetic exports; B has extra rows (see
`conftest.add_b_only_data`). Owner C holds only export v1, without messages.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from ingest.conftest import V1_AT, V2_AT
from mcp_testing import TOKEN_A, TOKEN_B

from kinship.synth import build_world

from .conftest import B_ONLY_AT, B_ONLY_COMPANY, B_ONLY_URL, TOKEN_C

pytestmark = pytest.mark.anyio

WORLD = build_world()
TRUTH = WORLD.truth
URI = "kinship://imports/latest"
PERSON_KEYS = {"name", "title", "company", "profile_url", "connected_on"}


async def call(client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    summary, _, compact = result.content[0].text.partition("\n")
    assert summary and json.loads(compact) == result.structured_content
    assert "owner_id" not in compact
    return result.structured_content


async def read_json(client) -> dict[str, Any]:
    result = await client.read_resource(URI)
    (content,) = result.contents
    assert content.mime_type == "application/json"
    return json.loads(content.text)


def urls(items: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(item["profile_url"] for item in items))


# -- listing --------------------------------------------------------------------


async def test_tools_are_listed_read_only_with_described_parameters(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    expected = {"warm_paths": {"company", "limit"}, "network_changes": {"since", "limit"}}
    for name, parameters in expected.items():
        tool = tools[name]
        assert tool.description and tool.title
        assert tool.annotations.read_only_hint is True
        properties = tool.input_schema["properties"]
        assert set(properties) == parameters
        for parameter, schema in properties.items():
            assert len(schema.get("description", "")) > 20, (name, parameter)
    assert tools["warm_paths"].input_schema["required"] == ["company"]
    assert "required" not in tools["network_changes"].input_schema


# -- warm_paths -----------------------------------------------------------------


async def test_warm_paths_ranks_with_reasons(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as client:
        data = await call(client, "warm_paths", company="Gift Health, Inc.", limit=50)
        again = await call(client, "warm_paths", company="gifthealth", limit=50)

    at_company = WORLD.connections_at("gifthealth", 2)
    (matched,) = data["matched_companies"]
    assert matched["match"] == "exact"
    assert matched["current_connections"] == data["total_current_connections"] == len(at_company)
    assert again["paths"] == data["paths"]  # deterministic, whatever the spelling

    paths = data["paths"]
    assert sorted(p.get("profile_url", "") for p in paths) == sorted(c.url for c in at_company)
    scores = [p["score"] for p in paths]
    assert scores == sorted(scores, reverse=True)
    for path in paths:
        assert set(path) - PERSON_KEYS == {"score", "reasons"}
        assert path["reasons"] and all(isinstance(reason, str) for reason in path["reasons"])
        cold = path["reasons"][0].startswith("No recorded interaction")
        assert cold == (path["score"] == 0)

    by_name = {p["name"]: p for p in paths}
    casey = by_name["Casey Foxglove"]
    assert casey["reasons"][0] == "Wrote you a recommendation in 2023"
    assert any(r.startswith("You exchanged 3 messages, last on ") for r in casey["reasons"])
    assert any(r.startswith("You endorsed them for ") for r in casey["reasons"])
    elio = by_name["Elio Emberly"]  # moved from Acme, where the owner also worked
    assert "You both worked at Acme" in elio["reasons"]
    assert "You wrote a note about them" in elio["reasons"]

    cold_paths = [p for p in paths if p["score"] == 0]
    assert cold_paths
    assert paths.index(casey) < paths.index(cold_paths[0])
    assert all("connected since" in p["reasons"][0] for p in cold_paths if "connected_on" in p)
    assert "messages_used" not in data


async def test_warm_paths_limit_and_former_employees(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as client:
        full = await call(client, "warm_paths", company="acme", limit=50)
        top = await call(client, "warm_paths", company="acme", limit=2)

    assert top["paths"] == full["paths"][:2]
    assert top["total_current_connections"] == len(WORLD.connections_at("acme", 2))
    v1_keys = {c.url: c.company_key for c in WORLD.connections(1) if c.url}
    moved_out = sorted(
        c.url for c in WORLD.connections(2) if v1_keys.get(c.url) == "acme" != c.company_key
    )
    assert moved_out
    assert urls(full["former_employees"]) == tuple(moved_out)
    assert full["total_former_employees"] == len(moved_out)
    for person in full["former_employees"]:
        assert person["company"] not in WORLD.company_groups["acme"]
        assert person["reasons"]


async def test_warm_paths_partial_and_unknown_company(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as client:
        partial = await call(client, "warm_paths", company="quill")
        unknown = await call(client, "warm_paths", company="Zzyzx Unheard Of")
        near_miss = await call(client, "warm_paths", company="Acme Rocketry Division")

    assert [m["match"] for m in partial["matched_companies"]] == ["prefix"]
    assert len(partial["paths"]) == len(WORLD.connections_at("quillfeather", 2))

    assert unknown == {"company": "Zzyzx Unheard Of", "matched_companies": [], "paths": []}
    assert near_miss["matched_companies"] == [] and near_miss["paths"] == []
    assert near_miss["suggestions"] == ["Acme"]


async def test_warm_paths_without_messages(mcp_client) -> None:
    """Owner C's export has no messages: ranking still works and says so."""
    async with mcp_client(TOKEN_C) as client:
        data = await call(client, "warm_paths", company="gift health", limit=50)
        summary = await call(client, "network_summary")
    assert summary["counts"]["messages"] == 0
    assert data["messages_used"] is False
    paths = data["paths"]
    assert len(paths) == len(WORLD.connections_at("gifthealth", 1))
    assert not any("message" in reason for p in paths for reason in p["reasons"])
    by_name = {p["name"]: p for p in paths}
    assert by_name["Casey Foxglove"]["reasons"][0] == "Wrote you a recommendation in 2023"
    assert by_name["Casey Foxglove"]["score"] >= 30
    assert paths[-1]["score"] == 0
    assert [p["score"] for p in paths] == sorted((p["score"] for p in paths), reverse=True)


# -- network_changes ------------------------------------------------------------


def assert_matches_truth(data: dict[str, Any]) -> None:
    changes = data["job_changes"]
    assert {c["change"] for c in changes} == {"company", "title", "company_and_title"}
    moved = [c for c in changes if c["change"] in ("company", "company_and_title")]
    retitled = [c for c in changes if c["change"] in ("title", "company_and_title")]
    assert urls(moved) == TRUTH.company_changed
    assert urls(retitled) == TRUTH.title_changed
    assert urls(data["company_spelling_changes"]) == TRUTH.company_renamed
    assert urls(data["new_connections"]) == TRUTH.added
    assert urls(data["removed_connections"]) == TRUTH.removed
    assert urls([*changes, *data["company_spelling_changes"]]) == TRUTH.job_changed
    assert data["counts"] == {
        "job_changes": len(changes),
        "company_moves": len(TRUTH.company_changed),
        "title_only_changes": len(changes) - len(TRUTH.company_changed),
        "company_spelling_changes": 1,
        "new_connections": len(TRUTH.added),
        "removed_connections": len(TRUTH.removed),
        "applied_companies": len(data["applied_companies"]),
    }


async def test_network_changes_default_window_matches_diff_truth(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as client:
        data = await call(client, "network_changes", limit=200)

    assert data["window"] == {
        "since": V1_AT.isoformat(),
        "latest_export": V2_AT.isoformat(),
        "imports_considered": 1,
        "imports_total": 2,
    }
    assert_matches_truth(data)

    before = {c.url: c for c in WORLD.connections(1)}
    after = {c.url: c for c in WORLD.connections(2)}
    for change in data["job_changes"]:
        url = change["profile_url"]
        assert change["from"] == {"company": before[url].company, "title": before[url].position}
        assert change["to"] == {"company": after[url].company, "title": after[url].position}
        assert change["as_of"] == V2_AT.date().isoformat()
        moved = before[url].company_key != after[url].company_key
        assert change["change"].startswith("company") == moved
    (renamed,) = data["company_spelling_changes"]
    assert renamed["from"] != renamed["to"]
    assert {renamed["from"], renamed["to"]} <= set(WORLD.company_groups["quillfeather"])
    for person in data["new_connections"]:
        assert person["first_seen"] == V2_AT.date().isoformat()
    for person in data["removed_connections"]:
        assert person["removed_as_of"] == V2_AT.date().isoformat()

    # No application after export v1 is at a company with current connections,
    # so the tool falls back to all applications and says so.
    assert data["applied_companies_scope"] == "all"
    current = {c.company_key for c in WORLD.connections(2)}
    expected = [a.company for a in WORLD.job_applications if a.company_key in current - {""}]
    assert [a["company"] for a in data["applied_companies"]] == expected
    for item in data["applied_companies"]:
        assert 1 <= len(item["connections"]) <= 3
        assert item["connection_count"] >= len(item["connections"])
        assert item["job_title"] and item["applied_at"]


@pytest.mark.parametrize("since", ["2020-01-01", "2025-05-10T12:00:00Z", "2026-11-02"])
async def test_network_changes_since_before_the_latest_export(mcp_client, since: str) -> None:
    """Any `since` before export v2 gives the same diff: export v1 is the baseline."""
    async with mcp_client(TOKEN_A) as client:
        data = await call(client, "network_changes", since=since, limit=200)
    assert_matches_truth(data)
    assert data["window"]["imports_considered"] == (2 if since == "2020-01-01" else 1)


async def test_network_changes_after_the_latest_export_is_empty(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as client:
        data = await call(client, "network_changes", since="2026-11-03")
    assert data["window"]["imports_considered"] == 0
    assert data["window"]["since"] == "2026-11-03T00:00:00+00:00"
    counts = dict(data["counts"])
    assert counts.pop("applied_companies") > 0  # the "all" fallback
    assert set(counts.values()) == {0}
    assert not {"job_changes", "new_connections", "removed_connections"} & set(data)


async def test_network_changes_limit_applies_per_list_and_counts_stay_total(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as client:
        data = await call(client, "network_changes", limit=2)
        full = await call(client, "network_changes", limit=200)
    assert data["counts"] == full["counts"]
    for key in ("job_changes", "new_connections", "removed_connections", "applied_companies"):
        assert data[key] == full[key][:2]
        assert len(full[key]) > 2


@pytest.mark.parametrize("since", ["yesterday", "2026-13-01", "11/02/2026"])
async def test_network_changes_rejects_an_invalid_date(mcp_client, since: str) -> None:
    async with mcp_client(TOKEN_A) as client:
        result = await client.call_tool("network_changes", {"since": since})
    assert result.is_error
    assert "Invalid `since`" in result.content[0].text and since in result.content[0].text
    assert "ISO date" in result.content[0].text


async def test_network_changes_with_a_single_export(mcp_client) -> None:
    async with mcp_client(TOKEN_C) as client:
        result = await client.call_tool("network_changes", {})
        early = await call(client, "network_changes", since="2000-01-01")
    assert result.content[0].text.startswith("Only one export has been imported")
    for data in (result.structured_content, early):
        assert data["window"]["imports_total"] == 1
        assert data["counts"]["new_connections"] == 0
        assert data["counts"]["job_changes"] == 0
        assert data["counts"]["removed_connections"] == 0
        assert "new_connections" not in data


# -- resource -------------------------------------------------------------------


async def test_imports_latest_resource_is_listed_and_per_owner(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as a, mcp_client(TOKEN_B) as b, mcp_client(TOKEN_C) as c:
        listed = {str(r.uri): r for r in (await a.list_resources()).resources}
        first_a = await read_json(a)
        from_b = await read_json(b)
        from_c = await read_json(c)
        second_a = await read_json(a)
        summary_a = await call(a, "network_summary")

    assert listed[URI].mime_type == "application/json"
    assert listed[URI].description

    assert first_a == second_a
    assert first_a["import_count"] == 2
    latest = first_a["latest_import"]
    assert latest["exported_at"] == V2_AT.isoformat()
    assert latest["imported_at"]
    assert len(latest["sha256_prefix"]) == 12
    assert latest["file_counts"]["connections"] == TRUTH.v2_count
    assert first_a["row_counts"] == {t: n for t, n in summary_a["counts"].items() if n}
    assert first_a["row_counts"]["imports"] == 2
    assert first_a["row_counts"]["connections"] == TRUTH.v1_count + len(TRUTH.added)

    assert from_b["import_count"] == 3
    assert from_b["latest_import"]["exported_at"] == B_ONLY_AT.isoformat()
    assert from_b["latest_import"]["sha256_prefix"] == "sha-b-only"
    assert from_b["row_counts"]["connections"] == first_a["row_counts"]["connections"] + 1
    assert from_b["row_counts"]["companies"] == first_a["row_counts"]["companies"] + 1

    assert from_c["import_count"] == 1
    assert from_c["latest_import"]["exported_at"] == V1_AT.isoformat()
    assert "messages" not in from_c["row_counts"]
    assert "owner_id" not in json.dumps([first_a, from_b, from_c])


async def test_resource_without_an_authenticated_request_fails(two_networks) -> None:
    from mcp import Client

    from kinship.server.app import build_mcp

    async with Client(build_mcp(two_networks.session_factory)) as client:
        with pytest.raises(Exception, match=r"(?i)owner|resource|error"):
            await client.read_resource(URI)


# -- prompt ---------------------------------------------------------------------


async def test_draft_intro_request_prompt_is_listed_and_renders(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as client:
        prompts = {p.name: p for p in (await client.list_prompts()).prompts}
        rendered = await client.get_prompt(
            "draft_intro_request",
            {"connection": " Casey Foxglove ", "job": "Staff Data Engineer at {Gift Health}"},
        )

    prompt = prompts["draft_intro_request"]
    assert prompt.description
    arguments = {a.name: a for a in prompt.arguments}
    assert set(arguments) == {"connection", "job"}
    assert all(a.required and a.description for a in arguments.values())

    (message,) = rendered.messages
    assert message.role == "user"
    text = message.content.text
    assert "Help me ask Casey Foxglove for an introduction" in text
    assert "Staff Data Engineer at {Gift Health}" in text
    for expected in ("`get_connection`", "`warm_paths`", "`match_job`", "Never invent", "easy out"):
        assert expected in text
    assert "{connection}" not in text and "{job}" not in text


# -- isolation ------------------------------------------------------------------


async def test_a_cannot_reach_b_data_through_the_tools(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as a, mcp_client(TOKEN_B) as b:
        # Interleaved on purpose.
        b_company = await call(b, "warm_paths", company=B_ONLY_COMPANY)
        a_company = await call(a, "warm_paths", company=B_ONLY_COMPANY)
        b_acme = await call(b, "warm_paths", company="acme", limit=50)
        a_acme = await call(a, "warm_paths", company="acme", limit=50)
        a_changes = await call(a, "network_changes", since="2020-01-01", limit=200)
        b_changes = await call(b, "network_changes", limit=200)
        a_latest = await call(a, "network_changes", since="2026-11-03")

    # B sees its own extra person, with B-only signals.
    assert urls(b_company["paths"]) == (B_ONLY_URL,)
    assert b_company["paths"][0]["reasons"][:2] == [
        "Wrote you a recommendation in 2026",
        "You exchanged 1 message on 2026-11-20",
    ]
    assert B_ONLY_URL in urls(b_acme["former_employees"])
    assert b_changes["window"]["since"] == V2_AT.isoformat()
    assert urls(b_changes["job_changes"]) == (B_ONLY_URL,)
    assert b_changes["applied_companies_scope"] == "since"
    assert [item["company"] for item in b_changes["applied_companies"]] == [B_ONLY_COMPANY]

    # A sees none of it, by name, in rankings, or in changes.
    assert a_company == {"company": B_ONLY_COMPANY, "matched_companies": [], "paths": []}
    everything_a = json.dumps([a_acme, a_changes, a_latest])
    assert B_ONLY_URL not in everything_a
    assert B_ONLY_COMPANY not in everything_a
    assert "Onlyb" not in everything_a
    assert_matches_truth(a_changes)
    assert a_acme["total_current_connections"] == len(WORLD.connections_at("acme", 2))
    assert a_acme["paths"] == [
        {**path, "score": a_path["score"]}
        for path, a_path in zip(b_acme["paths"], a_acme["paths"], strict=True)
    ]  # same people in the same order; only message recency differs (B's later export)
