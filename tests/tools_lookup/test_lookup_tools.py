"""The three lookup tools end to end over MCP (HTTP), and isolation between tokens."""

from __future__ import annotations

import json

import pytest
from server.mcp_testing import TOKEN_A, TOKEN_B

from kinship.synth import V2_EXPORTED_AT

from .conftest import B_ONLY_COMPANY, B_ONLY_PEOPLE, WORLD

pytestmark = pytest.mark.anyio

TOOLS = ("find_connections_at_company", "search_connections", "get_connection")
TRUTH = WORLD.truth
V1 = {c.url: c for c in WORLD.connections(1) if c.url}
GIFTHEALTH = sorted(f"{c.first_name} {c.last_name}" for c in WORLD.connections_at("gifthealth", 2))


async def call(client, tool: str, **arguments) -> tuple[str, dict]:
    """Call a tool; return its one-line summary and structured data."""
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    summary, _, compact = result.content[0].text.partition("\n")
    assert json.loads(compact) == result.structured_content
    assert "\n" not in compact and ", " not in compact.split('"')[0]
    assert "owner_id" not in compact
    return summary, result.structured_content


def names(data: dict, key: str = "connections") -> list[str]:
    return sorted(person["name"] for person in data[key])


async def test_tools_are_listed_with_descriptions(mcp_client) -> None:
    async with mcp_client(TOKEN_A) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(TOOLS) <= set(tools)
    expected_parameters = {
        "find_connections_at_company": {"company", "include_former"},
        "search_connections": {"query", "title", "company", "limit"},
        "get_connection": {"name_or_url"},
    }
    for name in TOOLS:
        tool = tools[name]
        assert tool.description and len(tool.description) > 80, name
        assert tool.annotations.read_only_hint is True
        properties = tool.input_schema["properties"]
        assert set(properties) == expected_parameters[name]
        for parameter, schema in properties.items():
            assert len(schema.get("description", "")) > 20, f"{name}.{parameter}"
    assert tools["find_connections_at_company"].input_schema["required"] == ["company"]
    assert tools["get_connection"].input_schema["required"] == ["name_or_url"]
    assert "required" not in tools["search_connections"].input_schema


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_find_connections_at_company(mcp_client, mode) -> None:
    async with mcp_client(TOKEN_A, mode=mode) as client:
        summary, data = await call(client, "find_connections_at_company", company="gift health")
        _, acme = await call(
            client, "find_connections_at_company", company="ACME", include_former=True
        )
        typo_summary, typo = await call(client, "find_connections_at_company", company="Gifthelth")

    (company,) = data["matched_companies"]
    assert {company["display_name"], *company["aliases"]} == set(WORLD.company_groups["gifthealth"])
    assert names(data) == GIFTHEALTH
    assert data["current_count"] == len(GIFTHEALTH)
    assert "former_count" not in data and "suggestions" not in data
    assert summary == f"{len(GIFTHEALTH)} current connections at {company['display_name']}."
    assert set(data["connections"][0]) == {
        "name",
        "title",
        "company",
        "profile_url",
        "connected_on",
        "status",
    }

    moved_away = sorted(u for u in TRUTH.company_changed if V1[u].company_key == "acme")
    former = [p for p in acme["connections"] if p["status"] == "former"]
    assert moved_away and sorted(p["profile_url"] for p in former) == moved_away
    assert acme["former_count"] == len(former)
    assert acme["current_count"] == len(WORLD.connections_at("acme", 2))
    assert {p["left_by"] for p in former} == {V2_EXPORTED_AT.isoformat()}
    assert all(p["former_company"] in WORLD.company_groups["acme"] for p in former)

    assert typo == {
        "matched_companies": [],
        "suggestions": [company["display_name"]],
        "current_count": 0,
        "connections": [],
    }
    assert typo_summary == (
        'No company matching "Gifthelth" among your connections. '
        f"Similar companies: {company['display_name']}."
    )


async def test_search_connections(mcp_client) -> None:
    current = WORLD.connections(2)
    async with mcp_client(TOKEN_A) as client:
        summary, data = await call(client, "search_connections", query="engineer", limit=2)
        _, filtered = await call(
            client, "search_connections", title="engineer", company="gift health", limit=100
        )
        everyone_summary, everyone = await call(client, "search_connections")
        nobody_summary, nobody = await call(client, "search_connections", query="zzzzzz")

    engineers = [c for c in current if "engineer" in c.position.lower()]
    assert data["total"] == len(engineers) and len(data["connections"]) == 2
    assert summary == f"{len(engineers)} connections match; showing the first 2."
    assert set(data["connections"][0]) == {
        "name",
        "title",
        "company",
        "profile_url",
        "connected_on",
    }

    assert names(filtered) == sorted(
        f"{c.first_name} {c.last_name}" for c in engineers if c.company_key == "gifthealth"
    )
    assert [m["match"] for m in filtered["matched_companies"]] == ["exact"]

    assert everyone["total"] == len(current) and len(everyone["connections"]) == 20
    assert everyone_summary == f"{len(current)} connections match; showing the first 20."
    assert nobody == {"total": 0, "connections": []}
    assert nobody_summary == "No connections match."


async def test_get_connection(mcp_client) -> None:
    url = WORLD.messages[0].sender_url
    async with mcp_client(TOKEN_A) as client:
        summary, data = await call(client, "get_connection", name_or_url=url)
        _, by_name = await call(client, "get_connection", name_or_url="elio emberly")
        several_summary, several = await call(client, "get_connection", name_or_url="Elio")
        none_summary, none = await call(client, "get_connection", name_or_url="Elio Nobody")
        _, removed = await call(client, "get_connection", name_or_url=TRUTH.removed[0])

    assert data["match"] == "found" and by_name == data
    person = data["connection"]
    assert person["name"] == "Elio Emberly" and person["profile_url"] == url
    assert [role["as_of"] for role in person["role_history"]] == [
        WORLD.v1_exported_at.isoformat(),
        WORLD.v2_exported_at.isoformat(),
    ]
    assert person["message_count"] == 3
    assert person["endorsements"]["received"][0].keys() == {"skill", "date"}
    assert person["notes"][0].keys() == {"note", "date"}
    assert summary == (
        f"Elio Emberly: {person['title']} at {person['company']} "
        f"(connected {person['connected_on']}; 3 messages, last contact {person['last_contact']})."
    )

    elios = sorted(
        f"{c.first_name} {c.last_name}" for c in WORLD.connections(2) if c.first_name == "Elio"
    )
    assert several["match"] == "ambiguous" and names(several, "candidates") == elios
    assert "connection" not in several
    assert several_summary == (
        f'{len(elios)} connections match "Elio". Call get_connection again with the '
        "profile_url of the one you mean."
    )
    assert none["match"] == "none" and names(none, "suggestions") == elios
    assert none_summary.startswith('No connection matches "Elio Nobody". Similar names: Elio ')

    assert removed["connection"]["removed"] is True
    assert removed["connection"]["removed_on"] == V2_EXPORTED_AT.isoformat()


async def test_each_token_sees_only_its_own_connections(mcp_client) -> None:
    b_urls = sorted(p.profile_url for p in B_ONLY_PEOPLE)
    async with mcp_client(TOKEN_A) as a, mcp_client(TOKEN_B) as b:
        # Interleaved on purpose.
        _, at_a = await call(a, "find_connections_at_company", company="gift health")
        _, at_b = await call(b, "find_connections_at_company", company="gift health")
        _, own_a = await call(a, "find_connections_at_company", company=B_ONLY_COMPANY)
        _, own_b = await call(b, "find_connections_at_company", company=B_ONLY_COMPANY)

        _, search_a = await call(a, "search_connections", query="onlybee")
        _, search_b = await call(b, "search_connections", query="onlybee")
        _, company_a = await call(a, "search_connections", company=B_ONLY_COMPANY)
        _, all_a = await call(a, "search_connections", limit=100)
        _, all_b = await call(b, "search_connections", limit=100)

        get_a = [(await call(a, "get_connection", name_or_url=url))[1] for url in b_urls]
        get_b = [(await call(b, "get_connection", name_or_url=url))[1] for url in b_urls]
        _, name_a = await call(a, "get_connection", name_or_url="Bryony Onlybee")
        _, part_a = await call(a, "get_connection", name_or_url="onlybee")

    assert names(at_a) == GIFTHEALTH
    assert names(at_b) == sorted([*GIFTHEALTH, "Beatrix Onlybee", "Bertram Onlybee"])
    assert own_a == {"matched_companies": [], "current_count": 0, "connections": []}
    assert names(own_b) == ["Bryony Onlybee"]

    assert search_a == {"total": 0, "connections": []} == company_a
    assert search_b["total"] == 3
    assert all_a["total"] == len(WORLD.connections(2))
    assert all_b["total"] == len(WORLD.connections(2)) + 3
    assert not any("Onlybee" in person["name"] for person in all_a["connections"])

    assert get_a == [{"match": "none"}] * 3
    assert [found["connection"]["profile_url"] for found in get_b] == b_urls
    assert name_a == {"match": "none"} == part_a
