# Kinship

Your LinkedIn network, available to AI assistants.

Kinship imports a LinkedIn data export into Postgres and serves it through a
hosted [Model Context Protocol](https://modelcontextprotocol.io) server. An
assistant can ask who you know at a company, find warm introduction paths, and
search your recommendations and work history to tailor a job application.

**Status:** early prototype. See [docs/DESIGN.md](docs/DESIGN.md) and the
[issues](https://github.com/naryga/kinship/issues).

## Stack

Python · official MCP SDK (FastMCP, Streamable HTTP) · Postgres + pgvector ·
hybrid retrieval (full-text + vectors) · Fly.io

## Development

```
docker compose up -d db
cp .env.example .env
uv sync
uv run pytest
```

## Run the MCP server locally

Configuration comes from the environment (or `.env`):

| Variable | Meaning |
|---|---|
| `DATABASE_URL` | Postgres URL, e.g. `postgresql+psycopg://kinship:kinship@localhost:5432/kinship` |
| `KINSHIP_TOKENS` | Comma-separated `token:owner_id` pairs, e.g. `dev-token:1`. With none set, every MCP request is refused. |

```
uv run alembic upgrade head
uv run kinship serve            # http://127.0.0.1:8000, --host / --port to change
curl http://127.0.0.1:8000/healthz
```

The MCP endpoint is `/mcp` (Streamable HTTP) and needs
`Authorization: Bearer <token>`. To connect Claude Code:

```
claude mcp add --transport http kinship http://127.0.0.1:8000/mcp \
  --header "Authorization: Bearer dev-token"
```

Each token sees only its own owner's data. Tools read through the repository
layer in `src/kinship/repo/`, which always filters by owner.

This repository contains no real LinkedIn data. Tests and demos use a synthetic
export.
