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

This repository contains no real LinkedIn data. Tests and demos use a synthetic
export.
