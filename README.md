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

## Retrieval evaluation

`semantic_search` is measured, not assumed. [`evals/golden.jsonl`](evals/golden.jsonl)
holds 24 queries over the synthetic export, each with the recommendations,
positions or profile that should come back: 8 exact-keyword queries, 8
natural-language questions that share some words with their target, and 8
paraphrases that share almost none (see [`evals/README.md`](evals/README.md)).

```
make eval                                # fake embedder, offline
make eval ARGS="--embedder openai"       # a real model; needs OPENAI_API_KEY
uv run kinship eval --help               # --k, --format markdown|json, --min-recall, ...
```

The command needs a source checkout and a Postgres server it may create
databases on (`DATABASE_URL`). It creates a throwaway database, migrates it,
imports the synthetic export, embeds it, runs the queries in each retrieval
mode and drops the database again. It never writes to an existing database.

- **recall@5**: the share of queries with at least one expected item in the top 5.
- **MRR**: the mean of 1 / (rank of the first expected item), over the top 20.

Results with the fake embedder (this is what CI runs; a test keeps this table
in step with the code):

<!-- eval-results:fake:start -->
Embedder: `fake-hash-v1` · 24 queries · recall@5 · MRR over the top 20

| Queries | n | Full-text recall@5 | Full-text MRR | Vector recall@5 | Vector MRR | Hybrid recall@5 | Hybrid MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 24 | 0.75 | 0.75 | 0.75 | 0.70 | 0.75 | 0.75 |
| keyword | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| question | 8 | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 |
| paraphrase | 8 | 0.25 | 0.25 | 0.25 | 0.19 | 0.25 | 0.25 |
<!-- eval-results:fake:end -->

Read these numbers for what they are. The fake embedder is a hashed bag of
words, a lexical stand-in for a real model, so here "vector" search finds what
full-text search finds and hybrid adds nothing. Both fail on the paraphrases,
which is the gap a real embedding model is there to close. The synthetic corpus
is also small (20 chunks, one topic each), so any query that shares a word with
its target is easy.

Results for `text-embedding-3-small` are not measured yet (no API key has been
used with this harness so far). Run `make eval ARGS="--embedder openai --format markdown"`
and add the table here; it embeds 20 short chunks and 24 queries, which costs a
fraction of a cent. The same command compares any other model behind the
`Embedder` protocol.

CI runs the harness with `--min-recall 0.65`, so a change that pushes hybrid
recall@5 below that fails the build.

This repository contains no real LinkedIn data. Tests and demos use a synthetic
export.
