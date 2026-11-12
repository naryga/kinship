# Kinship — Design

Kinship loads a person's LinkedIn data export into Postgres and serves it to AI
assistants through a hosted, remote MCP server. Assistants can ask "who do I know
at this company?", find warm introduction paths, and run retrieval (RAG) over
recommendations, positions and messages to tailor job applications.

This file is the source of truth for the prototype. GitHub Issues track the work.

## Goals

- **Prototype (v0/v1):** one real user (Nathan), hosted, usable from Claude as a
  remote MCP connector, with a measured RAG layer.
- **Later (v2):** multi-user. A shared MCP server where each user can only see
  their own network. User model, OAuth 2.1, Postgres row-level security.

Non-goals: scraping LinkedIn, writing back to LinkedIn, a web UI.

## Decisions

| Area | Decision | Why |
|---|---|---|
| Language | Python 3.12+, managed with `uv` | Default ecosystem for both MCP and RAG |
| MCP | Official `mcp` Python SDK (`FastMCP`), Streamable HTTP transport | Remote, hosted server |
| Database | Postgres 16+ with `pgvector` | Real DB for a hosted service; FTS + vectors in one place |
| Data access | SQLAlchemy 2.x + Alembic migrations, psycopg 3 | Typed, migrated, testable |
| Embeddings | OpenAI `text-embedding-3-small` (1536-d) behind an `Embedder` protocol | Cheapest good option; swappable. A deterministic fake embedder is used in tests |
| Hosting | Fly.io app; Postgres on Neon or Fly (pending Nathan's pick) | |
| Tracking | GitHub Issues + milestones | |
| Quality | `ruff` (lint + format), `pytest`, CI on every PR | |

## Multi-tenancy from day one

Even though v0 has one user, **every owned table carries `owner_id`**, and
**every query goes through a repository layer that requires an owner**. No query
in tool code may touch tables directly. That makes v2 an auth change, not a data
migration. In v2, Postgres row-level security adds a second, database-level guard.

v0 auth: requests carry a bearer token; the server maps token → `owner_id`
(configured via the `KINSHIP_TOKENS` secret). See the auth spike issue: the
client is a claude.ai Project, and its custom-connector auth options decide
whether v0 can use a static token or needs OAuth sooner.

## Data model

Every table below except `owners` has `owner_id` (FK) and is indexed on it.

- `owners(id, display_name, created_at)`
- `imports(id, owner_id, source_sha256, exported_at, imported_at, file_counts jsonb)`
  — unique on `(owner_id, source_sha256)`; re-importing the same export is a no-op.
- `companies(id, owner_id, name_normalized, display_name, aliases jsonb)`
- `connections(id, owner_id, profile_url, first_name, last_name, email, company_raw,
  company_id, position, connected_on, first_seen_import_id, last_seen_import_id,
  removed_at)` — unique on `(owner_id, profile_url)`.
- `connection_history(id, owner_id, connection_id, import_id, company_raw, position)`
- `messages(id, owner_id, conversation_id, from_name, to_names, sent_at, subject,
  body, connection_id NULL)`
- `my_positions(id, owner_id, company, title, started_on, ended_on, description)`
- `endorsements(...)`, `recommendations(...)` — received; text kept for embedding.
- `chunks(id, owner_id, source_table, source_id, text, tsv tsvector,
  embedding vector, model, dim, created_at)`

## Import

Input: a LinkedIn export `.zip` or an extracted directory.

- **Idempotent.** Keyed on the export's SHA-256 per owner; the same file twice
  does nothing. Rows are upserted on natural keys (`profile_url`, falling back to
  name + `connected_on`).
- **Diffing between exports.** New connections are inserted; a changed company or
  title writes a `connection_history` row; a connection missing from a newer
  export gets `removed_at` (never deleted).
- **Order-tolerant.** Exports may be imported in any order; `exported_at` (from
  file metadata or the CSVs) decides what is "newer".
- **CSV quirks.** LinkedIn files can start with a "Notes:" preamble before the
  header, use mixed date formats, and have blank emails. The parser handles all
  three.
- CLI: `kinship import PATH --owner OWNER`. Run locally against any
  `DATABASE_URL`, including production.

An older full export can be imported first; the newer one becomes import #2 and
exercises the diff path with real data.

## MCP surface

All tools are read-only and owner-scoped. Each returns compact JSON plus a short
text summary.

1. `find_connections_at_company(company, include_former=false)`
2. `search_connections(query, title?, company?, limit)`
3. `get_connection(name_or_url)` — role history, last contact, endorsements
4. `warm_paths(company)` — ranked by message recency, endorsements and
   recommendations, and shared past employers
5. `network_changes(since)` — job changes and new connections
6. `semantic_search(query, sources, k)` — the RAG retrieval tool
7. `match_job(description_text)` — relevant parts of the owner's history plus
   connections near the role

Resource: `kinship://imports/latest`. Prompt: `draft_intro_request(connection, job)`.

## RAG

- **Chunking:** one chunk per recommendation and per position; messages grouped
  by conversation, capped near 500 tokens.
- **Message bodies are not embedded by default.** Embedding them requires an
  explicit flag.
- **Retrieval:** hybrid. Postgres full-text search (`ts_rank`) and pgvector
  cosine similarity, merged with reciprocal rank fusion.
- **Evaluation:** a golden set of ~20 queries over the synthetic dataset, with
  `recall@5` reported by `make eval` and run in CI. Model swaps (for example
  Voyage) are compared with this harness.

## Privacy

- The real export contains other people's data. **It is never committed.**
  `data/`, `*.zip` and `*.csv` outside `tests/fixtures/` are gitignored.
- The public repo ships code plus a **synthetic** export generator and fixtures.
- Production secrets live in Fly and GitHub Actions secrets only.

## Milestones

1. **Foundation** — scaffold, CI, local Postgres (done in the initial commit)
2. **Ingest** — schema, synthetic data, parser, idempotent import, companies
3. **MCP** — server, owner scoping, tools
4. **RAG** — embeddings, hybrid retrieval, `match_job`, eval
5. **Ship** — auth spike, Fly deploy, README
6. **v2** — multi-user OAuth + RLS (later)
