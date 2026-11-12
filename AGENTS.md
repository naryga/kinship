# Agent guide for Kinship

Read `docs/DESIGN.md` before starting any issue. It is the source of truth.

## Rules

- **Owner scoping is mandatory.** Every owned table has `owner_id`. Tool and
  service code reads data only through the repository layer, which requires an
  owner. Never write a query that can return another owner's rows. Add a test
  for owner isolation whenever you add a query.
- **Never commit real LinkedIn data.** Use the synthetic generator and
  `tests/fixtures/`. If you see a real export in the working tree, stop.
- **No secrets in code.** Configuration comes from environment variables
  (see `.env.example`).
- Tests must not call paid APIs. Use the fake embedder.

## Workflow

- One issue per branch: `issue-<number>-<short-slug>`.
- Open a PR that says `Closes #<number>` and lists what you verified.
- Before pushing, run:
  ```
  uv run ruff check .
  uv run ruff format --check .
  uv run pytest
  ```
- Keep PRs small. If an issue grows, split it and note the new issue.

## Local setup

```
docker compose up -d db
cp .env.example .env
uv sync
uv run pytest
```
