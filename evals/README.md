# Retrieval golden set

`golden.jsonl` is the query set behind `make eval` (`kinship eval`). It is
written against the synthetic export (`kinship synth`, seed 42, version 2; the
same data as `tests/fixtures/synthetic/export_v2`). Everything in it is fake.

## Format

One JSON object per line:

```json
{"id": "p01", "kind": "paraphrase",
 "query": "Who can vouch for my performance tuning of batch jobs?",
 "expected": [{"source": "recommendations", "direction": "received", "name": "Freya Rookwood"}]}
```

| Field | Meaning |
|---|---|
| `id` | Unique, short. The prefix follows the kind (`k`, `q`, `p`). |
| `kind` | `keyword`, `question` or `paraphrase` (below). |
| `query` | The text passed to `kinship.rag.search.search`. |
| `expected` | 1–3 relevant items. A query counts as answered when any one of them is in the top k. |
| `sources` | Optional. Limits the search to these source tables, as the `semantic_search` tool can. |

Expected items are named by natural keys, never by database ids, and are looked
up at run time through the owner-scoped repository:

- `{"source": "recommendations", "direction": "received" | "given", "name": "First Last"}`
- `{"source": "my_positions", "company": "...", "title": "..."}`
- `{"source": "profiles"}` (the owner's profile)

Matching ignores case and extra spaces. An item that matches no row, or more
than one, stops the run: it means the generator and this file have drifted
apart. Message chunks are out of scope, because messages are not embedded by
default.

## Query kinds

- **keyword** — the words of the target, as someone would type into a search box.
- **question** — a natural-language question that shares some words with the target.
- **paraphrase** — the same need in different words, with little or no overlap
  with the target text. Full-text search and the fake embedder are expected to
  miss most of these; they are here to show what a real embedding model adds.

## Changing it

Write the query a person would ask, then record what should come back. Do not
reword a query because a retriever misses it: the misses are the measurement.
After a change, run `make eval ARGS="--format markdown"`, paste the table into
the "Retrieval evaluation" section of the top-level README (a test compares
them), and check the floor in `.github/workflows/ci.yml` still makes sense.
