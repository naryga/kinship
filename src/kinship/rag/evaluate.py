"""Retrieval evaluation: a golden set of queries scored against `kinship.rag.search`.

    golden = load_golden(default_golden_path())
    report = run_eval(repo, embedder, golden, k=5)
    print(report.to_text())

The golden set (`evals/golden.jsonl`) is written against the synthetic export
(seed 42, version 2). Each line is one query:

    {"id": "p01", "kind": "paraphrase", "query": "...",
     "expected": [{"source": "recommendations", "direction": "received",
                   "name": "Freya Rookwood"}],
     "sources": ["recommendations"]}        <- optional search filter

Expected items are named by natural keys, never database ids, and are resolved
to `(source_table, source_id)` through the owner-bound `Repository`:

- `{"source": "recommendations", "direction": "received"|"given", "name": "<first last>"}`
- `{"source": "my_positions", "company": "<company>", "title": "<title>"}`
- `{"source": "profiles"}` (the owner's one profile)

An item that matches no row, or more than one, is an error: the generator and
the golden set have drifted apart.

Metrics, per retrieval mode (`fts`, `vector`, `hybrid`):

- **recall@k**: the fraction of queries with *at least one* expected item in
  the top `k` results (also called hit rate; with one expected item per query
  it is plain recall).
- **mean recall@k**: the mean over queries of (expected items in the top `k`) /
  (expected items). It is lower than recall@k only where a query expects
  several items.
- **MRR**: the mean over queries of 1 / (rank of the first expected item),
  looking at the top `mrr_depth` results; 0 for a query with none there.
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kinship.rag.chunking import POSITIONS, PROFILES, RECOMMENDATIONS, SOURCE_TABLES
from kinship.rag.embedder import Embedder
from kinship.rag.search import MODES, search

if TYPE_CHECKING:
    from datetime import datetime

    from kinship.repo import Repository

KINDS = ("keyword", "question", "paraphrase")
DEFAULT_MODES = ("fts", "vector", "hybrid")
DEFAULT_K = 5
MRR_DEPTH = 20
MODE_LABELS = {"fts": "Full-text", "vector": "Vector", "hybrid": "Hybrid"}

SYNTH_SEED = 42
SYNTH_VERSION = 2

_ITEM_FIELDS = {
    RECOMMENDATIONS: ("direction", "name"),
    POSITIONS: ("company", "title"),
    PROFILES: (),
}

SourceKey = tuple[str, int]


class GoldenError(ValueError):
    """The golden set is malformed, or does not match the data it is run against."""


# -- golden set ---------------------------------------------------------------


@dataclass(frozen=True)
class GoldenItem:
    """One relevant row, named by a natural key (see the module docstring)."""

    source: str
    direction: str | None = None
    name: str | None = None
    company: str | None = None
    title: str | None = None

    def describe(self) -> str:
        fields = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(fields, ensure_ascii=False)


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    kind: str
    query: str
    expected: tuple[GoldenItem, ...]
    sources: tuple[str, ...] | None = None


def default_golden_path() -> Path:
    """`evals/golden.jsonl` in the source checkout."""
    return Path(__file__).resolve().parents[3] / "evals" / "golden.jsonl"


def _parse_item(raw: Any, where: str) -> GoldenItem:
    if not isinstance(raw, dict):
        raise GoldenError(f"{where}: an expected item must be an object")
    source = raw.get("source")
    if source not in _ITEM_FIELDS:
        raise GoldenError(
            f"{where}: source must be one of {', '.join(_ITEM_FIELDS)}, got {source!r}"
        )
    wanted = _ITEM_FIELDS[source]
    if set(raw) != {"source", *wanted}:
        raise GoldenError(
            f"{where}: a {source} item needs exactly the fields "
            f"{', '.join(('source', *wanted))}; got {', '.join(sorted(raw))}"
        )
    for name in wanted:
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise GoldenError(f"{where}: {name} must be a non-empty string")
    if source == RECOMMENDATIONS and raw["direction"] not in ("received", "given"):
        raise GoldenError(f"{where}: direction must be 'received' or 'given'")
    return GoldenItem(**raw)


def parse_golden(lines: Iterable[str], *, name: str = "golden set") -> list[GoldenQuery]:
    """Parse JSON Lines into queries; blank lines are skipped. Raises `GoldenError`."""
    queries: list[GoldenQuery] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{name}, line {number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise GoldenError(f"{where}: not valid JSON ({error.msg})") from error
        if not isinstance(raw, dict):
            raise GoldenError(f"{where}: each line must be a JSON object")
        unknown = set(raw) - {"id", "kind", "query", "expected", "sources"}
        if unknown:
            raise GoldenError(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")
        for key in ("id", "query"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise GoldenError(f"{where}: {key} must be a non-empty string")
        if raw["id"] in seen:
            raise GoldenError(f"{where}: duplicate id {raw['id']!r}")
        seen.add(raw["id"])
        if raw.get("kind") not in KINDS:
            raise GoldenError(f"{where}: kind must be one of {', '.join(KINDS)}")
        expected = raw.get("expected")
        if not isinstance(expected, list) or not expected:
            raise GoldenError(f"{where}: expected must be a non-empty list")
        items = tuple(_parse_item(item, where) for item in expected)
        if len(set(items)) != len(items):
            raise GoldenError(f"{where}: expected lists the same item twice")
        sources = raw.get("sources")
        if sources is not None:
            if not isinstance(sources, list) or not sources:
                raise GoldenError(f"{where}: sources must be a non-empty list when given")
            bad = [s for s in sources if s not in SOURCE_TABLES]
            if bad:
                raise GoldenError(f"{where}: unknown source(s) {', '.join(map(repr, bad))}")
            outside = [item.source for item in items if item.source not in sources]
            if outside:
                raise GoldenError(
                    f"{where}: expects a {outside[0]} item but searches only {', '.join(sources)}"
                )
        queries.append(
            GoldenQuery(
                id=raw["id"],
                kind=raw["kind"],
                query=raw["query"].strip(),
                expected=items,
                sources=tuple(sources) if sources else None,
            )
        )
    if not queries:
        raise GoldenError(f"{name}: no queries")
    return queries


def load_golden(path: Path) -> list[GoldenQuery]:
    """Read and validate a golden set file (JSON Lines)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise GoldenError(f"cannot read golden set {path}: {error.strerror or error}") from error
    return parse_golden(text.splitlines(), name=path.name)


def _norm(value: str | None) -> str:
    return " ".join((value or "").split()).casefold()


def resolve_golden(repo: Repository, golden: Sequence[GoldenQuery]) -> dict[str, set[SourceKey]]:
    """Query id -> the `(source_table, source_id)` pairs of its expected items.

    Rows are read through `repo`, so only its owner's rows can match. Raises
    `GoldenError` naming every item that matches no row or more than one.
    """
    rows = repo.chunk_sources()
    index: dict[GoldenItem, list[int]] = {}
    for rec in rows.recommendations:
        name = " ".join(filter(None, [_norm(rec.first_name), _norm(rec.last_name)]))
        key = GoldenItem(RECOMMENDATIONS, direction=rec.direction, name=name)
        index.setdefault(key, []).append(rec.id)
    for position in rows.positions:
        key = GoldenItem(POSITIONS, company=_norm(position.company), title=_norm(position.title))
        index.setdefault(key, []).append(position.id)
    if rows.profile is not None:
        index[GoldenItem(PROFILES)] = [rows.profile.id]

    resolved: dict[str, set[SourceKey]] = {}
    problems: list[str] = []
    for query in golden:
        keys: set[SourceKey] = set()
        for item in query.expected:
            lookup = GoldenItem(
                item.source,
                direction=item.direction,
                name=_norm(item.name) if item.name is not None else None,
                company=_norm(item.company) if item.company is not None else None,
                title=_norm(item.title) if item.title is not None else None,
            )
            ids = index.get(lookup, [])
            if len(ids) == 1:
                keys.add((item.source, ids[0]))
            else:
                found = "no row" if not ids else f"{len(ids)} rows"
                problems.append(f"{query.id}: {item.describe()} matches {found}")
        resolved[query.id] = keys
    if problems:
        raise GoldenError(
            "The golden set does not match this owner's data (was it written for another "
            "export, or did the synthetic generator change?):\n  " + "\n  ".join(problems)
        )
    return resolved


# -- metrics (pure) -----------------------------------------------------------


def first_relevant_rank(ranking: Sequence[Hashable], relevant: Iterable[Hashable]) -> int | None:
    """1-based position of the first relevant item in `ranking`, or None."""
    wanted = set(relevant)
    for position, item in enumerate(ranking, start=1):
        if item in wanted:
            return position
    return None


def recall_at_k(ranking: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """The fraction of the relevant items that appear in the first `k` of `ranking`."""
    wanted = set(relevant)
    if not wanted:
        raise ValueError("relevant must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")
    return len(wanted.intersection(ranking[:k])) / len(wanted)


def reciprocal_rank(rank: int | None) -> float:
    """1 / rank, or 0 when there was no relevant item."""
    return 0.0 if rank is None else 1.0 / rank


@dataclass(frozen=True)
class Metrics:
    """Aggregates over `queries` queries; see the module docstring for the definitions."""

    queries: int
    recall_at_k: float
    mean_recall: float
    mrr: float


@dataclass(frozen=True)
class QueryOutcome:
    """One query's result per mode.

    `ranks[mode]` is the rank of the first expected item within the top
    `mrr_depth` results (None if it is not there); `recall[mode]` is the
    fraction of the expected items in the top `k`.
    """

    id: str
    kind: str
    query: str
    expected: int
    ranks: dict[str, int | None]
    recall: dict[str, float]


def summarize(outcomes: Sequence[QueryOutcome], mode: str, k: int) -> Metrics:
    """Aggregate `outcomes` for one mode. With no outcomes every metric is 0."""
    n = len(outcomes)
    if n == 0:
        return Metrics(queries=0, recall_at_k=0.0, mean_recall=0.0, mrr=0.0)
    ranks = [outcome.ranks[mode] for outcome in outcomes]
    return Metrics(
        queries=n,
        recall_at_k=sum(1 for rank in ranks if rank is not None and rank <= k) / n,
        mean_recall=sum(outcome.recall[mode] for outcome in outcomes) / n,
        mrr=sum(reciprocal_rank(rank) for rank in ranks) / n,
    )


# -- report -------------------------------------------------------------------


@dataclass(frozen=True)
class EvalReport:
    """Metrics per mode, overall and by query kind, plus the per-query ranks."""

    embedder: str
    k: int
    mrr_depth: int
    modes: tuple[str, ...]
    overall: dict[str, Metrics]
    by_kind: dict[str, dict[str, Metrics]]
    queries: list[QueryOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def _rows(self) -> list[tuple[str, dict[str, Metrics]]]:
        return [("all", self.overall), *self.by_kind.items()]

    def to_markdown(self) -> str:
        """The results table (as pasted into the README), with a one-line caption."""
        header = ["Queries", "n"]
        for mode in self.modes:
            label = MODE_LABELS.get(mode, mode)
            header += [f"{label} recall@{self.k}", f"{label} MRR"]
        lines = [
            f"Embedder: `{self.embedder}` · {len(self.queries)} queries · "
            f"recall@{self.k} · MRR over the top {self.mrr_depth}",
            "",
            "| " + " | ".join(header) + " |",
            "|---|" + "---:|" * (len(header) - 1),
        ]
        for name, metrics in self._rows():
            cells = [name, str(next(iter(metrics.values())).queries)]
            for mode in self.modes:
                cells += [f"{metrics[mode].recall_at_k:.2f}", f"{metrics[mode].mrr:.2f}"]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def to_text(self) -> str:
        """For the terminal: the summary, mean recall, and the rank of each query's first hit."""
        lines = [
            f"Embedder: {self.embedder}   queries: {len(self.queries)}   "
            f"k: {self.k}   MRR depth: {self.mrr_depth}",
            "",
        ]
        head = f"{'queries':<12}{'n':>4}"
        for mode in self.modes:
            head += f"{mode + ' r@' + str(self.k):>14}{mode + ' MRR':>12}"
        lines.append(head)
        for name, metrics in self._rows():
            row = f"{name:<12}{next(iter(metrics.values())).queries:>4}"
            for mode in self.modes:
                row += f"{metrics[mode].recall_at_k:>14.2f}{metrics[mode].mrr:>12.2f}"
            lines.append(row)
        lines += [
            "",
            f"Mean per-query recall@{self.k} (share of expected items found): "
            + ", ".join(f"{mode} {self.overall[mode].mean_recall:.2f}" for mode in self.modes),
            "",
            f"Rank of the first expected item (- = not in the top {self.mrr_depth}):",
            f"{'id':<6}{'kind':<12}" + "".join(f"{mode:>8}" for mode in self.modes) + "  query",
        ]
        for outcome in self.queries:
            ranks = "".join(f"{outcome.ranks[mode] or '-':>8}" for mode in self.modes)
            lines.append(f"{outcome.id:<6}{outcome.kind:<12}{ranks}  {outcome.query}")
        lines += [f"Note: {note}" for note in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# -- running ------------------------------------------------------------------


def run_eval(
    repo: Repository,
    embedder: Embedder | None,
    golden: Sequence[GoldenQuery],
    *,
    k: int = DEFAULT_K,
    modes: Sequence[str] = DEFAULT_MODES,
    mrr_depth: int = MRR_DEPTH,
) -> EvalReport:
    """Run every golden query in each mode against `repo`'s owner and score the results.

    The owner's chunks must already be embedded with `embedder` (`sync_chunks`)
    for the `vector` and `hybrid` modes; `embedder` may be None for `fts` alone.
    Raises `GoldenError` when an expected item does not resolve, and
    `ValueError` for a bad `k`, an unknown mode, or missing vectors.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    modes = tuple(dict.fromkeys(modes))
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown or not modes:
        raise ValueError(f"modes must be some of {', '.join(MODES)}; got {list(modes)}")
    needs_vectors = any(mode != "fts" for mode in modes)
    if needs_vectors and embedder is None:
        raise ValueError("the vector and hybrid modes need an embedder")
    if needs_vectors and repo.embedded_count(embedder.model) == 0:  # type: ignore[union-attr]
        raise ValueError(
            f"None of this owner's chunks has a vector from {embedder.model}; "  # type: ignore[union-attr]
            "run sync_chunks with the same embedder first."
        )
    relevant = resolve_golden(repo, golden)
    depth = max(k, mrr_depth)

    outcomes: list[QueryOutcome] = []
    notes: list[str] = []
    for query in golden:
        ranks: dict[str, int | None] = {}
        recall: dict[str, float] = {}
        for mode in modes:
            result = search(
                repo,
                embedder,
                query.query,
                sources=query.sources,
                k=depth,
                mode=mode,  # type: ignore[arg-type]
            )
            if result.note:
                notes.append(f"{query.id} ({mode}): {result.note}")
            ranking = list(dict.fromkeys((hit.source_table, hit.source_id) for hit in result.hits))
            ranks[mode] = first_relevant_rank(ranking, relevant[query.id])
            recall[mode] = recall_at_k(ranking, relevant[query.id], k)
        outcomes.append(
            QueryOutcome(
                id=query.id,
                kind=query.kind,
                query=query.query,
                expected=len(relevant[query.id]),
                ranks=ranks,
                recall=recall,
            )
        )

    by_kind = {
        kind: {mode: summarize(subset, mode, k) for mode in modes}
        for kind in KINDS
        if (subset := [outcome for outcome in outcomes if outcome.kind == kind])
    }
    return EvalReport(
        embedder=embedder.model if embedder is not None else "none",
        k=k,
        mrr_depth=depth,
        modes=modes,
        overall={mode: summarize(outcomes, mode, k) for mode in modes},
        by_kind=by_kind,
        queries=outcomes,
        notes=notes,
    )


def synthetic_exported_at() -> datetime:
    """The export time used for the generated synthetic export (noon UTC on its date)."""
    from datetime import UTC, datetime, time

    from kinship.synth import V2_EXPORTED_AT

    return datetime.combine(V2_EXPORTED_AT, time(12), tzinfo=UTC)


def evaluate_in_scratch_database(
    embedder: Embedder,
    golden: Sequence[GoldenQuery],
    *,
    k: int = DEFAULT_K,
    export_path: Path | None = None,
    server_url: str | None = None,
) -> EvalReport:
    """Import an export into a throwaway database, embed it and run the golden set.

    A uniquely named database is created on `server_url` (default: the
    configured `DATABASE_URL`), migrated, and always dropped again; no existing
    database is written to. Without `export_path` the synthetic export the
    golden set was written for (seed 42, version 2) is generated in a temporary
    directory. Needs a source checkout (see `kinship.db.scratch`).
    """
    import tempfile

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from kinship.db.models import Owner
    from kinship.db.scratch import scratch_database
    from kinship.ingest import import_export
    from kinship.parser import parse_export
    from kinship.rag.indexer import sync_chunks
    from kinship.repo import Repository
    from kinship.synth import generate_export

    with tempfile.TemporaryDirectory(prefix="kinship-eval-") as tmp:
        if export_path is None:
            generated = generate_export(
                Path(tmp) / "export", seed=SYNTH_SEED, version=SYNTH_VERSION
            )
            parsed = parse_export(generated, exported_at=synthetic_exported_at())
        else:
            parsed = parse_export(export_path)

        with scratch_database(server_url, prefix="kinship_eval") as url:
            engine = create_engine(url)
            try:
                with Session(engine) as session:
                    owner = Owner(display_name="Eval owner")
                    session.add(owner)
                    session.flush()
                    import_export(session, owner.id, parsed)
                    sync_chunks(session, owner.id, embedder)
                    session.flush()
                    return run_eval(Repository(session, owner.id), embedder, golden, k=k)
            finally:
                engine.dispose()
