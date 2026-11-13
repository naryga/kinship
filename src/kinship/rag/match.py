"""`match_job`: what in the owner's data is close to a job description.

    from kinship.rag.match import match_job

    result = match_job(repo, embedder, pasted_job_description)
    for item in result.history:
        print(item.title, "-", item.why)

Two parts, both read through the owner-bound `Repository`:

1. **History.** The owner's positions, recommendations and profile that best
   fit the description (never messages). A job description is long and full of
   boilerplate, so it is not searched as one query. `analyse` extracts its
   salient terms; one *focused* query (the title plus the most frequent terms)
   and one query per substantial section are each run through hybrid
   `kinship.rag.search.search`, and the rankings are merged with reciprocal
   rank fusion. Every item says why it matched: the description's terms that
   really occur in the chunk, and which retrievers found it.
2. **Connections.** People near the role, in three groups: at the company now,
   with a lexically similar title anywhere, and formerly at the company.

The role's title and company come from the caller or are inferred from the
text. The company is only ever matched against the owner's own companies.

Everything lexical here is deliberately cheap: a tiny suffix stemmer, stop-word
lists and token overlap. It imports nothing from the server.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kinship.rag.chunking import POSITIONS, PROFILES, RECOMMENDATIONS
from kinship.rag.embedder import Embedder
from kinship.rag.fusion import reciprocal_rank_fusion
from kinship.rag.search import SearchHit, search

if TYPE_CHECKING:
    from kinship.repo import Repository
    from kinship.repo.network import NetworkPerson

HISTORY_SOURCES = (POSITIONS, RECOMMENDATIONS, PROFILES)
"""What `match_job` searches. Messages are never part of it."""

MAX_DESCRIPTION_CHARS = 20_000
DEFAULT_K = 5
DEFAULT_GROUP_LIMIT = 5
MAX_FOCUSED_TERMS = 24
MAX_SECTION_TERMS = 24
MAX_SECTIONS = 4
MIN_SECTION_TERMS = 6
CANDIDATES_PER_QUERY = 20
MAX_EVIDENCE_TERMS = 6
MAX_EVIDENCE_PHRASES = 3
PROMINENT_TERMS = 30
MIN_REPEATS = 3
MAX_MISSING_TERMS = 8
MAX_TITLE_CHARS = 90
MAX_TITLE_WORDS = 10
MAX_OTHER_COMPANIES = 4

MISSING_TERMS_NOTE = (
    "Heuristic: prominent words of the description that appear nowhere in your positions, "
    "recommendations, profile or skills. Some are gaps, some are just the employer's wording."
)

_WORD = re.compile(r"[^\W_]+")


def _wordset(text: str) -> frozenset[str]:
    return frozenset(text.split())


SENIORITY_WORDS = _wordset(
    "senior sr junior jr staff principal lead leading associate assistant intern internship "
    "trainee apprentice graduate entry mid level chief head deputy i ii iii iv v 1 2 3"
)
"""Words that say how senior a role is, not what it is."""

GENERIC_ROLE_WORDS = _wordset(
    "engineer developer manager analyst specialist consultant officer director coordinator "
    "administrator executive representative member technician partner professional expert "
    "worker person team"
)
"""Role nouns shared by unrelated jobs: sharing only these does not make titles similar."""

_ROLE_NOUNS = GENERIC_ROLE_WORDS | _wordset(
    "architect scientist designer researcher writer recruiter auditor programmer strategist "
    "accountant advocate lead head intern president vp cto ceo cfo coo founder editor "
    "marketer owner planner operator supervisor trainer nurse teacher counsel attorney "
    "buyer agent clerk producer sre"
)

STOP_WORDS = _wordset(
    """
    a about above across after again against all also am an and any are aren as at be because
    been before being below between both but by can cannot could did do does doing don down
    during each either etc few for from further had has have having he her here hers him his
    how i if in into is isn it its itself just ll me more most must my no nor not now of off
    on once only or other our ours out over own per re s same she should so some such t than
    that the their theirs them then there these they this those through to too under until up
    us ve very via was we were what when where which while who whom why will with within
    without would you your yours
    """
)

BOILERPLATE_WORDS = _wordset(
    """
    ability able about accommodation applicant applicants application apply applying based
    benefit benefits bonus candidate candidates career careers company compensation
    competitive day days description desired diverse diversity employee employees employer
    employment environment equal equity etc excellent experience experienced experiences
    fast following full good great help hire hiring holiday ideal ideally include included
    includes including insurance job join knowledge like looking love make minimum month
    months need needs new nice offer offers office one opportunity opportunities paced part
    passion passionate people perks plus position preferred proven qualification
    qualifications range related relevant remote require required requirement requirements
    responsibilities responsibility role salary skill skills strong success successful team
    teams time title two three understanding use used using want well work working world
    year years
    """
)
"""Words every job description uses; they say nothing about this one."""

_BOILERPLATE_HEADING = re.compile(
    r"^\W*(benefits|perks|compensation|salary|what we offer|we offer|why join|why us|"
    r"equal opportunit|eeo|diversity|how to apply|to apply|about us|about the company|"
    r"who we are|our values|location|legal|privacy)",
    re.IGNORECASE,
)
_NOT_A_TITLE = re.compile(
    r"^\W*(job description|description|overview|about|summary|the role|the opportunity|"
    r"responsibilities|requirements|qualifications|who you are|what you|we are|we're|"
    r"our |location|posted|apply|full[- ]time|part[- ]time|remote|hybrid|on[- ]site)",
    re.IGNORECASE,
)
_LABEL = r"(?:[ \t]*:[ \t]*|[ \t]+[\u2013\u2014-][ \t]+)(\S[^\n]{0,118})$"
_TITLE_LABEL = re.compile(
    r"^[ \t>*#_-]*(?:job[ \t]+title|position[ \t]+title|title|position|role)\**" + _LABEL,
    re.IGNORECASE | re.MULTILINE,
)
_COMPANY_LABEL = re.compile(
    r"^[ \t>*#_-]*(?:company(?:[ \t]+name)?|employer|organi[sz]ation)\**" + _LABEL,
    re.IGNORECASE | re.MULTILINE,
)
_HIRING_PREFIX = re.compile(
    r"^(?:we(?:'re| are)[ \t]+)?(?:now[ \t]+)?(?:hiring|job opening|opening|vacancy)"
    r"[ \t]*[:\u2013\u2014-]?[ \t]+(?:an?[ \t]+)?",
    re.IGNORECASE,
)
_HIRING_SENTENCE = re.compile(
    r"(?:hiring|looking for|seeking|searching for|in search of)[ \t]+"
    r"(?:an?|our next|our first|another)[ \t]+"
    r"((?:[A-Z][\w/&+.-]*[ \t]+){0,5}[A-Z][\w/&+.-]*)"
)
_BRACKETED = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
_TITLE_SEPARATORS = re.compile(
    r"\s+(?:at|@)\s+|\s+[|\u2013\u2014\u2022-]\s+|\s*,\s+", re.IGNORECASE
)
_AT = re.compile(r"\s+(?:at|@)\s+", re.IGNORECASE)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_SENTENCE_END = re.compile(r"[.!?:;]")


# -- words --------------------------------------------------------------------


def stem(word: str) -> str:
    """A cheap stem: plural and -ing / -ed endings removed. Not linguistics.

    Good enough to see that "pipelines" and "pipeline", or "mentoring" and
    "mentor", are the same word; it is only used to compare two texts.
    """
    w = word.casefold()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def words(text: str) -> list[str]:
    """The lower-cased words of `text`, in order."""
    return [word.casefold() for word in _WORD.findall(text)]


def stems(text: str) -> list[str]:
    return [stem(word) for word in words(text)]


def _salient(word: str) -> bool:
    """Whether a (lower-cased) word can say something about a job."""
    return (
        len(word) >= 2
        and not word.isdigit()
        and word not in STOP_WORDS
        and word not in BOILERPLATE_WORDS
        and word not in SENIORITY_WORDS
    )


def _spaced(parts: Iterable[str]) -> str:
    """Words joined into one string with a space at both ends, for phrase lookups."""
    return f" {' '.join(parts)} "


# -- analysing the description ------------------------------------------------


@dataclass(frozen=True)
class Term:
    """A salient word (or two-word phrase) of the description.

    `text` is how the description spells it most often (lower-cased), `key` its
    stemmed form (`"data pipeline"` for "data pipelines") and `count` how often
    it occurs. `proper` says it is written with a capital in the middle of a
    sentence somewhere: probably a name (a tool, a product, a place).
    """

    text: str
    key: str
    count: int
    in_title: bool = False
    proper: bool = False


@dataclass(frozen=True)
class Analysis:
    """A job description reduced to what retrieval and the explanations need."""

    text: str
    truncated: bool
    title: str | None
    company_hint: str | None
    terms: tuple[Term, ...]
    """Salient single words, most prominent first."""
    phrases: tuple[Term, ...]
    """Adjacent pairs of salient words, most frequent first."""
    focused_query: str
    section_queries: tuple[str, ...]
    stem_text: str
    """Every word of the description, stemmed, space-separated and space-padded."""


def _clean_title(line: str) -> str:
    line = _BRACKETED.sub("", line).strip(" \t*#_>-:|\u2022")
    return " ".join(_HIRING_PREFIX.sub("", line).split())


def _has_role_noun(text: str) -> bool:
    return any(stem(word) in _ROLE_NOUNS or word in _ROLE_NOUNS for word in words(text))


def infer_title(text: str) -> tuple[str | None, str | None]:
    """Guess `(title, company)` from a job description; either may be None.

    In order: a "Title: ..." style line; the first short heading-like line
    near the top that names a role ("Senior Data Engineer at Acme" also gives
    the company); a "we are hiring a ..." sentence; the first short line.
    The company here is only what the text says next to the title; matching
    the owner's known companies is the repository's job.
    """
    company = None
    if found := _COMPANY_LABEL.search(text):
        company = _clean_title(found.group(1)) or None
    if found := _TITLE_LABEL.search(text):
        parts = _AT.split(found.group(1), maxsplit=1)
        labelled = _clean_title(parts[0])
        if labelled and len(labelled.split()) <= MAX_TITLE_WORDS and not labelled.endswith("."):
            if len(parts) == 2 and company is None:
                company = _clean_title(parts[1]) or None
            return labelled, company

    top = [line.strip() for line in text.splitlines() if line.strip()][:6]
    fallback = None
    for line in top:
        if _TITLE_LABEL.match(line) or _COMPANY_LABEL.match(line) or _NOT_A_TITLE.match(line):
            continue
        cleaned = _clean_title(line)
        if (
            not cleaned
            or len(cleaned) > MAX_TITLE_CHARS
            or cleaned.endswith((".", "!", "?"))
            or len(cleaned.split()) > MAX_TITLE_WORDS
        ):
            continue
        at_parts = _AT.split(cleaned, maxsplit=1)
        for segment in _TITLE_SEPARATORS.split(cleaned):
            if segment and _has_role_noun(segment):
                if company is None and len(at_parts) == 2 and at_parts[0].startswith(segment):
                    company = at_parts[1].strip() or None
                return segment.strip(), company
        if fallback is None and len(cleaned.split()) <= 6:
            fallback = cleaned

    if found := _HIRING_SENTENCE.search(text):
        return found.group(1).strip(), company
    return fallback, company


def _count_terms(
    text: str, title: str | None, ignore: frozenset[str] = frozenset()
) -> tuple[list[Term], list[Term]]:
    """Salient words and adjacent salient pairs of `text`, most prominent first.

    Words whose stem is in `ignore` are treated like stop words.
    """
    title_stems = {stem(word) for word in words(title or "") if _salient(word)}
    spellings: dict[str, Counter[str]] = {}
    order: dict[str, int] = {}
    proper: set[str] = set()
    pair_spellings: dict[str, Counter[str]] = {}
    for line in [title or "", *text.splitlines()]:
        previous: str | None = None
        sentence_start = True
        last_end = 0
        for found in _WORD.finditer(line):
            raw, word = found.group(), found.group().casefold()
            if _SENTENCE_END.search(line, last_end, found.start()):
                sentence_start = True
            last_end = found.end()
            capitalised = raw[0].isupper() and not sentence_start
            sentence_start = False
            key = stem(word)
            if not _salient(word) or key in ignore:
                previous = None
                continue
            spellings.setdefault(key, Counter())[word] += 1
            order.setdefault(key, len(order))
            if capitalised and line is not title:
                proper.add(key)
            if previous is not None and stem(previous) != key:
                pair_key = f"{stem(previous)} {key}"
                pair_spellings.setdefault(pair_key, Counter())[f"{previous} {word}"] += 1
            previous = word

    def term(key: str, counter: Counter[str]) -> Term:
        spelling = counter.most_common(1)[0][0]
        return Term(spelling, key, sum(counter.values()), key in title_stems, key in proper)

    terms = [term(key, counter) for key, counter in spellings.items()]
    terms.sort(key=lambda t: (not t.in_title, -t.count, order[t.key]))
    phrases = [term(key, counter) for key, counter in pair_spellings.items()]
    phrases.sort(key=lambda t: (-t.count, t.text))
    return terms, phrases


def _sections(text: str, ignore: frozenset[str]) -> list[str]:
    """Paragraphs of `text` worth a query of their own: no boilerplate, enough terms.

    A paragraph needs `MIN_SECTION_TERMS` distinct salient words besides those
    in `ignore` (the title and the company), which leaves out the heading block.
    """
    found = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph or _BOILERPLATE_HEADING.match(paragraph):
            continue
        keys = {stem(word) for word in words(paragraph) if _salient(word)} - ignore
        if len(keys) >= MIN_SECTION_TERMS:
            found.append((len(keys), len(found), paragraph))
    largest = sorted(found, key=lambda item: (-item[0], item[1]))[:MAX_SECTIONS]
    return [paragraph for _, _, paragraph in sorted(largest, key=lambda item: item[1])]


def _query(terms: Sequence[Term], limit: int) -> str:
    return " ".join(term.text for term in terms[:limit])


def clean(description_text: str) -> tuple[str, bool]:
    """The description as it is read: trimmed and cut to `MAX_DESCRIPTION_CHARS`."""
    raw = (description_text or "").replace("\x00", " ").strip()
    text = raw[:MAX_DESCRIPTION_CHARS]
    return text, len(raw) > len(text)


def analyse(
    description_text: str, *, title: str | None = None, company: str | None = None
) -> Analysis:
    """Reduce a job description to a title, salient terms and retrieval queries.

    `title` overrides the inferred one. The words of `company` (the hiring
    company's name) are left out of the terms: the name says nothing about
    whether the owner's history fits the job.
    """
    text, truncated = clean(description_text)
    inferred_title, company_hint = infer_title(text)
    title = " ".join((title or "").split()) or inferred_title
    ignore = frozenset(stems(company or ""))
    terms, phrases = _count_terms(text, title, ignore)

    focused = _query(terms, MAX_FOCUSED_TERMS)
    section_queries: list[str] = []
    for section in _sections(text, ignore | {t.key for t in terms if t.in_title}):
        section_terms, _ = _count_terms(section, None, ignore)
        query = _query(section_terms, MAX_SECTION_TERMS)
        if query and query != focused and query not in section_queries:
            section_queries.append(query)
    return Analysis(
        text=text,
        truncated=truncated,
        title=title or None,
        company_hint=company_hint,
        terms=tuple(terms),
        phrases=tuple(phrases),
        focused_query=focused,
        section_queries=tuple(section_queries),
        stem_text=_spaced(stems(text)),
    )


# -- results ------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    """The job as understood. `*_inferred` is True when the text, not the caller, said so.

    `company_known` says the company is one of the owner's companies (somebody
    they know works or worked there, or they did); only then can the company
    groups be filled. `other_companies` are further companies of the owner the
    text names, `company_suggestions` near-misses for a company that is not known.
    """

    title: str | None = None
    company: str | None = None
    title_inferred: bool = False
    company_inferred: bool = False
    company_known: bool = False
    other_companies: tuple[str, ...] = ()
    company_suggestions: tuple[str, ...] = ()


@dataclass(frozen=True)
class HistoryItem:
    """A part of the owner's history that fits the description, and why.

    `matched_terms` are the description's words and phrases that occur in
    `text` (compared by stem); `retrievers` is `("fts", "vector")` or one of them.
    """

    source: str
    source_id: int
    title: str
    text: str
    score: float
    matched_terms: tuple[str, ...]
    retrievers: tuple[str, ...]
    why: str


@dataclass(frozen=True)
class PersonMatch:
    person: NetworkPerson
    why: str


@dataclass(frozen=True)
class JobMatch:
    role: Role
    history: list[HistoryItem] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    at_company: list[PersonMatch] = field(default_factory=list)
    similar_titles: list[PersonMatch] = field(default_factory=list)
    former_at_company: list[PersonMatch] = field(default_factory=list)
    totals: dict[str, int] = field(default_factory=dict)
    """How many people each connection group has before the cap."""
    mode_used: str | None = None
    """`hybrid` or `fts`; None when the description had nothing to search for."""
    note: str | None = None
    truncated: bool = False


# -- part 1: history ----------------------------------------------------------

_RETRIEVER_NAMES = {"fts": "full-text", "vector": "vector"}


def evidence(analysis: Analysis, chunk_text: str) -> tuple[str, ...]:
    """The description's terms that occur in `chunk_text`, most telling first.

    Candidates are the description's salient words and its two-word phrases
    that are adjacent in both texts. They are ordered by how often the
    description uses them (a phrase beats a word used equally often, title
    words get a bonus); a word that is part of a listed phrase is not listed
    by itself. At most
    `MAX_EVIDENCE_TERMS`, compared by stem.
    """
    chunk_stems = stems(chunk_text)
    present = set(chunk_stems)
    chunk_stem_text = _spaced(chunk_stems)
    title_bonus = {term.key for term in analysis.terms if term.in_title}
    candidates: list[tuple[float, int, Term]] = []
    for phrase in analysis.phrases:
        if f" {phrase.key} " in chunk_stem_text:
            bonus = sum(key in title_bonus for key in phrase.key.split())
            candidates.append((phrase.count + 0.5 + bonus, len(candidates), phrase))
    for term in analysis.terms:
        if term.key in present:
            candidates.append((term.count + (1 if term.in_title else 0), len(candidates), term))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    chosen: list[Term] = []
    in_phrases: set[str] = set()
    for _, _, term in candidates:
        keys = term.key.split()
        is_phrase = len(keys) > 1
        if all(key in in_phrases for key in keys):
            continue
        if is_phrase:
            # A phrase replaces its own words if they were already chosen.
            chosen = [other for other in chosen if other.key not in keys]
            in_phrases.update(keys)
        elif any(other.key == term.key for other in chosen):
            continue
        chosen.append(term)
        if len(chosen) >= MAX_EVIDENCE_TERMS:
            break
    shown = [term.text for term in chosen]
    return tuple(shown)


def explain(matched_terms: Sequence[str], retrievers: Sequence[str]) -> str:
    """ "matched on: airflow, data pipelines; full-text + vector"."""
    how = " + ".join(_RETRIEVER_NAMES[name] for name in ("fts", "vector") if name in retrievers)
    if matched_terms:
        return f"matched on: {', '.join(matched_terms)}; {how}"
    return f"no keyword in common with the description; similar in meaning ({how})"


def _history(
    repo: Repository, embedder: Embedder | None, analysis: Analysis, k: int
) -> tuple[list[HistoryItem], str | None, str | None]:
    queries = [q for q in (analysis.focused_query, *analysis.section_queries) if q]
    if not queries:
        return [], None, "The description has no searchable words."
    hits: dict[int, SearchHit] = {}
    retrievers: dict[int, set[str]] = {}
    rankings: list[list[int]] = []
    mode_used, note = "hybrid", None
    for query in queries:
        result = search(
            repo,
            embedder,
            query,
            sources=HISTORY_SOURCES,
            k=max(CANDIDATES_PER_QUERY, k),
            mode="hybrid",
        )
        if result.mode_used != "hybrid":
            mode_used, note = result.mode_used, note or result.note
        rankings.append([hit.chunk_id for hit in result.hits])
        for hit in result.hits:
            hits.setdefault(hit.chunk_id, hit)
            retrievers.setdefault(hit.chunk_id, set()).update(hit.matched)

    items = []
    for chunk_id, score in reciprocal_rank_fusion(rankings)[:k]:
        hit = hits[chunk_id]
        found_by = tuple(name for name in ("fts", "vector") if name in retrievers[chunk_id])
        matched = evidence(analysis, hit.text)
        items.append(
            HistoryItem(
                source=hit.source_table,
                source_id=hit.source_id,
                title=hit.title,
                text=hit.text,
                score=score,
                matched_terms=matched,
                retrievers=found_by,
                why=explain(matched, found_by),
            )
        )
    return items, mode_used, note


def _matched_skills(skills: Sequence[str], analysis: Analysis) -> list[str]:
    """The skills named in the description, as whole words, compared by stem."""
    found = []
    for skill in skills:
        skill_stems = stems(skill)
        if skill_stems and f" {' '.join(skill_stems)} " in analysis.stem_text:
            found.append(skill)
    return found


def _missing_terms(repo: Repository, analysis: Analysis, skills: Sequence[str]) -> list[str]:
    """Prominent terms of the description that the owner's chunks and skills never use.

    "Prominent" is a heuristic: words written like names mid-sentence (tools,
    products) and words the description repeats.
    """
    ignore = {stem(word) for skill in skills for word in words(skill)}
    prominent = [
        term
        for term in sorted(analysis.terms, key=lambda t: (not t.proper, -t.count))
        if (term.proper or term.count >= MIN_REPEATS) and len(term.text) > 2
    ]
    candidates = [term for term in prominent if term.key not in ignore][:PROMINENT_TERMS]
    if not candidates:
        return []
    present = repo.terms_in_chunks([t.text for t in candidates], sources=HISTORY_SOURCES)
    return [t.text for t in candidates if t.text not in present][:MAX_MISSING_TERMS]


# -- part 2: connections ------------------------------------------------------


@dataclass(frozen=True)
class TitleOverlap:
    """What two job titles share, ignoring seniority. `substantive` excludes generic nouns."""

    shared: tuple[str, ...] = ()
    substantive: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.substantive)


def title_words(title: str | None) -> dict[str, str]:
    """`{stem: word}` for the words of a title that say what the job is."""
    found: dict[str, str] = {}
    for word in words(title or ""):
        if word in SENIORITY_WORDS or word in STOP_WORDS or word.isdigit() or len(word) < 2:
            continue
        found.setdefault(stem(word), word)
    return found


def _substantive(title: dict[str, str]) -> set[str]:
    """The stems that carry a title; all of them if it has only generic nouns."""
    specific = {key for key in title if key not in GENERIC_ROLE_WORDS}
    return specific or set(title)


def title_overlap(role_title: str | None, other_title: str | None) -> TitleOverlap:
    """Token overlap of two titles. Truthy only with a substantive word in common.

    "Senior Data Engineer" and "Data Engineer II" share `data` and `engineer`;
    "Data Engineer" and "Firmware Engineer" share only the generic `engineer`,
    which is not a match.
    """
    role, other = title_words(role_title), title_words(other_title)
    shared = [key for key in role if key in other]
    wanted = _substantive(role)
    return TitleOverlap(
        shared=tuple(role[key] for key in shared),
        substantive=tuple(role[key] for key in shared if key in wanted),
    )


def _search_words(role_title: str | None) -> list[str]:
    """Substrings to pre-filter titles with: the common prefix of each word and its stem."""
    role = title_words(role_title)
    found = []
    for key in _substantive(role):
        word = role[key]
        prefix = word
        while not key.startswith(prefix):
            prefix = prefix[:-1]
        found.append(prefix if len(prefix) >= 3 else word)
    return found


def _connections(
    repo: Repository, role: Role, company_id: int | None, limit: int
) -> tuple[list[PersonMatch], list[PersonMatch], list[PersonMatch], dict[str, int]]:
    company_ids = [company_id] if company_id is not None else []

    current = repo.connections_at_companies(company_ids)
    overlaps = {person.id: title_overlap(role.title, person.title) for person in current}
    current = sorted(current, key=lambda p: -len(overlaps[p.id].substantive))  # stable: by name
    at_company = []
    for person in current:
        why = f"Works at {role.company} now"
        if overlaps[person.id]:
            why += f"; similar title ({', '.join(overlaps[person.id].shared)})"
        at_company.append(PersonMatch(person, why))

    scored = []
    for person in repo.connections_with_title_words(_search_words(role.title)):
        overlap = title_overlap(role.title, person.title)
        if not overlap:
            continue
        there = company_id is not None and person.company_id == company_id
        key = (-len(overlap.substantive), -len(overlap.shared), not there)
        why = (
            f"Similar title: {person.title} \N{LEFT RIGHT ARROW} {role.title} "
            f"(shared: {', '.join(overlap.shared)})"
        )
        if there:
            why += f"; works at {role.company}"
        scored.append((key, len(scored), PersonMatch(person, why)))
    similar = [match for _, _, match in sorted(scored, key=lambda item: item[:2])]

    former_people = repo.former_connections_of_companies(company_ids)
    departures = repo.company_departures(company_ids, [person.id for person in former_people])
    former = []
    for person in former_people:
        left = departures.get(person.id)
        why = f"Worked at {role.company}"
        if left and left.title:
            why += f" as {left.title}"
        if left and left.moved_by:
            why += f"; had moved on by {left.moved_by.date().isoformat()}"
        else:
            why += " before"
        former.append((left.moved_by if left else None, PersonMatch(person, why)))
    # Most recent leavers first; undated ones last, in name order.
    former.sort(key=lambda item: -(item[0].timestamp() if item[0] else 0.0))
    former_matches = [match for _, match in former]

    totals = {
        "at_company": len(at_company),
        "similar_titles": len(similar),
        "former_at_company": len(former_matches),
    }
    return at_company[:limit], similar[:limit], former_matches[:limit], totals


# -- the role -----------------------------------------------------------------


def _resolve_role(
    repo: Repository, text: str, company: str | None, title: str | None
) -> tuple[Role, int | None]:
    """Decide title and company; returns the role and the known company's id."""
    given_title = " ".join((title or "").split())
    given_company = " ".join((company or "").split())
    inferred_title, company_hint = infer_title(text)
    role_title = given_title or inferred_title
    title_inferred = bool(role_title) and not given_title

    named = repo.companies_named_in(text, limit=MAX_OTHER_COMPANIES + 1)

    def others(chosen_id: int | None) -> tuple[str, ...]:
        names = [c.display_name for c in named if c.id != chosen_id]
        return tuple(names[:MAX_OTHER_COMPANIES])

    hint = given_company or company_hint
    if hint:
        matches = repo.find_companies(hint, limit=MAX_OTHER_COMPANIES + 1)
        exact = next((m for m in matches if m.match == "exact"), None)
        if exact is not None:
            role = Role(
                role_title,
                exact.display_name,
                title_inferred,
                not given_company,
                True,
                others(exact.id),
            )
            return role, exact.id
        if given_company or not named:
            role = Role(
                role_title,
                hint,
                title_inferred,
                not given_company,
                False,
                others(None),
                tuple(m.display_name for m in matches[:MAX_OTHER_COMPANIES]),
            )
            return role, None
    if named:
        best = named[0]
        return Role(role_title, best.display_name, title_inferred, True, True, others(best.id)), (
            best.id
        )
    return Role(role_title, None, title_inferred), None


# -- entry point --------------------------------------------------------------


def match_job(
    repo: Repository,
    embedder: Embedder | None,
    description_text: str,
    *,
    company: str | None = None,
    title: str | None = None,
    k: int = DEFAULT_K,
    group_limit: int = DEFAULT_GROUP_LIMIT,
) -> JobMatch:
    """Match a job description against the history and network of `repo`'s owner.

    `company` and `title` override what would otherwise be inferred from the
    text. `k` caps the history items, `group_limit` each connection group.
    Without an `embedder` (or without stored vectors) retrieval is full-text
    only: `mode_used` is then `"fts"` and `note` says why. Raises `ValueError`
    for an empty description or `k < 1`; the text is read up to
    `MAX_DESCRIPTION_CHARS` characters.
    """
    if not (description_text or "").strip():
        raise ValueError("description_text must not be empty")
    if k < 1 or group_limit < 1:
        raise ValueError("k and group_limit must be at least 1")

    text, _ = clean(description_text)
    role, company_id = _resolve_role(repo, text, company, title)
    analysis = analyse(text, title=role.title, company=role.company)
    history, mode_used, note = _history(repo, embedder, analysis, k)
    skills = repo.skill_names()
    at_company, similar, former, totals = _connections(repo, role, company_id, group_limit)
    return JobMatch(
        role=role,
        history=history,
        matched_skills=_matched_skills(skills, analysis),
        missing_terms=_missing_terms(repo, analysis, skills),
        at_company=at_company,
        similar_titles=similar,
        former_at_company=former,
        totals=totals,
        mode_used=mode_used,
        note=note,
        truncated=len((description_text or "").strip()) > MAX_DESCRIPTION_CHARS,
    )
