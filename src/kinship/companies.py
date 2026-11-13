"""Company name normalization. Pure functions, no database access.

LinkedIn exports spell one employer many ways ("Gifthealth", "Gift Health,
Inc.", "gift health"). Two functions turn a raw spelling into something
comparable:

`normalize_company_name(raw)`
    The human-readable normalized name: Unicode NFKC, case-folded, accents
    stripped, `&` read as "and", punctuation dropped, trailing legal suffixes
    (Inc, LLC, Ltd, Corp, Co, ...) removed, whitespace collapsed.
    "Gift Health, Inc." -> "gift health"; "ACME Corp" -> "acme".

`company_key(raw)`
    The normalized name with all whitespace removed: "gifthealth". This is the
    grouping key stored in `companies.name_normalized`; two spellings are the
    same company exactly when their keys are equal.

Known trade-offs of grouping on the key:

- Names that differ only in spacing merge. That is what makes "Gifthealth" and
  "Gift Health" one company, and it will occasionally merge two different
  companies ("Sun Life" and "Sunlife").
- Names that differ only in a legal suffix merge ("Quillfeather Systems" and
  "Quillfeather Systems, LLC"), which is intended; two distinct legal entities
  that share a trading name are not told apart.
- Nothing else is guessed: abbreviations, a leading "The", former names and
  parent/subsidiary relations are left as different companies.

A blank name is no company (`None`). A name is never normalized down to
nothing: a company literally called "Co" stays "co".
"""

from __future__ import annotations

import unicodedata

LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "llc",
        "ltd",
        "limited",
        "corp",
        "corporation",
        "co",
        "company",
        "plc",
        "gmbh",
        "lp",
        "llp",
        "pbc",
    }
)
"""Trailing words that say what kind of legal entity a company is, not which one."""

# Dropped without leaving a gap, so "L.L.C." reads "llc" and "O'Neil" reads "oneil".
_JOINING_PUNCTUATION = frozenset(".'\u2019\u2018`")


def _fold(raw: str) -> str:
    """NFKC, case-folded, without accents."""
    text = unicodedata.normalize("NFKC", raw).casefold()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_company_name(raw: str | None) -> str | None:
    """The normalized, space-separated form of a company name; None if it is blank."""
    if raw is None:
        return None
    folded = _fold(raw)
    if not folded.strip():
        return None

    chars: list[str] = []
    for char in folded.replace("&", " and "):
        if char.isalnum():
            chars.append(char)
        elif char not in _JOINING_PUNCTUATION:
            chars.append(" ")
    tokens = "".join(chars).split()
    if not tokens:
        # Punctuation only: keep it as written rather than lose the company.
        return " ".join(folded.split())

    stripped = False
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
        stripped = True
    # "Tiffany & Co." has become "tiffany and": the "and" belonged to the suffix.
    if stripped and len(tokens) > 1 and tokens[-1] == "and":
        tokens.pop()
    return " ".join(tokens)


def company_key(raw: str | None) -> str | None:
    """The grouping key of a company name: its normalized form without whitespace."""
    normalized = normalize_company_name(raw)
    if normalized is None:
        return None
    return "".join(normalized.split())
