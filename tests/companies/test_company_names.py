"""`normalize_company_name` and `company_key`: pure, table-driven."""

import pytest

from kinship.companies import LEGAL_SUFFIXES, company_key, normalize_company_name
from kinship.synth import build_world


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        # case and whitespace
        ("Acme", "acme"),
        ("ACME", "acme"),
        ("  Gift   Health \t", "gift health"),
        # punctuation
        ("Acme, Inc.", "acme"),
        ("Yahoo!", "yahoo"),
        ("Coca-Cola", "coca cola"),
        ("Amazon.com", "amazoncom"),
        ("O'Reilly Media", "oreilly media"),
        ("O\u2019Reilly Media", "oreilly media"),
        ("Ernst/Young (US)", "ernst young us"),
        # legal suffixes, with and without dots and commas, repeated
        ("ACME Corp", "acme"),
        ("Acme Corp.", "acme"),
        ("Acme Corporation", "acme"),
        ("Acme Incorporated", "acme"),
        ("Acme LLC", "acme"),
        ("Acme, L.L.C.", "acme"),
        ("Acme Ltd", "acme"),
        ("Acme Limited", "acme"),
        ("Acme Co", "acme"),
        ("Acme Co.", "acme"),
        ("Acme Company", "acme"),
        ("Acme Co., Ltd.", "acme"),
        ("Acme Holdings Company, Inc.", "acme holdings"),
        ("Acme plc", "acme"),
        ("Acme GmbH", "acme"),
        ("Acme LP", "acme"),
        ("Acme LLP", "acme"),
        ("Acme, PBC", "acme"),
        ("Gift Health, Inc.", "gift health"),
        ("Quillfeather Systems, LLC", "quillfeather systems"),
        # a suffix word elsewhere in the name stays
        ("Co-op Bank", "co op bank"),
        ("Inc Magazine", "inc magazine"),
        ("Limited Brands", "limited brands"),
        # ampersand
        ("Juniper & Wren Studio", "juniper and wren studio"),
        ("Juniper and Wren Studio", "juniper and wren studio"),
        ("AT&T", "at and t"),
        ("Tiffany & Co.", "tiffany"),
        ("Johnson & Johnson", "johnson and johnson"),
        # unicode
        ("Nestlé", "nestle"),
        ("Société Générale", "societe generale"),
        ("\uff21\uff43\uff4d\uff45 Inc", "acme"),  # full-width letters (NFKC)
        ("Straße GmbH", "strasse"),
        ("株式会社トヨタ", "株式会社トヨタ"),
        # never empty
        ("Co", "co"),
        ("Co.", "co"),
        ("Inc", "inc"),
        ("Company, Inc.", "company"),
        ("LLC Ltd", "llc"),
        ("!!!", "!!!"),
        ("&", "and"),
    ],
)
def test_normalize_company_name(raw: str, normalized: str) -> None:
    assert normalize_company_name(raw) == normalized
    assert company_key(raw) == normalized.replace(" ", "")
    # Normalizing is stable: the normalized name is its own normal form.
    assert normalize_company_name(normalized) == normalized


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n", "　"])
def test_blank_is_no_company(raw: str | None) -> None:
    assert normalize_company_name(raw) is None
    assert company_key(raw) is None


@pytest.mark.parametrize("suffix", sorted(LEGAL_SUFFIXES))
def test_every_legal_suffix_is_stripped_but_never_to_nothing(suffix: str) -> None:
    for spelling in (suffix, suffix.upper(), suffix.capitalize() + "."):
        assert company_key(f"Acme {spelling}") == "acme"
        assert company_key(f"Acme, {spelling}") == "acme"
        assert company_key(spelling) == suffix


def test_issue_examples() -> None:
    assert company_key("gift health") == company_key("Gifthealth") == "gifthealth"
    assert company_key("Gift Health, Inc.") == "gifthealth"
    assert normalize_company_name("Gift Health, Inc.") == "gift health"
    assert normalize_company_name("Gifthealth") == "gifthealth"


def test_known_trade_offs() -> None:
    # Spacing is ignored, so these two different companies would merge.
    assert company_key("Sun Life") == company_key("Sunlife")
    # Nothing beyond the documented rules is guessed.
    assert company_key("The Acme Company") != company_key("Acme")
    assert company_key("IBM") != company_key("International Business Machines")


def test_generator_variants_group_exactly_as_the_ground_truth() -> None:
    """Each synthetic company group has one key, and no two groups share one."""
    groups = build_world().company_groups
    keys: dict[str, str] = {}
    for name, spellings in groups.items():
        group_keys = {company_key(spelling) for spelling in spellings}
        assert len(group_keys) == 1, (name, group_keys)
        key = group_keys.pop()
        assert key and key not in keys, (name, keys.get(key))
        keys[key] = name
    assert sum(len(spellings) > 1 for spellings in groups.values()) == 3
    assert (keys["acme"], keys["gifthealth"], keys["quillfeathersystems"]) == (
        "acme",
        "gifthealth",
        "quillfeather",
    )
