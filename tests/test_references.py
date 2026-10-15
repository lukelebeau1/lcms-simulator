"""Citations: every cited key exists and is verified, every reference is cited, and docs/references.md is current."""

from pathlib import Path

import pytest

from lcms_sim.factors import BENCH, STAGE
from lcms_sim.faults.library import LIBRARY
from lcms_sim.faults.matcher import default_checks
from lcms_sim.paths import ROOT
from lcms_sim.references import (CLAIMS, STAGE_SOURCES, citing_sites, docs_citations, load_references, markdown_cite,
                                 ref, references_markdown, short_cite, unsourced)
from lcms_sim.trace import STAGE_KEYS


@pytest.fixture(scope="module")
def refs():
    return load_references()


def test_every_reference_is_verified_and_locatable(refs):
    assert len(refs) >= 30
    for r in refs.values():
        assert r.kind and r.title and r.verified, r.key
        assert r.doi or r.isbn or r.url, f"{r.key} has no DOI, ISBN or URL"
        assert "2026" in r.verified or "2025" in r.verified, f"{r.key}: verification is not dated"
        assert r.short() and r.full()


def test_every_cited_key_exists(refs):
    sites = citing_sites()
    missing = sorted({s.key for s in sites if s.key not in refs})
    assert not missing, missing
    assert len(sites) > 60


def test_every_reference_is_cited_somewhere(refs):
    cited = {s.key for s in citing_sites()}
    unused = sorted(set(refs) - cited)
    assert not unused, unused


def test_objects_carry_their_sources(refs):
    sourced = [f for f in BENCH + STAGE if f.sources]
    assert len(sourced) >= 14
    for f in BENCH + STAGE:
        for k in f.sources:
            assert k in refs, (f.key, k)
    assert sum(1 for fl in LIBRARY.values() if fl.sources) >= 15
    for c in default_checks():
        for k in c.sources:
            assert k in refs, (c.name, k)
    assert set(STAGE_SOURCES) == set(("spray",) + STAGE_KEYS)
    for c in CLAIMS:
        assert c.sources and c.status in ("sourced", "partly", "assumed")


def test_unsourced_claims_are_the_known_ones():
    names = {what for _, what, _ in unsourced()}
    # Pure arithmetic faults and the drift fault state model behaviour, not literature; everything else cites.
    allowed = {"Extraction volume error", "Low injection volume", "Sensitivity drift over the sequence",
               "Sensitivity drift per injection"}
    prefixes = ("Real sorption events last", "Very hot drying gas", "The rod gap (4 mm)", "The Gaussian pass window")
    assert all(n in allowed or n.startswith(prefixes) for n in names), names


def test_docs_cite_with_keys_that_exist(refs):
    sites = docs_citations()
    assert sites, "no [key] citations in docs"
    assert {s.key for s in sites} <= set(refs)
    assert any("lc.md" in s.where for s in sites) and any("ms.md" in s.where for s in sites)


def test_references_markdown_is_current():
    want = references_markdown()
    have = (ROOT / "docs" / "references.md").read_text(encoding="utf-8")
    assert have == want, "docs/references.md is stale: run scripts/make_references_md.py"


def test_short_and_markdown_citations():
    assert ref("kebarle_verkerk_2009").short() == "Kebarle and Verkerk 2009"
    assert ref("page_2007").short() == "Page et al. 2007"
    assert ref("dolan_2006_dwell").short() == "Dolan 2006"
    assert ref("sante_11312_2021").short() == "SANTE/11312/2021"
    assert ref("hoffmann_stroobant_2007").short() == "de Hoffmann and Stroobant 2007"   # particle kept
    assert ref("van_berkel_1995").short() == "Van Berkel and Zhou 1995"                # two-word surname
    assert ref("dawson_1995").short() == "Dawson 1995" and ref("crc_handbook_97").short() == "CRC Handbook 97th ed."
    assert short_cite(["knoll_2010", "dawson_1995"]) == "Knoll 2010; Dawson 1995"
    md = markdown_cite(["king_2000"])
    assert md.startswith("[King et al. 2000](https://doi.org/10.1016/S1044-0305(00)00163-X)")
    with pytest.raises(KeyError):
        ref("not_a_key")
