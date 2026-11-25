"""Regression tests for the red-team findings in docs/RED_TEAM_AUTOPSY.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fundmatch.engine import get_engine  # noqa: E402
from fundmatch.privacy import redact_org_name, redact_person_names  # noqa: E402
from fundmatch.text import MIN_WORDS, check_input  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def test_rt01_engine_matches_original_sklearn_pipeline(engine):
    """Logits must match the pickled training pipeline, not a re-export of it."""
    fixture = json.loads((ROOT / "tests" / "fixtures" / "sklearn_parity.json").read_text())
    ours = engine.logits(engine.vectorize(fixture["texts"]), scaled=False)
    reference = np.array(fixture["logits"])
    assert np.max(np.abs(ours - reference)) < 1e-4  # sklearn used float32
    assert (ours.argmax(1) == reference.argmax(1)).all()


def test_rt01_stop_words_are_removed_before_bigrams(engine):
    tokens = engine._tokens("Cultural and scientific perceptions of the sea")
    assert tokens == ["cultural", "scientific", "perceptions", "sea"]
    assert "cultural scientific" in engine.index


@pytest.mark.parametrize("abstract", ["!!! " * 40, "12345 " * 40, "robot " * 40, "xqzt vbnm qwrp lkjh " * 10, "   \n\t "])
def test_rt03_padding_and_repetition_are_refused(abstract):
    errors, _ = check_input("", abstract)
    assert errors


def test_rt03_word_boundary():
    pool = ("soil carbon microbes nitrogen forest climate model field survey water plant root crop "
            "yield drought farm data sensor network policy health river fungi decay wood test "
            "species habitat pollen insect bird coast").split()
    assert len(pool) >= MIN_WORDS
    short_errors, _ = check_input("", " ".join(pool[: MIN_WORDS - 1]))
    assert short_errors, "one word under the minimum must be refused"
    errors, _ = check_input("", " ".join(pool[:MIN_WORDS]))
    assert errors == [], "exactly the minimum number of distinct real words must pass"
    padded_errors, _ = check_input("", " ".join(pool[: MIN_WORDS - 1]) + " 2024 !!! 3.5")
    assert padded_errors, "numbers and symbols must not top up the count"


def test_rt03_engine_refuses_zero_vocabulary(engine):
    result = engine.analyze("Qwxv", " ".join(f"zq{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}xv" for i in range(40)))
    assert not result.ok


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Established Career Fellowship for Professor Jane Example", "Established Career Fellowship for Professor [name]"),
        ("Visit of Prof. Oliver Example", "Visit of Prof. [name]"),
        ("A Visiting Fellowship for Dr. C.V. Example", "A Visiting Fellowship for Dr. [name]"),
        ("Overseas Travel Grant for Dr L Example", "Overseas Travel Grant for Dr [name]"),
        ("Quantum sensing for navigation", "Quantum sensing for navigation"),
    ],
)
def test_rt05_person_names_redacted_in_titles(raw, expected):
    assert redact_person_names(raw) == expected


def test_rt05_sole_trader_org_names_redacted():
    assert redact_org_name("Mr J S Example") == "Sole trader (name withheld)"
    assert redact_org_name("A B EXAMPLE T/A WIDGET WORKS") == "WIDGET WORKS"
    assert redact_org_name("University of Leeds") == "University of Leeds"


def test_rt05_engine_table_has_no_honorific_names(engine):
    titles = engine.frame["title"].astype(str)
    leaked = titles.str.contains(r"\b(?:Professor|Prof\.?|Dr\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+", regex=True)
    assert int(leaked.sum()) == 0
    orgs = engine.frame["lead_org_name"].astype(str)
    assert not orgs.str.contains(r"\bT/A\b", regex=True).any()


def test_rt06_low_confidence_thresholds_match_calibration_bins():
    info = json.loads((ROOT / "artifacts" / "model_info.json").read_text())
    bins = info["accuracy_by_confidence"]
    assert bins["(0.8, 1.0]"] > 0.9 > bins["(0.6, 0.8]"] > bins["(0.4, 0.6]"]
    from fundmatch import engine as eng

    assert eng.MODERATE == 0.60


def test_rt08_gtr_links_use_unique_grant_refs(engine):
    refs = engine.frame["grant_ref"].astype(str)
    assert refs.is_unique
    assert engine.frame["gtr_url"].astype(str).str.startswith("https://gtr.ukri.org/projects?ref=").all()


def test_rt10_training_requirements_cover_notebook_imports():
    pins = (ROOT / "training" / "requirements-training.txt").read_text().lower()
    for package in ("lightgbm", "sentence-transformers", "scikit-learn", "faiss-cpu", "rank-bm25"):
        assert package in pins, package


def test_fr8_recency_rule_excludes_recent_awards(engine):
    from fundmatch.engine import RECENT_START_YEAR

    frame = engine.frame
    years = pd.to_numeric(frame["start_year"], errors="coerce")
    priced = frame["award_gbp"].astype(float) > 0
    recent = np.flatnonzero((years >= RECENT_START_YEAR).to_numpy() & priced.to_numpy())[:5]
    older = np.flatnonzero((years < RECENT_START_YEAR).to_numpy() & priced.to_numpy())[:5]
    band = engine._award(np.concatenate([recent, older]), cpih=False, same_scheme=False, scheme=None)
    assert len(recent) == 5 and band.n_positive == 5
    assert "recency rule" in band.zero_note
