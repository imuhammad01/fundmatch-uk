"""V5 Proposal Studio: every number comes from the deployed model, the test profile or GtR rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fundmatch import insights  # noqa: E402
from fundmatch.brand import APP_EDITION, APP_VERSION  # noqa: E402
from fundmatch.demos import DEMOS  # noqa: E402
from fundmatch.engine import get_engine  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def _demo(i=0):
    return DEMOS[i]["title"], DEMOS[i]["abstract"]


def test_brand_is_v5():
    assert APP_VERSION.startswith("5.")
    assert APP_EDITION.startswith("V5")


def test_reliability_uses_published_bin(engine):
    title, abstract = _demo()
    result = engine.analyze(title, abstract)
    rel = insights.reliability(engine, result)
    assert result.top_council == "EPSRC"
    assert round(result.councils[0].probability, 2) == 0.88
    assert rel["bin"] == "(0.8, 1.0]"
    assert rel["bin_test_accuracy"] == engine.model_info["accuracy_by_confidence"]["(0.8, 1.0]"]
    assert rel["margin"] == pytest.approx(result.councils[0].probability - result.councils[1].probability, abs=1e-4)
    assert rel["set90"] == result.conformal_set_90
    assert insights.reliability(engine, None) == {}


def test_decision_drivers_match_the_explanation(engine):
    title, abstract = _demo()
    drivers = insights.decision_drivers(engine, title, abstract)
    assert drivers["council"] == "EPSRC"
    toward = [r for r in drivers["rows"] if r["direction"] == "toward"]
    away = [r for r in drivers["rows"] if r["direction"] == "away"]
    assert toward and all(r["points"] > 0 for r in toward)
    assert all(r["points"] < 0 for r in away)
    assert toward[0]["term"] == "quantum"
    explanation = engine.explain_council(title, abstract, "EPSRC", k=10)
    assert toward[0]["points"] == pytest.approx(explanation.positive[0].contribution / engine.temperature, abs=1e-3)
    other = insights.decision_drivers(engine, title, abstract, "MRC")
    assert other["council"] == "MRC"


def test_checklist_is_evidence_based(engine):
    title, abstract = _demo()
    check = insights.abstract_checklist(engine, title, abstract)
    assert check["total"] == len(check["items"]) == 10
    assert all(item["status"] in {"met", "partial", "missing"} for item in check["items"])
    assert all(item["evidence"] for item in check["items"])
    assert "not a prediction of funding success" in check["note"]
    thin = insights.abstract_checklist(engine, "Notes", "Some words about a topic that matters.")
    by_id = {item["id"]: item for item in thin["items"]}
    assert by_id["length"]["status"] == "missing"
    assert by_id["aim"]["status"] == "missing"
    assert thin["met"] < check["met"]


def test_keyword_radar_counts_real_titles(engine):
    title, abstract = _demo()
    radar = insights.keyword_radar(engine, title, abstract)
    assert radar["neighbours"] == 50
    assert radar["suggest"] and radar["already"]
    text = f"{title} {abstract}".lower()
    for row in radar["suggest"]:
        assert row["weight"] > 0 and 2 <= row["titles"] <= 50
        assert row["term"] in engine.index
        assert f" {row['term']} " not in f" {text} "
    assert any(row["term"] == "quantum" for row in radar["already"])


def test_compare_two_ideas(engine):
    out = insights.compare_two(engine, DEMOS[0], DEMOS[1])
    assert out["ok"]
    assert out["a"]["top_council"] == "EPSRC" and out["b"]["top_council"] == "MRC"
    assert -1.0 <= out["semantic_cosine"] <= 1.0
    assert 0 <= out["shared_precedents"] <= out["neighbours"]
    assert sum(r["idea_a"] for r in out["councils"]) == pytest.approx(1.0, abs=1e-3)
    same = insights.compare_two(engine, DEMOS[0], DEMOS[0])
    assert same["semantic_cosine"] == pytest.approx(1.0, abs=1e-4)
    assert same["neighbour_jaccard"] == pytest.approx(1.0)
    bad = insights.compare_two(engine, DEMOS[0], {"title": "", "abstract": ""})
    assert not bad["ok"] and bad["errors"]


def test_filter_precedents():
    rows = [
        {"lead_funder": "EPSRC", "start_year": 2020, "award_gbp": 500000.0},
        {"lead_funder": "EPSRC", "start_year": 2012, "award_gbp": 0.0},
        {"lead_funder": "MRC", "start_year": 2021, "award_gbp": 1200000.0},
        {"lead_funder": "MRC", "start_year": None, "award_gbp": 90000.0},
    ]
    assert len(insights.filter_precedents(rows)) == 4
    assert len(insights.filter_precedents(rows, councils=["MRC"])) == 2
    assert len(insights.filter_precedents(rows, years=(2015, 2025))) == 2
    assert len(insights.filter_precedents(rows, priced_only=True)) == 3
    assert len(insights.filter_precedents(rows, award=(100000.0, 600000.0))) == 2  # £0 rows kept unless priced_only
    assert len(insights.filter_precedents(rows, award=(100000.0, 600000.0), priced_only=True)) == 1


def test_neighbourhood_pulse(engine):
    title, abstract = _demo()
    pulse = insights.neighbourhood_pulse(engine, title, abstract)
    assert pulse["neighbours"] == 200
    assert sum(row["projects"] for row in pulse["council_mix"]) == 200
    assert pulse["council_mix"][0]["council"] == "EPSRC"
    assert insights.neighbourhood_pulse(engine, "", "") == {}


def test_pdf_brief_is_a_real_pdf(engine):
    title, abstract = _demo()
    result = engine.analyze(title, abstract)
    pdf = insights.pdf_brief(
        title, abstract, result, insights.abstract_checklist(engine, title, abstract),
        insights.decision_drivers(engine, title, abstract), insights.keyword_radar(engine, title, abstract), version="v5",
    )
    assert pdf[:5] == b"%PDF-" and len(pdf) > 2000
    failed = insights.pdf_brief("", "", engine.analyze("", ""))
    assert failed[:5] == b"%PDF-"
    assert insights._latin("£ – “x” ≥ 中") == '£ - "x" >= ?'


def test_v5_studio_renders_in_streamlit():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 9
    labels = [tab.label for tab in at.tabs]
    assert labels[0] == "✦ Proposal Studio"
    for name in ("Your journey", "AI Coach & Red-Team", "UK Research Hubs", "Waves & fusion", "AI Matcher",
                 "Constellation & What-if", "Compare & Batch", "Model Lab & Ethics"):
        assert name in labels
    page = " ".join(m.value for m in at.markdown)
    assert "Mission brief" in page and "94.3%" in page and "EPSRC" in page
    for tool in ("Draft checklist", "Vocabulary", "Precedents", "Idea vs idea", "Export"):
        at.session_state["studio_tool"] = tool
        at.run()
        assert not at.exception, tool
    at.session_state["studio_tool"] = "Idea vs idea"
    at.run()
    at.button(key="vs_go").click().run()
    assert not at.exception
    assert "Semantic closeness" in " ".join(m.value for m in at.markdown)
    at.session_state["studio_tool"] = "Export"
    at.run()
    at.button(key="pdf_go").click().run()
    assert not at.exception
    assert at.session_state["pdf_bytes"][:5] == b"%PDF-"
