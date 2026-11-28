"""V5.2: Gap finder and 3D idea map.

Pure-logic tests use hand-made sub-topics so the rules are checked exactly. Engine
tests use the real 113,233-row table and recompute the numbers independently.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fundmatch import gaps  # noqa: E402
from fundmatch.demos import DEMOS  # noqa: E402
from fundmatch.engine import get_engine  # noqa: E402

EXAMPLE = (DEMOS[0]["title"], DEMOS[0]["abstract"])  # the example idea shown first in the app


@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module")
def example(engine):
    title, abstract = EXAMPLE
    hood = gaps.neighbourhood(engine, title, abstract)
    return hood, gaps.analyse(engine, title, abstract, hood=hood)


# ---------------------------------------------------------------- pure logic

def test_trend_label_thresholds_and_small_samples():
    assert gaps.trend_label(30, 10, 60, 60)["label"] == "growing"          # 75% vs 50% = +25 points
    assert gaps.trend_label(10, 30, 60, 60)["label"] == "shrinking"
    assert gaps.trend_label(21, 19, 60, 60)["label"] == "flat"
    assert gaps.trend_label(21, 19, 60, 60)["delta_pp"] == pytest.approx(2.5)
    few = gaps.trend_label(3, 2, 60, 60)
    assert few["label"] == "too few dated projects" and few["delta_pp"] is None
    # exactly at the threshold counts as a trend
    assert gaps.trend_label(29, 21, 50, 50)["label"] == "growing"


def test_trend_is_relative_to_the_neighbourhood_not_absolute():
    # the whole neighbourhood doubled, this sub-topic only kept pace: flat, not growing
    assert gaps.trend_label(40, 20, 400, 200)["label"] == "flat"


def _topic(i, projects, total, median, priced=None):
    return {"id": i, "projects": projects, "total_gbp": float(total), "median_gbp": median,
            "priced_projects": projects if priced is None else priced}


def test_classify_flags_small_and_poor_and_crowded():
    topics = [
        _topic(0, 100, 50e6, 400e3), _topic(1, 90, 40e6, 350e3), _topic(2, 80, 30e6, 300e3),
        _topic(3, 12, 0.2e6, 20e3), _topic(4, 20, 1.0e6, 60e3), _topic(5, 60, 20e6, 300e3),
    ]
    gaps.classify(topics)
    by_id = {t["id"]: t for t in topics}
    assert by_id[3]["status"] == "under-explored"
    assert by_id[4]["status"] == "under-explored"
    assert by_id[0]["status"] == "crowded" and by_id[1]["status"] == "crowded"
    assert by_id[2]["status"] == "balanced" and by_id[5]["status"] == "balanced"
    assert by_id[3]["thin"] < by_id[4]["thin"]                        # the thinner one ranks first


def test_classify_never_flags_tiny_or_above_average_topics():
    topics = [_topic(0, 40, 10e6, 200e3), _topic(1, 5, 0.1e6, 5e3), _topic(2, 38, 9e6, 190e3), _topic(3, 20, 12e6, 500e3)]
    gaps.classify(topics)
    assert topics[1]["status"] != "under-explored"                    # fewer than MIN_TOPIC_PROJECTS
    assert topics[3]["status"] != "under-explored"                    # below-average count but above-average money
    assert all(t["count_index"] >= 0 for t in topics)


def test_classify_caps_the_lists_and_marks_unpriced_funding():
    topics = [_topic(i, 10 + i, 1e5 * (i + 1), 1e4) for i in range(6)] + [_topic(6, 400, 5e7, 3e5), _topic(7, 380, 4e7, 3e5)]
    topics[0]["priced_projects"] = 1
    gaps.classify(topics, max_gaps=3, max_crowded=1)
    assert sum(t["status"] == "under-explored" for t in topics) == 3
    assert sum(t["status"] == "crowded" for t in topics) == 1
    assert topics[0]["funding_known"] is False and topics[1]["funding_known"] is True


def test_pct_rank_handles_ties_and_single_values():
    assert gaps._pct_rank([5.0]) == [0.5]
    assert gaps._pct_rank([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert gaps._pct_rank([2.0, 2.0, 5.0]) == [0.25, 0.25, 1.0]


# ---------------------------------------------------------------- real data

def test_every_neighbour_is_in_exactly_one_subtopic(engine, example):
    hood, data = example
    assert data["ok"] and len(hood.idx) == data["fetched"] == gaps.NEIGHBOURS
    assert data["neighbours"] + data["left_out"] == data["fetched"] and data["neighbours"] >= gaps.MIN_KEPT
    rows = [r for t in data["topics"] for r in t["rows"]]
    assert len(rows) == len(set(rows)) and set(rows) <= {int(i) for i in hood.idx}
    assert sum(t["projects"] for t in data["topics"]) == data["neighbours"]
    assert 3 <= data["k"] <= gaps.MAX_TOPICS


def test_funding_numbers_match_an_independent_recomputation(engine, example):
    _hood, data = example
    for topic in data["topics"]:
        frame = engine.frame.iloc[topic["rows"]]
        awards = frame["award_gbp"].to_numpy(dtype=float)
        priced = awards[awards > 0]
        assert topic["total_gbp"] == pytest.approx(awards.sum())
        assert topic["priced_projects"] == len(priced)
        if len(priced):
            assert topic["median_gbp"] == pytest.approx(float(np.median(priced)))
        else:
            assert topic["median_gbp"] is None
        top = frame["lead_funder"].value_counts()
        assert topic["council"] in set(top[top == top.max()].index)
        years = frame["start_year"].dropna().astype(int)
        assert sum(topic["by_year"].values()) == len(years)
        (r0, r1), (e0, e1) = gaps._windows(engine)
        assert topic["recent"] == int(((years >= r0) & (years <= r1)).sum())
        assert topic["earlier"] == int(((years >= e0) & (years <= e1)).sum())


def test_flags_follow_the_published_rules_and_stay_near_the_idea(engine, example):
    hood, data = example
    sim = engine.z_unit[[r for t in data["topics"] for r in t["rows"]]].astype(float) @ hood.query
    assert sim.min() >= data["relatedness_cutoff"] - 1e-9               # far-away neighbours were dropped
    for t in data["topics"]:
        assert t["status"] in {"under-explored", "crowded", "balanced"}
        if t["status"] == "under-explored":
            assert t["count_index"] <= gaps.UNDER_INDEX and t["funding_index"] <= gaps.UNDER_INDEX and t["projects"] >= gaps.MIN_TOPIC_PROJECTS
        if t["status"] == "crowded":
            assert t["count_index"] >= gaps.CROWDED_COUNT_INDEX
    assert sum(t["status"] == "under-explored" for t in data["topics"]) <= 3
    assert any(t["status"] == "under-explored" for t in data["topics"]), "the example idea should show at least one gap"
    order = [t["status"] for t in data["topics"]]
    assert order == sorted(order, key=["under-explored", "crowded", "balanced"].index)


def test_examples_are_real_projects_with_gtr_links(engine, example):
    _hood, data = example
    known = set(engine.frame["gtr_url"])
    for t in data["topics"]:
        assert 1 <= len(t["examples"]) <= 3
        assert len({e["title"].lower() for e in t["examples"]}) == len(t["examples"])
        for e in t["examples"]:
            assert e["gtr_url"] in known and e["gtr_url"].startswith("https://gtr.ukri.org/projects?ref=")


def test_key_terms_are_title_words_not_funder_names(example):
    _hood, data = example
    for t in data["topics"]:
        assert 1 <= len(t["terms"]) <= 6
        for term in t["terms"]:
            assert not set(term.split()) & gaps._FUNDER_WORDS


def test_caveats_say_the_three_honest_things(example):
    _hood, data = example
    text = " ".join(data["caveats"])
    assert "not the same as" in text and "opportunity" in text
    assert "not calls that are open" in text
    assert "£0" in text and "Unpriced" in text


def test_result_is_deterministic_and_search_is_not_disturbed(engine):
    title, abstract = EXAMPLE
    before = engine.z_unit[:50].copy()
    a = gaps.analyse(engine, title, abstract)
    b = gaps.analyse(engine, title, abstract)
    for key in ("projects", "total_gbp", "status", "terms", "rows"):
        assert [t[key] for t in a["topics"]] == [t[key] for t in b["topics"]]
    assert np.array_equal(before, engine.z_unit[:50])


def test_invalid_idea_gets_an_error_not_a_crash(engine):
    data = gaps.analyse(engine, "", "")
    assert data["ok"] is False and data["errors"]
    assert gaps.idea_map_3d(engine, gaps.neighbourhood(engine, "", ""))["ok"] is False


def test_gap_finder_is_fast_enough(engine):
    title, abstract = EXAMPLE
    gaps.analyse(engine, title, abstract)
    runs = []
    for _ in range(3):
        started = time.perf_counter()
        hood = gaps.neighbourhood(engine, title, abstract)
        gaps.analyse(engine, title, abstract, hood=hood)
        gaps.idea_map_3d(engine, hood)
        runs.append(time.perf_counter() - started)
    assert min(runs) < 2.0, runs


def test_works_for_every_demo_idea(engine):
    for demo in DEMOS:
        data = gaps.analyse(engine, demo["title"], demo["abstract"])
        assert data["ok"] or data["errors"]
        if data["ok"]:
            assert sum(t["projects"] for t in data["topics"]) == data["neighbours"]


def test_no_personal_fields_in_gap_output(example):
    _hood, data = example
    blob = repr(data).lower()
    for banned in ("pi_name", "orcid", "researcher", "investigator"):
        assert banned not in blob


# ---------------------------------------------------------------- 3D map

def test_3d_map_has_three_axes_and_the_idea(engine, example):
    hood, _data = example
    m = gaps.idea_map_3d(engine, hood)
    assert m["ok"] and m["n"] == gaps.MAP_POINTS == len(m["points"])
    assert len(m["idea"]) == 3 and np.isfinite(m["idea"]).all()
    e = m["explained"]
    assert e[0] >= e[1] >= e[2] > 0 and sum(e) <= 1.0
    xyz = np.array([[p["x"], p["y"], p["z"]] for p in m["points"]])
    assert np.allclose(xyz.mean(axis=0), 0, atol=1e-8)                 # PCA scores are centred
    assert np.allclose(np.corrcoef(xyz.T) - np.eye(3), 0, atol=1e-6)   # and uncorrelated
    assert "distort" in m["note"]
    assert [p["gtr_url"] for p in m["points"]][:3] == [engine.frame.iloc[i]["gtr_url"] for i in hood.idx[:3]]


def test_3d_map_keeps_the_ideas_place_among_its_neighbours(engine, example):
    hood, _data = example
    m = gaps.idea_map_3d(engine, hood)
    xyz = np.array([[p["x"], p["y"], p["z"]] for p in m["points"]])
    spread = np.linalg.norm(xyz - xyz.mean(axis=0), axis=1).max()
    assert np.linalg.norm(np.array(m["idea"])) < spread * 1.5


# ---------------------------------------------------------------- the page

def _page(tool):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    at.session_state["studio_tool"] = tool
    at.run()
    return at


def test_example_still_gives_epsrc_about_88_percent(engine):
    result = engine.analyze(*EXAMPLE)
    assert result.top_council == "EPSRC"
    assert 0.85 <= result.councils[0].probability <= 0.92


def test_gap_finder_page_shows_caveat_cards_chart_and_link(monkeypatch):
    import streamlit as st

    figures = []
    original = st.plotly_chart
    monkeypatch.setattr(st, "plotly_chart", lambda fig, *a, **k: (figures.append(fig), original(fig, *a, **k))[1])
    at = _page("Gap finder")
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "Gap finder" in page and "not the same as" in page and "past awards" in page
    assert "Under-explored directions near your idea" in page
    assert "gtr.ukri.org/projects?ref=" in page and "Funding Finder" in page
    assert figures, "the crowdedness-against-growth chart is missing"
    mine = [f.to_dict()["layout"] for f in figures if "Crowdedness" in ((f.to_dict()["layout"].get("xaxis") or {}).get("title") or {}).get("text", "")]
    assert mine, "the crowdedness-against-growth chart is missing"
    assert "Growth" in mine[-1]["yaxis"]["title"]["text"]


def test_3d_map_page_labels_axes_and_warns_about_distortion(monkeypatch):
    import streamlit as st

    figures = []
    original = st.plotly_chart
    monkeypatch.setattr(st, "plotly_chart", lambda fig, *a, **k: (figures.append(fig), original(fig, *a, **k))[1])
    at = _page("3D idea map")
    assert not at.exception
    maps = [f.to_dict() for f in figures if any(t["type"] == "scatter3d" for t in f.to_dict()["data"])]
    assert maps, "the 3D map was not drawn"
    data = maps[-1]
    scene = data["layout"]["scene"]
    for axis, n in (("xaxis", 1), ("yaxis", 2), ("zaxis", 3)):
        assert scene[axis]["title"]["text"].startswith(f"Semantic axis {n}")
    names = [t.get("name") for t in data["data"]]
    assert "Your idea" in names                                     # the highlighted marker
    page = " ".join(m.value for m in at.markdown)
    assert "distort" in page
    at.radio(key="map3d_colour").set_value("Gap-finder sub-topic").run()
    assert not at.exception


def test_numpy_kmeans_separates_blobs_and_is_deterministic():
    rng = np.random.RandomState(1)
    centres = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    x = np.vstack([c + rng.normal(0, 0.3, size=(30, 2)) for c in centres])
    labels, found = gaps._kmeans(x, 4)
    assert sorted(np.bincount(labels)) == [30, 30, 30, 30]
    assert all(len(set(labels[i * 30:(i + 1) * 30])) == 1 for i in range(4))
    again, _ = gaps._kmeans(x, 4)
    assert np.array_equal(labels, again)
    assert "sklearn.cluster" not in open(gaps.__file__, encoding="utf-8").read().replace("`sklearn.cluster`", "")
