"""Engine, API, CLI and Streamlit checks against the real 113,233-project index."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from typer.testing import CliRunner

from fundmatch.cli import app as cli
from fundmatch.demos import DEMOS
from fundmatch.engine import TEMPERATURE, get_engine
from fundmatch.pipeline.preprocess import clean_frame
from fundmatch.pipeline.train import temperature_scale
from fundmatch.quantize import export_int8, fidelity
from fundmatch.text import check_input, english_share, mask_funders

ROOT = Path(__file__).resolve().parents[1]
BANNED = ("researcher", "pi_name", "orcid", "investigator", "surname")
_SKIP_DIRS = {".venv", "venv", "env", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}


def _project_files(pattern: str) -> list[Path]:
    """Project files only. A professor's .venv must not fail the packaging checks."""
    return [path for path in ROOT.rglob(pattern) if _SKIP_DIRS.isdisjoint(path.parts)]

TECHNICAL = (
    "Low-temperature hydrometallurgical recovery of LiCoO2 and Ni-rich NMC-811 cathodes "
    "uses a choline chloride–urea deep eutectic solvent. We will measure leaching kinetics "
    "for cobalt, nickel and lithium, compare the process energy with pyrometallurgy, and "
    "publish the flowsheet. Species names such as Escherichia coli are not in this study. "
    "The work is chemical engineering in a UK laboratory, with UK spelling of sulphide where "
    "it appears in the ore mineralogy, and with formulae written as Li2CO3 and Co(OH)2."
)


@pytest.fixture(scope="session")
def engine():
    return get_engine()


def test_short_text_is_an_error():
    errors, warnings = check_input("Too short", "Only a few words.")
    assert errors
    assert not warnings


def test_technical_english_is_not_flagged():
    errors, warnings = check_input("Cathode recycling", TECHNICAL)
    assert not errors
    assert not warnings
    assert english_share(f"Cathode recycling. {TECHNICAL}") >= 0.15
    for demo in DEMOS:
        demo_errors, demo_warnings = check_input(demo["title"], demo["abstract"])
        assert not demo_errors
        assert not any("does not look like English" in item for item in demo_warnings)


def test_german_and_gibberish_warn_but_do_not_block():
    german = (
        "Diese Untersuchung betrachtet die langfristige Entwicklung regionaler Arbeitsmärkte "
        "nach großen Betriebsschließungen und vergleicht anschließend die Einkommensverläufe "
        "von betroffenen Haushalten über einen Zeitraum von zehn Jahren anhand administrativer "
        "Lohnunterlagen aus drei mittelgroßen Städten in verschiedenen Bundesländern mit "
        "unterschiedlicher Industriestruktur und unterschiedlicher Pendelentfernung."
    )
    errors, warnings = check_input("Arbeitsmarkt", german)
    assert not errors
    assert warnings and "English" in warnings[0]
    errors, warnings = check_input("xqv", "blorpt zindle quax florn mipp glarf sned wobble plix snard " * 4)
    assert not errors
    assert warnings


def test_mask_replaces_council_names():
    masked = mask_funders("The EPSRC and Innovate UK programme on photonics")
    assert "EPSRC" not in masked
    assert "Innovate" not in masked
    assert "photonics" in masked
    assert "funderx" in masked


def test_bit_exact_logits(engine):
    meta = json.loads((ROOT / "artifacts" / "tfidf_lr_meta.json").read_text())
    weights = np.load(ROOT / "artifacts" / "tfidf_lr_weights.npz")
    vocab = {term: i for i, term in enumerate(meta["vocab"])}
    vector = TfidfVectorizer(
        ngram_range=tuple(meta["ngram_range"]), lowercase=True, token_pattern=meta["token_pattern"],
        sublinear_tf=True, norm="l2", use_idf=True, smooth_idf=True,
        stop_words="english", vocabulary=vocab, dtype=np.float64,  # as trained (RT-01)
    )
    vector.idf_ = weights["idf"].astype(np.float64)
    vector._tfidf._idf_diag = sp.diags(vector.idf_, offsets=0, format="csr", dtype=np.float64)
    text = mask_funders(f"{DEMOS[0]['title']}. {DEMOS[0]['abstract']}")
    sklearn_logits = np.asarray(vector.transform([text]) @ weights["coef"].T).ravel() + weights["intercept"]
    ours = engine.logits(engine.vectorize([text]), scaled=False)[0]
    assert np.max(np.abs(sklearn_logits - ours)) <= 1e-6
    probs = engine.predict_proba_masked(text)
    assert abs(probs.sum() - 1) < 1e-6
    # Temperature on probabilities matches logits divided by T.
    raw = np.exp(sklearn_logits - sklearn_logits.max())
    raw = raw / raw.sum()
    scaled = temperature_scale(raw, TEMPERATURE)
    direct = np.exp((sklearn_logits / TEMPERATURE) - (sklearn_logits / TEMPERATURE).max())
    direct = direct / direct.sum()
    assert np.max(np.abs(scaled - direct)) < 1e-8
    assert engine.temperature == pytest.approx(1.1314305006097425)


def test_confidence_tiers(engine):
    strong = engine.analyze(DEMOS[1]["title"], DEMOS[1]["abstract"])
    assert strong.ok
    assert strong.confidence in {"high", "moderate", "low"}
    assert abs(sum(row.probability for row in strong.councils) - 1) < 1e-5
    assert strong.councils[0].funding_finder_url.startswith("https://www.ukri.org/opportunity/?filter_council")
    # A near-empty-but-long string of rare tokens should not be high confidence.
    # A long string that repeats one word is now refused, not scored (RT-03).
    vague = engine.analyze("Notes", "something something " * 20)
    assert not vague.ok and vague.errors


def test_explain_and_contrast(engine):
    view = engine.explain_council(DEMOS[0]["title"], DEMOS[0]["abstract"], "EPSRC")
    assert view.positive
    text = f"{DEMOS[0]['title']}. {DEMOS[0]['abstract']}"
    for span in view.highlights:
        assert text[span.start:span.end].lower() == span.feature
        assert span.contribution != 0
    contrast = engine.contrast(DEMOS[4]["title"], DEMOS[4]["abstract"], "EPSRC", "Innovate UK")
    assert contrast.toward_a or contrast.toward_b
    assert contrast.council_a == "EPSRC" and contrast.council_b == "Innovate UK"


def test_retrieval_filters_and_privacy(engine):
    result = engine.analyze(DEMOS[2]["title"], DEMOS[2]["abstract"], top_k=10)
    assert len(result.similar) == 10
    assert len(engine.frame) == 113233
    blob = result.model_dump_json().lower()
    for banned in BANNED:
        assert banned not in blob
    for row in result.similar:
        assert row.gtr_url.startswith("https://gtr.ukri.org/projects?ref=")
        assert 0 <= row.similarity <= 1.01
    filtered = engine.analyze(
        DEMOS[2]["title"], DEMOS[2]["abstract"], councils=["NERC"], year_min=2012, year_max=2022,
        scheme="Research Grant", same_scheme=True, top_k=8,
    )
    assert len(filtered.similar) == 8
    for row in filtered.similar:
        assert row.lead_funder == "NERC"
        assert row.grant_category == "Research Grant"
        assert row.start_year is None or 2012 <= row.start_year <= 2022


def test_same_scheme_and_cpih_recompute(engine):
    base = engine.analyze(DEMOS[0]["title"], DEMOS[0]["abstract"])
    scheme = engine.analyze(
        DEMOS[0]["title"], DEMOS[0]["abstract"], scheme="Studentship", same_scheme=True,
    )
    real = engine.analyze(DEMOS[0]["title"], DEMOS[0]["abstract"], cpih=True)
    assert base.award and scheme.award and real.award
    assert base.award.basis == "nominal" and real.award.basis == "real_2026"
    assert base.award.n_zero_excluded >= 0
    assert "£0" in base.award.zero_note
    assert "not a funding quote" in base.award.caveat
    # Studentships are a different money distribution from the mixed neighbour set.
    assert (scheme.award.median, scheme.award.p25, scheme.award.p75) != (base.award.median, base.award.p25, base.award.p75)
    assert real.award.median != base.award.median
    assert real.award.p10 != base.award.p10
    assert real.trends and base.trends
    assert real.trends[0].total_gbp != base.trends[0].total_gbp or real.award.median != base.award.median


def test_int8_fidelity(engine, tmp_path):
    report = export_int8(engine, tmp_path / "w.npz")
    texts = [f"{item['title']}. {item['abstract']}" for item in DEMOS]
    texts += engine.frame["title"].head(30).tolist()
    stats = fidelity(engine, texts, report["coef_hat"])
    assert stats["top1_agreement"] >= 0.9
    assert stats["max_probability_delta"] < 0.05
    assert report["int8_file_bytes"] < report["fp32_coef_bytes"]


def test_cli_commands():
    runner = CliRunner()
    analyzed = runner.invoke(cli, ["analyze", "--demo", "epsrc-quantum", "--top-k", "5"])
    assert analyzed.exit_code == 0, analyzed.output
    assert "EPSRC" in analyzed.output or "Top council" in analyzed.output
    searched = runner.invoke(cli, ["search", "--query", "peatland carbon drought water table", "--top-k", "3", "--council", "NERC"])
    assert searched.exit_code == 0, searched.output
    explained = runner.invoke(cli, ["explain", "--demo", "borderline", "--council", "EPSRC", "--vs", "Innovate UK"])
    assert explained.exit_code == 0, explained.output
    bench = runner.invoke(cli, ["benchmark"])
    assert bench.exit_code == 0, bench.output
    assert "warm_latency_ms" in bench.output
    demonstrated = runner.invoke(cli, ["demo", "mrc-biomedical"])
    assert demonstrated.exit_code == 0, demonstrated.output


def test_api_routes():
    from fundmatch.api import app as api

    client = TestClient(api)
    health = client.get("/health").json()
    assert health["projects"] == 113233
    assert health["snapshot"] == "2026-11-23"
    body = {"title": DEMOS[3]["title"], "abstract": DEMOS[3]["abstract"], "top_k": 5}
    analyzed = client.post("/api/v1/analyze", json=body).json()
    assert analyzed["ok"] and abs(sum(row["probability"] for row in analyzed["councils"]) - 1) < 1e-5
    predicted = client.post("/api/v1/predict", json=body).json()
    assert predicted["top_council"]
    explained = client.post("/api/v1/explain", json={**body, "council": "AHRC", "versus": "ESRC"}).json()
    assert explained["explanation"]["council"] == "AHRC"
    assert "contrast" in explained
    found = client.post("/api/v1/search", json={"query": "parish archives handwriting", "top_k": 4, "councils": ["AHRC"]}).json()
    assert found["results"]
    assert all(row["lead_funder"] == "AHRC" for row in found["results"])
    benchmarks = client.get("/api/v1/benchmarks").json()
    assert benchmarks["model_info"]["test_macro_f1"] == 0.8615
    stats = client.get("/api/v1/corpus/stats").json()
    assert stats["projects"] == 113233
    assert len(client.get("/api/v1/demos").json()["demos"]) == 10
    what = client.post("/api/v1/whatif", json={**body, "target_council": "EPSRC", "add_terms": "archive handwriting"}).json()
    assert what["simulation"]["deltas"]
    assert "honesty" in what["simulation"]
    compared = client.post("/api/v1/compare", json={"proposals": [
        {"label": "a", "title": DEMOS[0]["title"], "abstract": DEMOS[0]["abstract"]},
        {"label": "b", "title": DEMOS[1]["title"], "abstract": DEMOS[1]["abstract"]},
    ]}).json()
    assert len(compared["rows"]) == 2
    batched = client.post("/api/v1/batch", json={"records": [{"id": "1", "title": DEMOS[2]["title"], "abstract": DEMOS[2]["abstract"]}]}).json()
    assert batched["n"] == 1
    telemetry = client.get("/api/v1/metrics").json()
    assert telemetry["stored_abstracts"] == 0
    assert telemetry["projects"] == 113233


def test_preprocess_rules_drop_short_and_other_funders():
    import pandas as pd
    frame = pd.DataFrame([
        {"lead_funder": "EPSRC", "title": "A", "abstract": "word " * 40},
        {"lead_funder": "ISCF", "title": "B", "abstract": "word " * 40},
        {"lead_funder": "MRC", "title": "C", "abstract": "too short"},
    ])
    cleaned = clean_frame(frame)
    assert set(cleaned["lead_funder"]) == {"EPSRC"}


def test_streamlit_toggles_change_the_band():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.button(key="demo_epsrc-quantum").click().run()
    assert not at.exception
    medians = [m.value for m in at.metric if m.label == "Median"]
    assert medians, "the award median should be on the page after a demo"
    nominal = medians[0]
    at.toggle(key="cpih").set_value(True).run()
    assert not at.exception
    updated = [m.value for m in at.metric if m.label == "Median"][0]
    assert updated != nominal
    at.toggle(key="same_scheme").set_value(True).run()
    assert not at.exception
    scheme_value = [m.value for m in at.metric if m.label == "Median"][0]
    assert scheme_value


def test_conformal_novelty_constellation_and_hosts(engine):
    peaked = engine.analyze(DEMOS[0]["title"], DEMOS[0]["abstract"])
    flat = engine.analyze(
        "Notes",
        "This project studies people, places, data and change over time in many settings, working with "
        "partners to understand what matters, why it matters and how it might be improved for communities "
        "and organisations across the country.",
    )
    assert flat.ok
    assert peaked.conformal_set_90 == ["EPSRC"]
    assert peaked.top_council in peaked.conformal_set_95
    assert len(peaked.conformal_set_95) == 3
    expected = [flat.top_council] if flat.councils[0].probability >= 0.80 else [row.council for row in flat.councils[:3]]
    assert flat.conformal_set_90 == expected
    assert "0.943" in peaked.conformal_note or "94.3" in peaked.conformal_note
    assert peaked.novelty["band"] in {
        "Core UKRI Territory", "Emerging Interdisciplinary Frontier", "High-Novelty / Potential Out-of-Scope",
    }
    kinds = {point["kind"] for point in peaked.constellation}
    assert {"query", "neighbour", "centroid", "landmark"} <= kinds
    assert peaked.host_institutions
    assert peaked.scheme_fit
    blob = peaked.model_dump_json().lower()
    for banned in BANNED:
        assert banned not in blob
    assert peaked.similar[0].shared_terms is not None
    q = engine.analyze(DEMOS[0]["title"], DEMOS[0]["abstract"], quantized=True)
    assert q.precision == "int8" and q.top_council == "EPSRC"
    shift = engine.simulate_what_if(DEMOS[0]["title"], DEMOS[0]["abstract"], add_terms="parish archive handwriting", target_council="AHRC")
    assert shift["target"]["council"] == "AHRC"
    assert shift["target"]["after"] != shift["target"]["before"]
    assert "honest" in shift["honesty"].lower()
    arena = engine.compare_proposals([
        {"label": "q", "title": DEMOS[0]["title"], "abstract": DEMOS[0]["abstract"]},
        {"label": "m", "title": DEMOS[1]["title"], "abstract": DEMOS[1]["abstract"]},
    ])
    assert {row["top_council"] for row in arena["rows"]} >= {"EPSRC"}
    batch = engine.analyze_batch([{"id": "n", "title": DEMOS[2]["title"], "abstract": DEMOS[2]["abstract"]}])
    assert batch["rows"][0]["top_council"] == "NERC"


def test_new_cli_commands(tmp_path):
    runner = CliRunner()
    student = runner.invoke(cli, ["analyze", "--demo", "epsrc-quantum", "--student", "--cpih"])
    assert student.exit_code == 0, student.output
    assert "Student coach" in student.output
    assert "90% set" in student.output
    flipped = runner.invoke(cli, ["whatif", "--demo", "borderline", "--target", "MRC", "--add", "clinical cohort"])
    assert flipped.exit_code == 0, flipped.output
    assert "Delta" in flipped.output
    arena = runner.invoke(cli, ["compare", "--demos", "epsrc-quantum,mrc-biomedical,interdisciplinary-ai"])
    assert arena.exit_code == 0, arena.output
    table = tmp_path / "in.csv"
    table.write_text(
        "title,abstract\n"
        "Heat and carbon,\"We will measure heat content and particulate carbon export in the subpolar North Atlantic "
        "using autonomous gliders, shipboard hydrography and biogeochemical sensors along a repeat winter section west of the British Isles over four years.\"\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    batched = runner.invoke(cli, ["batch", "--input", str(table), "--output", str(out)])
    assert batched.exit_code == 0, batched.output
    assert out.exists() and "NERC" in out.read_text(encoding="utf-8")


def test_student_stage_tools_are_grounded(engine):
    from fundmatch.student import PERSONAS
    assert set(PERSONAS) == {"undergrad", "masters", "phd", "researcher", "international", "innovator"}
    interest = (
        "music information retrieval and machine learning for archive recordings of folk songs "
        "in British collections, with a clear evaluation on a public dataset"
    )
    topics = engine.suggest_topics(interest, "undergrad")
    assert len(topics["topics"]) == 5
    ids = {row["project_id"] for topic in topics["topics"] for row in topic["precedents"]}
    assert ids <= set(engine.frame["project_id"].astype(str))
    trend = engine.topic_trends(interest)
    assert trend["band"] in {"Surging", "Rising", "Steady", "Mature", "insufficient"}
    assert trend["series"]
    score = engine.readiness(DEMOS[0]["title"], DEMOS[0]["abstract"], "masters")
    assert 0 <= score["overall"] <= 100
    assert len(score["dimensions"]) == 5
    fused = engine.fuse("nursing", "robotics")
    assert fused["projects"]
    blob = json.dumps(fused).lower()
    for banned in ("pi_name", "orcid", "surname"):
        assert banned not in blob
    pack = engine.starter_pack(DEMOS[3]["title"], DEMOS[3]["abstract"], "international")
    assert "Open Government Licence" in pack["markdown"]
    assert "Twelve-week" in pack["markdown"]
    assert engine.rss_mb() > 0


def test_v3_api_and_cli(tmp_path):
    from fundmatch.api import app as api
    client = TestClient(api)
    body = {"interest": "wheat drought genetics and controlled water table experiments in the uk", "persona": "masters"}
    assert len(client.post("/api/v1/topics", json=body).json()["topics"]) == 5
    assert client.post("/api/v1/trends", json=body).json()["band"]
    fused = client.post("/api/v1/fuse", json={"field_a": "music", "field_b": "machine learning"}).json()
    assert fused["projects"]
    ready = client.post("/api/v1/readiness", json={"title": DEMOS[0]["title"], "abstract": DEMOS[0]["abstract"], "persona": "phd"}).json()
    assert ready["overall"] > 0
    pack = client.post("/api/v1/starter-pack", json={"title": DEMOS[5]["title"], "abstract": DEMOS[5]["abstract"], "persona": "phd"}).json()
    assert pack["markdown"]
    assert len(client.get("/api/v1/personas").json()["personas"]) == 6
    runner = CliRunner()
    listed = runner.invoke(cli, ["topics", "--interest", body["interest"], "--persona", "innovator"])
    assert listed.exit_code == 0, listed.output
    wave = runner.invoke(cli, ["trends", "--interest", body["interest"]])
    assert wave.exit_code == 0, wave.output
    bridge = runner.invoke(cli, ["fuse", "--a", "law", "--b", "climate"])
    assert bridge.exit_code == 0, bridge.output
    scored = runner.invoke(cli, ["score", "--demo", "epsrc-robotics", "--persona", "undergrad"])
    assert scored.exit_code == 0, scored.output
    dest = tmp_path / "pack.md"
    packed = runner.invoke(cli, ["starter-pack", "--demo", "bbsrc-genetics", "--persona", "masters", "--output", str(dest)])
    assert packed.exit_code == 0, packed.output
    assert dest.exists()
    assert not _project_files("*.pkl")


def test_working_titles_are_unique_and_scored(engine):
    topics = engine.suggest_topics(f"{DEMOS[0]['title']}. {DEMOS[0]['abstract']}", "undergrad")
    rows = topics["topics"]
    assert len(rows) == 5
    titles = [row["working_title"].strip() for row in rows]
    assert all(titles)
    assert len({title.lower() for title in titles}) == 5
    bands = {
        "Core UKRI Territory",
        "Emerging Interdisciplinary Frontier",
        "High-Novelty / Potential Out-of-Scope",
    }
    for row in rows:
        assert row["predicted_council"] in engine.classes
        assert row["calibrated_probability"] is not None
        assert 0 < float(row["calibrated_probability"]) <= 1
        assert row["confidence"] in {"high", "moderate", "low"}
        assert row["novelty_band"] in bands
        assert "none" not in row["working_title"].lower().split()


def test_topics_endpoint_is_fast_and_openapi_is_v3():
    from fundmatch.api import app as api

    client = TestClient(api)
    body = {
        "interest": "superconducting silicon quantum processors and microwave qubit control",
        "persona": "masters",
    }
    warmed = client.post("/api/v1/topics", json=body)
    assert warmed.status_code == 200
    started = time.perf_counter()
    response = client.post("/api/v1/topics", json=body)
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < 1.5, elapsed
    rows = response.json()["topics"]
    assert len(rows) == 5
    assert len({row["working_title"].lower() for row in rows}) == 5
    assert all(row["predicted_council"] and row["calibrated_probability"] is not None and row["novelty_band"] for row in rows)
    assert api.version == "4.0.0"
    assert api.openapi()["info"]["version"] == "4.0.0"
    assert api.title == "ScholarPulse UK API"


def test_scholarpulse_brand_coach_redteam_hubs_and_dossier(engine):
    import re
    from fundmatch.brand import APP_DISPLAY_NAME, APP_VERSION
    from fundmatch.futuristic import coach, hubs, redteam, research_dossier

    assert APP_DISPLAY_NAME == "ScholarPulse UK"
    assert APP_VERSION == "4.0.0"
    title, abstract = DEMOS[0]["title"], DEMOS[0]["abstract"]
    guided = coach(engine, title, abstract, "masters")
    assert 6 <= len(guided["missing_signals"]) <= 8
    assert guided["before"]["council"] in engine.classes
    assert guided["after"]["council"] in engine.classes
    assert guided["before"]["readiness"] is not None and guided["after"]["readiness"] is not None
    assert isinstance(guided["delta_readiness"], float)
    assert guided["roadmap"]["timeline"] and guided["roadmap"]["checklist"]
    assert "[Professor / Supervisor Name]" in guided["roadmap"]["email_template"]
    assert "simulation" in guided["simulation_note"].lower()
    review = redteam(engine, title, abstract, "phd")
    assert len(review["questions"]) == 5
    assert len(review["matrix"]) == 5
    assert {row["dimension"] for row in review["matrix"]} == {
        "Novelty Differentiation", "Stage Scope Fit", "Method Clarity", "Council Positioning", "Field Momentum",
    }
    assert all(item["question"] and item["why"] and item["defence"] for item in review["questions"])
    centres = hubs(engine, title, abstract)
    assert len(centres["hubs"]) == 10
    assert all(row["organisation"] and row["awards"] > 0 and row["total_gbp_real_2026"] >= 0 for row in centres["hubs"])
    pack = research_dossier(engine, title, abstract, "international")
    assert "Open Government Licence v3.0" in pack["markdown"]
    assert "Five working titles" in pack["markdown"]
    assert "Md Shamim Mia" in pack["markdown"]
    blob = json.dumps({"coach": guided, "redteam": review, "hubs": centres, "dossier": pack["markdown"]})
    for banned in ("pi_name", "orcid", "surname"):
        assert banned not in blob.lower()
    assert re.search(r"\b(?:Prof|Dr)\.?\s+[A-Z]", blob.replace("[Professor / Supervisor Name]", "")) is None
    from fundmatch.api import app as api
    client = TestClient(api)
    body = {"title": title, "abstract": abstract, "persona": "undergrad"}
    for path in ("/api/v1/coach", "/api/v1/redteam", "/api/v1/hubs", "/api/v1/topics"):
        assert client.post(path, json=body).status_code == 200
        started = time.perf_counter()
        response = client.post(path, json=body)
        assert time.perf_counter() - started < 1.5, path
        assert response.status_code == 200
    surfaces = [ROOT / "app.py", ROOT / "README.md", ROOT / "SUBMISSION_CHECKLIST.md", ROOT / "src"]
    text = []
    for surface in surfaces:
        if surface.is_dir():
            for path in _project_files("*.py"):
                if surface in path.parents or path == surface:
                    text.append(path.read_text(encoding="utf-8"))
        elif surface.exists():
            text.append(surface.read_text(encoding="utf-8"))
    joined = "\n".join(text).replace("[Professor / Supervisor Name]", "")
    assert re.search(r"\b(?:Prof|Dr)\.?\s+[A-Z]", joined) is None
    assert not _project_files("*.pkl")
    for pattern in ("*.js", "*.jsx", "*.ts", "*.tsx"):
        assert not _project_files(pattern)
    assert not (ROOT / "package.json").exists()
