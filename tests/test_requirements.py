"""Tests added by the requirements audit (docs/REQUIREMENTS_AUDIT.md).

Each test names the requirement it checks. None of them calls the network.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fundmatch.pipeline import collect_gtr as cg  # noqa: E402
from fundmatch.pipeline.preprocess import (  # noqa: E402
    COUNCILS, FUNNEL, clean_frame, clean_frame_with_report, date_filter_report, normalise_funder,
)


def _page(n_pages: int, page: int) -> dict:
    return {"totalPages": n_pages, "totalSize": n_pages * 2, "page": page,
            "project": [{"id": f"p{page}a"}, {"id": f"p{page}b"}]}


# ---- FR-1 / FR-3 / FR-4: collector ------------------------------------------------

def test_fr1_url_uses_v7_endpoint_and_page_size():
    url = cg.page_url("projects", 3)
    assert url == "https://gtr.ukri.org/gtr/api/projects?p=3&s=100"
    assert cg.ACCEPT == "application/vnd.rcuk.gtr.json-v7"


def test_nfr9_person_endpoint_is_refused():
    with pytest.raises(ValueError):
        cg.page_url("persons", 1)


def test_fr3_rate_limiter_keeps_one_to_two_requests_per_second():
    assert 0.5 <= cg.PAUSE_S <= 1.0  # 1-2 requests a second
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = cg.RateLimiter(0.6, clock=lambda: now[0], sleep=sleep)
    for _ in range(4):
        limiter.wait()
    assert slept == pytest.approx([0.6, 0.6, 0.6])


def test_fr3_retries_with_exponential_backoff_then_succeeds():
    calls = {"n": 0}
    slept: list[float] = []

    def getter(url):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(url, 503, "busy", {}, io.BytesIO(b""))
        return _page(1, 1)

    data = cg.fetch_page("projects", 1, pause=0, getter=getter, sleep=slept.append)
    assert data["page"] == 1 and calls["n"] == 3
    backoffs = [s for s in slept if s > 0]
    assert backoffs == [1.0, 2.0]


def test_fr3_non_retryable_error_is_raised():
    def getter(url):
        raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(b""))

    with pytest.raises(urllib.error.HTTPError):
        cg.fetch_page("projects", 1, pause=0, getter=getter, sleep=lambda s: None)


def test_fr3_fr4_collect_pages_checkpoints_resumes_and_logs(tmp_path):
    fetched: list[int] = []

    def fetch(endpoint, page):
        fetched.append(page)
        return _page(4, page)

    dest = tmp_path / "raw" / "projects"
    first = cg.collect("projects", dest, pause=0, fetch=fetch)
    assert fetched == [1, 2, 3, 4]
    assert first["api_total_pages"] == 4 and first["pages_fetched_this_run"] == 4
    (dest / "00003.json").unlink()  # simulate an interrupted run
    fetched.clear()
    second = cg.collect("projects", dest, pause=0, fetch=fetch)
    assert fetched == [3]  # only the missing page is fetched again
    assert second["pages_fetched_this_run"] == 1
    log = json.loads((tmp_path / "raw" / "collection_log.json").read_text())
    assert log["projects"]["api_total_records"] == 8
    assert log["projects"]["collected_utc"].startswith("20")
    assert len(list(cg.iter_records(dest))) == 8
    assert not list(dest.glob("*.tmp"))


def test_fr4_dated_snapshot_and_sidecar(tmp_path):
    frame = pd.DataFrame({"project_id": ["a", "b"], "title": ["x", "y"]})
    meta = cg.save_snapshot(frame, tmp_path, collected_utc="2026-11-23T19:35:16+00:00")
    assert meta["file"] == "gtr_projects_2026-11-23.parquet" and meta["rows"] == 2
    assert len(pd.read_parquet(tmp_path / meta["file"])) == 2
    assert json.loads((tmp_path / "gtr_projects_2026-11-23.json").read_text())["rows"] == 2


def test_fr4_shipped_collection_log_records_date_and_counts():
    log = json.loads((ROOT / "training" / "logs" / "collection_log.json").read_text())
    assert log["projects"]["api_total_records"] == 158710
    assert log["projects"]["collected_utc"].startswith("2026-11-23")
    assert log["projects"]["delay_s"] >= 0.5


# ---- FR-6 / FR-7 / FR-8: preprocessing --------------------------------------------

def test_fr7_funder_labels_normalised():
    assert normalise_funder("epsrc") == "EPSRC"
    assert normalise_funder(" Innovate  UK ") == "Innovate UK"
    assert normalise_funder("Technology Strategy Board") == "Innovate UK"
    assert normalise_funder("Medical Research Council") == "MRC"
    for programme in ["ISCF", "GCRF", "Horizon Europe Guarantee", None, float("nan")]:
        assert normalise_funder(programme) == "Other"


def test_fr6_report_counts_every_step():
    words = " ".join(f"word{i}" for i in range(40))
    rows = [
        {"lead_funder": "epsrc", "title": "A", "abstract": words + " alpha"},
        {"lead_funder": "EPSRC", "title": "A", "abstract": words + " alpha"},  # exact duplicate
        {"lead_funder": "GCRF", "title": "B", "abstract": words + " beta"},  # programme funder
        {"lead_funder": "MRC", "title": "C", "abstract": "Abstracts are not currently available in GtR " + words},
        {"lead_funder": "NERC", "title": "D", "abstract": "too short"},
        {"lead_funder": "NERC", "title": "E", "abstract": ""},
        {"lead_funder": "AHRC", "title": "F", "abstract": words + " gamma"},
        {"lead_funder": "ESRC", "title": "F", "abstract": words + " gamma"},  # conflicting funders
    ]
    clean, report = clean_frame_with_report(pd.DataFrame(rows))
    removed = dict(zip(report["step"], report["removed"]))
    assert report["rows"].iloc[0] == 8 and report["rows"].iloc[-1] == len(clean) == 1
    assert removed["keep 8 councils"] == 1
    assert removed["drop placeholder abstracts"] == 1
    assert removed["drop empty or abstracts under 30 words"] == 2
    assert removed["drop duplicates with conflicting funders"] == 2
    assert removed["drop exact duplicates"] == 1
    assert int(report["removed"].sum()) == 7
    assert set(clean["lead_funder"]) <= set(COUNCILS)
    assert len(clean_frame(pd.DataFrame(rows))) == 1


def test_fr6_published_funnel_is_consistent():
    for before, after in zip(FUNNEL, FUNNEL[1:]):
        assert before["rows"] - after["removed"] == after["rows"]
    assert FUNNEL[-1]["rows"] == 113233


def test_fr8_date_filter_reports_counts():
    frame = pd.DataFrame({"start_year": [2005, 2010, 2020, 2026, 2027, None]})
    kept, report = date_filter_report(frame, first_year=2006, snapshot_year=2026)
    assert list(kept["start_year"]) == [2010, 2020]
    counts = dict(zip(report["rule"], report["rows"]))
    assert counts["rows in"] == 6 and counts["rows kept"] == 2
    assert counts["started 2026 or later (too recent)"] == 2


# ---- FR-11 .. FR-18 evidence shipped with the repo ---------------------------------

@pytest.fixture(scope="module")
def summary():
    return json.loads((ROOT / "training" / "models" / "summary.json").read_text())


def test_fr12_nfr1_deployed_model_beats_baselines(summary):
    res = {row["model"]: row for row in summary["results"]}
    deployed = res["TF-IDF + LogReg"]
    assert deployed["macro_F1"] > res["Majority class"]["macro_F1"]
    assert deployed["macro_F1"] > res["TF-IDF + ComplementNB"]["macro_F1"]
    assert {"TF-IDF + LinearSVM", "MINILM emb + LogReg", "BGE emb + LogReg"} <= set(res)
    assert deployed["top3_acc"] > 0.95


def test_fr11_learning_curve_and_fr10_time_split(summary):
    curve = summary["learning_curve"]
    assert curve[-1]["n_train"] == 90585 and len(curve) >= 5
    assert curve[-1]["TF-IDF + LogReg"] > curve[0]["TF-IDF + LogReg"]
    assert summary["temporal"][0]["macro_F1 (train <2021, test >=2021)"] > 0.8


def test_fr13_fr14_retrieval_and_award_metrics(summary):
    retrievers = {row["retriever"] for row in summary["retrieval_metrics"]}
    assert {"TF-IDF cosine", "MINILM cosine", "BGE cosine"} <= retrievers
    assert all("Recall@10" in row and "MRR@10" in row for row in summary["retrieval_metrics"])
    models = " ".join(row["model"] for row in summary["award_regression"])
    assert "Ridge" in models and "LightGBM" in models
    assert any(row["coverage"] for row in summary["award_regression"])


def test_fr15_per_class_f1_and_fr16_tuning(summary):
    info = json.loads((ROOT / "artifacts" / "model_info.json").read_text())
    assert set(info["per_class_f1"]) == set(COUNCILS)
    assert summary["tuning"]["lr"][0] == 32.0


def test_fr17_deployment_table_lists_runners_up():
    table = pd.read_csv(ROOT / "artifacts" / "deployment_selection.csv")
    assert table.iloc[0]["configuration"] == "TF-IDF + LogReg"
    assert {"peak RAM MB", "latency ms (median)", "disk MB"} <= set(table.columns)
    assert len(table) >= 5


def test_nfr5_training_notebook_has_outputs_and_no_errors():
    nb = json.loads((ROOT / "training" / "FundMatch_Task1.ipynb").read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) >= 30 and all(c.get("outputs") for c in code)
    assert not any(o.get("output_type") == "error" for c in code for o in c["outputs"])


def test_nfr13_no_obvious_secrets_in_repo():
    import re

    patterns = [
        re.compile(r"gh[po]_[A-Za-z0-9]{30,}"),
        re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
        re.compile(r"sk-[A-Za-z0-9]{32,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile("-----BEGIN " + r"(RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    ]
    for path in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.toml")) + list(ROOT.rglob("*.md")):
        if ".venv" in path.parts or "site-packages" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            assert not pattern.search(text), f"{pattern.pattern} in {path}"


def test_nfr9_shipped_table_has_no_person_columns():
    columns = set(pd.read_parquet(ROOT / "data" / "projects_meta.parquet").columns)
    assert not {c for c in columns if any(k in c.lower() for k in ("person", "investigator", "pi_", "researcher"))}
