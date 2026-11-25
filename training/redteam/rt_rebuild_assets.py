"""RT-01 follow-up: rebuild the conformal quantiles and novelty reference with the fixed
vectoriser, and re-measure int8 fidelity and engine timing. Writes results/assets_rebuild.json."""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
started = time.perf_counter()
from fundmatch.engine import FundMatchEngine  # noqa: E402
from fundmatch.intelligence import build_assets  # noqa: E402
from fundmatch.quantize import export_int8, fidelity  # noqa: E402
from fundmatch.cli import DEMOS  # noqa: E402
from rt_common import save_result  # noqa: E402

before = json.loads((ROOT / "artifacts/conformal_aps.json").read_text())
nov_before = np.load(ROOT / "artifacts/novelty_reference.npz")["max_sim"]
engine = FundMatchEngine()
cold = time.perf_counter() - started
assets = build_assets(engine)
after = json.loads((ROOT / "artifacts/conformal_aps.json").read_text())
nov_after = np.load(ROOT / "artifacts/novelty_reference.npz")["max_sim"]

with tempfile.TemporaryDirectory() as tmp:
    report = export_int8(engine, Path(tmp) / "w.npz")
hat = report.pop("coef_hat")
texts45 = [f"{d['title']}. {d['abstract']}" for d in DEMOS] + engine.frame["title"].head(40).tolist()
rng = np.random.default_rng(7)
idx = rng.choice(len(engine.frame), size=2000, replace=False)
texts2k = [f"{engine.frame.title.iloc[i]}. {engine.frame.grant_category.iloc[i]}." for i in idx]
times = []
for d in DEMOS:
    times.append(engine.analyze(d["title"], d["abstract"]).latency_ms)
for _ in range(3):
    for d in DEMOS:
        times.append(engine.analyze(d["title"], d["abstract"]).latency_ms)
save_result("assets_rebuild", {
    "conformal_before": {k: before[k] for k in ("quantile_90", "quantile_95", "score_median")},
    "conformal_after": {k: after[k] for k in ("quantile_90", "quantile_95", "score_median")},
    "novelty_median_before": float(np.median(nov_before)), "novelty_median_after": float(np.median(nov_after)),
    "int8_fidelity_45_texts": fidelity(engine, texts45, hat),
    "int8_fidelity_2000_titles": fidelity(engine, texts2k, hat),
    "int8_max_abs_coef_error": report["max_abs_coef_error"],
    "cold_start_s_incl_imports": round(cold, 3), "engine_cold_s": round(engine.cold_s, 3),
    "warm_latency_ms_median": float(np.median(times[len(DEMOS):])),
    "warm_latency_ms_p95": float(np.percentile(times[len(DEMOS):], 95)),
    "rss_mb": engine.rss_mb(), "assets": assets,
})
print(open(Path(__file__).parent / "results/assets_rebuild.json").read())
