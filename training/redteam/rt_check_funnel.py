"""RT-07: check that the packaged cleaning code reproduces the published funnel on the raw pull."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fundmatch.pipeline.preprocess import FUNNEL, clean_frame_with_report  # noqa: E402
from rt_common import RAW_PARQUET, save_result  # noqa: E402

raw = pd.read_parquet(RAW_PARQUET, columns=["title", "abstract", "lead_funder", "award_gbp"])
clean, report = clean_frame_with_report(raw)
got = report["rows"].tolist()
want = [row["rows"] for row in FUNNEL]
print(report.to_string(index=False))
save_result("funnel_check", {"package_rows": got, "published_rows": want, "match": got == want})
print("match", got == want)
