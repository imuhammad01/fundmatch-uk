"""Cleaning rules that took the API pull from 158,710 rows to 113,233.

The counts below are the verified coursework funnel. ``clean_frame`` applies
the same rules to a dataframe that already has ``lead_funder``, ``title`` and
``abstract`` columns, so the logic can be tested without the raw API dump.
"""

from __future__ import annotations

import re

import pandas as pd

COUNCILS = ["AHRC", "BBSRC", "EPSRC", "ESRC", "Innovate UK", "MRC", "NERC", "STFC"]
PLACEHOLDER = re.compile(
    r"abstracts are not currently available|no abstract available|not required at the time of proposal",
    re.IGNORECASE,
)

# Verified funnel from summary.json (23 Nov 2026).
FUNNEL = [
    {"step": "raw projects from API", "rows": 158710, "removed": 0},
    {"step": "keep 8 councils", "rows": 141538, "removed": 17172},
    {"step": "drop placeholder abstracts", "rows": 125815, "removed": 15723},
    {"step": "drop abstracts under 30 words", "rows": 117986, "removed": 7829},
    {"step": "drop boilerplate reused by 5 or more titles", "rows": 117759, "removed": 227},
    {"step": "drop duplicates with conflicting funders", "rows": 117719, "removed": 40},
    {"step": "drop exact duplicates", "rows": 113233, "removed": 4486},
]


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the coursework filters. Does not call the network."""
    out = frame.loc[frame["lead_funder"].isin(COUNCILS)].copy()
    abstract = out["abstract"].fillna("")
    out = out.loc[~abstract.map(lambda text: bool(PLACEHOLDER.search(text)))].copy()
    abstract = out["abstract"].fillna("")
    out = out.loc[abstract.map(lambda text: len(str(text).split()) >= 30)].copy()
    key = out["abstract"].fillna("").map(lambda text: " ".join(str(text).lower().split()))
    boilerplate = set(key.value_counts()[lambda s: s >= 5].index)
    out = out.loc[~key.isin(boilerplate)].copy()
    group = out["title"].fillna("").str.lower() + "||" + out["abstract"].fillna("").str.lower().str.slice(0, 400)
    conflict = set(out.groupby(group)["lead_funder"].nunique().loc[lambda s: s > 1].index)
    out = out.loc[~group.isin(conflict)].copy()
    out = out.drop_duplicates(["title", "abstract", "lead_funder"])
    return out.reset_index(drop=True)
