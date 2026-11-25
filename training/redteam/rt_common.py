"""Shared helpers for the red-team evaluations (docs/RED_TEAM_AUTOPSY.md).

``build_split()`` repeats the notebook's Section 3 cleaning and Section 4
group-aware stratified split exactly (same rules, same seed 42), so every
red-team number is measured on the same train/test sets as the published
results. Run from ``fundmatch-uk/training`` after ``build_dataset.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

SEED = 42
HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
RAW_PARQUET = Path(os.environ.get("FM_RAW_PARQUET", TRAINING / "data" / "processed" / "gtr_projects.parquet"))
CACHE = Path(os.environ.get("FM_RT_CACHE", TRAINING / "data" / "redteam_cache"))
RESULTS = HERE / "results"
COUNCILS = ["AHRC", "BBSRC", "EPSRC", "ESRC", "Innovate UK", "MRC", "NERC", "STFC"]
PLACEHOLDER = re.compile(r"(?:abstracts are not currently available|awaiting public project summary|no public description|"
                         r"public description for this project has been requested|^\s*(?:tbc|tba|to be confirmed|n/?a|none|"
                         r"no abstract available\.?)\s*$|doctoral training partnerships: a range of postgraduate)", re.I)
FUNDER_TERMS = [r"engineering and physical sciences research council", r"biotechnology and biological sciences research council",
    r"economic and social research council", r"medical research council", r"arts and humanities research council",
    r"natural environment research council", r"science and technology facilities council", r"innovate ?uk",
    r"technology strategy board", r"uk research and innovation", r"research councils? uk",
    r"epsrc", r"bbsrc", r"esrc", r"mrc", r"ahrc", r"nerc", r"stfc", r"ukri", r"rcuk", r"ktp",
    r"knowledge transfer partnerships?"]
FUNDER_RE = re.compile(r"\b(?:" + "|".join(FUNDER_TERMS) + r")\b", re.I)


def mask_funders(text: str) -> str:
    return FUNDER_RE.sub(" funderx ", text)


def norm_text(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def build_split(use_cache: bool = True):
    """Return ``(train, test)`` dataframes identical to notebook Sections 3-4."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tr_path, te_path = CACHE / "train.parquet", CACHE / "test.parquet"
    if use_cache and tr_path.exists() and te_path.exists():
        return pd.read_parquet(tr_path), pd.read_parquet(te_path)
    cols = ["project_id", "title", "abstract", "lead_funder", "grant_category", "award_gbp", "fund_start",
            "fund_end", "research_topics"]
    raw = pd.read_parquet(RAW_PARQUET, columns=cols)
    raw["n_words"] = raw.abstract.fillna("").str.split().str.len()
    raw["start_year"] = raw.fund_start.dt.year
    df = raw[raw.lead_funder.isin(COUNCILS)].copy()
    df = df[~df.abstract.fillna("").str.contains(PLACEHOLDER)]
    df = df[df.n_words >= 30]
    df["abs_key"] = df.abstract.map(norm_text)
    reuse = df.groupby("abs_key").title.nunique()
    df = df[~df.abs_key.isin(reuse[reuse >= 5].index)]
    df["dup_key"] = (df.title.map(norm_text) + "||" + df.abs_key).map(lambda s: hashlib.md5(s.encode()).hexdigest())
    conflict = df.groupby("dup_key").lead_funder.nunique()
    df = df[~df.dup_key.isin(conflict[conflict > 1].index)]
    df = df.sort_values("award_gbp", ascending=False, kind="stable").drop_duplicates("dup_key")
    df["text"] = (df.title.fillna("") + ". " + df.abstract.fillna("")).map(mask_funders)
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    df = df.reset_index(drop=True)
    title_key = "T:" + df.title.map(norm_text)
    abs_key = "A:" + df.abs_key.str.split().str[:25].str.join(" ")
    for i, (t, a) in enumerate(zip(title_key, abs_key)):
        union(f"R{i}", t)
        union(f"R{i}", a)
    df["group"] = [find(f"R{i}") for i in range(len(df))]
    le = LabelEncoder().fit(COUNCILS)
    y = le.transform(df.lead_funder)
    train_idx, test_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, groups=df.group))
    keep = ["project_id", "title", "abstract", "text", "lead_funder", "grant_category", "award_gbp", "fund_start",
            "fund_end", "start_year", "research_topics", "group"]
    tr = df.iloc[train_idx][keep].reset_index(drop=True)
    te = df.iloc[test_idx][keep].reset_index(drop=True)
    tr.to_parquet(tr_path, index=False)
    te.to_parquet(te_path, index=False)
    return tr, te


def save_result(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    return path


def tfidf_lr():
    """The deployed configuration (notebook Section 5/6): TF-IDF + class-weighted LR, C=32."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True, stop_words="english",
                          max_features=200_000, dtype=np.float32)
    clf = LogisticRegression(C=32.0, class_weight="balanced", max_iter=2000)
    return vec, clf
