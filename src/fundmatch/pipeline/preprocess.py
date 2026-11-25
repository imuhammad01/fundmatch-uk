"""Cleaning rules that took the API pull from 158,710 rows to 113,233 (FR-6, FR-7, FR-8).

The counts in ``FUNNEL`` are the verified coursework funnel from
``training/models/summary.json`` (23 Nov 2026). ``clean_frame_with_report``
applies the same rules to any dataframe with ``lead_funder``, ``title`` and
``abstract`` columns and returns the rows removed by every step, so the logic
can be tested without the raw API dump. The full notebook version is Section 3
of ``training/FundMatch_Task1.ipynb``.
"""

from __future__ import annotations

import re

import pandas as pd

COUNCILS = ["AHRC", "BBSRC", "EPSRC", "ESRC", "Innovate UK", "MRC", "NERC", "STFC"]
MIN_ABSTRACT_WORDS = 30
BOILERPLATE_MIN_TITLES = 5
# Same placeholder rule as notebook Section 3 (red-team RT-07: the earlier copy was narrower).
PLACEHOLDER = re.compile(
    r"(?:abstracts are not currently available|awaiting public project summary|no public description|"
    r"public description for this project has been requested|^\s*(?:tbc|tba|to be confirmed|n/?a|none|"
    r"no abstract available\.?)\s*$|doctoral training partnerships: a range of postgraduate)",
    re.IGNORECASE,
)


def norm_text(value) -> str:
    """Lower-case, keep letters and digits, collapse spaces (duplicate keys only)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(value).lower())).strip()


# FR-7: spelling variants seen in GtR and other UKRI sources -> one label.
# Programme budgets (ISCF, GCRF, COVID, Horizon Europe Guarantee, Newton Fund,
# UKRI FLF, ...) are not councils an applicant can choose, they overlap the
# councils, and each is under 2.5% of the pull, so they map to "Other" and are
# dropped from the classifier (SCOPE_DECISIONS.md, D8).
FUNDER_ALIASES = {
    "ahrc": "AHRC", "arts and humanities research council": "AHRC",
    "bbsrc": "BBSRC", "biotechnology and biological sciences research council": "BBSRC",
    "epsrc": "EPSRC", "engineering and physical sciences research council": "EPSRC",
    "esrc": "ESRC", "economic and social research council": "ESRC",
    "innovate uk": "Innovate UK", "innovateuk": "Innovate UK", "iuk": "Innovate UK",
    "technology strategy board": "Innovate UK", "tsb": "Innovate UK",
    "mrc": "MRC", "medical research council": "MRC",
    "nerc": "NERC", "natural environment research council": "NERC",
    "stfc": "STFC", "science and technology facilities council": "STFC",
}
OTHER = "Other"

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


def normalise_funder(label) -> str:
    """Map a raw lead-funder string to one of ``COUNCILS`` or ``"Other"`` (FR-7)."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return OTHER
    key = " ".join(str(label).replace("-", " ").split()).lower()
    return FUNDER_ALIASES.get(key, OTHER)


def clean_frame_with_report(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the notebook Section 3 filters and return ``(clean, report)`` (FR-6).

    ``report`` has one row per step with ``rows`` left and rows ``removed``.
    On the 23 Nov 2026 pull this reproduces ``FUNNEL`` exactly (checked by
    ``training/redteam/rt_check_funnel.py``). Does not call the network.
    """
    steps: list[tuple[str, int]] = [("raw projects", len(frame))]
    out = frame.copy()
    out["lead_funder"] = out["lead_funder"].map(normalise_funder)
    out = out.loc[out["lead_funder"].isin(COUNCILS)].copy()
    steps.append(("keep 8 councils", len(out)))
    abstract = out["abstract"].fillna("").astype(str)
    out = out.loc[~abstract.str.contains(PLACEHOLDER)].copy()
    steps.append(("drop placeholder abstracts", len(out)))
    n_words = out["abstract"].fillna("").astype(str).str.split().str.len()
    out = out.loc[n_words >= MIN_ABSTRACT_WORDS].copy()
    steps.append((f"drop empty or abstracts under {MIN_ABSTRACT_WORDS} words", len(out)))
    out["_abs_key"] = out["abstract"].map(norm_text)
    reuse = out.groupby("_abs_key")["title"].nunique()
    out = out.loc[~out["_abs_key"].isin(reuse[reuse >= BOILERPLATE_MIN_TITLES].index)].copy()
    steps.append((f"drop boilerplate reused by {BOILERPLATE_MIN_TITLES} or more titles", len(out)))
    out["_dup_key"] = out["title"].map(norm_text) + "||" + out["_abs_key"]
    n_funders = out.groupby("_dup_key")["lead_funder"].nunique()
    out = out.loc[~out["_dup_key"].isin(n_funders[n_funders > 1].index)].copy()
    steps.append(("drop duplicates with conflicting funders", len(out)))
    if "award_gbp" in out.columns:  # keep the joint-grant copy with the largest award, as the notebook does
        out = out.sort_values("award_gbp", ascending=False, kind="stable")
    out = out.drop_duplicates("_dup_key").drop(columns=["_abs_key", "_dup_key"])
    steps.append(("drop exact duplicates", len(out)))
    rows = []
    previous = steps[0][1]
    for name, left in steps:
        rows.append({"step": name, "rows": left, "removed": previous - left})
        previous = left
    return out.reset_index(drop=True), pd.DataFrame(rows)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the coursework filters. Does not call the network."""
    return clean_frame_with_report(frame)[0]


def date_filter_report(frame: pd.DataFrame, *, first_year: int = 2006, snapshot_year: int = 2026,
                       year_col: str = "start_year") -> tuple[pd.DataFrame, pd.DataFrame]:
    """FR-8 date rule for award statistics, with counts.

    Keeps projects that started between ``first_year`` and ``snapshot_year - 1``.
    Projects starting in the snapshot year or later have not run long enough to
    show a settled award or outcome, and pre-``first_year`` records are sparse.
    The classifier is trained on every year (a time-based test checks drift:
    train before 2021, test 2021+); this filter is for comparing amounts.
    """
    years = pd.to_numeric(frame[year_col], errors="coerce")
    too_old = years < first_year
    too_recent = years >= snapshot_year
    missing = years.isna()
    keep = ~(too_old | too_recent | missing)
    report = pd.DataFrame([
        {"rule": "rows in", "rows": int(len(frame))},
        {"rule": "missing start year", "rows": int(missing.sum())},
        {"rule": f"started before {first_year}", "rows": int(too_old.sum())},
        {"rule": f"started {snapshot_year} or later (too recent)", "rows": int(too_recent.sum())},
        {"rule": "rows kept", "rows": int(keep.sum())},
    ])
    return frame.loc[keep].reset_index(drop=True), report
