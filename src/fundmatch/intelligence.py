"""Conformal sets, novelty, and the 2D constellation.

The shipped table has titles, organisations and awards. It does not contain
abstracts or the original test-split membership. Calibration therefore uses
real project titles (plus department and scheme, which are in the table), not
invented abstracts and not the unpublished CS5079 out-of-fold score file.

The quantiles follow the split-conformal recipe. They are not a re-measurement
of coverage on the original 22,648-row test set, because that row list was not
exported. The student-facing text says this in plain language.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fundmatch.text import mask_funders

ART = Path(__file__).resolve().parents[2] / "artifacts"
CALIBRATION_NOTE = (
    "Split-conformal thresholds were fit on real Gateway to Research titles "
    "(with the recorded department and scheme), 250 projects per council, seed 42. "
    "Abstracts are not in this snapshot, and the original test-split scores were not exported, "
    "so the 90% and 95% levels are quantiles of that title sample. They are not a new coverage "
    "certificate for the 22,648-row CS5079 test set. The published test calibration "
    "(ECE 0.0336 to 0.0180, and accuracy by confidence) is unchanged and separate."
)
HONESTY = (
    "Only add a phrase if it is true of the work. Stay scientifically honest: a higher "
    "council probability from a rewritten sentence is not a reason to overstate the science."
)


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    """Upper quantile with the usual (n+1) split-conformal correction."""
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    n = ordered.size
    rank = int(np.ceil((n + 1) * coverage)) 
    rank = min(max(rank, 1), n)
    return float(ordered[rank - 1])


def prediction_set(classes: list[str], probabilities: np.ndarray, quantile: float) -> list[str]:
    """Smallest top-ranked set whose probability mass reaches the quantile."""
    order = np.argsort(-probabilities)
    chosen: list[str] = []
    total = 0.0
    for index in order:
        chosen.append(classes[int(index)])
        total += float(probabilities[int(index)])
        if total + 1e-12 >= quantile:
            break
    return chosen


def empirical_sets(classes: list[str], probabilities: np.ndarray) -> tuple[list[str], list[str]]:
    """Sets justified by the published CS5079 test profile, not by the title quantile.

    The 0.80–1.00 confidence bin was correct on 94.3% of the held-out test, so a
    singleton clears 90% and does not clear 95%. Top-3 accuracy was 0.9916, so the
    top three councils clear both 90% and 95% when the top probability is below 0.80.
    """
    order = [classes[int(i)] for i in np.argsort(-probabilities)]
    top = float(probabilities.max())
    set_90 = order[:1] if top >= 0.80 else order[:3]
    set_95 = order[:3]
    return set_90, set_95


def explain_set(councils: list[str], level: str) -> str:
    if len(councils) == 1:
        return (
            f"At the {level} title-calibrated level the plausible set is only {councils[0]}. "
            "The model puts almost all of its probability there. Still read the runner-up if the idea is joint."
        )
    names = ", ".join(councils)
    return (
        f"At the {level} title-calibrated level the plausible set is {names}. "
        "More than one council is still inside the set, which is what an interdisciplinary or "
        "industry-facing idea often looks like. The set is a shortlist, not a decision."
    )


def novelty_band(percentile: float) -> str:
    """Percentile is how many calibration titles are less close to the corpus than this query."""
    if percentile >= 40:
        return "Core UKRI Territory"
    if percentile >= 10:
        return "Emerging Interdisciplinary Frontier"
    return "High-Novelty / Potential Out-of-Scope"


def novelty_note(band: str, max_sim: float, percentile: float) -> str:
    if band == "Core UKRI Territory":
        return (
            f"The closest funded abstract in the 64-d map has cosine {max_sim:.2f} "
            f"(about the {percentile:.0f}th percentile of real UKRI titles). "
            "This sits among ordinary funded topics."
        )
    if band == "Emerging Interdisciplinary Frontier":
        return (
            f"Closest cosine {max_sim:.2f}, around the {percentile:.0f}th percentile of real titles. "
            "The topic is less typical than most funded titles. Check the runner-up council and the neighbours."
        )
    return (
        f"Closest cosine {max_sim:.2f}, around the {percentile:.0f}th percentile of real titles. "
        "This is far from the historical UKRI title cloud. It may be new, or it may be outside what these eight councils fund. "
        "Do not read a low similarity as a judgement of quality."
    )


def project_query(q_svd: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (q_svd.astype(np.float64) - mean) @ components


def build_assets(engine, art: Path = ART) -> dict:
    """Fit conformal quantiles, a novelty reference, and the 2D map. Writes three artifact files."""
    frame = engine.frame
    rng = np.random.default_rng(42)
    classes = engine.classes
    per_council = 250
    chosen = []
    for council in classes:
        pool = np.flatnonzero(frame["lead_funder"].astype(str).to_numpy() == council)
        take = min(per_council, pool.size)
        chosen.append(rng.choice(pool, size=take, replace=False))
    index = np.concatenate(chosen)
    texts = []
    labels = []
    for row_i in index:
        row = frame.iloc[int(row_i)]
        department = "" if pd.isna(row["department"]) else str(row["department"])
        texts.append(mask_funders(f"{row['title']}. {department}. {row['grant_category']}."))
        labels.append(classes.index(str(row["lead_funder"])))
    matrix = engine.vectorize(texts)
    logits = engine.logits(matrix, scaled=True)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    scores = []
    for row_i, label in enumerate(labels):
        order = np.argsort(-probs[row_i])
        total = 0.0
        for col in order:
            total += float(probs[row_i, int(col)])
            if int(col) == label:
                scores.append(total)
                break
    scores_arr = np.asarray(scores, dtype=np.float64)
    q90 = finite_sample_quantile(scores_arr, 0.90)
    q95 = finite_sample_quantile(scores_arr, 0.95)
    conformal = {
        "method": "adaptive prediction set, split-conformal quantile of probability mass until the recorded council",
        "calibration_n": int(scores_arr.size),
        "per_council": per_council,
        "seed": 42,
        "text": "title + department + grant category (abstracts are not in the snapshot)",
        "quantile_90": q90,
        "quantile_95": q95,
        "score_median": float(np.median(scores_arr)),
        "published_test_ece_raw": 0.0336,
        "published_test_ece_scaled": 0.0180,
        "published_accuracy_by_confidence": engine.model_info.get("accuracy_by_confidence"),
        "note": CALIBRATION_NOTE,
    }
    (art / "conformal_aps.json").write_text(json.dumps(conformal, indent=2), encoding="utf-8")

    # Novelty: max abstract-space cosine of the same real title texts, in chunks.
    raw = np.asarray(matrix @ engine.c_rec.T, dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    queries = raw / norms
    max_sims = np.empty(queries.shape[0], dtype=np.float32)
    for start in range(0, queries.shape[0], 64):
        block = queries[start:start + 64]
        sims = engine.z_unit @ block.T
        max_sims[start:start + 64] = sims.max(axis=0)
    np.savez_compressed(art / "novelty_reference.npz", max_sim=max_sims)

    # 2D map from a stratified slice of the abstract SVD rows.
    sample = []
    for council in classes:
        pool = np.flatnonzero(frame["lead_funder"].astype(str).to_numpy() == council)
        sample.append(rng.choice(pool, size=min(400, pool.size), replace=False))
    sample_idx = np.concatenate(sample)
    cloud = engine.z_unit[sample_idx].astype(np.float64)
    mean = cloud.mean(axis=0)
    centred = cloud - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:2].T
    xy = centred @ components
    councils = frame["lead_funder"].astype(str).to_numpy()[sample_idx]
    landmark_idx = []
    for council in classes:
        local = np.flatnonzero(councils == council)
        landmark_idx.append(rng.choice(local, size=min(24, local.size), replace=False))
    keep = np.concatenate(landmark_idx)
    np.savez_compressed(
        art / "constellation_2d.npz",
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        landmark_xy=xy[keep].astype(np.float32),
        landmark_row=sample_idx[keep].astype(np.int32),
    )
    return {"quantile_90": q90, "quantile_95": q95, "novelty_n": int(max_sims.size), "landmarks": int(keep.size)}
