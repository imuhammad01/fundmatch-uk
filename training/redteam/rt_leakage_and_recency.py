"""Red-team checks RT-02 (near-duplicate leakage), RT-04 (time mix) and FR-8 (recency filter).

1. Refit the deployed configuration (TF-IDF + class-weighted LR, C=32) on the
   notebook's training split and score the notebook's test split. This must
   reproduce the published macro-F1 (0.8615) before anything else is trusted.
2. For every test project, find the most similar training project (TF-IDF
   cosine). Report how many test projects have a near twin in training and the
   macro-F1 once those are removed.
3. FR-8: award-band coverage on the test set with and without the recency rule
   (drop projects starting in or after the 2026 snapshot year) and with CPIH
   real-terms pounds, using the deployed 30-neighbour P25-P75 band.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from rt_common import COUNCILS, build_split, save_result, tfidf_lr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fundmatch.cpih import to_real_gbp  # noqa: E402

SNAPSHOT_YEAR = 2026


def main() -> None:
    tr, te = build_split()
    y_tr = np.array([COUNCILS.index(c) for c in tr.lead_funder])
    y_te = np.array([COUNCILS.index(c) for c in te.lead_funder])
    vec, clf = tfidf_lr()
    t0 = time.time()
    X_tr = vec.fit_transform(tr.text)
    X_te = vec.transform(te.text)
    clf.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    proba = clf.predict_proba(X_te)
    pred = proba.argmax(1)
    out = {"n_train": len(tr), "n_test": len(te), "fit_seconds": round(fit_s, 1),
           "reproduced": {"macro_f1": f1_score(y_te, pred, average="macro"), "accuracy": accuracy_score(y_te, pred),
                          "top3": top_k_accuracy_score(y_te, proba, k=3, labels=range(8))}}
    print("reproduced", out["reproduced"], flush=True)

    # --- near-duplicate leakage -------------------------------------------------------
    best = np.zeros(len(te), dtype=np.float32)
    best_idx = np.zeros(len(te), dtype=np.int64)
    Xt = X_tr.T.tocsr()
    for start in range(0, len(te), 1000):
        S = (X_te[start:start + 1000] @ Xt).toarray()
        best[start:start + 1000] = S.max(1)
        best_idx[start:start + 1000] = S.argmax(1)
    leak = {}
    for thr in (0.95, 0.9, 0.8, 0.7):
        twin = best >= thr
        keep = ~twin
        same_label = float((y_tr[best_idx[twin]] == y_te[twin]).mean()) if twin.any() else None
        leak[str(thr)] = {"test_with_train_twin": int(twin.sum()), "share": float(twin.mean()),
                          "twin_has_same_council": same_label,
                          "macro_f1_without_twins": f1_score(y_te[keep], pred[keep], average="macro"),
                          "accuracy_without_twins": accuracy_score(y_te[keep], pred[keep])}
    out["near_duplicates"] = leak
    out["max_cos_quantiles"] = {q: float(np.quantile(best, q)) for q in (0.5, 0.9, 0.99)}
    print("leak", leak, flush=True)

    # --- time mix inside the random split ------------------------------------------------
    yr = te.start_year.to_numpy()
    out["time_mix"] = {
        "test_share_start_2021_plus": float(np.mean(yr >= 2021)),
        "macro_f1_test_start_lt_2021": f1_score(y_te[yr < 2021], pred[yr < 2021], average="macro"),
        "macro_f1_test_start_2021_plus": f1_score(y_te[yr >= 2021], pred[yr >= 2021], average="macro"),
        "macro_f1_test_start_2026_plus": f1_score(y_te[yr >= SNAPSHOT_YEAR], pred[yr >= SNAPSHOT_YEAR], average="macro"),
        "n_test_start_2026_plus": int(np.sum(yr >= SNAPSHOT_YEAR)),
    }

    # --- FR-8: recency rule and comparable pounds for the award band -------------------
    a_tr = tr.award_gbp.to_numpy(float)
    a_te = te.award_gbp.to_numpy(float)
    y0_tr = tr.start_year.to_numpy()
    priced_tr = a_tr > 0
    rng = np.random.RandomState(42)
    cand = np.where(a_te > 0)[0]
    qs = rng.choice(cand, min(5000, len(cand)), replace=False)
    S = None
    neighbours = []
    for start in range(0, len(qs), 500):
        block = (X_te[qs[start:start + 500]] @ Xt).toarray()
        block[:, ~priced_tr] = -1
        neighbours.append(np.argsort(-block, axis=1)[:, :200])
    neighbours = np.vstack(neighbours)

    def coverage(recency: bool, real: bool, test_mask=None):
        hits, widths, used = [], [], 0
        for qi, row in zip(qs, neighbours):
            if test_mask is not None and not test_mask[qi]:
                continue
            idx = [j for j in row if (not recency or y0_tr[j] < SNAPSHOT_YEAR)][:30]
            vals = np.array([to_real_gbp(a_tr[j], int(y0_tr[j])) if real else a_tr[j] for j in idx])
            truth = to_real_gbp(a_te[qi], int(yr[qi])) if real else a_te[qi]
            lo, hi = np.percentile(vals, [25, 75])
            hits.append(lo <= truth <= hi)
            widths.append(hi / max(lo, 1))
            used += 1
        return {"n_queries": used, "coverage_p25_p75": float(np.mean(hits)), "median_width_hi_over_lo": float(np.median(widths))}

    not_recent = yr < SNAPSHOT_YEAR
    out["fr8_award_band"] = {
        "nominal, all neighbours (as deployed)": coverage(False, False),
        "nominal, recency rule (neighbours and queries start < 2026)": coverage(True, False, not_recent),
        "CPIH real 2026 pounds, all neighbours": coverage(False, True),
        "CPIH real 2026 pounds, recency rule": coverage(True, True, not_recent),
        "share_of_priced_train_neighbour_pool_removed_by_rule": float(np.mean(y0_tr[priced_tr] >= SNAPSHOT_YEAR)),
    }
    print("fr8", out["fr8_award_band"], flush=True)
    save_result("leakage_time_recency", out)


if __name__ == "__main__":
    main()
