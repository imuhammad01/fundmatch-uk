"""V5.2 Gap finder and 3D idea map.

Both tools work on the idea's *neighbourhood*: the funded GtR projects nearest to
the idea in the deployed 64-number SVD space (hybrid SVD + title ranking, the same
search the matcher uses). Nothing new is trained and nothing is invented.

Gap finder
    The neighbourhood (400 projects by default) is split into sub-topics with a
    small k-means on the existing unit SVD vectors. Each sub-topic is described by
    its project count, total and median priced award, and its trend, which is the
    change in the share of its projects that started in the latest four complete
    years against the four years before that, compared with the same change for the
    whole neighbourhood. A sub-topic is called *under-explored* when it is at least 15% below
    average on both project count and total funding **and** is among the thinnest
    on count, total and median award. It is called *crowded* when it has at least 20% more
    projects than the average sub-topic. These are descriptive labels about past awards. They are not
    advice that a gap is an opportunity.

3D idea map
    The nearest 300 projects are reduced to three axes with PCA fitted on those
    neighbours only, and the idea is projected onto the same axes.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .insights import _STOP
from .text import STOP_WORDS, check_input, mask_funders, proposal_text

NEIGHBOURS = 400
MAP_POINTS = 300
MAX_TOPICS = 10
RELATED_RATIO = 0.75     # a neighbour is kept only if its closeness to the idea is at least this share of the neighbourhood median
MIN_KEPT = 60
MIN_TOPIC_PROJECTS = 8
GROWTH_PP = 8.0          # percentage-point change in recent share that counts as a trend
MIN_PRICED = 3
UNDER_INDEX = 0.85          # at least 15% below the average sub-topic on both count and total award
CROWDED_COUNT_INDEX = 1.2  # at least 20% more projects than the average sub-topic
MIN_DATED = 8            # dated projects needed in the two windows before a trend is claimed

CAVEATS = (
    "“Little funding” is not the same as “opportunity”. A thin sub-topic can reflect low national priority, "
    "work filed under other words, or gaps in the data, as well as a true opening.",
    "GtR shows awards that were made in the past, not calls that are open now. Check the council's Funding "
    "Finder before you plan anything.",
    "Unpriced records such as studentships appear as £0. They count as projects but add nothing to totals, and "
    "medians use priced projects only.",
    "Sub-topics come from a small clustering of the neighbouring projects, so another run on a different idea can "
    "draw the borders differently. Amounts are nominal pounds.",
)

_GENERIC = {"work", "support", "investigating", "exploring", "enabling", "annually", "via", "effect", "effects", "impact", "using", "use", "uses", "case", "evidence", "case", "study", "studies", "towards", "improve", "improving", "multi", "scale", "large", "small", "two", "one"}
_FUNDER_WORDS = {"epsrc", "ukri", "bbsrc", "mrc", "nerc", "stfc", "ahrc", "esrc", "innovate", "uk", "ukri's"}


@dataclass
class Hood:
    """The idea's nearest funded projects. Small: indices and a 64-number query vector."""
    idx: np.ndarray
    query: np.ndarray
    ok: bool = True
    errors: list[str] | None = None
    n_words: int = 0


def neighbourhood(engine, title: str, abstract: str, n: int = NEIGHBOURS) -> Hood:
    text = proposal_text(title, abstract)
    masked = mask_funders(text)
    recognised = int(engine.vectorize([masked]).nnz) if text else 0
    errors, _warnings = check_input(title, abstract, recognised if text else None)
    if errors:
        return Hood(idx=np.array([], dtype=int), query=np.zeros(1), ok=False, errors=list(errors), n_words=len(text.split()))
    rrf, _svd, _lex, q = engine._search(masked)
    idx = engine._top(rrf, n)
    return Hood(idx=idx, query=np.asarray(q, dtype=np.float64), ok=True, errors=[], n_words=len(text.split()))


# ---------------------------------------------------------------- pure logic

def trend_label(recent: int, earlier: int, nb_recent: int, nb_earlier: int) -> dict:
    """Relative trend of a sub-topic against its neighbourhood.

    ``share`` is recent / (recent + earlier) for the sub-topic. The label compares it
    with the same share for the whole neighbourhood, so a general rise or fall in
    the number of GtR projects does not count as a sub-topic trend.
    """
    dated = recent + earlier
    nb_dated = nb_recent + nb_earlier
    if dated < MIN_DATED or nb_dated == 0:
        return {"label": "too few dated projects", "delta_pp": None, "share": None if dated == 0 else recent / dated}
    share = recent / dated
    delta = round((share - nb_recent / nb_dated) * 100.0, 1)
    label = "growing" if delta >= GROWTH_PP else "shrinking" if delta <= -GROWTH_PP else "flat"
    return {"label": label, "delta_pp": float(delta), "share": round(float(share), 4)}


def _pct_rank(values: list[float]) -> list[float]:
    """0 = smallest, 1 = largest; ties share the average rank."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return [0.5] * len(arr)
    order = arr.argsort(kind="mergesort")
    ranks = np.empty(len(arr))
    ranks[order] = np.arange(len(arr))
    for value in np.unique(arr):
        same = arr == value
        ranks[same] = ranks[same].mean()
    return list((ranks / (len(arr) - 1)).round(4))


def classify(topics: list[dict], max_gaps: int = 3, max_crowded: int = 2) -> list[dict]:
    """Add ``count_index``, ``funding_index``, ``thin``, ``crowd``, ``funding_known`` and ``status``.

    Indices are relative to the average sub-topic in this neighbourhood (1.0 = average).
    A median award is used only for sub-topics with at least ``MIN_PRICED`` priced
    projects; otherwise the funding picture is mostly unpriced records (£0) and the
    sub-topic is flagged ``funding_known = False``.
    """
    k = len(topics)
    if not k:
        return topics
    mean_count = sum(t["projects"] for t in topics) / k
    mean_total = sum(t["total_gbp"] for t in topics) / k
    counts = _pct_rank([t["projects"] for t in topics])
    totals = _pct_rank([t["total_gbp"] for t in topics])
    medians = _pct_rank([t["median_gbp"] or 0.0 for t in topics])
    for t, c, m, d in zip(topics, counts, totals, medians):
        t["funding_known"] = t.get("priced_projects", 0) >= MIN_PRICED
        t["count_index"] = round(t["projects"] / mean_count, 3) if mean_count else 0.0
        t["funding_index"] = round(t["total_gbp"] / mean_total, 3) if mean_total else 0.0
        t["thin"] = round((c + m + d) / 3.0 if t["funding_known"] else (c + m) / 2.0, 4)
        t["crowd"] = round((c + m) / 2.0, 4)
        t["status"] = "balanced"
    candidates = [t for t in topics if t["count_index"] <= UNDER_INDEX and t["funding_index"] <= UNDER_INDEX and t["projects"] >= MIN_TOPIC_PROJECTS]
    for t in sorted(candidates, key=lambda t: (t["thin"], t["projects"]))[:max_gaps]:
        t["status"] = "under-explored"
    crowded = [t for t in topics if t["count_index"] >= CROWDED_COUNT_INDEX]
    for t in sorted(crowded, key=lambda t: (-t["count_index"], -t["total_gbp"]))[:max_crowded]:
        t["status"] = "crowded"
    return topics


def _kmeans(x: np.ndarray, k: int, n_init: int = 5, iters: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic k-means (k-means++ start, best of ``n_init`` runs) in plain NumPy.

    Written out because ``sklearn.cluster`` adds about 11 MB of imports to a server that
    is held under 1 GB, and the data here is only a few hundred 64-number vectors.
    """
    rng = np.random.RandomState(seed)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    sq = (x ** 2).sum(axis=1)
    for _ in range(n_init):
        centres = [x[rng.randint(len(x))]]
        for _c in range(1, k):
            d2 = np.min([sq - 2 * x @ c + (c @ c) for c in centres], axis=0).clip(min=0)
            total = d2.sum()
            centres.append(x[rng.choice(len(x), p=d2 / total)] if total > 0 else x[rng.randint(len(x))])
        centres = np.array(centres)
        labels = np.zeros(len(x), dtype=int)
        for step in range(iters):
            dist = sq[:, None] - 2 * x @ centres.T + (centres ** 2).sum(axis=1)[None, :]
            new = dist.argmin(axis=1)
            if step and np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                members = x[labels == c]
                if len(members):
                    centres[c] = members.mean(axis=0)
                else:  # re-seed an empty cluster at the point farthest from its centre
                    far = int(dist[np.arange(len(x)), labels].argmax())
                    centres[c] = x[far]
                    labels[far] = c
        inertia = float(((x - centres[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy(), centres.copy())
    assert best is not None
    return best[1], best[2]


def _terms(titles: list[str], labels: np.ndarray, k: int, per_topic: int = 6) -> list[list[str]]:
    """Distinctive title words per sub-topic: mean TF-IDF in the sub-topic minus the
    neighbourhood mean. Fitted on ~400 short titles, so it is instant."""
    stop = sorted(set(STOP_WORDS) | _STOP | _GENERIC | _FUNDER_WORDS)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=stop,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b", dtype=np.float32)
    try:
        mat = vec.fit_transform(titles)
    except ValueError:
        return [[] for _ in range(k)]
    names = np.asarray(vec.get_feature_names_out())
    base = np.asarray(mat.mean(axis=0)).ravel()
    out = []
    for c in range(k):
        rows = np.flatnonzero(labels == c)
        if not len(rows):
            out.append([])
            continue
        lift = np.asarray(mat[rows].mean(axis=0)).ravel() - base
        picked: list[str] = []
        for j in np.argsort(-lift)[: per_topic * 6]:
            if lift[j] <= 0:
                break
            term = str(names[j])
            if any(term in p or p in term for p in picked):
                continue
            picked.append(term)
            if len(picked) == per_topic:
                break
        out.append(picked)
    return out


def _windows(engine) -> tuple[tuple[int, int], tuple[int, int]]:
    collected = str(engine.model_info.get("data_collected", ""))[:4]
    last = int(collected) - 1 if collected.isdigit() else 2025
    return (last - 3, last), (last - 7, last - 4)


def analyse(engine, title: str, abstract: str, n: int = NEIGHBOURS, hood: Hood | None = None) -> dict:
    hood = hood or neighbourhood(engine, title, abstract, n)
    if not hood.ok:
        return {"ok": False, "errors": hood.errors or [], "caveats": list(CAVEATS)}
    all_idx = hood.idx
    n_found = len(all_idx)
    # Drop neighbours that sit far from the idea in the SVD map, so no sub-topic is an unrelated
    # corner of the corpus. The cut is relative to this idea's own neighbourhood.
    sim_all = engine.z_unit[all_idx].astype(np.float64) @ hood.query
    cutoff = RELATED_RATIO * float(np.median(sim_all))
    keep = sim_all >= cutoff
    if int(keep.sum()) < MIN_KEPT:
        keep = np.ones(n_found, dtype=bool)
    idx = all_idx[keep]
    sim_to_idea = sim_all[keep]
    n_kept = len(idx)
    frame = engine.frame.iloc[idx]
    k = int(min(MAX_TOPICS, max(3, round(n_kept / 35))))
    vectors = engine.z_unit[idx].astype(np.float64)
    labels, centres = _kmeans(vectors, k)
    titles = [str(t) for t in frame["title"].fillna("")]
    years = frame["start_year"].to_numpy(dtype="float64", na_value=np.nan)
    awards = frame["award_gbp"].to_numpy(dtype=np.float64)
    funders = frame["lead_funder"].astype(str).to_numpy()
    (r0, r1), (e0, e1) = _windows(engine)
    in_recent = (years >= r0) & (years <= r1)
    in_earlier = (years >= e0) & (years <= e1)
    nb_recent, nb_earlier = int(in_recent.sum()), int(in_earlier.sum())
    terms = _terms([html_unescape(t) for t in titles], labels, k)
    topics = []
    for c in range(k):
        rows = np.flatnonzero(labels == c)
        if not len(rows):
            continue
        priced = awards[rows][awards[rows] > 0]
        centre = centres[c]
        centre = centre / (np.linalg.norm(centre) or 1.0)
        near = rows[np.argsort(-(vectors[rows] @ centre), kind="mergesort")]
        examples, seen = [], set()
        for r in near:
            name = html_unescape(titles[r]).strip().lower()
            if name in seen:
                continue
            seen.add(name)
            row = frame.iloc[int(r)]
            examples.append({
                "title": html_unescape(str(row["title"] or "")), "council": str(row["lead_funder"]),
                "year": None if not np.isfinite(years[r]) else int(years[r]), "award_gbp": float(row["award_gbp"]),
                "gtr_url": str(row["gtr_url"] or ""), "organisation": html_unescape(str(row["lead_org_name"] or "")),
            })
            if len(examples) == 3:
                break
        by_funder: dict[str, int] = {}
        for f in funders[rows]:
            by_funder[f] = by_funder.get(f, 0) + 1
        council, c_n = max(by_funder.items(), key=lambda kv: (kv[1], kv[0]))
        by_year: dict[int, int] = {}
        for y in years[rows][np.isfinite(years[rows])]:
            by_year[int(y)] = by_year.get(int(y), 0) + 1
        topics.append({
            "id": int(c), "terms": terms[c], "projects": int(len(rows)),
            "priced_projects": int(len(priced)), "total_gbp": float(awards[rows].sum()),
            "median_gbp": None if not len(priced) else float(np.median(priced)),
            "recent": int(in_recent[rows].sum()), "earlier": int(in_earlier[rows].sum()),
            "trend": trend_label(int(in_recent[rows].sum()), int(in_earlier[rows].sum()), nb_recent, nb_earlier),
            "by_year": dict(sorted(by_year.items())),
            "council": council, "council_share": round(c_n / len(rows), 3),
            "closeness_to_idea": round(float(sim_to_idea[rows].mean()), 4),
            "examples": examples, "rows": [int(i) for i in idx[rows]],
        })
    classify(topics)
    order = {"under-explored": 0, "crowded": 1, "balanced": 2}
    topics.sort(key=lambda t: (order[t["status"]], -t["crowd"] if t["status"] == "crowded" else t["thin"]))
    nb_priced = awards[awards > 0]
    return {
        "ok": True, "errors": [], "neighbours": int(n_kept), "fetched": int(n_found), "left_out": int(n_found - n_kept),
        "topics": topics, "k": len(topics), "relatedness_cutoff": round(cutoff, 4),
        "windows": {"recent": [r0, r1], "earlier": [e0, e1], "nb_recent": nb_recent, "nb_earlier": nb_earlier},
        "neighbourhood": {
            "total_gbp": float(awards.sum()), "median_gbp": None if not len(nb_priced) else float(np.median(nb_priced)),
            "priced_share": round(float(len(nb_priced) / n_kept), 3),
        },
        "method": (
            f"The {n_found} funded projects nearest your idea were fetched; {n_found - n_kept} that sit far from it in the SVD map were left out, "
            f"and the other {n_kept} were split into {len(topics)} sub-topics by k-means on their existing 64-number SVD vectors. "
            "Key terms are the title words that are most distinctive for each sub-topic. "
            f"Trend compares the share of a sub-topic's projects that started in {r0}–{r1} with {e0}–{e1}, relative to all {n_kept} projects."
        ),
        "caveats": list(CAVEATS),
    }


def html_unescape(text: str) -> str:
    return html.unescape(text or "")


# ---------------------------------------------------------------- 3D map

def idea_map_3d(engine, hood: Hood, n: int = MAP_POINTS, topic_of_row: dict[int, int] | None = None) -> dict:
    """PCA (fitted on the neighbours only) of the neighbours' SVD vectors, plus the idea."""
    if not hood.ok or not len(hood.idx):
        return {"ok": False}
    idx = hood.idx[:n]
    x = engine.z_unit[idx].astype(np.float64)
    mean = x.mean(axis=0)
    centred = x - mean
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    comps = vt[:3]
    coords = centred @ comps.T
    idea = (hood.query - mean) @ comps.T
    var = (s ** 2) / max(float((s ** 2).sum()), 1e-12)
    frame = engine.frame.iloc[idx]
    years = frame["start_year"].to_numpy(dtype="float64", na_value=np.nan)
    points = []
    for i, row_i in enumerate(idx):
        row = frame.iloc[i]
        points.append({
            "x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2]),
            "title": html_unescape(str(row["title"] or "")), "council": str(row["lead_funder"]),
            "year": None if not np.isfinite(years[i]) else int(years[i]), "award_gbp": float(row["award_gbp"]),
            "gtr_url": str(row["gtr_url"] or ""), "topic": None if topic_of_row is None else topic_of_row.get(int(row_i)),
        })
    return {
        "ok": True, "points": points, "idea": [float(v) for v in idea],
        "explained": [round(float(v), 4) for v in var[:3]], "n": len(points),
        "note": (
            "A 3D picture of a 64-number map. The three axes are the directions in which these neighbours differ most "
            f"(together {var[:3].sum():.0%} of their spread), so distances on screen distort the real ones: points that look "
            "close can be far apart in the full map, and the other way round. Axes have no unit."
        ),
    }
