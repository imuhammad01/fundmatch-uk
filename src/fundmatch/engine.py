"""In-memory FundMatch engine.

Loads the portable TF-IDF logistic regression exported from the CS5079 notebook
and the int8 TruncatedSVD index of all 113,233 abstracts. User text is not written
anywhere. Project rows never include a person's name.
"""

from __future__ import annotations

import csv
import ctypes
import gc
import html
import io
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from fundmatch.privacy import redact_org_name, redact_person_names
from fundmatch.cpih import CAVEAT, TARGET_2026, TARGET_LABEL, SOURCE_URL, annual_index, to_real_gbp
from fundmatch.intelligence import (
    CALIBRATION_NOTE,
    HONESTY,
    build_assets,
    empirical_sets,
    explain_set,
    novelty_band,
    novelty_note,
    project_query,
)
from fundmatch.text import STOP_WORDS

RECENT_START_YEAR = 2026  # FR-8: award bands use projects that started before this year
from fundmatch.text import (
    TOKEN_RE,
    check_input,
    english_share,
    mask_funders,
    proposal_text,
)

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
DATA = ROOT / "data"
TEMPERATURE = 1.1314305006097425
RRF_K = 60
AWARD_NEIGHBOURS = 30
TREND_NEIGHBOURS = 200
HIGH = 0.75
MODERATE = 0.60

COUNCILS = ["AHRC", "BBSRC", "EPSRC", "ESRC", "Innovate UK", "MRC", "NERC", "STFC"]
COUNCIL_INFO = {
    "AHRC": "Arts and Humanities Research Council",
    "BBSRC": "Biotechnology and Biological Sciences Research Council",
    "EPSRC": "Engineering and Physical Sciences Research Council",
    "ESRC": "Economic and Social Research Council",
    "Innovate UK": "Innovate UK (business-led innovation)",
    "MRC": "Medical Research Council",
    "NERC": "Natural Environment Research Council",
    "STFC": "Science and Technology Facilities Council",
}
# Okabe–Ito, plus a wine tone so the eighth council stays distinct (NFR-8).
COUNCIL_COLOURS = {
    "AHRC": "#E69F00",
    "BBSRC": "#009E73",
    "EPSRC": "#0072B2",
    "ESRC": "#CC79A7",
    "Innovate UK": "#882255",
    "MRC": "#D55E00",
    "NERC": "#56B4E9",
    "STFC": "#F0E442",
}
FUNDING_FINDER_IDS = {
    "AHRC": 814, "BBSRC": 816, "ESRC": 818, "EPSRC": 820,
    "Innovate UK": 822, "MRC": 824, "NERC": 826, "STFC": 830,
}
FINDER = "https://www.ukri.org/opportunity/"
IFS = "https://apply-for-innovation-funding.service.gov.uk/competition/search"
OGL = "Contains public sector information licensed under the Open Government Licence v3.0."
DISCLAIMER = (
    "Student project for CS5079 Applied AI, University of Aberdeen (Md Shamim Mia, 52536283). "
    "Not official UKRI funding advice. UKRI does not endorse this tool."
)


def funding_finder_url(council: str) -> str:
    cid = FUNDING_FINDER_IDS.get(council)
    if cid is None:
        return FINDER
    return f"{FINDER}?filter_council%5B%5D={cid}&filter_status%5B%5D=open&filter_submitted=true"


class CouncilScore(BaseModel):
    council: str
    probability: float
    rank: int
    colour: str
    funding_finder_url: str
    full_name: str


class TermContribution(BaseModel):
    feature: str
    contribution: float


class Highlight(BaseModel):
    start: int
    end: int
    feature: str
    contribution: float
    polarity: str


class Explanation(BaseModel):
    council: str
    positive: list[TermContribution]
    negative: list[TermContribution]
    highlights: list[Highlight]
    logit: float


class Contrast(BaseModel):
    council_a: str
    council_b: str
    toward_a: list[TermContribution]
    toward_b: list[TermContribution]


class SimilarProject(BaseModel):
    project_id: str
    grant_ref: str
    title: str
    lead_funder: str
    colour: str
    award_gbp: float
    award_gbp_real: float
    fund_start: str | None
    fund_end: str | None
    start_year: int | None
    grant_category: str
    lead_org_name: str
    department: str
    similarity: float
    svd_cosine: float
    lexical_score: float = 0.0
    rrf_score: float = 0.0
    shared_terms: list[str] = Field(default_factory=list)
    gtr_url: str


class AwardBand(BaseModel):
    p10: float | None
    p25: float | None
    median: float | None
    p75: float | None
    p90: float | None
    n_positive: int
    n_zero_excluded: int
    neighbours: int
    basis: str
    same_scheme: bool
    scheme: str | None
    caveat: str
    zero_note: str
    cpih_target: str
    cpih_index: float


class TrendPoint(BaseModel):
    year: int
    council: str
    projects: int
    total_gbp: float
    median_gbp: float | None


class Analysis(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    word_count: int = 0
    english_share: float = 0.0
    latency_ms: float = 0.0
    councils: list[CouncilScore] = Field(default_factory=list)
    top_council: str | None = None
    confidence: str | None = None
    confidence_note: str | None = None
    entropy: float | None = None
    entropy_normalized: float | None = None
    explanation: Explanation | None = None
    contrast: Contrast | None = None
    award: AwardBand | None = None
    trends: list[TrendPoint] = Field(default_factory=list)
    similar: list[SimilarProject] = Field(default_factory=list)
    temperature: float = TEMPERATURE
    quantized: bool = False
    precision: str = "fp64"
    conformal_set_90: list[str] = Field(default_factory=list)
    conformal_set_95: list[str] = Field(default_factory=list)
    conformal_note: str = ""
    conformal_explanation: str = ""
    novelty: dict = Field(default_factory=dict)
    constellation: list[dict] = Field(default_factory=list)
    host_institutions: list[dict] = Field(default_factory=list)
    scheme_fit: list[dict] = Field(default_factory=list)
    student_summary: str = ""


def _student_summary(council: str, tier: str, conformal: list[str], band: str, award: AwardBand | None) -> str:
    money = "a typical award band is not available from the neighbours" if award is None or award.median is None else (
        f"the middle of similar priced awards is about £{award.median:,.0f} ({award.basis.replace('_', ' ')})"
    )
    shortlist = ", ".join(conformal) if conformal else council
    return (
        f"For an MSc, PhD or early-career application, read this as a shortlist. "
        f"The closest council is {council} ({tier} confidence). "
        f"The 90% plausible set is {shortlist}. "
        f"The topic looks like {band}. {money}. "
        "None of this is a promise of funding, and it does not name or rank any person."
    )


def _drop_term(text: str, term: str) -> str:
    import re
    return re.sub(rf"\b{re.escape(term)}\b", " ", text, flags=re.IGNORECASE)


def _tier(probability: float) -> tuple[str, str]:
    if probability >= HIGH:
        return "high", (
            f"High confidence ({probability:.0%}). On the held-out test, predictions at or above 0.80 "
            "were correct 94.3% of the time. Read the runner-up if the idea is joint with industry."
        )
    if probability >= MODERATE:
        return "moderate", (
            f"Moderate confidence ({probability:.0%}). The 0.60–0.80 test band was correct 68.2% of the time. "
            "Compare the top two councils in the X-Ray before you rely on the leader."
        )
    return "low", (
        f"Low confidence ({probability:.0%}). Below 0.60 the held-out accuracy was about 52% or worse. "
        "The idea may be interdisciplinary. Treat the top three as a shortlist, not a decision."
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


class FundMatchEngine:
    """Thread-safe, read-only after construction."""

    def __init__(self, art: Path = ART, data: Path = DATA):
        self._lock = threading.Lock()
        started = time.perf_counter()
        # RT-21: build the lexical index first, while little else is resident, so the
        # vectoriser's temporary n-gram counts do not stack on top of the weights.
        # Arrow-backed strings keep the 113,233-row table compact. The OS is then
        # asked to return the vectorizer's temporary buffers so the resident set
        # stays inside a 1 GB Streamlit container with room to spare.
        frame = pd.read_parquet(data / "projects_meta.parquet", dtype_backend="pyarrow")
        banned = {"pi_name", "researcher", "person_name", "investigator", "orcid", "surname"}
        if banned.intersection({c.lower() for c in frame.columns}):
            raise RuntimeError("project table contains personal columns")
        self.frame = frame.reset_index(drop=True)
        del frame
        # NFR-9: a few public titles and sole-trader organisation names contain a
        # person's name. Redact them once here so no view can show them.
        for column, rule in (("title", redact_person_names), ("lead_org_name", redact_org_name)):
            original = self.frame[column].fillna("").astype(str)
            cleaned = original.map(rule)
            changed = cleaned != original
            if bool(changed.any()):
                self.frame.loc[changed, column] = cleaned[changed]
        lexical_text = (
            self.frame["title"].fillna("").astype(str)
            + " "
            + self.frame["department"].fillna("").astype(str)
            + " "
            + self.frame["grant_category"].fillna("").astype(str)
        )
        self.lexical = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=80_000,
            sublinear_tf=True, norm="l2", lowercase=True, dtype=np.float32,
        )
        self.x_lexical = self.lexical.fit_transform(lexical_text.tolist())
        del lexical_text
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        meta = json.loads((art / "tfidf_lr_meta.json").read_text(encoding="utf-8"))
        weights = np.load(art / "tfidf_lr_weights.npz")
        self.classes: list[str] = list(meta["classes"])
        self.vocab: list[str] = list(meta["vocab"])
        self.index = {term: i for i, term in enumerate(self.vocab)}
        self.idf = weights["idf"].astype(np.float64)
        self.coef = weights["coef"].astype(np.float64)
        self.intercept = weights["intercept"].astype(np.float64)
        self.temperature = TEMPERATURE
        self.token_re = TOKEN_RE
        index = np.load(art / "svd_index_int8.npz")
        # Dequantise in place: one float32 copy per matrix instead of three
        # temporaries (RT-21, load-time peak memory).
        self.z_unit = index["Z_int8"].astype(np.float32)
        self.z_unit *= (index["z_scales"].astype(np.float32) / np.float32(127.0))[:, None]
        self.c_rec = index["C_int8"].astype(np.float32)
        self.c_rec *= (index["c_scales"].astype(np.float32) / np.float32(127.0))[:, None]
        del index
        if len(self.frame) != self.z_unit.shape[0]:
            raise RuntimeError("SVD rows and projects_meta.parquet disagree")
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        info_text = (art / "model_info.json").read_text(encoding="utf-8")
        self.model_info = json.loads(info_text)
        summary_path = art / "summary.json"
        self.summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        weights_i8 = np.load(art / "tfidf_lr_weights_int8.npz")
        scales = weights_i8["scales"].astype(np.float64)
        self.coef_int8 = (weights_i8["coef_int8"].astype(np.float64) / 127.0) * scales[:, None]
        self.lex_names = np.asarray(self.lexical.get_feature_names_out())
        if not (art / "conformal_aps.json").exists() or not (art / "constellation_2d.npz").exists():
            build_assets(self, art)
        self.conformal = json.loads((art / "conformal_aps.json").read_text(encoding="utf-8"))
        novelty = np.load(art / "novelty_reference.npz")
        self.novelty_reference = np.sort(novelty["max_sim"].astype(np.float64))
        sky = np.load(art / "constellation_2d.npz")
        self.sky_mean = sky["mean"].astype(np.float64)
        self.sky_components = sky["components"].astype(np.float64)
        self.landmark_xy = sky["landmark_xy"].astype(np.float64)
        self.landmark_row = sky["landmark_row"].astype(np.int32)
        self._requests = 0
        self._latencies: deque[float] = deque(maxlen=400)
        self.cold_s = time.perf_counter() - started
        self._stats: dict | None = None

    def _tokens(self, masked: str) -> list[str]:
        """Tokenise like the training ``TfidfVectorizer``: lowercase, ``\\b\\w\\w+\\b``,
        then drop English stop words before n-grams are formed."""
        return [token for token in self.token_re.findall(masked.lower()) if token not in STOP_WORDS]

    def vectorize(self, masked_texts: list[str]):
        """Sparse L2-normalised sublinear TF-IDF. ``masked_texts`` are already masked."""
        import scipy.sparse as sp

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row, text in enumerate(masked_texts):
            tokens = self._tokens(text)
            counts: dict[int, int] = {}
            for term in tokens:
                idx = self.index.get(term)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
            if len(tokens) >= 2:
                for a, b in zip(tokens, tokens[1:]):
                    idx = self.index.get(f"{a} {b}")
                    if idx is not None:
                        counts[idx] = counts.get(idx, 0) + 1
            if not counts:
                continue
            weighted = {idx: (1.0 + np.log(count)) * float(self.idf[idx]) for idx, count in counts.items()}
            norm = float(np.sqrt(sum(value * value for value in weighted.values())))
            if norm == 0:
                continue
            for idx, value in weighted.items():
                rows.append(row)
                cols.append(idx)
                data.append(value / norm)
        return sp.csr_matrix((data, (rows, cols)), shape=(len(masked_texts), len(self.vocab)), dtype=np.float64)

    def logits(self, matrix, scaled: bool = True, quantized: bool = False) -> np.ndarray:
        coef = self.coef_int8 if quantized else self.coef
        raw = np.asarray(matrix @ coef.T, dtype=np.float64) + self.intercept
        return raw / self.temperature if scaled else raw

    def predict_proba_masked(self, masked: str, quantized: bool = False) -> np.ndarray:
        return _softmax(self.logits(self.vectorize([masked]), scaled=True, quantized=quantized)[0])

    def _contributions(self, masked: str, weight_row: np.ndarray) -> list[tuple[str, float, int]]:
        matrix = self.vectorize([masked]).tocsr()
        if matrix.nnz == 0:
            return []
        scores = matrix.data * weight_row[matrix.indices]
        out = []
        for index, score in zip(matrix.indices, scores):
            feature = self.vocab[int(index)]
            if "funderx" in feature:
                continue
            out.append((feature, float(score), int(index)))
        return out

    def explain_council(self, title: str, abstract: str, council: str, k: int = 8) -> Explanation:
        text = proposal_text(title, abstract)
        masked = mask_funders(text)
        class_index = self.classes.index(council)
        parts = self._contributions(masked, self.coef[class_index])
        parts.sort(key=lambda item: item[1], reverse=True)
        positive = [TermContribution(feature=f, contribution=round(s, 6)) for f, s, _ in parts if s > 0][:k]
        negative = [TermContribution(feature=f, contribution=round(s, 6)) for f, s, _ in sorted(parts, key=lambda item: item[1]) if s < 0][:k]
        wanted = {item.feature: item.contribution for item in positive + negative}
        logit = float(self.intercept[class_index] / self.temperature + sum(s for _, s, _ in parts) / self.temperature)
        # Contributions above are on the unscaled tf-idf × coefficient. The reported
        # logit is the temperature-scaled one, matching softmax(z / T).
        return Explanation(
            council=council,
            positive=positive,
            negative=negative,
            highlights=_highlights(text, wanted),
            logit=round(logit, 6),
        )

    def contrast(self, title: str, abstract: str, council_a: str, council_b: str, k: int = 6) -> Contrast:
        masked = mask_funders(proposal_text(title, abstract))
        ia, ib = self.classes.index(council_a), self.classes.index(council_b)
        delta = self.coef[ia] - self.coef[ib]
        parts = self._contributions(masked, delta)
        parts.sort(key=lambda item: item[1], reverse=True)
        toward_a = [TermContribution(feature=f, contribution=round(s, 6)) for f, s, _ in parts if s > 0][:k]
        toward_b = [TermContribution(feature=f, contribution=round(s, 6)) for f, s, _ in sorted(parts, key=lambda item: item[1]) if s < 0][:k]
        return Contrast(council_a=council_a, council_b=council_b, toward_a=toward_a, toward_b=toward_b)

    def _search(self, masked: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        matrix = self.vectorize([masked])
        # V5.1 (NFR-3): multiply only the query's non-zero terms by their rows of the
        # 64 x 200,000 projection. ``matrix @ self.c_rec.T`` made a float64 copy of the
        # whole projection (about 100 MB) on every first query, which set the server's
        # memory peak. The numbers are the same.
        csr = matrix.tocsr()
        if csr.nnz:
            raw = (self.c_rec[:, csr.indices].astype(np.float64) @ csr.data.astype(np.float64)).astype(np.float32)
        else:
            raw = np.zeros(self.c_rec.shape[0], dtype=np.float32)
        norm = float(np.linalg.norm(raw))
        q = raw / norm if norm else raw
        svd = self.z_unit @ q
        lexical_q = self.lexical.transform([masked])
        lex = np.asarray((self.x_lexical @ lexical_q.T).toarray(), dtype=np.float32).ravel()
        svd_rank = np.empty(len(svd), dtype=np.int32)
        lex_rank = np.empty(len(lex), dtype=np.int32)
        svd_rank[np.argsort(-svd, kind="mergesort")] = np.arange(1, len(svd) + 1)
        lex_rank[np.argsort(-lex, kind="mergesort")] = np.arange(1, len(lex) + 1)
        rrf = (1.0 / (RRF_K + svd_rank) + 1.0 / (RRF_K + lex_rank)).astype(np.float32)
        return rrf, svd, lex, q

    def _mask(self, councils: list[str] | None, year_min: int | None, year_max: int | None, scheme: str | None) -> np.ndarray:
        mask = np.ones(len(self.frame), dtype=bool)
        if councils:
            allowed = [c for c in councils if c in COUNCIL_INFO]
            if allowed:
                mask &= self.frame["lead_funder"].isin(allowed).to_numpy()
        years = pd.to_numeric(self.frame["start_year"], errors="coerce").to_numpy(dtype=np.float64)
        if year_min is not None:
            mask &= np.isfinite(years) & (years >= year_min)
        if year_max is not None:
            mask &= np.isfinite(years) & (years <= year_max)
        if scheme:
            mask &= (self.frame["grant_category"].fillna("") == scheme).to_numpy()
        return mask

    def _top(self, scores: np.ndarray, k: int) -> np.ndarray:
        valid = int(np.sum(np.isfinite(scores) & (scores > -1e20)))
        k = min(k, valid)
        if k <= 0:
            return np.array([], dtype=int)
        pick = np.argpartition(-scores, kth=k - 1)[:k]
        return pick[np.argsort(-scores[pick])]

    def analyze(
        self,
        title: str = "",
        abstract: str = "",
        *,
        councils: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        scheme: str | None = None,
        same_scheme: bool = False,
        cpih: bool = False,
        top_k: int = 10,
        compare: str | None = None,
        quantized: bool = False,
    ) -> Analysis:
        started = time.perf_counter()
        text = proposal_text(title, abstract)
        masked = mask_funders(text)
        recognised = int(self.vectorize([masked]).nnz) if text else 0
        errors, warnings = check_input(title, abstract, recognised if text else None)
        if errors:
            return Analysis(
                ok=False, errors=errors, warnings=warnings,
                word_count=len(text.split()), english_share=round(english_share(text), 4),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        probs = self.predict_proba_masked(masked, quantized=quantized)
        order = np.argsort(-probs)
        council_rows = []
        for rank, idx in enumerate(order, start=1):
            name = self.classes[int(idx)]
            council_rows.append(CouncilScore(
                council=name, probability=round(float(probs[idx]), 6), rank=rank,
                colour=COUNCIL_COLOURS[name], funding_finder_url=funding_finder_url(name),
                full_name=COUNCIL_INFO[name],
            ))
        top = council_rows[0]
        tier, note = _tier(top.probability)
        explanation = self.explain_council(title, abstract, top.council)
        other = compare if compare in self.classes and compare != top.council else council_rows[1].council
        contrast = self.contrast(title, abstract, top.council, other)
        rrf, svd, lex, q_svd = self._search(masked)
        # A grant category filters the neighbour pool when the caller asks for it,
        # which is how the same-scheme toggle recomputes the band and the trend.
        active_scheme = scheme if (scheme and (same_scheme or scheme)) else None
        if scheme and not same_scheme:
            active_scheme = scheme
        if same_scheme and not scheme:
            active_scheme = None
        pool_mask = self._mask(councils, year_min, year_max, active_scheme if (same_scheme or scheme) else None)
        scores = rrf.copy()
        scores[~pool_mask] = -np.inf
        top_k = max(1, min(50, int(top_k)))
        similar_idx = self._top(scores, top_k)
        trend_idx = self._top(scores, TREND_NEIGHBOURS)
        award_idx = self._top(scores, AWARD_NEIGHBOURS)
        # When same-scheme is off, the band uses the unfiltered-by-scheme neighbour set
        # even if a category was not requested. When it is on, award_idx is already filtered.
        award = self._award(award_idx, cpih=cpih, same_scheme=bool(same_scheme and scheme), scheme=scheme if same_scheme else None)
        trends = self._trends(trend_idx, cpih=cpih)
        similar = self._projects(similar_idx, rrf, svd, lex, masked)
        hosts = self._hosts(similar_idx)
        schemes = self._schemes(similar_idx)
        set90, set95 = empirical_sets(self.classes, probs)
        novelty = self._novelty(svd)
        sky = self._constellation(q_svd, self._top(scores, 30))
        ent = _entropy(probs)
        summary = _student_summary(top.council, tier, set90, novelty["band"], award)
        self._requests += 1
        self._latencies.append(round((time.perf_counter() - started) * 1000, 2))
        return Analysis(
            ok=True, errors=[], warnings=warnings,
            word_count=len(text.split()), english_share=round(english_share(text), 4),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            councils=council_rows, top_council=top.council, confidence=tier, confidence_note=note,
            entropy=round(ent, 4), entropy_normalized=round(ent / np.log(len(self.classes)), 4),
            explanation=explanation, contrast=contrast, award=award, trends=trends, similar=similar,
            quantized=quantized, precision="int8" if quantized else "fp64",
            conformal_set_90=set90, conformal_set_95=set95,
            conformal_note=(
                f"{CALIBRATION_NOTE} Title-sample quantiles were q90={self.conformal['quantile_90']:.3f} "
                f"and q95={self.conformal['quantile_95']:.3f} (n={self.conformal['calibration_n']}). "
                "Those quantiles are near 1 because a title is not an abstract, so the displayed sets "
                "instead follow the published test profile: singleton only when probability is at least "
                "0.80 (that bin was 94.3% accurate), otherwise the top three (top-3 accuracy 0.9916)."
            ),
            conformal_explanation=explain_set(set90, "90%"),
            novelty=novelty, constellation=sky, host_institutions=hosts, scheme_fit=schemes,
            student_summary=summary,
        )

    def search(self, query: str, **filters) -> list[SimilarProject]:
        masked = mask_funders(query or "")
        rrf, svd, lex, _q = self._search(masked)
        scores = rrf.copy()
        scores[~self._mask(filters.get("councils"), filters.get("year_min"), filters.get("year_max"), filters.get("scheme"))] = -np.inf
        idx = self._top(scores, max(1, min(50, int(filters.get("top_k") or 10))))
        return self._projects(idx, rrf, svd, lex, masked)

    def _award(self, indices: np.ndarray, *, cpih: bool, same_scheme: bool, scheme: str | None) -> AwardBand:
        if len(indices) == 0:
            return AwardBand(
                p10=None, p25=None, median=None, p75=None, p90=None, n_positive=0, n_zero_excluded=0,
                neighbours=0, basis="real_2026" if cpih else "nominal", same_scheme=same_scheme, scheme=scheme,
                caveat=CAVEAT, zero_note="No neighbours in this filter.", cpih_target=TARGET_LABEL, cpih_index=TARGET_2026,
            )
        subset = self.frame.iloc[indices]
        amounts = subset["award_gbp"].to_numpy(dtype=np.float64)
        years = pd.to_numeric(subset["start_year"], errors="coerce").to_numpy()
        zero = int(np.sum(~np.isfinite(amounts) | (amounts <= 0)))
        # FR-8 recency rule: projects that start in 2026 or later may not have their
        # full award recorded yet, so they are left out of the percentiles. On a
        # 5,000-query check this changes P25-P75 coverage by under 0.1 points
        # (training/redteam/results/leakage_time_recency.json).
        recent = int(np.sum(np.isfinite(amounts) & (amounts > 0) & np.isfinite(years) & (years >= RECENT_START_YEAR)))
        values = []
        for amount, year in zip(amounts, years):
            if not np.isfinite(amount) or amount <= 0:
                continue
            if np.isfinite(year) and year >= RECENT_START_YEAR:
                continue
            year_i = None if not np.isfinite(year) else int(year)
            values.append(to_real_gbp(float(amount), year_i) if cpih else float(amount))
        arr = np.array(values, dtype=np.float64)
        if arr.size == 0:
            pct = [None] * 5
        else:
            pct = [round(float(v), 2) for v in np.percentile(arr, [10, 25, 50, 75, 90])]
        return AwardBand(
            p10=pct[0], p25=pct[1], median=pct[2], p75=pct[3], p90=pct[4],
            n_positive=int(arr.size), n_zero_excluded=zero, neighbours=int(len(indices)),
            basis="real_2026" if cpih else "nominal", same_scheme=same_scheme, scheme=scheme,
            caveat=CAVEAT,
            zero_note=(
                f"{zero} of the {len(indices)} nearest projects have a £0 award "
                "(studentships and other unpriced records) and are excluded from the percentiles."
                + (f" {recent} priced project(s) starting in {RECENT_START_YEAR} or later are also left out (FR-8 recency rule)." if recent else "")
            ),
            cpih_target=TARGET_LABEL, cpih_index=TARGET_2026,
        )

    def _trends(self, indices: np.ndarray, *, cpih: bool) -> list[TrendPoint]:
        if len(indices) == 0:
            return []
        subset = self.frame.iloc[indices][["start_year", "lead_funder", "award_gbp"]].copy()
        subset["year"] = pd.to_numeric(subset["start_year"], errors="coerce")
        values = []
        for amount, year in zip(subset["award_gbp"].to_numpy(dtype=np.float64), subset["year"].to_numpy()):
            if not np.isfinite(amount) or amount < 0:
                values.append(0.0)
                continue
            year_i = None if not np.isfinite(year) else int(year)
            values.append(to_real_gbp(float(amount), year_i) if cpih else float(amount))
        subset["value"] = values
        subset = subset.dropna(subset=["year"])
        points = []
        for (year, council), part in subset.groupby(["year", "lead_funder"], sort=True):
            positive = part.loc[part["award_gbp"] > 0, "value"]
            points.append(TrendPoint(
                year=int(year), council=str(council), projects=int(len(part)),
                total_gbp=round(float(part["value"].sum()), 2),
                median_gbp=None if positive.empty else round(float(positive.median()), 2),
            ))
        return points

    def _projects(self, indices: np.ndarray, rrf: np.ndarray, svd: np.ndarray, lex: np.ndarray, masked: str) -> list[SimilarProject]:
        perfect = 2.0 / (RRF_K + 1)
        query_terms = self._query_lexical_terms(masked)
        rows = []
        for idx in indices:
            row = self.frame.iloc[int(idx)]
            amount = float(row["award_gbp"])
            year = int(row["start_year"]) if pd.notna(row["start_year"]) else None
            start = _date(row["fund_start"])
            end = _date(row["fund_end"])
            shared = self._shared_terms(int(idx), query_terms)
            rows.append(SimilarProject(
                project_id=str(row["project_id"]),
                grant_ref=str(row["grant_ref"] or ""),
                title=html.unescape(str(row["title"] or "")),
                lead_funder=str(row["lead_funder"]),
                colour=COUNCIL_COLOURS.get(row["lead_funder"], "#888888"),
                award_gbp=amount,
                award_gbp_real=to_real_gbp(amount, year) or 0.0,
                fund_start=start, fund_end=end, start_year=year,
                grant_category=html.unescape(str(row["grant_category"] or "")),
                lead_org_name=html.unescape(str(row["lead_org_name"] or "")),
                department="" if pd.isna(row["department"]) else html.unescape(str(row["department"])),
                similarity=round(float(rrf[idx] / perfect), 4),
                svd_cosine=round(float(max(svd[idx], 0.0)), 4),
                lexical_score=round(float(max(lex[idx], 0.0)), 4),
                rrf_score=round(float(rrf[idx]), 6),
                shared_terms=shared,
                gtr_url=str(row["gtr_url"] or ""),
            ))
        return rows

    def _query_lexical_terms(self, masked: str) -> set[int]:
        matrix = self.lexical.transform([masked])
        return set(int(i) for i in matrix.indices)

    def _shared_terms(self, row_index: int, query_terms: set[int]) -> list[str]:
        row = self.x_lexical[row_index]
        both = query_terms.intersection(int(i) for i in row.indices)
        names = [str(self.lex_names[i]) for i in list(both)[:8]]
        names.sort()
        return names

    def _hosts(self, indices: np.ndarray) -> list[dict]:
        if len(indices) == 0:
            return []
        subset = self.frame.iloc[indices]
        rows = []
        for org, part in subset.groupby(subset["lead_org_name"].astype(str)):
            if not str(org).strip():
                continue
            amounts = part["award_gbp"].to_numpy(dtype=np.float64)
            positive = amounts[amounts > 0]
            mode = part["lead_funder"].astype(str).mode()
            rows.append({
                "organisation": html.unescape(str(org)),
                "projects": int(len(part)),
                "median_gbp": None if positive.size == 0 else round(float(np.median(positive)), 2),
                "top_council": "" if mode.empty else str(mode.iloc[0]),
            })
        rows.sort(key=lambda item: (-item["projects"], item["organisation"]))
        return rows[:8]

    def _schemes(self, indices: np.ndarray) -> list[dict]:
        if len(indices) == 0:
            return []
        counts = self.frame.iloc[indices]["grant_category"].astype(str).value_counts()
        total = int(counts.sum())
        return [
            {"scheme": html.unescape(str(name)), "projects": int(count), "share": round(int(count) / total, 4)}
            for name, count in counts.items()
        ]

    def _novelty(self, svd: np.ndarray) -> dict:
        order = np.argsort(-svd)
        top = svd[order[:10]]
        max_sim = float(top[0]) if top.size else 0.0
        mean_top10 = float(top.mean()) if top.size else 0.0
        percentile = float(np.searchsorted(self.novelty_reference, max_sim, side="right") / max(1, self.novelty_reference.size) * 100)
        band = novelty_band(percentile)
        return {
            "max_sim": round(max_sim, 4),
            "mean_top10_sim": round(mean_top10, 4),
            "percentile": round(percentile, 2),
            "band": band,
            "note": novelty_note(band, max_sim, percentile),
            "reference_n": int(self.novelty_reference.size),
            "reference": "max cosine of real UKRI titles (plus department and scheme) against the abstract SVD index",
        }

    def _constellation(self, q_svd: np.ndarray, neighbour_idx: np.ndarray) -> list[dict]:
        query_xy = project_query(q_svd, self.sky_mean, self.sky_components)
        points = [{
            "x": round(float(query_xy[0]), 4),
            "y": round(float(query_xy[1]), 4),
            "kind": "query",
            "title": "This proposal",
            "council": "",
            "award_gbp": None,
            "organisation": "",
            "similarity": None,
            "gtr_url": "",
        }]
        if len(neighbour_idx):
            cloud = self.z_unit[neighbour_idx].astype(np.float64)
            xy = (cloud - self.sky_mean) @ self.sky_components
            for pos, row_i in enumerate(neighbour_idx):
                row = self.frame.iloc[int(row_i)]
                points.append({
                    "x": round(float(xy[pos, 0]), 4),
                    "y": round(float(xy[pos, 1]), 4),
                    "kind": "neighbour",
                    "title": html.unescape(str(row["title"] or "")),
                    "council": str(row["lead_funder"]),
                    "award_gbp": float(row["award_gbp"]),
                    "organisation": html.unescape(str(row["lead_org_name"] or "")),
                    "similarity": None,
                    "gtr_url": str(row["gtr_url"] or ""),
                })
        centroids: dict[str, list[np.ndarray]] = {name: [] for name in self.classes}
        landmark_xy = (self.z_unit[self.landmark_row].astype(np.float64) - self.sky_mean) @ self.sky_components
        labels = self.frame["lead_funder"].astype(str).to_numpy()[self.landmark_row]
        for xy_row, label in zip(landmark_xy, labels):
            points.append({
                "x": round(float(xy_row[0]), 4),
                "y": round(float(xy_row[1]), 4),
                "kind": "landmark",
                "title": "",
                "council": str(label),
                "award_gbp": None,
                "organisation": "",
                "similarity": None,
                "gtr_url": "",
            })
            centroids[str(label)].append(xy_row)
        for council, rows in centroids.items():
            if not rows:
                continue
            centre = np.mean(rows, axis=0)
            points.append({
                "x": round(float(centre[0]), 4),
                "y": round(float(centre[1]), 4),
                "kind": "centroid",
                "title": council,
                "council": council,
                "award_gbp": None,
                "organisation": "",
                "similarity": None,
                "gtr_url": "",
            })
        return points

    def corpus_stats(self) -> dict:
        if self._stats is not None:
            return self._stats
        frame = self.frame
        amounts = frame["award_gbp"].astype(np.float64)
        zero = int((amounts <= 0).sum())
        by = (
            frame.assign(award=amounts)
            .groupby("lead_funder")
            .agg(projects=("project_id", "size"), total_gbp=("award", "sum"), median_gbp=("award", "median"))
            .reindex(COUNCILS)
        )
        yearly = (
            frame.assign(award=amounts)
            .groupby(["start_year", "lead_funder"], sort=True)
            .agg(projects=("project_id", "size"), total_gbp=("award", "sum"))
            .reset_index()
        )
        cats = frame.groupby("grant_category").size().sort_values(ascending=False)
        self._stats = {
            "projects": int(len(frame)),
            "total_gbp": round(float(amounts.sum()), 2),
            "zero_awards": zero,
            "zero_share": round(zero / len(frame), 4),
            "snapshot": "2026-11-23",
            "collected_at": self.model_info.get("data_collected"),
            "councils": [
                {
                    "council": name,
                    "projects": int(by.loc[name, "projects"]),
                    "total_gbp": round(float(by.loc[name, "total_gbp"]), 2),
                    "median_gbp": round(float(by.loc[name, "median_gbp"]), 2),
                    "colour": COUNCIL_COLOURS[name],
                }
                for name in COUNCILS
            ],
            "yearly": [
                {"year": int(r.start_year), "council": r.lead_funder, "projects": int(r.projects), "total_gbp": round(float(r.total_gbp), 2)}
                for r in yearly.itertuples(index=False)
            ],
            "grant_categories": [{"category": str(i), "projects": int(n)} for i, n in cats.items()],
            "year_min": int(frame["start_year"].min()),
            "year_max": int(frame["start_year"].max()),
            "cpih_annual": {str(y): v for y, v in sorted(annual_index().items()) if y >= 2006},
            "cpih_target_2026": TARGET_2026,
            "cpih_target_label": TARGET_LABEL,
            "cpih_source": SOURCE_URL,
        }
        return self._stats

    def rss_mb(self) -> float:
        """Resident memory in MB. Windows working set, Linux VmRSS, then POSIX rusage."""
        if sys.platform == "win32":
            return _windows_working_set_mb()
        status = Path("/proc/self/status")
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return round(float(line.split()[1]) / 1024.0, 1)
        except OSError:
            pass
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux and other POSIX report kilobytes. macOS reports bytes.
            scale = 1024 * 1024 if sys.platform == "darwin" else 1024
            return round(rss / scale, 1)
        except (ImportError, AttributeError, OSError, ValueError):
            return 0.0

    def to_csv(self, projects: list[SimilarProject]) -> bytes:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "similarity", "svd_cosine", "title", "lead_funder", "grant_category",
            "award_gbp_nominal", "award_gbp_real_2026", "fund_start", "fund_end",
            "lead_org_name", "department", "grant_ref", "gtr_url",
        ])
        for row in projects:
            writer.writerow([
                row.similarity, row.svd_cosine, row.title, row.lead_funder, row.grant_category,
                row.award_gbp, row.award_gbp_real, row.fund_start, row.fund_end,
                row.lead_org_name, row.department, row.grant_ref, row.gtr_url,
            ])
        return buf.getvalue().encode("utf-8")

    def to_markdown(self, title: str, analysis: Analysis) -> str:
        lines = [
            "# ScholarPulse UK brief",
            "",
            f"**{title or 'Untitled'}**",
            "",
            f"Top council: {analysis.top_council} ({analysis.councils[0].probability:.1%})" if analysis.councils else "",
            f"Confidence: {analysis.confidence}",
            f"Entropy: {analysis.entropy}",
            f"Latency: {analysis.latency_ms} ms",
            "",
            analysis.confidence_note or "",
            "",
            "## Councils",
        ]
        lines += [f"- {row.council}: {row.probability:.1%}" for row in analysis.councils]
        if analysis.award:
            lines += [
                "",
                f"## Award band ({analysis.award.basis})",
                f"P25–P75: £{analysis.award.p25:,.0f} – £{analysis.award.p75:,.0f}" if analysis.award.p25 is not None else "Not enough priced awards.",
                f"P10–P90: £{analysis.award.p10:,.0f} – £{analysis.award.p90:,.0f}" if analysis.award.p10 is not None else "",
                f"Median: £{analysis.award.median:,.0f}" if analysis.award.median is not None else "",
                analysis.award.zero_note,
                analysis.award.caveat,
            ]
        lines += ["", "## Similar funded projects"]
        for i, row in enumerate(analysis.similar, start=1):
            lines.append(f"{i}. {row.title} — {row.lead_funder}, £{row.award_gbp:,.0f}, {row.lead_org_name}. {row.gtr_url}")
        lines += ["", DISCLAIMER, OGL]
        return "\n".join(lines)

    def to_json(self, analysis: Analysis) -> str:
        return analysis.model_dump_json(indent=2)

    def simulate_what_if(self, title: str, abstract: str, add_terms: str = "", remove_terms: str = "", target_council: str | None = None) -> dict:
        before = self.predict_proba_masked(mask_funders(proposal_text(title, abstract)))
        edited = proposal_text(title, abstract)
        for term in [part.strip() for part in (remove_terms or "").split(",") if part.strip()]:
            edited = _drop_term(edited, term)
        if add_terms.strip():
            edited = f"{edited} {add_terms.strip()}"
        after = self.predict_proba_masked(mask_funders(edited))
        deltas = []
        for index, council in enumerate(self.classes):
            deltas.append({
                "council": council,
                "before": round(float(before[index]), 6),
                "after": round(float(after[index]), 6),
                "delta": round(float(after[index] - before[index]), 6),
            })
        deltas.sort(key=lambda item: item["delta"], reverse=True)
        target = None
        if target_council in self.classes:
            target = next(item for item in deltas if item["council"] == target_council)
        return {
            "deltas": deltas,
            "target": target,
            "edited_preview": edited[:500],
            "honesty": HONESTY,
        }

    def suggest_counterfactuals(self, title: str, abstract: str, target_council: str) -> dict:
        if target_council not in self.classes:
            return {"error": f"Unknown council {target_council}", "honesty": HONESTY}
        text = proposal_text(title, abstract)
        masked = mask_funders(text)
        before = self.predict_proba_masked(masked)
        current = self.classes[int(before.argmax())]
        present = self._contributions(masked, self.coef[self.classes.index(target_council)] - self.coef[self.classes.index(current)])
        opposing = [feature for feature, score, _ in sorted(present, key=lambda item: item[1]) if score < 0][:5]
        additions = self._council_phrases(target_council, avoid=masked.lower())
        add_text = " ".join(additions[:3])
        simulated = self.simulate_what_if(title, abstract, add_terms=add_text, remove_terms="", target_council=target_council)
        return {
            "current_council": current,
            "target_council": target_council,
            "opposing_phrases": opposing,
            "suggested_additions": additions[:5],
            "simulation_if_added": simulated["target"],
            "honesty": HONESTY,
            "source": "Suggestions are high-weight phrases for the target council that also appear in real funded titles of that council.",
        }

    def _council_phrases(self, council: str, avoid: str) -> list[str]:
        cache = getattr(self, "_phrase_cache", None)
        if cache is None:
            self._phrase_cache = {}
            cache = self._phrase_cache
        if council not in cache:
            pool = self.frame.index[self.frame["lead_funder"].astype(str) == council]
            sample = pool[:2500]
            counts: dict[str, int] = {}
            for title in self.frame.loc[sample, "title"].astype(str):
                tokens = self.token_re.findall(title.lower())
                seen = set(tokens)
                seen.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
                for term in seen:
                    if term in self.index and "funderx" not in term and len(term) > 3:
                        counts[term] = counts.get(term, 0) + 1
            class_index = self.classes.index(council)
            ranked = sorted(counts, key=lambda term: (self.coef[class_index, self.index[term]], counts[term]), reverse=True)
            cache[council] = [term for term in ranked if counts[term] >= 8][:40]
        return [term for term in cache[council] if term not in avoid][:8]

    def compare_proposals(self, proposals: list[dict]) -> dict:
        rows = []
        for item in proposals[:5]:
            result = self.analyze(item.get("title", ""), item.get("abstract", ""), top_k=10, cpih=True)
            host = result.host_institutions[0]["organisation"] if result.host_institutions else ""
            rows.append({
                "label": item.get("label") or item.get("title") or "Untitled",
                "ok": result.ok,
                "top_council": result.top_council,
                "probability": result.councils[0].probability if result.councils else None,
                "confidence": result.confidence,
                "conformal_set_90": result.conformal_set_90,
                "conformal_size": len(result.conformal_set_90),
                "novelty_band": (result.novelty or {}).get("band"),
                "median_gbp_2026": result.award.median if result.award else None,
                "top_host": host,
            })
        return {"rows": rows, "honesty": "Each row is a separate live score. It is not a ranking of the people who wrote the ideas."}

    def analyze_batch(self, records: list[dict]) -> dict:
        rows = []
        for record in records[:200]:
            result = self.analyze(record.get("title", ""), record.get("abstract", ""), top_k=5, cpih=bool(record.get("cpih")))
            rows.append({
                "id": record.get("id") or record.get("title") or "",
                "ok": result.ok,
                "errors": result.errors,
                "top_council": result.top_council,
                "probability": result.councils[0].probability if result.councils else None,
                "confidence": result.confidence,
                "conformal_set_90": result.conformal_set_90,
                "novelty_band": (result.novelty or {}).get("band"),
                "median_gbp": result.award.median if result.award else None,
                "latency_ms": result.latency_ms,
            })
        return {"n": len(rows), "rows": rows}

    def metrics(self) -> dict:
        values = list(self._latencies)
        ordered = sorted(values)
        def pct(q: float) -> float | None:
            if not ordered:
                return None
            pos = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
            return ordered[pos]
        return {
            "requests": self._requests,
            "latency_p50_ms": pct(0.50),
            "latency_p95_ms": pct(0.95),
            "rss_mb": self.rss_mb(),
            "projects": int(len(self.frame)),
            "precision_available": ["fp64", "int8"],
            "stored_abstracts": 0,
        }

    def persona(self, persona_id: str) -> dict:
        from fundmatch.student import persona_brief
        return persona_brief(persona_id)

    def suggest_topics(self, interest: str, persona: str = "masters", n: int = 5) -> dict:
        from fundmatch.student import suggest_topics
        return suggest_topics(self, interest, persona, n)

    def topic_trends(self, text: str) -> dict:
        from fundmatch.student import topic_trends
        return topic_trends(self, text)

    def readiness(self, title: str, abstract: str, persona: str = "masters") -> dict:
        from fundmatch.student import readiness
        return readiness(self, title, abstract, persona)

    def fuse(self, field_a: str, field_b: str) -> dict:
        from fundmatch.student import fuse
        return fuse(self, field_a, field_b)

    def starter_pack(self, title: str, abstract: str, persona: str = "masters") -> dict:
        from fundmatch.student import starter_pack
        return starter_pack(self, title, abstract, persona)


def _date(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _highlights(text: str, wanted: dict[str, float]) -> list[Highlight]:
    """Find the contributing phrases in the original text, longest first."""
    if not text or not wanted:
        return []
    lowered = text.lower()
    occupied = np.zeros(len(text), dtype=bool)
    spans: list[Highlight] = []
    for feature, score in sorted(wanted.items(), key=lambda item: (len(item[0].split()), abs(item[1])), reverse=True):
        needle = feature.lower()
        start = 0
        while True:
            at = lowered.find(needle, start)
            if at < 0:
                break
            end = at + len(needle)
            # Require boundaries so "ion" does not light up inside "innovation".
            left_ok = at == 0 or not lowered[at - 1].isalnum()
            right_ok = end == len(lowered) or not lowered[end].isalnum()
            if left_ok and right_ok and not occupied[at:end].any():
                occupied[at:end] = True
                spans.append(Highlight(
                    start=at, end=end, feature=feature, contribution=round(score, 6),
                    polarity="positive" if score >= 0 else "negative",
                ))
            start = end
    spans.sort(key=lambda item: item.start)
    return spans


def _windows_working_set_mb() -> float:
    """Working set via kernel32/psapi. No extra packages. 0.0 if the call fails."""
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        process = kernel32.GetCurrentProcess()
        getter = None
        for library, name in (("kernel32", "K32GetProcessMemoryInfo"), ("psapi", "GetProcessMemoryInfo")):
            try:
                dll = kernel32 if library == "kernel32" else ctypes.WinDLL("psapi", use_last_error=True)
                getter = getattr(dll, name)
                break
            except (AttributeError, OSError):
                getter = None
        if getter is None:
            return 0.0
        getter.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        getter.restype = wintypes.BOOL
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not getter(process, ctypes.byref(counters), counters.cb):
            return 0.0
        return round(counters.WorkingSetSize / (1024 * 1024), 1)
    except (AttributeError, OSError, ValueError):
        return 0.0


_ENGINE: FundMatchEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> FundMatchEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = FundMatchEngine()
    return _ENGINE
