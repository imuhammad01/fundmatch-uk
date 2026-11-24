"""
fundmatch_core.py - model logic of FundMatch UK, kept free of any user-interface code so it can be unit
tested (tests/test_core.py) and reused outside Streamlit.

Functions
  load_assets(dir)        load the configuration the notebook exported (assets/model_info.json)
  check_input(title, abstract, vec)  validation: empty, too short, probably not English, no known words
  predict(text, assets)   council probabilities (temperature-calibrated) and the query vector for search
  explain(text, council, assets)     words that pushed the TF-IDF + LogReg model towards a council
  similar(q, assets, ...) exact cosine search with council / year / scheme filters
  award_range(amounts)    percentile band of the neighbours' awards
"""
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

MIN_WORDS = 30
LOW_CONF = 0.60  # test accuracy was only ~50 % when the top probability was below this (notebook Section 8)
COUNCIL_INFO = {
    "AHRC": "Arts and Humanities Research Council", "BBSRC": "Biotechnology and Biological Sciences Research Council",
    "EPSRC": "Engineering and Physical Sciences Research Council", "ESRC": "Economic and Social Research Council",
    "Innovate UK": "Innovate UK (business-led innovation)", "MRC": "Medical Research Council",
    "NERC": "Natural Environment Research Council", "STFC": "Science and Technology Facilities Council"}

# UKRI Funding Finder council filter IDs (read from the filter checkboxes on https://www.ukri.org/opportunity/
# on 23 Nov 2026; every URL below returned HTTP 200 and listed only that council's open calls).
FUNDING_FINDER_IDS = {"AHRC": 814, "BBSRC": 816, "ESRC": 818, "EPSRC": 820, "Innovate UK": 822, "MRC": 824,
                      "NERC": 826, "STFC": 830}
FUNDING_FINDER = "https://www.ukri.org/opportunity/"
INNOVATION_FUNDING_SERVICE = "https://apply-for-innovation-funding.service.gov.uk/competition/search"


def funding_finder_url(council: str) -> str:
    """Link to the council's currently open calls on the UKRI Funding Finder (general finder if unknown)."""
    cid = FUNDING_FINDER_IDS.get(council)
    if cid is None:
        return FUNDING_FINDER
    return f"{FUNDING_FINDER}?filter_council%5B%5D={cid}&filter_status%5B%5D=open&filter_submitted=true"


# Same masking as in the notebook (Section 3.1) so inputs match the training text.
FUNDER_TERMS = [r"engineering and physical sciences research council", r"biotechnology and biological sciences research council",
    r"economic and social research council", r"medical research council", r"arts and humanities research council",
    r"natural environment research council", r"science and technology facilities council", r"innovate ?uk",
    r"technology strategy board", r"uk research and innovation", r"research councils? uk",
    r"epsrc", r"bbsrc", r"esrc", r"mrc", r"ahrc", r"nerc", r"stfc", r"ukri", r"rcuk", r"ktp",
    r"knowledge transfer partnerships?"]
FUNDER_RE = re.compile(r"\b(?:" + "|".join(FUNDER_TERMS) + r")\b", re.I)
# Very common English function words; real English abstracts have well over 20 % of these.
EN_STOP = set("the and of to in a is for with on that this we will be are by as an from or at it its which "
              "these their our has have been can how into also such using use new not more".split())
ENCODERS = {"bge": "BAAI/bge-small-en-v1.5", "minilm": "sentence-transformers/all-MiniLM-L6-v2"}


def mask_funders(text):
    """Replace funder names and funder-exclusive scheme names with a neutral token."""
    return FUNDER_RE.sub(" funderx ", text)


def english_share(text):
    """Share of word tokens that are common English function words (0 for no words)."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    return sum(w in EN_STOP for w in words) / len(words) if words else 0.0


def check_input(title, abstract, vec=None):
    """Return (errors, warnings). Errors stop the search; warnings are shown with the results."""
    title, abstract = (title or "").strip(), (abstract or "").strip()
    text = f"{title}. {abstract}".strip(". ")
    errors, warnings = [], []
    n = len(text.split())
    if n == 0:
        return ["Please enter a title and an abstract."], []
    if n < MIN_WORDS:
        errors.append(f"Please give more detail: {n} words entered, at least {MIN_WORDS} are needed for a reliable match.")
        return errors, warnings
    if english_share(text) < 0.15:
        warnings.append("This text does not look like English. The model was trained on English UKRI abstracts, "
                        "so results for other languages are unreliable; an English summary works best.")
    if vec is not None and vec.transform([mask_funders(text)]).nnz < 10:
        errors.append("Too few recognised research words to compare with funded projects. Please describe the aims and methods.")
    return errors, warnings


def _stack_csr(paths):
    """Row-stack CSR shards into preallocated arrays, one shard in memory at a time (sp.vstack would briefly
    hold every shard twice, which raised the app's peak RAM by about 30 MB)."""
    import scipy.sparse as sp
    shapes, nnz = [], 0
    for p in paths:  # read only the small header arrays first
        with np.load(p) as z:
            shapes.append(tuple(z["shape"])); nnz += int(z["indptr"][-1])
    first = sp.load_npz(paths[0]).tocsr()
    data = np.empty(nnz, dtype=first.data.dtype); indices = np.empty(nnz, dtype=np.int32)
    indptr = np.empty(sum(s[0] for s in shapes) + 1, dtype=np.int64); indptr[0] = 0
    r = o = 0
    for i, p in enumerate(paths):
        X = first if i == 0 else sp.load_npz(p).tocsr()
        n, m = X.shape[0], X.nnz
        data[o:o + m] = X.data; indices[o:o + m] = X.indices; indptr[r + 1:r + n + 1] = X.indptr[1:] + o
        r += n; o += m; del X
        first = None
    return sp.csr_matrix((data, indices, indptr), shape=(r, shapes[0][1]))


def load_assets(assets_dir):
    """Load the deployed configuration described in model_info.json (written by the notebook, Section 10)."""
    info = json.load(open(os.path.join(assets_dir, "model_info.json")))
    clfs = [(c["input"], pickle.load(open(os.path.join(assets_dir, c["file"]), "rb"))) for c in info["classifiers"]]
    vec = next((m["vec"] for kind, m in clfs if kind == "tfidf"), None)
    if vec is None and os.path.exists(os.path.join(assets_dir, "tfidf_vec.pkl")):
        vec = pickle.load(open(os.path.join(assets_dir, "tfidf_vec.pkl"), "rb"))["vec"]
    if info["retriever"] == "tfidf":
        import scipy.sparse as sp
        # stored in row shards so every file stays under GitHub's 100 MB limit
        shards = info.get("index_shards") or [info["index"]]
        index = _stack_csr([os.path.join(assets_dir, f) for f in shards])
    else:
        index = np.load(os.path.join(assets_dir, info["index"]))
    encoders = {}
    needed = {kind for kind, _ in clfs if kind != "tfidf"} | ({info["retriever"]} if info["retriever"] != "tfidf" else set())
    if needed:  # torch is imported only when a neural encoder is part of the deployed configuration
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2)
        for k in needed:
            encoders[k] = SentenceTransformer(ENCODERS[k], device="cpu")
            encoders[k].max_seq_length = 256
    meta = pd.read_parquet(os.path.join(assets_dir, "projects_meta.parquet"))
    assert index.shape[0] == len(meta), "index and metadata are out of sync"
    return {"info": info, "clfs": clfs, "vec": vec, "index": index, "encoders": encoders, "meta": meta}


def temperature_scale(p, T):
    """Calibrate probabilities: p_i ** (1/T), renormalised (T fitted on out-of-fold data in the notebook)."""
    if T == 1.0:
        return p
    logp = np.log(np.clip(p, 1e-12, 1)) / T
    e = np.exp(logp - logp.max())
    return e / e.sum()


def predict(text, A):
    """Council probabilities (Series sorted high to low) and the query representation used for search."""
    masked = mask_funders(text)
    X = A["vec"].transform([masked]) if A["vec"] is not None else None
    E = {k: e.encode([masked], normalize_embeddings=True)[0].astype(np.float32) for k, e in A["encoders"].items()}
    p = np.mean([m["clf"].predict_proba(X if kind == "tfidf" else E[kind][None, :])[0] for kind, m in A["clfs"]], axis=0)
    p = temperature_scale(p, float(A["info"].get("temperature", 1.0)))
    classes = A["clfs"][0][1]["classes"]
    q = X if A["info"]["retriever"] == "tfidf" else E[A["info"]["retriever"]]
    return pd.Series(p, index=classes).sort_values(ascending=False), q


def explain(text, council, A, k=8):
    """Words (uni/bi-grams) with the largest positive contribution tfidf(w) x weight(council, w) in the
    TF-IDF + LogReg model. Exact for a linear model; empty if no TF-IDF LogReg model is deployed."""
    m = next((m for kind, m in A["clfs"] if kind == "tfidf" and hasattr(m["clf"], "coef_")), None)
    if m is None:
        return []
    x = m["vec"].transform([mask_funders(text)]).tocsr()
    c = list(m["classes"]).index(council)
    contrib = x.data * m["clf"].coef_[c, x.indices]
    order = [j for j in np.argsort(-contrib) if contrib[j] > 0][: k * 2]
    want = {int(x.indices[j]): float(contrib[j]) for j in order}
    names = {i: w for w, i in m["vec"].vocabulary_.items() if i in want}   # reverse lookup only for the few we need
    out = [(names[i], s) for i, s in want.items() if "funderx" not in names[i]]
    return out[:k]


def similar(q, A, k=10, councils=None, years=None, scheme="Any scheme"):
    """Exact cosine search (all vectors are L2-normalised) with optional council, year and scheme filters."""
    index, meta = A["index"], A["meta"]
    if hasattr(index, "tocsr"):            # sparse TF-IDF index
        scores = (index @ q.T).toarray().ravel()
    else:                                  # float16 dense index, converted in chunks for an accurate dot product
        scores = np.concatenate([index[i:i + 20000].astype(np.float32) @ q for i in range(0, index.shape[0], 20000)])
    mask = np.ones(len(meta), dtype=bool)
    if councils is not None:
        mask &= meta["lead_funder"].isin(councils).to_numpy(copy=True)   # copy: pandas 3 returns read-only views
    if years is not None:
        sy = meta["start_year"].to_numpy(dtype=float, na_value=np.nan)
        mask &= np.isnan(sy) | ((sy >= years[0]) & (sy <= years[1]))
    if scheme and scheme != "Any scheme":
        mask &= (meta["grant_category"] == scheme).to_numpy()
    scores[~mask] = -np.inf
    k = min(k, int(mask.sum()))
    if k <= 0:
        return meta.iloc[[]].assign(similarity=[])
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    out = meta.iloc[top].copy()
    out.insert(0, "similarity", scores[top].round(3))
    return out


def award_range(amounts, min_n=5):
    """P10/P25/P50/P75/P90 of positive award amounts, or None if fewer than min_n are known."""
    a = pd.Series(amounts, dtype=float)
    a = a[a > 0]
    if len(a) < min_n:
        return None
    return dict(zip(["p10", "p25", "p50", "p75", "p90"], np.percentile(a, [10, 25, 50, 75, 90]).tolist()))


def gbp(x):
    """Format pounds: £123k or £1.23m."""
    return "n/a" if x is None or pd.isna(x) else (f"£{x/1e6:,.2f}m" if x >= 1e6 else f"£{x/1e3:,.0f}k")
