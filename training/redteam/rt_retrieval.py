"""FR-13 red-team: BM25, TF-IDF, MiniLM, BGE and SPECTER, searched exactly and with FAISS.

Two label sources, on the notebook's test split:
(a) Known-item search, identical to notebook Section 7.2: the same 5,000 test
    abstracts (seed 42) form the corpus and 2,000 of their titles are queries.
(b) Topic pseudo-labels: 500 test queries (title + abstract) against a fixed
    random 5,000-project subset of the training split; a hit shares a GtR
    research topic. A subset is used because SPECTER (110M parameters) has to
    embed every corpus document on CPU and each stage must finish in ~10 min.

Run in stages (each merges into results/retrieval_fr13.json):
    python rt_retrieval.py lexical
    python rt_retrieval.py dense minilm|bge|specter
    python rt_retrieval.py faiss
FAISS: the full 113,233 x 384 BGE matrix from the notebook cache is indexed
with an exact IndexFlatIP and an approximate HNSW index; we report recall of
the exact top 10, query latency and index size.
"""

from __future__ import annotations

import json
import os
import sys
import re
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from rt_common import RESULTS, SEED, build_split, mask_funders, save_result

MODELS = {
    "minilm": ("MiniLM (all-MiniLM-L6-v2)", "sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge": ("BGE (bge-small-en-v1.5)", "BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    "specter": ("SPECTER (allenai-specter)", "sentence-transformers/allenai-specter", ""),
}
N_TOPIC_Q, N_TOPIC_C = 500, 5000
TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tok(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def rank_metrics(S: np.ndarray, k: int = 10) -> dict:
    s_true = S[np.arange(S.shape[0]), np.arange(S.shape[0])]
    rank = (S > s_true[:, None]).sum(1) + 1
    return {"Recall@10": float(np.mean(rank <= k)), "MRR@10": float(np.mean(np.where(rank <= k, 1 / rank, 0)))}


def topic_metrics(S: np.ndarray, q_topics, c_topics, k: int = 10) -> dict:
    top = np.argsort(-S, axis=1)[:, :k]
    hit, rr, prec = [], [], []
    for qt, row in zip(q_topics, top):
        rel = [len(qt & c_topics[j]) > 0 for j in row]
        hit.append(any(rel))
        rr.append(1 / (rel.index(True) + 1) if any(rel) else 0.0)
        prec.append(np.mean(rel))
    return {"Hit@10 (topic)": float(np.mean(hit)), "MRR@10 (topic)": float(np.mean(rr)), "P@10 (topic)": float(np.mean(prec))}


def bm25_scores(queries: list[str], docs: list[str], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> np.ndarray:
    """Okapi BM25 with the same formula and idf floor as ``rank_bm25.BM25Okapi``, vectorised with
    sparse matrices (the pure-Python library took hours on 1,000 x 20,000 full-text pairs).
    ``test_bm25_matches_rank_bm25`` in this folder checks it against the library on a small corpus."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(analyzer=tok)
    tf = cv.fit_transform(docs).tocsr().astype(np.float64)
    n_docs = tf.shape[0]
    dl = np.asarray(tf.sum(axis=1)).ravel()
    avgdl = dl.mean()
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
    floor = epsilon * idf.mean()
    idf[idf < 0] = floor
    norm = k1 * (1 - b + b * dl / avgdl)
    rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
    data = tf.data * (k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
    W = sp.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape)
    Q = cv.transform(queries).astype(np.float64)
    return np.asarray((Q @ W.T).todense(), dtype=np.float32)


def data():
    tr, te = build_split()
    topics = lambda s: s.fillna("").str.split("; ").map(lambda l: {x for x in l if x})  # noqa: E731
    rng = np.random.RandomState(SEED)  # (a) known-item, identical sampling to notebook 7.2
    corp = rng.choice(len(te), min(5000, len(te)), replace=False)
    qs = corp[:2000]
    d = {"docs": te.abstract.iloc[corp].fillna("").map(mask_funders).tolist(),
         "queries": te.title.iloc[qs].fillna("").map(mask_funders).tolist()}
    rng2 = np.random.RandomState(SEED + 1)  # (b) topic labels
    te_t, tr_t = topics(te.research_topics), topics(tr.research_topics)
    tq = rng2.choice(np.where(te_t.map(len) > 0)[0], N_TOPIC_Q, replace=False)
    tc = rng2.choice(len(tr), N_TOPIC_C, replace=False)
    d.update(t_queries=te.text.iloc[tq].tolist(), t_docs=tr.text.iloc[tc].tolist(),
             q_topics=te_t.iloc[tq].tolist(), c_topics=tr_t.iloc[tc].tolist(), tr_text=tr.text)
    return d


def merge(section: str, key: str, value) -> None:
    path = RESULTS / "retrieval_fr13.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out.setdefault(section, {})[key] = value
    save_result("retrieval_fr13", out)


def lexical() -> None:
    d = data()
    merge("setup", "known_item", {"queries": len(d["queries"]), "corpus": len(d["docs"])})
    merge("setup", "topic", {"queries": len(d["t_queries"]), "corpus": len(d["t_docs"]),
                             "corpus_share_with_topics": float(np.mean([len(c) > 0 for c in d["c_topics"]]))})
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True, stop_words="english",
                          max_features=200_000, dtype=np.float32).fit(d["tr_text"])
    merge("known_item", "TF-IDF cosine", rank_metrics((vec.transform(d["queries"]) @ vec.transform(d["docs"]).T).toarray()))
    merge("topic", "TF-IDF cosine", topic_metrics((vec.transform(d["t_queries"]) @ vec.transform(d["t_docs"]).T).toarray(),
                                                  d["q_topics"], d["c_topics"]))
    merge("known_item", "BM25 (Okapi)", rank_metrics(bm25_scores(d["queries"], d["docs"])))
    merge("topic", "BM25 (Okapi)", topic_metrics(bm25_scores(d["t_queries"], d["t_docs"]), d["q_topics"], d["c_topics"]))
    print(open(RESULTS / "retrieval_fr13.json").read(), flush=True)


def dense(key: str) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(os.cpu_count() or 4)
    name, model_id, prefix = MODELS[key]
    d = data()
    enc = SentenceTransformer(model_id, device="cpu")
    enc.max_seq_length = 256
    t0 = time.time()
    Ed = enc.encode(d["docs"], batch_size=32, normalize_embeddings=True)
    Eq = enc.encode([prefix + q for q in d["queries"]], batch_size=64, normalize_embeddings=True)
    merge("known_item", name, rank_metrics(Eq @ Ed.T))
    Etd = enc.encode(d["t_docs"], batch_size=32, normalize_embeddings=True)
    Etq = enc.encode(d["t_queries"], batch_size=32, normalize_embeddings=True)
    merge("topic", name, topic_metrics(Etq @ Etd.T, d["q_topics"], d["c_topics"]))
    n = sum(len(d[k]) for k in ("docs", "queries", "t_docs", "t_queries"))
    sec = time.time() - t0
    merge("encode_timing_cpu", name, {"seconds": round(sec, 1), "texts": n, "texts_per_second": round(n / sec, 1),
                                      "dim": int(Ed.shape[1]), "max_seq_length": 256})
    print(name, open(RESULTS / "retrieval_fr13.json").read(), flush=True)


def faiss_bench() -> None:
    import faiss

    emb_path = Path(os.environ.get("FM_BGE_NPY", Path(__file__).resolve().parents[1] / "data" / "processed" / "emb_bge.npy"))
    E = np.load(emb_path).astype(np.float32)
    faiss.normalize_L2(E)
    rngq = np.random.RandomState(7)
    Q = E[rngq.choice(len(E), 500, replace=False)] + 0.01 * rngq.standard_normal((500, E.shape[1])).astype(np.float32)
    faiss.normalize_L2(Q)
    t0 = time.time(); exact_np = np.argsort(-(Q @ E.T), axis=1)[:, :10]; np_ms = (time.time() - t0) * 1000 / len(Q)
    flat = faiss.IndexFlatIP(E.shape[1]); flat.add(E)
    t0 = time.time(); _, I_flat = flat.search(Q, 10); flat_ms = (time.time() - t0) * 1000 / len(Q)
    hnsw = faiss.IndexHNSWFlat(E.shape[1], 32, faiss.METRIC_INNER_PRODUCT); hnsw.hnsw.efSearch = 64
    t0 = time.time(); hnsw.add(E); build_s = time.time() - t0
    t0 = time.time(); _, I_h = hnsw.search(Q, 10); h_ms = (time.time() - t0) * 1000 / len(Q)
    single = []
    for q in Q[:100]:
        t1 = time.time(); hnsw.search(q[None, :], 10); single.append((time.time() - t1) * 1000)
    overlap = lambda A, B: float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(A, B)]))  # noqa: E731
    faiss.write_index(hnsw, "/tmp/hnsw.idx"); faiss.write_index(flat, "/tmp/flat.idx")
    merge("faiss", "bge_full_corpus", {
        "corpus": int(E.shape[0]), "dim": int(E.shape[1]), "queries": int(len(Q)),
        "numpy_exact_ms_per_query_batched": round(np_ms, 2),
        "faiss_flat_ms_per_query_batched": round(flat_ms, 2), "faiss_flat_recall_vs_exact": overlap(I_flat, exact_np),
        "faiss_hnsw_ms_per_query_batched": round(h_ms, 3), "faiss_hnsw_ms_single_query_median": round(float(np.median(single)), 3),
        "faiss_hnsw_recall_vs_exact": overlap(I_h, exact_np), "hnsw_build_seconds": round(build_s, 1),
        "flat_index_mb": round(os.path.getsize("/tmp/flat.idx") / 2**20, 1),
        "hnsw_index_mb": round(os.path.getsize("/tmp/hnsw.idx") / 2**20, 1)})
    print(open(RESULTS / "retrieval_fr13.json").read(), flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "lexical"
    {"lexical": lexical, "faiss": faiss_bench}.get(stage, lambda: dense(sys.argv[2]))()
