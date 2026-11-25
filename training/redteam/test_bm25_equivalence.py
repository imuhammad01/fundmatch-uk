"""Check the vectorised BM25 in rt_retrieval.py against rank_bm25 on a small corpus."""
import numpy as np
from rank_bm25 import BM25Okapi

from rt_retrieval import bm25_scores, tok

docs = ["soil carbon and fungi in forest soils", "quantum control of superconducting qubits", "fungal decay of wood",
        "soil microbes and nitrogen", "machine learning for qubits and quantum error", "forest carbon budgets in the uk"]
queries = ["fungi wood decay", "quantum qubits quantum", "carbon soil forest", "unknownword"]
ref = np.vstack([BM25Okapi([tok(d) for d in docs]).get_scores(tok(q)) for q in queries])
got = bm25_scores(queries, docs)
print("max abs diff", float(np.abs(ref - got).max()))
assert np.allclose(ref, got, atol=1e-5)
