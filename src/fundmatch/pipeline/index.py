"""How the 64-d int8 abstract index is queried.

``C`` is the TruncatedSVD component matrix in the same 200,000-term TF-IDF
space as the classifier. ``Z`` is the L2-normalised projection of every
abstract, stored as int8 with a per-row scale.

Reconstruction, as exported::

    C_rec = (C_int8.astype(float32) / 127) * c_scales[:, None]
    q_svd = normalise(q_tfidf @ C_rec.T)
    scores = ((Z_int8.astype(float32) @ q_svd) / 127) * z_scales

The engine keeps the reconstructed rows (about 29 MB) so a warm query stays
under a quarter of a second. Lexical ranks come from a TF-IDF index of title,
department and grant category, fused with reciprocal rank fusion at k = 60.
"""

from __future__ import annotations

RRF_K = 60
SVD_DIM = 64
N_PROJECTS = 113233
