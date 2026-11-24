"""Training settings of the deployed council classifier.

The live service does not retrain. It loads ``artifacts/tfidf_lr_weights.npz``,
which was exported from a scikit-learn pipeline with these exact settings and
checked to a maximum logit difference of 0 against that pipeline:

- ``TfidfVectorizer`` word and bigram features, at most 200,000 terms
- ``sublinear_tf=True``, ``norm='l2'``, ``min_df=3``, ``max_df=0.5``
- class-weighted ``LogisticRegression``, ``C=32``
- temperature ``T = 1.1314305006097425`` fitted on out-of-fold train probabilities
- funder names masked before the vectorizer sees the text
- stratified group split, 90,585 train and 22,648 test, seed 42

Held-out test of that fit: macro-F1 0.8615, accuracy 0.8693, top-3 0.9916.
ECE moved from 0.0336 to 0.0180 after temperature scaling. Accuracy did not change.
"""

from __future__ import annotations

TEMPERATURE = 1.1314305006097425
TRAIN_ROWS = 90585
TEST_ROWS = 22648
TEST_MACRO_F1 = 0.8615
TEST_ACCURACY = 0.8693
TEST_TOP3 = 0.9916

VECTORIZER = {
    "ngram_range": (1, 2),
    "max_features": 200_000,
    "min_df": 3,
    "max_df": 0.5,
    "sublinear_tf": True,
    "norm": "l2",
    "lowercase": True,
    "token_pattern": r"(?u)\b\w\w+\b",
}

CLASSIFIER = {"C": 32.0, "class_weight": "balanced", "solver": "lbfgs"}


def temperature_scale(probabilities, temperature: float = TEMPERATURE):
    """``p ** (1/T)`` renormalised. Equivalent to dividing the logits by T."""
    import numpy as np

    logp = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    shifted = np.exp(logp - np.max(logp))
    return shifted / shifted.sum()
