"""Int8 export of the logistic-regression weights (CS5079 lecture 8).

The float32 coefficient matrix is the deployed model. This module writes a
per-class int8 copy and measures how often it picks the same council.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from fundmatch.engine import ART, FundMatchEngine


def export_int8(engine: FundMatchEngine, path: Path | None = None) -> dict:
    coef = engine.coef.astype(np.float64)
    scales = np.max(np.abs(coef), axis=1)
    scales[scales == 0] = 1.0
    coef_i8 = np.clip(np.rint(coef / scales[:, None] * 127.0), -127, 127).astype(np.int8)
    dest = path or (ART / "tfidf_lr_weights_int8.npz")
    np.savez_compressed(dest, coef_int8=coef_i8, scales=scales.astype(np.float32), intercept=engine.intercept.astype(np.float32))
    hat = (coef_i8.astype(np.float64) / 127.0) * scales[:, None]
    fp32_bytes = coef.astype(np.float32).nbytes + engine.intercept.astype(np.float32).nbytes
    int8_bytes = dest.stat().st_size
    return {
        "path": str(dest),
        "fp32_coef_bytes": int(fp32_bytes),
        "int8_file_bytes": int(int8_bytes),
        "bytes_saved": int(fp32_bytes - int8_bytes),
        "max_abs_coef_error": float(np.max(np.abs(coef - hat))),
        "coef_hat": hat,
    }


def fidelity(engine: FundMatchEngine, texts: list[str], coef_hat: np.ndarray) -> dict:
    """Top-1 agreement and max probability gap of int8 weights against float64."""
    from fundmatch.text import mask_funders

    agree = 0
    max_delta = 0.0
    # Treat the float64 prediction as the reference label so we can also report
    # a macro-F1 of the int8 model against that reference.
    ref_labels = []
    pred_labels = []
    started = time.perf_counter()
    for text in texts:
        masked = mask_funders(text)
        matrix = engine.vectorize([masked])
        logits = np.asarray(matrix @ engine.coef.T).ravel() + engine.intercept
        logits = logits / engine.temperature
        logits_q = np.asarray(matrix @ coef_hat.T).ravel() + engine.intercept
        logits_q = logits_q / engine.temperature
        probs = _stable_softmax(logits)
        probs_q = _stable_softmax(logits_q)
        ref = int(probs.argmax())
        pred = int(probs_q.argmax())
        ref_labels.append(ref)
        pred_labels.append(pred)
        agree += int(ref == pred)
        max_delta = max(max_delta, float(np.max(np.abs(probs - probs_q))))
    elapsed = (time.perf_counter() - started) * 1000 / max(1, len(texts))
    labels = list(range(len(engine.classes)))
    return {
        "n": len(texts),
        "top1_agreement": round(agree / max(1, len(texts)), 6),
        "max_probability_delta": round(max_delta, 6),
        "macro_f1_vs_fp64": round(_macro_f1(ref_labels, pred_labels, labels), 6),
        "ms_per_text": round(elapsed, 3),
    }


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def _macro_f1(gold: list[int], pred: list[int], labels: list[int]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(scores)) if scores else 1.0
