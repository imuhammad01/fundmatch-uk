"""FR-12 red-team: a real (small, CPU) fine-tune of a transformer classifier.

Fine-tuning SciBERT on all 90,585 training projects needs a GPU. To give an
honest, measured comparison we fine-tune on a stratified subsample and compare
against TF-IDF + LR trained on the *same* subsample and scored on the *same*
test subsample:

- train: N_TRAIN projects (default 3,000) from the notebook's training split (stratified, seed 42)
- validation: N_VAL (default 500) other training projects, used to pick the epoch (no test peeking)
- test: N_TEST (default 1,500) projects from the notebook's test split (stratified, seed 42)
- training stops early if it exceeds MAX_TRAIN_SECONDS; the result records how many steps ran
- class-weighted cross-entropy, AdamW lr 3e-5, 10% warm-up, batch 16,
  first 128 word-pieces of "title. abstract" (funder names masked), 2 epochs.

Usage (two bounded stages, merged into results/finetune_<model>.json):
    python rt_finetune.py baseline
    python rt_finetune.py finetune allenai/scibert_scivocab_uncased
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

from rt_common import COUNCILS, SEED, build_split, save_result, tfidf_lr

N_TRAIN = int(os.environ.get("FT_N_TRAIN", 3000))
N_VAL = int(os.environ.get("FT_N_VAL", 500))
N_TEST = int(os.environ.get("FT_N_TEST", 1500))
MAX_LEN, BATCH, EPOCHS, LR = int(os.environ.get("FT_MAX_LEN", 128)), 16, int(os.environ.get("FT_EPOCHS", 2)), 3e-5
MAX_TRAIN_SECONDS = int(os.environ.get("FT_MAX_SECONDS", 480))  # hard budget so a CPU run stays bounded


def stratified(frame, n, seed):
    idx, _ = train_test_split(np.arange(len(frame)), train_size=n, stratify=frame.lead_funder, random_state=seed)
    return frame.iloc[idx].reset_index(drop=True)


def metrics(y, proba) -> dict:
    pred = proba.argmax(1)
    return {"macro_f1": float(f1_score(y, pred, average="macro")), "accuracy": float(accuracy_score(y, pred)),
            "top3": float(top_k_accuracy_score(y, proba, k=3, labels=range(8))),
            "per_class_f1": dict(zip(COUNCILS, map(float, f1_score(y, pred, average=None, labels=range(8)))))}


def subsets():
    tr, te = build_split()
    pool = stratified(tr, N_TRAIN + N_VAL, SEED)
    sub_tr, sub_val = train_test_split(pool, test_size=N_VAL, stratify=pool.lead_funder, random_state=SEED)
    sub_te = stratified(te, N_TEST, SEED)
    return tr, sub_tr, sub_val, sub_te


def enc(frame):
    return np.array([COUNCILS.index(c) for c in frame.lead_funder])


def baseline() -> None:
    import json

    tr, sub_tr, sub_val, sub_te = subsets()
    y_tr, y_te = enc(sub_tr), enc(sub_te)
    out = {"n_train": len(sub_tr), "n_val": len(sub_val), "n_test": len(sub_te), "seed": SEED}
    vec, clf = tfidf_lr()
    vec.set_params(min_df=2)
    t0 = time.time()
    clf.fit(vec.fit_transform(sub_tr.text), y_tr)
    out["tfidf_lr_same_subsample"] = metrics(y_te, clf.predict_proba(vec.transform(sub_te.text)))
    out["tfidf_lr_same_subsample"]["train_seconds"] = round(time.time() - t0, 1)
    vec_f, clf_f = tfidf_lr()
    t0 = time.time()
    clf_f.fit(vec_f.fit_transform(tr.text), enc(tr))
    out["tfidf_lr_full_train_on_same_test"] = metrics(y_te, clf_f.predict_proba(vec_f.transform(sub_te.text)))
    out["tfidf_lr_full_train_on_same_test"]["train_seconds"] = round(time.time() - t0, 1)
    save_result("finetune_baselines", out)
    print(json.dumps(out, indent=1)[:1500], flush=True)


def main(model_id: str) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 4)
    _, sub_tr, sub_val, sub_te = subsets()
    y_tr, y_val, y_te = enc(sub_tr), enc(sub_val), enc(sub_te)
    out = {"model": model_id, "n_train": len(sub_tr), "n_val": len(sub_val), "n_test": len(sub_te),
           "max_len": MAX_LEN, "batch": BATCH, "epochs": EPOCHS, "lr": LR, "seed": SEED,
           "max_train_seconds": MAX_TRAIN_SECONDS, "threads": torch.get_num_threads()}

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=8)
    out["parameters_millions"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)

    def tensors(frame, y):
        t = tok(frame.text.tolist(), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        return TensorDataset(t["input_ids"], t["attention_mask"], torch.tensor(y))

    dl_tr = DataLoader(tensors(sub_tr, y_tr), batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    dl_val = DataLoader(tensors(sub_val, y_val), batch_size=64)
    dl_te = DataLoader(tensors(sub_te, y_te), batch_size=64)
    counts = np.bincount(y_tr, minlength=8)
    weights = torch.tensor(len(y_tr) / (8 * counts), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(dl_tr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    def predict(dl):
        model.eval()
        outs = []
        with torch.no_grad():
            for ids, mask, _ in dl:
                outs.append(torch.softmax(model(input_ids=ids, attention_mask=mask).logits, -1).numpy())
        return np.vstack(outs)

    t0 = time.time()
    history, best_f1, best_state = [], -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        stopped = False
        for step, (ids, mask, y) in enumerate(dl_tr):
            if time.time() - t0 > MAX_TRAIN_SECONDS:
                stopped = True
                break
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(dl_tr)} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        val = metrics(y_val, predict(dl_val))
        history.append({"epoch": epoch + 1, "steps_done": step + (0 if stopped else 1), "steps_per_epoch": len(dl_tr),
                        "val_macro_f1": val["macro_f1"], "elapsed_s": round(time.time() - t0), "stopped_by_budget": stopped})
        print(history[-1], flush=True)
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if stopped:
            break
    out["train_seconds_cpu"] = round(time.time() - t0)
    out["history"] = history
    model.load_state_dict(best_state)
    t1 = time.time()
    out["finetuned_test"] = metrics(y_te, predict(dl_te))
    out["test_inference_ms_per_doc_cpu"] = round((time.time() - t1) * 1000 / len(sub_te), 1)
    print("finetuned", out["finetuned_test"]["macro_f1"], flush=True)
    save_result("finetune_" + model_id.split("/")[-1], out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        baseline()
    else:
        main(sys.argv[2] if len(sys.argv) > 2 else "allenai/scibert_scivocab_uncased")
