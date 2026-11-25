# Training and evaluation: how the deployed model was built

This folder holds the CS5079 Task 1 work behind the ScholarPulse UK / FundMatch UK app: data collection, cleaning, model comparison, evaluation and export. The app in the parent folder only loads the exported weights in `../artifacts/`. It never retrains or calls the API.

| Path | What it is | Requirements |
|---|---|---|
| `collect_gtr.py` | Our own collector for the UKRI Gateway to Research API v7 (`/projects`, `/funds`, `/organisations`). It pages 100 records at a time, waits at least 1 s between requests on each endpoint, retries with exponential back-off, and keeps a resumable gzipped page cache. | FR-1, FR-3, FR-4 |
| `build_dataset.py` | Joins `/funds` (award and dates, through the `FUNDED` link) and `/organisations` (lead organisation, through `LEAD_ORG`) to the projects. Writes `data/processed/gtr_projects.parquet`: 158,710 rows × 20 columns. | FR-2 |
| `collect_international.py` | Small NSF, NIH and ARC test sets for the cross-country check. Used for evaluation only, never for training. Investigator names are dropped. | evaluation |
| `FundMatch_Task1.ipynb` | **The executed notebook, with outputs kept** (36 code cells, no errors). It covers EDA, cleaning, leakage checks, the grouped stratified split and a time-based check. It then compares 7 models and an ensemble, with CV tuning, a learning curve, calibration, the cross-country test, Recall@10 and MRR retrieval metrics, Ridge against LightGBM for awards, error analysis, the deployment benchmark and the export. | FR-6 to FR-18, NFR-1 to NFR-3, NFR-5 |
| `notebook_src.py`, `py2nb.py` | Percent-format source of the notebook and the converter. Run `python py2nb.py notebook_src.py FundMatch_Task1.ipynb`. | NFR-6 |
| `logs/collection_log.json` | Dated record of the collection: 158,710 projects at 2026-11-23T19:35:16Z, 158,710 funds and 97,102 organisations. | FR-4 |
| `logs/collect_*stdout.log` | Console logs showing real HTTP 503 retries. | FR-3 |
| `models/summary.json` | Every result table in the notebook, as JSON. | FR-11 to FR-17 |
| `models/deployment_selection.csv` | Accuracy, size, memory and latency for each candidate configuration. | FR-17 |
| `models/manual_relevance_sheet.csv` | 50 queries × top results for human relevance rating. The `relevant (group)` column is still to be filled in by the group. | FR-13 |
| `figures/` | Learning curve, confusion matrix, reliability diagram, deployment trade-off and EDA plots. | FR-7, FR-11, FR-12, FR-17 |
| `SCOPE_DECISIONS.md` | Decisions D1 to D19, where the build differs from the requirements, and why. Examples: no fine-tuned SciBERT, no FAISS, and a time-based test instead of a too-recent filter. | all |
| `report/` | Group report (PDF and DOCX, 1,423 words) and the script that builds it from `models/summary.json`. | NFR-14 |

## Data policy (FR-5, NFR-9, NFR-10)

Every record was fetched by this code from public APIs. **No Kaggle data, GtR bulk export or GtR-Extended dataset was used**. These are mentioned only for comparison. GtR person records were never requested. GtR data is © UKRI and licensed under the Open Government Licence v3.0.

## Reproduce

Python 3.13 was used. A full run took about 1.5 hours on an 8-core CPU with no GPU, plus about 1 hour of polite collection.

```bash
cd fundmatch-uk/training
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-training.txt
python collect_gtr.py --endpoint both            # projects + funds, about 30 min each (two threads, ~1 request/s each)
python collect_gtr.py --endpoint organisations   # about 17 min
python collect_international.py                  # optional cross-country test sets, about 3 min
python build_dataset.py                          # -> data/processed/gtr_projects.parquet
jupyter nbconvert --to notebook --execute --inplace FundMatch_Task1.ipynb
```

- Seeds are fixed at `SEED = 42`.
- Sentence embeddings are cached in chunks under `data/processed/emb_parts/`, so a stopped run resumes.
- Raw pages (about 340 MB) and the processed parquet (270 MB) are not committed, because of GitHub's 100 MB file limit. The collector rebuilds them.
- To check the collector without a full run: `python -m fundmatch.pipeline.collect_gtr projects /tmp/gtr/projects --max-pages 2` from `fundmatch-uk/` with `src` on the path.
- On 25 Nov 2026 the API still reported 158,710 projects.

## Headline results (test set of 22,648 projects; from `models/summary.json`)

| Model | Macro-F1 | Top-3 accuracy |
|---|---|---|
| Majority class | 0.054 | 0.606 |
| TF-IDF + Complement NB | 0.807 | 0.975 |
| **TF-IDF + Logistic Regression (deployed)** | **0.8615** | **0.9916** |
| TF-IDF + Linear SVM | 0.862 | 0.987 |
| MiniLM embeddings + LogReg | 0.784 | 0.981 |
| BGE embeddings + LogReg | 0.793 | 0.982 |
| BGE embeddings + MLP | 0.820 | 0.988 |
| Ensemble TF-IDF LR + BGE LR | 0.858 | 0.992 |

**Why TF-IDF + Logistic Regression was deployed:**
- **Linear SVM:** level on macro-F1, but it gives no probabilities, and the app needs calibrated ones.
- **Ensemble:** it needs PyTorch plus BGE, with about 790 MB peak RAM and 190 ms per query. TF-IDF + LR needs 370 MB and 57 ms, and the ensemble scores 0.004 lower macro-F1.
- **Fine-tuned SciBERT:** not run. It needs a GPU, and its weights plus PyTorch exceed the free tier (decisions D4 and D19).

## Red-team checks (`redteam/`, pass 2, 25 Nov 2026)

These scripts re-derive the notebook split (Sections 3–4) and test the claims in `../docs/RED_TEAM_AUTOPSY.md`. Each writes a JSON file in `redteam/results/`. Set `FM_RAW_PARQUET` to the raw pull (`data/processed/gtr_projects.parquet`) and `FM_RT_CACHE` to a scratch folder for the cached split.

| Script | What it checks | Result file |
|---|---|---|
| `rt_check_funnel.py` | The packaged `preprocess.py` reproduces the published cleaning funnel | `funnel_check.json` |
| `rt_leakage_and_recency.py` | Near-duplicate leakage, the time mix of the random split, and the FR-8 recency rule's effect on award-band coverage | `leakage_time_recency.json` |
| `rt_retrieval.py` | FR-13: TF-IDF, BM25, MiniLM, BGE and SPECTER on known-item and topic search, plus FAISS Flat and HNSW on the full 113k BGE matrix | `retrieval_fr13.json` |
| `rt_finetune.py` | FR-12: a small CPU fine-tune of SciBERT against TF-IDF + LR on the same rows | `finetune_*.json` |
| `rt_rebuild_assets.py` | Rebuilds the conformal and novelty assets after the RT-01 fix, and re-measures int8 fidelity | `assets_rebuild.json` |
| `test_bm25_equivalence.py` | The vectorised BM25 equals `rank_bm25` | (assert) |

`models/manual_relevance_sheet.csv` now has a column `relevance (assistant draft, 0-2, NOT human-judged)`. These are drafts made from titles only, waiting for the group's review. The `relevant (group)` column is still empty. Its summary is in `redteam/results/manual_relevance_draft_summary.json`.
