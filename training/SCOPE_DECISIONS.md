# FundMatch UK — scope decisions (Task 1)

The product idea from `Product requirements.md` is kept: title + abstract in → likely UKRI council,
similar funded projects, typical award range. The scope is trimmed so that every part can be justified
inside 1,500 words and the 30-mark scheme.

| # | Decision | Why |
|---|---|---|
| D1 | **One main model with the full comparison: the funder classifier** (8 classes: AHRC, BBSRC, EPSRC, ESRC, Innovate UK, MRC, NERC, STFC). | The brief's 3+3+6 marks (candidates, training, deployment choice) need depth on one model, not shallow coverage of three. |
| D2 | **Similar-project search re-uses the classifier's sentence embeddings**, evaluated lightly (P@10 same council / shared GtR research topic) against TF-IDF and MiniLM. No FAISS, BM25 or SPECTER. | Exact numpy search over ~1e5 x 384 float16 vectors is fast enough and needs no extra library; SPECTER (768-d) doubles memory. |
| D3 | **Funding range = 25th–75th percentile of the neighbours' awards**, evaluated by coverage of the true award vs. a council-wide baseline. Ridge/LightGBM regression dropped in v1; **v3: now run** (notebook 7.3): Ridge on TF-IDF MAE 0.800 log £ vs 0.836 for the neighbour median; LightGBM + metadata 0.462 but needs inputs users rarely know and its P25–P75 quantiles cover only 42% (neighbour band 50.5%), so the app keeps the neighbour band. | Gives the user a useful range with zero extra models; avoids a third modelling section. |
| D4 | **No fine-tuned SciBERT.** Transformers are used as frozen sentence encoders (all-MiniLM-L6-v2, bge-small-en-v1.5) + linear/MLP heads. | Fine-tuning a 110M model on CPU is slow and the model + torch may not fit Streamlit's free memory; frozen small encoders are a defensible trade-off and are measured. |
| D5 | **FR-2 fixed:** award amounts and dates come from `/funds` (joined on the project id in each fund's FUNDED link); lead organisation names from `/organisations` (joined on LEAD_ORG link), not from project records. | Project records have no amount/dates; verified on the live API 23 Nov 2026. |
| D6 | **FR-8 changed:** no "too recent" filter (we do not predict outcomes). Instead a *time-based check*: train < 2021, test ≥ 2021. | The original rule had no reason in this product; a temporal test is a real robustness check. |
| D7 | **Leakage controls added:** funder names/KTP masked in text (and the effect measured), `grantCategory` and grant reference never used as features, placeholder/boilerplate abstracts removed, exact duplicates (joint grants) collapsed, group-aware split (same title or abstract opening ⇒ same group). | These would otherwise inflate scores. |
| D8 | **Programme funders dropped** (ISCF, GCRF, COVID, Horizon Europe Guarantee, Newton, UKRI FLF…). | They are budgets, not councils a user can apply to; tiny and overlapping classes. |
| D9 | **Kept "should/could" features that are cheap:** example abstracts, low-confidence warning, per-class accuracy in About, council/year filters, CSV download, trend chart of similar projects, link to UKRI Funding finder. | They help the "excellent functionalities" band without new models. |
| D10 | **Won't have:** predicting success, writing proposals, ranking researchers (no personal data shown). | Ethics and scope. |
| D11 | Collection uses the whole API (all pages of `/projects` and `/funds`, ~158.7k each) rather than a sample. | "Data collected is huge and diverse" is the CGS A descriptor; at ~1 request/s each pull is ~30–50 min. |

## Second pass (v2) — added only where an experiment showed a measurable benefit
| # | Decision | Evidence / reason |
|---|---|---|
| D12 | **Calibrated probabilities**: one temperature T fitted on out-of-fold training probabilities, applied in the app. | Lower test ECE / Brier / log loss with identical accuracy (notebook 6.4). |
| D13 | **Exact word-level explanations** ("Why this council?") from tfidf × coefficient of the deployed linear model. | Deletion test: removing the 5 top words drops the predicted-council probability far more than removing 5 random words (notebook 8.1). |
| D14 | **Cross-country test** on our own collected NSF + NIH (USA) and ARC (Australia) abstracts mapped to UK councils; evaluation only, not training. | Measures out-of-distribution robustness (notebook 6.5). ARC data is CC BY-NC-ND, so it is used for evaluation only and not shipped with the app. |
| D15 | **Hybrid TF-IDF + BGE search (RRF)** evaluated but **not deployed** unless it clearly beats TF-IDF within the memory limit. | Notebook 7.1 vs Section 9 memory figures. |
| D16 | **Not collected:** CORDIS (EU) and NSERC/CIHR (Canada) are distributed mainly as bulk files rather than a paged API, and EU/Canadian programme structures do not map cleanly onto UK councils → roadmap. | Time and mapping validity. |
| D17 | App refactored: `app/fundmatch_core.py` (logic, unit tested) + `app/app.py` (UI only); input checks for empty, short, non-English and no-known-words text. | Maintainability and robustness (tests/). |
| D18 | **v3: FR-13 retrieval metrics added** (notebook 7.2): known-item Recall@10/MRR@10 (2,000 titles vs 5,000 test abstracts) and topic Hit@10/MRR@10; exact cosine search kept instead of FAISS (113k docs, ~50 ms per query). A 50-query sheet with automatic proxy checks is exported for the group's own human rating; no human judgements are claimed. | Closes FR-13 honestly. |
| D19 | **v3: fine-tuned SciBERT still not run** (GPU needed; ~440 MB weights + PyTorch exceed the free tier). No SciBERT score is reported; the notebook explains why frozen MiniLM/BGE encoders stand in for the transformer route. | FR-12 deviation documented. |
