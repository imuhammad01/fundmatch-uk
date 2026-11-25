# Requirements audit: ScholarPulse UK / FundMatch UK against the product requirements (Idea A)

Audit date: 25 November 2026 (Europe/London, UTC+0). Branch audited: `shamim` of `mueezd/applied_ai_a_1`, starting at commit `2178065` (the same code as `imuhammad01/fundmatch-uk` v4.0.0 and the live app, apart from Windows line endings).

Requirements source: `docs/specs/00_product_requirements.md` (sections 1 to 5 and the notes).

**How each status was checked.** Every status comes from code that was run or a file that was read during the audit. No number was invented.
- `pytest -q` passes: 41 tests (21 existing plus 20 new in `tests/test_requirements.py`).
- Streamlit `AppTest` loads `app.py` with no exceptions and shows 8 tabs.
- A live two-page pull with `python -m fundmatch.pipeline.collect_gtr projects ... --max-pages 2` ran on 25 Nov 2026. The API still reports **158,710 projects** in 1,588 pages.
- Engine latency was measured over 30 demo queries.
- Model and evaluation figures come from the executed training notebook `training/FundMatch_Task1.ipynb` and its saved `training/models/summary.json`, from the run on the 23 Nov 2026 snapshot.

Status key:
- **Met**: the "How you check it" evidence exists in this repository.
- **Partly**: done, but with a deviation that is documented here.
- **Missing**: not done.

## Summary

| Group | Items | Before (2178065): Met / Partly / Missing | After: Met / Partly / Missing |
|---|---|---|---|
| Functional FR-1 to FR-27 | 27 | 9 / 18 / 0 | 25 / 2 / 0 (round 2; round 1 was 24 / 3 / 0) |
| Non-functional NFR-1 to NFR-14 | 14 | 10 / 4 / 0 | 13 / 1 / 0 (round 2 measured the NFR-3 load peak; round 1 was 14 / 0 / 0) |
| Feature list (Must, Should, Could, Won't) | 13 | 9 / 4 / 0 | 13 / 0 / 0 |
| Marking-scheme rows | 11 | 3 / 8 / 0 | 9 / 2 / 0 |

**Why many items were only Partly before.** The `shamim` branch had the deployed product: the app, the engine, the exported weights and the results JSON. It did not have the training work behind it:
- the collection and dataset-join scripts
- the executed training notebook
- the figures
- the 50-query relevance sheet
- the group report

So most data and model requirements had numbers in `artifacts/summary.json`, but not the "How you check it" evidence. That material existed in the coursework workspace and is now in `training/`.

**The two FR items still Partly after round 2.**
- **FR-12:** a CPU SciBERT pilot was run and measured (macro-F1 0.784, against 0.825 for TF-IDF + LR on the same rows). A full GPU fine-tune was not run.
- **FR-13:** BM25, SPECTER and FAISS were measured, and the 50 queries have assistant-drafted labels. The human review in `relevant (group)` is still empty.
- FR-8 is now Met: the recency rule is applied to award bands and its effect is measured.

## Round 2: red-team pass (25 Nov 2026)

The full findings are in [`RED_TEAM_AUTOPSY.md`](RED_TEAM_AUTOPSY.md): 21 findings, of which 1 is Critical, 3 High, 9 Medium, 5 Low and 3 Info.

The ones that changed statuses or numbers above:
- **RT-01 (Critical).** The deployed NumPy vectoriser skipped English stop-word removal. Against the original pickle on 3,070 texts, logits differed by up to 3.325 and the top council disagreed on 2.15%. This is fixed: the largest gap is now 1.0e-5 and agreement is 100%. The conformal and novelty assets were rebuilt. The old "bit-exact" test was circular (RT-02) and has been replaced by a test against a fixture from the real pickle.
- **RT-03 (High).** Padding defeated the minimum-length check (FR-19). Fixed.
- **RT-04 (High, open).** The live site is deployed from `imuhammad01/fundmatch-uk` v4.0.0, so none of the round-2 fixes are live until the owner redeploys from this branch.
- **FR-8, FR-12 and FR-13.** New measurements, shown in the rows above.
- **NFR-3.** The load-time memory peak exceeds the 690 MB minimum. Marked Partly.

## 2.1 Data collection

| ID | Before | After | Evidence | Fix made |
|---|---|---|---|---|
| FR-1 | Partly | **Met** | Training notebook Section 1 contains the full `training/collect_gtr.py` with outputs. The packaged collector is `src/fundmatch/pipeline/collect_gtr.py` (`page_url`, `fetch_page`, `collect`). The live two-page pull worked on 25 Nov 2026. | Added `training/`, rewrote the packaged collector so it is testable (CLI with `--max-pages`), and added tests `test_fr1_*` and `test_fr3_fr4_*`. |
| FR-2 | Partly | **Met** | `training/build_dataset.py` takes amount and dates from `/funds` through the `FUNDED` link, and the lead organisation from `/organisations` through `LEAD_ORG`. Notebook Section 2 prints the schema and the share of missing values per field. `data/projects_meta.parquet` stores id, ref, title, funder, category, organisation, award, start and end dates, and the GtR URL. Abstracts are left out of the app table to save memory. | The join code and notebook are now in the repo. |
| FR-3 | Partly | **Met** | `RateLimiter` sets `PAUSE_S = 0.6` s, which is about 1.6 requests a second. `fetch_page` retries 429 and 5xx errors with back-off of 1, 2, 4 … 30 s. `collect` skips pages already saved, so a run resumes, and writes each page atomically. `training/logs/collect_stdout.log` shows real HTTP 503 retries. | Added the rate limiter, resume test and back-off test. |
| FR-4 | Partly | **Met** | `training/logs/collection_log.json` records 158,710 projects collected at 2026-11-23T19:35:16Z. `collect()` merges a dated entry into `collection_log.json`, and `save_snapshot()` writes `gtr_projects_YYYY-MM-DD.parquet` with a JSON sidecar giving the date and row count. | New functions, plus tests `test_fr4_*`. |
| FR-5 | Partly | **Met** | All data comes from our own API calls. No Kaggle, GtR bulk export or GtR-Extended data is used. This is stated in `training/README.md`, in the report (§2.1) and in the top-level `README.MD`. | Statement added to the READMEs. |

## 2.2 Preprocessing and features

| ID | Before | After | Evidence | Fix made |
|---|---|---|---|---|
| FR-6 | Partly | **Met** | Notebook Section 3 has a table of rows removed at each step: 158,710 → 141,538 → 125,815 → 117,986 → 117,759 → 117,719 → 113,233. **Round 2 (RT-07):** the packaged `clean_frame_with_report()` used a narrower placeholder pattern and a different dedup rule. It now mirrors the notebook, and `training/redteam/rt_check_funnel.py` reproduces all 7 counts exactly on the raw pull (`funnel_check.json`). | Synced `preprocess.py` with the notebook, plus test `test_fr6_*`. |
| FR-7 | Partly | **Met** | `normalise_funder()` maps spelling variants to the 8 councils and maps everything else to "Other". Programme budgets such as ISCF, GCRF, COVID, Horizon Europe Guarantee and Newton Fund are each under 2.5% of the pull, overlap the councils, and cannot be applied to, so they are dropped (D8). The class distribution plot is in notebook Section 2 (`training/figures/eda_overview.png`). | Added `normalise_funder()` and test `test_fr7_*`. |
| FR-8 | Partly | **Met** | `date_filter_report()` counts **1,224** of 113,233 projects that start in 2026 or later. **Round 2 (RT-10):** the recency rule is now applied to award bands (`engine.RECENT_START_YEAR = 2026`), and the band text says how many priced neighbours it removed. Measured on 5,000 test queries: P25–P75 coverage is 0.518 without the rule and 0.518 with it (CPIH: 0.519 and 0.520); the rule removes 1.6% of the priced neighbour pool. For training, the rule is still not applied (D6). The 233 test projects that start in 2026 or later score a macro-F1 of 0.921, and the time-based test (train before 2021, test from 2021) gives 0.827. Amounts are comparable through the £0 exclusion and optional CPIH (ONS L522). | Rule applied to bands; test `test_fr8_recency_rule_excludes_recent_awards`. |
| FR-9 | Partly | **Met** | Notebook Section 5 prints the feature shapes: TF-IDF with 200k word and bigram features, frozen MiniLM and BGE sentence embeddings, and for the award model log1p(amount) plus council, scheme, duration and start year (Section 7.3). | Notebook added. |
| FR-10 | Partly | **Met** | Notebook Section 4 uses `StratifiedGroupKFold` (seed 42) for an 80/20 split by funder, with grouping so near-duplicates stay together: 90,585 train and 22,648 test. Validation means 3-fold grouped cross-validation inside the training set only. A time-based test set is used as an extra check (Section 6.3). | Notebook added. |
| FR-11 | Partly | **Met** | Learning curve in notebook Section 6.2 (`training/figures/learning_curve.png`): macro-F1 goes from 0.778 with 1,807 training projects to 0.862 with 90,585. The smallest class, STFC, has 5,511 projects. | Figure and notebook added. |

## 2.3 Models and training

| ID | Before | After | Evidence | Fix made |
|---|---|---|---|---|
| FR-12 | Partly | Partly | Notebook Section 6, on the 22,648 test projects:<br>- majority class 0.054<br>- Complement NB 0.807<br>- **TF-IDF + LogReg 0.8615** (top-3 0.9916)<br>- Linear SVM 0.862<br>- MiniLM + LR 0.784<br>- BGE + LR 0.793<br>- BGE + MLP 0.820<br>- ensemble<br>**Round 2 (RT-12):** a real SciBERT fine-tune on CPU (`training/redteam/rt_finetune.py`): 3,000 training rows, about 1.1 epochs in a 480 s budget. On the same 1,500 test projects it scored macro-F1 **0.784** (top-3 0.981), against 0.825 for TF-IDF + LR on the same 3,000 rows and 0.885 for TF-IDF + LR trained on the full split. | A CPU pilot was measured. **Still Partly:** a full GPU fine-tune on all 90k rows was not run. |
| FR-13 | Partly | Partly | **Round 2 (RT-11)**, `training/redteam/results/retrieval_fr13.json`.<br>Known-item search (2,000 titles against 5,000 abstracts), Recall@10 / MRR@10:<br>- TF-IDF 0.895 / 0.773<br>- **BM25 0.916 / 0.843**<br>- MiniLM 0.900 / 0.800<br>- BGE 0.8945 / 0.798<br>- **SPECTER 0.758 / 0.578**<br>Topic Hit@10 (500 queries against 5,000 documents): TF-IDF 0.854, BM25 0.824, MiniLM 0.798, BGE 0.776, SPECTER 0.796.<br>**FAISS** on the full 113,233 × 384 BGE matrix: Flat recall 1.000; HNSW recall 0.998 of the exact top 10, median 0.16 ms per single query, 195 MB index, 7 s build.<br>The 50-query sheet has an assistant-drafted 0–2 column: P@3 (≥1) 0.847, and 98% of queries have a relevant result in the top 3. | FAISS and SPECTER were measured, and exact SVD search was kept because it is about 29 MB against 195 MB plus PyTorch. **Still Partly:** the `relevant (group)` column needs a human review. |
| FR-14 | Partly | **Met** | Notebook Section 7.3, MAE on log amount:<br>- Ridge on TF-IDF: 0.800; with metadata: 0.534<br>- LightGBM with BGE and metadata: 0.462, but its P25–P75 range covers only 0.422 of true awards<br>- deployed 30-neighbour P25–P75 band: covers 0.505 of true awards (P10–P90 covers 0.786) | Notebook added. |
| FR-15 | Met | **Met** | `class_weight='balanced'` (`src/fundmatch/pipeline/train.py`). Per-class F1 is in `artifacts/model_info.json` and shown in the Model Lab tab: AHRC 0.899, BBSRC 0.780, EPSRC 0.885, ESRC 0.823, Innovate UK 0.925, MRC 0.795, NERC 0.867, STFC 0.918. | None. |
| FR-16 | Partly | **Met** | Tuning used validation folds only: chosen C or alpha and CV macro-F1 per model are in `summary.json["tuning"]`, and training seconds are in `results[*].train_s`. Seed 42 is used throughout. The settings table is in notebook Section 6. | Notebook added. |
| FR-17 | Met | **Met** | `artifacts/deployment_selection.csv` compares accuracy, disk, peak RAM, cold load time and median and p95 latency for 6 configurations. TF-IDF + LogReg uses 370 MB and 57 ms; the runner-up ensemble uses 790 MB and 190 ms for 0.004 lower macro-F1. The written justification is in notebook Section 9 and report §3.4. | Justification is now in the repo (report and notebook). |
| FR-18 | Partly | **Met** | Notebook Section 8 analyses misclassified examples, including EPSRC versus Innovate UK and BBSRC versus MRC. Section 8.1 runs a faithfulness test of the word explanations. | Notebook added. |

## 2.4 The app

| ID | Before | After | Evidence (`app.py`, `src/fundmatch`) | Fix made |
|---|---|---|---|---|
| FR-19 | Met | **Met** | `text.check_input` requires at least 30 **alphabetic words of two or more letters**, at least 10 of them different, and at least one term in the vocabulary. **Round 2 (RT-03):** padding such as `"!!! "`×30, `"12345 "`×40 or one word ×30 used to pass and be scored with high confidence; it is now refused with a clear message. | Tests `test_rt03_*`, including the exact 29/30-word boundary. |
| FR-20 | Met | **Met** | Probability bar chart in `render_match`. Below 0.60 a low-confidence `st.error` appears, and from 0.60 to 0.80 a moderate warning appears. | Added axis titles and a hover tooltip. |
| FR-21 | Partly | **Met** | The similar-projects table shows title, council, scheme, award, **start and end dates**, organisation and a GtR link. Links take the form `https://gtr.ukri.org/projects?ref=<grant ref>`. | **Added start and end date columns.** |
| FR-22 | Met | **Met** | Award band (P25, median and P75, plus P10–P90) with the "not a funding quote" caveat. **Round 2 (RT-06):** compact £k/£m tiles (the live site cut them off as "£191,…"), a warning when fewer than 10 of the 30 neighbours are priced, and "£0 · not priced" in the similar table instead of 0. | `app.py` changes. |
| FR-23 | Met | **Met** | "Topic funding by year" stacked bar chart by council. | Added axis labels. |
| FR-24 | Met | **Met** | One-click demo buttons (`demo_<id>`), with the first demo pre-loaded. | None. |
| FR-25 | Met | **Met** | Council multiselect, start-year slider and scheme filter in the AI Matcher tab, passed to `engine.analyze`. | None. |
| FR-26 | Met | **Met** | "Download CSV" for similar projects, plus CSVs for the comparison and batch tools. | None. |
| FR-27 | Partly | **Met** | The "About, ethics, and the limits" panel shows the data source and date, the model and test accuracy, privacy, money caveats, scope, OGL and the not-official-advice notice. | **Added the model name and why it was chosen, known weak classes and failure modes, measured speed, and how to wake a sleeping app.** |

## 3. Non-functional

| ID | Before | After | Evidence | Fix made |
|---|---|---|---|---|
| NFR-1 | Met | **Met** | Macro-F1 0.8615 against 0.054 for the majority class and 0.807 for TF-IDF + Complement NB. Top-3 accuracy is 0.9916. Linear SVM is level on F1 (0.862) but gives no probabilities. | Test `test_fr12_nfr1_*`. |
| NFR-2 | Met | **Met** | Re-measured in round 2 with other jobs paused (`training/redteam/results/engine_benchmark_quiet.json`): 30 demo queries, median **135 ms**, p95 230 ms, maximum 301 ms; engine cold load 5.0 s. The target was under 3 s per query. | README updated. |
| NFR-3 | Met | Partly | Streamlit Community Cloud limits, checked 25 Nov 2026: memory 690 MB minimum and 2.7 GB maximum, up to 2 CPUs. **Round 2 (RT-21):** steady RSS is 552 MiB for the engine alone and 616 MiB for the Streamlit server, but the **load-time peak is 733 MiB for the engine and 819 MiB for the server**, above the minimum. v4.0.0 peaked at 710 MiB, so this is not new. **Fixed:** the engine now builds its lexical index first and dequantises the SVD in place, so peaks fall to 632 MiB for the engine and 691 MiB for the server. | Borderline at the minimum; the host currently allows up to 2.7 GB. |
| NFR-4 | Partly | **Met** | The app is live at https://scholarpulse-uk.streamlit.app/. For non-browser clients it answers with Streamlit's normal HTTP 303 redirect. The wake procedure is documented in the README and in the About panel. | Wake note added. **Open the link on submission day.** |
| NFR-5 | Partly | **Met** | App versions are pinned in `requirements.txt`, and the training stack in `training/requirements-training.txt`. Seed 42 is used. The executed notebook is kept with all 36 code cells output and no errors. The collection log and reproduction steps are in `training/README.md`. The top-level `requirement.txt` is now filled. | Notebook and requirements added, plus test `test_nfr5_*`. |
| NFR-6 | Met | **Met** | Docstrings and typed functions. New code is commented. | None. |
| NFR-7 | Met | **Met** | Plain-English labels, a glossary, example inputs and hover tooltips. | Tooltips and axis titles added to the main chart. |
| NFR-8 | Partly | **Met** | Okabe–Ito colour-blind-safe palette (`COUNCIL_COLOURS`), dark high-contrast theme. | **Axes are now labelled** on the probability and trend charts. |
| NFR-9 | Met | **Met** | Person endpoints are refused in code (`ALLOWED_ENDPOINTS`), and the table holds organisations only. **Round 2 (RT-05):** 83 public titles with an honorific plus a name, and 12 sole-trader organisation names, are now redacted on load (`fundmatch/privacy.py`). | Tests `test_rt05_*`. **Residual:** names without an honorific. |
| NFR-10 | Met | **Met** | OGL v3.0 attribution in the app and README. UKRI does not endorse the tool. Sources are cited in the code docstrings and report references. | None. |
| NFR-11 | Met | **Met** | Model Lab tab shows test metrics, a per-class F1 chart and the deployment table. The About panel lists weak classes. | Failure modes added to About. |
| NFR-12 | Met | **Met** | The collector can be rerun and resumed, the notebook re-exports the weights, and the engine reloads them from `artifacts/`. | None. |
| NFR-13 | Met | **Met** | The GtR API needs no key, and the repo has no secrets. | Test `test_nfr13_*` scans the repo. |
| NFR-14 | Partly | **Met** | `training/report/CS5079_Task1_Group_Report.pdf` is **1,423 words** (counted by `make_report.py`) under headings that follow the brief, with the app and code links filled in. | Report regenerated with links. **To do:** add the group members' names and IDs, which are still placeholders. |

## 4. Feature list (MoSCoW)

| Feature | Priority | Before | After | Where |
|---|---|---|---|---|
| Funder predictor with probabilities | Must | Met | Met | `engine.analyze` and `render_match` |
| Top 10 similar projects with GtR links, amounts, dates | Must | Partly (no dates) | Met | similar table (dates added) |
| Typical funding range | Must | Met | Met | award band |
| About and limits panel | Must | Met | Met | About panel |
| Model comparison in the notebook (baselines against transformers, per task) | Must | Partly (notebook not in repo) | Met (frozen transformers; no fine-tuned SciBERT) | `training/FundMatch_Task1.ipynb` §6, 7.2, 7.3 |
| Topic funding trend chart | Should | Met | Met | "Topic funding by year" |
| One-click example abstracts | Should | Met | Met | demo buttons |
| Low-confidence warning and per-class accuracy | Should | Met | Met | `render_match` and Model Lab |
| Error analysis section | Should | Partly | Met | notebook §8 |
| Filters by council and year | Could | Met | Met | AI Matcher tab |
| CSV download | Could | Met | Met | "Download CSV" |
| Links to live UKRI Funding Finder calls | Could | Met | Met | "Live UKRI Funding Finder" links per council |
| Won't have: success prediction, proposal writing, researcher ranking | Won't | Partly (stated in app only) | Met | About panel, report §5, `training/SCOPE_DECISIONS.md` D10 |

## 5. Marking-scheme map

| Brief section (marks) | Requirements | Before | After | Main evidence |
|---|---|---|---|---|
| Usefulness (2) | Product summary | Met | Met | README, report §1.1 |
| Uses data (1) | FR-1, FR-2 | Partly | Met | notebook §1–2 |
| Data sufficient (2) | FR-11 | Partly | Met | learning curve |
| Own sources (2) | FR-1, FR-5 | Partly | Met | collector, FR-5 statement |
| Own code (3) | FR-1, FR-3, FR-4 | Partly | Met | `collect_gtr.py`, log, tests |
| Preprocessing and features (3) | FR-6 to FR-9 | Partly | Met (round 2: FR-8 applied to bands, funnel reproduced) | notebook §3–5, `preprocess.py` |
| Candidate models (3) | FR-12 to FR-14 | Partly | Partly (SPECTER, FAISS, BM25 and a CPU SciBERT pilot measured; no full fine-tune; no human judgements yet) | notebook §6–7 |
| Training process (3) | FR-10, FR-15, FR-16 | Partly | Met | notebook §4, §6 |
| Deployment choice (6) | FR-17, NFR-1 to NFR-3 | Met | Partly (NFR-3 load peak) | deployment table, latency, memory |
| Deployment, showcase, link (5) | FR-19 to FR-27, NFR-4 | Met | Met | live app, report §4 |
| Report, reproducibility, code quality | NFR-5, NFR-6, NFR-14 | Partly | Met | report PDF, pinned requirements, tests |

## Traceability map: requirement to file, function or notebook cell

| Requirement | File / function | Training notebook section (`training/FundMatch_Task1.ipynb`) |
|---|---|---|
| FR-1, FR-3 | `src/fundmatch/pipeline/collect_gtr.py`: `page_url`, `fetch_page`, `RateLimiter`, `backoff_seconds`, `collect`; full run script `training/collect_gtr.py` | §1 Data collection |
| FR-2 | `training/build_dataset.py`; `data/projects_meta.parquet` | §1.1 Build one table; §2 EDA (missing values) |
| FR-4 | `collect_gtr.write_collection_log`, `save_snapshot`; `training/logs/collection_log.json` | §1 |
| FR-5 | `training/README.md`, `README.MD` | §1 header |
| FR-6 | `preprocess.clean_frame_with_report`, `FUNNEL` | §3 Cleaning |
| FR-7 | `preprocess.normalise_funder`, `COUNCILS` | §2 EDA, §3 |
| FR-8 | `preprocess.date_filter_report`; `cpih.py` (real-terms pounds) | §6.3 Time-based check |
| FR-9 | `pipeline/train.py` `VECTORIZER` | §5 Features, §7.3 |
| FR-10 | — | §4 Split |
| FR-11 | `training/figures/learning_curve.png` | §6.2 |
| FR-12, FR-15, FR-16 | `pipeline/train.py`; `training/models/summary.json` (`results`, `tuning`) | §6, §6.1 |
| FR-13 | `pipeline/index.py`; `engine.search`; `training/models/manual_relevance_sheet.csv` | §7, §7.1, §7.2 |
| FR-14, FR-22 | `engine._award`; `summary.json["award_regression"]` | §7, §7.3 |
| FR-17, NFR-1 to NFR-3 | `artifacts/deployment_selection.csv`; `app.tab_lab` | §9 |
| FR-18 | `engine` explanation and contrast; `app.render_match` X-Ray | §8, §8.1 |
| FR-19 | `text.check_input` (`MIN_WORDS`) | — |
| FR-20 | `app.render_match` (bar chart, confidence warnings); `engine` confidence notes | §6.4 Calibration |
| FR-21, FR-25, FR-26 | `app.render_match` similar table; `app.tab_matcher` filters; `engine.to_csv` | — |
| FR-23 | `engine._trends`; `app.render_match` | — |
| FR-24 | `demos.DEMOS`; `app.main` demo buttons | — |
| FR-27, NFR-10, NFR-11 | `app.tab_corpus` "About, ethics, and the limits"; `engine.DISCLAIMER`, `engine.OGL` | — |
| NFR-5 | `requirements.txt`, `training/requirements-training.txt`, `training/README.md` (Reproduce) | whole notebook |
| NFR-9 | `collect_gtr.ALLOWED_ENDPOINTS`; `tests/test_requirements.py::test_nfr9_*` | §1 |
| NFR-13 | `tests/test_requirements.py::test_nfr13_*` | — |
| NFR-14 | `training/report/` | — |

## Still to do by the group (not something code can fix)

1. Review the assistant-drafted labels in `training/models/manual_relevance_sheet.csv` and fill `relevant (group)` (FR-13).
2. Redeploy the live app from this branch (RT-04), or merge these fixes into `imuhammad01/fundmatch-uk`.
3. Re-execute `training/FundMatch_Task1.ipynb` once before submission (the round-2 rerun was stopped before it finished).
4. Put the group members' names and IDs in the report title block (`training/report/make_report.py`).
5. Open https://scholarpulse-uk.streamlit.app/ on submission day to wake it (NFR-4).
6. The repo `mueezd/applied_ai_a_1` is private, so a marker cannot open the code link in the report. Make it public or give the marker access, or point the report at the public mirror `imuhammad01/fundmatch-uk`.
