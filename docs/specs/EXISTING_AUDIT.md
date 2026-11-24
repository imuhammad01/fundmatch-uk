# FundMatch UK — Existing-Work Audit (CS5079 Assessment 1)

Audit date: Tuesday 24 November 2026, ~01:15–02:30 (Europe/London, UTC+0).
Auditor role: Senior AI Researcher / Academic Supervisor (Grok Bot), for Md Shamim Mia (MSc AI, University of Aberdeen, ID 52536283).
Method: read-only inspection of `/workspace/cs5079-a1/` (nothing there was edited), plus live checks listed in `VERIFIED_SOURCES.md`.

Evidence labels used throughout:
- **[VERIFIED-FILE]** — read directly from a file in `/workspace/cs5079-a1/` today.
- **[VERIFIED-LIVE]** — checked live on the web/API today (24 Nov 2026).
- **[PROPOSED]** — a recommendation or target, not yet done or measured.

---

## 1. Official brief and marking scheme (from `/workspace/cs5079-a1/Final/`)

### 1.1 Files in `Final/`
| File | What it is | Notes |
|---|---|---|
| `Assessment 1 --Brief (2026-27).pdf` | 2-page "Microsoft Print to PDF" of the MyAberdeen assessment page (text is image-only; read by rendering the pages) | Title "Project and implementation, 50%". Links to `CS5079_Assessment_1_2026-2027.pdf`, `TeamAssessmentContribution.docx`, `GP-CGS-lookup-table.pdf`, `NCS_School_Marking_Conversion.pdf`. "Submission Guidelines" heading is present but **empty in the printout**. No deadline date appears in either PDF. **[VERIFIED-FILE]** |
| `CS5079_Assessment_1_2026-2027 (1).pdf` | The actual 4-page briefing ("CA1, Briefing Document") | Full text extracted. **[VERIFIED-FILE]** |
| `Product requirements.md` | FundMatch UK requirements (Idea A) | **Byte-identical** to `fundmatch-uk/00_product_requirements.md` (`diff` returned no differences). **[VERIFIED-FILE]** |
| `TeamAssessmentContribution.docx` | Team assessment profile form | "Marks awarded for the team work will be distributed to individuals based upon the agreed assessment profile"; must be completed electronically; per-member % per task + report writing. **Not yet filled in.** **[VERIFIED-FILE]** |
| `GP-CGS-lookup-table.pdf`, `NCS_School_Marking_Conversion.pdf` | Referenced, **not on the box** | Needed only to map marks → CGS grade. |

### 1.2 Assessment 1 overall
- Weight: **50%** of CS5079. Two parts: **Task 1 group (30 marks, ≤1,500 words)** and **Task 2 individual (20 marks, ≤1,200 words; FrozenLake-v1 6×6 DQN)**.
- Learning objectives: libraries for AI system creation; combining components end-to-end; validating effectiveness; documenting the approach; applying knowledge to a practical problem; reflecting and modifying design in response to validation.
- Submission: a **PDF report per task** (answers easy to match to questions) + **one Python notebook in a ZIP with outputs kept**. Group report submitted by one student. Any open-source code must be credited **in code and report** with modifications described (misconduct otherwise).
- Overall marking criteria (apply to both tasks): report quality (structure, clarity, English, **brevity**); **reproducibility**; **understanding**; quality of experiments (design + presentation); **in-depth analysis** (critical evaluation, conclusions); **source-code quality** (documentation, comments).

### 1.3 Task 1 — the 30 marks
| # | Brief item | Marks |
|---|---|---|
| 1a | Usefulness of the product clearly described | 2 |
| 1b | Product uses data | 1 |
| 1c | Trained on a large dataset; explain why the data is sufficient | 2 |
| 2a | Collect your own data from any source (no Kaggle / already-prepared data) | 2 |
| 2b | Write your own collection code | 3 |
| 3a | Pre-processing and feature engineering | 3 |
| 3b | Candidate ML/AI models and why they suit the project | 3 |
| 3c | Training process | 3 |
| 3d | Why the chosen model is suitable for deployment vs competitors | 6 |
| 4 | Deployment (e.g. Streamlit): process explained, showcased, **link provided** | 5 |
| | **Total** | **30** |

CGS A descriptor (target): "an excellent AI product, which could be made publicly available… all components working perfectly, with perfect justification… data collected is huge and diverse… **key AI techniques delivered in the course** were used… deployment is excellent, providing excellent functionalities… could be deployed in the real world."

---

## 2. What is already built and measured (`task1/`, `deploy_repo/`)

### 2.1 Data collection — [VERIFIED-FILE] `task1/data/raw/collection_log.json`, `collect_gtr.py`, `build_dataset.py`
| Endpoint | Records | Pages (100/page) | Time | Collected (UTC = London time today) |
|---|---|---|---|---|
| `/projects` | 158,710 | 1,588 | 1,911 s (~32 min) | 23 Nov 2026 19:35 |
| `/funds` | 158,710 | 1,588 | 1,663 s (~28 min) | 23 Nov 2026 19:31 |
| `/organisations` | 97,102 | 972 | 1,041 s (~17 min) | 23 Nov 2026 19:41 |

- Collector: own code, `Accept: application/vnd.rcuk.gtr.json-v7`, 100 records/page, ≥1.0 s between requests per endpoint (two threads ⇒ ≤2 req/s total), exponential back-off on 429/5xx, atomic page cache + resume, JSON log. Report says the run survived 26 transient errors.
- `build_dataset.py` joins **funds → projects via each fund's `rel="FUNDED"` link** (amount = `valuePounds.amount`, dates = fund `start`/`end`), and **organisations → projects via `LEAD_ORG`** (fallback: Innovate UK `LEAD_PARTICIPANT` name). Output `data/processed/gtr_projects.parquet` = 158,710 rows × 20 columns; every project matched a fund (`award_gbp` 0 nulls; `n_funds` > 1 for 0 projects).
- International evaluation sets (own code, `collect_international.py`): NSF 2,000 (2023 awards), NIH RePORTER 1,000 (FY2023), ARC 2,000 (Australia) = 5,000 raw; 3,852 mapped to UK councils for the cross-country test. PI/investigator names dropped at collection. ARC (CC BY-NC-ND) used for evaluation only.
- **New data-quality facts found in this audit (read-only profiling of the parquet):**
  - **47,416 projects (29.9%) have `award_gbp = 0`; 47,005 of them are `Studentship`** (EPSRC 18,490 of the zeros). Award modelling must exclude or separately model studentships (the notebook's award sets are 59,324 train / 14,925 test, consistent with such filtering).
  - `fund_start` spans 2006-01-01 → 2031-03-30; **56 projects start in 2027 or later** (future-dated records) — flag in EDA.
  - `department` missing for 54,088; `fund_end` missing for 17; `status` missing for 15.
  - `fund_category`: INCOME_ACTUAL 158,529; SUPPLEMENTARY_UNIT_AWARDS 114; EXPENDITURE_ACTUAL 67.
  - Lead-funder counts (raw): EPSRC 36,366; Innovate UK 33,170; BBSRC 16,490; MRC 13,325; ESRC 12,848; NERC 11,208; AHRC 10,795; STFC 7,336; then programme funders (Horizon Europe Guarantee 3,568; COVID 2,909; ISCF 2,530; GCRF 1,244; …).

### 2.2 Cleaning — [VERIFIED-FILE] `models/summary.json`
| Step | Rows left | Removed |
|---|---|---|
| Raw API projects | 158,710 | — |
| Keep 8 councils (drop programme funders) | 141,538 | 17,172 |
| Drop placeholder abstracts | 125,815 | 15,723 |
| Drop abstracts < 30 words | 117,986 | 7,829 |
| Drop boilerplate reused by ≥ 5 titles | 117,759 | 227 |
| Drop duplicates with conflicting funders | 117,719 | 40 |
| Drop exact duplicates (joint grants) | **113,233** | 4,486 |

Leakage controls: funder names masked in text (unmasked model 0.8663 vs masked 0.8615 macro-F1; on texts naming a funder 0.9289 vs 0.8875); `grantCategory`/grant reference excluded as features; group-aware split (109,104 groups). Split: **90,585 train / 22,648 test** (StratifiedGroupKFold). Smallest class STFC 5,511 projects (report).

### 2.3 Funder classifier results (held-out test, 8 classes) — [VERIFIED-FILE]
| Model | CV macro-F1 (±sd) | Test macro-F1 | Accuracy | Top-3 acc | Train s |
|---|---|---|---|---|---|
| Majority class | — | 0.0541 | 0.2761 | 0.606 | 0.01 |
| TF-IDF + ComplementNB (α 0.1) | 0.8078 ± 0.0010 | 0.8069 | 0.8088 | 0.9751 | 0.4 |
| **TF-IDF + LogReg (C 32) — deployed** | 0.8585 ± 0.0018 | **0.8615** | **0.8693** | **0.9916** | 80.3 |
| TF-IDF + LinearSVM (C 1) | 0.8599 ± 0.0012 | 0.8622 | 0.8702 | 0.9866 | 9.9 |
| MiniLM emb + LogReg (C 16) | 0.7833 ± 0.0010 | 0.7835 | 0.7840 | 0.9808 | 23.2 |
| BGE-small emb + LogReg (C 4) | 0.7926 ± 0.0013 | 0.7934 | 0.7946 | 0.9824 | 21.0 |
| BGE emb + MLP (1024) | 0.8187 ± 0.0018 | 0.8197 | 0.8252 | 0.9875 | 318.6 |
| Ensemble TF-IDF LR + BGE LR | 0.8562 ± 0.0005 | 0.8577 | 0.8644 | 0.9916 | 101.3 |

- Per-class F1 (deployed): AHRC 0.899, BBSRC 0.780, EPSRC 0.885, ESRC 0.823, Innovate UK 0.925, MRC 0.795, NERC 0.867, STFC 0.918.
- Learning curve (TF-IDF+LR test macro-F1): 1,807 → 0.7782; 4,525 → 0.8127; 9,054 → 0.8256; 22,643 → 0.8434; 45,290 → 0.8523; 90,585 → 0.8615.
- Temporal test (train < 2021, test ≥ 2021): TF-IDF+LR 0.827; BGE+LR 0.7611; Ensemble 0.8281.
- Calibration: raw ECE 0.0336 / Brier 0.1954 / log loss 0.3846 → temperature T = 1.1314: ECE **0.018** / Brier 0.1935 / log loss 0.3755 (accuracy unchanged 0.8693).
- Accuracy by calibrated confidence: (0, 0.4] 0.428; (0.4, 0.6] 0.517; (0.6, 0.8] 0.682; (0.8, 1.0] **0.943** (18,318 of 22,648 test cases = 80.9% fall in the top bin). App low-confidence threshold `LOW_CONF = 0.60`.
- Explanation faithfulness (deletion test): removing top-5 explanation words drops predicted-council probability by 0.221 and flips 26.5% of predictions vs 0.007 / 2.2% for 5 random words.
- Cross-country transfer (unchanged UK models): overall TF-IDF LR acc 0.671 / macro-F1 0.623 / top-3 0.949; BGE LR 0.680 / 0.617 / 0.959; Ensemble 0.697 / 0.649 / 0.962. By source (TF-IDF LR acc): NSF 0.753 (n 1,168), NIH 0.706 (n 940), ARC 0.596 (n 1,744).

### 2.4 Similar-project search — [VERIFIED-FILE]
| Retriever | P@10 same council | P@10 shares topic | Recall@10 (known item) | MRR@10 | Hit@10 topic | MRR@10 topic |
|---|---|---|---|---|---|---|
| **TF-IDF cosine (deployed)** | 0.7021 | 0.3502 | 0.895 | 0.7727 | 0.8713 | 0.5788 |
| MiniLM cosine | 0.6929 | 0.3458 | 0.900 | 0.7998 | 0.8673 | 0.5663 |
| BGE cosine | 0.6923 | 0.3302 | 0.8945 | 0.7979 | 0.8554 | 0.5518 |
| Hybrid RRF (TF-IDF + BGE) | 0.7112 | 0.3612 | **0.921** | **0.8203** | 0.8812 | 0.5936 |

Known-item protocol: 2,000 titles queried against 5,000 test abstracts. **The 50-query × top-3 manual relevance sheet (`models/manual_relevance_sheet.csv`, 150 rows) has 150/150 blank human judgements** — no human relevance evaluation exists yet.

### 2.5 Award range — [VERIFIED-FILE] (log-£ MAE on 14,925 test projects; 59,324 train)
| Method | MAE log | Median abs error £k | Coverage | Median width (hi/lo) |
|---|---|---|---|---|
| Baseline: training median | 1.12 | 186.3 | — | — |
| Ridge, TF-IDF text only (α 1) | 0.800 | 119.6 | — | — |
| Ridge, TF-IDF + metadata (α 3) | 0.534 | 74.6 | — | — |
| LightGBM, BGE text only | 0.885 | 124.2 | — | — |
| LightGBM, BGE + metadata, quantile P25–P75 | 0.462 | 65.2 | 0.422 | 1.803 |
| LightGBM, BGE + metadata, quantile P10–P90 | 0.462 | 65.2 | 0.687 | 3.400 |
| **30 TF-IDF neighbours, P25–P75 (deployed)** | 0.836 | 125.5 | **0.505** | 3.858 |
| 30 TF-IDF neighbours, P10–P90 (deployed) | 0.836 | 125.5 | 0.786 | 13.13 |
Earlier neighbour-band variants (summary `award`): council-wide oracle P25–P75 coverage 0.512 (width 5.444); 30 same-scheme TF-IDF neighbours P25–P75 0.520 (width 2.967).

### 2.6 Deployment selection — [VERIFIED-FILE] `models/deployment_selection.csv`
Pre-registered rule: peak RAM < 1 GB and p95 latency < 1 s, then best CV macro-F1 among probabilistic models, preferring the lighter within 0.01.
| Configuration | Search | CV F1 | Test F1 | Disk MB | Peak RAM MB | Cold load s | Median ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| **TF-IDF + LogReg** | TF-IDF | 0.858 | 0.862 | 116.5 | **369.6** | 2.81 | **56.6** | 68.3 |
| Ensemble TF-IDF LR + BGE LR | BGE | 0.856 | 0.858 | 218.5 | 789.9 | 3.76 | 189.6 | 205.8 |
| BGE + MLP | BGE | 0.819 | 0.820 | 208.9 | 756.4 | 3.84 | 191.8 | 198.6 |
| TF-IDF + ComplementNB | TF-IDF | 0.808 | 0.807 | 137.3 | 392.8 | 1.75 | 56.5 | 71.3 |
| BGE + LogReg | BGE | 0.793 | 0.793 | 204.1 | 756.5 | 3.88 | 190.2 | 216.0 |
| MiniLM + LogReg | MiniLM | 0.783 | 0.784 | 161.2 | 713.8 | 4.17 | 173.9 | 205.4 |

### 2.7 App (`deploy_repo/` == `task1/app/` code; `diff` identical) — [VERIFIED-FILE]
- `app.py` (Streamlit UI) + `fundmatch_core.py` (logic: input checks incl. empty/short/non-English/no-known-words, funder masking, temperature scaling, word-level explanations, filtered search, award range, Funding Finder deep links for 8 councils, Innovation Funding Service link).
- UI: example abstracts (sidebar), filters (councils, start-year slider, grant category, k 5–30), probability bar chart (Altair), low-confidence warning (< 60%), award band + trend chart, similar projects table + **CSV download**, "About, accuracy and limits" expander with OGL v3.0 attribution.
- `requirements.txt` pinned: streamlit 1.67.0, numpy 2.5.5, pandas 3.0.6, pyarrow 26.0.1, scipy 1.18.1, scikit-learn 1.9.1, altair 6.3.0 (no PyTorch). Python 3.13 required (pickles).
- Assets: `tfidf_lr.pkl` 14.4 MB, `projects_meta.parquet` 13.5 MB, TF-IDF index in 8 shards (12.3–20.4 MB each; deploy repo), `model_info.json`.
- Tests: **28 pytest** (17 `test_core.py` + 11 `test_app_e2e.py`); PROGRESS log reports 28/28 pass in task1, clean venv and extracted zip. Local server VmHWM **651 MiB**; warm queries 0.18–0.43 s.
- Notebook: `FundMatch_Task1.ipynb` 61 cells (36 code), **0 error outputs**; identical to `FundMatch_executed.ipynb`.
- ZIPs: lean `CS5079_Task1_FundMatch_code.zip` 53 files (2.6 MB); full with assets 58 files (158.5 MB).
- `deploy_repo/` is **not a git repository** (no `.git`); PROGRESS says "GitHub repo still 404". **There is no live Streamlit URL yet.**

### 2.8 Report — [VERIFIED-FILE] `report/CS5079_Task1_Group_Report.pdf`
6 pages; header word count 1,419 (headings, body, captions, tables; excl. title, code images, references); PROGRESS notes 1,461 incl. title and 1,477 by `pdftotext` before References. Sections map 1:1 to the brief (1 definition, 2 collection, 3 creation 3.1–3.5, 4 deployment, 5 critical evaluation + roadmap). **Placeholders still present:** `[Member 2–4 name (ID)]`, `[APP LINK]`, `[GITHUB LINK]`, `Generative AI use: [state the group's use here…]`.

---

## 3. Strengths (evidence-backed)
1. Whole-population collection with own polite, resumable code (158,710 projects + funds + 97,102 orgs) and a correct funds/organisations join — directly hits 2a/2b and the CGS-A "huge and diverse" descriptor.
2. Rigorous protocol: leakage masking with a measured effect, group-aware split, CV-only tuning, a test set used once, learning curve, temporal test, calibration, cross-country OOD test, faithfulness test. This exceeds a typical MSc coursework baseline.
3. Honest, pre-registered deployment rule with measured RAM/latency — exactly what the 6-mark item rewards.
4. Engineering quality: core/UI split, 28 automated tests, pinned deps, sharded assets for GitHub limits, privacy-by-design (no person records, no user text stored).

## 4. Gaps and grading risks (ranked by marks at stake)
| # | Risk | Marks exposed | Evidence | Fix |
|---|---|---|---|---|
| R1 | **No public deployment link / no GitHub repo** | up to 5 (item 4) and credibility of 6 (item 3d) | `[APP LINK]` in report; no `.git`; PROGRESS "GitHub repo still 404" | Push `deploy_repo/` to a public GitHub repo, deploy on Streamlit Community Cloud (Python 3.13), screenshot the live app, rebuild the report with `--app-url/--github-url` |
| R2 | Memory headroom on free tier | 3d / 4 | VmHWM 651 MiB vs Streamlit documented minimum 690 MB (limits "as of February 2024", may change) **[VERIFIED-LIVE]** | Keep `prune_index.py 100` ready (halves index RAM, 79% top-10 unchanged); consider float32→float16 CSR data, `HashingVectorizer`; optionally apply for Streamlit "good-for-the-world" increased resources |
| R3 | Placeholders (members, AI-use statement) and missing contribution form | Report quality; individual mark distribution | Report text; blank docx | Fill names/IDs, write a truthful Generative-AI statement per course policy, complete `TeamAssessmentContribution.docx` with the group |
| R4 | Human relevance never rated | 3b/3d "understanding / in-depth analysis" | 150/150 blank | 2–3 group members rate the 150 rows; report Cohen's κ and P@3 (human) next to proxy P@10 |
| R5 | Requirements deviations (no fine-tuned SciBERT, no SPECTER/FAISS) | 3b | SCOPE_DECISIONS D2, D4, D19 | Already justified; optionally add a small SPECTER2 or SciBERT **frozen-embedding** row on a 20k subsample to close the comparison honestly |
| R6 | Word cap pressure | Report quality / brevity | 1,419–1,477 words depending on counting | Do not add prose; put new results in tables/figures; state the counting rule |
| R7 | Temporal drift (0.862 → 0.827) and lead-funder-only labels | Analysis | summary.json | Already discussed; propose multi-label + yearly retrain in roadmap |
| R8 | Award amounts not inflation-adjusted; 29.9% zero awards (studentships) | Analysis of FR-14/FR-22 | Parquet profile | Deflate with ONS CPIH (series L522) to constant 2025 £; state studentship handling explicitly |
| R9 | Course alignment could be made more visible | CGS A "key AI techniques delivered in the course" | Report cites methods but not the lecture mapping | One table row/caption tying models to CS5079 lectures (TF-IDF/LR/SVM L5–L7, frozen sentence embeddings L6, LightGBM L8, CV/tuning/interpretation/Streamlit L4, feature engineering L2); optionally a LIME cross-check (L4) on 20 examples |
| R10 | Brief's deadline/submission guidelines not on box | Process | Empty "Submission Guidelines" in printout | Confirm date and portal on MyAberdeen |

## 5. Live verification that changes the record (24 Nov 2026) — details in VERIFIED_SOURCES.md
- GtR API v7 still reports **158,710 projects / 158,710 funds / 97,102 organisations** (identical to the 23 Nov collection and the 20 Nov test call).
- `s` must be **10–100**: `s=2` → HTTP 400 "Page size cannot be less than 10"; `s=101` → HTTP 400 "Page size cannot be greater than 100".
- Project records carry **no amount and `start`/`end` = null** (10/10 sampled); each project has a **`rel="FUND"` link** to `/funds/{id}`; each fund carries `valuePounds {currencyCode: GBP, amount}`, `start`/`end` (epoch ms), `category` (e.g. `INCOME_ACTUAL`), a **`FUNDED`** link back to the project and a **`FUNDER`** link to the funder organisation. This **resolves Note #2** of the requirements and confirms `build_dataset.py`'s join.
- Innovate UK (and some programme) projects additionally expose `participantValues.participant[]` with `organisationName`, `role`, `projectCost`, `grantOffer` — a second award signal the current build does not use.
- GtR API docs moved: `https://gtr.ukri.org/resources/api.html` → HTTP 308 → `https://gtr.ukri.org/find-research-and-innovation/getting-started-with-the-gtr-api` (rate limit stated: 10 requests/second per user, "[Frequency to be confirmed]").
