# Red-team autopsy: ScholarPulse UK / FundMatch UK

Date: 25 November 2026 (Europe/London, UTC+0). Branch: `shamim` of `mueezd/applied_ai_a_1`, after the pass-1 audit (commits `88eadce` to `9468588`).

**Method.** We assumed a hostile marker and tried to break every claim. Each finding was re-checked by running code on the box, not by reading the docs:
- the original scikit-learn pickle from the coursework workspace
- the 23 Nov 2026 raw GtR pull (158,710 projects)
- the shipped engine and app
- the live site <https://scholarpulse-uk.streamlit.app/>, loaded in a headless browser

Each finding lists:
- what was wrong
- the evidence, with the script and result file in `training/redteam/`
- the fix, or the risk that is left

Severity:
- **Critical:** a headline claim is false, or the product gives wrong answers.
- **High:** a user or marker would be misled.
- **Medium:** a requirement is weakened.
- **Low:** cosmetic.
- **Info:** checked and found sound.

## Scorecard

| ID | Severity | Finding | Status |
|---|---|---|---|
| RT-01 | Critical | The deployed vectoriser did not match training: stop words were not removed before bigrams | Fixed |
| RT-02 | High | The "bit-exact" parity test was circular, and the `max_logit_diff <= 1e-6` claim was false | Fixed |
| RT-03 | High | Padding passed the 30-word check (`"!!! "`×30, `"12345 "`×40, one word ×30) and was scored with high confidence | Fixed |
| RT-04 | High | The live app is deployed from `imuhammad01/fundmatch-uk` v4.0.0, so none of the fixes here are live | **Open: needs the owner's approval** |
| RT-05 | Medium | Person names appeared in public titles and sole-trader organisation names (NFR-9) | Fixed for honorific patterns; residual risk remains |
| RT-06 | Medium | Live UI: award tiles cut off (`£191,…`), studentships shown as an award of `0`, and bands built from very few priced neighbours with no warning | Fixed on `shamim` |
| RT-07 | Medium | The packaged `preprocess.py` did not reproduce the notebook funnel: different placeholder pattern, dedup key and kept copy | Fixed; reproduces all 7 funnel rows exactly |
| RT-08 | Medium | Stale v3 and FundMatch-era claims in the README, checklist, model card and report §4–5 | Fixed |
| RT-09 | Medium | `requirements-training.txt` was missing `lightgbm`, which notebook 7.3 needs | Fixed; pins checked against the working venv |
| RT-10 | Medium | FR-8 recency rule was not applied | Applied to award bands; effect measured |
| RT-11 | Medium | FR-13: no FAISS, no SPECTER, and no judgements on the 50 queries | FAISS and SPECTER measured; judgements drafted, pending human review |
| RT-12 | Medium | FR-12: no fine-tuned transformer | CPU pilot measured: SciBERT loses to TF-IDF + LR on the same rows |
| RT-13 | Low | Near-duplicate leakage across the split | Measured: negligible |
| RT-14 | Low | Time leakage from a random split | Measured: small |
| RT-15 | Low | Gibberish that contains a few real words is still scored, with only a warning | Residual risk |
| RT-16 | Low | 4 projects end before they start; 34.4% of shipped rows have a £0 award | Documented; £0 is excluded from bands |
| RT-17 | Low | `artifacts/tfidf_lr_meta.json` lists the wrong `stop_words` (7 single letters) | Documented; the engine no longer reads it |
| RT-18 | Info | Confidence thresholds against calibration | Sound |
| RT-19 | Info | GtR links, grant references and the fund-to-project join | Sound |
| RT-21 | Medium | Load-time memory peak goes above the 690 MB free-tier minimum | Reduced: engine peak 733 → 632 MiB, server 819 → 691 MiB |
| RT-20 | Info | OGL attribution, no secrets, no Kaggle data, pinned versions install, notebook reruns | Sound (see below) |

Totals: 1 Critical, 3 High, 9 Medium, 5 Low, 3 Info (21 findings).

## Findings

### RT-01 (Critical): the deployed model was not the trained model
- **What.** Training used `TfidfVectorizer(stop_words="english")`, so "carbon and soil" became the bigram "carbon soil". The v4 NumPy engine took its stop-word list from `tfidf_lr_meta.json`. That list held only seven single letters (`e, g, h, …`), and the token pattern already removes single letters, so in practice no stop words were removed. Bigrams such as "and soil" are not in the vocabulary, while "carbon soil" was never formed.
- **Evidence.**
  - Comparison against the original pickle (`deploy_repo/assets/tfidf_lr.pkl`) on 3,070 texts: 10 demos, 2,000 titles and 1,060 abstracts.
  - Before the fix, the largest logit gap was **3.325** and the top council agreed on **97.85%** of texts (97.93% on abstracts alone).
  - So about 1 query in 47 got a different council from the model whose test metrics are reported.
- **Fix.**
  - `text.STOP_WORDS` is now scikit-learn's `ENGLISH_STOP_WORDS`, and `engine._tokens` removes those words before n-grams, exactly as training does.
  - After the fix, the largest logit gap is **1.0e-5** and top-council agreement is **100%** on all 3,070 texts.
  - `tests/test_red_team.py::test_rt01_*` keeps 100 masked texts and their original-pickle logits in `tests/fixtures/sklearn_parity.json` as a regression guard. The fixture has no names; texts are demos, titles and abstracts.
- **Knock-on.** The conformal quantiles and the novelty reference had been built with the broken vectoriser, so we rebuilt them (`results/assets_rebuild.json`):
  - q90 went from 0.99845 to 0.99811.
  - The median novelty cosine went from 0.8081 to 0.8075.
  - The 2D map does not depend on the vectoriser and came out byte-for-byte identical.
  - Int8 fidelity was re-measured: 0.98 top-1 agreement on 50 texts and 0.991 on 2,000 titles, with the largest probability gap at 0.039.

### RT-02 (High): the test that should have caught RT-01 tested the bug
- `test_numpy_matches_sklearn` built its scikit-learn reference from the same wrong meta file, so it compared the bug with itself.
- It now uses `stop_words="english"`. The independent fixture test from RT-01 checks against the real pickle.
- The model card claim of ≤1e-6 has been corrected.

### RT-03 (High): the minimum-length check could be gamed
- **Before.** The check counted whitespace tokens. All of these passed and were scored, for example Innovate UK at 90% with high confidence:
  - `"!!! "`×30
  - `"12345 "`×40
  - `"robot "`×30
  - four words repeated
- **Fix.** `check_input` now counts only alphabetic words of two or more letters. It needs at least 30 such words, at least 10 different words, and at least one word in the 200,000-term vocabulary.
- **Tests:**
  - padding cases
  - the exact boundary: 29 distinct words fail, 30 pass, and numbers or symbols do not top up the count
  - the zero-vocabulary case
- **Residual (RT-15).** Made-up words mixed with a few real research words are still scored. The non-English warning is shown, and the low-confidence tier usually applies.

### RT-04 (High, open): the fixes are not on the live site
- The live app (<https://scholarpulse-uk.streamlit.app/>, screenshot 25 Nov 2026) is deployed from `imuhammad01/fundmatch-uk`, not from this branch. It still shows:
  - the RT-01 model
  - the RT-06 UI defects
  - unredacted names
- **Fix needed (owner's decision).** Either merge `shamim` into that repo, or point the Streamlit app at `mueezd/applied_ai_a_1`, branch `shamim`, folder `fundmatch-uk`. We did not push to that repo because it was not authorised.

### RT-05 (Medium): personal names (NFR-9)
- No person endpoints were called, and the table has no person columns; a test enforces this.
- However, about 58 or more public **titles** carry an honorific plus a name, for example fellowship titles, and some lead organisations are sole traders named after a person.
- **Fix.** `fundmatch.privacy` redacts these on load, so every view, CSV and the API share the same redacted table:
  - "Dr A B Name" style titles become "[name]".
  - Sole-trader organisations become "Sole trader (name withheld)".
  - "X T/A Y" becomes "Y".
- On the shipped table this changes **83 titles and 12 organisation names**. Tests cover titles, organisations and the engine table.
- **Residual.** A name with no honorific, for example "The Smith lab", is not caught. GtR URLs still lead to public GtR pages, which show the investigators.

### RT-06 (Medium): live UI defects (fixed on `shamim`)
- **Award tiles.** They showed "£191,…" because the full figure did not fit. They now use compact £k and £m values, with the full value in the help text.
- **£0 awards.** The similar-projects table showed `0` for studentships. The "award (£)" column now shows "£0 · not priced" for these rows, with a caption explaining that they are left out of the band. A local screenshot after the first fix showed `None`, so the column was changed to text.
- **Thin bands.** For the quantum demo, the band used only **8 priced neighbours out of 30**. A warning now shows when fewer than 10 neighbours have a priced award.

### RT-07 (Medium): two cleaning pipelines that disagreed
- The packaged `preprocess.py` (used by tests and the CLI) differed from notebook Section 3:
  - a narrower placeholder pattern
  - deduplication on raw text with the first copy kept, where the notebook keeps the largest award
  - boilerplate detection on a 400-character prefix
- It now matches the notebook. `training/redteam/rt_check_funnel.py` runs it on the raw 158,710-row pull and reproduces every funnel row: 158,710 → 141,538 → 125,815 → 117,986 → 117,759 → 117,719 → **113,233** (`results/funnel_check.json`, `match: true`).

### RT-08 (Medium): stale claims
- README: "90 ms, 2.72 s, 568 MB" were v3 numbers.
- Checklist: "21 passed".
- Model card: "≤1e-6" and "0.978 on 45 texts" (the set actually has 50 texts).
- Report §4: described the v3 FundMatch app (`fundmatch_core.py`, battery screenshot, "28 tests", "651 MiB").
- Report §5: "awards ignore inflation", although v4 offers CPIH.
- All were replaced with numbers measured in this pass. Report §4 now describes the live ScholarPulse UK app, with a screenshot taken on 25 Nov 2026.

### RT-09 (Medium): the training environment could not be rebuilt
- `lightgbm` was used in notebook 7.3 but was not pinned.
- `requirements-training.txt` now pins `lightgbm==4.7.0`, `faiss-cpu==1.16.0` and `rank-bm25==0.2.2`.
- Every pin was checked against `pip freeze` of the environment that ran the notebook. The app's `requirements.txt` pins were checked against a clean venv that passes the test suite.

### RT-10 (Medium): FR-8 recency rule
- 1,224 of 113,233 shipped projects start in 2026 or later, and their awards may not be fully recorded yet.
- **Measured** (`results/leakage_time_recency.json`, 5,000 test queries, 30 TF-IDF neighbours from training):

| Award band | P25–P75 coverage (ideal 0.50) | Median band width (P75/P25) |
|---|---|---|
| Nominal, all neighbours (v4) | 0.518 | 4.07 |
| Nominal, recency rule | 0.518 | 4.03 |
| CPIH real, all neighbours | 0.519 | 4.03 |
| CPIH real, recency rule | 0.520 | 4.00 |

- The rule removes 1.6% of the priced neighbour pool and changes coverage by under 0.1 points.
- **Decision.** Apply it to award bands (`RECENT_START_YEAR = 2026` in `engine._award`, with a note in the band text). It costs nothing and follows FR-8.
- For **training**, the rule is still not applied (D6). The 233 test projects that start in 2026 or later score a macro-F1 of 0.921, so they do not hurt the reported metric, and the time-based test (macro-F1 0.827) is the stricter check.

### RT-11 (Medium): FR-13 search
- **Measured** in `training/redteam/rt_retrieval.py` and `results/retrieval_fr13.json` on the notebook's test split.
- **Known-item search** uses the same 2,000 title queries against 5,000 test abstracts as notebook 7.2.
- **Topic search** uses 500 test queries against 5,000 training projects; a hit shares a GtR research topic. Only 33% of those projects carry topics, so absolute values are low.
- The TF-IDF, MiniLM and BGE known-item figures reproduce the notebook (0.895, 0.900 and 0.8945), which shows the split and sampling are the same.

| Retriever | Known-item Recall@10 | Known-item MRR@10 | Topic Hit@10 | Topic P@10 | CPU encode speed |
|---|---|---|---|---|---|
| TF-IDF cosine (deployed lexical part) | 0.895 | 0.773 | **0.854** | 0.244 | n/a |
| BM25 (Okapi) | **0.916** | **0.843** | 0.824 | **0.246** | n/a |
| MiniLM (all-MiniLM-L6-v2) | 0.900 | 0.800 | 0.798 | 0.216 | 45 texts/s |
| BGE (bge-small-en-v1.5) | 0.8945 | 0.798 | 0.776 | 0.220 | 80 texts/s |
| SPECTER (allenai-specter) | 0.758 | 0.578 | 0.796 | 0.220 | 20 texts/s |

- **SPECTER** was trained on title + abstract pairs for citation prediction. On this task it is the weakest at finding a project from its title, and no better than MiniLM on topics, while being 4 times slower than BGE. It does not earn a place.
- **BM25** is the best lexical baseline for known-item search: +2.1 points Recall@10 over TF-IDF. TF-IDF stays ahead on topic Hit@10.
- **FAISS** was benchmarked on the full 113,233 × 384 BGE matrix with 500 queries:
  - Exact `IndexFlatIP` reproduces NumPy exact search (recall 1.000).
  - HNSW (M = 32, efSearch = 64) keeps 99.8% of the exact top 10. It answers a single query in a median 0.16 ms, against 2.7 ms per query for batched NumPy.
  - The HNSW index takes 195 MB, against 166 MB for Flat, and builds in 7 s.
- **Decision.** The deployed engine keeps exact search over a 64-d int8 SVD index, about 29 MB in RAM, because it is already well under NFR-2. A 195 MB FAISS index plus BGE and PyTorch would not fit the 690 MB minimum tier. FAISS is the documented route if the corpus grows by about 10 times.
- **50 queries judged by hand.** `training/models/manual_relevance_sheet.csv` now has an assistant-drafted 0–2 relevance column, judged from titles only and **labelled as not human-judged**. The draft gives:
  - P@3 (relevant ≥1): 0.847
  - share rated highly relevant: 0.267
  - queries with at least one relevant result in the top 3: 98%
  - queries with a highly relevant result in the top 3: 60%
  - Results are in `results/manual_relevance_draft_summary.json`.
  - The `relevant (group)` column is still empty. **Residual:** the group must review and confirm the labels before the report cites them as human judgements.

### RT-12 (Medium): FR-12 fine-tuned transformer
- **Measured** with `training/redteam/rt_finetune.py`, results in `results/finetune_*.json`. It was a bounded CPU pilot: 8 threads and a 480 s training budget.
- **Data.** SciBERT (`allenai/scibert_scivocab_uncased`, 109.9M parameters) was fine-tuned on 3,000 stratified training projects. 500 more training projects were held out to pick the checkpoint, and it was tested on 1,500 stratified test projects (seed 42).
- **Settings.** Inputs were the first 128 word-pieces, with funder names masked; class-weighted loss, AdamW, learning rate 3e-5, batch 16.
- **How far it got.** It completed 1 full epoch plus 18 of 188 steps of the second epoch before the budget stopped it. Validation macro-F1 was 0.733, then 0.804.

| Model (same 1,500 test projects) | Macro-F1 | Accuracy | Top-3 |
|---|---|---|---|
| SciBERT fine-tuned, 3,000 rows, about 1.1 epochs, CPU | 0.784 | 0.779 | 0.981 |
| TF-IDF + LR, same 3,000 rows (4 s to train) | 0.825 | 0.833 | 0.988 |
| TF-IDF + LR, full 90,585-row training split (137 s) | **0.885** | 0.890 | 0.993 |

- **Reading.** With equal data, the linear model beats the CPU-budget transformer by 4 points of macro-F1. SciBERT also needs 23.6 ms per document for inference, plus about 440 MB of weights and PyTorch.
- This does not prove that a full GPU fine-tune would lose. It does show that the deployed model is not an under-powered choice on this hardware.
- **Residual.** A full fine-tune on all 90k rows needs a GPU. It is on the report roadmap.

### RT-21 (Medium): memory peak at load (NFR-3)
- Measured on the box with other jobs paused (`results/engine_benchmark_quiet.json`):
  - Engine alone: steady RSS 552 MiB, load peak (VmHWM) 733 MiB. The v4.0.0 code peaked at 710 MiB, so redaction adds about 23 MiB.
  - Streamlit server after one browser session on the matcher tab: steady 616 MiB, peak 819 MiB.
  - Cold load: 5.0 s.
  - Warm query over 30 demo runs: median 135 ms, p95 230 ms.
- The steady state is under the 690 MB minimum, but the load peak is above it. The live app runs, so the host currently gives more than the minimum (up to 2.7 GB).
- **Fix (after the owner approved redeploying).** The engine now builds its lexical TF-IDF index before loading the weights, so the vectoriser's temporary n-gram counts no longer stack on top of them. It also dequantises the SVD matrices in place, keeping one copy instead of three temporaries.
  - Engine alone: load peak 733 → **632 MiB**, steady 544 MiB.
  - Streamlit server after one session: peak 819 → **691 MiB**, steady 600 MiB.
  - Results are identical: the demo still scores EPSRC 0.8795, and all tests pass.
- **Residual.** The server peak is now about at the 690 MB minimum, and the host currently allows more.

### RT-13 (Low): near-duplicate leakage
- For every one of the 22,648 test projects, we found its most similar training project by TF-IDF cosine (`results/leakage_time_recency.json`).
- 33 test projects (0.15%) have a training twin at cosine ≥0.95; 63 have one at ≥0.90 and 281 at ≥0.70.
- Removing all twins at ≥0.70 moves macro-F1 from 0.8620 to 0.8612.
- The median nearest-training cosine is 0.19. The group split works.

### RT-14 (Low): time leakage
- The main split is random by group, so 36.5% of test projects start in 2021 or later. That mixes past and future.
- Macro-F1 is 0.866 on test projects that start before 2021 and 0.855 on those from 2021 on.
- The honest forward-looking figure is the separate time-based test: train before 2021, test 2021 and later, macro-F1 **0.827**. The report states both.

### RT-16 (Low): data oddities
- 4 projects have an end date before their start date.
- 47,416 raw projects (29.9%) and 38,984 shipped projects (34.4%) have a £0 award, and 38,856 of the shipped £0 rows are studentships.
- Every one of the 158,710 projects joined exactly one fund record (`n_funds = 1`).
- £0 rows are excluded from award percentiles, and the count excluded is shown to the user.

### RT-18 (Info): confidence thresholds
- Low confidence is below 0.60 and high is 0.80 or above.
- On the held-out test set, accuracy is 0.517 in the 0.4–0.6 bin, 0.682 in 0.6–0.8 and 0.943 in 0.8–1.0.
- The thresholds sit where accuracy crosses about two-thirds and above 90%. They are consistent.

### RT-19 (Info): links and joins
- Ten random shipped rows were checked against the live GtR API on 25 Nov 2026. All 10 matched on title, grant reference and lead funder.
- Grant references are unique in the shipped table; a test checks this.
- The `gtr.ukri.org/projects?ref=` page returns HTTP 202, because it is a single-page app, and renders in a browser.

### RT-20 (Info): everything else that held up
- OGL v3.0 attribution appears in the About panel, the hub and brief text, and the README.
- No Kaggle or prepared dataset was used: our own collector pulled every record.
- A secret scan passes.
- `pytest` passes, with the count in the commit notes.
- **Notebook rerun (not completed).** A full top-to-bottom re-execution of `training/FundMatch_Task1.ipynb` was started on the box. It was stopped after about 85 minutes, before it finished, so that the bounded red-team jobs could run. The committed notebook is the pass-1 execution, with every code cell output and no errors.
  - Independent reruns reproduce its key numbers exactly: the split (90,585 / 22,648), the deployed LR test macro-F1 (0.8620, accuracy 0.8696, top-3 0.9915), the cleaning funnel, and the known-item search figures.
  - **Residual:** the group should run `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 training/FundMatch_Task1.ipynb` once before submission (about 2 h on 8 CPUs). The audit cited a `training/run_notebook.sh` that does not exist; that reference has been corrected.

## ELI10: what went wrong, in plain words

Imagine a robot librarian that reads your project idea and guesses which of eight funding offices would pay for it.

1. **The robot read differently at work than at school (RT-01).** At school, it learned by first crossing out little words like "and" and "the". At work, nobody told it to cross them out, so about 1 idea in 47 went to a different office than its school results promised. We taught it to cross them out again. Now it gives exactly the school answer every time, and a test checks this forever.
2. **You could trick it with nonsense (RT-03).** It only counted words, so "!!! !!! !!!" thirty times counted as thirty words. Now it only counts real words, and it wants at least ten different ones.
3. **It sometimes said people's names (RT-05).** Some project titles contain "Dr So-and-so". It now hides those names.
4. **The shop window was wrong (RT-06, RT-08).** Some prices were cut off ("£191,…"), studentships looked like they cost £0, and old leaflets described an older robot. All of that is fixed and the leaflets are rewritten.
5. **Checks that came out fine.** Did it peek at the answers? Hardly: only 0.15% of test ideas had a near-copy in its training books, and removing them changes nothing. Do the links work? 10 out of 10 did.
6. **Still to do.** The robot in the real shop window (the live website) is still the old one, because it lives in a different repository. Its owner has to press "update". A person also needs to check the 50 example searches we pre-marked.
