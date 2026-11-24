# FundMatch UK — Master Research, Architecture & A1/A+ Strategy

Prepared 24 Nov 2026 (Europe/London) for Md Shamim Mia (MSc AI, University of Aberdeen, 52536283).
Companion files: `00_product_requirements.md` (verbatim brief from Shamim), `EXISTING_AUDIT.md` (what exists, with numbers), `VERIFIED_SOURCES.md` (every link and person checked today).

**Evidence labels:** **[V-FILE]** measured/present in `/workspace/cs5079-a1/` · **[V-LIVE]** checked on the web/API on 24 Nov 2026 · **[PROPOSED]** recommendation or target, not done · **[ASSUMPTION]** reasoned but unverified.

**Two tracks, kept separate everywhere in this document**
- **Track A — CS5079 Assessment 1 submission (v1, frozen scope).** The existing Task 1 build is already near CGS-A quality. Only low-risk, high-mark actions are recommended (deploy, fill placeholders, human relevance rating, small course-alignment additions). Do **not** re-architect Track A before submission.
- **Track B — FundMatch UK v2 product (post-submission).** The research-grade, publicly deployable product and possible paper. Built on a new branch/repo so the marked submission stays reproducible.

---

## Section 1 — Project research, competitor analysis, literature review, research gaps

### 1.1 Purpose, problem, users
- **Problem statement (for the report and proposal):** UK applicants must choose among seven research councils plus Innovate UK, whose remits overlap (the deployed model's main confusions are BBSRC↔MRC, EPSRC↔Innovate UK, AHRC↔ESRC [V-FILE]). Evidence about what has already been funded, and for how much, sits in Gateway to Research as keyword search over 158,710 records [V-LIVE] — not in a form that answers "where does my idea fit, what is similar, what size is typical?"
- **Users:** early-career researchers and PhD applicants; university research-development offices; SMEs applying to Innovate UK; (v2) funders' portfolio analysts and policy researchers.
- **Objectives (measurable):** (O1) council recommendation with calibrated probabilities, top-3 accuracy ≥ 0.99 on held-out UK data [V-FILE: 0.9916]; (O2) evidence retrieval of similar funded projects, Recall@10 ≥ 0.89 known-item [V-FILE: 0.895]; (O3) a typical award band whose nominal 50% band covers ≈ 50% of true awards [V-FILE: 0.505]; (O4) faithful explanations [V-FILE: deletion test 0.221 vs 0.007]; (O5) runs free and public with < 1 s latency [V-FILE: 57 ms median benchmark].
- **Explicit non-goals:** predicting funding success; writing proposals; ranking or profiling people (NFR-9, D10).

### 1.2 Competitor analysis (verified facts in VERIFIED_SOURCES §C)
| Tool | Access | Strength | Weakness for this need | FundMatch UK difference |
|---|---|---|---|---|
| UKRI GtR | Free, API | Authoritative, complete UKRI award record, OGL | Keyword search; no abstract-in prediction, no award band | Reads an abstract; returns council + evidence + band |
| UKRI Funding Finder | Free | Live open calls by council | No matching to the user's text | App deep-links each predicted council to its open calls (already built) |
| NIH RePORTER Matchmaker | Free | Abstract-in matching | US NIH only | UK multi-council |
| Dimensions | Free personal non-commercial version; paid platform | Grants–publications–patents graph | Grant analytics in paid tier; not an applicant recommender | Open, free, UK-specific, calibrated |
| Pivot-RP / Research Professional (Clarivate) | Institutional subscription | Global opportunity DB, alerts, editorial profiles | Paywalled; profile-based (personal data) | No personal data; past-award evidence |
| GrantForward / Instrumentl | Subscription / trial | Large opportunity databases | US federal/foundation focus | UK research councils |
| OpenAlex / Semantic Scholar | Free (OpenAlex key since Feb 2026) | Open scholarly graph, APIs | Papers, not funder fit | Uses them only as optional enrichment |
| CORDIS / EU F&T portal | Free | EU projects and calls | EU programmes | UK; EU as future extension |
| GtR-Extended (2026 research dataset) | Open (Zenodo) | First unified UK funding-lifecycle DB | Dataset, not an applicant tool; forbidden as Track A training data (FR-5) | Applicant-facing ML product built from own collection |

Honest market claim: "In a desk check of public pages on 24 Nov 2026 we did not find a free tool that takes a UK abstract and returns council probabilities, similar funded UKRI projects and a typical award range together." Not "no tool exists".

### 1.3 Critical literature review (themes, what each implies for design)
1. **Scientific text representation.** SciBERT (Beltagy et al., 2019) domain-pretrains BERT on scientific text; SPECTER (Cohan et al., 2020) learns document embeddings from citation triplets; SciRepEval/SPECTER2 (Singh et al., 2023) shows task-format-specific adapters matter; Sentence-BERT (Reimers & Gurevych, 2019) makes sentence embeddings efficient. *Implication:* frozen general encoders (MiniLM, BGE) underperformed TF-IDF on this task (0.78–0.79 vs 0.86 macro-F1 [V-FILE]) — consistent with council labels depending on discriminative vocabulary; a scientific encoder (SPECTER2 classification adapter) is the principled next comparison, not a guaranteed gain.
2. **Grant/abstract classification.** Tattershall & Yang (2017, STFC tech report) classified GtR abstracts into a two-level ontology (≈80% precision, 34 categories); Goh et al. (2020, Scientometrics) found SVM beats trained human classifiers at assigning ERC abstracts to panels; Coute et al. (2025) compare embedding+LR with LLM few-/zero-shot on NIH grants (few-shot best on a 1,505-grant task). *Implication:* linear models on lexical features are strong, well-evidenced baselines; LLM few-shot is a credible v2 comparator for low-data labels (e.g. new programmes) but costs latency/privacy.
3. **Grant recommendation.** Zhu et al. (2023, PLOS ONE) BERT recommender for NIH RFAs vs BM25/TF-IDF/doc2vec/NB; Zhang, Yaseen & Wu (2024) funded-grant recommender with biomedical embeddings; GotFunding (Zeng & Acuna, 2024 preprint) learning-to-rank on NIH grant–publication pairs. *Gap:* all US/NIH-centric; none deploy calibrated council probabilities, retrieval evidence and award ranges for the UK.
4. **UK funding data infrastructure.** GtR API (UKRI); GtR-Extended (Thorne, Shepherd & Maynard, 2026) links GtR to opportunities and panel outcomes (38,862 applications; 27.1% linked to projects). *Implication:* confirms opportunity linking is feasible and valuable; panel outcomes raise ethical questions for any "success" feature (kept out of scope).
5. **Trustworthy ML.** Calibration (Guo et al., 2017) — temperature scaling already cut ECE 0.034→0.018 [V-FILE]; conformal prediction (Angelopoulos & Bates, 2021; Romano et al., 2019 CQR) gives distribution-free coverage guarantees — directly applicable to award bands and to council prediction sets; explanations via LIME (Ribeiro et al., 2016), SHAP (Lundberg & Lee, 2017), InterpretML/EBM (Nori et al., 2019) — CS5079 L4 covers LIME and InterpretML.
6. **Retrieval.** BM25 (Robertson & Zaragoza, 2009), dense retrieval (Karpukhin et al., 2020), RRF fusion (Cormack et al., 2009 — measured +2.6 pts Recall@10 here [V-FILE]), FAISS (Johnson et al.; Douze et al., 2024) for scale.
7. **Responsible metrics / research-on-research.** Leiden Manifesto (Hicks et al., 2015), The Metric Tide (Wilsdon et al., 2015), science of science (Fortunato et al., 2018; Wang & Barabási, 2021). *Implication:* present outputs as decision support with uncertainty, never as an assessment of researchers.

### 1.4 Research gaps and novelty (what is defensible)
| Gap | Evidence | FundMatch UK contribution | Status |
|---|---|---|---|
| G1 No open UK applicant-facing tool combining council fit + similar funded evidence + award band | §1.2 desk check | Integrated product | Built (Track A) [V-FILE] |
| G2 Grant classifiers rarely report calibration, faithfulness and OOD transfer together | §1.3 items 2–3 | Calibrated (ECE 0.018), deletion-tested explanations, NSF/NIH/ARC transfer test | Measured [V-FILE] |
| G3 Award "ranges" in tools are uncalibrated | §1.3 item 5 | Coverage-evaluated neighbour band (0.505 for nominal 50%); v2 conformal bands with guaranteed coverage | Partly measured; conformal [PROPOSED] |
| G4 Temporal robustness of funder classifiers is rarely reported | §1.3 | Pre/post-2021 test: 0.827 | Measured [V-FILE] |
| G5 Efficiency–accuracy trade-off for free public hosting | CS5079 L8 | Pre-registered deployment rule; cascade design | Rule measured; cascade [PROPOSED] |

**Novel v2 contributions to aim for (publishable angle):** (a) a confidence-gated *cascade* (cheap TF-IDF LR answers when calibrated p ≥ 0.8 — 80.9% of test cases at 0.943 accuracy [V-FILE] — and escalates the rest to a heavier hybrid model), (b) conformal council *prediction sets* and conformalised award intervals, (c) cross-funder transfer benchmark UK→US/AU/EU built from official APIs.

---

## Section 2 — Complete features and functionalities

Format per feature: **Purpose · Problem · How it works · Benefit · Algorithms/tech (CS5079 link) · Implementation · Testing · Status.**

### 2.1 Core MVP (Track A — all built unless marked)
1. **Funder predictor.** Purpose: rank 8 councils. Problem: remit confusion. How: masked text → TF-IDF (1–2-grams, ≤200k) → class-weighted LogReg (C 32) → temperature T 1.13. Benefit: calibrated probabilities, top-3 0.9916. Tech: TF-IDF + LR (L5), SVM comparator (L7). Impl: `fundmatch_core.predict`. Test: unit tests, held-out macro-F1 0.8615, ECE 0.018. Status: built [V-FILE].
2. **Low-confidence warning.** Warns when top p < 0.60 (accuracy 0.517 in 0.4–0.6 bin). Test: AppTest with ambiguous input. Built.
3. **"Why this council?" explanation.** tf-idf × coefficient per word; deletion test faithful (0.221 vs 0.007). L4 interpretation. Built.
4. **Similar funded projects (top-k, GtR links).** TF-IDF cosine over 113,233 projects; filters by council/year/grant category; Recall@10 0.895. Built.
5. **Typical award band.** P25–P75 and P10–P90 of 30 nearest neighbours' awards; coverage 0.505 / 0.786. Caveat "not a quote". Built.
6. **Funding trend chart** of similar projects by year/council (Altair). Built.
7. **Example abstracts, CSV download, Funding Finder deep links, About/limits panel with OGL attribution.** Built.
8. **Input validation** (empty, < MIN_WORDS, non-English, no known vocabulary). 28 tests. Built.
9. **Public deployment link.** Streamlit Community Cloud. **Not done** — highest priority [V-FILE].

### 2.2 Advanced features (Track B v2; some are cheap enough for Track A if time)
| # | Feature | Purpose / problem | How it works | Algorithms (CS5079) | Implementation | Testing | Priority |
|---|---|---|---|---|---|---|---|
| A1 | Conformal council prediction set | Users need "which councils are plausible" with a guarantee | Split-conformal on calibrated probabilities; return smallest set with 90% coverage | Calibration + conformal (L4 validation) | `mapie` or 20 lines of numpy on OOF scores | Empirical coverage on test ≥ 0.90, mean set size | High |
| A2 | Conformalised award interval | Current band is empirical | CQR on LightGBM quantile models (Ridge baseline), or conformal on neighbour medians | LightGBM (L8), quantile regression | Fit on train, calibrate on held-out split | Coverage at 50/80/90%, width | High |
| A3 | Inflation-adjusted awards | Bands mix 2006 and 2025 pounds | Deflate `award_gbp` by ONS CPIH index (L522) to constant 2025 £; show both | L2 transformation (log, scaling) | Monthly CPIH table joined on `fund_start` month | Unit test vs ONS values; report change in band | High (cheap) |
| A4 | Studentship-aware awards | 29.9% of awards are £0, mostly studentships | Separate band per grant category; explicit message for studentships | L2 categorical handling | Filter in `award_range` | Coverage per category | High (cheap) |
| A5 | Cascade inference | Better accuracy on hard cases without paying for every query | If calibrated p ≥ τ answer with TF-IDF LR; else run hybrid BGE + TF-IDF | L8 cascades, duty cycling | Lazy-load encoder only on escalation; τ tuned on validation | Accuracy/latency/RAM curve vs τ | High (v2) |
| A6 | Hybrid lexical+dense search (RRF) | +2.6 pts Recall@10 measured | RRF of TF-IDF and BGE ranks | L6 sentence embeddings | int8/float16 embeddings, FAISS `IndexFlatIP` or HNSW | Recall@10, MRR, human P@3 | Medium |
| A7 | Scientific encoder comparison (SPECTER2, SciBERT frozen) | Close FR-12/13 honestly | Encode 20k subsample; LR head; retrieval | L6 | GPU-free batch on box/Colab | Macro-F1, Recall@10 vs MiniLM/BGE | Medium |
| A8 | Multi-label / co-funding awareness | Lead-funder-only label penalises joint grants | Use GtR `FUNDER`/co-funder links; one-vs-rest LR; evaluate Hamming/F1 | L5/L7 linear models | Build co-funder table from fund `FUNDER` links | Micro/macro-F1, subset accuracy | Medium |
| A9 | LIME / SHAP cross-check | Show explanations agree across methods | LIME on 50 test docs vs exact linear attributions | L4 LIME, InterpretML | `lime.lime_text` | Rank correlation, deletion test | Medium (cheap; good for CGS-A "course techniques") |
| A10 | Live open-calls panel | Past awards ≠ open calls (main market gap) | Scrape/parse UKRI Funding Finder listing (respect robots/terms) and match to user text with the same TF-IDF space | Retrieval | Nightly job; cache | Precision@5 judged by hand | Medium |
| A11 | Topic map & trend analytics | Portfolio context | UMAP/PCA of embeddings + clusters; time series per council | L2 PCA, KMeans | Precomputed 2-D coordinates | Cluster purity vs research topics | Low |
| A12 | Feedback loop | Learn what users find relevant | Thumbs up/down per result, stored anonymously (no abstract text) | Evaluation data | Supabase/Postgres table of (result_id, rank, vote, model_version) | Monitored relevance rate | Medium |

### 2.3 Future enhancements (commercial / research)
- **F1 Proposal-to-call matcher** for institutions (research offices upload anonymised call lists). **F2 Federated fine-tuning** across universities on unfunded proposals that cannot leave institutions (L8 federated averaging w = Σ (n_k/n) w_k; note gradient-leakage risks from the lecture; add secure aggregation/DP). **F3 Multi-country funders** (NSF, NIH, ARC, CORDIS, NSERC) with funder-agnostic taxonomy. **F4 LLM-assisted rationale** (retrieval-augmented, citing GtR records; never inventing calls). **F5 API + institutional SSO tier**. **F6 Fairness dashboard** by institution type/region (aggregate only). **F7 Welsh/Gaelic UI and plain-language mode**. **F8 Edge/offline build** (quantised ONNX/TFLite encoder; L8 TinyML idea) for privacy-sensitive offices.

Won't have (all tracks): success prediction for individual applications; researcher ranking; storing pasted abstracts.

---

## Section 3 — Data collection, mining, dataset strategy, data dictionary

### 3.1 FR-5 rule (non-negotiable for Track A)
Primary training data **must** come from Shamim's group's own collection code against the GtR API (done: 158,710 projects/funds, 97,102 orgs [V-FILE]). GtR-Extended, Kaggle or GtR bulk exports may be **cited for comparison/discussion only**. International APIs are used **for evaluation only** in Track A.

### 3.2 Dataset catalogue
| Dataset | Source / link | Variables & format | Use in project | Collection | Limitations & ethics |
|---|---|---|---|---|---|
| GtR projects | UKRI · https://gtr.ukri.org/gtr/api/projects | JSON v7: id, title, abstractText, techAbstractText, potentialImpact, leadFunder, grantCategory, status, researchSubjects/Topics, identifiers, links | Training text + labels | Own paginated collector (s 10–100) | OGL v3.0; lead-funder label only; abstracts polished post-award |
| GtR funds | https://gtr.ukri.org/gtr/api/funds | valuePounds.amount (GBP), start/end (epoch ms), category, FUNDED/FUNDER links | Award amounts, dates | Same collector; join on FUNDED | 29.9% zero (studentships); nominal £ |
| GtR organisations | https://gtr.ukri.org/gtr/api/organisations | id, name, address | Lead organisation names | Same | Organisation IDs not unique across UKRI systems (GtR-Extended paper) |
| GtR persons | /persons | names, ORCID | **Not collected** | — | Personal data; excluded by design (NFR-9) |
| Innovate UK participant values | inside project records | organisationName, role, projectCost, grantOffer | v2: business award signal | Already in raw pages | Commercial organisations only |
| NSF awards | https://api.nsf.gov/services/v1/awards.json | id, title, abstractText, directorate, amount, dates | OOD evaluation | Own code (done: 2,000) | Public domain; mapping to UK councils is approximate |
| NIH RePORTER | https://api.reporter.nih.gov/ | appl_id, title, abstract, IC, award_amount | OOD evaluation | Own code (done: 1,000) | ≤ 1 req/s; PI names dropped |
| ARC grants | https://dataportal.arc.gov.au/NCGP/API/grants | summary, scheme, field, amount, year | OOD evaluation | Own code (done: 2,000) | CC BY-NC-ND — evaluation only, never shipped |
| ONS CPIH index | https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l522/mm23 | monthly index 2015=100 (CSV) | Real-terms £ | Download CSV | Official statistic; OGL |
| UKRI Funding Finder | https://www.ukri.org/opportunity/ | title, funder, dates, status, award text | Link-outs (v1); open-call matching (v2) | Respectful scraping or links only | Check terms; snapshot dating |
| OpenAlex | https://api.openalex.org | works, funders (e.g. EPSRC F4320334627), grants | v2 enrichment (publications per funded project) | Free key (since Feb 2026) | CC0; coverage varies |
| Crossref | https://api.crossref.org | DOIs, funder IDs | Enrichment, reference checking | Polite pool with mailto | Metadata only |
| ROR | https://api.ror.org/v2/organizations | ror id, names, types, location | Normalise organisation names | API | CC0 |
| CORDIS | https://data.europa.eu/data/datasets/cordis-eu-research-projects-under-horizon-europe-2021-2027 | projects, objectives, funding | v2 EU extension / OOD test | Bulk download (not paged API; D16) | Different programme structure |
| UKRI panel outcomes | https://www.ukri.org/publications/epsrc-funding-application-outcomes/ | application outcomes, panel data | Discussion only | — | Contains personal data in some files; success prediction out of scope |
| GtR-Extended | https://zenodo.org/records/19243841 | unified PostgreSQL dump | **Comparison only (FR-5)** | — | Not own collection |
| User feedback (v2) | own app | result_id, vote, model_version, timestamp | Relevance evaluation | Opt-in | No abstract text stored; GDPR lawful basis |

### 3.3 Data dictionary — `gtr_projects.parquet` (158,710 rows) [V-FILE]
| Column | Type | Source | Nulls | Notes |
|---|---|---|---|---|
| project_id | string | project.id | 0 | GtR UUID; primary key |
| grant_ref | string | identifiers[0].value | 0 | Never a feature (leaks funder) |
| title | string | title | 0 | Model input |
| abstract | string | abstractText | — | Model input (masked funder names) |
| tech_abstract | string | techAbstractText | — | Not used v1 |
| potential_impact | string | potentialImpact | — | Not used v1 |
| lead_funder | string | leadFunder | 0 | Target; 8 councils kept |
| grant_category | string | grantCategory | 0 | Filter/award stratifier; never a classifier feature |
| status | string | status | 15 | Active/Closed |
| department | string | leadOrganisationDepartment | 54,088 | Sparse |
| lead_org_id | string | LEAD_ORG link | 0 | Join key |
| lead_org_name | string | organisations / LEAD_PARTICIPANT | 0 | Display only |
| research_subjects, research_topics | string (`; `-joined) | taxonomies | 0 | Topic proxy labels for retrieval eval |
| gtr_url | string | built from grant_ref | 0 | Link-out |
| award_gbp | int64 | fund.valuePounds.amount | 0 | 47,416 zeros (47,005 studentships) |
| fund_start / fund_end | timestamp[ms] | fund.start / end | 0 / 17 | 2006-01-01 → 2031-03-30 (56 rows ≥ 2027) |
| fund_category | string | fund.category | 0 | INCOME_ACTUAL 158,529 |
| n_funds | int64 | count of funds per project | 0 | Always 1 in this snapshot |
| **[PROPOSED v2]** award_gbp_real2025 | float | award × CPIH(2025)/CPIH(start month) | — | Real-terms award |
| **[PROPOSED v2]** cofunders | list | fund FUNDER links / partner roles | — | Multi-label target |

### 3.4 Preprocessing plan (done [V-FILE] + proposed)
Done: programme-funder removal; placeholder, short (<30 words), boilerplate and duplicate removal (table in audit); funder-name masking; group-aware stratified split; TF-IDF (min_df 3, max_df 0.5, sublinear, ≤200k features); frozen 384-d embeddings cached in 5,000-row chunks.
Proposed v2: (1) CPIH deflation; (2) studentship stratification; (3) language ID and Unicode NFKD normalisation (L5); (4) near-duplicate detection with MinHash on shingles; (5) drop or flag future-dated records; (6) `HashingVectorizer` (L2 feature hashing) to remove the vocabulary dictionary from RAM; (7) PCA (L2) or int8 quantisation (L8) of embeddings for memory.

### 3.5 Quality, bias, privacy, licensing
- **Quality checks:** schema assertions, join coverage (100% funds matched [V-FILE]), per-step row counts, snapshot date logged, duplicate audit.
- **Bias:** class imbalance (EPSRC 36k vs STFC 7k raw) handled by class weights; institutional skew (research-intensive universities dominate awards) → report retrieval/award results by organisation type in v2; abstracts are of *funded* work only (survivorship) — never imply success likelihood.
- **Privacy (UK GDPR, ICO guidance https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/):** no person endpoint collected; foreign PI names dropped; user abstracts processed in memory only; v2 feedback stores no text; DPIA-lite before any login feature.
- **Licensing:** GtR under OGL v3.0 — attribute "Contains public sector information licensed under the Open Government Licence v3.0" (already in app); no UKRI endorsement implied; NSF public domain; NIH public; ARC CC BY-NC-ND (evaluation only); OpenAlex CC0; code MIT/Apache-2.0 [PROPOSED].

---

## Section 4 — University & research institution landscape (potential contacts only)

**Disclaimer:** every entry is a *potential academic contact or related research group*. **No collaboration, endorsement or agreement exists or is implied.** Affiliations verified 24 Nov 2026 (VERIFIED_SOURCES §D).

| Region | Institution / unit | Researcher (verified role) | Key publication | Relation to FundMatch UK | Realistic opportunity |
|---|---|---|---|---|---|
| UK | Univ. of Aberdeen, Computing Science (AberdeenML) | Dr Mingjun Zhong (Lecturer in ML; CS5079) | — | Course coordinator | Formative feedback through normal course channels only |
| UK | Univ. of Aberdeen, Computing Science | Dr Dewei Yi (Senior Lecturer; Director MSc AI) | — | Edge AI / federated learning teaching | MSc project supervision conversation for v2 |
| UK | UCL STEaPP / Research on Research Institute | Prof James Wilsdon (Prof. of Research Policy; Exec. Director RoRI) | The Metric Tide (2015) | Responsible use of funding data | Short expert-feedback email on responsible framing |
| UK | Univ. of Sheffield, CS NLP group (GATE) | Dr Diana Maynard (Senior Research Fellow) | GtR-Extended, NSLP 2026 | Closest UK dataset work | Ask for comments on evaluation; cite properly |
| UK | Univ. of Sheffield, Information, Journalism & Communication | Prof Mike Thelwall (Prof. of Data Science) | Kousha & Thelwall 2023 | AI in research assessment | Expert review of limitations section |
| UK | Univ. of Sussex, SPRU | Dr Ohid Yaqub | Yaqub, Coburn & Moore 2024 (Research Policy) | Funding allocation research | Feedback on funder-fit framing |
| UK | Alan Turing Institute | institution only | — | National AI community | Public events/talks |
| Europe | Leiden Univ., CWTS | Prof Ludo Waltman (Scientific Director) ; Nees Jan van Eck | Leiden Manifesto 2015; VOSviewer 2010 | Science mapping for trend view | Methods feedback; VOSviewer-style maps |
| Europe | Fraunhofer ISI, Innovation & Knowledge Economy | Dr Rainer Frietsch | Neuhäusler & Frietsch 2012 | Indicators | Background reading |
| Europe | TU Darmstadt, UKP Lab | Prof Iryna Gurevych | Sentence-BERT 2019 | Embedding methods | Citation; possible student workshop |
| US | Allen Institute for AI (Semantic Scholar) | Doug Downey, Sergey Feldman (Ai2 directory) | SPECTER 2020; SciRepEval 2023 | Scientific embeddings (SPECTER2) | Use open models; issues on GitHub |
| US | Yale, CS (Yale NLP) | Asst Prof Arman Cohan | SPECTER; SciRepEval | Scientific NLP | Citation |
| US | Northwestern, Kellogg CSSI | Prof Dashun Wang | The Science of Science 2021 | Science of science | Background; possible research question framing |
| US | Indiana Univ., CNS Center | Prof Katy Börner | Science of science 2018 | Visualising funding | Visual design inspiration |
| Africa | Stellenbosch Univ., CREST / SciSTIP | Prof Johann Mouton | Science-granting councils in SSA (2019) | Funding systems beyond UK | Future extension to African councils |
| Africa | Univ. of Pretoria, DSFSI | Prof Vukosi Marivate (ABSA UP Chair of Data Science) | Masakhane (Findings EMNLP 2020) | Low-resource NLP transfer | Multilingual extension |
| Africa | Masakhane; AIMS | communities/institution | Nekoto et al. 2020 | Open NLP communities | Open-source contributors/testers |

How to approach (v2, after marks are released): one short, specific email per person with the public demo link, a 1-page results summary and one concrete question; never claim affiliation; record replies in a contacts log; cite their work regardless of reply.

---

## Section 5 — AI/ML algorithms & full technical architecture

### 5.1 CS5079 lecture → product alignment
| Lecture (course notes) | Technique taught | Where it powers FundMatch UK | Status |
|---|---|---|---|
| L1 Introduction + practical | sklearn estimator/transformer API, `GridSearchCV`, pandas, Keras MLP | Whole pipeline in sklearn; CV tuning of C/α; BGE+MLP head (0.8197) | Built |
| L2 Working with features | Cleaning, log transform, binning, encoding, feature hashing, scaling, χ²/RFE selection, PCA | Log-award target; grant-category stratification; TF-IDF features; v2: `HashingVectorizer`, PCA-compressed embeddings | Built / v2 |
| L3 RL with OpenAI Gym (file pending in notes; used in Task 2 FrozenLake DQN) | Q-learning / DQN | Not used in Track A product (honest); v2 idea: contextual-bandit re-ranking from feedback (needs real traffic) | Future |
| L4 Tuning, interpretation, deployment | Bias–variance, K-fold CV, Grid/Randomized search, LIME, InterpretML, joblib/pickle, IaaS/PaaS, Streamlit 3-file app | Group CV, pre-registered deployment rule, exact linear explanations + faithfulness test, pickled assets, `core`/`app` split on Streamlit | Built; LIME cross-check v2 |
| L5 Sentiment prediction | Text normalisation, BOW/TF-IDF (1,2)-grams, LogReg, SGD/linear SVM, Word2Vec, LSTM | Deployed TF-IDF (1,2)-gram + LogReg; linear SVM comparator | Built |
| L6 Text classification with LLMs | Frozen sentence-transformer embeddings + LR; task-specific transformers | MiniLM/BGE frozen embeddings + LR/MLP; ensemble; v2 SPECTER2/SciBERT | Built / v2 |
| L7 Support vector machines | Margin, soft margin C, kernels, dual | LinearSVM (C 1) tied best F1 0.8622 but no native probabilities → justified not deployed | Built |
| L8 AI for embedded devices | Edge/fog/cloud trade-offs, quantisation, pruning, cascades, duty cycling, federated learning, XGBoost/LightGBM | Memory-bounded deployment (651 MiB vs 690 MB tier), index pruning (`prune_index.py`), sharding, hibernation handling, LightGBM award model; v2 cascade + int8 embeddings + federated idea | Built / v2 |

### 5.2 Model recommendations with justification
| Task | Recommended (deploy) | Why | Limitations | Alternatives worth testing | Train/validate/test | Metrics | Improving reliability/fairness/scale |
|---|---|---|---|---|---|---|---|
| Council classification | TF-IDF (1–2-gram) + class-weighted LogReg + temperature scaling | Best CV-F1 among probabilistic models, 370 MB / 57 ms [V-FILE]; exact explanations | Lexical; drift (0.827 post-2021); lead-funder only | LinearSVM + Platt; SPECTER2 + LR; fine-tuned SciBERT (GPU); LLM few-shot | StratifiedGroupKFold 3-fold CV on train; one test pass; temporal split | Macro-F1, per-class F1, top-3, ECE/Brier, OOD accuracy | Conformal sets; yearly retrain; multi-label; per-council error audit |
| Similar projects | TF-IDF cosine (exact, sparse) | Highest topic P@10 (0.350); shares vectoriser | Vocabulary mismatch | Hybrid RRF (+2.6 pts R@10), BM25, SPECTER2 + FAISS HNSW | Known-item + topic-proxy + human P@3 | Recall@10, MRR@10, nDCG@10, human κ | Hybrid in cascade; FAISS for >1M docs |
| Award band | 30-NN P25–P75/P10–P90 | Coverage closest to nominal (0.505) | Wide (×3.9) | LightGBM quantile + CQR; Ridge | Train/val/test on non-zero awards; by category | MAE log £, coverage, width | CPIH deflation; per-category bands; conformal |
| Explanation | Exact tf-idf × coef | Faithful by construction; tested | Word-level only | LIME, SHAP, EBM for award model | Deletion/insertion tests | Prob. drop, flip rate | Show top-k positive and negative words |

### 5.3 Architecture A — CS5079 free-tier (Track A, current)
Browser → Streamlit Community Cloud (Python 3.13) → `app.py` (UI) → `fundmatch_core.py` → in-memory assets (`tfidf_lr.pkl`, sharded CSR index, `projects_meta.parquet`, `model_info.json`) loaded once with `st.cache_resource`. Offline: `collect_gtr.py` → raw gz JSON → `build_dataset.py` → Parquet → notebook (train/evaluate/select/export) → assets → GitHub → Streamlit. No database, no auth, no secrets (GtR needs no key). Resource evidence: peak 651 MiB vs documented 690 MB min / 2.7 GB max [V-FILE, V-LIVE].

### 5.4 Architecture B — Production SaaS (Track B)
| Layer | Choice | Reason |
|---|---|---|
| Frontend | Next.js (React, TypeScript) on Vercel; accessible component library; charts | SEO landing page, fast UI, WCAG AA |
| API | FastAPI (Python) container on Google Cloud Run (or Render/Fly) | Same Python ML stack; scales to zero |
| Models | Versioned artefacts (sklearn pickles + ONNX encoder) in object storage; loaded at container start; cascade | Reproducible, cheap |
| Vector search | FAISS (in-process) or pgvector in Postgres | 113k–1M docs fit in memory |
| Database | Postgres (Supabase/Neon): projects metadata, model registry, feedback, saved searches | Relational + vector |
| Batch pipeline | Scheduled job (GitHub Actions/Cloud Scheduler) weekly: GtR delta collection → validation → retrain → evaluation gate → promote | Data freshness; drift control |
| Auth | Optional accounts for saved searches (Clerk/Supabase Auth, OAuth); anonymous use stays free | Least data |
| Security | HTTPS everywhere, secrets in platform secret manager, rate limiting/WAF, dependency scanning (Dependabot), no abstract logging, CSP headers | OWASP basics |
| Observability | Structured logs without user text, latency/error metrics, model-version tags | Reliability |
| Hosting/domain | Custom domain (e.g. `fundmatch.uk` if available — not checked) with managed TLS (Let's Encrypt via platform) | Professional |

### 5.5 System architecture diagram (text form; the Page draws it)
```
[GtR API v7] --own collector--> [Raw JSON cache] --> [build_dataset: join funds/orgs] --> [Parquet snapshot]
     [ONS CPIH] ------------------------------------------------------------^ (v2 deflation)
[Parquet] --> [Notebook: clean • mask • group split • TF-IDF/embeddings • CV • calibrate • evaluate • select] --> [Model + index artefacts]
[Artefacts] --> [Streamlit app (Track A)]  or  [FastAPI service + Postgres/FAISS (Track B)] <-- [User browser]
[User feedback (v2, no text)] --> [Postgres] --> [Evaluation & retraining gate]
[NSF / NIH / ARC APIs] --> [OOD test sets] --> [Evaluation only]
```

---

## Section 6 — 10-stage development roadmap

Status key: ✅ done [V-FILE] · 🟡 partly · ⬜ not started. Priority P0 (submission-critical) → P3.

| Stage | 1 What | 2 Why | 3 Tools | 4 Data | 5 How | 6 Test | 7 Output | 8 Exit criterion | Status / priority |
|---|---|---|---|---|---|---|---|---|---|
| 1 Research & planning | Problem, requirements, scope decisions, literature | Marks for usefulness/justification | Markdown, Zotero | Brief, rubric | Requirements doc + SCOPE_DECISIONS D1–D19 | Supervisor-style review vs rubric | Requirements, gap table | Every rubric item mapped | ✅ (v2 refresh 🟡) P1 |
| 2 Data collection | Own GtR collector + intl. test sets | 2a/2b marks; data size | requests, gzip | GtR, NSF, NIH, ARC | Paged, rate-limited, resumable | Counts vs API totalSize; log | 158,710/158,710/97,102 + 5,000 | Totals equal; log saved | ✅ |
| 3 Preprocessing | Clean, mask, dedupe, split | 3a marks; leakage | pandas, sklearn | Parquet | Step table; group split | Before/after counts; leakage experiment | 113,233 clean; 90,585/22,648 | No group overlap; counts reported | ✅ (v2: CPIH, studentships ⬜ P1) |
| 4 Model development | Baselines → linear → embeddings → ensemble; search; award | 3b/3c marks | sklearn, sentence-transformers, LightGBM | Train split | CV tuning; one test pass | Macro-F1, top-3, ECE, R@10, coverage | summary.json, figures | Test used once; seeds fixed | ✅ (v2 SPECTER2/conformal ⬜ P2) |
| 5 Backend | Core logic module | Maintainability | Python | Artefacts | `fundmatch_core.py` | 17 unit tests | Tested core | All tests pass | ✅ (v2 FastAPI ⬜ P3) |
| 6 Frontend | Streamlit UI | Deployment marks | Streamlit, Altair | Metadata | Examples, filters, charts, About | AppTest e2e (11) | App | e2e pass; accessibility check | ✅ (v2 Next.js ⬜ P3) |
| 7 Integration | Export → app → shards | Reproducibility | Python scripts | Artefacts | Sharding <100 MB/file | Clean venv test | deploy_repo | 28/28 pass in clean venv | ✅ |
| 8 Testing & evaluation | Calibration, OOD, faithfulness, human relevance | In-depth analysis | numpy, pandas | Test sets | Protocols in §7 | Pre-registered metrics | Tables/figures | Human ratings complete | 🟡 human ratings ⬜ **P0** |
| 9 Deployment | Public GitHub + Streamlit Cloud | 5 marks | git, GitHub, Streamlit | deploy_repo | README steps | Live smoke test, screenshot, memory | Public URL | URL works in private window; wakes from sleep | ⬜ **P0** |
| 10 Documentation & presentation | Report, README, demo video, contribution form | Report quality | docx/pdf, OBS | Results | Fill placeholders; rebuild report | Word count ≤ 1,500; link check | PDF + ZIP + form | All placeholders gone; ZIP rebuilt | 🟡 **P0** |

Dependencies: 9 → (screenshot, links) → 10; 8 (human ratings) → 10 (one table row); v2 stages restart at 3 on a separate branch.

---

## Section 7 — Testing, evaluation & research validation

### 7.1 Framework
| Layer | What | Method | Criterion | Status |
|---|---|---|---|---|
| Functional | Input checks, predictions, search filters, CSV | pytest unit (17) + Streamlit AppTest e2e (11) | 100% pass | ✅ 28/28 [V-FILE] |
| ML performance | Macro-F1, per-class F1, top-3, confusion | Group CV + single test pass | Beat majority (0.054) and NB (0.807) | ✅ 0.8615 |
| Calibration/reliability | ECE, Brier, reliability diagram, accuracy by confidence | Temperature on OOF | ECE ≤ 0.02 | ✅ 0.018 |
| Temporal robustness | Train <2021, test ≥2021 | Forward split | Report drop | ✅ 0.827 |
| OOD transfer | NSF/NIH/ARC mapped to councils | Unchanged models | Report by source | ✅ 0.596–0.753 acc |
| Retrieval | Known-item Recall@10/MRR; topic Hit@10; **human P@3, κ** | 2,000 queries; 50×3 sheet | Human P@3 reported | 🟡 human ⬜ |
| Award band | Coverage, width, MAE log £ | 14,925 test | Coverage within ±3 pts of nominal | ✅ 0.505 / 0.786 |
| Explanation | Deletion test | top-5 vs random-5 | ≥ 10× random drop | ✅ 0.221 vs 0.007 |
| Bias/fairness | Per-council F1; v2: by institution type/region (Russell Group vs others, nations) for retrieval relevance and band coverage | Stratified metrics | Gaps reported, not hidden | 🟡 per-council only |
| Security/privacy | No secrets in repo; no text logging; dependency audit; input length limits | `gitleaks`/`pip-audit` [PROPOSED]; code review | 0 findings | ⬜ |
| Performance | Cold load, median/p95 latency, peak RAM | Benchmark + live VmHWM | p95 < 1 s; RAM < 690 MB | ✅ 68 ms p95; 651 MiB |
| UAT | 5–8 target users (PhD students, research office staff) | Think-aloud tasks + SUS questionnaire | SUS ≥ 70 [PROPOSED target] | ⬜ |
| Real-world scenarios | Interdisciplinary, industry, humanities, very short, non-English abstracts | Scripted cases | Sensible warnings | 🟡 (German input tested) |

### 7.2 Baseline vs advanced (verified numbers only; targets marked)
| Model | Test macro-F1 | Top-3 | Peak RAM | Latency | Evidence |
|---|---|---|---|---|---|
| Majority | 0.054 | 0.606 | — | — | [V-FILE] |
| TF-IDF + CNB | 0.807 | 0.975 | 393 MB | 57 ms | [V-FILE] |
| **TF-IDF + LR (deployed)** | **0.862** | **0.992** | 370 MB | 57 ms | [V-FILE] |
| TF-IDF + LinearSVM | 0.862 | 0.987 | — | — | [V-FILE] |
| BGE + MLP | 0.820 | 0.988 | 756 MB | 192 ms | [V-FILE] |
| Ensemble TF-IDF + BGE | 0.858 | 0.992 | 790 MB | 190 ms | [V-FILE] |
| SPECTER2 + LR | target: report, no claim | — | — | — | [PROPOSED] |
| Cascade (τ = 0.8) | target: ≥ 0.862 with < 1 GB | — | — | — | [PROPOSED] |

### 7.3 Documentation of experiments
Each experiment: hypothesis, data snapshot date, split, seed (42), hyper-parameter grid, metric, result table, figure, one-sentence interpretation, threat to validity. Keep `models/summary.json` as the single source of truth; the report generator already reads it (`write_facts.py`) — keep that pattern so no number is typed by hand.

---

## Section 8 — Professional UI/UX & product design

| Area | Track A (Streamlit, now) | Track B (v2 product) |
|---|---|---|
| Landing | Title + one-line value + example buttons | Marketing landing: problem, 3-step "paste → see fit → explore evidence", trust strip (data source, date, accuracy), privacy promise |
| Workspace | Title/abstract inputs, primary button, results in numbered sections | Split view: editor left, results right; sticky summary of top council + band |
| Explainability visuals | Probability bars, words behind the prediction | Highlighted words in the user's own text (positive/negative), conformal set chips, reliability badge ("predictions like this are right 94% of the time" from the 0.8–1.0 bin) |
| Evidence | Table of similar projects with GtR links, filters, CSV | Cards with council, amount (nominal and real £), dates, organisation, "why similar" shared terms; compare-two view |
| Analytics | Trend chart | Topic trend by council/year, award distribution histogram, open calls panel |
| User management | none (privacy) | Optional account for saved searches/alerts; delete-my-data |
| Admin | none | Model registry, data freshness, drift dashboard, feedback rates, error logs (no user text) |
| Responsive | Streamlit default | Mobile-first breakpoints |
| Accessibility | Labelled charts, colour-blind-safe palette (NFR-8) | WCAG 2.2 AA (https://www.w3.org/TR/WCAG22/): contrast ≥ 4.5:1, keyboard navigation, focus states, ARIA labels on charts, text alternatives for every chart |
| Feedback | none | Thumbs on each result + "report a problem"; quarterly UAT with SUS |
| Copy & ethics | "Not official UKRI advice"; OGL attribution | Same, plus plain-language limits next to every number |

---

## Section 9 — Deployment & publication strategy

### 9.1 Track A deployment (do this first; P0)
1. Create a **public** GitHub repo (e.g. `fundmatch-uk`) from the *contents* of `deploy_repo/` (`app.py`, `fundmatch_core.py`, `requirements.txt`, `README.md`, `assets/` — 8 shards, each < 21 MB, so no Git LFS needed). Use `git push` (web uploader rejects > 25 MB).
2. share.streamlit.io → Create app → repo/branch `main`/`app.py` → Advanced settings **Python 3.13** (pickles built with scikit-learn 1.9.1). No secrets needed.
3. Smoke test in a private window: load an example, run, check GtR links, CSV, About panel. Note RAM in "Manage app" logs; if "over its resource limits", run `prune_index.py 100 --write` and redeploy.
4. Screenshot the live app; rebuild report with `python report/make_report.py --app-url … --github-url …`; rebuild ZIPs.
5. Keep-alive: apps sleep after 12 h without traffic [V-LIVE]; open the link on submission day and before marking; state the wake-up step in the README/report.
6. Optional: apply for Streamlit "good-for-the-world" increased resources (educational, open-source) [V-LIVE].

### 9.2 Track B production path
- **Hosting:** frontend on Vercel; API container on Cloud Run/Render/Fly; Postgres on Supabase/Neon. Hugging Face Spaces CPU Basic is 2 vCPU/16 GB but compute Spaces now need a paid plan [V-LIVE] — use only if budget allows.
- **Domain & SSL:** register a domain (availability not checked), point DNS (CNAME/A) to the host; managed TLS certificates are automatic on these platforms; enforce HTTPS + HSTS.
- **Model serving:** load versioned artefacts at start-up; health endpoint; cascade; cache embeddings for metadata; 1 worker per 1 GB RAM.
- **Secrets:** platform secret manager / `.streamlit/secrets.toml` (never committed); `.gitignore`; rotate keys (OpenAlex key, DB URL).
- **GitHub repo standard:** README (problem, demo GIF, quick start, results table, data licence, limitations), `LICENSE` (MIT/Apache-2.0 for code; data stays OGL), `CITATION.cff`, `CONTRIBUTING.md`, issue templates, GitHub Actions (pytest, lint, `pip-audit`), releases tagged with data snapshot date, model card and data card (`docs/`).
- **Technical documentation:** architecture page, data dictionary (Section 3.3), evaluation protocol (Section 7), API reference (FastAPI OpenAPI auto-docs).
- **Demo video (2–3 min):** problem (15 s) → paste abstract (30 s) → council + why (30 s) → similar projects + band (30 s) → limits/ethics (15 s) → architecture + metrics (30 s). Captions on; host on YouTube unlisted + link in README.
- **Portfolio:** case-study page (problem, role, data, models, numbers, trade-offs, lessons), link to repo, video and live app; LinkedIn post after marks are released.

### 9.3 Four different "publication" routes
| Route | What it is | Minimum bar | Practical strategy for FundMatch UK |
|---|---|---|---|
| 1 Working software product | A live URL people can use reliably | Uptime, tests, monitoring, clear limits | Track A live app now; Track B with monitoring + data refresh |
| 2 Open-source project | Public code others can reuse/contribute to | Licence, README, reproducible setup, CI, issue tracker | Release collector + pipeline + app under MIT/Apache-2.0 after marks are returned (check course rules on sharing assessed work before making the repo public beyond the marker link — **confirm with coordinator**) |
| 3 Academic research paper | Peer-reviewed contribution with novelty and rigorous evaluation | Clear research question, related work, reproducible experiments, ethics | Target a workshop (e.g. scholarly-document/NLP-for-science workshops, or NSLP-type venues) with: calibrated + conformal funder prediction, cascade efficiency, UK→US/AU transfer benchmark, human relevance study. Co-authorship only with real contributors; ask the supervisor about authorship norms |
| 4 Commercialisation | Paying customers / sustainable service | Value proposition, legal terms, data licence compliance, support | Free for individuals; paid institutional tier (open-call matching, bulk portfolio analysis, API, SSO). OGL permits commercial reuse with attribution; ARC data (NC-ND) must never be in a commercial product. Explore university enterprise support only after grading |

---

## Section 10 — A1/A+ academic excellence strategy (mapped to the verified brief)

### 10.1 Line-by-line rubric map (Task 1, 30 marks; ≤1,500 words)
| Item (marks) | What markers need | Current evidence [V-FILE] | Risk | Action to secure full marks |
|---|---|---|---|---|
| Usefulness (2) | Clear user + problem + why existing tools fall short | Report §1.1 with Funding Finder, Matchmaker, Dimensions, prior research | Low | Add one clause on verified GtR scale (158,710) if words allow |
| Uses data (1) | Data is central | §1.2 | Low | — |
| Data sufficiency (2) | Argument + evidence | Learning curve 0.778 → 0.862; smallest class 5,511 | Low | — |
| Own sources (2) | Not prepared datasets | GtR all pages + NSF/NIH/ARC; FR-5 statement | Low | Mention GtR-Extended exists and was deliberately **not** used (shows awareness) |
| Own code (3) | Collector quality | `collect_gtr.py` (rate limit, back-off, resume, log), Figure 2 | Low | Note live check: page size 10–100 enforced by API |
| Preprocessing/features (3) | Steps + justification | Table 1 cleaning, leakage controls, TF-IDF/embeddings | Low | Optional: one line on studentship £0 awards and CPIH caveat |
| Candidate models (3) | Why each suits | §3.2 with citations | Low–Med | Add a short course link per model family (L5–L8) |
| Training (3) | Process detail | Group CV, class weights, tuning by CV, one test pass, seeds | Low | — |
| Deployment choice (6) | Compare vs competitors on accuracy **and** deployability | Pre-registered rule, Table 2 RAM/latency, Figure 4, award-model trade-off | Med (depends on working live app) | Add live-host RAM reading from Streamlit logs once deployed |
| Deployment (5) | Process, showcase, **link** | Process + screenshots; **link placeholder** | **High** | Deploy, live screenshot, working link, GitHub link |
| Overall: quality, reproducibility, understanding, experiments, analysis, code | | 28 tests, executed notebook 61 cells/0 errors, README reproduce steps, critical evaluation §5 | Med | Fill placeholders; Gen-AI statement; human relevance results; contribution form |

### 10.2 Grading risks and mitigations
1. **No live link (R1)** → deploy this week; test from a phone and a private window; keep-alive before marking.
2. **Free-tier memory (R2)** → pruning fallback ready; document measured RAM.
3. **Placeholders / AI-use statement / contribution form (R3)** → complete and have every member sign the form electronically.
4. **"No human evaluation" (R4)** → rate the 150-row sheet with two raters; report P@3 and κ in one table row.
5. **Requirements deviations (R5)** → keep the honest justification (GPU, memory) — markers reward justified scope; do not add half-finished SciBERT.
6. **Word cap (R6)** → net-zero words: every addition replaces text; prefer table rows.
7. **Course alignment visibility (R9)** → one compact table or caption mapping models to lectures.
8. **Academic integrity** → all borrowed code credited (report already states none copied); generative-AI use declared honestly per course policy.
9. **Reproducibility** → ZIP includes notebook with outputs, `requirements.txt`, README; the marker should not need to re-collect 30+ minutes of data — mention cached raw pages are excluded/included explicitly.

### 10.3 What makes it A1/A+ rather than A5 (examiner's view)
Demonstrated: huge, diverse, self-collected data; leakage-aware protocol; multiple model families from the course; calibration; OOD and temporal robustness; faithfulness test; quantified deployability; working public product with tests. Missing only: live link and human relevance evidence. With those two closed and placeholders removed, Track A is evidence-complete for every rubric line.

---

## Section 11 — Final deliverables tracker (status as of 24 Nov 2026)
| # | Deliverable | Status | Evidence | Next action |
|---|---|---|---|---|
| 1 | Complete project strategy | ✅ Done | This file | Revisit after deployment |
| 2 | Research proposal | 🟡 Partial | Requirements + Section 1 | Write 2-page v2 proposal (question, method, evaluation) |
| 3 | Literature review | 🟡 Partial | Report §1.1 (short); Section 1.3 here | Expand to v2 review (~2,000 words) with verified refs only |
| 4 | Research gap & novelty analysis | ✅ Done | Section 1.4 | — |
| 5 | Dataset collection & mining strategy | ✅ Done (Track A) | `collect_gtr.py`, logs; Section 3 | v2: CPIH, OpenAlex enrichment |
| 6 | Data dictionary | ✅ Done | Section 3.3 | Add v2 columns when built |
| 7 | Data preprocessing plan | ✅ Done (Track A) | Notebook §3; Section 3.4 | v2 items |
| 8 | Technical architecture | ✅ Done | Section 5 | — |
| 9 | AI/ML algorithm selection | ✅ Done | Notebook §6–9; Section 5.2 | v2 SPECTER2/conformal experiments |
| 10 | Feature & functionality list | ✅ Done | Section 2 | — |
| 11 | Development roadmap | ✅ Done | Section 6 | — |
| 12 | Implementation plan | 🟡 Partial | Track A implemented | v2 sprint plan (backlog from Section 2.2) |
| 13 | Testing & evaluation framework | 🟡 Partial | 28 tests + notebook metrics | Human relevance rating; UAT; security scan |
| 14 | Professional UI/UX design | 🟡 Partial | Streamlit app | v2 wireframes |
| 15 | Deployment strategy | 🟡 Partial | README steps | **Execute: public GitHub + Streamlit URL** |
| 16 | GitHub repository & documentation | ⬜ Not started (no repo) | `deploy_repo/` not a git repo | Create public repo, README, licence, CI |
| 17 | Final academic report | 🟡 Nearly done | 6-page PDF, 1,419 words | Fill links, members, Gen-AI statement; rebuild |
| 18 | Final presentation | ⬜ Not started | — | 8–10 slide deck if required (no presentation is required by the brief text) |
| 19 | Professional product demonstration | ⬜ Not started | Screenshots only | 2–3 min demo video after deployment |

---

## Section 12 — Evidence standards, verified vs proposed, honest limitations
- **Verified (this audit):** every number tagged [V-FILE] comes from `models/summary.json`, `deployment_selection.csv`, `model_info.json`, `collection_log.json`, the report PDF or the parquet profile; [V-LIVE] facts were fetched on 24 Nov 2026 (VERIFIED_SOURCES).
- **Proposed:** SPECTER2/SciBERT rows, cascade, conformal sets/intervals, CPIH deflation, multi-label, UAT/SUS targets, production stack — none measured; do not report them as results.
- **Assumptions:** that markers can open a Streamlit app at marking time; that Streamlit limits documented "as of February 2024" still hold (the doc itself warns they may change); domain availability; that the course allows a public repo (confirm).
- **Known limitations of the product:** lead-funder-only labels; funded abstracts only (survivorship; no success inference); nominal £; temporal drift; lexical model's sensitivity to wording; OOD accuracy 0.60–0.75; proxy-only retrieval relevance; organisation IDs imperfect across UKRI systems.
- **Limitations of this research pass:** competitor facts limited to what public pages state (Pivot-RP page bot-blocked; confirmed via search snippets); no paywalled papers read in full; course notes for L3 RL and two notebooks are still pending; Iz Beltagy's and Kyle Lo's affiliations changed recently — cite their papers, don't rely on roles.
- **Never fabricated:** no metric, paper, dataset or collaboration was invented; unverifiable people were omitted (Section 4 note).
