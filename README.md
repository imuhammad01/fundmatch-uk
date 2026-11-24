# ScholarPulse UK

CS5079 Applied AI final submission. Md Shamim Mia, Student ID 52536283, University of Aberdeen.

ScholarPulse UK is a local research and funding command deck for every UK student stage: undergraduate dissertation, MSc taught project, PhD applicant, current PhD, international student, and student founder. The display name lives in `src/fundmatch/brand.py`. The import package and the command stay `fundmatch`, so existing imports do not change.

Paste a title and an abstract. The app returns a calibrated shortlist of UKRI councils, a conformal set tied to the published test profile, similar funded projects, a band of past awards, five working titles, a readiness radar, an upgrade coach, reviewer questions, and the UK organisations nearest that topic. The classifier is the exported CS5079 model: 200,000 sublinear word and bigram features, class-weighted logistic regression, temperature **1.1314305006097425**. Held-out macro-F1 **0.8615**, accuracy **0.8693**, top-3 **0.9916**, ECE **0.0336 → 0.0180**. The index is all **113,233** projects collected on 23 November 2026. The table has organisations, not people. Pasted abstracts are not stored.

This is not official UKRI funding advice. UKRI does not endorse it.

```
title + abstract
      │
      ▼
FundMatchEngine  ── artifacts/tfidf_lr_weights.npz
      │            ── artifacts/svd_index_int8.npz
      │            ── data/projects_meta.parquet
      ├── fundmatch              Rich CLI
      ├── streamlit run app.py   ScholarPulse UK
      ├── fundmatch serve        FastAPI, version 4.0.0
      └── notebooks/01_fundmatch_pipeline_and_evaluation.ipynb
```

## Quick start

Python 3.11 or 3.12. The same code runs on Windows Python 3.14.

```bash
pip install -r requirements.txt
pip install -e .
fundmatch ui
fundmatch serve --port 8741
fundmatch analyze --demo epsrc-quantum --student --cpih
fundmatch coach --demo epsrc-quantum --persona masters
fundmatch redteam --demo epsrc-quantum --persona phd
fundmatch hubs --demo epsrc-quantum
fundmatch dossier --demo epsrc-quantum --persona undergrad --output scholarpulse_research_dossier.md
fundmatch topics --interest "school exclusion and later earnings" --persona undergrad
fundmatch trends --interest "wheat drought genetics"
fundmatch fuse --a "music" --b "machine learning"
fundmatch score --demo epsrc-robotics --persona masters
pytest -q
jupyter nbconvert --to notebook --execute --inplace notebooks/01_fundmatch_pipeline_and_evaluation.ipynb
```

`./start.sh` and `docker compose up --build` start Streamlit on port 8765.

## What a student can do

- Read eight council probabilities, a 90% and 95% set, and live Funding Finder links. A single council is shown at 90% only when the probability is at least 0.80, because that test bin was 94.3% accurate. Otherwise the set is the top three (test top-3 accuracy 0.9916).
- See how close the text sits to the 113,233-project map, and open the nearest funded projects by organisation.
- Generate five working titles from words in those real titles, each with a council, a calibrated percentage, and a novelty band.
- Ask the upgrade coach for 6–8 terms that appear in funded neighbours and are missing from the draft, then read a labelled before/after simulation. The after-score is not a new scientific claim.
- Read five viva questions and a five-dimension feasibility matrix.
- See the ten organisations with the most awards among the 60 nearest projects, in October 2026 pounds.
- Download one Markdown dossier: idea, radar, councils, titles, hubs, coach, email template, red-team, funding wave, and the OGL v3.0 footer.

## Tests and speed

```bash
pytest -q
```

GitHub Actions runs the suite on Python 3.12. A warm council query is about 90 ms. `fundmatch benchmark` on the v3 build reported a cold start of 2.72 s, a median warm query of 90.1 ms, an int8 query of 86.8 ms, and 568 MB resident after load. Warm `POST /api/v1/topics`, `/coach`, `/redteam`, and `/hubs` stay under 150 ms because each uses one neighbour search and the calibrated logits, not five full analyses.

Real-terms pounds use ONS CPIH series L522. October 2026 is index 144.6, the latest month in the 18 November 2026 release. A full-year 2026 average is not published, so it is not used.

## Ethics

Contains public sector information licensed under the Open Government Licence v3.0. https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

The score is resemblance to funded text, not a chance of success. It does not write the science, rank people, or store the abstract you paste. Organisations and award figures come from the 23 November 2026 Gateway to Research snapshot.
