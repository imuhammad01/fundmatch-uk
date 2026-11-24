# FundMatch UK

Paste a research title and abstract to see:

1. which UKRI council most likely funds work like yours (calibrated probabilities, the words that drove the
   prediction, and links to that council's open calls on the UKRI Funding Finder);
2. a typical award range (P25–P75 and P10–P90 of the 30 most similar funded projects), with a trend chart;
3. the 10 most similar projects UKRI has already funded, with Gateway to Research links and a CSV download.

Student project for CS5079 Applied AI (University of Aberdeen). **Not official UKRI funding advice**; UKRI does
not endorse it.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Push this folder to a public GitHub repository, then on share.streamlit.io choose the repository, branch `main`,
main file `app.py`. The app peaks at about 0.6 GiB RAM, inside the free tier.

## Contents

| Path | What it is |
|---|---|
| `app.py` | Streamlit user interface |
| `fundmatch_core.py` | Model logic: input checks, prediction, explanation, search, award range |
| `assets/tfidf_lr.pkl` | TF-IDF vectoriser + logistic regression (scikit-learn 1.9.1) |
| `assets/index_tfidf_part0-7.npz` | TF-IDF search index, split into 8 files under 20 MiB each |
| `assets/projects_meta.parquet` | Project metadata shown in results (no researcher names) |
| `assets/model_info.json` | Test metrics, calibration temperature and other settings |

## Data and licence

Data from the UKRI Gateway to Research API, collected on 23 November 2026.
Contains public sector information licensed under the Open Government Licence v3.0
(https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Privacy

Titles and abstracts are processed in memory only and never stored or logged. No researcher names are shown.
