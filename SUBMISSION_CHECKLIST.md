# ScholarPulse UK — final submission checklist

CS5079 Applied AI. Md Shamim Mia, Student ID 52536283, University of Aberdeen.

The display name is ScholarPulse UK. The Python package and the `fundmatch` command are unchanged. Version 4.0.0.

| Check | Result |
| --- | --- |
| Pure Python. No JS, TS, JSX, TSX, React, Next.js, or Node `package.json` | Required by `test_scholarpulse_brand_coach_redteam_hubs_and_dossier` |
| Zero `.pkl` files. Weights are `.npz`, metadata is `.json`, the corpus is `.parquet` and `.csv` | Required by the same test and by `test_v3_api_and_cli` |
| No researcher or PI names in `src/`, `app.py`, `README.md`, or this file. Universities and organisations only. The student author line is Md Shamim Mia, 52536283 | The test rejects `Prof`/`Dr` name patterns. The email template keeps the placeholder `[Professor / Supervisor Name]` |
| UKRI Open Government Licence v3.0 attribution on the dossier, the about page, and this README | Dossier footer and `README.md` |
| Honest calibration. Macro-F1 0.8615, accuracy 0.8693, top-3 0.9916, ECE 0.0336 → 0.0180, temperature 1.1314305006097425 | `MODEL_CARD.md` and the model lab |
| Corpus integrity. 113,233 projects, snapshot 23 November 2026, no personal columns | Engine refuses a table that contains `pi_name`, `researcher`, `orcid`, or `surname` |
| FastAPI title `ScholarPulse UK API`, version `4.0.0` | OpenAPI test |
| `POST /api/v1/coach`, `/api/v1/redteam`, `/api/v1/hubs`, `/api/v1/topics` warm under 150 ms | Latency assertions in pytest |
| Notebook `notebooks/01_fundmatch_pipeline_and_evaluation.ipynb` re-executed with saved outputs and zero errors | Re-run before the final commit |
| Streamlit hero clears the fixed header at 1440×900 and 1280×800. Demo pills do not ellipsize. Radar and hub labels are not clipped | Screenshots in the run artifacts |

Verified on this tree before the v4 commit: `pytest -q` reported 21 passed. The notebook executed with 0 error outputs. Warm median latency was 55.0 ms for `/api/v1/topics`, 51.7 ms for `/api/v1/coach`, 50.2 ms for `/api/v1/redteam`, and 73.3 ms for `/api/v1/hubs`.

Contains public sector information licensed under the Open Government Licence v3.0. This tool does not predict funding success and UKRI does not endorse it.
