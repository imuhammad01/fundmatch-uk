# Model card — ScholarPulse UK

Format follows Mitchell et al. (2019), “Model Cards for Model Reporting”.

## Overview

ScholarPulse UK (package `fundmatch`) scores a research title and abstract against eight UKRI councils and retrieves similar funded Gateway to Research projects. The classifier is a class-weighted logistic regression on 200,000 sublinear word and bigram TF-IDF features, with temperature scaling. It was trained for CS5079 Applied AI at the University of Aberdeen and exported so that numpy logits match the original scikit-learn pipeline (`max_logit_diff <= 1e-6`).

- **Version:** 4.0.0
- **Snapshot:** 23 November 2026, 113,233 cleaned projects
- **Developer:** Md Shamim Mia, MSc Artificial Intelligence, University of Aberdeen (52536283)
- **Licence of the records:** Open Government Licence v3.0

## Intended users

MSc and PhD applicants, early-career researchers, and research-office staff who want a shortlist of councils, a band of similar past awards, and links to those projects. It is also a coursework deployment, so a marker can see the model, the calibration, and the memory budget.

## Out of scope

It does not predict whether a proposal will be funded. It does not write or edit the science. It does not rank or name researchers. Lecture 3 (reinforcement learning) is not used.

## Metrics (held-out CS5079 test, unchanged)

| Metric | Value |
| --- | --- |
| Macro-F1 | 0.8615 |
| Accuracy | 0.8693 |
| Top-3 accuracy | 0.9916 |
| ECE, raw → temperature-scaled | 0.0336 → 0.0180 |
| Temperature | 1.1314305006097425 |
| Accuracy when probability ≥ 0.80 | 0.943 |
| Weaker classes | BBSRC 0.780, MRC 0.795 |

## Factors and limits

Lead funder is a single label. Joint grants and interdisciplinary work are the main confusions (BBSRC with MRC, EPSRC with Innovate UK, AHRC with ESRC). Abstracts in the training set are of work that was funded, so the score is resemblance to funded text, not a chance of success.

## Conformal sets

The displayed 90% set is the top council only when its probability is at least 0.80, because that published confidence bin was 94.3% accurate. Otherwise, and always for the 95% set, the set is the top three councils, because top-3 test accuracy was 0.9916.

A separate split-conformal quantile was fit on 2,000 real project titles (250 per council, seed 42), using title + department + scheme because abstracts are not in the shipped table and the original test-row scores were not exported. Those title quantiles are near 1 and are stored in `artifacts/conformal_aps.json`. They are not used as the displayed set, and they are not a new coverage certificate for the 22,648-row test set.

## Novelty

Novelty compares the proposal’s maximum cosine in the 64-d abstract index with the same statistic on the 2,000 calibration titles. Bands are Core UKRI Territory (at or above the 40th percentile of those titles), Emerging Interdisciplinary Frontier (10th to 40th), and High-Novelty / Potential Out-of-Scope (below the 10th). A low score is not a judgement of quality.

## Quantisation

`artifacts/tfidf_lr_weights_int8.npz` is about 0.78 MB. On a 45-text check against the float model, top-1 agreement was 0.978 and the largest probability gap was 0.030. The float model remains the default.

## Student-stage tools (v3)

Working titles are composed from words that occur in the nearest real funded titles. Each one cites those projects. They are not grants. The funding wave is a ratio of project counts in the 200 nearest dated rows: the last three years over the three years before that. Surging means the ratio is at least 1.5, Rising at least 1.15, Steady at least 0.85, otherwise Mature, and the label is withheld when either window has fewer than three projects. The readiness score is an average of five documented rules (specificity, council alignment, distance from the 40th novelty percentile, presence of problem/method/data/impact words, and the size of the second council probability). The fusion score is the lower of the two abstract-space cosines.

## Ethics and privacy

Person endpoints were not collected. The neighbour table shows organisations, departments, awards and GtR links. User text is not written to disk or to the metrics endpoint. What-if suggestions tell the reader to add a phrase only if it is true.

## Caveats

Award bands are historical percentiles of similar funded projects after £0 studentships are excluded and counted. Real-terms figures use ONS CPIH L522, October 2026 index 144.6, because a full-year 2026 average is not in the 18 November 2026 release. This is not a UKRI quote. UKRI does not endorse the tool.
