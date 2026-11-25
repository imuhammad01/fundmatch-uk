"""Facts and result prose for the report (v3, concise). Numbers come from ../models/summary.json,
../../artifacts/model_info.json and verify_facts.json (app memory, tests, example output measured in the v2 test run);
the constants marked below were read from executed notebook outputs and collection logs."""
import json, os
R = os.path.dirname(os.path.abspath(__file__)); T = os.path.dirname(R)
S = json.load(open(f"{T}/models/summary.json")); I = json.load(open(f"{T}/../artifacts/model_info.json"))
L = json.load(open(f"{T}/logs/collection_log.json")); V = json.load(open(f"{R}/verify_facts.json"))
res = {r["model"]: r for r in S["results"]}; sel = {r["configuration"]: r for r in S["selection"]}
aw = {r["method"]: r for r in S["award"]}
tmp = {r["model"]: r["macro_F1 (train <2021, test >=2021)"] for r in S["temporal"]}
lr, svm = res["TF-IDF + LogReg"], res["TF-IDF + LinearSVM"]
D, E = sel["TF-IDF + LogReg"], sel["Ensemble: TF-IDF LR + BGE LR"]
pcf = I["per_class_f1"]; cal = S["calibration"]; fa = S["faithfulness"]
cc = {r["model"]: r for r in S["cross_country"]}; hy = {r["retriever"]: r for r in S["hybrid"]}
rm = {r["retriever"]: r for r in S["retrieval_metrics"]}; ar = {r["model"].split(" (alpha")[0]: r for r in S["award_regression"]}
n_intl = sum(int(r["n"]) for r in S["cross_by_source"]); lc = S["learning_curve"]; lk = S["leak"]
a30, a30s, ora = aw["TF-IDF: 30 neighbours, P25-P75"], aw["TF-IDF: 30 same-scheme neighbours, P25-P75"], aw["council-wide P25-P75 (oracle: true council)"]
MIN_CLASS, MIN_N = "STFC", 5511                       # notebook Section 3 output
N_ERR, SHARE_MENTIONS, CRD = 26, 0.11, 1.0            # collect_*_stdout.log; notebook Section 3.1
N_KQ, N_KC = 2000, 5000                              # notebook Section 7.2 output (known-item queries, corpus)
k = "macro_F1 on test texts that name a funder"
X = {
 "use": (
   "Researchers, PhD applicants, research offices and small firms often do not know which UK Research and Innovation "
   "(UKRI) council fits an idea, what similar work was funded, or for how much. GtR search and UKRI's Funding Finder do "
   "not read the user's abstract; NIH RePORTER Matchmaker does so only for US health research (NIH, 2026); Dimensions is "
   "paywalled (Hook et al., 2018); research prototypes classify GtR subjects (Tattershall and Yang, 2017) or map grants "
   "across funders (Khor et al., 2018). FundMatch UK takes a title and abstract and returns the likely council with "
   "calibrated probabilities and the words behind them, the most similar funded projects, and a typical award range. "
   "It does not predict success."),
 "data": (
   f"All {L['projects']['api_total_records']:,} Gateway to Research (GtR) projects, joined to {L['funds']['api_total_records']:,} "
   f"fund records (amounts, dates) and {L['organisations']['api_total_records']:,} organisations. Input: title and abstract; "
   "target: lead funder among eight councils (AHRC, BBSRC, EPSRC, ESRC, Innovate UK, MRC, NERC, STFC). For a "
   f"cross-country test we also collected {S['n_intl_raw']:,} US and Australian grants."),
 "sufficient": (
   f"After cleaning, {S['n_clean']:,} projects remain; the smallest class has {MIN_N:,} ({MIN_CLASS}). Macro-F1 rises from "
   f"{lc[0]['TF-IDF + LogReg']:.3f} with {lc[0]['n_train']:,} training projects to {lc[-1]['TF-IDF + LogReg']:.3f} with "
   f"{lc[-1]['n_train']:,}, and the last doubling adds only {100*(lc[-1]['TF-IDF + LogReg']-lc[-2]['TF-IDF + LogReg']):.1f} "
   "points (Figure 1): the data is large enough."),
 "ethics": (
   "Every record was fetched by our own code from public APIs; no Kaggle or prepared datasets. GtR is under the Open "
   "Government Licence v3.0 (credited in the app); NSF data is public domain; NIH's one-request-per-second guidance was "
   "respected; ARC data (CC BY-NC-ND 4.0) is used only for non-commercial evaluation. No GtR person records were "
   "collected and foreign investigator names were dropped; the app shows no researcher names and stores no user text."),
 "code": (
   "collect_gtr.py pages through /projects, /funds and /organisations (100 records per page), waits 1 s between requests, "
   "retries 429/5xx errors with exponential back-off and caches every page, so runs resume (Figure 2). The "
   f"{L['projects']['collected_utc'][:10]} run took {L['projects']['seconds']/60:.0f} min for projects and "
   f"{L['funds']['seconds']/60:.0f} min for funds and survived {N_ERR} transient errors. build_dataset.py joins funds to "
   "projects through FUNDED links (100% matched); collect_international.py applies the same policy to NSF, NIH and ARC."),
 "prep": (
   "Programme budgets (ISCF, GCRF…), which users cannot apply to, were removed, as were placeholders, short texts, "
   f"boilerplate and joint-grant duplicates (Table 1). Leakage controls: (i) {100*SHARE_MENTIONS:.0f}% of texts name a "
   f"funder, so names are masked; on those texts an unmasked model scores {lk[0][k]:.3f} against {lk[1][k]:.3f}; "
   f"(ii) grant category and reference reveal the funder (\"Collaborative R&D\" is {100*CRD:.0f}% Innovate UK), so they "
   f"are not features; (iii) near-duplicates form {S['n_groups']:,} groups, each kept on one side of the split. Features: "
   "TF-IDF uni/bi-grams (min_df 3, max_df 0.5, sublinear, ≤200k features) and frozen 384-d MiniLM and BGE sentence embeddings."),
 "models": (
   "This is imbalanced multi-class text classification. Candidates: a majority floor; Complement NB, built for imbalanced "
   "text (Rennie et al., 2003); logistic regression and a linear SVM on TF-IDF, strong text classifiers (Joachims, 1998), "
   "the former giving probabilities; logistic regression and an MLP on transformer embeddings (Reimers and Gurevych, 2019; "
   "Xiao et al., 2023), which capture meaning beyond shared words; and a TF-IDF + BGE ensemble. A CPU-budget SciBERT "
   "fine-tune (Beltagy et al., 2019) on 3,000 projects scored 0.784 macro-F1 against 0.825 for TF-IDF + LR on the same rows; "
   "a full run needs a GPU."),
 "training": (
   f"StratifiedGroupKFold gave {S['n_train']:,} training and {S['n_test']:,} test projects with no shared group. "
   "Hyper-parameters were tuned by macro-F1 in 3-fold group cross-validation on training data with balanced class "
   f"weights; the test set was used once (Table 2). TF-IDF + LR reached {lr['macro_F1']:.3f} macro-F1 with the right council "
   f"in its top three {100*lr['top3_acc']:.1f}% of the time; the SVM tied but gives no probabilities. Embeddings were weaker "
   "and the ensemble did not help: councils differ in specific words (\"clinical\", \"crop\", \"market\") that TF-IDF keeps. "
   f"Per-council F1 is {min(pcf.values()):.2f}–{max(pcf.values()):.2f}; the main confusions (BBSRC/MRC, EPSRC/Innovate UK, "
   f"AHRC/ESRC; Figure 3) mirror real overlaps. Trained before 2021 and tested after, F1 is {tmp['TF-IDF + LogReg']:.3f}, "
   "so yearly retraining is needed."),
 "beyond": (
   f"Calibration: raw probabilities were slightly over-confident (ECE {cal[0]['ECE']:.3f}; Guo et al., 2017). A temperature "
   f"(T = {S['temperature']:.2f}) fitted on out-of-fold training predictions cut ECE to {cal[1]['ECE']:.3f} without changing "
   "any prediction. Explanation: tfidf × coefficient gives each word's exact contribution, shown in the app. Deleting a "
   f"text's five top words lowered the predicted council's probability by {fa[0]['mean drop in predicted-council probability']:.2f} "
   f"and changed {100*fa[0]['prediction changed']:.0f}% of predictions, against {fa[1]['mean drop in predicted-council probability']:.2f} "
   f"and {100*fa[1]['prediction changed']:.0f}% for five random words, so explanations are faithful. Transfer: on {n_intl:,} "
   f"foreign grants mapped to UK councils (Table 3), accuracy falls to {cc['TF-IDF + LogReg']['accuracy']:.2f} but top-3 "
   f"stays at {100*cc['TF-IDF + LogReg']['top3_acc']:.0f}%; embeddings transfer slightly better (ensemble "
   f"{cc['Ensemble']['accuracy']:.2f}). Search: given {N_KQ:,} titles, TF-IDF ranked the project's own abstract "
   f"among {N_KC:,} in the top 10 for {100*rm['TF-IDF cosine']['Recall@10']:.1f}% (MRR {rm['TF-IDF cosine']['MRR@10']:.3f}); "
   f"hybrid rank fusion (Cormack et al., 2009) reached {100*rm['Hybrid RRF (TF-IDF + BGE)']['Recall@10']:.1f}% "
   f"({rm['Hybrid RRF (TF-IDF + BGE)']['MRR@10']:.3f}) but needs PyTorch; BM25 reached 91.6%, SPECTER 75.8%."),
 "deploy": (
   "A rule fixed before measuring: peak RAM under 1 GB and p95 latency under 1 s, then the best CV macro-F1 among "
   f"probabilistic models, preferring a lighter option within 0.01. TF-IDF + LR wins (CV {D['CV macro-F1']:.3f}; Figure 4). "
   f"The ensemble needs PyTorch, {E['peak RAM MB']:.0f} against {D['peak RAM MB']:.0f} MB and {E['latency ms (median)']:.0f} "
   f"against {D['latency ms (median)']:.0f} ms per query, for no gain; better transfer abroad does not justify exceeding the "
   f"690 MB free tier. TF-IDF search is also the most on-topic (P@10 {D['search P@10 topic']:.3f} against "
   f"{E['search P@10 topic']:.3f} for BGE), and one vectoriser serves prediction, search and explanation. Awards (MAE in log £, "
   f"{S['n_award_test']:,} test projects): the 30-neighbour median scores {ar['30 TF-IDF neighbours, P25-P75 (deployed)']['MAE log']:.3f}, "
   f"Ridge on TF-IDF {ar['Ridge: TF-IDF text only']['MAE log']:.3f}; LightGBM with council, scheme and duration "
   f"{ar['LightGBM: BGE + metadata, quantile P25-P75']['MAE log']:.3f}, but users rarely know these and its P25–P75 "
   f"quantiles held only {100*ar['LightGBM: BGE + metadata, quantile P25-P75']['coverage']:.0f}% of true awards, against "
   f"{100*ar['30 TF-IDF neighbours, P25-P75 (deployed)']['coverage']:.0f}% (ideal 50%) for the neighbour band the app keeps."),
 "how": (
   "The notebook refits TF-IDF + LR on all 113,233 clean projects and exports NumPy weights (float64 plus an int8 copy), "
   "a 64-d int8 SVD index of the abstracts and a metadata table, so the app needs neither the pickle nor PyTorch. "
   "src/fundmatch/engine.py rebuilds the vectoriser (Figure 5); a red-team check found and fixed a stop-word mismatch, and "
   "logits now match scikit-learn within 1e-5 on 3,070 texts. app.py is the Streamlit UI; the code is on GitHub and "
   "Streamlit Community Cloud serves it (Python 3.13, pinned requirements)."),
 "showcase": (
   "The live app, ScholarPulse UK (Figure 6), was opened on 25 November 2026 with a quantum-control idea. It returned EPSRC "
   "at 89% (high confidence: test predictions at or above 0.80 were right 94.3% of the time), a 90% conformal set of EPSRC "
   "alone, a novelty band, the words behind the score, Funding Finder links, similar GtR projects, a CPIH-adjustable award "
   "band and a trend chart. "
   f"{V['n_tests']} automated tests pass. The server holds about {V['rss_short']} once loaded but peaks near 819 MiB "
   "while starting, above the 690 MB minimum, so memory remains a risk."),
 "limits": (
   "The label is the lead funder only, so co-funded work counts as an error; search relevance rests on topic proxies and "
   "assistant-drafted labels that the group must still review; a third of "
   "projects (mostly studentships) have no award, so some bands rest on few neighbours, and the app warns when fewer than 10 "
   "are priced. Table 4 gives the roadmap."),
 "roadmap": [
   ["User study of search relevance", "Proxy labels only"],
   ["Multi-label councils; NIHR, Wellcome", "Co-funding missed"],
   ["Hybrid search on a ≥2 GB tier", "Better transfer (Table 3)"],
   ["Monthly data refresh and retraining", f"Temporal F1 {tmp['TF-IDF + LogReg']:.3f}"],
   ["Full SciBERT fine-tune on a GPU", "CPU pilot only (§3.2)"],
   ["EU (CORDIS), Canada (NSERC), Australia", "Cross-country pipeline exists"],
 ],
}
json.dump(X, open(f"{R}/extra_facts.json", "w"), indent=1); print("ok")
