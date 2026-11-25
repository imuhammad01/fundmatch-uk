"""Build the Task 1 group report (docx + PDF) from saved results (v4, ScholarPulse UK).
Reads ../models/summary.json, ../logs/collection_log.json, extra_facts.json (python write_facts.py) and ../figures/*.png.

Usage:
  python write_facts.py && python make_report.py
  python make_report.py --app-url https://scholarpulse-uk.streamlit.app --github-url https://github.com/mueezd/applied_ai_a_1 \
                        --live-screenshot ../figures/app_live_scholarpulse.png
Environment variables APP_URL / GITHUB_URL / LIVE_SCREENSHOT work too. Needs python-docx and LibreOffice (PDF)."""
import argparse, json, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_helpers import new_doc, para, table, figure

ap = argparse.ArgumentParser()
ap.add_argument("--app-url", default=os.environ.get("APP_URL", "[APP LINK]"))
ap.add_argument("--github-url", default=os.environ.get("GITHUB_URL", "[GITHUB LINK]"))
ap.add_argument("--live-screenshot", default=os.environ.get("LIVE_SCREENSHOT"), help="PNG of the deployed app (replaces the local screenshot)")
A = ap.parse_args()

R = os.path.dirname(os.path.abspath(__file__)); T = os.path.dirname(R)
S = json.load(open(f"{T}/models/summary.json"))
X = json.load(open(f"{R}/extra_facts.json"))
res = {r["model"]: r for r in S["results"]}
sel = {r["configuration"]: r for r in S["selection"]}
f3 = lambda x: f"{float(x):.3f}"
COUNT = {"body": [], "captions": [], "tables": []}
W = lambda t: len(re.findall(r"[A-Za-z0-9£%×≤][\w'’.,/%£×≤–+-]*", t))

def P(d, text, **kw):
    COUNT["body"].append(text); return para(d, text, **kw)
def H(d, text, level):
    COUNT["body"].append(text); return d.add_heading(text, level)
def TAB(d, header, rows, caption, **kw):
    COUNT["tables"] += header + [c for r in rows for c in r]; COUNT["captions"].append(caption)
    t = table(d, header, rows, caption=caption, **kw)
    for row in t.rows[:-1]:                       # keep the table on one page
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True
    return t
def FIG(d, path, width, caption):
    COUNT["captions"].append(caption); return figure(d, path, width, caption)

d = new_doc()
para(d, "ScholarPulse UK (FundMatch UK): matching research ideas to UKRI funders", size=17, align="center")
para(d, "CS5079 Applied Artificial Intelligence, Assessment 1, Task 1 (group report)", size=10.5, align="center")
para(d, "Group: Md Shamim Mia (52536283), [Member 2 name (ID)], [Member 3 name (ID)], [Member 4 name (ID)]", size=10, align="center")
para(d, f"Deployed app: {A.app_url}   |   Code: {A.github_url}", size=10, align="center")
WC = para(d, "Word count: __WC__", size=9, italic=True, align="center")

H(d, "1. Project definition", 1)
H(d, "1.1 Usefulness", 2); P(d, X["use"])
H(d, "1.2 Data used", 2); P(d, X["data"])
H(d, "1.3 Why the data is sufficient", 2); P(d, X["sufficient"])
FIG(d, f"{T}/figures/learning_curve.png", 8, "Figure 1. Learning curve (test macro-F1).")

H(d, "2. Data collection", 1)
H(d, "2.1 Own sources, licences and ethics", 2); P(d, X["ethics"])
H(d, "2.2 Own collection code", 2); P(d, X["code"])
FIG(d, f"{T}/figures/code_collect.png", 15, "Figure 2. Core of our collector (full code: notebook Section 1).")

H(d, "3. Product creation", 1)
H(d, "3.1 Pre-processing and feature engineering", 2)
short = {"raw projects from API": "Raw API records", "keep 8 councils (drop programme funders)": "Keep 8 councils",
         "drop placeholder abstracts": "Drop placeholders", "drop abstracts < 30 words": "Drop <30 words",
         "drop boilerplate reused by >= 5 titles": "Drop boilerplate", "drop duplicates with conflicting funders": "Drop conflicting duplicates",
         "drop exact duplicates (joint grants)": "Drop joint-grant duplicates"}
TAB(d, ["Step", "Rows left"], [[short.get(r["step"], r["step"]), f"{r['rows left']:,}"] for r in S["clean_table"]],
    "Table 1. Cleaning (notebook Section 3).", widths=[6, 3])
P(d, X["prep"])
H(d, "3.2 Candidate models", 2); P(d, X["models"])
H(d, "3.3 Training process and results", 2); P(d, X["training"])
names = [("Majority", "Majority class"), ("TF-IDF+CNB", "TF-IDF + ComplementNB"), ("TF-IDF+LR", "TF-IDF + LogReg"),
         ("TF-IDF+SVM", "TF-IDF + LinearSVM"), ("MiniLM+LR", "MINILM emb + LogReg"), ("BGE+LR", "BGE emb + LogReg"),
         ("BGE+MLP", "BGE emb + MLP"), ("Ensemble", "Ensemble: TF-IDF LR + BGE LR")]
rows = []
for short_n, m in names:
    r, s = res[m], sel.get(m)
    cv = "–" if r["cv_macroF1"] != r["cv_macroF1"] or r["cv_macroF1"] is None else f"{r['cv_macroF1']:.3f}±{r['cv_sd']:.3f}"
    rows.append([short_n, cv, f3(r["macro_F1"]), f3(r["accuracy"]), f3(r["top3_acc"]),
                 f"{s['peak RAM MB']:.0f}" if s else "–", f"{s['latency ms (median)']:.0f}" if s else "–"])
TAB(d, ["Model", "CV F1", "Test F1", "Acc.", "Top-3", "RAM MB", "ms"], rows,
    "Table 2. Held-out results; RAM and latency per query.", widths=[3, 3, 1.8, 1.8, 1.8, 1.8, 1.4], highlight_row=2)
FIG(d, f"{T}/figures/confusion_matrix.png", 7.5, "Figure 3. Confusion matrix, TF-IDF+LR (test).")
H(d, "3.4 Beyond accuracy", 2); P(d, X["beyond"])
src = {"NSF": "NSF (US)", "NIH": "NIH (US)", "ARC": "ARC (AU)"}
TAB(d, ["Test set", "n", "TF-IDF", "BGE", "Top-3"],
    [[src[r["source"]], f"{r['n']:,}", f3(r["TF-IDF LR acc"]), f3(r["BGE LR acc"]), f3(r["TF-IDF LR top3"])] for r in S["cross_by_source"]] +
    [["UK test", f"{S['n_test']:,}", f3(res['TF-IDF + LogReg']['accuracy']), f3(res['BGE emb + LogReg']['accuracy']),
      f3(res['TF-IDF + LogReg']['top3_acc'])]],
    "Table 3. Accuracy of unchanged UK models abroad.", widths=[3, 2, 2, 2, 2])
H(d, "3.5 Why this model is deployed", 2); P(d, X["deploy"])
FIG(d, f"{T}/figures/deployment_tradeoff_report.png", 11, "Figure 4. Quality against memory (bubble size: latency).")

H(d, "4. Deployment", 1)
H(d, "4.1 Process", 2); P(d, X["how"])
FIG(d, f"{T}/figures/code_query.png", 15, "Figure 5. One query in the app (simplified).")
H(d, "4.2 Showcase", 2); P(d, X["showcase"])
shot = A.live_screenshot or f"{T}/figures/app_live_scholarpulse.png"
FIG(d, shot, 12, "Figure 6. The live ScholarPulse UK app, 25 November 2026, quantum-control example.")
H(d, "4.3 Link", 2); P(d, f"App: {A.app_url}. Code: {A.github_url}.")

H(d, "5. Critical evaluation and future work", 1); P(d, X["limits"])
TAB(d, ["Next step", "Evidence"], X["roadmap"], "Table 4. Roadmap.", widths=[7, 5])

d.add_heading("References", 1)
REFS = [
    "Beltagy, I., Lo, K. and Cohan, A. (2019) 'SciBERT: a pretrained language model for scientific text', Proc. EMNLP-IJCNLP, pp. 3615–3620.",
    "Cormack, G.V., Clarke, C.L.A. and Büttcher, S. (2009) 'Reciprocal rank fusion outperforms Condorcet and individual rank learning methods', Proc. SIGIR, pp. 758–759.",
    "Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) 'On calibration of modern neural networks', Proc. ICML, pp. 1321–1330.",
    "Hook, D.W., Porter, S.J. and Herzog, C. (2018) 'Dimensions: building context for search and evaluation', Frontiers in Research Metrics and Analytics, 3, 23.",
    "Joachims, T. (1998) 'Text categorization with support vector machines', Proc. ECML, pp. 137–142.",
    "Khor, K.A., Ko, G. and Walter, T. (2018) 'Applying machine learning to compare research grant programs', Proc. STI 2018, Leiden: CWTS, pp. 816–824.",
    "NIH (2026) RePORTER Matchmaker. Available at: https://reporter.nih.gov/matchmaker (Accessed: 23 November 2026).",
    "Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', Journal of Machine Learning Research, 12, pp. 2825–2830.",
    "Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: sentence embeddings using Siamese BERT-networks', Proc. EMNLP-IJCNLP, pp. 3982–3992.",
    "Rennie, J.D.M., Shih, L., Teevan, J. and Karger, D.R. (2003) 'Tackling the poor assumptions of naive Bayes text classifiers', Proc. ICML, pp. 616–623.",
    "Streamlit (2026) Community Cloud: resource limits. Available at: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app (Accessed: 23 November 2026).",
    "Tattershall, E. and Yang, E. (2017) A machine learning approach to the classification of technical abstracts in a two-level ontology. RAL-TR-2017-004. STFC. doi:10.5286/raltr.2017004.",
    "UKRI (2026) Gateway to Research API 2. Available at: https://gtr.ukri.org/resources/api.html (Accessed: 23 November 2026). Open Government Licence v3.0.",
    "Wang, W. et al. (2020) 'MiniLM: deep self-attention distillation for task-agnostic compression of pre-trained transformers', NeurIPS 33.",
    "Xiao, S., Liu, Z., Zhang, P. and Muennighoff, N. (2023) 'C-Pack: packaged resources to advance general Chinese embedding' (BGE models), arXiv:2309.07597.",
    "Data: NSF Awards API (https://api.nsf.gov, public domain); NIH RePORTER API (https://api.reporter.nih.gov); ARC Grants API (https://dataportal.arc.gov.au/NCGP/API/grants, CC BY-NC-ND 4.0). Accessed 23 November 2026.",
]
for ref in REFS:
    para(d, ref, size=8.5)
END = ("Code sources: no code was copied; scikit-learn, sentence-transformers, pandas and Streamlit were used through their "
       "documented APIs (cited in the notebook and app). Generative AI use: [state the group's use here, as the course policy requires].")
para(d, END, size=8.5, italic=True)

c = {k: sum(W(t) for t in v) for k, v in COUNT.items()}
total = c["body"] + c["captions"] + c["tables"]
for r in WC.runs:
    r.text = r.text.replace("__WC__", f"{total} (headings, body, captions and tables; excluding title block, code images and references)")
out = f"{R}/CS5079_Task1_Group_Report.docx"; d.save(out)
title_w = sum(W(t) for t in [p.text for p in d.paragraphs[:4]]); refs_w = sum(W(t) for t in REFS + [END])
print(f"body+headings {c['body']} | +captions {c['body']+c['captions']} | +tables {total} | title {title_w} | refs {refs_w}")
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", R, out], check=True, capture_output=True)
print("pdf:", out.replace(".docx", ".pdf"))
