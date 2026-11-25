"""Re-draw the deployment trade-off figure for the report (labels next to points, no overlapping legend).
Data: ../models/deployment_selection.csv written by the notebook (Section 9)."""
import pandas as pd, matplotlib.pyplot as plt
s = pd.read_csv("../models/deployment_selection.csv", index_col=0)
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=150)
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
for i, (n, r) in enumerate(s.iterrows()):
    ax.scatter(r["peak RAM MB"], r["test macro-F1"], s=30 + r["latency ms (median)"], color=CB[i % 6],
               edgecolor="k" if n == "TF-IDF + LogReg" else "none", linewidth=1.5, zorder=3)
    ax.annotate(f"{n} ({r['latency ms (median)']:.0f} ms)", (r["peak RAM MB"], r["test macro-F1"]), textcoords="offset points",
                xytext=(8, -12) if n.startswith("Ensemble") else (8, -3), fontsize=7)
ax.axvline(1000, ls="--", c="grey", lw=1); ax.text(1010, 0.80, "1 GB rule\nlimit", fontsize=7, color="grey")
ax.axvline(690, ls=":", c="#D55E00", lw=1); ax.text(545, 0.835, "690 MB free tier", fontsize=7, color="#D55E00")
ax.set_xlim(300, 1300); ax.set_xlabel("peak RAM of configuration (MB)"); ax.set_ylabel("test macro-F1"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("../figures/deployment_tradeoff_report.png"); print("ok")
