"""ScholarPulse UK — Streamlit research command deck. Pure Python. Abstracts are not stored."""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fundmatch.brand import APP_DISPLAY_NAME, APP_EDITION, APP_TAGLINE
from fundmatch.futuristic import research_dossier
from fundmatch.cpih import CAVEAT, SOURCE_URL, TARGET_LABEL
from fundmatch.demos import DEMOS
from fundmatch.engine import COUNCIL_COLOURS, COUNCIL_INFO, DISCLAIMER, IFS, OGL, get_engine
from fundmatch.student import PERSONAS

st.set_page_config(page_title=APP_DISPLAY_NAME, page_icon="◈", layout="wide")
ENGINE = get_engine()

GLOSSARY = [
    ("Calibration", "A calibrated probability of 0.80 should be right about 80% of the time. Temperature scaling moved this model's test ECE from 0.0336 to 0.0180."),
    ("ECE", "Expected calibration error. Lower means the stated probabilities match how often the council was actually right."),
    ("Conformal set", "The shortlist of councils that the published test profile can support. A singleton appears only when the probability is at least 0.80, because that bin was 94.3% accurate. Otherwise the set is the top three, because top-3 accuracy was 0.9916."),
    ("SVD", "A 64-number summary of each funded abstract. Your text is placed in the same map so closeness is a cosine, not a keyword count."),
    ("RRF", "Reciprocal rank fusion. A project ranks highly if it is near your text in the abstract map or shares exact title words, or both."),
    ("Int8 quantisation", "The council weights are stored as small integers. The file is 0.78 MB instead of 6.4 MB. The live toggle shows whether the top council changes."),
]

LECTURES = [
    ("L1", "sklearn pipelines and grouped validation", "used"),
    ("L2", "TF-IDF, cleaning, SVD / PCA map", "used"),
    ("L3", "Reinforcement learning", "out of scope — no policy over people"),
    ("L4", "Temperature scaling, linear attribution, deployment", "used"),
    ("L5", "Word and bigram logistic regression", "used"),
    ("L6", "Frozen MiniLM and BGE embeddings", "compared, not deployed"),
    ("L7", "Linear SVM", "compared; no probabilities, not deployed"),
    ("L8", "int8 weights, memory budget, cascades", "used"),
]

st.markdown(
    """
    <style>
    [data-testid="stHeader"] {
      background: rgba(5, 8, 16, 0.55) !important;
      backdrop-filter: blur(10px);
      height: 2.6rem;
    }
    [data-testid="stAppViewContainer"], .stApp {
      background:
        radial-gradient(900px 420px at 8% -10%, rgba(94, 234, 212, 0.16), transparent 55%),
        radial-gradient(800px 480px at 100% 0%, rgba(167, 139, 250, 0.16), transparent 50%),
        radial-gradient(700px 400px at 70% 110%, rgba(16, 185, 129, 0.10), transparent 45%),
        #07090f;
      color: #e7eef6;
    }
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewContainer"] > .main .block-container,
    .block-container {
      padding-top: 5.5rem !important;
      padding-bottom: 3rem;
      max-width: 1180px;
    }
    .fm-kicker {letter-spacing: .18em; text-transform: uppercase; font-size: .72rem; color: #9fb0c3;}
    .fm-card, .fm-hero {
      border: 1px solid rgba(94, 234, 212, 0.35);
      border-radius: 18px;
      padding: 16px 18px;
      background: rgba(12, 18, 28, 0.72);
      box-shadow: 0 0 0 1px rgba(167, 139, 250, 0.08), 0 18px 50px rgba(0, 0, 0, 0.28);
      margin-bottom: 12px;
    }
    .fm-hero {margin-top: 0.35rem;}
    .fm-kpis {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;}
    .fm-kpi, .fm-stage {
      border: 1px solid rgba(94, 234, 212, 0.45);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.82rem;
      color: #e7eef6;
      background: rgba(94, 234, 212, 0.10);
    }
    .fm-stage {
      display: inline-block;
      margin: 0.15rem 0 0.7rem;
      border-color: rgba(196, 181, 253, 0.75);
      color: #f3e8ff;
      background: rgba(167, 139, 250, 0.16);
      box-shadow: 0 0 22px rgba(167, 139, 250, 0.28);
    }
    .fm-high {border-color: #fb7185; color: #ffe4e6; background: rgba(244, 63, 94, 0.16);}
    .fm-watch {border-color: #fbbf24; color: #fef3c7; background: rgba(245, 158, 11, 0.16);}
    .fm-solid {border-color: #34d399; color: #d1fae5; background: rgba(16, 185, 129, 0.16);}
    div.stButton > button {
      border-radius: 999px;
      border: 1px solid rgba(231, 238, 246, 0.18);
      background: rgba(16, 22, 32, 0.9);
      color: #e7eef6;
      white-space: nowrap !important;
      overflow: visible !important;
      text-overflow: clip !important;
      min-height: 2.5rem;
      padding: 0.35rem 0.7rem;
      font-size: 0.86rem;
    }
    div.stButton > button p {
      white-space: nowrap !important;
      overflow: visible !important;
      text-overflow: clip !important;
    }
    div.stButton > button:hover {border-color: #5eead4; color: #5eead4;}
    div.stButton > button[kind="primary"] {
      background: linear-gradient(90deg, #0f766e, #5eead4);
      color: #04120f;
      border: none;
      font-weight: 650;
    }
    mark {border-radius: 3px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"£{value:,.0f}"


def _money_short(value: float | None) -> str:
    """Compact pounds for metric tiles, so values are not cut off ("£191,..." on narrow columns)."""
    if value is None:
        return "—"
    if abs(value) >= 1_000_000:
        return f"£{value / 1_000_000:.2f}m"
    if abs(value) >= 1_000:
        return f"£{value / 1_000:.0f}k"
    return f"£{value:,.0f}"


def _paint(text: str, highlights: list) -> str:
    if not text:
        return ""
    pieces: list[str] = []
    cursor = 0
    for mark in sorted(highlights, key=lambda item: item.start):
        if mark.start < cursor or mark.end > len(text):
            continue
        pieces.append(html.escape(text[cursor:mark.start]))
        colour = "rgba(0,158,115,0.45)" if mark.polarity == "positive" else "rgba(213,94,0,0.45)"
        tip = html.escape(f"{mark.feature}: {mark.contribution:+.3f}")
        pieces.append(f'<mark title="{tip}" style="background:{colour};color:inherit">{html.escape(text[mark.start:mark.end])}</mark>')
        cursor = mark.end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def render_match(result, expert: bool) -> None:
    for warning in result.warnings:
        st.warning(warning)
    if not result.ok:
        for error in result.errors:
            st.error(error)
        return
    st.markdown('<div class="fm-card">', unsafe_allow_html=True)
    st.markdown(f"### {result.top_council}")
    st.caption(f"{COUNCIL_INFO[result.top_council]} · {result.latency_ms:.0f} ms · {result.precision} · entropy {result.entropy:.2f}")
    if result.confidence == "low":
        st.error(result.confidence_note)
    elif result.confidence == "moderate":
        st.warning(result.confidence_note)
    else:
        st.success(result.confidence_note)
    st.markdown(
        f"**90% set:** {', '.join(result.conformal_set_90)}  ·  **95% set:** {', '.join(result.conformal_set_95)}"
    )
    st.caption(result.conformal_explanation)
    if result.novelty:
        st.info(f"{result.novelty['band']}. {result.novelty['note']}")
    st.markdown("</div>", unsafe_allow_html=True)
    if not expert:
        st.markdown(f'<div class="fm-card"><h3>Student coach</h3>{html.escape(result.student_summary)}</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 0.85])
    with left:
        frame = pd.DataFrame([{"council": row.council, "probability": row.probability} for row in result.councils])
        chart = px.bar(frame, x="probability", y="council", orientation="h", color="council", color_discrete_map=COUNCIL_COLOURS)
        chart.update_layout(showlegend=False, height=320, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7eef6"), xaxis=dict(range=[0, 1], tickformat=".0%", title="Calibrated probability"), yaxis=dict(title="UKRI council", categoryorder="total ascending"))
        chart.update_traces(hovertemplate="%{y}: %{x:.1%} calibrated probability<extra></extra>")
        st.plotly_chart(chart, width="stretch")
        links = " · ".join(f"[{row.council} calls]({row.funding_finder_url})" for row in result.councils[:3])
        st.markdown(f"Live UKRI Funding Finder: {links} · [Innovation Funding Service]({IFS})")
    with right:
        band = result.award
        st.markdown("### Award band")
        if band and band.median is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("P25", _money_short(band.p25), help=_money(band.p25))
            c2.metric("Median", _money_short(band.median), help=_money(band.median))
            c3.metric("P75", _money_short(band.p75), help=_money(band.p75))
            if band.n_positive < 10:
                st.warning(
                    f"Only {band.n_positive} of the {band.neighbours} nearest projects have a priced award, "
                    "so this band rests on very few grants. Treat it as a rough guide."
                )
            st.caption(f"P10–P90 {_money(band.p10)} – {_money(band.p90)} · {band.basis} · {band.n_positive} priced of {band.neighbours}")
            st.caption(band.zero_note)
            st.caption(band.caveat)
            st.caption(f"{TARGET_LABEL}. [ONS L522]({SOURCE_URL}).")
        else:
            st.info("Not enough priced awards in this neighbour set.")
    if result.explanation:
        st.markdown(f"### X-Ray · why {result.explanation.council}")
        joined = f"{(st.session_state.get('title') or '').strip()}. {(st.session_state.get('abstract') or '').strip()}".strip(". ")
        st.markdown(_paint(joined, result.explanation.highlights), unsafe_allow_html=True)
        if result.contrast:
            st.caption(
                f"Contrast {result.contrast.council_a} vs {result.contrast.council_b}. "
                f"Toward {result.contrast.council_a}: " + ", ".join(item.feature for item in result.contrast.toward_a[:5])
                + ". Toward " + result.contrast.council_b + ": " + ", ".join(item.feature for item in result.contrast.toward_b[:5]) + "."
            )
    hosts, schemes = st.columns(2)
    with hosts:
        st.markdown("### Host organisations")
        if result.host_institutions:
            st.dataframe(pd.DataFrame(result.host_institutions), hide_index=True, width="stretch")
        st.caption("Organisations only. No researcher names.")
    with schemes:
        st.markdown("### Scheme fit")
        if result.scheme_fit:
            st.dataframe(pd.DataFrame(result.scheme_fit), hide_index=True, width="stretch")
    if result.trends:
        trend = pd.DataFrame([row.model_dump() for row in result.trends])
        fig = px.bar(trend, x="year", y="total_gbp", color="council", color_discrete_map=COUNCIL_COLOURS,
                     labels={"year": "Start year", "total_gbp": "Total awarded (£)", "council": "Council"})
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7eef6"), legend_title="")
        st.markdown("### Topic funding by year")
        st.plotly_chart(fig, width="stretch")
    if result.similar:
        st.markdown("### Similar funded projects")
        columns = {
            "similarity": [row.similarity for row in result.similar],
            "title": [row.title for row in result.similar],
            "council": [row.lead_funder for row in result.similar],
            "scheme": [row.grant_category for row in result.similar],
            "award (£)": [f"£{row.award_gbp:,.0f}" if row.award_gbp > 0 else "£0 · not priced" for row in result.similar],
            "start": [row.fund_start for row in result.similar],
            "end": [row.fund_end for row in result.similar],
            "organisation": [row.lead_org_name for row in result.similar],
            "GtR": [row.gtr_url for row in result.similar],
        }
        if expert:
            columns["semantic"] = [row.svd_cosine for row in result.similar]
            columns["lexical"] = [row.lexical_score for row in result.similar]
            columns["shared"] = [", ".join(row.shared_terms) for row in result.similar]
        st.dataframe(
            pd.DataFrame(columns), hide_index=True, width="stretch",
            column_config={
                "GtR": st.column_config.LinkColumn("GtR", display_text="open"),
                "award (£)": st.column_config.TextColumn("award (£)", help="£0 · not priced means GtR records £0: studentships and other unpriced records."),
            },
        )
        st.caption("“£0 · not priced” means GtR records £0 for that project (mostly studentships); these are left out of the award band. Organisations only, no researcher names.")
        st.download_button("Download CSV", ENGINE.to_csv(result.similar), "fundmatch-similar.csv", "text/csv", key="csv")
        st.download_button("Download Markdown brief", ENGINE.to_markdown(st.session_state.get("title", ""), result), "fundmatch-brief.md", "text/markdown", key="md")


def tab_matcher(expert: bool) -> None:
    st.markdown('<p class="fm-kicker">Mission control · UKRI funding intelligence</p>', unsafe_allow_html=True)
    st.markdown("## Match a proposal")
    title = st.session_state.get("title") or ""
    abstract = st.session_state.get("abstract") or ""
    if not expert:
        st.caption("The idea above is the text being scored. Read the council shortlist, then open the similar funded projects.")
    else:
        st.caption(f"113,233 projects · T = {ENGINE.temperature:.4f} · RSS {ENGINE.rss_mb():.0f} MB · abstracts are not stored.")
    words = len(f"{title.strip()}. {abstract.strip()}".split())
    st.caption(f"{words} words" + (" · enough to score" if words >= 30 else " · at least 30 words"))
    c1, c2, c3, c4 = st.columns(4)
    schemes = ["Research Grant"] + [row for row in ENGINE.model_info.get("grant_categories", []) if row != "Research Grant"]
    scheme = c1.selectbox("Grant scheme", schemes, key="scheme")
    same = c2.toggle("Same scheme", key="same_scheme")
    cpih = c3.toggle("CPIH 2026 pounds", key="cpih")
    top_k = c4.slider("Similar projects", 5, 30, 10, key="top_k")
    quantized = False
    if expert:
        quantized = st.toggle("Score with int8 weights", key="quantized", help="0.78 MB integer weights instead of the float model.")
    councils = st.multiselect("Council filter", list(COUNCIL_INFO), default=list(COUNCIL_INFO), key="councils")
    year_min, year_max = int(ENGINE.frame["start_year"].min()), int(ENGINE.frame["start_year"].max())
    years = st.slider("Start year", year_min, year_max, (year_min, year_max), key="years")
    compare = st.selectbox("Contrast the leader with", ["runner-up"] + list(COUNCIL_INFO), key="compare")
    go = st.button("Analyze proposal", type="primary", key="analyze")
    if not expert:
        with st.expander("AI and viva glossary"):
            for term, meaning in GLOSSARY:
                st.markdown(f"**{term}.** {meaning}")
    ready = bool((title or "").strip() or (abstract or "").strip())
    if go:
        st.session_state["_scored"] = True
    if ready and st.session_state.get("_scored"):
        result = ENGINE.analyze(
            title or "", abstract or "", councils=councils or None,
            year_min=int(years[0]), year_max=int(years[1]),
            scheme=scheme if same else None, same_scheme=same, cpih=cpih,
            top_k=int(top_k), compare=None if compare == "runner-up" else compare,
            quantized=bool(quantized),
        )
        render_match(result, expert)
    elif not ready:
        st.info("Paste a title and abstract, or load one of the demos.")


def tab_sky() -> None:
    st.markdown("## Where this idea sits")
    st.caption("The map is a 2D view of the 64-d abstract index. Landmarks are real funded projects. Centroids are the mean of each council's landmarks.")
    title = st.session_state.get("title") or ""
    abstract = st.session_state.get("abstract") or ""
    result = ENGINE.analyze(title, abstract, top_k=10) if (title or abstract) else None
    if result is None or not result.ok:
        st.info("Score a proposal on the matcher tab first. The map uses that text.")
    if result is not None and result.ok:
        frame = pd.DataFrame(result.constellation)
        frame["label"] = frame.apply(lambda row: row["title"] or row["council"] or row["kind"], axis=1)
        fig = px.scatter(
            frame, x="x", y="y", color="council", symbol="kind", hover_name="label",
            hover_data={"organisation": True, "award_gbp": True, "gtr_url": True, "kind": True, "x": False, "y": False},
            color_discrete_map=COUNCIL_COLOURS,
        )
        fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,9,15,0.35)", font=dict(color="#e7eef6"), legend_title="", margin=dict(l=40, r=24, t=36, b=40))
        st.plotly_chart(fig, width="stretch")
    st.markdown("## What-if remixer")
    st.caption("Only add a phrase if it is true of the work. A higher council probability is not a reason to overstate the science.")
    target = st.selectbox("Target council", list(COUNCIL_INFO), key="what_target")
    add = st.text_input("Phrases to add, if they are true", key="what_add", placeholder="comma-separated phrases")
    remove = st.text_input("Phrases to remove", key="what_remove")
    if st.button("Simulate edit", key="what_go"):
        st.session_state["what_result"] = ENGINE.simulate_what_if(title, abstract, add_terms=add, remove_terms=remove, target_council=target)
        st.session_state["what_suggest"] = ENGINE.suggest_counterfactuals(title, abstract, target)
    suggestion = st.session_state.get("what_suggest")
    if suggestion and suggestion.get("target_council") == target:
        st.write("Phrases in your text that pull away:", ", ".join(suggestion["opposing_phrases"]) or "none of the strongest ones")
        st.write("Phrases common in real", target, "titles:", ", ".join(suggestion["suggested_additions"]))
    simulated = st.session_state.get("what_result")
    if simulated:
        chart = pd.DataFrame(simulated["deltas"])
        fig = go.Figure()
        fig.add_bar(name="before", x=chart["council"], y=chart["before"], marker_color="#56B4E9")
        fig.add_bar(name="after", x=chart["council"], y=chart["after"], marker_color="#009E73")
        fig.update_layout(barmode="group", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7eef6"), yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")


def tab_compare() -> None:
    st.markdown("## Compare ideas")
    picked = st.multiselect("Demos", [item["label"] for item in DEMOS], default=[DEMOS[0]["label"], DEMOS[1]["label"]], key="arena")
    if st.button("Compare selected", key="arena_go") and len(picked) >= 2:
        proposals = [item for item in DEMOS if item["label"] in picked]
        st.session_state["arena_rows"] = ENGINE.compare_proposals(proposals)["rows"]
    rows = st.session_state.get("arena_rows")
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame, hide_index=True, width="stretch")
        if frame["probability"].notna().any():
            fig = px.bar(frame, x="label", y="probability", color="top_council", color_discrete_map=COUNCIL_COLOURS)
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7eef6"), yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
        st.download_button("Download comparison CSV", frame.to_csv(index=False).encode(), "fundmatch-compare.csv", "text/csv", key="arena_csv")
    st.markdown("## Batch")
    upload = st.file_uploader("CSV with title and abstract columns", type=["csv"], key="batch_file")
    if upload is not None and st.button("Score batch", key="batch_go"):
        frame = pd.read_csv(upload)
        result = ENGINE.analyze_batch(frame.to_dict(orient="records"))
        out = pd.DataFrame(result["rows"])
        st.dataframe(out, hide_index=True, width="stretch")
        st.download_button("Download batch CSV", out.to_csv(index=False).encode(), "fundmatch-batch.csv", "text/csv", key="batch_csv")


def tab_lab(expert: bool) -> None:
    st.markdown("## CS5079 model lab")
    info = ENGINE.model_info
    a, b, c, d = st.columns(4)
    a.metric("Macro-F1", f"{info['test_macro_f1']:.4f}")
    b.metric("Accuracy", f"{info['test_accuracy']:.4f}")
    c.metric("Top-3", f"{info['test_top3']:.4f}")
    d.metric("RSS now", f"{ENGINE.rss_mb():.0f} MB")
    st.dataframe(pd.DataFrame(info["calibration"]), hide_index=True, width="stretch")
    per = pd.DataFrame([{"council": k, "f1": v} for k, v in info["per_class_f1"].items()])
    fig = px.bar(per, x="council", y="f1", color="council", color_discrete_map=COUNCIL_COLOURS)
    fig.update_layout(showlegend=False, yaxis_range=[0.7, 1], height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7eef6"))
    st.plotly_chart(fig, width="stretch")
    st.caption("BBSRC 0.780 and MRC 0.795 are the weaker classes. EPSRC and Innovate UK are the usual borderline pair.")
    st.dataframe(pd.read_csv(ROOT / "artifacts" / "deployment_selection.csv"), hide_index=True, width="stretch")
    if st.button("Compare float and int8 on the quantum demo", key="quant_go"):
        demo = DEMOS[0]
        fp = ENGINE.analyze(demo["title"], demo["abstract"])
        iq = ENGINE.analyze(demo["title"], demo["abstract"], quantized=True)
        st.session_state["quant_pair"] = (fp, iq)
    pair = st.session_state.get("quant_pair")
    if pair:
        fp, iq = pair
        st.write(f"Float top {fp.top_council} {fp.councils[0].probability:.1%} in {fp.latency_ms:.0f} ms. Int8 top {iq.top_council} {iq.councils[0].probability:.1%} in {iq.latency_ms:.0f} ms.")
    st.markdown("### Lectures 1–8")
    st.dataframe(pd.DataFrame(LECTURES, columns=["Lecture", "Technique", "In FundMatch"]), hide_index=True, width="stretch")
    if expert:
        st.json(ENGINE.metrics())


def tab_corpus() -> None:
    stats = ENGINE.corpus_stats()
    st.markdown(f"## {stats['projects']:,} funded projects")
    st.caption(f"Snapshot {stats['snapshot']}. Nominal total £{stats['total_gbp']/1e9:.2f} billion. £0 awards: {stats['zero_awards']:,} ({stats['zero_share']:.1%}).")
    councils = pd.DataFrame(stats["councils"])
    fig = px.bar(councils, x="council", y="projects", color="council", color_discrete_map=COUNCIL_COLOURS)
    fig.update_layout(showlegend=False, height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7eef6"))
    st.plotly_chart(fig, width="stretch")
    probe = st.text_input("Semantic probe", key="probe", placeholder="A short research phrase")
    if probe and len(probe.split()) >= 3:
        rows = ENGINE.search(probe, top_k=8)
        st.dataframe(pd.DataFrame([{"similarity": r.similarity, "title": r.title, "council": r.lead_funder, "organisation": r.lead_org_name} for r in rows]), hide_index=True, width="stretch")
    st.markdown("## About, ethics, and the limits")
    st.markdown(
        f"""
{DISCLAIMER}

**Data.** {ENGINE.model_info['n_projects_index']:,} cleaned UKRI projects, collected {str(ENGINE.model_info['data_collected'])[:10]}. Person endpoints were not called. {OGL}

**Model.** TF-IDF word and bigram features with class-weighted logistic regression, chosen over Linear SVM, Complement NB, MiniLM/BGE embedding models and an ensemble for accuracy, speed and memory (see the deployment table above). Temperature T = {ENGINE.temperature:.4f}. Held-out macro-F1 {ENGINE.model_info['test_macro_f1']}, accuracy {ENGINE.model_info['test_accuracy']}, top-3 {ENGINE.model_info['test_top3']} on 22,648 test projects. ECE 0.0336 → 0.0180.

**Where it fails.** Weakest councils: BBSRC (F1 {ENGINE.model_info['per_class_f1']['BBSRC']}) and MRC (F1 {ENGINE.model_info['per_class_f1']['MRC']}), which share biomedical vocabulary; EPSRC and Innovate UK are the usual borderline pair. When the top probability is under 0.60, held-out accuracy was about 52% or worse, so the app shows a low-confidence warning. Projects that start after 2021 score lower (time-based macro-F1 0.827). Grants from other countries transfer less well (macro-F1 0.623).

**Speed and availability.** A warm query takes about 0.1–0.2 s on the server. The free host sleeps after 12 hours without visits; click "Yes, get this app back up!" and wait about a minute.

**Privacy.** Pasted abstracts stay in memory for the request. They are not logged. The neighbour table has organisations, not people.

**Money.** {CAVEAT} {TARGET_LABEL}.

**Out of scope.** No success prediction, no proposal writing, no researcher ranking. Lecture 3 is not used.
"""
    )
    rows = [
        ("FR-19", "Minimum length and English warning"),
        ("FR-20", "Eight probabilities and Funding Finder links"),
        ("FR-21", "Similar projects with GtR links"),
        ("FR-22", "Award band, £0 count, not a quote"),
        ("FR-23", "Topic trend"),
        ("FR-24", "One-click demos, including a studentship"),
        ("FR-25", "Council, year and scheme filters"),
        ("FR-26", "CSV and Markdown"),
        ("FR-27", "This page"),
        ("NFR-8", "Okabe–Ito council colours"),
        ("NFR-9", "No researcher names, no stored abstracts"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Id", "Where it is met"]), hide_index=True, width="stretch")


def tab_journey(persona_id: str) -> None:
    st.markdown("## Tailor the advice, not the data")
    brief = PERSONAS[persona_id]
    st.markdown(f'<div class="fm-card"><h3>{html.escape(brief["label"])}</h3><p>{html.escape(brief["scheme_emphasis"])}</p><p>{html.escape(brief["jargon"])}</p></div>', unsafe_allow_html=True)
    title = st.session_state.get("title") or ""
    abstract = st.session_state.get("abstract") or ""
    if not (title or abstract):
        st.info("Load a demo or paste an idea in the matcher tab. This page will score that same text.")
        return
    bundle = _bundle(persona_id)
    score = bundle["readiness"]
    topics = bundle["topics"]
    left, right = st.columns([0.9, 1.1])
    with left:
        st.metric("Readiness", f"{score['overall']:.0f} / 100")
        st.caption(score["tip"])
        radar = pd.DataFrame(score["dimensions"])
        fig = go.Figure()
        short = {
            "Topic specificity and lexical clarity": "Specificity",
            "Council and domain alignment": "Council fit",
            "Originality sweet-spot": "Originality",
            "Methodology and impact signals": "Method",
            "Interdisciplinary bridge potential": "Bridge",
        }
        radar["axis"] = radar["label"].map(lambda name: short.get(name, name))
        fig.add_trace(go.Scatterpolar(r=radar["score"], theta=radar["axis"], fill="toself", line_color="#5eead4"))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100], color="#e7eef6"), angularaxis=dict(color="#e7eef6"), bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7eef6"), height=460, showlegend=False,
            margin=dict(l=90, r=90, t=50, b=50),
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("### Five working titles from real neighbours")
        for topic in topics:
            precedent = topic["precedents"][0]
            probability = topic["calibrated_probability"]
            shown = f"{probability:.0%}" if isinstance(probability, float) else "n/a"
            council = topic["predicted_council"] or "Unscored"
            band = topic["novelty_band"] or "Unscored"
            st.markdown(
                f'<div class="fm-card"><strong>{html.escape(topic["working_title"])}</strong><br>'
                f'<span class="fm-kpi">{html.escape(str(council))}</span> '
                f'<span class="fm-kpi">{html.escape(shown)}</span> '
                f'<span class="fm-kpi">{html.escape(str(band))}</span><br>'
                f'Precedent: <a href="{html.escape(precedent["gtr_url"])}">{html.escape(precedent["title"])}</a>'
                f' · {html.escape(precedent["organisation"])}</div>',
                unsafe_allow_html=True,
            )
    st.download_button(
        "Download research dossier",
        bundle["markdown"],
        "scholarpulse_research_dossier.md",
        "text/markdown",
        key="dossier_md",
    )


def tab_waves() -> None:
    st.markdown("## Is this neighbourhood getting busier?")
    title = st.session_state.get("title") or ""
    abstract = st.session_state.get("abstract") or ""
    text = f"{title}. {abstract}".strip()
    if len(text.split()) < 3:
        st.info("Add an idea first. The wave is counted only inside the 200 nearest funded projects.")
        return
    trend = ENGINE.topic_trends(text)
    st.markdown(f"### {trend['band']}")
    st.write(trend["note"])
    if trend["series"]:
        frame = pd.DataFrame(trend["series"])
        bars = px.bar(frame, x="year", y="projects", labels={"projects": "Funded projects", "year": "Start year"})
        bars.update_traces(marker_color="#5eead4")
        bars.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,9,15,0.2)", font=dict(color="#e7eef6"), showlegend=False, margin=dict(l=48, r=24, t=36, b=40))
        st.plotly_chart(bars, width="stretch")
        line = px.line(frame, x="year", y="total_gbp_real_2026", labels={"total_gbp_real_2026": "£ at Oct 2026 prices", "year": "Start year"})
        line.update_traces(line_color="#c4b5fd")
        line.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,9,15,0.2)", font=dict(color="#e7eef6"), showlegend=False, margin=dict(l=48, r=24, t=36, b=40))
        st.plotly_chart(line, width="stretch")
    if trend["rising_terms"]:
        st.markdown("### Terms more common in the recent window")
        st.dataframe(pd.DataFrame(trend["rising_terms"]), hide_index=True, width="stretch")
    st.markdown("## Two fields at once")
    c1, c2 = st.columns(2)
    field_a = c1.text_input("Field A", value="music", key="fuse_a")
    field_b = c2.text_input("Field B", value="machine learning", key="fuse_b")
    if st.button("Find the bridge", key="fuse_go"):
        st.session_state["fusion"] = ENGINE.fuse(field_a, field_b)
    fusion = st.session_state.get("fusion")
    if fusion:
        st.write("Councils near both:", ", ".join(fusion["bridge_councils"]) or "none in the top 30 of each")
        for angle in fusion["angles"]:
            st.markdown(f'<div class="fm-card">{html.escape(angle)}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(fusion["projects"])[["title", "lead_funder", "organisation", "similarity", "gtr_url"]], hide_index=True, width="stretch")


def _bundle(persona_id: str) -> dict:
    title = st.session_state.get("title") or ""
    abstract = st.session_state.get("abstract") or ""
    key = (title, abstract, persona_id)
    if st.session_state.get("_bundle_key") != key:
        st.session_state["_bundle"] = research_dossier(ENGINE, title, abstract, persona_id)
        st.session_state["_bundle_key"] = key
    return st.session_state["_bundle"]


def _axis_org(name: str) -> str:
    text = " ".join(str(name).split())
    upper = text.upper()
    for prefix in (
        "THE CHANCELLOR, MASTERS AND SCHOLARS OF THE UNIVERSITY OF ",
        "THE CHANCELLOR MASTERS AND SCHOLARS OF THE UNIVERSITY OF ",
    ):
        if upper.startswith(prefix):
            text = "University of " + text[len(prefix):]
            break
    return text


def tab_coach(persona_id: str) -> None:
    st.markdown("## Upgrade the draft, then hear the hard questions")
    title = st.session_state.get("title") or ""
    abstract = st.session_state.get("abstract") or ""
    if not (title or abstract):
        st.info("The idea above is empty. Load a demo or type a title and abstract, then come back.")
        return
    bundle = _bundle(persona_id)
    view = bundle["coach"]
    review = bundle["redteam"]
    before, after, delta = st.columns(3)
    before.metric("Before readiness", f"{view['before']['readiness']:.0f} / 100")
    after.metric("After simulation", f"{view['after']['readiness']:.0f} / 100")
    delta.metric("Readiness change", f"{view['delta_readiness']:+.1f}")
    st.caption(
        f"{view['before']['council']} {view['before']['probability']:.0%} · 90% set size {view['before']['conformal_set_size']}"
        f" → {view['after']['council']} {view['after']['probability']:.0%} · 90% set size {view['after']['conformal_set_size']}"
    )
    st.info(view["simulation_note"])
    st.markdown("### Missing signals in funded neighbours")
    chips = " ".join(f'<span class="fm-kpi">{html.escape(term)}</span>' for term in view["missing_signals"])
    st.markdown(chips or "No extra neighbour terms cleared the filter.", unsafe_allow_html=True)
    st.markdown("### Stage roadmap")
    for step in view["roadmap"]["timeline"]:
        st.markdown(f"- {html.escape(step)}")
    st.markdown("### Deliverable checklist")
    for item in view["roadmap"]["checklist"]:
        st.markdown(f"- {html.escape(item)}")
    st.markdown("### First-contact email")
    st.code(view["roadmap"]["email_template"])
    st.markdown("## Reviewer questions")
    tone = {"High": "fm-high", "Watch": "fm-watch", "Solid": "fm-solid"}
    for item in review["questions"]:
        st.markdown(
            f'<div class="fm-card"><span class="fm-kpi {tone[item["severity"]]}">{html.escape(item["severity"])}</span>'
            f"<p><strong>{html.escape(item['question'])}</strong></p>"
            f"<p>{html.escape(item['why'])}</p>"
            f"<p>{html.escape(item['defence'])}</p></div>",
            unsafe_allow_html=True,
        )
    st.markdown("## Five-dimension feasibility")
    matrix_tone = {"Green": "fm-solid", "Amber": "fm-watch", "Red": "fm-high"}
    for item in review["matrix"]:
        st.markdown(
            f'<div class="fm-card"><span class="fm-kpi {matrix_tone[item["status"]]}">{html.escape(item["status"])}</span> '
            f"<strong>{html.escape(item['dimension'])}</strong><br>{html.escape(item['fact'])}</div>",
            unsafe_allow_html=True,
        )


def tab_hubs(persona_id: str) -> None:
    st.markdown("## Where this topic is already being funded")
    title = st.session_state.get("title") or ""
    abstract = st.session_state.get("abstract") or ""
    if not (title or abstract):
        st.info("Add an idea first. Hubs are counted only inside the nearest funded projects.")
        return
    rows = _bundle(persona_id)["hubs"]["hubs"]
    if not rows:
        st.info("No organisation names were available in this neighbourhood.")
        return
    frame = pd.DataFrame(rows)
    frame["axis"] = frame["organisation"].map(_axis_org)
    fig = px.bar(
        frame, x="awards", y="axis", orientation="h", color="dominant_council",
        color_discrete_map=COUNCIL_COLOURS, labels={"awards": "Nearby awards", "axis": ""},
    )
    fig.update_layout(
        height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,9,15,0.2)",
        font=dict(color="#e7eef6", size=12), legend_title="",
        margin=dict(l=210, r=24, t=36, b=40), yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Counts are among the 60 nearest awards. Pounds are October 2026 CPIH prices. Organisations only.")
    for hub in rows:
        links = " · ".join(
            f'<a href="{html.escape(item["gtr_url"])}">{html.escape(item["title"])}</a>' for item in hub["precedents"]
        )
        st.markdown(
            f'<div class="fm-card"><strong>{html.escape(hub["organisation"])}</strong><br>'
            f'<span class="fm-kpi">{hub["awards"]} awards</span> '
            f'<span class="fm-kpi">£{hub["total_gbp_real_2026"]:,.0f}</span> '
            f'<span class="fm-kpi">median £{hub["median_gbp_real_2026"]:,.0f}</span> '
            f'<span class="fm-kpi">{html.escape(hub["dominant_council"])}</span><br>{links}</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    if "bootstrapped" not in st.session_state:
        opening = DEMOS[0]
        st.session_state["title"] = opening["title"]
        st.session_state["abstract"] = opening["abstract"]
        st.session_state["scheme"] = opening["scheme"]
        st.session_state["_scored"] = True
        st.session_state["bootstrapped"] = True
    projects = f"{len(ENGINE.frame):,}"
    st.markdown(
        f'<div class="fm-hero"><p class="fm-kicker">{APP_EDITION}</p>'
        f'<h1 style="margin:0.2rem 0">{APP_DISPLAY_NAME}</h1><p>{APP_TAGLINE}</p>'
        f'<div class="fm-kpis">'
        f'<span class="fm-kpi">{projects} real UKRI grants</span>'
        f'<span class="fm-kpi">{len(COUNCIL_INFO)} scored councils</span>'
        f'<span class="fm-kpi">{len(PERSONAS)} student stages</span>'
        f'<span class="fm-kpi">Local calibrated AI</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    labels = {key: value["label"] for key, value in PERSONAS.items()}
    choice = st.selectbox("Tailor to your journey", list(labels), format_func=lambda key: labels[key], key="persona")
    st.markdown(f'<span class="fm-stage">{html.escape(labels[choice])}</span>', unsafe_allow_html=True)
    for start in range(0, len(DEMOS), 5):
        chips = st.columns(5)
        for column, demo in zip(chips, DEMOS[start:start + 5]):
            if column.button(demo.get("pill", demo["label"]), key=f"demo_{demo['id']}"):
                st.session_state["title"] = demo["title"]
                st.session_state["abstract"] = demo["abstract"]
                st.session_state["scheme"] = demo["scheme"]
                st.session_state["_scored"] = True
    st.text_input("Idea title", key="title")
    st.text_area("Abstract or one-paragraph idea", key="abstract", height=140)
    words = len(f"{(st.session_state.get('title') or '').strip()}. {(st.session_state.get('abstract') or '').strip()}".split())
    st.caption(f"{words} words in the idea" + (" · ready to score" if words >= 30 else " · add words until you pass 30"))
    if st.button("Analyze Idea", type="primary", key="analyze_idea"):
        st.session_state["_scored"] = True
    journey, coach_tab, hub_tab, waves, matcher, sky, arena, lab = st.tabs([
        "Your journey", "AI Coach & Red-Team", "UK Research Hubs", "Waves & fusion",
        "AI Matcher", "Constellation & What-if", "Compare & Batch", "Model Lab & Ethics",
    ])
    with journey:
        tab_journey(choice)
    with coach_tab:
        tab_coach(choice)
    with hub_tab:
        tab_hubs(choice)
    with waves:
        tab_waves()
    with matcher:
        tab_matcher(False)
    with sky:
        tab_sky()
    with arena:
        tab_compare()
    with lab:
        tab_lab(True)
        tab_corpus()


main()
