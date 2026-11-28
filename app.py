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

from fundmatch import insights
from fundmatch.brand import APP_DISPLAY_NAME, APP_EDITION, APP_TAGLINE, APP_VERSION
from fundmatch.futuristic import research_dossier
from fundmatch.cpih import CAVEAT, SOURCE_URL, TARGET_LABEL
from fundmatch.demos import DEMOS
from fundmatch.engine import COUNCIL_COLOURS, COUNCIL_INFO, DISCLAIMER, IFS, OGL, funding_finder_url, get_engine
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

CSS = (ROOT / "src" / "fundmatch" / "assets" / "neon.css").read_text(encoding="utf-8")
st.markdown(CSS, unsafe_allow_html=True)


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
        fig.update_layout(barmode="group", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7eef6"), yaxis_tickformat=".0%", xaxis_title="UKRI council", yaxis_title="Calibrated probability", legend_title="")
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

**Privacy.** Pasted abstracts live only in your browser session's memory while the page is open. They are never written to disk, logged, or sent to another service, and the app calls no paid AI API. The neighbour table has organisations, not people.

**Money.** {CAVEAT} {TARGET_LABEL}.

**Out of scope.** The app does not predict whether a grant will be funded, does not write or edit proposals (the Idea Studio only checks wording with rules, lists vocabulary used by nearby funded titles, and runs labelled what-if simulations; it never rewrites your text), and does not rank or profile researchers (only organisations are counted). Lecture 3 is not used.
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
        st.caption("Readiness is a transparent writing-signal heuristic (specificity, council fit, originality, method, bridge). It is not a prediction that an application would be funded.")
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
    st.markdown("## Test the vocabulary, then hear the hard questions")
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
        color_discrete_map=COUNCIL_COLOURS, labels={"awards": "Nearby awards", "axis": "Organisation"},
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


# ----------------------------------------------------------------------------- V5 · Neon Edition

def _style(fig, height: int = 320):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,9,15,0.25)",
        font=dict(color="#e7eef6", size=12), margin=dict(l=10, r=16, t=30, b=30), legend_title="",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.25)")
    return fig


def _idea() -> tuple[str, str]:
    return (st.session_state.get("title") or ""), (st.session_state.get("abstract") or "")


def _live(top_k: int = 10):
    """One cached analysis of the current idea per (text, top_k)."""
    title, abstract = _idea()
    key = (title, abstract, top_k)
    cache = st.session_state.setdefault("_live_cache", {})
    if key not in cache:
        if len(cache) > 6:
            cache.clear()
        cache[key] = ENGINE.analyze(title, abstract, top_k=top_k)
    return cache[key]


def _tile(label: str, value: str, sub: str = "", tone: str = "sp-neon", bar: float | None = None) -> str:
    meter = f'<div class="sp-bar"><span style="width:{max(0.0, min(1.0, bar)) * 100:.1f}%"></span></div>' if bar is not None else ""
    return (
        f'<div class="sp-tile"><div class="sp-label">{html.escape(label)}</div>'
        f'<div class="sp-value {tone}">{html.escape(value)}</div>'
        f'<div class="sp-sub">{html.escape(sub)}</div>{meter}</div>'
    )


def onboarding() -> None:
    if st.session_state.get("guide_hidden"):
        return
    with st.expander("✦ New here? How to use ScholarPulse in four steps", expanded=True):
        steps = [
            ("Describe the idea", "Pick your stage, then type a title and a short abstract, or tap a demo chip. 30 words is the minimum; 100+ is better."),
            ("Read the mission brief", "The panel under the button shows the most likely UKRI council, how reliable that number is, and the honest uncertainty."),
            ("Open Idea Studio", "See which words drive the prediction, run the draft checklist, find vocabulary used by nearby funded projects, and compare two ideas."),
            ("Check precedents and export", "Filter real funded projects by council, year and amount, then download a PDF brief and open the live calls on UKRI Funding Finder."),
        ]
        st.markdown(
            '<div class="sp-grid sp-steps">' + "".join(
                f'<div class="sp-tile sp-step"><div class="sp-label">{html.escape(head)}</div><div class="sp-sub">{html.escape(body)}</div></div>'
                for head, body in steps
            ) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Everything is computed live from 113,233 real UKRI awards and a calibrated local model. No paid AI service sees your text, and abstracts are not stored.")
        if st.button("Got it, hide the guide", key="guide_hide"):
            st.session_state["guide_hidden"] = True
            st.rerun()


def mission_brief() -> None:
    title, abstract = _idea()
    if not (title.strip() or abstract.strip()) or not st.session_state.get("_scored"):
        return
    result = _live(10)
    st.markdown('<p class="fm-kicker"><span class="sp-dot"></span>Mission brief · live result for the idea above</p>', unsafe_allow_html=True)
    if not result.ok:
        for error in result.errors:
            st.error(error)
        return
    rel = insights.reliability(ENGINE, result)
    top = result.councils[0]
    gauge, tiles = st.columns([0.9, 2.1])
    with gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=round(top.probability * 100, 1), number={"suffix": "%", "font": {"size": 40, "color": "#f8fafc"}},
            title={"text": f"{top.council} · calibrated probability", "font": {"size": 13, "color": "#a3b3c6"}},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "tickcolor": "#64748b", "tickfont": {"color": "#a3b3c6", "size": 10}},
                "bar": {"color": "#5eead4", "thickness": 0.28},
                "bgcolor": "rgba(15,23,42,0.6)", "borderwidth": 0,
                "steps": [
                    {"range": [0, 60], "color": "rgba(244,63,94,0.16)"},
                    {"range": [60, 75], "color": "rgba(245,158,11,0.16)"},
                    {"range": [75, 100], "color": "rgba(16,185,129,0.16)"},
                ],
            },
        ))
        _style(fig, 240).update_layout(margin=dict(l=40, r=40, t=50, b=8))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="brief_gauge")
    with tiles:
        accuracy = rel.get("bin_test_accuracy")
        band = result.award
        median = _money_short(band.median) if band and band.median is not None else "—"
        st.markdown(
            '<div class="sp-grid">'
            + _tile("Most likely council", top.council, COUNCIL_INFO[top.council])
            + _tile("Test accuracy at this confidence", f"{accuracy:.1%}" if accuracy is not None else "—",
                    f"Test predictions in the {rel.get('bin')} band", "sp-violet", accuracy)
            + _tile("Lead over runner-up", f"{rel['margin'] * 100:+.0f} pts", f"vs {rel['runner_up']} · {rel['margin_note']}")
            + _tile("90% shortlist", ", ".join(result.conformal_set_90), f"95%: {', '.join(result.conformal_set_95)}", "sp-violet")
            + _tile("Novelty", (result.novelty or {}).get("band", "—"), "Distance to the closest funded abstracts")
            + _tile("Typical nearby award", median, f"Median of {band.n_positive} priced neighbours · not a quote" if band else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    if result.confidence == "low":
        st.error(result.confidence_note)
    elif result.confidence == "moderate":
        st.warning(result.confidence_note)
    links = st.columns(min(3, len(result.conformal_set_95)) + 1)
    for column, council in zip(links, result.conformal_set_95[:3]):
        column.link_button(f"Open {council} calls ↗", funding_finder_url(council), width="stretch")
    links[-1].link_button("Innovation Funding Service ↗", IFS, width="stretch")
    st.markdown(
        '<p class="sp-caveat">Probabilities say which council has funded similar text before. They are not a chance of being funded. '
        f'{html.escape(rel.get("note", ""))}</p>',
        unsafe_allow_html=True,
    )


def _studio_explain(title: str, abstract: str, result) -> None:
    st.markdown("### Why the model chose this council")
    council = st.selectbox("Explain council", [row.council for row in result.councils], key="studio_council")
    drivers = insights.decision_drivers(ENGINE, title, abstract, council, k=10)
    rows = pd.DataFrame(drivers["rows"])
    if rows.empty:
        st.info("No recognised words to explain.")
    else:
        rows["colour"] = rows["direction"].map({"toward": "#5eead4", "away": "#fb7185"})
        rows = rows.sort_values("points")
        fig = go.Figure(go.Bar(
            x=rows["points"], y=rows["term"], orientation="h", marker_color=rows["colour"],
            hovertemplate="%{y}: %{x:+.3f} logit points<extra></extra>",
        ))
        _style(fig, 60 + 24 * len(rows)).update_layout(xaxis_title=f"Push toward (+) or away from (−) {council}", yaxis_title="Word or phrase")
        st.plotly_chart(fig, width="stretch", key="drivers_chart")
    st.markdown(f'<p class="sp-caveat">{html.escape(drivers.get("note", ""))} The model reads words, not meaning: '
                'a phrase like “not a compiler” still counts the word “compiler”.</p>', unsafe_allow_html=True)
    rel = insights.reliability(ENGINE, result)
    bins = ENGINE.model_info.get("accuracy_by_confidence", {})
    if bins:
        st.markdown("### How reliable is that number?")
        frame = pd.DataFrame([{"band": label, "accuracy": acc} for label, acc in bins.items()])
        frame["you"] = frame["band"].map(lambda b: "your idea" if b == rel.get("bin") else "other bands")
        fig = px.bar(frame, x="band", y="accuracy", color="you", text_auto=".1%",
                     color_discrete_map={"your idea": "#a78bfa", "other bands": "rgba(148,163,184,0.45)"},
                     labels={"band": "Top-council probability band", "accuracy": "Held-out accuracy"})
        fig.update_traces(textposition="outside", textfont_color="#f8fafc", cliponaxis=False)
        _style(fig, 300).update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.08])
        st.plotly_chart(fig, width="stretch", key="reliability_chart")
        st.caption(f"From the held-out test split (model_info.json). Top-3 accuracy was {rel['test_top3']:.2%}, which is why low-confidence ideas get a three-council shortlist.")


def _studio_checklist(title: str, abstract: str) -> None:
    check = insights.abstract_checklist(ENGINE, title, abstract)
    st.markdown(f"### Draft checklist · {check['met']} of {check['total']} met" + (f", {check['partial']} partly" if check["partial"] else ""))
    icon = {"met": "✓", "partial": "~", "missing": "✗"}
    st.markdown(
        '<div class="sp-grid">' + "".join(
            f'<div class="sp-tile"><div class="sp-check"><span class="sp-icon sp-{item["status"]}" aria-label="{item["status"]}">{icon[item["status"]]}</span>'
            f'<div><strong>{html.escape(item["label"])}</strong><div class="sp-sub">{html.escape(item["evidence"])}</div>'
            f'<div class="sp-sub" style="color:#a3b3c6;margin-top:4px">{html.escape(item["why"])}</div></div></div></div>'
            for item in check["items"]
        ) + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="sp-caveat">{html.escape(check["note"])}</p>', unsafe_allow_html=True)


def _studio_vocabulary(title: str, abstract: str, result) -> None:
    council = result.top_council
    radar = insights.keyword_radar(ENGINE, title, abstract, council)
    st.markdown(f"### Vocabulary radar · {radar['neighbours']} nearest funded titles")
    left, right = st.columns(2)
    with left:
        st.markdown("**Used by nearby funded projects, not in your draft**")
        chips = "".join(f'<span class="sp-chip">{html.escape(r["term"])}<b>{r["titles"]}/{r["of"]}</b></span>' for r in radar["suggest"])
        st.markdown(chips or "Nothing common is missing.", unsafe_allow_html=True)
    with right:
        st.markdown("**Already in your draft**")
        chips = "".join(f'<span class="sp-chip sp-have">{html.escape(r["term"])}<b>{r["titles"]}/{r["of"]}</b></span>' for r in radar["already"])
        st.markdown(chips or "None of the common neighbour terms yet.", unsafe_allow_html=True)
    st.markdown(f'<p class="sp-caveat">{html.escape(radar["note"])}</p>', unsafe_allow_html=True)
    options = [r["term"] for r in radar["suggest"]]
    if options:
        picked = st.multiselect("Test terms that are genuinely true of your work", options, key="radar_pick")
        if picked and st.button("Simulate with these terms", key="radar_go"):
            st.session_state["radar_sim"] = (tuple(picked), ENGINE.simulate_what_if(title, abstract, add_terms=", ".join(picked), target_council=council))
        sim = st.session_state.get("radar_sim")
        if sim and sim[1].get("target"):
            target = sim[1]["target"]
            a, b, c = st.columns(3)
            a.metric(f"{council} before", f"{target['before']:.1%}")
            b.metric(f"{council} after", f"{target['after']:.1%}", f"{target['delta'] * 100:+.1f} pts")
            c.metric("Terms tested", str(len(sim[0])))
            st.caption("A simulation on the deployed model. A higher probability is not a reason to claim work you are not doing.")


def _studio_explorer(result_wide) -> None:
    st.markdown("### Precedent explorer · 50 nearest funded projects")
    rows = [p.model_dump() for p in result_wide.similar]
    if not rows:
        st.info("No neighbours to explore.")
        return
    years = [r["start_year"] for r in rows if r["start_year"]]
    amounts = [r["award_gbp"] for r in rows if r["award_gbp"] > 0]
    c1, c2 = st.columns(2)
    councils = c1.multiselect("Councils", sorted({r["lead_funder"] for r in rows}), key="ex_councils")
    priced = c2.toggle("Only priced awards", key="ex_priced")
    year_range = None
    if years and min(years) < max(years):
        year_range = st.slider("Start year", int(min(years)), int(max(years)), (int(min(years)), int(max(years))), key="ex_years")
    award_range = None
    if len(amounts) > 1:
        top_k = int(max(amounts) / 1000) + 1
        lo_hi = st.slider("Award (£ thousands, nominal)", 0, top_k, (0, top_k), key="ex_award")
        award_range = (lo_hi[0] * 1000.0, lo_hi[1] * 1000.0)
    shown = insights.filter_precedents(rows, councils or None, year_range, award_range, priced)
    st.caption(f"{len(shown)} of {len(rows)} projects match. £0 awards (mostly studentships) are kept unless you tick “Only priced awards”.")
    if not shown:
        return
    frame = pd.DataFrame(shown)
    plot = frame[frame["award_gbp"] > 0]
    if len(plot):
        fig = px.scatter(plot, x="start_year", y="award_gbp", color="lead_funder", size="similarity", size_max=22,
                         hover_name="title", hover_data={"lead_org_name": True, "similarity": ":.3f", "start_year": True, "award_gbp": ":,.0f"},
                         color_discrete_map=COUNCIL_COLOURS, log_y=True,
                         labels={"start_year": "Start year", "award_gbp": "Award (£, log scale)", "lead_funder": "Council"})
        st.plotly_chart(_style(fig, 380), width="stretch", key="explorer_chart")
    table = pd.DataFrame({
        "similarity": frame["similarity"], "title": frame["title"], "council": frame["lead_funder"],
        "year": frame["start_year"], "award (£)": frame["award_gbp"].map(lambda v: f"£{v:,.0f}" if v > 0 else "£0 · not priced"),
        "organisation": frame["lead_org_name"], "GtR": frame["gtr_url"],
    })
    st.dataframe(table, hide_index=True, width="stretch", column_config={"GtR": st.column_config.LinkColumn("GtR", display_text="open")})
    st.download_button("Download these precedents (CSV)", table.to_csv(index=False).encode(), "scholarpulse-precedents.csv", "text/csv", key="ex_csv")


def _studio_compare(title: str, abstract: str) -> None:
    st.markdown("### Idea vs idea")
    st.caption("Idea A is the text at the top of the page. Write idea B below, or load a demo into it.")
    if "vs_title" not in st.session_state:
        starter = next((d for d in DEMOS if d["id"] == "borderline"), DEMOS[1])
        st.session_state["vs_title"], st.session_state["vs_abstract"] = starter["title"], starter["abstract"]
    demo_labels = ["(keep my text)"] + [d["label"] for d in DEMOS]
    pick = st.selectbox("Load a demo into idea B", demo_labels, key="vs_demo")
    if pick != "(keep my text)" and st.session_state.get("_vs_loaded") != pick:
        demo = next(d for d in DEMOS if d["label"] == pick)
        st.session_state["vs_title"], st.session_state["vs_abstract"] = demo["title"], demo["abstract"]
        st.session_state["_vs_loaded"] = pick
    st.text_input("Idea B title", key="vs_title")
    st.text_area("Idea B abstract", key="vs_abstract", height=110)
    if st.button("Compare A and B", type="primary", key="vs_go"):
        st.session_state["vs_result"] = insights.compare_two(
            ENGINE, {"title": title, "abstract": abstract},
            {"title": st.session_state.get("vs_title", ""), "abstract": st.session_state.get("vs_abstract", "")},
        )
    out = st.session_state.get("vs_result")
    if not out:
        return
    if not out["ok"]:
        for error in out.get("errors", []):
            st.error(error)
        return
    a, b = out["a"], out["b"]
    st.markdown(
        '<div class="sp-grid">'
        + _tile("Idea A", f"{a['top_council']} {a['probability']:.0%}", f"90% set {', '.join(a['set90'])} · {a['novelty']}")
        + _tile("Idea B", f"{b['top_council']} {b['probability']:.0%}", f"90% set {', '.join(b['set90'])} · {b['novelty']}", "sp-violet")
        + _tile("Semantic closeness", f"{out['semantic_cosine']:.2f}", "Cosine in the abstract map (1 = same direction)", "sp-neon", max(0.0, out["semantic_cosine"]))
        + _tile("Shared precedents", f"{out['shared_precedents']} / {out['neighbours']}", f"Overlap (Jaccard) {out['neighbour_jaccard']:.2f}", "sp-violet")
        + "</div>",
        unsafe_allow_html=True,
    )
    frame = pd.DataFrame(out["councils"])
    fig = go.Figure()
    fig.add_bar(name="Idea A", x=frame["council"], y=frame["idea_a"], marker_color="#5eead4")
    fig.add_bar(name="Idea B", x=frame["council"], y=frame["idea_b"], marker_color="#a78bfa")
    _style(fig, 320).update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_title="Calibrated probability")
    st.plotly_chart(fig, width="stretch", key="vs_chart")
    medians = f"Typical nearby award (CPIH 2026): A {_money(a['median_gbp_2026'])} from {a['priced_neighbours']} priced, B {_money(b['median_gbp_2026'])} from {b['priced_neighbours']} priced."
    st.caption(f"{out['note']} {medians}")


def _studio_export(title: str, abstract: str, result) -> None:
    st.markdown("### Export the brief")
    st.caption("A PDF with the council probabilities, shortlist, award band, word drivers, draft checklist, vocabulary radar, nearest projects and live-call links, plus the caveats.")
    if st.button("Build PDF brief", type="primary", key="pdf_go"):
        with st.spinner("Building the PDF"):
            st.session_state["pdf_bytes"] = insights.pdf_brief(
                title, abstract, result,
                insights.abstract_checklist(ENGINE, title, abstract),
                insights.decision_drivers(ENGINE, title, abstract, result.top_council),
                insights.keyword_radar(ENGINE, title, abstract, result.top_council),
                APP_DISPLAY_NAME, f"v{APP_VERSION}",
            )
            st.session_state["pdf_key"] = (title, abstract)
    if st.session_state.get("pdf_bytes") and st.session_state.get("pdf_key") == (title, abstract):
        st.download_button("Download PDF brief", st.session_state["pdf_bytes"], "scholarpulse-brief.pdf", "application/pdf", key="pdf_dl")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Markdown brief", ENGINE.to_markdown(title, result), "scholarpulse-brief.md", "text/markdown", key="studio_md")
    c2.download_button("Similar projects CSV", ENGINE.to_csv(result.similar), "scholarpulse-similar.csv", "text/csv", key="studio_csv")
    c3.download_button("Full result JSON", ENGINE.to_json(result), "scholarpulse-result.json", "application/json", key="studio_json")


STUDIO_TOOLS = ["Explain", "Draft checklist", "Vocabulary", "Precedents", "Idea vs idea", "Export"]


def tab_studio() -> None:
    st.markdown('<p class="fm-kicker">V5 · Idea Studio</p>', unsafe_allow_html=True)
    st.markdown("## Idea Studio")
    title, abstract = _idea()
    if not (title.strip() or abstract.strip()):
        st.info("Type an idea at the top of the page or tap a demo chip. The studio works on that text.")
        return
    result = _live(10)
    if not result.ok:
        for error in result.errors:
            st.error(error)
        return
    tool = st.segmented_control("Studio tool", STUDIO_TOOLS, default="Explain", key="studio_tool") or "Explain"
    if tool == "Explain":
        _studio_explain(title, abstract, result)
    elif tool == "Draft checklist":
        _studio_checklist(title, abstract)
    elif tool == "Vocabulary":
        _studio_vocabulary(title, abstract, result)
    elif tool == "Precedents":
        _studio_explorer(_live(50))
    elif tool == "Idea vs idea":
        _studio_compare(title, abstract)
    else:
        _studio_export(title, abstract, result)


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
        f'<div class="fm-hero"><p class="fm-kicker"><span class="sp-dot"></span>{APP_EDITION}</p>'
        f'<h1 class="sp-title">{APP_DISPLAY_NAME}</h1><p class="sp-tagline">{APP_TAGLINE}</p>'
        f'<div class="fm-kpis">'
        f'<span class="fm-kpi">{projects} real UKRI grants</span>'
        f'<span class="fm-kpi">{len(COUNCIL_INFO)} scored councils</span>'
        f'<span class="fm-kpi">{len(PERSONAS)} student stages</span>'
        f'<span class="fm-kpi">Calibrated local AI · no paid API</span>'
        f'<span class="fm-kpi">Explainable · exportable</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    onboarding()
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
    mission_brief()
    studio_tab, journey, coach_tab, hub_tab, waves, matcher, sky, arena, lab = st.tabs([
        "✦ Idea Studio", "Your journey", "AI Coach & Red-Team", "UK Research Hubs", "Waves & fusion",
        "AI Matcher", "Constellation & What-if", "Compare & Batch", "Model Lab & Ethics",
    ])
    with studio_tab:
        tab_studio()
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
    st.markdown(
        '<p class="sp-caveat" style="margin-top:1.5rem">ScholarPulse UK is a student project, not official UKRI funding advice, and UKRI does not endorse it. '
        'Contains public sector information licensed under the '
        '<a href="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/">Open Government Licence v3.0</a> '
        '(UKRI Gateway to Research). No researcher names are shown and abstracts are not stored.</p>',
        unsafe_allow_html=True,
    )


main()
