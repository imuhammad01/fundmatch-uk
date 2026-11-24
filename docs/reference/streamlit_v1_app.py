"""
FundMatch UK - Streamlit user interface (CS5079 Applied AI, Assessment 1, Task 1).

Paste a research title and abstract. The app shows:
  1. which UKRI council most likely funds this kind of work (calibrated probabilities) and the words
     that drove that prediction,
  2. a typical award range taken from the most similar funded projects, and a trend chart,
  3. the most similar projects UKRI has already funded (links to Gateway to Research).

All model logic lives in fundmatch_core.py (unit tested in tests/test_core.py); this file is only the UI.
Data: UKRI Gateway to Research API, Open Government Licence v3.0. Not official UKRI advice.
Privacy: user text is processed in memory only and never stored or logged.
"""
import os

import altair as alt
import streamlit as st

from fundmatch_core import (COUNCIL_INFO, FUNDING_FINDER, INNOVATION_FUNDING_SERVICE, LOW_CONF, MIN_WORDS, award_range,
                            check_input, explain, funding_finder_url, gbp, load_assets, predict, similar)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

EXAMPLES = {
    "Battery recycling (engineering)": (
        "Low-energy recovery of lithium and cobalt from end-of-life electric vehicle batteries",
        "Electric vehicle batteries contain valuable metals that are rarely recovered. This project will develop a "
        "low-temperature hydrometallurgical process that uses deep eutectic solvents to dissolve cathode materials "
        "and selectively recover lithium, cobalt and nickel. We will model the reaction kinetics, build a "
        "laboratory-scale flow reactor and compare the energy use and carbon footprint of the process with "
        "pyrometallurgy. The outcome will be a scalable route to a circular supply of battery materials."),
    "Loneliness in older adults (social science)": (
        "Digital inclusion and loneliness among older adults in rural Scotland",
        "Loneliness is a growing public health and social problem for older people living in rural areas. Using a "
        "mixed-methods design, we will survey 1,500 adults aged over 70 and run interviews and focus groups to "
        "understand how access to broadband, digital skills and local community groups shape social connection. "
        "The project will work with local councils and charities to co-design policy recommendations and evaluate "
        "a peer-mentoring scheme that supports older people to use video calling and online services."),
    "Antibiotic resistance (medical)": (
        "Phage therapy to treat drug-resistant urinary tract infections",
        "Urinary tract infections caused by multidrug-resistant Escherichia coli are increasingly hard to treat. We "
        "will isolate bacteriophages from wastewater, characterise their host range against clinical isolates and "
        "test phage cocktails in a mouse model of infection. In parallel, we will study how resistance to phages "
        "emerges and whether it restores antibiotic sensitivity. The work will provide preclinical evidence for an "
        "early-phase clinical trial in patients with recurrent infections."),
    "AI for crop disease (agri-food AI)": (
        "Machine learning for early detection of fungal disease in Scottish barley from drone imagery",
        "Fungal diseases such as Ramularia leaf spot reduce barley yields and drive heavy fungicide use. This project "
        "will combine multispectral drone images, weather records and field surveys from farms in north-east Scotland "
        "to train deep learning models that detect infection before symptoms are visible. We will test how well the "
        "models transfer between farms and seasons, explain their predictions to agronomists, and evaluate whether "
        "targeted spraying guided by the models can cut fungicide use while protecting yield and grain quality."),
}


@st.cache_resource(show_spinner="Loading models and index (the first load can take about a minute)...")
def load():
    return load_assets(ASSETS)


def main():
    st.set_page_config(page_title="FundMatch UK", page_icon="🔎", layout="wide")
    st.title("🔎 FundMatch UK")
    st.markdown("**Before you write a UK research proposal:** see which UKRI council funds work like yours, why, "
                "what similar projects it has already funded and for how much, in seconds, from your abstract.")
    st.caption("Built on 113k cleaned UKRI Gateway to Research projects. A guide for early planning, not official UKRI advice.")
    A = load()
    info, meta = A["info"], A["meta"]

    with st.sidebar:
        st.header("Try an example")
        for name, (t, a) in EXAMPLES.items():
            if st.button(name, width="stretch"):
                st.session_state["title"], st.session_state["abstract"] = t, a
        st.header("Filters for similar projects")
        councils = st.multiselect("Councils", list(COUNCIL_INFO), default=list(COUNCIL_INFO))
        ymin, ymax = int(meta["start_year"].min()), int(meta["start_year"].max())
        years = st.slider("Start year", ymin, ymax, (max(ymin, 2006), ymax))
        scheme = st.selectbox("Grant category (e.g. Studentship for PhD projects; makes the award range more precise)",
                              ["Any scheme"] + info.get("grant_categories", []))
        k = st.slider("Number of similar projects", 5, 30, 10)

    title = st.text_input("Project title", key="title", placeholder="e.g. Low-cost sensors for river pollution")
    abstract = st.text_area("Abstract or summary of your idea", key="abstract", height=180,
                            placeholder=f"Paste at least {MIN_WORDS} words describing aims, methods and impact.")
    go = st.button("Find funders and similar projects", type="primary")

    if go:
        errors, warns = check_input(title, abstract, A["vec"])
        for e in errors:
            st.error(e)
        if not errors and not councils:
            st.error("Select at least one council in the filters."); errors = ["no council"]
        if errors:
            st.session_state.pop("result", None)
            st.stop()
        text = f"{(title or '').strip()}. {(abstract or '').strip()}".strip(". ")
        probs, q = predict(text, A)
        sim30 = similar(q, A, max(k, 30), councils, years, scheme)   # 30 neighbours for the award range
        st.session_state["result"] = (probs, sim30.head(k), sim30.head(30), scheme, warns,
                                      explain(text, probs.index[0], A))

    if "result" not in st.session_state:
        st.info("Enter a title and abstract, or load an example from the sidebar.")
    else:
        probs, sim, sim30, scheme_used, warns, why = st.session_state["result"]
        for w in warns:
            st.warning(w)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("1. Most likely funder")
            top, p = probs.index[0], probs.iloc[0]
            st.metric("Top match", f"{top}")
            st.write(f"**{p:.0%} probability** · {COUNCIL_INFO[top]}")
            if p < LOW_CONF:
                st.warning(f"Low confidence: the top probability is under {LOW_CONF:.0%}. Your idea may sit between "
                           f"councils (cross-council work is common). Consider the top two or three.")
            chart = alt.Chart(probs.rename("probability").rename_axis("council").reset_index()).mark_bar(color="#0072B2").encode(
                x=alt.X("probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("council:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=160)), tooltip=["council", alt.Tooltip("probability:Q", format=".1%")])
            st.altair_chart(chart.properties(height=250), width="stretch")
            links = " · ".join(f"[{c} open calls]({funding_finder_url(c)})" for c in probs.index[:3])
            st.markdown(f"**Live calls on the UKRI Funding Finder:** {links}" +
                        (f" · [Innovation Funding Service]({INNOVATION_FUNDING_SERVICE})" if "Innovate UK" in probs.index[:3] else ""))
            if why:
                st.markdown(f"**Why {top}?** Words in your text that pushed the model towards {top}: " +
                            " · ".join(f"`{w}`" for w, _ in why))
            st.caption(f"Calibrated probabilities from {info['deployed_model']}. Right council in the top 3 in "
                       f"{info['test_top3']:.0%} of held-out tests. All open calls: [UKRI Funding Finder]({FUNDING_FINDER}).")
        with c2:
            st.subheader("2. Typical award for similar projects")
            band = award_range(sim30["award_gbp"])
            if band:
                st.metric("Typical award (middle 50% of the 30 most similar)", f"{gbp(band['p25'])} – {gbp(band['p75'])}")
                st.write(f"Median **{gbp(band['p50'])}** · wider range (10th–90th percentile): "
                         f"**{gbp(band['p10'])} – {gbp(band['p90'])}**")
            else:
                st.write("Not enough award amounts among the matches.")
            cov = info.get("award_coverage", {})
            rname = "TF-IDF" if info["retriever"] == "tfidf" else info["retriever"].upper()
            base = f"{rname}: 30 same-scheme neighbours" if scheme_used != "Any scheme" else f"{rname}: 30 neighbours"
            st.caption("A guide only, **not a quote and not an eligibility guarantee**: awards depend on the scheme, duration and institution. In held-out tests the "
                       f"real award fell inside the middle-50% range {cov.get(base + ', P25-P75', float('nan')):.0%} and inside "
                       f"the wider range {cov.get(base + ', P10-P90', float('nan')):.0%} of the time" +
                       (". Pick a scheme in the sidebar for a tighter match." if scheme_used == "Any scheme" else "."))
            yr = sim30.dropna(subset=["start_year"]).groupby(["start_year", "lead_funder"]).size().rename("projects").reset_index()
            if len(yr):
                st.write("Trend: the 30 most similar projects by start year and council")
                trend = alt.Chart(yr).mark_bar().encode(
                    x=alt.X("start_year:O", title="start year"), y=alt.Y("projects:Q", title="projects"),
                    color=alt.Color("lead_funder:N", title="council"),
                    tooltip=[alt.Tooltip("start_year:O", title="year"), alt.Tooltip("lead_funder:N", title="council"), "projects:Q"])
                st.altair_chart(trend.properties(height=220), width="stretch")
            if scheme_used == "Studentship":
                st.info("Studentships shown are past funded PhD projects, not open vacancies. Look for live PhD places on "
                        "university and doctoral-training-centre pages.")

        st.subheader("3. Most similar funded projects")
        show = sim.assign(award=sim["award_gbp"].map(gbp), start=sim["fund_start"].dt.strftime("%Y-%m"),
                          end=sim["fund_end"].dt.strftime("%Y-%m"))[
            ["similarity", "title", "lead_funder", "grant_category", "lead_org_name", "award", "start", "end", "gtr_url"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "gtr_url": st.column_config.LinkColumn("GtR link", display_text="open"),
            "lead_funder": "council", "lead_org_name": "lead organisation", "grant_category": "scheme"})
        st.download_button("Download these results (CSV)", show.to_csv(index=False).encode(), "fundmatch_results.csv", "text/csv")

    with st.expander("About, accuracy and limits"):
        cal = {r["probabilities"][:3]: r for r in info.get("calibration", [])}
        ece = (f"; calibration error (ECE) {cal['raw']['ECE']:.3f} before and {cal['tem']['ECE']:.3f} after temperature scaling"
               if "raw" in cal and "tem" in cal else "")
        weak = sorted(info["per_class_f1"].items(), key=lambda kv: kv[1])[:2]
        st.markdown(f"""
* **Data source:** {info['n_projects_index']:,} cleaned UKRI projects collected from the
  [Gateway to Research API](https://gtr.ukri.org/) on {info['data_collected'][:10]} with our own code.
* Contains public sector information licensed under the Open Government Licence v3.0
  ([licence text](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)).
* **Model:** {info['deployed_model']}; similar-project search uses {info['retriever'].upper()} cosine similarity. Held-out test:
  macro-F1 {info['test_macro_f1']:.3f}, accuracy {info['test_accuracy']:.1%}, right council in the top 3 {info['test_top3']:.1%}{ece}.
* **Per-council F1:** {", ".join(f"{k} {v:.2f}" for k, v in info['per_class_f1'].items())}. Weakest: {", ".join(f"{k} ({v:.2f})" for k, v in weak)}.
* **Limits:** it learns from *past lead funders*, so cross-council and new schemes are harder; programme
  budgets (ISCF, GCRF…) are not predicted; award ranges ignore inflation. It does not predict success.
  Past studentships are not PhD vacancies.
* **Privacy:** your title and abstract are processed in memory only and are never stored or logged.
  Researcher names are never shown or downloaded.
* **Not official UKRI funding advice.** This is a student project (University of Aberdeen, CS5079); UKRI does not endorse it.
""")


if __name__ == "__main__":
    main()
