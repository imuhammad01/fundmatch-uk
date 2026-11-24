"""Offline student tools grounded in the loaded UKRI table.

Coach, red-team, and hub radar each run one neighbour search and the calibrated
logits. They do not call the full analyzer, so a warm request stays well under
150 ms. Nothing here writes a claim the student did not already type: added
terms are a labelled simulation.
"""

from __future__ import annotations

import html
import re
from collections import Counter

import numpy as np
import pandas as pd

from fundmatch.brand import APP_DISPLAY_NAME
from fundmatch.cpih import to_real_gbp
from fundmatch.intelligence import empirical_sets, novelty_band
from fundmatch.student import (
    PERSONAS,
    _DATA,
    _EXTRA_STOP,
    _IMPACT,
    _METHOD,
    _PROBLEM,
    _bigrams,
    _terms,
    persona_brief,
    suggest_topics,
    topic_trends,
)
from fundmatch.text import TOKEN_RE, mask_funders

_BOILER = set(_EXTRA_STOP) | {
    "university", "universities", "college", "colleges", "london", "cambridge", "oxford",
    "limited", "ltd", "institute", "school", "department", "faculty", "centre", "center",
    "national", "british", "royal", "group", "programme", "program", "funded", "grant",
    "award", "council", "studentship", "fellowship", "network", "laboratory", "professor",
    "doctor", "towards", "using", "based", "within", "across", "between", "through",
    "including", "related", "associated", "general", "public", "social", "human",
    "analysis", "analytical", "application", "applications", "understanding",
    "investigating", "investigation", "elucidating", "towards", "impact", "impacts",
}

_NAME_RE = re.compile(r"\b(prof|dr|professor|mr|mrs|ms)\.?\s+[A-Z][a-z]+")


def _entropy_normalized(probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = float(-np.sum(clipped * np.log(clipped)))
    return entropy / float(np.log(len(clipped)))


def _scores(engine, text: str) -> dict:
    masked = mask_funders(text)
    _rrf, svd, _lex, _query = engine._search(masked)
    probabilities = engine.predict_proba_masked(masked)
    order = np.argsort(-probabilities)
    top_i = int(order[0])
    second_i = int(order[1]) if len(order) > 1 else top_i
    set90, set95 = empirical_sets(engine.classes, probabilities)
    max_sim = float(np.max(svd)) if svd.size else 0.0
    percentile = float(np.searchsorted(engine.novelty_reference, max_sim, side="right") / max(1, engine.novelty_reference.size) * 100)
    neighbour_idx = np.argsort(-svd)[:200]
    return {
        "svd": svd,
        "neighbour_idx": neighbour_idx,
        "probabilities": probabilities,
        "top_council": engine.classes[top_i],
        "top_p": float(probabilities[top_i]),
        "second_p": float(probabilities[second_i]),
        "set90": set90,
        "set95": set95,
        "percentile": percentile,
        "novelty_band": novelty_band(percentile),
        "entropy_normalized": _entropy_normalized(probabilities),
        "max_sim": max_sim,
    }


def _readiness_numbers(engine, text: str, signals: dict, persona: str) -> dict:
    tokens = _terms(text)
    unique = len(set(tokens))
    raw_tokens = TOKEN_RE.findall(text.lower())
    share = sum(1 for token in raw_tokens if token in engine.index) / len(raw_tokens) if raw_tokens else 0.0
    word_count = len(text.split())
    specificity = min(100.0, unique * 4.0 + (20 if share > 0.4 else 0))
    if word_count < 40:
        specificity = min(specificity, 45)
    alignment = signals["top_p"] * 100
    if len(signals["set90"]) == 1:
        alignment = min(100.0, alignment + 8)
    elif len(signals["set90"]) >= 3:
        alignment = max(0.0, alignment - 10)
    originality = max(0.0, 100 - abs(signals["percentile"] - 40) * 1.6)
    lowered = text.lower()

    def hits(words: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in words)

    method_score = 25 * sum(1 for group in (_PROBLEM, _METHOD, _DATA, _IMPACT) if hits(group))
    bridge = min(100.0, signals["second_p"] * 180 + signals["entropy_normalized"] * 40)
    dimensions = [
        {"id": "specificity", "label": "Topic specificity and lexical clarity", "score": round(specificity, 1)},
        {"id": "alignment", "label": "Council and domain alignment", "score": round(alignment, 1)},
        {"id": "originality", "label": "Originality sweet-spot", "score": round(originality, 1)},
        {"id": "method", "label": "Methodology and impact signals", "score": round(float(method_score), 1)},
        {"id": "bridge", "label": "Interdisciplinary bridge potential", "score": round(bridge, 1)},
    ]
    overall = round(sum(item["score"] for item in dimensions) / len(dimensions), 1)
    return {
        "overall": overall,
        "dimensions": dimensions,
        "persona": persona_brief(persona),
        "council": signals["top_council"],
        "probability": round(signals["top_p"], 6),
        "conformal_set_90": list(signals["set90"]),
        "conformal_set_95": list(signals["set95"]),
        "novelty_band": signals["novelty_band"],
        "novelty_percentile": round(signals["percentile"], 2),
    }


def _rows(engine, indices: np.ndarray) -> pd.DataFrame:
    frame = engine.frame.iloc[np.asarray(indices)].copy()
    frame["organisation"] = frame["lead_org_name"].fillna("").map(lambda value: html.unescape(str(value)))
    frame["title_text"] = frame["title"].fillna("").map(lambda value: html.unescape(str(value)))
    frame["funder"] = frame["lead_funder"].fillna("").astype(str)
    frame["year"] = pd.to_numeric(frame["start_year"], errors="coerce")
    frame["award"] = pd.to_numeric(frame["award_gbp"], errors="coerce")
    frame["url"] = frame["gtr_url"].fillna("").astype(str)
    return frame


def _missing_terms(text: str, frame: pd.DataFrame) -> list[str]:
    owned = set(_terms(text))
    counts: Counter[str] = Counter()
    recent = frame.loc[frame["year"].fillna(0) >= 2018]
    pool = recent if len(recent) >= 12 else frame
    for title in pool["title_text"].head(40):
        for term in _terms(title):
            if term in owned or term in _BOILER or len(term) < 5:
                continue
            counts[term] += 1
        for phrase in _bigrams(title):
            parts = phrase.split()
            if any(part in owned or part in _BOILER for part in parts):
                continue
            if all(len(part) >= 4 for part in parts):
                counts[phrase] += 1
    ranked = [term for term, count in counts.most_common(30) if count >= 2] or [term for term, _count in counts.most_common(12)]
    picked: list[str] = []
    stems: set[str] = set()
    for term in ranked:
        stem = term.split()[0]
        if stem in stems:
            continue
        stems.add(stem)
        picked.append(term)
        if len(picked) == 8:
            break
    if len(picked) < 6:
        for term, _count in counts.most_common(40):
            stem = term.split()[0]
            if stem in stems or term in picked:
                continue
            stems.add(stem)
            picked.append(term)
            if len(picked) == 6:
                break
    return picked[:8]


def _precedents(frame: pd.DataFrame, limit: int = 3) -> list[dict]:
    rows = []
    for record in frame.head(limit).itertuples(index=False):
        rows.append({
            "title": record.title_text,
            "organisation": record.organisation,
            "lead_funder": record.funder,
            "gtr_url": record.url,
            "start_year": None if pd.isna(record.year) else int(record.year),
        })
    return rows


def _wave(frame: pd.DataFrame) -> dict:
    usable = frame.loc[frame["year"].notna()].copy()
    if usable.empty:
        return {"band": "insufficient", "momentum": "unclear", "note": "No dated neighbours.", "ratio": None}
    usable["year"] = usable["year"].astype(int)
    latest = int(usable["year"].max())
    recent_n = int((usable["year"] >= latest - 2).sum())
    prior_n = int(((usable["year"] >= latest - 5) & (usable["year"] < latest - 2)).sum())
    ratio = (recent_n / prior_n) if prior_n else None
    if prior_n < 3 or recent_n < 3:
        band, momentum = "insufficient", "unclear"
        note = f"Only {recent_n} projects in {latest - 2}–{latest} and {prior_n} in the three years before."
    elif ratio >= 1.5:
        band, momentum = "Surging", "rising"
        note = f"{recent_n} projects in {latest - 2}–{latest} versus {prior_n} earlier (ratio {ratio:.2f})."
    elif ratio >= 1.15:
        band, momentum = "Rising", "rising"
        note = f"{recent_n} recent projects versus {prior_n} in the previous window (ratio {ratio:.2f})."
    elif ratio >= 0.85:
        band, momentum = "Steady", "steady"
        note = f"Counts are close: {recent_n} recent versus {prior_n} earlier (ratio {ratio:.2f})."
    else:
        band, momentum = "Mature", "cooling"
        note = f"{recent_n} recent projects versus {prior_n} earlier (ratio {ratio:.2f}). The neighbourhood is quieter than it was."
    return {"band": band, "momentum": momentum, "note": note, "ratio": None if ratio is None else round(float(ratio), 3), "recent_projects": recent_n, "prior_projects": prior_n}


def _status(score: float, green: float, amber: float) -> str:
    if score >= green:
        return "Green"
    if score >= amber:
        return "Amber"
    return "Red"


def _roadmap(persona: str, council: str, precedents: list[dict]) -> dict:
    brief = persona_brief(persona)
    reading = ", ".join(f"{row['title']} ({row['organisation']})" for row in precedents[:2]) or "the nearest funded projects"
    plans = {
        "undergrad": {
            "timeline": ["Week 1: read five funded titles and write one sentence you can finish this semester.", "Weeks 2–3: choose one method and one data source.", "Weeks 4–8: do the method on a bounded sample.", "Weeks 9–12: write, and do not copy a multi-year grant shape."],
            "checklist": ["One question", "One method", "One real data source or archive", "Five precedent projects named by organisation", "A limit on what you will not do"],
        },
        "masters": {
            "timeline": ["Weeks 1–2: map the council neighbourhood and cut the question to one comparison.", "Weeks 3–6: methods and a pilot.", "Weeks 7–10: results.", "Weeks 11–12: thesis chapter and a ten-minute talk."],
            "checklist": ["Comparison stated", "Dataset or archive named", "What is out of scope", "Council shortlist checked", "Precedent organisations listed"],
        },
        "phd": {
            "timeline": ["Weeks 1–3: match the question to Studentship or Fellowship neighbours.", "Weeks 4–6: one-page proposal and the 90% council set.", "Weeks 7–9: talk to a supervisor using the precedent list.", "Weeks 10–12: revise the gap so it still has funded neighbours."],
            "checklist": ["Gap that still has neighbours", "Training route named as a hypothesis, not a fact", "90% council set", "Two host organisations", "Eligibility checked by you, not by this tool"],
        },
        "researcher": {
            "timeline": ["Weeks 1–2: place the chapter against the nearest funded projects.", "Weeks 3–6: run the method.", "Weeks 7–9: write the result and the limit.", "Weeks 10–12: answer the red-team questions aloud."],
            "checklist": ["Chapter question", "Method you will actually run", "Precedent list for the supervisor", "Viva answers for the weakest radar axis", "No claim that a score is a funding decision"],
        },
        "international": {
            "timeline": ["Weeks 1–2: learn the council names that match the subject.", "Weeks 3–5: read precedent organisations, not adverts alone.", "Weeks 6–8: draft one page.", "Weeks 9–12: check studentship eligibility yourself. This tool does not."],
            "checklist": ["Council named", "Two UK organisations in the neighbourhood", "One-page idea", "Eligibility source saved", "Visa and fee questions taken to the university, not to this score"],
        },
        "innovator": {
            "timeline": ["Weeks 1–2: name the user and the partner organisation.", "Weeks 3–6: a small pilot.", "Weeks 7–9: cost and route.", "Weeks 10–12: a one-page brief a supervisor and a partner can both read."],
            "checklist": ["User named", "Partner organisation named", "Innovate UK or collaborative neighbours read", "Pilot scope", "What the university owns versus what the partner owns, left for a real agreement"],
        },
    }
    plan = plans.get(persona if persona in PERSONAS else "masters", plans["masters"])
    email = (
        f"Subject: First conversation about a {brief['label']} idea nearest {council}\n\n"
        "Dear [Professor / Supervisor Name],\n\n"
        f"I am shaping a {brief['label']} on work that a local reading of 113,233 UKRI awards places nearest {council}. "
        f"The nearest public projects include {reading}.\n\n"
        "I am not asking you to confirm funding. I would value twenty minutes on whether the scope fits this stage, "
        "and which of those organisations is the right reading list.\n\n"
        "The precedent rows are public sector information licensed under the Open Government Licence v3.0. "
        "The draft wording is mine.\n\n"
        "Kind regards"
    )
    return {
        "stage": brief["label"],
        "timeline": plan["timeline"],
        "checklist": plan["checklist"],
        "email_template": email,
        "note": "The salutation keeps a placeholder. This tool does not store or suggest a person's name.",
    }


def coach(engine, title: str, abstract: str, persona: str = "masters") -> dict:
    text = f"{title}. {abstract}".strip()
    before = _scores(engine, text)
    frame = _rows(engine, before["neighbour_idx"][:80])
    missing = _missing_terms(text, frame)
    simulated = text if not missing else text + ". The draft is simulated to mention " + ", ".join(missing) + "."
    after_probs = engine.predict_proba_masked(mask_funders(simulated))
    after_order = np.argsort(-after_probs)
    after_set90, after_set95 = empirical_sets(engine.classes, after_probs)
    after_signals = {
        **before,
        "probabilities": after_probs,
        "top_council": engine.classes[int(after_order[0])],
        "top_p": float(after_probs[int(after_order[0])]),
        "second_p": float(after_probs[int(after_order[1])]),
        "set90": after_set90,
        "set95": after_set95,
        "entropy_normalized": _entropy_normalized(after_probs),
    }
    before_score = _readiness_numbers(engine, text, before, persona)
    after_score = _readiness_numbers(engine, simulated, after_signals, persona)
    precedents = _precedents(frame, 3)
    return {
        "ok": True,
        "persona": persona_brief(persona),
        "missing_signals": missing,
        "signal_count": len(missing),
        "before": {
            "readiness": before_score["overall"],
            "council": before_score["council"],
            "probability": before_score["probability"],
            "conformal_set_90": before_score["conformal_set_90"],
            "conformal_set_95": before_score["conformal_set_95"],
            "conformal_set_size": len(before_score["conformal_set_90"]),
        },
        "after": {
            "readiness": after_score["overall"],
            "council": after_score["council"],
            "probability": after_score["probability"],
            "conformal_set_90": after_score["conformal_set_90"],
            "conformal_set_size": len(after_score["conformal_set_90"]),
        },
        "delta_readiness": round(after_score["overall"] - before_score["overall"], 1),
        "precedents": precedents,
        "roadmap": _roadmap(persona, before_score["council"], precedents),
        "simulation_note": (
            "Corpus-alignment simulation only. The after-score inserts neighbour vocabulary into a copy of the draft. "
            "It is not a scientific result and not a reason to add a phrase unless it is true of the work."
        ),
    }


def _severity(status: str) -> str:
    return {"Red": "High", "Amber": "Watch", "Green": "Solid"}[status]


def redteam(engine, title: str, abstract: str, persona: str = "masters") -> dict:
    text = f"{title}. {abstract}".strip()
    signals = _scores(engine, text)
    frame = _rows(engine, signals["neighbour_idx"])
    score = _readiness_numbers(engine, text, signals, persona)
    wave = _wave(frame)
    precedents = _precedents(frame, 2)
    cited = precedents[0] if precedents else {"title": "a nearby funded project", "organisation": "a UK organisation"}
    other = precedents[1] if len(precedents) > 1 else cited
    by_id = {item["id"]: item["score"] for item in score["dimensions"]}
    novelty_status = _status(by_id["originality"], 70, 45)
    method_status = _status(by_id["method"], 75, 50)
    council_status = "Green" if len(signals["set90"]) == 1 and signals["top_p"] >= 0.80 else "Amber" if signals["top_p"] >= 0.60 else "Red"
    momentum_status = {"rising": "Green", "steady": "Amber", "cooling": "Red", "unclear": "Amber"}[wave["momentum"]]
    words = len(text.split())
    if persona == "undergrad":
        scope_status = "Red" if words < 40 else "Amber" if words > 280 else "Green"
    elif persona in {"phd", "researcher"}:
        scope_status = "Amber" if words < 80 else "Green"
    else:
        scope_status = "Red" if words < 40 else "Green"
    matrix = [
        {"dimension": "Novelty Differentiation", "status": novelty_status, "fact": f"{signals['novelty_band']} at the {signals['percentile']:.0f}th percentile of real titles."},
        {"dimension": "Stage Scope Fit", "status": scope_status, "fact": f"{words} words for a {persona_brief(persona)['label']}."},
        {"dimension": "Method Clarity", "status": method_status, "fact": f"Method and impact axis {by_id['method']:.0f} / 100."},
        {"dimension": "Council Positioning", "status": council_status, "fact": f"{signals['top_council']} at {signals['top_p']:.0%}; 90% set size {len(signals['set90'])}."},
        {"dimension": "Field Momentum", "status": momentum_status, "fact": f"{wave['band']} ({wave['momentum']}): {wave['note']}"},
    ]
    questions = [
        {
            "question": "Where does this sit against work UKRI has already funded, and what exactly is new?",
            "severity": _severity(novelty_status),
            "why": matrix[0]["fact"],
            "defence": f"Name the difference from {cited['title']} at {cited['organisation']}, then say which part you will not try to repeat.",
        },
        {
            "question": "Is this one council's question, or are you asking two councils to own it?",
            "severity": _severity(council_status),
            "why": matrix[3]["fact"],
            "defence": f"Lead with {signals['top_council']}. Mention the rest of the 90% set only where the method truly crosses, using {other['title']} as the comparison.",
        },
        {
            "question": "What will you measure, on what material, and how will you know it worked?",
            "severity": _severity(method_status),
            "why": matrix[2]["fact"],
            "defence": "Add one sentence each for the problem, the method, the data or archive, and the user. Do not add a method you will not run.",
        },
        {
            "question": "Why this question now, if the nearby funded set is rising, steady, or quieter?",
            "severity": _severity(momentum_status),
            "why": matrix[4]["fact"],
            "defence": "Treat the count as context. A quieter neighbourhood can still be the right dissertation if the method is smaller than the grants around it.",
        },
        {
            "question": f"What will you finish at the {persona_brief(persona)['label']} scale, and what are you leaving out?",
            "severity": _severity(scope_status),
            "why": matrix[1]["fact"] + " " + persona_brief(persona)["scheme_emphasis"],
            "defence": f"Write the limit in one sentence. Use {cited['organisation']} as a reading, not as a template for a larger bid.",
        },
    ]
    return {
        "ok": True,
        "persona": persona_brief(persona),
        "readiness": score["overall"],
        "novelty_band": signals["novelty_band"],
        "wave": wave,
        "questions": questions,
        "matrix": matrix,
        "precedents": precedents,
    }


def hubs(engine, title: str, abstract: str, top_n: int = 10) -> dict:
    text = f"{title}. {abstract}".strip()
    signals = _scores(engine, text)
    frame = _rows(engine, signals["neighbour_idx"][:60])
    grouped: list[dict] = []
    for organisation, part in frame.groupby("organisation"):
        if not str(organisation).strip():
            continue
        priced = part.loc[part["award"].fillna(0) > 0]
        real_values = []
        for amount, year in zip(priced["award"], priced["year"]):
            year_i = None if pd.isna(year) else int(year)
            real_values.append(to_real_gbp(float(amount), year_i) or 0.0)
        council = part["funder"].value_counts().index[0] if len(part) else ""
        links = []
        for record in part.head(2).itertuples(index=False):
            links.append({"title": record.title_text, "gtr_url": record.url, "lead_funder": record.funder})
        grouped.append({
            "organisation": str(organisation),
            "awards": int(len(part)),
            "total_gbp_real_2026": round(float(sum(real_values)), 2),
            "median_gbp_real_2026": round(float(np.median(real_values)), 2) if real_values else 0.0,
            "dominant_council": str(council),
            "precedents": links,
        })
    grouped.sort(key=lambda row: (row["awards"], row["total_gbp_real_2026"]), reverse=True)
    return {
        "ok": True,
        "neighbour_count": int(min(60, len(frame))),
        "price_level": "October 2026 CPIH index 144.6",
        "hubs": grouped[:top_n],
        "note": "Organisations only. Counts are among the nearest funded projects, not a league table of universities.",
    }


def _money(value: float) -> str:
    return f"£{value:,.0f}"


def research_dossier(engine, title: str, abstract: str, persona: str = "masters") -> dict:
    from fundmatch.student import readiness

    text = f"{title}. {abstract}".strip()
    score = readiness(engine, title, abstract, persona)
    topics = suggest_topics(engine, text, persona)
    coach_view = coach(engine, title, abstract, persona)
    review = redteam(engine, title, abstract, persona)
    centres = hubs(engine, title, abstract)
    trend = topic_trends(engine, text)
    lines = [
        f"# {APP_DISPLAY_NAME} research dossier",
        "",
        f"**Idea.** {title or 'Untitled'}",
        f"**Stage.** {persona_brief(persona)['label']}",
        "",
        "## Readiness",
        "",
        f"Overall: {score['overall']} / 100.",
        "",
        "| Axis | Score |",
        "| --- | ---: |",
    ]
    for item in score["dimensions"]:
        lines.append(f"| {item['label']} | {item['score']:.0f} |")
    lines += [
        "",
        "## Council prediction",
        "",
        f"Primary council: {score.get('council') or coach_view['before']['council']} ({coach_view['before']['probability']:.0%}).",
        f"90% set: {', '.join(coach_view['before']['conformal_set_90'])}.",
        f"95% set: {', '.join(coach_view['before']['conformal_set_95'])}.",
        "",
        "## Five working titles",
        "",
    ]
    for topic in topics["topics"]:
        lines.append(
            f"- {topic['working_title']} — {topic['predicted_council']} "
            f"{topic['calibrated_probability']:.0%}, {topic['novelty_band']}."
        )
    lines += ["", "## UK university and research hubs", ""]
    for hub in centres["hubs"]:
        link = hub["precedents"][0]["gtr_url"] if hub["precedents"] else ""
        lines.append(
            f"- {hub['organisation']}: {hub['awards']} nearby awards, "
            f"{_money(hub['total_gbp_real_2026'])} at Oct 2026 prices, "
            f"median {_money(hub['median_gbp_real_2026'])}, mostly {hub['dominant_council']}. {link}"
        )
    lines += [
        "",
        "## Upgrade blueprint",
        "",
        "Missing signals from funded neighbours: " + ", ".join(coach_view["missing_signals"]) + ".",
        f"Before readiness {coach_view['before']['readiness']} / 100 "
        f"({coach_view['before']['council']} {coach_view['before']['probability']:.0%}, "
        f"90% set size {coach_view['before']['conformal_set_size']}).",
        f"After simulated alignment {coach_view['after']['readiness']} / 100 "
        f"({coach_view['after']['council']} {coach_view['after']['probability']:.0%}, "
        f"90% set size {coach_view['after']['conformal_set_size']}).",
        coach_view["simulation_note"],
        "",
        "### Stage roadmap",
        "",
    ]
    for step in coach_view["roadmap"]["timeline"]:
        lines.append(f"- {step}")
    lines += ["", "### Deliverable checklist", ""]
    for item in coach_view["roadmap"]["checklist"]:
        lines.append(f"- {item}")
    lines += ["", "### First-contact email", "", "```", coach_view["roadmap"]["email_template"], "```", "", "## Reviewer questions", ""]
    for item in review["questions"]:
        lines.append(f"- **{item['severity']}.** {item['question']} {item['why']} Defence: {item['defence']}")
    lines += ["", "### Feasibility matrix", ""]
    for item in review["matrix"]:
        lines.append(f"- {item['dimension']}: {item['status']}. {item['fact']}")
    lines += [
        "",
        "## Future-wave trajectory",
        "",
        f"{trend['band']}. {trend['note']}",
        "",
        "## Ethics and licence",
        "",
        "Contains public sector information licensed under the Open Government Licence v3.0.",
        f"{APP_DISPLAY_NAME} is a CS5079 Applied AI submission by Md Shamim Mia (52536283), University of Aberdeen.",
        "It does not predict whether a proposal will be funded, it does not name people, and UKRI does not endorse it.",
        "Organisations and award figures come from the 23 November 2026 Gateway to Research snapshot.",
    ]
    markdown = "\n".join(lines) + "\n"
    return {
        "markdown": markdown,
        "readiness": score,
        "topics": topics["topics"],
        "coach": coach_view,
        "redteam": review,
        "hubs": centres,
        "trends": {"band": trend["band"], "note": trend["note"], "series": trend["series"]},
        "title": title,
        "persona": persona_brief(persona),
    }
