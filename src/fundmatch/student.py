"""Student-stage tools. Every number comes from the loaded UKRI table or the exported model.

Suggested titles are working titles composed from words that actually appear in
the nearest funded titles. They are not existing grants and not a promise that
a council will fund them. Precedent rows are real projects.
"""

from __future__ import annotations

import html
import re
from collections import Counter

import numpy as np

from fundmatch.cpih import to_real_gbp
from fundmatch.text import EN_STOP, TOKEN_RE, mask_funders

PERSONAS: dict[str, dict[str, str]] = {
    "undergrad": {
        "label": "Undergraduate / final-year dissertation",
        "scheme_emphasis": "A one-semester question with one method and one real data source.",
        "milestones": "Weeks 1–2 read five funded projects. Weeks 3–4 fix the question. Weeks 5–8 do the method. Weeks 9–10 check it. Weeks 11–12 write.",
        "jargon": "DTP and CDT are doctoral training routes, not undergraduate grants. For a dissertation, use funded projects as reading, not as a template for a multi-year bid.",
    },
    "masters": {
        "label": "MSc taught project",
        "scheme_emphasis": "A three-to-four month thesis: one comparison, one dataset or archive, and a clear limit on what you will not do.",
        "milestones": "Weeks 1–2 map the neighbourhood. Weeks 3–5 methods. Weeks 6–9 results. Weeks 10–12 thesis and a short talk.",
        "jargon": "A studentship is a funded PhD place. A research grant is usually held by staff. Your thesis can still learn from both.",
    },
    "phd": {
        "label": "PhD applicant / proposal",
        "scheme_emphasis": "Look at Studentship and Fellowship rows. A PhD question needs a gap that still has funded neighbours, not an empty part of the map.",
        "milestones": "Weeks 1–3 council and DTP fit. Weeks 4–6 proposal. Weeks 7–9 supervisor conversations using the precedent list. Weeks 10–12 refine the 90% council set.",
        "jargon": "DTP: a doctoral training partnership, usually a group of universities with studentships. CDT: a centre for doctoral training, cohort-based and often with a partner. KTP: a knowledge transfer partnership, a graduate in a company with an academic supervisor.",
    },
    "international": {
        "label": "International student in the UK",
        "scheme_emphasis": "Start from the council that matches the subject, then read how UK studentships are advertised. Eligibility rules differ by scheme and are not decided here.",
        "milestones": "Weeks 1–2 learn the council names. Weeks 3–5 read precedent projects and their organisations. Weeks 6–8 draft a one-page idea. Weeks 9–12 match it to advertised studentships yourself.",
        "jargon": "UKRI is the umbrella. The seven research councils plus Innovate UK sit under it. A studentship pays a doctoral place. A grant usually pays a project led by staff. This tool does not check visas or fee status.",
    },
    "researcher": {
        "label": "Current PhD researcher",
        "scheme_emphasis": "A chapter-sized question inside a PhD: one gap, one method, and a precedent list a supervisor can check.",
        "milestones": "Weeks 1–2 place the chapter against funded neighbours. Weeks 3–6 run the method. Weeks 7–9 write the result. Weeks 10–12 prepare the viva defence from the red-team questions.",
        "jargon": "A fellowship is usually held by the doctoral candidate or a staff member. A research grant is usually held by staff. This score does not decide which route you can apply to.",
    },
    "innovator": {
        "label": "Student founder / spinout",
        "scheme_emphasis": "Innovate UK, Collaborative R&D, and Knowledge Transfer Partnership rows matter more than a pure studentship.",
        "milestones": "Weeks 1–2 name the user and the partner. Weeks 3–6 a small pilot. Weeks 7–9 cost and route to market. Weeks 10–12 a one-page brief a supervisor can read.",
        "jargon": "KTP places a graduate in a company with a university supervisor. Collaborative R&D is a business-led project with an academic partner. Neither is a dissertation mark.",
    },
}

_EXTRA_STOP = EN_STOP | {
    "project", "research", "study", "using", "based", "towards", "toward", "into", "from",
    "with", "without", "across", "within", "between", "their", "this", "that", "will",
    "develop", "development", "new", "novel", "approach", "system", "systems", "model",
    "many", "much", "more", "most", "other", "such", "very", "also", "via", "high", "low",
    "large", "small", "first", "next", "early", "late", "real", "main", "multi", "single",
    "intrinsically", "does", "did", "what", "when", "where", "which", "than", "then",
    "only", "just", "even", "still", "well", "make", "made", "makes", "making", "over",
    "long", "body", "mediating", "unpacking", "your", "during", "title", "work", "role",
}

_METHOD = ("method", "measure", "model", "experiment", "trial", "interview", "sensor", "dataset", "archive", "simulate", "sequence", "cohort")
_DATA = ("data", "dataset", "sample", "archive", "sensor", "survey", "cohort", "image", "record", "glider", "scan")
_IMPACT = ("patient", "clinical", "policy", "community", "climate", "energy", "health", "industry", "fleet", "school", "city")
_PROBLEM = ("limited", "poorly", "risk", "gap", "failure", "delay", "loss", "uncertain", "constraint", "challenge")


def persona_brief(persona: str) -> dict:
    if persona == "expert":
        persona = "researcher"
    key = persona if persona in PERSONAS else "masters"
    row = PERSONAS[key]
    return {"id": key, **row}


def _terms(text: str) -> list[str]:
    out = []
    for token in TOKEN_RE.findall((text or "").lower()):
        if token in _EXTRA_STOP or len(token) < 4:
            continue
        if token.isnumeric():
            continue
        out.append(token)
    return out


def _bigrams(text: str) -> list[str]:
    tokens = [tok for tok in TOKEN_RE.findall((text or "").lower()) if tok not in _EXTRA_STOP and len(tok) > 3]
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def _project_public(row, similarity: float | None = None) -> dict:
    year = int(row["start_year"]) if row["start_year"] == row["start_year"] else None
    try:
        import pandas as pd
        if pd.isna(row["start_year"]):
            year = None
        else:
            year = int(row["start_year"])
    except (TypeError, ValueError):
        year = None
    amount = float(row["award_gbp"]) if row["award_gbp"] == row["award_gbp"] else 0.0
    return {
        "project_id": str(row["project_id"]),
        "title": html.unescape(str(row["title"] or "")),
        "lead_funder": str(row["lead_funder"]),
        "organisation": html.unescape(str(row["lead_org_name"] or "")),
        "grant_category": html.unescape(str(row["grant_category"] or "")),
        "award_gbp": amount,
        "start_year": year,
        "gtr_url": str(row["gtr_url"] or ""),
        "similarity": similarity,
    }


def _stem(term: str) -> str:
    word = term.lower()
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        return word[:-1]
    return word


def _nice(phrase: str) -> str:
    text = " ".join(phrase.split())
    if not text:
        return "Funded work"
    return text[0].upper() + text[1:]


def _confidence(probability: float) -> str:
    if probability >= 0.80:
        return "high"
    if probability >= 0.60:
        return "moderate"
    return "low"


def _score_blurb(engine, blurb: str, density: float) -> dict:
    """Council probabilities only. No second search over the 113,233 rows."""
    from fundmatch.intelligence import novelty_band

    probabilities = engine.predict_proba_masked(mask_funders(blurb))
    index = int(np.argmax(probabilities))
    probability = float(probabilities[index])
    percentile = float(
        np.searchsorted(engine.novelty_reference, density, side="right")
        / max(1, engine.novelty_reference.size)
        * 100
    )
    return {
        "predicted_council": engine.classes[index],
        "calibrated_probability": round(probability, 6),
        "confidence": _confidence(probability),
        "novelty_band": novelty_band(percentile),
    }


def suggest_topics(engine, interest: str, persona: str = "masters", n: int = 5) -> dict:
    """Five working titles from one neighbour search. Each title is scored with logits only."""
    brief = persona_brief(persona)
    neighbours = engine.search(interest, top_k=40)
    if not neighbours:
        return {"persona": brief, "topics": [], "note": "No neighbours. Add a few more concrete words."}
    slices = [neighbours[i::n] for i in range(n)]
    bags: list[Counter[str]] = []
    for group in slices:
        bag: Counter[str] = Counter()
        for item in group:
            bag.update(_terms(item.title))
            bag.update(_bigrams(item.title))
        bags.append(bag)
    used_titles: set[str] = set()
    used_stems: set[str] = set()
    topics = []
    for group, bag in zip(slices, bags):
        if not group:
            continue
        phrases = [term for term, _count in bag.most_common(12)] or [group[0].lead_funder.lower()]
        unigrams = [term for term, _count in bag.most_common() if " " not in term] or phrases
        lead = next((term for term in unigrams if _stem(term) not in used_stems), None)
        if lead is None:
            lead = next((term for term in phrases if _stem(term) not in used_stems), phrases[0])
        used_stems.add(_stem(lead))
        second = next(
            (term for term in unigrams if _stem(term) not in used_stems),
            group[0].grant_category or "funded uk work",
        )
        weak_mod = {"scalable", "dimensional", "spatial", "mediating", "protected", "digital", "fast", "deep", "cross", "reduced"}
        for phrase, count in bag.most_common():
            parts = phrase.split()
            if count < 2 or len(parts) != 2 or any(part in weak_mod for part in parts):
                continue
            if second in parts and _stem(lead) not in phrase and _stem(phrase) not in used_stems:
                second = phrase
                break
        used_stems.add(_stem(second))
        title = f"{_nice(lead)}: a bounded study of {second}"
        if title.lower() in used_titles:
            title = f"{_nice(lead)} in {group[0].lead_funder.lower()} settings: {second}"
        spare = 2
        while title.lower() in used_titles:
            title = f"{group[0].title.rstrip('.')} — student focus {spare} on {second}"
            spare += 1
        used_titles.add(title.lower())
        precedents = [
            {
                "project_id": item.project_id,
                "title": item.title,
                "organisation": item.lead_org_name,
                "lead_funder": item.lead_funder,
                "gtr_url": item.gtr_url,
                "award_gbp": item.award_gbp,
                "start_year": item.start_year,
            }
            for item in group[:2]
        ]
        blurb = title + ". " + " ".join(item.title for item in group[:3])
        density = float(np.mean([item.svd_cosine for item in group])) if group else 0.0
        scored = _score_blurb(engine, blurb, density)
        councils = Counter(item.lead_funder for item in group)
        topics.append({
            "working_title": title,
            "terms_from_real_titles": phrases[:5],
            "predicted_council": scored["predicted_council"],
            "calibrated_probability": scored["calibrated_probability"],
            "confidence": scored["confidence"],
            "novelty_band": scored["novelty_band"],
            "neighbour_density": round(density, 4),
            "why_timely": _why_timely(group),
            "scope": brief["scheme_emphasis"],
            "precedents": precedents,
            "cluster_council_mode": councils.most_common(1)[0][0],
            "disclaimer": "Working title composed from words in real funded titles. It is not an existing grant.",
        })
    return {
        "persona": brief,
        "interest": interest,
        "topics": topics,
        "ogl": "Precedent rows contain public sector information licensed under the Open Government Licence v3.0.",
    }


def _why_timely(group) -> str:
    years = [item.start_year for item in group if item.start_year]
    if not years:
        return "These neighbours have no start year, so timing is not claimed."
    latest = max(years)
    recent = sum(1 for year in years if year >= latest - 2)
    return (
        f"{recent} of {len(years)} dated neighbours in this slice start in {latest - 2}–{latest}. "
        "That is a count, not a forecast."
    )


def topic_trends(engine, text: str) -> dict:
    """Year-over-year counts inside the 200 nearest funded projects."""
    _rrf, svd, _lex, _q = engine._search(mask_funders(text))
    scores = svd.copy()
    order = np.argsort(-scores)[:200]
    frame = engine.frame.iloc[order]
    years = frame["start_year"]
    import pandas as pd
    usable = frame.loc[pd.to_numeric(years, errors="coerce").notna()].copy()
    usable["year"] = pd.to_numeric(usable["start_year"], errors="coerce").astype(int)
    if usable.empty:
        return {"band": "insufficient", "note": "No dated neighbours.", "series": [], "rising_terms": []}
    counts = usable.groupby("year").size()
    totals = []
    for year, part in usable.groupby("year"):
        nominal = float(part["award_gbp"].fillna(0).sum())
        real = 0.0
        for amount in part["award_gbp"].fillna(0):
            real += to_real_gbp(float(amount), int(year)) or 0.0
        totals.append({
            "year": int(year),
            "projects": int(counts.loc[year]),
            "total_gbp_nominal": round(nominal, 2),
            "total_gbp_real_2026": round(real, 2),
        })
    latest = max(row["year"] for row in totals)
    recent_years = [row for row in totals if row["year"] >= latest - 2]
    prior_years = [row for row in totals if latest - 5 <= row["year"] < latest - 2]
    recent_n = sum(row["projects"] for row in recent_years)
    prior_n = sum(row["projects"] for row in prior_years)
    ratio = (recent_n / prior_n) if prior_n else None
    if prior_n < 3 or recent_n < 3:
        band = "insufficient"
        note = f"Only {recent_n} projects in {latest - 2}–{latest} and {prior_n} in the three years before. Too few to name a wave."
    elif ratio >= 1.5:
        band = "Surging"
        note = f"{recent_n} projects in {latest - 2}–{latest} versus {prior_n} in {latest - 5}–{latest - 3} (ratio {ratio:.2f})."
    elif ratio >= 1.15:
        band = "Rising"
        note = f"{recent_n} projects in {latest - 2}–{latest} versus {prior_n} in the previous three-year window (ratio {ratio:.2f})."
    elif ratio >= 0.85:
        band = "Steady"
        note = f"Counts are close: {recent_n} recent versus {prior_n} in the previous window (ratio {ratio:.2f})."
    else:
        band = "Mature"
        note = f"{recent_n} projects in {latest - 2}–{latest} versus {prior_n} earlier (ratio {ratio:.2f}). The neighbourhood is quieter than it was."
    recent_terms = Counter()
    older_terms = Counter()
    for row in usable.itertuples(index=False):
        bag = Counter(_terms(str(row.title)))
        if row.year >= latest - 2:
            recent_terms.update(bag)
        else:
            older_terms.update(bag)
    rising = []
    for term, count in recent_terms.items():
        if count < 2:
            continue
        before = older_terms.get(term, 0)
        if count > before:
            rising.append({"term": term, "recent": count, "earlier": before})
    rising.sort(key=lambda item: (item["recent"] - item["earlier"], item["recent"]), reverse=True)
    return {
        "band": band,
        "note": note,
        "window_recent": [latest - 2, latest],
        "window_prior": [latest - 5, latest - 3],
        "recent_projects": recent_n,
        "prior_projects": prior_n,
        "ratio": None if ratio is None else round(ratio, 3),
        "series": totals,
        "rising_terms": rising[:8],
        "neighbour_projects": int(len(order)),
    }


def readiness(engine, title: str, abstract: str, persona: str = "masters") -> dict:
    text = f"{title}. {abstract}".strip()
    analysis = engine.analyze(title, abstract, top_k=10)
    tokens = _terms(text)
    unique = len(set(tokens))
    raw_tokens = TOKEN_RE.findall(text.lower())
    share = sum(1 for token in raw_tokens if token in engine.index) / len(raw_tokens) if raw_tokens else 0
    specificity = min(100.0, unique * 4.0 + (20 if share > 0.4 else 0))
    if analysis.word_count < 40:
        specificity = min(specificity, 45)
    top_p = analysis.councils[0].probability if analysis.councils else 0.0
    alignment = top_p * 100
    if len(analysis.conformal_set_90) == 1:
        alignment = min(100.0, alignment + 8)
    elif len(analysis.conformal_set_90) >= 3:
        alignment = max(0.0, alignment - 10)
    percentile = float((analysis.novelty or {}).get("percentile") or 0)
    originality = max(0.0, 100 - abs(percentile - 40) * 1.6)
    lowered = text.lower()
    def hits(words: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in words)
    method_score = 25 * sum(1 for group in (_PROBLEM, _METHOD, _DATA, _IMPACT) if hits(group))
    second = analysis.councils[1].probability if len(analysis.councils) > 1 else 0
    entropy = analysis.entropy_normalized or 0
    bridge = min(100.0, second * 180 + entropy * 40)
    dims = [
        {"id": "specificity", "label": "Topic specificity and lexical clarity", "score": round(specificity, 1)},
        {"id": "alignment", "label": "Council and domain alignment", "score": round(alignment, 1)},
        {"id": "originality", "label": "Originality sweet-spot", "score": round(originality, 1)},
        {"id": "method", "label": "Methodology and impact signals", "score": round(float(method_score), 1)},
        {"id": "bridge", "label": "Interdisciplinary bridge potential", "score": round(bridge, 1)},
    ]
    overall = round(sum(item["score"] for item in dims) / len(dims), 1)
    weakest = min(dims, key=lambda item: item["score"])
    return {
        "ok": analysis.ok,
        "errors": analysis.errors,
        "persona": persona_brief(persona),
        "overall": overall,
        "dimensions": dims,
        "tip": _tip(weakest["id"], persona_brief(persona)),
        "council": analysis.top_council,
        "probability": top_p if analysis.councils else None,
        "conformal_set_90": analysis.conformal_set_90,
        "novelty": analysis.novelty,
    }


def _tip(dimension: str, brief: dict) -> str:
    tips = {
        "specificity": "Name the place, the material, the archive, or the disease. Vague words such as 'system' do not separate you from thousands of titles.",
        "alignment": "Say who the work is for in council language: a patient cohort, a field site, a company process, or an archive. Then re-read the 90% set.",
        "originality": "If the score is low because you sit in a crowded part of the map, narrow the slice. If it is low because you are far from every funded abstract, add a method someone has already used.",
        "method": "Add one sentence each for the problem, the method, the data or archive, and who would use the result.",
        "bridge": "A bridge needs a second field that is actually in the work. Name it, or accept a single-council project.",
    }
    return tips.get(dimension, "") + " " + brief["scheme_emphasis"]


def fuse(engine, field_a: str, field_b: str) -> dict:
    masked_a = mask_funders(field_a)
    masked_b = mask_funders(field_b)
    _ra, svd_a, _la, _qa = engine._search(masked_a)
    _rb, svd_b, _lb, _qb = engine._search(masked_b)
    bridge = np.minimum(svd_a, svd_b)
    order = np.argsort(-bridge)[:12]
    rows = []
    for idx in order:
        row = engine.frame.iloc[int(idx)]
        rows.append({
            **_project_public(row, similarity=round(float(bridge[idx]), 4)),
            "similarity_a": round(float(svd_a[idx]), 4),
            "similarity_b": round(float(svd_b[idx]), 4),
        })
    top_a = engine.search(field_a, top_k=30)
    top_b = engine.search(field_b, top_k=30)
    councils_a = {item.lead_funder for item in top_a}
    councils_b = {item.lead_funder for item in top_b}
    both = sorted(councils_a & councils_b)
    angles = []
    if rows:
        terms_a = _terms(field_a)[:2] or [field_a]
        terms_b = _terms(field_b)[:2] or [field_b]
        angles.append(
            f"A bounded study of {terms_a[0]} alongside {terms_b[0]}, using the same kind of material as {rows[0]['title']}."
        )
        if len(rows) > 1:
            angles.append(
                f"Compare {rows[0]['organisation']} and {rows[1]['organisation']} as host types for work that has to satisfy both fields."
            )
    return {
        "field_a": field_a,
        "field_b": field_b,
        "bridge_councils": both,
        "councils_only_a": sorted(councils_a - councils_b),
        "councils_only_b": sorted(councils_b - councils_a),
        "projects": rows,
        "angles": angles,
        "note": "Bridge score is the lower of the two abstract-space cosines. A project must be near both fields, not just one.",
    }


def starter_pack(engine, title: str, abstract: str, persona: str = "masters") -> dict:
    brief = persona_brief(persona)
    analysis = engine.analyze(title, abstract, top_k=5, cpih=True)
    score = readiness(engine, title, abstract, persona)
    topics = suggest_topics(engine, f"{title}. {abstract}", persona, n=5)
    precedents = []
    for item in analysis.similar[:5]:
        precedents.append({
            "title": item.title,
            "organisation": item.lead_org_name,
            "lead_funder": item.lead_funder,
            "gtr_url": item.gtr_url,
            "award_gbp": item.award_gbp,
            "grant_category": item.grant_category,
        })
    glossary = []
    for item in analysis.similar:
        for term in item.shared_terms:
            if term not in {row["term"] for row in glossary}:
                glossary.append({"term": term, "meaning": "This phrase is in your text and in a nearby funded title."})
            if len(glossary) >= 8:
                break
    pack = {
        "display_name": "ScholarPulse UK",
        "persona": brief,
        "summary": analysis.student_summary,
        "council": analysis.top_council,
        "probability": analysis.councils[0].probability if analysis.councils else None,
        "conformal_set_90": analysis.conformal_set_90,
        "readiness": score,
        "topics": topics["topics"],
        "precedents": precedents,
        "hosts": analysis.host_institutions[:5],
        "glossary": glossary,
        "roadmap": brief["milestones"],
        "ogl": "Contains public sector information licensed under the Open Government Licence v3.0.",
        "disclaimer": "Not official UKRI advice. Precedent projects are real. Working titles are not grants.",
    }
    pack["markdown"] = render_starter_markdown(pack, title)
    return pack


def render_starter_markdown(pack: dict, title: str) -> str:
    lines = [
        f"# {pack['display_name']} research starter pack",
        "",
        f"**Idea.** {title or 'Untitled'}",
        f"**Stage.** {pack['persona']['label']}",
        "",
        pack["summary"] or "",
        "",
        f"Readiness score: {pack['readiness']['overall']} / 100.",
        pack["readiness"]["tip"],
        "",
        "## Five real precedent projects",
    ]
    for index, item in enumerate(pack["precedents"], start=1):
        lines.append(
            f"{index}. {item['title']} — {item['lead_funder']}, {item['organisation']}. {item['gtr_url']}"
        )
    lines += ["", "## Host organisations"]
    for host in pack["hosts"]:
        lines.append(f"- {host['organisation']} ({host['projects']} of the neighbour set, mostly {host['top_council']})")
    lines += ["", "## Words in common with funded titles"]
    for item in pack["glossary"]:
        lines.append(f"- {item['term']}: {item['meaning']}")
    lines += ["", "## Twelve-week outline", pack["roadmap"], "", pack["ogl"], pack["disclaimer"]]
    return "\n".join(lines)
