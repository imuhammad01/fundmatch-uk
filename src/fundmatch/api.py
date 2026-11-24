"""Headless inference API. The process does not store pasted abstracts."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from fundmatch.brand import APP_DISPLAY_NAME, APP_VERSION
from fundmatch.demos import DEMOS
from fundmatch.engine import DISCLAIMER, OGL, get_engine
from fundmatch.futuristic import coach as coach_idea
from fundmatch.futuristic import hubs as research_hubs
from fundmatch.futuristic import redteam as redteam_idea
from fundmatch.futuristic import research_dossier

app = FastAPI(title=f"{APP_DISPLAY_NAME} API", version=APP_VERSION, description=DISCLAIMER)


class TextBody(BaseModel):
    title: str = ""
    abstract: str = ""
    query: str = ""
    councils: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None
    scheme: str | None = None
    same_scheme: bool = False
    cpih: bool = False
    top_k: int = Field(default=10, ge=1, le=50)
    compare: str | None = None
    council: str | None = None
    versus: str | None = None
    quantized: bool = False
    add_terms: str = ""
    remove_terms: str = ""
    target_council: str | None = None
    proposals: list[dict] = Field(default_factory=list)
    records: list[dict] = Field(default_factory=list)
    persona: str = "masters"
    interest: str = ""
    field_a: str = ""
    field_b: str = ""


@app.get("/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "projects": int(len(engine.frame)),
        "rss_mb": engine.rss_mb(),
        "snapshot": "2026-11-23",
        "temperature": engine.temperature,
        "ogl": OGL,
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/v1/analyze")
def analyze(body: TextBody) -> dict:
    return get_engine().analyze(
        body.title, body.abstract, councils=body.councils, year_min=body.year_min, year_max=body.year_max,
        scheme=body.scheme, same_scheme=body.same_scheme, cpih=body.cpih, top_k=body.top_k, compare=body.compare,
        quantized=body.quantized,
    ).model_dump()


@app.post("/api/v1/predict")
def predict(body: TextBody) -> dict:
    full = get_engine().analyze(body.title, body.abstract, top_k=1)
    return {
        "ok": full.ok,
        "errors": full.errors,
        "warnings": full.warnings,
        "councils": [row.model_dump() for row in full.councils],
        "top_council": full.top_council,
        "confidence": full.confidence,
        "entropy": full.entropy,
        "latency_ms": full.latency_ms,
    }


@app.post("/api/v1/explain")
def explain(body: TextBody) -> dict:
    engine = get_engine()
    council = body.council or "EPSRC"
    view = engine.explain_council(body.title, body.abstract, council)
    payload = {"explanation": view.model_dump()}
    if body.versus:
        payload["contrast"] = engine.contrast(body.title, body.abstract, council, body.versus).model_dump()
    return payload


@app.post("/api/v1/search")
def search(body: TextBody) -> dict:
    rows = get_engine().search(
        body.query or f"{body.title} {body.abstract}",
        councils=body.councils, year_min=body.year_min, year_max=body.year_max,
        scheme=body.scheme, top_k=body.top_k,
    )
    return {"results": [row.model_dump() for row in rows]}


@app.get("/api/v1/benchmarks")
def benchmarks() -> dict:
    engine = get_engine()
    return {"model_info": engine.model_info, "summary_keys": list(engine.summary)}


@app.get("/api/v1/corpus/stats")
def corpus() -> dict:
    return get_engine().corpus_stats()


@app.get("/api/v1/demos")
def demos() -> dict:
    return {"demos": DEMOS}


@app.post("/api/v1/whatif")
def what_if(body: TextBody) -> dict:
    engine = get_engine()
    simulated = engine.simulate_what_if(
        body.title, body.abstract, add_terms=body.add_terms, remove_terms=body.remove_terms,
        target_council=body.target_council,
    )
    suggested = None
    if body.target_council:
        suggested = engine.suggest_counterfactuals(body.title, body.abstract, body.target_council)
    return {"simulation": simulated, "suggestion": suggested}


@app.post("/api/v1/compare")
def compare(body: TextBody) -> dict:
    return get_engine().compare_proposals(body.proposals)


@app.post("/api/v1/batch")
def batch(body: TextBody) -> dict:
    return get_engine().analyze_batch(body.records)


@app.get("/api/v1/metrics")
def metrics() -> dict:
    return get_engine().metrics()


@app.post("/api/v1/topics")
def topics(body: TextBody) -> dict:
    text = body.interest or f"{body.title}. {body.abstract}"
    return get_engine().suggest_topics(text, body.persona)


@app.post("/api/v1/trends")
def trends(body: TextBody) -> dict:
    return get_engine().topic_trends(body.interest or f"{body.title}. {body.abstract}")


@app.post("/api/v1/fuse")
def fuse_fields(body: TextBody) -> dict:
    return get_engine().fuse(body.field_a, body.field_b)


@app.post("/api/v1/readiness")
def readiness_score(body: TextBody) -> dict:
    return get_engine().readiness(body.title, body.abstract, body.persona)


@app.post("/api/v1/starter-pack")
def starter(body: TextBody) -> dict:
    return get_engine().starter_pack(body.title, body.abstract, body.persona)


@app.get("/api/v1/personas")
def personas() -> dict:
    from fundmatch.student import PERSONAS
    return {"personas": [{"id": key, "label": value["label"]} for key, value in PERSONAS.items()]}


@app.post("/api/v1/coach")
def coach(body: TextBody) -> dict:
    return coach_idea(get_engine(), body.title, body.abstract or body.interest, body.persona)


@app.post("/api/v1/redteam")
def redteam(body: TextBody) -> dict:
    return redteam_idea(get_engine(), body.title, body.abstract or body.interest, body.persona)


@app.post("/api/v1/hubs")
def hubs(body: TextBody) -> dict:
    return research_hubs(get_engine(), body.title, body.abstract or body.interest)


@app.post("/api/v1/dossier")
def dossier(body: TextBody) -> dict:
    return research_dossier(get_engine(), body.title, body.abstract or body.interest, body.persona)
