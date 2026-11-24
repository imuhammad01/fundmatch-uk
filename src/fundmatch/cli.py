"""Rich terminal for ScholarPulse UK. The command name stays fundmatch."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fundmatch.demos import DEMOS, resolve_demo
from fundmatch.engine import COUNCIL_COLOURS, DISCLAIMER, IFS, get_engine

app = typer.Typer(add_completion=False, no_args_is_help=True, help="ScholarPulse UK — UKRI council intelligence. Command: fundmatch.")
console = Console()


def _gbp(value: float | None) -> str:
    if value is None:
        return "—"
    return f"£{value:,.0f}"


def _render(title: str, analysis, export_csv: Path | None, export_brief: Path | None, as_json: bool, student: bool = False) -> None:
    engine = get_engine()
    if as_json:
        console.print_json(analysis.model_dump_json())
    else:
        if analysis.errors:
            console.print(Panel("\n".join(analysis.errors), title="Cannot score", border_style="red"))
            return
        for warning in analysis.warnings:
            console.print(Panel(warning, title="Warning", border_style="yellow"))
        head = Text()
        head.append(analysis.top_council or "—", style=f"bold {COUNCIL_COLOURS.get(analysis.top_council, 'white')}")
        head.append(f"  {analysis.councils[0].probability:.1%}  ·  {analysis.confidence} confidence  ·  {analysis.latency_ms:.0f} ms")
        console.print(Panel(head, title="Top council", subtitle=analysis.confidence_note or ""))
        table = Table(title="Calibrated probabilities", expand=True)
        table.add_column("Council")
        table.add_column("Probability", justify="right")
        table.add_column("Open calls")
        for row in analysis.councils:
            table.add_row(row.council, f"{row.probability:.1%}", row.funding_finder_url)
        console.print(table)
        if analysis.explanation:
            pos = ", ".join(f"{item.feature} ({item.contribution:+.3f})" for item in analysis.explanation.positive[:6])
            neg = ", ".join(f"{item.feature} ({item.contribution:+.3f})" for item in analysis.explanation.negative[:4])
            console.print(Panel(f"For: {pos}\nAgainst: {neg}", title=f"X-Ray · {analysis.explanation.council}"))
        if analysis.contrast:
            toward = ", ".join(item.feature for item in analysis.contrast.toward_a[:5])
            against = ", ".join(item.feature for item in analysis.contrast.toward_b[:5])
            console.print(Panel(
                f"Toward {analysis.contrast.council_a}: {toward}\nToward {analysis.contrast.council_b}: {against}",
                title="Contrast",
            ))
        if analysis.award and analysis.award.median is not None:
            band = analysis.award
            console.print(Panel(
                f"P25–P75 {_gbp(band.p25)} – {_gbp(band.p75)}\n"
                f"Median {_gbp(band.median)} · P10–P90 {_gbp(band.p10)} – {_gbp(band.p90)}\n"
                f"{band.zero_note}\n{band.caveat}\nBasis: {band.basis}. {band.cpih_target}.",
                title="Award band",
            ))
        projects = Table(title="Similar funded projects", expand=True)
        for column in ("Sim", "Council", "Award", "Organisation", "Title"):
            projects.add_column(column, overflow="fold")
        for row in analysis.similar:
            projects.add_row(f"{row.similarity:.2f}", row.lead_funder, _gbp(row.award_gbp), row.lead_org_name, row.title)
        console.print(projects)
        if analysis.conformal_set_90:
            console.print(Panel(
                f"90% set: {', '.join(analysis.conformal_set_90)}\n"
                f"95% set: {', '.join(analysis.conformal_set_95)}\n"
                f"{analysis.conformal_explanation}",
                title="Plausible councils",
            ))
        if analysis.novelty:
            console.print(Panel(
                f"{analysis.novelty.get('band')}\n{analysis.novelty.get('note')}",
                title="Novelty",
            ))
        if student and analysis.student_summary:
            hosts = ", ".join(item["organisation"] for item in analysis.host_institutions[:3]) or "—"
            schemes = ", ".join(f"{item['scheme']} {item['share']:.0%}" for item in analysis.scheme_fit[:3]) or "—"
            console.print(Panel(
                f"{analysis.student_summary}\n\nHosts among neighbours: {hosts}\nSchemes: {schemes}",
                title="Student coach",
            ))
        console.print(f"[dim]{DISCLAIMER} Innovate UK calls: {IFS}[/dim]")
    if export_csv and analysis.similar:
        export_csv.write_bytes(engine.to_csv(analysis.similar))
        console.print(f"Wrote {export_csv}")
    if export_brief and analysis.ok:
        export_brief.write_text(engine.to_markdown(title, analysis), encoding="utf-8")
        console.print(f"Wrote {export_brief}")


@app.command()
def analyze(
    title: str = typer.Option("", "--title"),
    abstract: str = typer.Option("", "--abstract"),
    demo: str = typer.Option("", "--demo", help="Demo id, or 'all' is the demo command."),
    same_scheme: bool = typer.Option(False, "--same-scheme"),
    cpih: bool = typer.Option(False, "--cpih"),
    top_k: int = typer.Option(10, "--top-k"),
    compare_council: str = typer.Option("", "--compare-council"),
    council: list[str] = typer.Option(None, "--council"),
    year_min: int | None = typer.Option(None, "--year-min"),
    year_max: int | None = typer.Option(None, "--year-max"),
    scheme: str = typer.Option("", "--scheme"),
    export_csv: Path | None = typer.Option(None, "--export-csv"),
    export_brief: Path | None = typer.Option(None, "--export-brief"),
    as_json: bool = typer.Option(False, "--json"),
    student: bool = typer.Option(False, "--student"),
    quantized: bool = typer.Option(False, "--quantized"),
) -> None:
    """Score a title and abstract against the eight UKRI councils."""
    if demo:
        chosen = resolve_demo(demo)
        if chosen is None:
            raise typer.BadParameter(f"Unknown demo '{demo}'. Ids: {', '.join(item['id'] for item in DEMOS)}")
        title, abstract = chosen["title"], chosen["abstract"]
        if same_scheme and not scheme:
            scheme = chosen["scheme"]
    engine = get_engine()
    result = engine.analyze(
        title, abstract, councils=council or None, year_min=year_min, year_max=year_max,
        scheme=scheme or None, same_scheme=same_scheme, cpih=cpih, top_k=top_k,
        compare=compare_council or None, quantized=quantized,
    )
    _render(title, result, export_csv, export_brief, as_json, student=student)
    if not result.ok:
        raise typer.Exit(code=2)


@app.command("search")
def search_cmd(
    query: str = typer.Option(..., "--query"),
    top_k: int = typer.Option(10, "--top-k"),
    council: list[str] = typer.Option(None, "--council"),
    year_min: int | None = typer.Option(None, "--year-min"),
    year_max: int | None = typer.Option(None, "--year-max"),
    category: str = typer.Option("", "--category"),
) -> None:
    """Search the 113,233 funded projects."""
    rows = get_engine().search(query, councils=council or None, year_min=year_min, year_max=year_max, scheme=category or None, top_k=top_k)
    table = Table(title="Search", expand=True)
    for column in ("Sim", "Council", "Year", "Award", "Title"):
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(f"{row.similarity:.2f}", row.lead_funder, str(row.start_year or ""), _gbp(row.award_gbp), row.title)
    console.print(table)


@app.command()
def explain(
    title: str = typer.Option("", "--title"),
    abstract: str = typer.Option("", "--abstract"),
    council: str = typer.Option("EPSRC", "--council"),
    vs: str = typer.Option("", "--vs"),
    demo: str = typer.Option("", "--demo"),
) -> None:
    """Token attributions for one council, optionally contrasted with another."""
    if demo:
        chosen = next(item for item in DEMOS if item["id"] == demo)
        title, abstract = chosen["title"], chosen["abstract"]
    engine = get_engine()
    view = engine.explain_council(title, abstract, council)
    console.print(Panel(
        "For: " + ", ".join(f"{i.feature} ({i.contribution:+.3f})" for i in view.positive) + "\n"
        "Against: " + ", ".join(f"{i.feature} ({i.contribution:+.3f})" for i in view.negative),
        title=f"Why {council}",
    ))
    if vs:
        contrast = engine.contrast(title, abstract, council, vs)
        console.print(Panel(
            f"Toward {council}: " + ", ".join(i.feature for i in contrast.toward_a) + "\n"
            f"Toward {vs}: " + ", ".join(i.feature for i in contrast.toward_b),
            title="Contrast",
        ))


@app.command()
def demo(which: str = typer.Argument("all")) -> None:
    """Run one curated proposal, or all five."""
    chosen = DEMOS if which == "all" else [item for item in DEMOS if item["id"] == which or which.lower() in item["label"].lower()]
    if not chosen:
        raise typer.BadParameter("Unknown demo")
    engine = get_engine()
    for item in chosen:
        result = engine.analyze(item["title"], item["abstract"], top_k=5)
        console.rule(item["label"])
        _render(item["title"], result, None, None, False)


@app.command()
def benchmark() -> None:
    """Cold-start was already paid by loading. Report warm latency, RSS and int8 fidelity."""
    from fundmatch.quantize import export_int8, fidelity

    engine = get_engine()
    sample = engine.analyze(DEMOS[0]["title"], DEMOS[0]["abstract"])
    times = []
    for item in DEMOS:
        result = engine.analyze(item["title"], item["abstract"])
        times.append(result.latency_ms)
    report = export_int8(engine)
    texts = [f"{item['title']}. {item['abstract']}" for item in DEMOS]
    # Add real project titles so the fidelity set is not only the five demos.
    texts += engine.frame["title"].head(40).tolist()
    fit = fidelity(engine, texts, report.pop("coef_hat"))
    payload = {
        "projects": int(len(engine.frame)),
        "cold_start_s": round(engine.cold_s, 3),
        "warm_latency_ms_median": round(float(np_median(times)), 2),
        "warm_latency_ms_min": round(min(times), 2),
        "rss_mb": engine.rss_mb(),
        "sample_latency_ms": sample.latency_ms,
        "temperature": engine.temperature,
        "int8": {**{k: v for k, v in report.items() if k != "coef_hat"}, **fit},
        "coursework_macro_f1": engine.model_info.get("test_macro_f1"),
        "coursework_ece": "0.0336 -> 0.0180",
        "warm_int8_latency_ms": engine.analyze(DEMOS[0]["title"], DEMOS[0]["abstract"], quantized=True).latency_ms,
    }
    console.print_json(json.dumps(payload))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8741) -> None:
    """Start the FastAPI inference service."""
    import uvicorn
    uvicorn.run("fundmatch.api:app", host=host, port=port)


@app.command()
def whatif(
    demo: str = typer.Option("", "--demo"),
    title: str = typer.Option("", "--title"),
    abstract: str = typer.Option("", "--abstract"),
    add: str = typer.Option("", "--add"),
    remove: str = typer.Option("", "--remove"),
    target: str = typer.Option(..., "--target"),
) -> None:
    """Show how adding or removing phrases moves the council probabilities."""
    if demo:
        chosen = resolve_demo(demo)
        if chosen is None:
            raise typer.BadParameter(f"Unknown demo '{demo}'")
        title, abstract = chosen["title"], chosen["abstract"]
    engine = get_engine()
    result = engine.simulate_what_if(title, abstract, add_terms=add, remove_terms=remove, target_council=target)
    table = Table(title="What-if probability shift")
    table.add_column("Council")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Delta", justify="right")
    for row in result["deltas"]:
        table.add_row(row["council"], f"{row['before']:.1%}", f"{row['after']:.1%}", f"{row['delta']:+.1%}")
    console.print(table)
    console.print(Panel(result["honesty"], title="Keep the science honest"))


@app.command()
def compare(demos: str = typer.Option(..., "--demos", help="Comma-separated demo ids.")) -> None:
    """Compare two to five demo proposals side by side."""
    engine = get_engine()
    proposals = []
    for name in [part.strip() for part in demos.split(",") if part.strip()]:
        chosen = resolve_demo(name)
        if chosen is None:
            raise typer.BadParameter(f"Unknown demo '{name}'")
        proposals.append({**chosen, "label": chosen["label"]})
    result = engine.compare_proposals(proposals)
    table = Table(title="Compare arena", expand=True)
    for column in ("Idea", "Council", "p", "90% set", "Novelty", "2026 median", "Top host"):
        table.add_column(column, overflow="fold")
    for row in result["rows"]:
        table.add_row(
            str(row["label"]), str(row["top_council"]), f"{row['probability']:.0%}" if row["probability"] else "—",
            ", ".join(row["conformal_set_90"]), str(row["novelty_band"]),
            _gbp(row["median_gbp_2026"]), str(row["top_host"]),
        )
    console.print(table)


@app.command()
def batch(
    input_path: Path = typer.Option(..., "--input"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Score a CSV or JSONL file of title, abstract rows. Abstracts are not stored."""
    import pandas as pd
    if input_path.suffix.lower() == ".jsonl":
        frame = pd.read_json(input_path, lines=True)
    else:
        frame = pd.read_csv(input_path)
    records = frame.to_dict(orient="records")
    result = get_engine().analyze_batch(records)
    pd.DataFrame(result["rows"]).to_csv(output, index=False)
    console.print(f"Wrote {result['n']} rows to {output}")


@app.command()
def topics(
    interest: str = typer.Option(..., "--interest"),
    persona: str = typer.Option("masters", "--persona"),
) -> None:
    """Suggest five working titles from the nearest real funded projects."""
    result = get_engine().suggest_topics(interest, persona)
    for index, topic in enumerate(result["topics"], start=1):
        probability = topic["calibrated_probability"]
        shown = f"{probability:.0%}" if probability is not None else "—"
        precedent = topic["precedents"][0]["title"] if topic["precedents"] else "—"
        console.print(Panel(
            f"{topic['working_title']}\nCouncil {topic['predicted_council']} "
            f"({shown}, {topic['confidence']})\n"
            f"{topic['why_timely']}\nPrecedent: {precedent}",
            title=f"Topic {index}",
        ))


@app.command()
def trends(
    interest: str = typer.Option(..., "--interest"),
) -> None:
    """Year-over-year counts in the nearest funded neighbourhood."""
    result = get_engine().topic_trends(interest)
    console.print(Panel(f"{result['band']}\n{result['note']}", title="Funding wave"))


@app.command()
def fuse(
    field_a: str = typer.Option(..., "--a"),
    field_b: str = typer.Option(..., "--b"),
) -> None:
    """Find funded projects that sit near two fields at once."""
    result = get_engine().fuse(field_a, field_b)
    console.print(Panel(
        "Councils in both neighbourhoods: " + ", ".join(result["bridge_councils"]),
        title=f"{field_a} × {field_b}",
    ))
    table = Table(title="Bridge projects", expand=True)
    for column in ("Bridge", "Council", "Organisation", "Title"):
        table.add_column(column, overflow="fold")
    for row in result["projects"][:8]:
        table.add_row(f"{row['similarity']:.2f}", row["lead_funder"], row["organisation"], row["title"])
    console.print(table)


@app.command()
def score(
    title: str = typer.Option("", "--title"),
    abstract: str = typer.Option("", "--abstract"),
    demo: str = typer.Option("", "--demo"),
    persona: str = typer.Option("masters", "--persona"),
) -> None:
    """Score a draft on five plain-English dimensions."""
    if demo:
        chosen = resolve_demo(demo)
        if chosen is None:
            raise typer.BadParameter(f"Unknown demo '{demo}'")
        title, abstract = chosen["title"], chosen["abstract"]
    result = get_engine().readiness(title, abstract, persona)
    table = Table(title=f"Readiness {result['overall']} / 100")
    table.add_column("Dimension")
    table.add_column("Score", justify="right")
    for item in result["dimensions"]:
        table.add_row(item["label"], f"{item['score']:.0f}")
    console.print(table)
    console.print(Panel(result["tip"], title="Next edit"))


@app.command("starter-pack")
def starter_pack_cmd(
    demo: str = typer.Option("", "--demo"),
    title: str = typer.Option("", "--title"),
    abstract: str = typer.Option("", "--abstract"),
    persona: str = typer.Option("masters", "--persona"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Write a markdown starter pack. Precedent projects are real UKRI rows."""
    if demo:
        chosen = resolve_demo(demo)
        if chosen is None:
            raise typer.BadParameter(f"Unknown demo '{demo}'")
        title, abstract = chosen["title"], chosen["abstract"]
    pack = get_engine().starter_pack(title, abstract, persona)
    if output:
        output.write_text(pack["markdown"], encoding="utf-8")
        console.print(f"Wrote {output}")
    else:
        console.print(pack["markdown"])


def _load_text(demo: str, title: str, abstract: str) -> tuple[str, str]:
    if demo:
        chosen = resolve_demo(demo)
        if chosen is None:
            raise typer.BadParameter(f"Unknown demo '{demo}'")
        return chosen["title"], chosen["abstract"]
    return title, abstract


@app.command()
def coach(
    demo: str = typer.Option("", "--demo"),
    title: str = typer.Option("", "--title"),
    abstract: str = typer.Option("", "--abstract"),
    persona: str = typer.Option("masters", "--persona"),
) -> None:
    """Show missing neighbour terms and a before/after alignment simulation."""
    from fundmatch.futuristic import coach as coach_idea
    title, abstract = _load_text(demo, title, abstract)
    result = coach_idea(get_engine(), title, abstract, persona)
    console.print(Panel(
        f"Before {result['before']['readiness']} / 100 · {result['before']['council']} {result['before']['probability']:.0%}\n"
        f"After {result['after']['readiness']} / 100 · {result['after']['council']} {result['after']['probability']:.0%}\n"
        f"Signals: {', '.join(result['missing_signals'])}\n{result['simulation_note']}",
        title="Upgrade coach",
    ))


@app.command()
def redteam(
    demo: str = typer.Option("", "--demo"),
    title: str = typer.Option("", "--title"),
    abstract: str = typer.Option("", "--abstract"),
    persona: str = typer.Option("masters", "--persona"),
) -> None:
    """Five viva questions and a five-dimension risk matrix."""
    from fundmatch.futuristic import redteam as redteam_idea
    title, abstract = _load_text(demo, title, abstract)
    result = redteam_idea(get_engine(), title, abstract, persona)
    for item in result["questions"]:
        console.print(Panel(f"{item['why']}\n{item['defence']}", title=f"{item['severity']} · {item['question']}"))


@app.command()
def hubs(
    demo: str = typer.Option("", "--demo"),
    title: str = typer.Option("", "--title"),
    abstract: str = typer.Option("", "--abstract"),
) -> None:
    """Organisations in the nearest funded neighbourhood."""
    from fundmatch.futuristic import hubs as research_hubs
    title, abstract = _load_text(demo, title, abstract)
    result = research_hubs(get_engine(), title, abstract)
    table = Table(title="UK research hubs", expand=True)
    for column in ("Awards", "Oct 2026 £", "Council", "Organisation"):
        table.add_column(column, overflow="fold")
    for row in result["hubs"]:
        table.add_row(str(row["awards"]), f"£{row['total_gbp_real_2026']:,.0f}", row["dominant_council"], row["organisation"])
    console.print(table)


@app.command()
def dossier(
    demo: str = typer.Option("", "--demo"),
    title: str = typer.Option("", "--title"),
    abstract: str = typer.Option("", "--abstract"),
    persona: str = typer.Option("masters", "--persona"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Write the one-file student research dossier."""
    from fundmatch.futuristic import research_dossier
    title, abstract = _load_text(demo, title, abstract)
    result = research_dossier(get_engine(), title, abstract, persona)
    if output:
        output.write_text(result["markdown"], encoding="utf-8")
        console.print(f"Wrote {output}")
    else:
        console.print(result["markdown"])


@app.command()
def ui(port: int = 8765) -> None:
    """Start the Streamlit demo."""
    import subprocess
    import sys
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port), "--server.headless", "true"], check=False)


def np_median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
