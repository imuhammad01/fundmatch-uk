"""V5 re-audit: checks that the restyled app still meets the product requirements.

Each test names the requirement it covers. They run the real Streamlit page with
``AppTest``, so they test what a user sees, not only the engine.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fundmatch import insights  # noqa: E402
from fundmatch.demos import DEMOS  # noqa: E402
from fundmatch.engine import get_engine  # noqa: E402
from fundmatch.text import mask_funders  # noqa: E402

FLAT = (
    "Notes", "This project studies people, places, data and change over time in many settings, working with "
    "partners to understand what matters, why it matters and how it might be improved for communities "
    "and organisations across the country.",
)


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def _page(title: str, abstract: str):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.session_state["bootstrapped"] = True
    at.session_state["title"], at.session_state["abstract"] = title, abstract
    at.session_state["scheme"] = "Research Grant"
    at.session_state["_scored"] = True
    at.run()
    return at


def test_fr19_empty_and_short_input_get_a_clear_message_on_the_v5_page():
    for title, abstract in (("", ""), ("Short", "Too few words here.")):
        at = _page(title, abstract)
        assert not at.exception
        shown = " ".join(e.value for e in at.error) + " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info)
        assert "30" in shown or "Paste" in shown or "Type an idea" in shown, (title, shown[:200])


def test_fr20_low_confidence_warning_is_in_the_v5_mission_brief(engine):
    result = engine.analyze(*FLAT)
    assert result.ok and result.confidence in {"low", "moderate"}
    at = _page(*FLAT)
    assert not at.exception
    warned = [e.value for e in at.error] + [w.value for w in at.warning]
    assert any("confidence" in text.lower() for text in warned)
    strong = _page(DEMOS[0]["title"], DEMOS[0]["abstract"])
    assert not [e for e in strong.error if "confidence" in e.value.lower()]


def test_fr21_to_fr26_still_reachable_from_the_matcher_tab():
    at = _page(DEMOS[0]["title"], DEMOS[0]["abstract"])
    assert not at.exception
    assert at.slider(key="top_k").value == 10  # FR-21: ten similar projects by default
    assert at.multiselect(key="councils").value  # FR-25 council filter
    assert at.slider(key="years").value  # FR-25 year filter
    assert len(at.get("download_button")) >= 3  # FR-26: CSV, Markdown and dossier downloads
    page = " ".join(m.value for m in at.markdown)
    assert "Topic funding by year" in page  # FR-23
    assert "Award band" in page  # FR-22
    captions = " ".join(c.value for c in at.caption).lower()
    assert "not a funding quote" in captions or "not a quote" in captions or "quote" in captions
    demos = [b for b in at.button if (b.key or "").startswith("demo_")]
    assert len(demos) == len(DEMOS) >= 6  # FR-24


def test_every_chart_has_labelled_axes_nfr8(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    figures = []
    original = st.plotly_chart

    def spy(fig, *args, **kwargs):
        figures.append(fig)
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(st, "plotly_chart", spy)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    for tool in ("Draft checklist", "Vocabulary", "Precedents", "Idea vs idea", "Explain"):
        at.session_state["studio_tool"] = tool
        at.run()
    at.button(key="what_go").click().run()
    assert figures
    for fig in figures:
        data = fig.to_dict()
        kinds = {trace.get("type", "scatter") for trace in data["data"]}
        if kinds & {"indicator", "scatterpolar"}:
            continue
        layout = data["layout"]
        assert (layout.get("xaxis", {}).get("title") or {}).get("text"), kinds
        assert (layout.get("yaxis", {}).get("title") or {}).get("text"), kinds


def _luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_text_contrast_meets_wcag_aa_nfr8():
    css = (ROOT / "src" / "fundmatch" / "assets" / "neon.css").read_text(encoding="utf-8")
    colours = dict(re.findall(r"--sp-(text|muted|teal|violet|sky):\s*(#[0-9a-fA-F]{6})", css))
    for background in ("#05070d", "#0e1420", "#121820"):
        for name in ("text", "muted", "teal", "violet", "sky"):
            assert _contrast(colours[name], background) >= 4.5, (name, background)
    assert _contrast("#cbd5e1", "#0e1420") >= 4.5


def test_wont_have_items_are_not_claimed():
    """No success prediction, no proposal writing, no researcher ranking."""
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [ROOT / "app.py", *(ROOT / "src" / "fundmatch").glob("*.py")]
        if path.name not in {"cli.py"}
    )
    forbidden = [
        r"chance of (?:being )?funded(?<!not a chance of being funded)",
        r"likelihood of (?:being )?funded",
        r"probability of (?:being )?funded",
        r"will (?:probably |likely )?be funded",
        r"success rate",
        r"rewrit(?:e|ten) (?:your|the) (?:text|abstract|proposal)",
        r"write (?:your|the) proposal for you",
        r"rank(?:s|ing)? (?:researchers|investigators|applicants|people)",
    ]
    for pattern in forbidden:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            context = text[max(0, match.start() - 60):match.end() + 20].lower()
            assert "not " in context or "no " in context or "never" in context or "does not" in context, (pattern, context)
    at = _page(DEMOS[0]["title"], DEMOS[0]["abstract"])
    about = " ".join(m.value for m in at.markdown)
    assert "does not predict whether a grant will be funded" in about
    assert "does not write or edit proposals" in about
    assert "does not rank or profile researchers" in about
    # The studio never returns rewritten text: checklist, radar and drivers hold no draft text.
    engine = get_engine()
    for result in (
        insights.abstract_checklist(engine, DEMOS[0]["title"], DEMOS[0]["abstract"]),
        insights.keyword_radar(engine, DEMOS[0]["title"], DEMOS[0]["abstract"]),
    ):
        assert DEMOS[0]["abstract"][:60] not in repr(result)


def test_nfr9_nfr10_footer_and_no_people_in_v5_outputs(engine):
    at = _page(DEMOS[0]["title"], DEMOS[0]["abstract"])
    page = " ".join(m.value for m in at.markdown)
    assert "Open Government Licence v3.0" in page and "does not endorse" in page and "not official UKRI funding advice" in page
    honorific = re.compile(r"\b(?:Prof|Professor|Dr|Mr|Mrs|Ms)\.?\s+[A-Z][a-z]+")
    for demo in DEMOS:
        result = engine.analyze(demo["title"], demo["abstract"], top_k=50)
        blob = " ".join(f"{p.title} {p.lead_org_name} {p.department}" for p in result.similar)
        assert not honorific.search(blob), demo["id"]
        for col in ("pi_name", "orcid", "surname", "investigator"):
            assert col not in result.model_dump_json().lower()
    pdf_text = insights.pdf_brief(DEMOS[0]["title"], DEMOS[0]["abstract"], engine.analyze(DEMOS[0]["title"], DEMOS[0]["abstract"]))
    assert pdf_text[:5] == b"%PDF-"


def test_nfr9_user_text_is_never_written_to_disk_or_logged():
    for path in (ROOT / "app.py", ROOT / "src" / "fundmatch" / "insights.py"):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"open\([^)]*[\"'][wa]b?[\"']", source), path
        for banned in ("write_text(", "write_bytes(", "logging.", "logger.", "to_parquet(", ".to_csv(\""):
            assert banned not in source, (path.name, banned)


def test_search_vector_is_identical_after_the_memory_fix(engine):
    """NFR-3: the query projection no longer copies the 64 x 200,000 matrix, and gives the same numbers."""
    titles = engine.frame["title"].astype(str).sample(40, random_state=3).tolist() + [DEMOS[0]["abstract"], "", "zzzz qqqq"]
    for text in titles:
        masked = mask_funders(text)
        matrix = engine.vectorize([masked])
        old = np.asarray(matrix @ engine.c_rec.T, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(old))
        expected = old / norm if norm else old
        _rrf, _svd, _lex, q = engine._search(masked)
        assert q.shape == expected.shape
        assert np.allclose(q, expected, atol=1e-6, rtol=0)
