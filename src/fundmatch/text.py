"""Input checks and the funder-name mask used when the classifier was trained.

The mask and the English-function-word check follow the CS5079 implementation in
docs/reference/fundmatch_core.py. Funder names become the neutral token
``funderx`` so a council cannot be predicted from its own name.
"""

from __future__ import annotations

import re

MIN_WORDS = 30
# Real English abstracts sit well above 15% function words. The threshold is the
# one measured against the UKRI training text, not a general language detector.
ENGLISH_SHARE_MIN = 0.15
# Single-letter entries cannot be produced by the token pattern ``\b\w\w+\b``.
# They are stored because they are in the exported vectorizer configuration.
STOP_WORDS = frozenset({"e", "g", "h", "i", "l", "n", "s"})

_FUNDER_TERMS = [
    r"engineering and physical sciences research council",
    r"biotechnology and biological sciences research council",
    r"economic and social research council",
    r"medical research council",
    r"arts and humanities research council",
    r"natural environment research council",
    r"science and technology facilities council",
    r"innovate ?uk",
    r"technology strategy board",
    r"uk research and innovation",
    r"research councils? uk",
    r"epsrc",
    r"bbsrc",
    r"esrc",
    r"mrc",
    r"ahrc",
    r"nerc",
    r"stfc",
    r"ukri",
    r"rcuk",
    r"ktp",
    r"knowledge transfer partnerships?",
]
FUNDER_RE = re.compile(r"\b(?:" + "|".join(_FUNDER_TERMS) + r")\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

EN_STOP = frozenset(
    "the and of to in a is for with on that this we will be are by as an from or at it its which "
    "these their our has have been can how into also such using use new not more".split()
)


def mask_funders(text: str) -> str:
    """Replace funder names and funder-only scheme names with a neutral token."""
    return FUNDER_RE.sub(" funderx ", text or "")


def english_share(text: str) -> float:
    """Share of alphabetic word tokens that are common English function words."""
    words = _WORD_RE.findall((text or "").lower())
    if not words:
        return 0.0
    return sum(word in EN_STOP for word in words) / len(words)


def proposal_text(title: str, abstract: str) -> str:
    title, abstract = (title or "").strip(), (abstract or "").strip()
    return f"{title}. {abstract}".strip(". ")


def check_input(title: str, abstract: str, recognised_terms: int | None = None) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)``.

    Empty or very short text is an error. Non-English text and text with almost
    no research vocabulary are warnings: the score is still returned, with the
    warning beside it. Chemical formulas, Latin names and UK spellings stay
    above the English threshold because the function-word share, not the
    alphabet, is what is measured.
    """
    text = proposal_text(title, abstract)
    n_words = len(text.split())
    if n_words == 0:
        return ["Enter a title and an abstract. Empty text cannot be matched to a council."], []
    if n_words < MIN_WORDS:
        return [
            f"Please give more detail: {n_words} words entered, and at least {MIN_WORDS} are needed "
            "for a reliable council match. Add the aims, the methods, and who the work is for."
        ], []
    warnings: list[str] = []
    share = english_share(text)
    if share < ENGLISH_SHARE_MIN:
        warnings.append(
            "This text does not look like English. The model was trained on English UKRI abstracts, "
            "so a German, French, Spanish or gibberish input is not a reliable match. "
            "An English summary of the aims and methods works best."
        )
    if recognised_terms is not None and recognised_terms < 10:
        warnings.append(
            "Very few of these words are in the 200,000-term research vocabulary. "
            "The council ranking is shown, but it is a weak comparison. Describe the aims and methods "
            "in the language of the field."
        )
    return [], warnings
