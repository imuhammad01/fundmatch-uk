"""Input checks and the funder-name mask used when the classifier was trained.

The mask and the English-function-word check follow the CS5079 implementation in
docs/reference/fundmatch_core.py. Funder names become the neutral token
``funderx`` so a council cannot be predicted from its own name.
"""

from __future__ import annotations

import re

MIN_WORDS = 30
MIN_DISTINCT_WORDS = 10
# Real English abstracts sit well above 15% function words. The threshold is the
# one measured against the UKRI training text, not a general language detector.
ENGLISH_SHARE_MIN = 0.15
# The CS5079 classifier was trained with ``TfidfVectorizer(stop_words="english")``
# (checked against the original pickled pipeline). scikit-learn removes these words
# BEFORE it builds bigrams, so "cultural and scientific" yields the bigram
# "cultural scientific". The exported meta file listed only seven single letters,
# and v4.0.0 skipped this step, which changed the logits by up to 1.07 and the top
# council on 4 of 70 test texts (red-team finding RT-01). Use the real list.
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SK_STOP
except Exception:  # pragma: no cover - scikit-learn is pinned in requirements.txt
    _SK_STOP = frozenset()
STOP_WORDS = frozenset(_SK_STOP)

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
    # Count real words (letters only), so "!!! !!! ..." or "12345 ..." cannot pass the
    # length check by padding with symbols or numbers (red-team finding RT-03).
    words = [word.lower() for word in _WORD_RE.findall(text) if len(word) >= 2]
    n_words = len(words)
    if n_words == 0 and not text.strip():
        return ["Enter a title and an abstract. Empty text cannot be matched to a council."], []
    if n_words < MIN_WORDS:
        return [
            f"Please give more detail: {n_words} words entered, and at least {MIN_WORDS} are needed "
            "for a reliable council match. Add the aims, the methods, and who the work is for. "
            "Numbers and symbols do not count as words."
        ], []
    if len(set(words)) < MIN_DISTINCT_WORDS:
        return [
            f"The text repeats the same few words ({len(set(words))} different words). "
            f"Please write at least {MIN_DISTINCT_WORDS} different words describing the aims and methods."
        ], []
    if recognised_terms is not None and recognised_terms == 0:
        return [
            "None of these words are in the 200,000-term research vocabulary, so any council score "
            "would be meaningless. Describe the aims and methods in English."
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
