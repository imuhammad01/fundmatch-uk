"""Display-time redaction of personal names (NFR-9).

GtR never gave us person records, but a few public project titles and
lead-organisation names still contain a person's name, for example
"Fellowship for Professor <name>" or a sole trader registered as
"<name> T/A <trading name>". The app must never show researcher names, so these
strings are redacted once when the project table is loaded. The rules are
deliberately narrow (an honorific followed by capitalised words, a "T/A"
sole-trader prefix, an organisation that is only an honorific plus initials and a
surname) so ordinary titles and company names are left alone.
"""

from __future__ import annotations

import re

NAME_TOKEN = r"(?:(?:[A-Z]\.){1,3}|[A-Z][A-Za-z'’\-]+|[A-Z](?=\s))"
HONORIFIC = r"(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor)\.?(?:\s+Sir|\s+Dame)?"
# Honorific + 1 to 4 capitalised tokens ("Professor Sir A B Name").
_TITLE_NAME = re.compile(rf"\b({HONORIFIC})\s+{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,3}}")
# "Sir/Dame Firstname Surname" without another honorific.
_KNIGHT = re.compile(rf"\b(Sir|Dame)\s+{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{1,2}}")
# Sole traders: "D S NAME T/A TRADING NAME" -> keep the trading name only.
_TRADING_AS = re.compile(r"^.*?\bT/A\b\s*", re.IGNORECASE)
# An organisation that is only an honorific plus a name: "Mr J S Name".
_PERSON_ORG = re.compile(rf"^\s*(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,4}}\s*$")
REDACTED = "[name]"


def redact_person_names(text: str) -> str:
    """Replace "Professor <name>"-style mentions with "Professor [name]"."""
    if not text:
        return text
    out = _TITLE_NAME.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
    out = _KNIGHT.sub(lambda m: f"{m.group(1)} {REDACTED}", out)
    return out


def redact_org_name(name: str) -> str:
    """Hide the personal part of sole-trader organisation names."""
    if not name:
        return name
    if _PERSON_ORG.match(name):
        return "Sole trader (name withheld)"
    if re.search(r"\bT/A\b", name, re.IGNORECASE):
        trading = _TRADING_AS.sub("", name).strip(" ()|")
        return trading or "Sole trader (name withheld)"
    return redact_person_names(name)
