"""ONS CPIH series L522 (2015 = 100) deflator to a 2026 price level.

The annual index is the official ONS annual average, release 18 November 2026.
A full-year 2026 average is not in that release (the last annual observation is
2025 = 138.0). Real-terms 2026 pounds therefore use the latest published month,
October 2026 = 144.6, as the target price level. That figure is an official
monthly index, not an estimated annual average.

Source: https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l522/mm23
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

# October 2026, the last monthly observation in the 18 November 2026 release.
TARGET_2026 = 144.6
TARGET_LABEL = "October 2026 CPIH (index 144.6), the latest month in the 18 November 2026 ONS release"
SOURCE_URL = "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l522/mm23"


def _csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "cpih_l522.csv"


@lru_cache(maxsize=1)
def annual_index() -> dict[int, float]:
    """Official annual CPIH averages. Years are four-digit keys in the ONS file."""
    table: dict[int, float] = {}
    with _csv_path().open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or not row[0].strip().isdigit() or len(row[0].strip()) != 4:
                continue
            if not row[1].strip():
                continue
            table[int(row[0])] = float(row[1])
    if 2015 not in table or table[2015] != 100.0:
        raise RuntimeError("CPIH annual table is missing the 2015 = 100 base")
    if 2025 not in table:
        raise RuntimeError("CPIH annual table is missing 2025")
    return table


def factor(start_year: int | None) -> tuple[float, str, bool]:
    """Return ``(multiplier, index_year_used, extrapolated)``.

    ``award_2026 = award_nominal * TARGET_2026 / CPIH(start year)``.
    """
    table = annual_index()
    if start_year is None:
        return 1.0, "unknown", True
    if start_year >= 2026:
        return 1.0, "2026-10", False
    if start_year in table:
        return TARGET_2026 / table[start_year], str(start_year), False
    years = sorted(table)
    if start_year < years[0]:
        return TARGET_2026 / table[years[0]], str(years[0]), True
    # Between published years this file has every year from 1988, so this is a guard.
    return TARGET_2026 / table[years[-1]], str(years[-1]), True


def to_real_gbp(amount: float | None, start_year: int | None) -> float | None:
    if amount is None:
        return None
    multiplier, _, _ = factor(start_year)
    return round(float(amount) * multiplier, 2)


CAVEAT = (
    "Historical benchmark from similar funded projects, not a funding quote and not a prediction "
    "that a proposal will be awarded. UKRI does not publish this figure."
)
