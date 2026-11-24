"""Own-code collector for the UKRI Gateway to Research API v7.

``Accept: application/vnd.rcuk.gtr.json-v7``. Page size 100. One to two
requests a second, with back-off on 429 and 5xx, and a checkpoint per page so
a run can resume. Person endpoints are never requested (NFR-9).

The 23 November 2026 collection recorded 158,710 projects, 158,710 funds and
97,102 organisations. This module is the method. The shipped product reads the
cleaned parquet rather than calling the API on startup.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ACCEPT = "application/vnd.rcuk.gtr.json-v7"
USER_AGENT = "FundMatchUK/1.0 (CS5079 University of Aberdeen; educational; own collector)"
PAGE_SIZE = 100
PAUSE_S = 0.6  # about 1.6 requests/second, under the stated 10/second ceiling


def fetch_page(endpoint: str, page: int, pause: float = PAUSE_S) -> dict:
    url = f"https://gtr.ukri.org/gtr/api/{endpoint}?p={page}&s={PAGE_SIZE}"
    last: Exception | None = None
    for attempt in range(6):
        time.sleep(pause)
        request = urllib.request.Request(url, headers={"Accept": ACCEPT, "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(min(30, 2 ** attempt))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"failed {url}: {last}")


def collect(endpoint: str, dest: Path, pause: float = PAUSE_S) -> dict:
    """Download every page of one endpoint. Existing page files are skipped."""
    dest.mkdir(parents=True, exist_ok=True)
    first = fetch_page(endpoint, 1, pause)
    total = int(first["totalPages"])
    (dest / "00001.json").write_text(json.dumps(first), encoding="utf-8")
    for page in range(2, total + 1):
        path = dest / f"{page:05d}.json"
        if path.exists():
            continue
        path.write_text(json.dumps(fetch_page(endpoint, page, pause)), encoding="utf-8")
    return {"endpoint": endpoint, "pages": total, "total_size": int(first["totalSize"])}
