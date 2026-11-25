"""Own-code collector for the UKRI Gateway to Research API v7 (FR-1, FR-3, FR-4).

``Accept: application/vnd.rcuk.gtr.json-v7``. Page size 100 (the API maximum).
At most one request every ``PAUSE_S`` seconds (about 1.6 requests a second,
inside the requirement's 1-2 requests a second and well under the published
10 a second), exponential back-off on 429, 5xx and network errors, and a
checkpoint file per page so a stopped run resumes where it left off. Person
endpoints are never requested (NFR-9).

The 23 November 2026 collection recorded 158,710 projects, 158,710 funds and
97,102 organisations (``training/logs/collection_log.json``). The full
multi-endpoint script used for that run, with the funds/organisations join,
is ``training/collect_gtr.py`` and ``training/build_dataset.py``. This module
is the same method packaged with an injectable fetcher so it can be unit
tested without the network. The shipped app reads the cleaned parquet and
never calls the API on start-up.

Usage::

    python -m fundmatch.pipeline.collect_gtr projects data/raw/projects --max-pages 3
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

BASE_URL = "https://gtr.ukri.org/gtr/api"
ACCEPT = "application/vnd.rcuk.gtr.json-v7"
USER_AGENT = "FundMatchUK/1.0 (CS5079 University of Aberdeen; educational; own collector)"
PAGE_SIZE = 100
PAUSE_S = 0.6  # about 1.6 requests/second, under the stated 10/second ceiling
RETRY_STATUS = (429, 500, 502, 503, 504)
MAX_RETRIES = 6
ALLOWED_ENDPOINTS = ("projects", "funds", "organisations")  # never "persons" (NFR-9)


class RateLimiter:
    """Keeps at least ``min_gap`` seconds between consecutive calls to :meth:`wait`."""

    def __init__(self, min_gap: float = PAUSE_S, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.min_gap = float(min_gap)
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            gap = now - self._last
            if gap < self.min_gap:
                self._sleep(self.min_gap - gap)
                now = self._clock()
        self._last = now


def backoff_seconds(attempt: int, cap: float = 30.0) -> float:
    """1, 2, 4, 8, 16, 30, 30 ... seconds for attempt 0, 1, 2 ..."""
    return float(min(cap, 2 ** attempt))


def page_url(endpoint: str, page: int) -> str:
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"endpoint {endpoint!r} is not collected (allowed: {ALLOWED_ENDPOINTS})")
    return f"{BASE_URL}/{endpoint}?p={page}&s={PAGE_SIZE}"


def _http_get_json(url: str, timeout: float = 90) -> dict:
    request = urllib.request.Request(url, headers={"Accept": ACCEPT, "User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_page(endpoint: str, page: int, pause: float = PAUSE_S, *,
               getter: Callable[[str], dict] = _http_get_json,
               sleep: Callable[[float], None] = time.sleep,
               limiter: RateLimiter | None = None) -> dict:
    """Fetch one page, retrying with exponential back-off on 429/5xx/network errors."""
    url = page_url(endpoint, page)
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        if limiter is not None:
            limiter.wait()
        else:
            sleep(pause)
        try:
            return getter(url)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in RETRY_STATUS:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            last = exc
        sleep(backoff_seconds(attempt))
    raise RuntimeError(f"failed {url} after {MAX_RETRIES} attempts: {last}")


def collect(endpoint: str, dest: Path, pause: float = PAUSE_S, *, max_pages: int | None = None,
            fetch: Callable[..., dict] | None = None, log_path: Path | None = None) -> dict:
    """Download every page of one endpoint into ``dest`` (one JSON file per page).

    Pages already on disk are skipped, so the run resumes after an interruption
    (checkpointing). Writes are atomic (``.tmp`` then rename). A dated entry is
    merged into ``log_path`` (default ``dest.parent / 'collection_log.json'``).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(pause)
    fetch = fetch or (lambda ep, page: fetch_page(ep, page, pause, limiter=limiter))
    started = time.time()
    first_path = dest / "00001.json"
    if first_path.exists():
        first = json.loads(first_path.read_text(encoding="utf-8"))
        fetched = 0
    else:
        first = fetch(endpoint, 1)
        _atomic_write(first_path, first)
        fetched = 1
    total_pages = int(first["totalPages"])
    n_pages = total_pages if max_pages is None else min(total_pages, int(max_pages))
    skipped = 0
    for page in range(2, n_pages + 1):
        path = dest / f"{page:05d}.json"
        if path.exists():
            skipped += 1
            continue
        _atomic_write(path, fetch(endpoint, page))
        fetched += 1
    entry = {
        "endpoint": endpoint,
        "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api_total_records": int(first.get("totalSize", 0)),
        "api_total_pages": total_pages,
        "pages_requested": n_pages,
        "pages_fetched_this_run": fetched,
        "pages_resumed_from_checkpoint": skipped + (0 if fetched else 1),
        "seconds": round(time.time() - started, 1),
        "pause_s": pause,
    }
    write_collection_log(log_path or dest.parent / "collection_log.json", entry)
    return entry


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def write_collection_log(path: Path, entry: dict) -> dict:
    """Merge one endpoint entry into the JSON collection log and return the whole log."""
    path = Path(path)
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    log[entry["endpoint"]] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def iter_records(dest: Path, key: str = "project"):
    """Yield the records stored in the page files of one endpoint, in page order."""
    for path in sorted(Path(dest).glob("[0-9][0-9][0-9][0-9][0-9].json")):
        yield from json.loads(path.read_text(encoding="utf-8")).get(key, [])


def save_snapshot(frame, out_dir: Path, collected_utc: str | None = None, name: str = "gtr_projects") -> dict:
    """Save a dated Parquet snapshot plus a JSON sidecar with the date and row count (FR-4)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = collected_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    day = stamp[:10]
    path = out_dir / f"{name}_{day}.parquet"
    frame.to_parquet(path, index=False)
    meta = {"file": path.name, "collected_utc": stamp, "rows": int(len(frame)), "columns": list(map(str, frame.columns))}
    (out_dir / f"{name}_{day}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect one GtR endpoint with checkpoints.")
    parser.add_argument("endpoint", choices=ALLOWED_ENDPOINTS)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--pause", type=float, default=PAUSE_S)
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.endpoint, args.dest, args.pause, max_pages=args.max_pages), indent=2))


if __name__ == "__main__":
    main()
