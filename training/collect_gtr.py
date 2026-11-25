"""
collect_gtr.py - collect UKRI Gateway to Research (GtR) data with our own code.

CS5079 Applied AI, Assessment 1, Task 1 (group project "FundMatch UK").

What it does
------------
1. Pages through the public GtR API v7 endpoint `/projects` (100 records per page,
   the maximum the API accepts) and caches every page as gzipped JSON under
   data/raw/projects/page_XXXXX.json.gz.
2. Pages through `/funds` and `/organisations` the same way. Award amounts are NOT
   stored on project records; each fund record links to its project with a
   `rel="FUNDED"` link, so we join funds to projects by project id later
   (see build_dataset.py).

Politeness / robustness
-----------------------
* A fixed minimum gap between requests per endpoint (default 1.0 s, i.e. about
  1 request per second, well under GtR's published limit).
* Retries with exponential back-off on network errors, HTTP 429 and 5xx.
* Resumable: pages already on disk are skipped, so the script can be stopped and
  rerun at any time ("caching" + "checkpointing").
* A JSON log (data/raw/collection_log.json) records date, totals and timings.

No API key is needed. Data is (c) UKRI, Open Government Licence v3.0.
No ready-made dataset (Kaggle, GtR-Extended, bulk CSV export) is used.

Usage
-----
    python collect_gtr.py --endpoint projects            # all pages
    python collect_gtr.py --endpoint funds --max-pages 50
    python collect_gtr.py --endpoint both --delay 1.0   # projects + funds
    python collect_gtr.py --endpoint organisations      # lead organisation names

Written by the group; the retry pattern follows the standard `requests`
documentation idea of back-off (https://requests.readthedocs.io) but is our own code.
"""
import argparse
import gzip
import json
import os
import threading
import time
from datetime import datetime, timezone

import requests

BASE = "https://gtr.ukri.org/gtr/api"
HEADERS = {
    "Accept": "application/vnd.rcuk.gtr.json-v7",
    "User-Agent": "CS5079-student-project (University of Aberdeen; educational use)",
}
PAGE_SIZE = 100  # API maximum (requests with s>100 return HTTP 400)
# folder of this script (falls back to the working directory inside a notebook)
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
RAW_DIR = os.path.join(_HERE, "data", "raw")


def fetch_page(session, endpoint, page, max_retries=6, timeout=60):
    """Fetch one page of an endpoint, retrying with exponential back-off.

    Returns the decoded JSON dict. Raises RuntimeError after max_retries.
    """
    url = f"{BASE}/{endpoint}"
    params = {"p": page, "s": PAGE_SIZE}
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt  # 1, 2, 4, 8 ... seconds
                print(f"[{endpoint}] page {page}: HTTP {r.status_code}, retry in {wait}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except (requests.ConnectionError, requests.Timeout, ValueError) as e:
            wait = 2 ** attempt
            print(f"[{endpoint}] page {page}: {type(e).__name__}, retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"[{endpoint}] page {page} failed after {max_retries} retries")


def page_path(endpoint, page):
    return os.path.join(RAW_DIR, endpoint, f"page_{page:05d}.json.gz")


def collect(endpoint, max_pages=None, delay=1.0, log=None):
    """Download all (or the first max_pages) pages of an endpoint into the cache."""
    os.makedirs(os.path.join(RAW_DIR, endpoint), exist_ok=True)
    session = requests.Session()
    first = fetch_page(session, endpoint, 1)
    total_pages, total_size = first["totalPages"], first["totalSize"]
    n_pages = min(total_pages, max_pages or total_pages)
    print(f"[{endpoint}] {total_size} records in {total_pages} pages; collecting {n_pages}")
    with gzip.open(page_path(endpoint, 1), "wt", encoding="utf-8") as f:
        json.dump(first, f)
    t0, fetched, last = time.time(), 1, time.time()
    for page in range(2, n_pages + 1):
        if os.path.exists(page_path(endpoint, page)):
            continue  # cached by an earlier run -> resume
        gap = time.time() - last
        if gap < delay:
            time.sleep(delay - gap)  # rate limit
        last = time.time()
        data = fetch_page(session, endpoint, page)
        tmp = page_path(endpoint, page) + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, page_path(endpoint, page))  # atomic write: no half files
        fetched += 1
        if page % 50 == 0:
            print(f"[{endpoint}] page {page}/{n_pages}  ({time.time()-t0:.0f}s)", flush=True)
    if log is not None:
        log[endpoint] = {
            "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "api_total_records": total_size,
            "api_total_pages": total_pages,
            "pages_requested": n_pages,
            "pages_fetched_this_run": fetched,
            "seconds": round(time.time() - t0, 1),
            "delay_s": delay,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", choices=["projects", "funds", "organisations", "both"], default="both")
    ap.add_argument("--max-pages", type=int, default=None, help="limit pages (default: all)")
    ap.add_argument("--delay", type=float, default=1.0, help="min seconds between requests")
    args = ap.parse_args()

    log_path = os.path.join(RAW_DIR, "collection_log.json")
    log = {}
    endpoints = ["projects", "funds"] if args.endpoint == "both" else [args.endpoint]
    # The two endpoints run in parallel threads, each at ~1 req/s (<= 2 req/s total).
    threads = [threading.Thread(target=collect, args=(e, args.max_pages, args.delay, log)) for e in endpoints]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # re-read and merge so that runs for different endpoints never overwrite each other's entries
    merged = json.load(open(log_path)) if os.path.exists(log_path) else {}
    merged.update({k: v for k, v in log.items() if k in endpoints})
    with open(log_path, "w") as f:
        json.dump(merged, f, indent=2)
    log = merged
    print("done:", json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
