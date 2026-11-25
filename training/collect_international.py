"""
collect_international.py - small cross-country test sets collected with our own code (CS5079 Task 1, v2).

Purpose: check whether the UK council classifier transfers to grant abstracts from other countries, where
the "right" UK council is known from the foreign funder's structure:
  * NSF Awards API (USA)      https://api.nsf.gov/services/v1/awards.json   (public domain, US Government work)
  * NIH RePORTER API v2 (USA) https://api.reporter.nih.gov/v2/projects/search (public data; NIH asks <= 1 req/s)
  * ARC Grants API (Australia) https://dataportal.arc.gov.au/NCGP/API/grants (open JSON:API, no key)
Only the fields we need are kept: id, title, abstract, directorate/division or institute, amount, start date.
Personal data returned by the APIs (PI / investigator names, e-mails, phone numbers) is dropped before saving.

Usage: python collect_international.py [--nsf-pages 80] [--nih-pages 2] [--arc-pages 4]
"""
import argparse, gzip, json, os, time
import requests

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT = os.path.join(_HERE, "data", "raw", "international")
UA = {"User-Agent": "CS5079-student-project (University of Aberdeen; educational use)"}


def get_json(method, url, retries=6, **kw):
    """HTTP request with exponential back-off on errors (same policy as collect_gtr.py)."""
    for a in range(retries):
        try:
            r = requests.request(method, url, headers={**UA, **kw.pop("headers", {})}, timeout=60, **kw)
            if r.status_code == 200:
                return r.json()
            print(f"HTTP {r.status_code}, retry in {2**a}s")
        except (requests.ConnectionError, requests.Timeout, ValueError) as e:
            print(type(e).__name__, f"retry in {2**a}s")
        time.sleep(2 ** a)
    raise RuntimeError(url)


def collect_nsf(pages, delay=1.0):
    """NSF awards starting in 2023 (25 per page, the API's page size). Cached per page."""
    keep = ["id", "title", "abstractText", "dirAbbr", "divAbbr", "estimatedTotalAmt", "startDate"]
    for p in range(pages):
        path = os.path.join(OUT, f"nsf_{p:04d}.json.gz")
        if os.path.exists(path):
            continue
        d = get_json("GET", "https://api.nsf.gov/services/v1/awards.json",
                     params={"dateStart": "01/01/2023", "dateEnd": "12/31/2023", "rpp": 25, "offset": 1 + 25 * p})
        rows = [{k: a.get(k) for k in keep} for a in d["response"].get("award", [])]   # drops PI names / e-mails
        with gzip.open(path, "wt") as f:
            json.dump(rows, f)
        time.sleep(delay)


def collect_nih(pages, delay=1.5):
    """NIH RePORTER projects from fiscal year 2023 (500 per page)."""
    for p in range(pages):
        path = os.path.join(OUT, f"nih_{p:04d}.json.gz")
        if os.path.exists(path):
            continue
        body = {"criteria": {"fiscal_years": [2023]}, "offset": 500 * p, "limit": 500,
                "include_fields": ["ApplId", "ProjectTitle", "AbstractText", "AgencyIcAdmin", "AwardAmount", "ProjectStartDate"]}
        d = get_json("POST", "https://api.reporter.nih.gov/v2/projects/search", json=body)
        rows = [{"id": r.get("appl_id"), "title": r.get("project_title"), "abstractText": r.get("abstract_text"),
                 "institute": (r.get("agency_ic_admin") or {}).get("abbreviation"), "amount": r.get("award_amount"),
                 "startDate": r.get("project_start_date")} for r in d.get("results", [])]
        with gzip.open(path, "wt") as f:
            json.dump(rows, f)
        time.sleep(delay)


def collect_arc(pages, delay=1.5):
    """Most recent ARC grants (500 per page). Investigator names are dropped."""
    for p in range(pages):
        path = os.path.join(OUT, f"arc_{p:04d}.json.gz")
        if os.path.exists(path):
            continue
        d = get_json("GET", "https://dataportal.arc.gov.au/NCGP/API/grants",
                     params={"page[number]": p + 1, "page[size]": 500})
        rows = [{"id": g["id"], "summary": g["attributes"].get("grant-summary"), "scheme": g["attributes"].get("scheme-name"),
                 "field": g["attributes"].get("primary-field-of-research"), "amount": g["attributes"].get("current-funding-amount"),
                 "year": g["attributes"].get("funding-commencement-year")} for g in d.get("data", [])]
        with gzip.open(path, "wt") as f:
            json.dump(rows, f)
        time.sleep(delay)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--nsf-pages", type=int, default=80); ap.add_argument("--nih-pages", type=int, default=2)
    ap.add_argument("--arc-pages", type=int, default=4)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); collect_nsf(a.nsf_pages); collect_nih(a.nih_pages); collect_arc(a.arc_pages)
    log = {"collected": time.strftime("%Y-%m-%d %H:%M"), "nsf_pages": a.nsf_pages, "nih_pages": a.nih_pages, "arc_pages": a.arc_pages, "seconds": round(time.time() - t0)}
    json.dump(log, open(os.path.join(OUT, "collection_log.json"), "w"), indent=1); print(log)
