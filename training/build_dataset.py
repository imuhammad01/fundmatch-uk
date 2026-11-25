"""
build_dataset.py - turn the cached raw GtR pages into one tidy table.

Reads data/raw/projects/*.json.gz, data/raw/funds/*.json.gz and (if present)
data/raw/organisations/*.json.gz written by collect_gtr.py, and writes
data/processed/gtr_projects.parquet with one row per GtR project id:

    project_id, grant_ref, title, abstract, tech_abstract, potential_impact,
    lead_funder, grant_category, status, department, lead_org_id, lead_org_name,
    fund_start, fund_end, award_gbp, fund_category, n_funds,
    research_subjects, research_topics, gtr_url

Award amounts and dates come from the /funds records, joined on the project
id found in each fund's rel="FUNDED" link (they are not on project records).
"""
import glob
import gzip
import json
import os

import pandas as pd

# folder of this script (falls back to the working directory inside a notebook)
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
HERE = _HERE
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "processed", "gtr_projects.parquet")


def iter_records(endpoint, key):
    """Yield every record of one endpoint from the cached pages."""
    for path in sorted(glob.glob(os.path.join(RAW, endpoint, "page_*.json.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for rec in json.load(f).get(key) or []:
                yield rec


def link_ids(rec, rel):
    """Return (id, start, end) for each link of a given rel type."""
    out = []
    for l in (rec.get("links") or {}).get("link") or []:
        if l.get("rel") == rel:
            out.append((l["href"].rstrip("/").split("/")[-1], l.get("start"), l.get("end")))
    return out


def parse_project(p):
    lead = link_ids(p, "LEAD_ORG")
    ids = (p.get("identifiers") or {}).get("identifier") or []
    subj = [s["text"] for s in (p.get("researchSubjects") or {}).get("researchSubject") or []]
    topic = [t["text"] for t in (p.get("researchTopics") or {}).get("researchTopic") or []]
    # Innovate UK projects list the lead business as a participant (no LEAD_ORG name otherwise)
    lead_part = [x["organisationName"] for x in ((p.get("participantValues") or {}).get("participant") or [])
                 if x.get("role") == "LEAD_PARTICIPANT"]
    return {
        "project_id": p["id"],
        "grant_ref": ids[0]["value"] if ids else None,
        "title": p.get("title"),
        "abstract": p.get("abstractText"),
        "tech_abstract": p.get("techAbstractText"),
        "potential_impact": p.get("potentialImpact"),
        "lead_funder": p.get("leadFunder"),
        "grant_category": p.get("grantCategory"),
        "status": p.get("status"),
        "department": p.get("leadOrganisationDepartment"),
        "lead_org_id": lead[0][0] if lead else None,
        "lead_participant": lead_part[0] if lead_part else None,
        "research_subjects": "; ".join(subj),
        "research_topics": "; ".join(t for t in topic if t != "Unclassified"),
        "gtr_url": f"https://gtr.ukri.org/projects?ref={ids[0]['value']}" if ids else None,
    }


def main():
    projects = pd.DataFrame(parse_project(p) for p in iter_records("projects", "project"))
    print("project records read:", len(projects))
    n0 = len(projects)
    projects = projects.drop_duplicates("project_id")
    print("duplicate project ids across pages removed:", n0 - len(projects))

    funds = []
    for f in iter_records("funds", "fund"):
        for pid, _, _ in link_ids(f, "FUNDED"):
            funds.append({
                "project_id": pid,
                "fund_id": f["id"],
                "fund_start": f.get("start"),
                "fund_end": f.get("end"),
                "award_gbp": (f.get("valuePounds") or {}).get("amount"),
                "fund_category": f.get("category"),
            })
    funds = pd.DataFrame(funds).drop_duplicates("fund_id")
    print("fund records read:", len(funds))
    # A project can (rarely) have several fund records: sum amounts, take earliest start/latest end.
    agg = funds.groupby("project_id").agg(
        award_gbp=("award_gbp", "sum"), fund_start=("fund_start", "min"),
        fund_end=("fund_end", "max"), fund_category=("fund_category", "first"),
        n_funds=("fund_id", "size")).reset_index()
    df = projects.merge(agg, on="project_id", how="left")
    for c in ("fund_start", "fund_end"):
        df[c] = pd.to_datetime(df[c], unit="ms", errors="coerce")

    org_pages = glob.glob(os.path.join(RAW, "organisations", "page_*.json.gz"))
    if org_pages:
        orgs = {o["id"]: o.get("name") for o in iter_records("organisations", "organisation")}
        df["lead_org_name"] = df["lead_org_id"].map(orgs)
    else:
        df["lead_org_name"] = None
    df["lead_org_name"] = df["lead_org_name"].fillna(df["lead_participant"])
    df = df.drop(columns=["lead_participant"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print("saved", OUT, df.shape)
    print("projects with a joined fund record:", df["award_gbp"].notna().sum())


if __name__ == "__main__":
    main()
