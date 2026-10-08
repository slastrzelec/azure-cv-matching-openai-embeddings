"""Job offers: The Muse API client, formatting, local cache, CSV export."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

MUSE_URL = "https://www.themuse.com/api/public/jobs"
SEARCH_TERMS = ("machine learning", "data scientist", "python developer", "AI engineer", "deep learning")

DATA_DIR = Path("data")
CACHE_FILE = "muse_jobs.json"
STAMP_FILE = "last_update.txt"
SAMPLE_FILE = "sample_jobs.json"

_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """Remove tags and entities and collapse whitespace."""
    return " ".join(html.unescape(_TAG_RE.sub(" ", text or "")).split())


def safe_url(url: str | None) -> str:
    """Return the URL only when it is http(s); otherwise an empty string."""
    url = (url or "").strip()
    return url if url.lower().startswith(("http://", "https://")) else ""


def format_muse_job(job: dict) -> dict:
    """Convert one Muse API result into the structure used by the app."""
    locations = job.get("locations") or []
    tags = job.get("tags") or []
    return {
        "id": job.get("id"),
        "title": job.get("name") or "Unknown",
        "company": (job.get("company") or {}).get("name") or "Unknown",
        "description": strip_html(job.get("contents", ""))[:1000],
        "requirements": ", ".join(t.get("name", "") for t in tags) or "Not specified",
        "location": ", ".join(loc.get("name", "") for loc in locations) or "Remote",
        "salary": "Not specified",
        "link": safe_url((job.get("refs") or {}).get("landing_page")),
    }


def fetch_fresh_jobs(http_get=requests.get, terms=SEARCH_TERMS) -> list[dict]:
    """Fetch offers for each search term; duplicates (same id) are dropped.

    Network or parsing errors for one term are skipped, so the result can be empty.
    """
    seen: set = set()
    jobs: list[dict] = []
    for term in terms:
        try:
            response = http_get(MUSE_URL, params={"search": term, "page": 0}, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])
        except (requests.RequestException, ValueError):
            continue
        for raw in results:
            job_id = raw.get("id")
            if job_id in seen:
                continue
            seen.add(job_id)
            jobs.append(format_muse_job(raw))
    return jobs


def save_jobs(jobs: list[dict], data_dir: Path = DATA_DIR) -> None:
    """Persist offers and a timestamp; silently ignore read-only file systems."""
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / CACHE_FILE).write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")
        (data_dir / STAMP_FILE).write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    except OSError:
        pass


def load_jobs(data_dir: Path = DATA_DIR) -> tuple[list[dict], str]:
    """Load offers without touching the network.

    Returns ``(jobs, source)`` where source is ``"cache"`` (previously fetched
    real offers), ``"sample"`` (bundled fictional offers) or ``"none"``.
    """
    for name, source in ((CACHE_FILE, "cache"), (SAMPLE_FILE, "sample")):
        path = data_dir / name
        if path.exists():
            try:
                jobs = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(jobs, list) and jobs:
                return jobs, source
    return [], "none"


def last_update(data_dir: Path = DATA_DIR) -> str | None:
    path = data_dir / STAMP_FILE
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def csv_safe(value) -> str:
    """Neutralise spreadsheet formula injection in a text cell."""
    text = str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text


def export_results_to_csv(results: list[dict], rating_fn) -> str:
    """Render ranked results as CSV; ``rating_fn`` maps similarity to (emoji, label, colour)."""
    rows = []
    for rank, job in enumerate(results, 1):
        _, label, _ = rating_fn(job["similarity"])
        rows.append(
            {
                "Rank": rank,
                "Match Score": f"{job['similarity'] * 100:.1f}%",
                "Rating": label,
                "Title": csv_safe(job["title"]),
                "Company": csv_safe(job["company"]),
                "Location": csv_safe(job["location"]),
                "Salary": csv_safe(job.get("salary", "Not specified")),
                "Requirements": csv_safe(job["requirements"]),
                "Link": csv_safe(job.get("link", "")),
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)
