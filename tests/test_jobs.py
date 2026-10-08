import json

import pandas as pd
import requests

from utils import jobs as J
from utils.embeddings import get_similarity_rating


class Resp:
    def __init__(self, payload=None, error=False):
        self.payload, self.error = payload, error

    def raise_for_status(self):
        if self.error:
            raise requests.HTTPError("500")

    def json(self):
        return self.payload


def muse(job_id, name="ML Engineer", link="https://example.com/job"):
    return {
        "id": job_id,
        "name": name,
        "company": {"name": "Acme"},
        "contents": "<p>Build <b>models</b> &amp; pipelines</p>" + "x" * 2000,
        "locations": [{"name": "Remote"}, {"name": "Berlin"}],
        "tags": [{"name": "Python"}, {"name": "SQL"}],
        "refs": {"landing_page": link},
    }


def test_strip_html():
    assert J.strip_html("<p>a &amp; <b>b</b></p>\n  c") == "a & b c"


def test_format_muse_job():
    job = J.format_muse_job(muse(1))
    assert job["title"] == "ML Engineer" and job["company"] == "Acme"
    assert job["location"] == "Remote, Berlin" and job["requirements"] == "Python, SQL"
    assert "<" not in job["description"] and len(job["description"]) <= 1000


def test_format_handles_missing_fields():
    job = J.format_muse_job({"id": 5})
    assert job["title"] == "Unknown" and job["location"] == "Remote" and job["link"] == ""


def test_unsafe_links_are_dropped():
    assert J.safe_url("javascript:alert(1)") == ""
    assert J.safe_url("data:text/html,x") == ""
    assert J.safe_url(" https://ok.example/x ") == "https://ok.example/x"
    assert J.format_muse_job(muse(1, link="javascript:alert(1)"))["link"] == ""


def test_fetch_deduplicates_and_survives_errors():
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["search"])
        if params["search"] == "b":
            return Resp(error=True)
        if params["search"] == "c":
            raise requests.ConnectionError("down")
        return Resp({"results": [muse(1), muse(2)]})

    jobs = J.fetch_fresh_jobs(http_get=fake_get, terms=("a", "b", "c", "d"))
    assert calls == ["a", "b", "c", "d"]
    assert [j["id"] for j in jobs] == [1, 2]  # duplicates from "d" removed


def test_fetch_returns_empty_when_everything_fails():
    def fake_get(url, params, timeout):
        raise requests.Timeout()

    assert J.fetch_fresh_jobs(http_get=fake_get, terms=("a",)) == []


def test_load_prefers_cache_then_sample_then_none(tmp_path):
    assert J.load_jobs(tmp_path) == ([], "none")
    (tmp_path / "sample_jobs.json").write_text(json.dumps([{"id": 1}]))
    assert J.load_jobs(tmp_path) == ([{"id": 1}], "sample")
    (tmp_path / "muse_jobs.json").write_text(json.dumps([{"id": 2}]))
    assert J.load_jobs(tmp_path) == ([{"id": 2}], "cache")


def test_load_ignores_corrupt_cache(tmp_path):
    (tmp_path / "muse_jobs.json").write_text("{not json")
    (tmp_path / "sample_jobs.json").write_text(json.dumps([{"id": 1}]))
    assert J.load_jobs(tmp_path)[1] == "sample"


def test_save_then_load_roundtrip(tmp_path):
    J.save_jobs([{"id": 9}], tmp_path)
    assert J.load_jobs(tmp_path) == ([{"id": 9}], "cache")
    assert J.last_update(tmp_path)


def test_csv_neutralises_formula_injection():
    assert J.csv_safe("=HYPERLINK(...)") == "'=HYPERLINK(...)"
    assert J.csv_safe("@SUM(A1)") == "'@SUM(A1)"
    assert J.csv_safe("Data Scientist") == "Data Scientist"
    results = [{"title": "=1+1", "company": "+x", "location": "-y", "requirements": "@z", "similarity": 0.7, "link": ""}]
    csv = J.export_results_to_csv(results, get_similarity_rating)
    frame = pd.read_csv(__import__("io").StringIO(csv))
    assert frame.loc[0, "Title"] == "'=1+1" and frame.loc[0, "Rating"] == "Excellent match"
