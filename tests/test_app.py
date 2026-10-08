"""Streamlit AppTest: runs app.py with the bundled sample offers, no network, no API key."""

import json
import shutil

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT, FakeOpenAI


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    shutil.copy(ROOT / "data" / "sample_jobs.json", tmp_path / "data" / "sample_jobs.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.delenv("CV_STORAGE_ENABLED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    st.cache_data.clear()  # the embedding cache is process-wide
    return tmp_path


def run_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    return at.run()


def all_text(at):
    parts = [e.value for e in at.markdown] + [e.value for e in at.info] + [e.value for e in at.warning] + [e.value for e in at.success]
    parts += [e.value for e in at.caption] + [e.value for e in at.header]
    return "\n".join(parts)


def test_renders_with_sample_offers_and_privacy_notice(workdir):
    at = run_app()
    assert not at.exception
    text = all_text(at)
    assert "sent to OpenAI" in text and "not stored" in text
    assert "fictional sample offers" in text


def test_no_recent_cv_list_and_no_storage_checkbox_by_default(workdir):
    at = run_app()
    assert "Recent CVs" not in all_text(at)
    assert len(at.checkbox) == 0


def test_storage_checkbox_appears_only_when_enabled_and_defaults_off(workdir, monkeypatch):
    monkeypatch.setenv("CV_STORAGE_ENABLED", "true")
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")
    at = run_app()
    assert len(at.checkbox) == 1 and at.checkbox[0].value is False


def test_cached_real_offers_replace_sample(workdir):
    jobs = [{"id": 1, "title": "T", "company": "C", "description": "d", "requirements": "r", "location": "L", "salary": "", "link": ""}]
    (workdir / "data" / "muse_jobs.json").write_text(json.dumps(jobs))
    at = run_app()
    assert "fictional" not in all_text(at)
    assert "1 offers from The Muse API" in all_text(at)


def test_matching_end_to_end_with_fake_openai(workdir, monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr("utils.embeddings.get_openai_client", lambda api_key=None: fake)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["cv_text"] = "some cv text"
    at.session_state["cv_skills"] = "Python, PyTorch"
    at.run()
    assert not at.exception
    button = next(b for b in at.button if "Find matching offers" in b.label)
    button.click().run()
    assert not at.exception
    results = at.session_state["results"]
    assert len(results) == 10
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)
    # every ranked title really belongs to the offer that was embedded with it
    assert {r["title"] for r in results} <= {j["title"] for j in json.loads((workdir / "data" / "sample_jobs.json").read_text())}


def test_embedding_failure_produces_no_ranking(workdir, monkeypatch):
    fake = FakeOpenAI(fail_on_call=2)  # CV embedding ok, job batch fails
    monkeypatch.setattr("utils.embeddings.get_openai_client", lambda api_key=None: fake)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["cv_skills"] = "Python"
    at.session_state["cv_text"] = "cv"
    at.run()
    next(b for b in at.button if "Find matching offers" in b.label).click().run()
    assert at.session_state["results"] is None
    assert any("No ranking was produced" in e.value for e in at.error)
