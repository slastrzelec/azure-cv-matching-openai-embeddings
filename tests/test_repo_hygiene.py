import json
import re
import subprocess

import pytest

from tests.conftest import ROOT

pytestmark = pytest.mark.repo

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{2}[ -]?)?\d{3}[ -]\d{3}[ -]\d{3}(?!\d)")


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.splitlines()


def test_no_secrets_cv_or_generated_files_tracked():
    files = tracked_files()
    bad = [f for f in files if f == ".env" or f.endswith(".pdf") or f.endswith("muse_jobs.json") or f.endswith("last_update.txt")]
    assert bad == []


def test_gitignore_covers_env_and_cv():
    text = (ROOT / ".gitignore").read_text()
    assert ".env" in text and "data/cv_" in text


def test_tracked_notebooks_have_no_personal_data_in_outputs():
    for name in (f for f in tracked_files() if f.endswith(".ipynb")):
        path = ROOT / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            for out in cell.get("outputs", []):
                blob = json.dumps(out)
                assert not EMAIL.search(blob), f"e-mail address in outputs of {name}"
                assert not PHONE.search(blob), f"phone number in outputs of {name}"


def test_app_has_no_temp_files_or_recent_cv_list():
    code = (ROOT / "app.py").read_text() + "".join(p.read_text() for p in (ROOT / "utils").glob("*.py"))
    assert "Recent CVs" not in code and "temp_" not in code


def test_cv_is_never_stored_and_no_cloud_storage_code():
    """The app keeps the CV in memory only: no storage module, no cloud-storage dependency or settings."""
    assert not (ROOT / "utils" / "azure_storage.py").exists()
    scanned = [ROOT / "app.py", ROOT / "requirements.txt", ROOT / "requirements-dev.txt", ROOT / ".env.example"]
    scanned += list((ROOT / "utils").glob("*.py")) + list((ROOT / ".github").rglob("*.yml"))
    text = "\n".join(p.read_text(encoding="utf-8").lower() for p in scanned if p.exists())
    for forbidden in ("azure", "blob", "cv_storage", "upload_cv"):
        assert forbidden not in text, forbidden
