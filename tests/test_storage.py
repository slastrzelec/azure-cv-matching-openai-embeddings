import inspect

from utils import azure_storage as S


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CV_STORAGE_ENABLED", raising=False)
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "x")
    assert S.is_enabled() is False
    assert S.upload_cv(b"data") is False


def test_needs_both_flag_and_connection_string(monkeypatch):
    monkeypatch.setenv("CV_STORAGE_ENABLED", "true")
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    assert S.is_enabled() is False


def test_blob_name_is_random_and_has_no_user_input():
    a, b = S.new_blob_name(), S.new_blob_name()
    assert a != b and a.endswith(".pdf") and len(a) == 36
    assert "filename" not in inspect.signature(S.upload_cv).parameters


def test_no_listing_function_exists():
    public = [n for n in dir(S) if not n.startswith("_")]
    assert not any("list" in n.lower() for n in public)
