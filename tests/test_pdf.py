import io

import pytest

from utils.pdf_handler import MAX_PAGES, extract_text_from_pdf


@pytest.mark.parametrize("method", ["pdfplumber", "pypdf", "pypdf2"])
def test_extracts_text_from_bytes(make_pdf, method):
    pdf = make_pdf(["Python developer", "PyTorch and Docker"])
    text = extract_text_from_pdf(io.BytesIO(pdf), method=method)
    assert "Python developer" in text and "Docker" in text


def test_reads_from_path(make_pdf, tmp_path):
    path = tmp_path / "cv.pdf"
    path.write_bytes(make_pdf(["hello world"]))
    assert "hello world" in extract_text_from_pdf(str(path))


def test_garbage_returns_none():
    assert extract_text_from_pdf(io.BytesIO(b"not a pdf at all")) is None


def test_only_first_pages_are_read(make_pdf):
    pdf = make_pdf(["marker"], pages=MAX_PAGES + 3)
    text = extract_text_from_pdf(io.BytesIO(pdf))
    assert text.count("marker") == MAX_PAGES


def test_unknown_method_rejected(make_pdf):
    with pytest.raises(ValueError):
        extract_text_from_pdf(io.BytesIO(make_pdf(["x"])), method="ocr")
