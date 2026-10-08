"""PDF text extraction (pdfplumber or pypdf) from a path or an in-memory file."""

from __future__ import annotations

import logging
from typing import BinaryIO

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PAGES = 10

METHODS = ("pdfplumber", "pypdf")


def extract_text_from_pdf(
    source: str | BinaryIO,
    method: str = "pdfplumber",
    clean: bool = True,
    max_pages: int = MAX_PAGES,
) -> str | None:
    """Extract text from a PDF given as a path or a binary file object.

    Only the first ``max_pages`` pages are read. Returns ``None`` when the
    file cannot be parsed or contains no extractable text.
    """
    if method == "pypdf2":  # backwards-compatible alias used in the notebooks
        method = "pypdf"
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; use one of {METHODS}")

    try:
        if method == "pdfplumber":
            with pdfplumber.open(source) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages[:max_pages]]
        else:
            reader = PdfReader(source)
            pages = [page.extract_text() or "" for page in reader.pages[:max_pages]]
    except Exception as exc:  # parser errors are heterogeneous
        logger.warning("PDF extraction failed (%s): %s", method, type(exc).__name__)
        return None

    text = "\n".join(p for p in pages if p)
    if clean:
        text = " ".join(text.split())
    return text or None
