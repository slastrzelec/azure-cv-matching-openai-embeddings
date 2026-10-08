"""Shared fixtures: a deterministic fake OpenAI client (no network, no API key)."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _vector(text: str, dim: int = 16) -> list[float]:
    """Deterministic pseudo-embedding derived from the text."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    return np.random.default_rng(seed).normal(size=dim).tolist()


class FakeOpenAI:
    """Mimics the parts of the OpenAI client used by the app."""

    def __init__(self, fail_on_call: int | None = None, shuffle: bool = False):
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call
        self.shuffle = shuffle
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.chat_prompts: list[str] = []

    def _embed(self, input, model):  # noqa: A002 - mirrors the real signature
        self.calls.append(list(input))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("boom")
        items = [SimpleNamespace(index=i, embedding=_vector(t)) for i, t in enumerate(input)]
        if self.shuffle:
            items.reverse()  # real API may not guarantee list order; index is authoritative
        return SimpleNamespace(data=items)

    def _chat(self, model, messages, max_tokens, temperature):
        self.chat_prompts.append(messages[0]["content"])
        message = SimpleNamespace(content="  Python, PyTorch, Docker  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def fake_client():
    return FakeOpenAI()


@pytest.fixture
def make_pdf():
    """Build a small text PDF in memory."""
    from reportlab.pdfgen import canvas

    def _make(lines: list[str], pages: int = 1) -> bytes:
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        for _ in range(pages):
            y = 800
            for line in lines:
                c.drawString(72, y, line)
                y -= 20
            c.showPage()
        c.save()
        return buf.getvalue()

    return _make
