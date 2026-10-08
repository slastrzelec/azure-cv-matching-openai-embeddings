import numpy as np
import pytest

from tests.conftest import FakeOpenAI
from utils.embeddings import (
    BATCH_SIZE,
    MAX_CV_CHARS,
    MAX_EMBED_CHARS,
    EmbeddingError,
    cosine_similarities,
    embed_texts,
    expected_calls,
    extract_skills_with_ai,
    get_similarity_rating,
    rank_jobs,
)


def test_embed_texts_batches_and_keeps_order():
    client = FakeOpenAI()
    texts = [f"text {i}" for i in range(250)]
    out = embed_texts(texts, client=client)
    assert out.shape == (250, 16)
    assert len(client.calls) == expected_calls(250) == 3
    assert [len(c) for c in client.calls] == [BATCH_SIZE, BATCH_SIZE, 50]
    # row i belongs to text i (compare with an independent single call)
    single = embed_texts([texts[137]], client=FakeOpenAI())
    assert np.allclose(out[137], single[0])


def test_embed_texts_uses_api_index_not_list_order():
    texts = ["a", "b", "c"]
    normal = embed_texts(texts, client=FakeOpenAI())
    shuffled = embed_texts(texts, client=FakeOpenAI(shuffle=True))
    assert np.allclose(normal, shuffled)


def test_embed_texts_failure_is_all_or_nothing():
    client = FakeOpenAI(fail_on_call=2)
    with pytest.raises(EmbeddingError):
        embed_texts([f"t{i}" for i in range(150)], client=client)


def test_embed_texts_truncates_and_replaces_empty():
    client = FakeOpenAI()
    embed_texts(["x" * (MAX_EMBED_CHARS + 500), "   "], client=client)
    sent = client.calls[0]
    assert len(sent[0]) == MAX_EMBED_CHARS
    assert sent[1] == "(empty)"


def test_embed_texts_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbeddingError):
        embed_texts(["x"])


def test_cosine_similarities_known_vectors():
    q = np.array([1.0, 0.0])
    m = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0], [0.0, 0.0]])
    assert np.allclose(cosine_similarities(q, m), [1.0, 0.0, -1.0, 0.0])
    assert cosine_similarities(q, np.empty((0, 2))).size == 0


def test_rank_jobs_orders_and_does_not_mutate():
    jobs = [{"title": "a"}, {"title": "b"}, {"title": "c"}]
    ranked = rank_jobs(jobs, [0.2, 0.9, 0.5], top_n=2)
    assert [j["title"] for j in ranked] == ["b", "c"]
    assert all("similarity" not in j for j in jobs)


def test_rank_jobs_length_mismatch_raises():
    with pytest.raises(ValueError):
        rank_jobs([{"title": "a"}, {"title": "b"}], [0.5])


def test_rating_thresholds():
    assert get_similarity_rating(0.61)[1] == "Excellent match"
    assert get_similarity_rating(0.55)[1] == "Good match"
    assert get_similarity_rating(0.45)[1] == "Average match"
    assert get_similarity_rating(0.10)[1] == "Poor match"


def test_extract_skills_strips_and_truncates_cv():
    client = FakeOpenAI()
    skills = extract_skills_with_ai("x" * (MAX_CV_CHARS + 5000), client=client)
    assert skills == "Python, PyTorch, Docker"
    assert "x" * (MAX_CV_CHARS + 1) not in client.chat_prompts[0]


def test_extract_skills_returns_none_on_api_error():
    class Broken:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**_):
                    raise RuntimeError("down")

    assert extract_skills_with_ai("cv", client=Broken()) is None
