from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from sk_assistant.app import app
from sk_assistant.chunking import Chunk
from sk_assistant.dependencies import get_embedder, get_llm, get_qdrant
from sk_assistant.indexing import EMBEDDING_MODEL, index_chunks
from sk_assistant.llm import FakeLLM

CHUNKS = [
    Chunk(
        page="preprocessing",
        page_title="Preprocessing data",
        section_title="Scaling features to a range",
        url="https://scikit-learn.org/1.9/modules/preprocessing.html",
        text="An alternative standardization is scaling features to "
        "lie between a given minimum and maximum value, ",
        chunk_index=0,
    ),
    Chunk(
        page="clustering",
        page_title="Clustering",
        section_title="Clustering",
        url="https://scikit-learn.org/1.9/modules/clustering.html",
        text="Each clustering algorithm comes in two variants: a class, that "
        "implements "
        "the fit method to learn the clusters on train data, and a function,",
        chunk_index=1,
    ),
    Chunk(
        page="impute",
        page_title="Imputation of missing values",
        section_title="Imputation of missing values",
        url="https://scikit-learn.org/1.9/modules/impute.html",
        text="A basic strategy to use incomplete datasets is to discard entire "
        "rows and/or columns containing missing values.",
        chunk_index=2,
    ),
]
QUESTION = "How can I handle missing values?"


@pytest.fixture(scope="module")
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture(scope="module")
def client(fake_llm: FakeLLM) -> Iterator[TestClient]:
    # setup
    embedder = TextEmbedding(EMBEDDING_MODEL)
    qdrant = QdrantClient(":memory:")
    index_chunks(qdrant, embedder, CHUNKS)

    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_qdrant] = lambda: qdrant

    yield TestClient(app)

    # teardown
    app.dependency_overrides.clear()


def test_ask_rejects_missing_body(client: TestClient) -> None:
    response = client.post("/ask")
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "missing"
    assert error["loc"] == ["body"]


def test_ask_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "string_too_short"
    assert error["loc"] == ["body", "question"]


def test_ask_rejects_missing_question(client: TestClient) -> None:
    response = client.post("/ask", json={})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "missing"
    assert error["loc"] == ["body", "question"]


def test_ask_returns_llm_answer(client: TestClient) -> None:
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    assert response.json()["answer"] == "Fake answer, only to test"


def test_ask_returns_relevant_source(client: TestClient) -> None:
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    assert (
        response.json()["sources"][0]["url"] == "https://scikit-learn.org/1.9/modules/impute.html"
    )


def test_ask_sends_question_to_llm(client: TestClient, fake_llm: FakeLLM) -> None:
    client.post("/ask", json={"question": QUESTION})
    assert QUESTION in fake_llm.last_prompt
