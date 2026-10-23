from fastapi.testclient import TestClient

from sk_assistant.app import app

client = TestClient(app)


def test_ask_returns_fake_answer() -> None:
    response = client.post("/ask", json={"question": "How can I implement PCA"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Fake answer to: How can I implement PCA", "sources": []}


def test_ask_rejects_missing_body() -> None:
    response = client.post("/ask")
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "missing"
    assert error["loc"] == ["body"]


def test_ask_rejects_empty_question() -> None:
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "string_too_short"
    assert error["loc"] == ["body", "question"]


def test_ask_rejects_missing_question() -> None:
    response = client.post("/ask", json={})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "missing"
    assert error["loc"] == ["body", "question"]
