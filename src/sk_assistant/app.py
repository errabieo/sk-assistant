from fastapi import FastAPI

from sk_assistant.schemas import AskRequest, AskResponse


app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check: returns ok if the server is running"""

    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    """Answer a question about scikit-learn (fake answer for now)."""
    return AskResponse(answer=f"Fake answer to: {request.question}", sources=[])
