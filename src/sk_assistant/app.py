from fastapi import Depends, FastAPI
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from sk_assistant.dependencies import get_embedder, get_llm, get_qdrant
from sk_assistant.llm import LLMClient
from sk_assistant.prompt import build_prompt
from sk_assistant.retrieval import search
from sk_assistant.schemas import AskRequest, AskResponse, Source

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check: returns ok if the server is running"""

    return {"status": "ok"}


@app.post("/ask")
def ask(
    request: AskRequest,
    llm: LLMClient = Depends(get_llm),
    embedder: TextEmbedding = Depends(get_embedder),
    qdrant: QdrantClient = Depends(get_qdrant),
) -> AskResponse:
    """Answer a question about scikit-learn."""

    search_results = search(client=qdrant, embedder=embedder, question=request.question)
    prompt = build_prompt(question=request.question, results=search_results)
    answer = llm.generate(prompt=prompt)

    sources = [
        Source(
            title=f"{result.chunk.page_title} — {result.chunk.section_title}", url=result.chunk.url
        )
        for result in search_results
    ]

    return AskResponse(answer=answer, sources=sources)
