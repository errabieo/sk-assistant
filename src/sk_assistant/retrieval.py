import os

from fastembed import TextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient

from sk_assistant.chunking import Chunk
from sk_assistant.indexing import COLLECTION_NAME, EMBEDDING_MODEL


class SearchResult(BaseModel):
    chunk: Chunk
    score: float


def search(
    client: QdrantClient, embedder: TextEmbedding, question: str, k: int = 5
) -> list[SearchResult]:
    results: list[SearchResult] = []

    query_vector = next(iter(embedder.query_embed(query=question)))  # one numpy array
    hits = client.query_points(COLLECTION_NAME, query=query_vector.tolist(), limit=k).points
    for hit in hits:
        if hit.payload is None:
            raise ValueError(f"Point {hit.id} has no payload")
        chunk = Chunk.model_validate(hit.payload)
        results.append(SearchResult(chunk=chunk, score=hit.score))

    return results


if __name__ == "__main__":
    questions = [
        "What is the difference between StandardScaler and MinMaxScaler?",
        "How do I choose the number of clusters in KMeans?",
        "How does cross-validation work?",
        "What does the C parameter do in SVM?",
        "How do I handle missing values?",
    ]
    embedder = TextEmbedding(EMBEDDING_MODEL)
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    for question in questions:
        results = search(client=client, embedder=embedder, question=question)
        print(f"Results for the question: {question}")
        for result in results:
            print(
                f"Page: {result.chunk.page}, "
                f"Section title: {result.chunk.section_title}, "
                f"Score: {result.score}."
            )
