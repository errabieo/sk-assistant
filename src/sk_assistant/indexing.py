import os
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from sk_assistant.chunking import Chunk, chunk_section
from sk_assistant.parsing import parse_rst

COLLECTION_NAME = "sklearn_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """load data from docs_dir and parse it and chunk it"""

    chunks: list[Chunk] = []
    list_of_docs = sorted(list(docs_dir.glob("*.rst")))
    for doc in list_of_docs:
        page = doc.stem
        text = doc.read_text(encoding="utf-8")
        sections = parse_rst(text, page)
        for section in sections:
            chunks += chunk_section(section)

    return chunks


def index_chunks(client: QdrantClient, embedder: TextEmbedding, chunks: list[Chunk]) -> None:

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        COLLECTION_NAME, vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    texts = [chunk.text for chunk in chunks]
    vectors = list(embedder.passage_embed(texts))

    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(id=i, vector=vectors[i].tolist(), payload=chunk.model_dump()))
    client.upsert(COLLECTION_NAME, points=points)


def main() -> None:
    embedder = TextEmbedding(EMBEDDING_MODEL)
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    docs_dir = Path("data/raw/scikit-learn/doc/modules")
    chunks = load_chunks(docs_dir)
    index_chunks(client, embedder, chunks)


if __name__ == "__main__":
    main()
