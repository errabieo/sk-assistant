import pytest
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from sk_assistant.chunking import Chunk
from sk_assistant.indexing import COLLECTION_NAME, EMBEDDING_MODEL, index_chunks
from sk_assistant.retrieval import search


@pytest.fixture(scope="module")
def embedder() -> TextEmbedding:
    return TextEmbedding(EMBEDDING_MODEL)


chunks = [
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


def test_search_finds_imputations(embedder: TextEmbedding) -> None:
    client = QdrantClient(":memory:")
    index_chunks(client, embedder, chunks)
    result = search(client, embedder, "how to fill missing values", k=1)
    assert result[0].chunk.page == "impute"


def test_index_is_idempotent(embedder: TextEmbedding) -> None:
    client = QdrantClient(":memory:")
    index_chunks(client, embedder, chunks)
    index_chunks(client, embedder, chunks)
    assert client.count(COLLECTION_NAME).count == 3
