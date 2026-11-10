import os
from functools import lru_cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from sk_assistant.indexing import EMBEDDING_MODEL
from sk_assistant.llm import OLLAMA_HOST, LLMClient, OllamaLLM

LLM_MODEL = "granite4:micro"


@lru_cache
def get_llm() -> LLMClient:
    llm = OllamaLLM(model=LLM_MODEL, host=OLLAMA_HOST)

    return llm


@lru_cache
def get_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@lru_cache
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
