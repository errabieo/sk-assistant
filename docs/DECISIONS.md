# Decisions

## 001 — Embedding library: fastembed

- **Context:** We need an embedding model to map each chunk of the knowledge base to a vector of fixed dimension. Constraints: keep the Docker image and CI install as small as possible, and run on CPU only.
- **Decision:** Use fastembed (by Qdrant) with `BAAI/bge-small-en-v1.5` (384 dimensions). fastembed runs on ONNX Runtime, optimized for inference-only use cases.
- **Alternatives considered:** sentence-transformers (Hugging Face), better suited to research and fine-tuning. Rejected because it depends on PyTorch: a 555 MB wheel on Linux x86, plus several GB of CUDA libraries by default, versus ~170 MB for the whole fastembed environment.
- **Consequences:** We accept a smaller choice of models and no fine-tuning, in exchange for a light image and fast CI. fastembed's models are strong enough for our use case; the evaluation will confirm it.