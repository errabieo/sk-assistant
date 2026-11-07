# Decisions

## 001 — Embedding library: fastembed

- **Context:** We need an embedding model to map each chunk of the knowledge base to a vector of fixed dimension. Constraints: keep the Docker image and CI install as small as possible, and run on CPU only.
- **Decision:** Use fastembed (by Qdrant) with `BAAI/bge-small-en-v1.5` (384 dimensions). fastembed runs on ONNX Runtime, optimized for inference-only use cases.
- **Alternatives considered:** sentence-transformers (Hugging Face), better suited to research and fine-tuning. Rejected because it depends on PyTorch: a 555 MB wheel on Linux x86, plus several GB of CUDA libraries by default, versus ~170 MB for the whole fastembed environment.
- **Consequences:** We accept a smaller choice of models and no fine-tuning, in exchange for a light image and fast CI. fastembed's models are strong enough for our use case; the evaluation will confirm it.

## 002 - The LLM interface

- **Context:** We need an interface that describes any LLM client so that no real model is called in the CI and tests so that tests won't be slow and flaky. Also the backend can be swapped to a hosted API at deployment.
- **Decision:** Define an LLMClient protocol, following the Dependency Inversion Principle: `/ask` depends on the abstraction, not on a concrete model.
- **Alternatives considered:** Calling Ollama directly inside `/ask`.
- **Consequences:** One more abstraction to maintain, plus every new backend must respect the contract.

## 003 - Ollama client library

- **Context:** `OllamaLLM` needs to call the local Ollama server’s HTTP API.
- **Decision:** Use the official ollama library: it makes the HTTP request, returns a typed response, raises errors as exceptions, and is maintained by the Ollama team.
- **Alternatives considered:** We send a `POST` to `http://localhost:11434/api/generate` with a JSON body then we parse the JSON response.
- **Consequences:** One more dependency added in exchange for less code to write and to test.

## 004 — LLM for local development

- **Context:** We need a model that runs locally on a 16 GB MacBook, at no cost, and leaves room for Qdrant and the embedder
- **Decision:** `granite4:micro`.
- **Alternatives considered:** A larger model like `granite4:small-h` but it would not fit.
- **Consequences:** Modest answer quality, which is acceptable for a proof of concept, and the evaluation will measure it.
