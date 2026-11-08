import os
from typing import Protocol

import ollama

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class FakeLLM:
    last_prompt: str

    def __init__(self) -> None:
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "Fake answer, only to test"


class OllamaLLM:
    model: str
    host: str
    client: ollama.Client

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self.host = host
        self.client = ollama.Client(host=self.host)

    def generate(self, prompt: str) -> str:
        response = self.client.generate(prompt=prompt, model=self.model)
        if response.response is None:
            raise ValueError("The llm returned nothing")
        return response.response


if __name__ == "__main__":
    llm = OllamaLLM(model="granite4:micro", host=OLLAMA_HOST)
    response = llm.generate("How to implement kernelPCA?")
    print(response)
