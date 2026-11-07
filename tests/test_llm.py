from sk_assistant.llm import FakeLLM

PROMPT = "How to implement kernelPCA?"


def test_fake_llm() -> None:

    llm = FakeLLM()
    response = llm.generate(prompt=PROMPT)
    assert response == "Fake answer, only to test"
    assert llm.last_prompt == PROMPT
