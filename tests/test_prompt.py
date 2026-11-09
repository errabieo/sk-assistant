from sk_assistant.chunking import Chunk
from sk_assistant.prompt import I_DONT_KNOW, build_prompt
from sk_assistant.retrieval import SearchResult

RESULTS = [
    SearchResult(
        chunk=Chunk(
            page="preprocessing",
            page_title="Preprocessing data",
            section_title="Scaling features to a range",
            url="https://scikit-learn.org/1.9/modules/preprocessing.html",
            text="An alternative standardization is scaling features" + " to lie",
            chunk_index=0,
        ),
        score=0.9,
    ),
    SearchResult(
        chunk=Chunk(
            page="classification_threshold",
            page_title="Tuning the decision " + "threshold for class prediction",
            section_title="Post-tuning the decision threshold",
            url="https://scikit-learn.org/1.9/modules/" + "classification_threshold.html",
            text="One solution to address the problem stated in "
            + "the introduction is to tune the decision "
            + "threshold of the classifier once",
            chunk_index=0,
        ),
        score=0.8,
    ),
]

QUESTION = "What are the ways of preprocessing data?"
TEST_PROMPT = build_prompt(question=QUESTION, results=RESULTS)


def test_sources_are_numbered() -> None:
    assert "[1]" in TEST_PROMPT
    assert "[2]" in TEST_PROMPT
    assert TEST_PROMPT.index("[1]") < TEST_PROMPT.index("[2]")


def test_sources_contain_titles_and_text() -> None:
    assert RESULTS[0].chunk.page_title in TEST_PROMPT
    assert RESULTS[0].chunk.section_title in TEST_PROMPT
    assert RESULTS[0].chunk.text in TEST_PROMPT

    assert RESULTS[1].chunk.page_title in TEST_PROMPT
    assert RESULTS[1].chunk.section_title in TEST_PROMPT
    assert RESULTS[1].chunk.text in TEST_PROMPT


def test_question_comes_after_sources() -> None:
    assert TEST_PROMPT.index(RESULTS[-1].chunk.text) < TEST_PROMPT.index(QUESTION)


def test_prompt_contains_i_dont_know_instruction() -> None:
    assert I_DONT_KNOW in TEST_PROMPT
