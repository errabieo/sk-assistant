import pytest

from sk_assistant.chunking import chunk_section
from sk_assistant.parsing import Section

text = " ".join(f"w_{i}" for i in range(700))  # w_0, w_1, ..., w_699
section = Section(
    page="preprocessing",
    page_title="Preprocessing data",
    title="Standardization, or mean removal and variance scaling",
    text=text,
    url="https://scikit-learn.org/1.9/modules/preprocessing.html",
)
chunks = chunk_section(section)


def test_small_section() -> None:
    section_small = Section(
        page="", page_title="", title="", text=" ".join(text.split()[:100]), url=""
    )
    assert len(chunk_section(section_small)) == 1
    assert chunk_section(section_small)[0].text == section_small.text


def test_700_section() -> None:
    assert len(chunks) == 3


def test_boundary_case() -> None:
    section_boundary_case = Section(
        page="", page_title="", title="", text=" ".join(text.split()[:550]), url=""
    )
    assert len(chunk_section(section_boundary_case)) == 2


def test_overlap() -> None:
    for i in range(len(chunks) - 1):
        assert chunks[i].text.split()[-50:] == chunks[i + 1].text.split()[:50]


def test_metadata_copied() -> None:
    for i in range(len(chunks)):
        assert chunks[i].chunk_index == i
        assert chunks[i].page == section.page
        assert chunks[i].page_title == section.page_title
        assert chunks[i].section_title == section.title
        assert chunks[i].url == section.url


def test_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_section(section, max_words=50, overlap=50)
