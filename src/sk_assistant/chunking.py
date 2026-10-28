from pydantic import BaseModel

from sk_assistant.parsing import Section


class Chunk(BaseModel):
    page: str
    page_title: str
    section_title: str
    url: str
    text: str
    chunk_index: int


def chunk_section(section: Section, max_words: int = 300, overlap: int = 50) -> list[Chunk]:

    if overlap >= max_words:
        raise ValueError("The sliding window would never move")

    words = section.text.split()

    if len(words) <= max_words:
        return [
            Chunk(
                page=section.page,
                page_title=section.page_title,
                section_title=section.title,
                url=section.url,
                text=section.text,
                chunk_index=0,
            )
        ]

    step = max_words - overlap

    # find each chunk start
    chunk_start = [i for i in range(0, len(words), step)]

    chunks: list[Chunk] = []
    for i, k in enumerate(chunk_start):
        end = min(k + max_words, len(words))
        chunk_text = " ".join(words[k:end])
        chunks.append(
            Chunk(
                page=section.page,
                page_title=section.page_title,
                section_title=section.title,
                url=section.url,
                text=chunk_text,
                chunk_index=i,
            )
        )
        if end == len(words):
            break

    return chunks
