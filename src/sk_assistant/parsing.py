from pydantic import BaseModel


class Section(BaseModel):
    page: str
    page_title: str
    title: str
    text: str
    url: str


def parse_rst(text: str, page: str) -> list[Section]:
    """Parses the raw docs in .rst into sections"""

    lines_of_text = text.splitlines()

    titles_indices = [
        i
        for i in range(len(lines_of_text) - 1)
        if is_underline(line=lines_of_text[i + 1]) and lines_of_text[i] != ""
    ]

    if not titles_indices:
        return []

    page_title = lines_of_text[titles_indices[0]].strip()

    sections: list[Section] = []
    for k, start in enumerate(titles_indices):
        end = titles_indices[k + 1] if k + 1 < len(titles_indices) else len(lines_of_text)
        body = lines_of_text[start + 2 : end]
        kept = [line for line in body if not line.startswith("..") and not is_underline(line)]
        content = "\n".join(kept).strip()
        if content != "":
            sections.append(
                Section(
                    page=page,
                    page_title=page_title,
                    title=lines_of_text[start].strip(),
                    text=content,
                    url=f"https://scikit-learn.org/1.9/modules/{page}.html",
                )
            )

    return sections


def is_underline(line: str) -> bool:
    """Checks if a line is an underline."""
    if line == "":
        return False
    for elt in line:
        if elt not in ["=", "-", "^"]:
            return False
    return True
