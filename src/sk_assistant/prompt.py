from sk_assistant.retrieval import SearchResult

I_DONT_KNOW = "I don't know based on the scikit-learn documentation."


def build_prompt(question: str, results: list[SearchResult]) -> str:

    lines = [
        "You answer questions about scikit-learn using ONLY the numbered sources below.",
        "- Cite the sources you use with their number, e.g. [1].",
        f'- If the sources do not contain the answer, reply exactly: "{I_DONT_KNOW}"',
        "",
        "Sources:",
    ]

    for i, result in enumerate(results, start=1):
        lines.append(f"[{i}] {result.chunk.page_title} - {result.chunk.section_title}")
        lines.append(result.chunk.text)
        lines.append("")

    lines.append(f"Question: {question}")
    lines.append("Answer:")

    prompt = "\n".join(lines)

    return prompt
