from typing import List, TypedDict
from src.llm_client import get_llm

GENERATION_PROMPT = """You are answering questions about a bank's internal Arabic procedure \
manuals (alarm systems, mail/files handling, warehouse operations), using ONLY the context \
provided below. Answer in Arabic.

Rules:
- Base your answer strictly on the given context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so clearly in Arabic \
rather than guessing.
- Be direct and specific — cite the relevant procedure/step numbers from the context where applicable.

Context:
{context}

Question: {query}

Answer:"""


class GenerationResult(TypedDict):
    answer: str
    input_tokens: int
    output_tokens: int


def build_context(chunks: List[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (المصدر: {c['source_pdf']}, صفحة {c['page_number']})\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: List[dict]) -> GenerationResult:
    context = build_context(chunks)
    llm = get_llm()
    response = llm.invoke(GENERATION_PROMPT.format(context=context, query=query))

    usage = response.usage_metadata or {}
    return {
        "answer": response.content.strip(),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }