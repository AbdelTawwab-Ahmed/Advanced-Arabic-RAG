from typing import List
from src.llm_client import get_llm
from pydantic import BaseModel, Field


DECOMPOSE_PROMPT = """You are helping a RAG system over Arabic bank procedure manuals \
(alarm systems, mail/files handling, warehouse operations) handle multi-part questions.

The query below asks about multiple distinct sub-topics or steps that should be looked up \
separately. Break it into the smallest set of independent, self-contained Arabic sub-questions \
needed to fully answer it. Each sub-question must be understandable on its own, without \
needing the others for context. Do not add sub-questions beyond what the original query asks.

Original query: {query}
"""


class SubQuestions(BaseModel):
    sub_questions: List[str] = Field(description="Independent, self-contained Arabic sub-questions.")


def decompose_query(query: str) -> List[str]:
    llm = get_llm().with_structured_output(SubQuestions)
    result: SubQuestions = llm.invoke(DECOMPOSE_PROMPT.format(query=query))
    return result.sub_questions